"""
索引构建器：把「新闻表」变成「可检索的向量索引」。

与检索时的懒加载的区别
----------------------
懒加载是「第一次查询时现算」，适合小语料的即用即走场景。
本模块是显式的离线构建，适合语料较大、需要增量更新、需要版本管理的场景：

    增量：内容 hash 变了才重新向量化，未变更的文档直接复用已有向量
    续跑：中途失败时保留进度，重跑只处理未完成的部分
    版本：产出 manifest，换模型或语料变化时可判定失效并回滚

403 条语料下离线构建没有明显收益（懒加载一次也就 2.3s），
但它的**元数据**（manifest）和**增量/续跑能力**是懒加载方案给不了的，
这些在数据量增长后才显出价值，且现在接入成本很低。
"""
import asyncio
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ai.embeddings import VectorMatrix, embed_texts
from ai.ingest.manifest import (
    IndexManifest,
    make_index_id,
    now_iso,
    save_manifest,
    save_snapshot,
)
from ai.ingest.storage import (
    ensure_dir,
    load_matrix_file,
    save_manifest_file,
    save_matrix_file,
)
from ai.ingest.validate import (
    ValidationReport,
    build_content_hash_index,
    corpus_fingerprint,
    validate_corpus,
)
from ai.retriever import build_document
from utils.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class BuildResult:
    manifest: IndexManifest
    report: ValidationReport
    total: int = 0
    embedded: int = 0
    reused: int = 0
    skipped: int = 0
    duration_ms: float = 0.0
    resumed: bool = False
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- 进度持久化

# 进度存文件而非 Redis。理由与索引产物一致：Redis 可用性不能决定
# 「这次构建要不要重跑 403 篇」，否则 Redis 一抖动就会浪费大量付费 API 调用。
_PROGRESS_FILE = Path(__file__).with_name("index") / "progress.json"


async def _save_progress(done_ids: set[int]) -> None:
    """记录已完成的文档 id，重跑时据此跳过"""
    ensure_dir()
    try:
        with io.open(_PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump({"doneIds": sorted(done_ids)}, f)
    except Exception as exc:
        logger.warning("保存构建进度失败: %s", exc)


async def _load_progress() -> set[int]:
    if not _PROGRESS_FILE.exists():
        return set()
    try:
        data = json.load(io.open(_PROGRESS_FILE, encoding="utf-8"))
        return {int(x) for x in data.get("doneIds", [])}
    except Exception:
        return set()


async def _clear_progress() -> None:
    try:
        _PROGRESS_FILE.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------- 构建

# 增量构建时复用向量的进程内缓存。
# 不缓存的话每次构建都要把 .npy 读进来再按行转成 list（403 行 × 1024 维），
# 实测这一项就占了 2 秒以上，比真正要向量化的一篇贵得多。
_vector_pool: dict[int, list[float]] = {}
_VECTOR_POOL_MAX = 20000  # 超过就整体丢弃重建，避免内存无限增长


def _load_vector_pool() -> dict[int, list[float]]:
    """从已落盘的索引加载 {news_id: vector} 映射"""
    global _vector_pool

    if _vector_pool:
        return _vector_pool

    stored = load_matrix_file()
    if not stored:
        return {}

    matrix, news_ids, _fp = stored
    pool = {}
    for row, doc_id in enumerate(news_ids):
        if row < len(matrix):
            pool[doc_id] = matrix.row(row)

    if len(pool) > _VECTOR_POOL_MAX:
        # 语料规模超出预期时不做缓存，避免占用过多内存
        return {}

    _vector_pool = pool
    return pool


def clear_vector_pool() -> None:
    """语料或索引变更后清空复用缓存"""
    global _vector_pool
    _vector_pool = {}


async def build_index(
    docs: list,
    previous: IndexManifest | None = None,
    embed_model: str = "",
    resume: bool = True,
    strict: bool = False,
    embed_dim: int = 0,
) -> BuildResult:
    """
    构建（或增量更新）向量索引。

    docs        待索引的新闻列表
    previous    上一次的 manifest；为 None 时做全量构建
    embed_model 当前使用的 embedding 模型名，写进 manifest 用于失效判定
    embed_dim   当前向量维度。传入时（>0）会参与失效判定：同一个模型可以输出
                多种维度（qwen3.7-text-embedding 支持 256~2560），只比模型名会拿
                上一维度的向量和新维度的语料混用，检索结果是错的且不报错。
                留空则只比对模型名 —— 假向量/测试场景不固定维度时用。
    resume      是否续跑（读取上次进度跳过已完成文档）

    返回 BuildResult，包含 manifest 与本次构建的统计信息。
    """
    started = time.perf_counter()

    accepted, report = validate_corpus(docs, strict=strict)
    fingerprint = corpus_fingerprint(accepted)

    # 内容 hash -> [news_id, ...]，增量判断的依据。
    # 用列表是因为存在内容完全相同的重复新闻，用单 id 会覆盖掉一部分，
    # 那些文档就永远进不了索引且无任何提示。
    current_hashes = build_content_hash_index(accepted)

# 决定哪些文档需要重新向量化
    # embed_dim 未传（<=0）时只比模型名；传了就同时比维度。
    # 不从 config 取默认值：假向量和测试用的维度是固定的，取配置值会把它们
    # 误判成维度变更，导致每次都全量重建。
    same_model = bool(previous) and previous.embed_model == embed_model
    dim_ok = not embed_dim or (previous is not None and previous.embed_dim == embed_dim)
    if same_model and dim_ok:
        need_embed = previous.changed_docs(current_hashes)
    else:
        # 无历史 manifest，或换了模型/维度 -> 全量重建
        if previous:
            logger.info(
                "索引失效：模型 %s@%s -> %s@%s，全量重建",
                previous.embed_model,
                previous.embed_dim,
                embed_model,
                embed_dim,
            )
        need_embed = {n.id for n in accepted}

    done_ids: set[int] = set()
    resumed = False
    if resume:
        done_ids = await _load_progress()
        if done_ids:
            resumed = True
            logger.info("检测到上次构建进度，已完成 %s 篇，继续处理", len(done_ids))

    # 进度是「上次未跑完」的记录，不是「已完成」的记录。
    # 如果本次需要的文档里没有一篇在进度表里，说明进度属于另一次构建
    # （或上一次其实已经跑完了），此时续跑没有意义，直接清掉避免把所有文档误判为已完成。
    if done_ids and not (done_ids & need_embed):
        logger.info("已有进度与本次待处理文档无交集，判定为陈旧进度并清除")
        await _clear_progress()
        done_ids = set()
        resumed = False

    # 复用上一版的向量
    reused_vectors: dict[int, list[float]] = {}
    reused = 0
    if previous:
        # 从进程内的向量池复用，避免每次都重新解析 .npy
        pool = _load_vector_pool()
        logger.debug("复用池加载 %s 个向量，待复用文档 %s 篇", len(pool), len(accepted))
        # 注意 accepted 是 News 对象列表，必须取 .id 才能与池的键匹配
        for news in accepted:
            doc_id = news.id
            if doc_id in need_embed or doc_id in done_ids:
                continue
            vec = pool.get(doc_id)
            if vec is not None:
                reused_vectors[doc_id] = vec
                reused += 1

        if reused == 0:
            # 回落到逐篇读取（兼容只有 Redis 没有文件的旧数据）
            from config.cache_conf import redis_client
            import json

            for doc_id in accepted:
                if doc_id in need_embed or doc_id in done_ids:
                    continue
                try:
                    raw = await redis_client.get(f"ai:vec:{doc_id}")
                except Exception:
                    raw = None
                if raw:
                    try:
                        reused_vectors[doc_id] = json.loads(raw)
                        reused += 1
                    except json.JSONDecodeError:
                        pass

    # 只对需要处理的文档做向量化（断点续跑会跳过已完成的）
    pending = [
        n for n in accepted if n.id in need_embed and n.id not in done_ids
    ]
    logger.info(
        "索引构建：语料 %s 篇，需向量化 %s 篇，复用 %s 篇，已完成跳过 %s 篇",
        len(accepted), len(pending), reused, len(done_ids),
    )

    embedded = 0
    if pending:
        texts = [build_document(n) for n in pending]
        try:
            vectors = await embed_texts(texts, use_cache=True)
        except Exception as exc:
            logger.error("向量化失败: %s", exc)
            return BuildResult(
                manifest=IndexManifest(),
                report=report,
                total=len(accepted),
                embedded=0,
                reused=reused,
                skipped=report.skipped,
                duration_ms=(time.perf_counter() - started) * 1000,
                errors=[f"向量化失败: {exc}"],
            )

        from config.cache_conf import redis_client
        import json

        try:
            async with redis_client.pipeline(transaction=False) as pipe:
                for news, vector in zip(pending, vectors):
                    pipe.setex(f"ai:vec:{news.id}", 86400 * 7, json.dumps(vector))
                await pipe.execute()
        except Exception as exc:
            logger.warning("写入向量缓存失败: %s", exc)

        for news, vector in zip(pending, vectors):
            reused_vectors[news.id] = vector
            embedded += 1
            done_ids.add(news.id)

        if resume:
            await _save_progress(done_ids)

    # 组装最终矩阵。顺序必须与 accepted 一致，下标才能映射回新闻
    matrix_vectors = []
    ordered_news_ids = []
    missing = []
    for news in accepted:
        vec = reused_vectors.get(news.id)
        if vec is None:
            missing.append(news.id)
            continue
        matrix_vectors.append(vec)
        ordered_news_ids.append(news.id)

    if missing:
        # 进度里记为「已完成」但没拿到向量的文档，说明上一轮是在写向量之前就中断了。
        # 这种情况不能默默跳过 —— 否则索引会少数据且没人知道。
        logger.error(
            "有 %s 篇既未向量化也无缓存向量，其中 %s 篇被进度标记为已完成 —— "
            "说明上次构建在写入向量前中断，本次不能复用。缺失 id: %s",
            len(missing), len(set(missing) & done_ids), missing[:10],
        )
        done_ids -= set(missing)

    matrix = VectorMatrix(matrix_vectors)
    dim = matrix.dim

# 存索引矩阵与清单。ordered_news_ids 显式传下去——
    # 矩阵第 i 行必须对应第 i 个 news_id，顺序错了整个索引就是错的。
    await _store_matrix(matrix, ordered_news_ids, fingerprint)
    if missing:
        # 残缺矩阵仍然落盘，但要明确标记。
        # 不落盘更糟：下次构建会读到「上一次完整的矩阵」当作复用来源，
        # 而实际语料已经变了 —— 那才是真正的错位。
        logger.warning("本次索引不完整（缺 %s 篇），落盘矩阵已标记为待重建", len(missing))
    save_matrix_file(matrix, ordered_news_ids, fingerprint)
    clear_vector_pool()

    manifest = IndexManifest(
        index_id=make_index_id(embed_model or "unknown", dim, fingerprint),
        embed_model=embed_model,
        embed_dim=dim,
        corpus_fingerprint=fingerprint,
        document_count=len(matrix_vectors),
        doc_hashes=current_hashes,
        built_at=now_iso(),
        build_duration_ms=round((time.perf_counter() - started) * 1000, 2),
        stats={
            "inputDocs": len(docs),
            "acceptedDocs": len(accepted),
            "skippedDocs": report.skipped,
            "embedded": embedded,
            "reused": reused,
            "resumed": resumed,
            "validationErrors": len(report.errors),
            "validationWarnings": len(report.warnings),
        },
    )
    await save_manifest(manifest)
    # 文件是真相来源：Redis 可用时是快路径，不可用时靠它保住增量判定的依据
    save_manifest_file(manifest)

    if not missing:
        await _clear_progress()
    else:
        # 有缺失时不能留进度文件：它记录的是「已向量化」，但向量没落盘成功。
        # 下次构建会把这些文档全部误判为已完成，从而跳过它们——
        # 结果是索引缺数据且没有任何提示，这是最糟的失败方式。
        logger.warning(
            "存在缺失文档，清除构建进度以免下次误判为已完成：%s", missing[:10]
        )
        await _clear_progress()

    result = BuildResult(
        manifest=manifest,
        report=report,
        total=len(accepted),
        embedded=embedded,
        reused=reused,
        skipped=report.skipped,
        duration_ms=manifest.build_duration_ms,
        resumed=resumed,
    )
    logger.info(
        "索引构建完成：%s，用时 %.0fms（向量化 %s，复用 %s）",
        manifest.index_id, result.duration_ms, embedded, reused,
    )
    return result


async def _store_matrix(matrix: VectorMatrix, news_ids: list[int], fingerprint: str) -> None:
    """
    把矩阵和「行号 -> news_id」的映射一起存下来。

    news_ids 的顺序必须与 matrix 的行顺序严格对应，这是索引正确性的前提，
    所以由调用方直接传入组装矩阵时用的那个顺序，而不是在这里重新推导。
    """
    import json

    from config.cache_conf import redis_client

    if len(news_ids) != len(matrix):
        raise ValueError(
            f"索引行数与 news_ids 不一致: {len(matrix)} vs {len(news_ids)}，顺序可能已错乱"
        )

    payload = {"fingerprint": fingerprint, "matrix": matrix.to_list(), "newsIds": news_ids}
    try:
        await redis_client.setex("ai:index:matrix", 86400 * 7, json.dumps(payload))
    except Exception as exc:
        logger.warning("写入索引矩阵到 Redis 失败（文件已是真相来源，不影响构建）: %s", exc)


def load_index_matrix() -> tuple[VectorMatrix, list[int]] | None:
    """
    读回离线构建的索引。

    返回 None 表示没有可用的离线索引，调用方应回退到懒加载路径。
    """
    import asyncio
    import json

    from config.cache_conf import redis_client

    async def _read():
        try:
            return await redis_client.get("ai:index:matrix")
        except Exception:
            return None

    try:
        raw = asyncio.run(_read())
    except RuntimeError:
        # 已在事件循环里，直接用裸 Redis 同步客户端不适用，返回 None 让调用方降级
        return None

    if not raw:
        return None
    try:
        payload = json.loads(raw)
        return VectorMatrix(payload["matrix"]), payload["newsIds"]
    except Exception as exc:
        logger.warning("读取索引矩阵失败: %s", exc)
        return None