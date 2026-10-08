"""
新闻语料混合检索：BM25（词面）+ 向量（语义）+ RRF 融合。

为什么要两路
------------
两路检索的失败模式互补，这是混合检索（Hybrid Search）的核心价值：

- **BM25 词面**：对专有名词、数字、缩写特别准。「GDP 增长5.2%」这种查询，
  只有词面能精确命中，向量反而会把它和「经济形势」泛泛地拉近。
- **向量语义**：能跨越表述差异。「高铁延长运行时间」和「动车组新增班次」
  零字面重叠，BM25 分数是 0，但向量能判定它们语义相近。

实测中常见的结论是：只用向量会漏掉专名查询，只用 BM25 会漏掉同义表述，

> 本项目自身的实测（完整数据见 ai/eval/README.md）：
> BM25 单路 Recall@5 93.9%。向量路的真实增益需要有效 API Key 才能测 ——
> 无语义的假向量会让 MRR 从 90.5% 掉到 50.7%，说明融合机制在单路退化时
> 缺乏保护，这也是已记录的待改进项。
融合后召回率显著高于任一单路。

RRF 融合
--------
两路的分数不能直接相加 —— BM25 分数无上界，向量余弦在 [-1,1]，
量纲完全不同。RRF（Reciprocal Rank Fusion）只依赖**排名**不看分数，
公式是：
    score(d) = Σ 1 / (k + rank_i(d))
k 取 60 是原论文的经验值，作用是压制头部的过度影响，
避免「BM25 排第一且分数极高」的文档把其他文档的贡献淹没。

为什么不用向量数据库
--------------------
语料 403 条，全量扫描用 numpy 矩阵乘只要 0.26ms（实测），引 FAISS/Milvus
只会增加部署与运维成本而没有收益。真正需要 ANN 索引（HNSW/IVF）时是百万级以上，
届时替换 VectorMatrix.search() 的内部实现即可，调用方无需改动。

缓存层次
--------
检索有两处重开销，都随语料缓存而非每次重算：
- BM25 索引构造：实测 74.3ms/查询（tokenize 38.4ms + 建索引打分 35.9ms）
- 语料向量读取：逐条 GET 是 404 次 Redis 往返（实测 1497ms）
两者都由语料指纹做键，语料不变时直接复用。
"""
import json
import re
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.config import (
    INDEX_CACHE_TTL,
    MAX_DOC_CHARS,
    MIN_BM25_SCORE,
    MIN_FUSION_SCORE,
    MIN_VECTOR_SIM,
    RRF_K,
    SINGLE_PATH_WEIGHT,
    TOP_K_BM25,
    TOP_K_FINAL,
    TOP_K_VECTOR,
    TRACE_ENABLED,
    TRACE_TTL,
)
from ai.embeddings import VectorMatrix, embed_query, embed_texts
from ai.ingest.manifest import IndexManifest
from ai.ingest.validate import corpus_fingerprint, document_content_hash
from config.cache_conf import redis_client
from models.news import News
from utils.logging_conf import get_logger

logger = get_logger(__name__)

_CORPUS_CACHE_KEY = "ai:corpus:news"
_CORPUS_VECTORS_CACHE_KEY = "ai:corpus:vectors"
# 语料向量矩阵的进程内缓存：构造 numpy 矩阵有固定开销，
# 同一份语料连续多次查询不该重复构造。用 (语料版本, 长度) 做键。
_matrix_cache: dict[str, object] = {}
_MATRIX_CACHE_MAX = 2  # 只留最近几个版本，防止语料频繁变动时内存膨胀

# BM25 索引的进程内缓存：语料不变时索引也完全不变，缓存没有正确性风险
_bm25_cache: dict[str, tuple] = {}
_INDEX_CACHE_MAX = 2

# 连续的中日韩统一表意文字
_CJK = r"\u4e00-\u9fff\u3400-\u4dbf"
_CJK_RUN = re.compile(f"[{_CJK}]+")
_ASCII_WORD = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """
    分词。

    刻意不引入 jieba：为了中文分词多装一个依赖不划算，而且语料是新闻这种
    规整文本。用「单字 + 相邻二元组」的中文检索常用基线就能拿到不错的效果：
        「中国高铁」 -> 中 国 高 铁 / 中国 国高 高铁
    这样单字查询（「高铁」）和组合查询（「中国高铁」）都能命中。
    ASCII 单词和数字整体保留，因为「GDP」「5.2%」这类专名必须当成完整 token。

    试过改成「只保留二元组+三元组、不索引单字」，理由是单字歧义大。
    **实测结论：这个改动没有收益，已回退。**
    - 触发改动的怀疑用例是「搜人工智能，『人形机器人表演』排第一」。
      查了原文才发现那篇正文确实写着「人工智能与机器人技术深度融合」，
      是真实内容匹配，不是分词噪音 —— 最初的判断就是错的。
    - 用「高铁 / 科技 / 安全 / 发展」做探针对比，两种分词的 top3 结果完全一致。
    - 而代价是实的：MRR@5 从 90.5% 降到 88.4%，Recall@5 不变。
    没收益、有退化，不该留。单字的歧义在真实查询里被 IDF 自然压住了。
    """
    if not text:
        return []

    lowered = text.lower()
    tokens: list[str] = []

    for match in _ASCII_WORD.finditer(lowered):
        tokens.append(match.group())

    cjk_runs = _CJK_RUN.findall(lowered)
    for run in cjk_runs:
        tokens.extend(list(run))
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))

    return tokens


def build_document(news: News) -> str:
    """
    把新闻拼成用于检索的文本。

    标题权重最高：用户在提问里几乎一定会带上标题里的核心词，
    而正文里同样的词可能只出现一次，被长文本稀释。
    所以标题重复两遍，正文截断。
    """
    parts = [
        news.title or "",
        news.title or "",  # 标题加权
        news.description or "",
        (news.content or "")[:MAX_DOC_CHARS],
    ]
    return "\n".join(p for p in parts if p)


@dataclass
class RetrievedNews:
    """一条召回的新闻及其检索元信息"""

    news_id: int
    title: str
    excerpt: str
    category_id: int
    publish_time: datetime | None = None
    bm25_rank: int | None = None
    vector_rank: int | None = None
    vector_score: float | None = None
    fusion_score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)

    def to_citation(self) -> dict:
        """
        转成给前端的溯源信息。

        datetime 在这里就转成 ISO 字符串：SSE 事件走的是裸 json.dumps，
        直接把 datetime 塞进去会抛 TypeError。
        """
        return {
            "newsId": self.news_id,
            "title": self.title,
            "excerpt": self.excerpt,
            "categoryId": self.category_id,
            # 序列化放在 DTO 边界做，避免把 datetime 泄漏到 json.dumps 里
            "publishTime": self.publish_time.isoformat() if self.publish_time else None,
            "fusionScore": round(self.fusion_score, 6),
            "matchedTerms": self.matched_terms,
            "bm25Rank": self.bm25_rank,
            "vectorRank": self.vector_rank,
        }


async def load_corpus(db: AsyncSession, use_cache: bool = True) -> list[News]:
    """
    载入检索语料（全部新闻），带 Redis 缓存。

    新闻列表接口自己有分级缓存，但那是给前端用的分页视图；
    检索需要一次性拿全量，用独立 key 避免互相污染。
    """
    if use_cache:
        try:
            raw = await redis_client.get(_CORPUS_CACHE_KEY)
            if raw:
                return _rows_from_cache(json.loads(raw))
        except Exception as exc:
            logger.warning("读取语料缓存失败，回源数据库: %s", exc)

    stmt = select(News).order_by(News.id)
    result = await db.execute(stmt)
    rows = list(result.scalars().all())

    if use_cache:
        try:
            payload = [
                {
                    "id": n.id,
                    "title": n.title,
                    "description": n.description,
                    "content": n.content,
                    "category_id": n.category_id,
                    "publish_time": n.publish_time.isoformat() if n.publish_time else None,
                }
                for n in rows
            ]
            await redis_client.setex(
                _CORPUS_CACHE_KEY, INDEX_CACHE_TTL, json.dumps(payload, ensure_ascii=False)
            )
        except Exception as exc:
            logger.warning("写入语料缓存失败: %s", exc)

    return rows


def _rows_from_cache(payload: list[dict]) -> list[News]:
    """把缓存里的 dict 还原成 News ORM 对象，保持与其他 crud 层一致的返回类型"""
    rows = []
    for item in payload:
        rows.append(
            News(
                id=item["id"],
                title=item["title"],
                description=item.get("description"),
                content=item.get("content"),
                category_id=item["category_id"],
                publish_time=(
                    datetime.fromisoformat(item["publish_time"]) if item.get("publish_time") else None
                ),
            )
        )
    return rows


def bm25_search(
    corpus: list[News], query: str, top_k: int = TOP_K_BM25
) -> list[tuple[int, float, list[str]]]:
    """
    BM25 词面检索。

    返回 [(新闻下标, 分数, 命中词列表)]，按分数降序。

    两道过滤：

    1. 分数 <= 0 丢弃 —— BM25 只处理正相关，保留零分/负分只会引入噪声。
    2. **必须至少命中一个多字词** —— 中文单字区分度太低，「量子计算机」和
       「人工智能」共享「计」「算」这类字，加上「的/中/国」这种高频字，
       任何两条新闻之间都能凑出几个共同单字。只靠单字命中的结果全是误召回，
       会让无关问题也走上LLM，然后靠模型去编。所以必须命中二元组或 ASCII 词。

    **写小语料的单元测试时注意**：rank_bm25 的 IDF 在查询词出现在过半文档时
    会变成负数，配合第 1 条过滤会把所有结果清空，表现为「BM25 永远返回空」。
    只有 3~8 篇语料时几乎必然触发 —— 造测试数据要补够无关文档，
    别把这个现象误判成检索功能坏了。
    """
    if not corpus:
        return []

    bm25, docs = get_bm25_index(corpus)
    if bm25 is None:
        return []

    scores = bm25.get_scores(tokenize(query))

    query_terms = set(tokenize(query))
    # 查询里「有信息量」的词：多字词才有区分度
    discriminative = {t for t in query_terms if len(t) > 1}

    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    results = []
    for idx, score in ranked:
        if score <= 0:
            break
        doc_terms = set(docs[idx])
        overlap = doc_terms & query_terms
        # 没有命中任何多字词 -> 判定为单字噪声，不返回
        if discriminative and not (overlap & discriminative):
            continue
        if not discriminative and not overlap:
            continue
        results.append((idx, float(score), sorted(overlap)[:6]))
        if len(results) >= top_k:
            break
    return results


async def vector_search(
    corpus: list[News], query: str, top_k: int = TOP_K_VECTOR
) -> list[tuple[int, float]]:
    """
    向量语义检索，返回 [(新闻下标, 余弦相似度)]，按相似度降序。

    语料向量矩阵随语料缓存，避免每次查询都重新读取 403 条向量
    （实测逐条 GET 是 404 次往返 / 1497ms）。

    优先使用离线构建的索引（ai/ingest），它是显式构建产物，
    省去运行时拼装向量矩阵的开销，且带 manifest 可校验兼容性。

    向量化失败时返回空列表并降级为纯 BM25 —— 语义检索是增强项，
    不该因为它挂了就让整个问答不可用。
    """
    if not corpus:
        return []

    # 查询向量必须先拿到：它只有一条，缓存价值低但必须有
    try:
        query_vector = await embed_query(query)
    except Exception as exc:
        logger.warning("查询向量化失败，降级为纯 BM25: %s", exc)
        return []

    # 1) 离线索引
    offline = await _load_offline_index(corpus)
    if offline is not None:
        try:
            return offline.search(query_vector, top_k)
        except ValueError as exc:
            logger.warning("离线索引维度不匹配，回退到运行时构建: %s", exc)

    # 2) 运行时构建（语料首次被检索时懒加载）
    try:
        runtime_matrix = await get_corpus_vector_matrix(corpus)
        return runtime_matrix.search(query_vector, top_k)
    except Exception as exc:
        logger.warning("向量检索不可用，降级为纯 BM25: %s", exc)
        return []


async def _load_offline_index(corpus: list[News]):
    """
    加载离线构建的索引，并校验它对当前语料与模型是否仍然有效。

    三重校验，缺一不可：
    1. 语料指纹一致 —— 语料变了索引就是旧的
    2. embedding 模型一致 —— 换了模型旧向量不可用
    3. 矩阵行数与 news_ids 一致 —— 兜底，防止文件被部分覆盖导致错位

    任一项不满足就返回 None，由调用方回退到运行时构建路径。
    """
    from ai.config import DASHSCOPE_EMBED_MODEL
    from ai.ingest.storage import load_manifest_file, load_matrix_file

    try:
        stored = load_matrix_file()
    except Exception as exc:
        logger.debug("读取离线索引失败: %s", exc)
        return None

    if stored is None:
        return None

    matrix, news_ids, fingerprint = stored
    if not len(matrix) or len(matrix) != len(news_ids):
        logger.info("离线索引行数与 id 数不一致，视为损坏并回退")
        return None

    current_fp = _corpus_fingerprint(corpus)
    if fingerprint and fingerprint != current_fp:
        logger.info("离线索引语料指纹已过期（%s != %s），回退到运行时构建", fingerprint, current_fp)
        return None

    manifest_data = load_manifest_file()
    if manifest_data:
        manifest = IndexManifest.from_dict(manifest_data)
        compatible, reason = manifest.is_compatible_with(DASHSCOPE_EMBED_MODEL, matrix.dim)
        if not compatible:
            logger.info("离线索引不可用：%s，回退到运行时构建", reason)
            return None

    logger.debug("使用离线索引：%s 篇，维度 %s", len(matrix), matrix.dim)
    return matrix



def get_bm25_index(corpus: list[News]):
    """
    取 BM25 索引，索引随语料缓存。

    为什么必须缓存
    --------------
    BM25Okapi 构造时要遍历全部语料做词频统计。之前每次查询都重建，
    实测 403 条语料下 tokenize 38.4ms + 建索引打分 35.9ms = 74.3ms，
    而查询本身的打分只需要几毫秒 —— 90% 的时间花在准备索引上。
    语料不变时索引也完全不变，缓存没有正确性风险。

    返回 (BM25Okapi 实例, 分词后的文档列表)；语料为空时返回 (None, [])。
    """
    fingerprint = _corpus_fingerprint(corpus)

    if fingerprint in _bm25_cache:
        return _bm25_cache[fingerprint]

    from rank_bm25 import BM25Okapi

    docs = [tokenize(build_document(n)) for n in corpus]
    if not any(docs):
        return None, docs

    bm25 = BM25Okapi(docs)

    while len(_bm25_cache) >= _INDEX_CACHE_MAX:
        _bm25_cache.pop(next(iter(_bm25_cache)))
    _bm25_cache[fingerprint] = (bm25, docs)
    return bm25, docs


def clear_index_cache() -> None:
    """清空 BM25 索引的进程内缓存（语料内容变更后可调用）"""
    _bm25_cache.clear()


def _corpus_fingerprint(corpus: list[News]) -> str:
    """
    语料指纹：内容变化才会导致缓存失效。

    与 ai.ingest 的 corpus_fingerprint 是同一个算法 —— 两边必须一致，
    否则检索侧永远认为离线索引已过期，离线构建就白做了。
    """
    return corpus_fingerprint(corpus)


async def get_corpus_vector_matrix(corpus: list[News], force_rebuild: bool = False):
    """
    取语料向量矩阵，优先进程内缓存，其次 Redis，最后调 embedding API。

    三层缓存的原因：
    1. 进程内 —— numpy 矩阵构造有开销，构造一次约几毫秒
    2. Redis —— 避免重复调付费 API
    3. API —— 最终来源

    维度不一致时（换了 embedding 模型）自动失效重建，见 vector_search 的处理。
    """
    from ai.embeddings import VectorMatrix

    fingerprint = _corpus_fingerprint(corpus)

    if not force_rebuild and fingerprint in _matrix_cache:
        return _matrix_cache[fingerprint]

    vectors: list[list[float]] | None = None
    if not force_rebuild:
        try:
            raw = await redis_client.get(_CORPUS_VECTORS_CACHE_KEY)
            if raw:
                payload = json.loads(raw)
                # 缓存必须和当前语料一一对应，否则下标映射会错位
                if payload.get("fingerprint") == fingerprint:
                    vectors = payload.get("vectors")
        except Exception as exc:
            logger.warning("读取语料向量缓存失败: %s", exc)

    if vectors is None:
        documents = [build_document(n) for n in corpus]
        # batch_lookup=True：403 条从 404 次往返降到 1 次
        vectors = await embed_texts(documents, batch_lookup=True)
        try:
            await redis_client.setex(
                _CORPUS_VECTORS_CACHE_KEY,
                INDEX_CACHE_TTL,
                json.dumps(
                    {"fingerprint": fingerprint, "vectors": vectors}, ensure_ascii=False
                ),
            )
        except Exception as exc:
            logger.warning("写入语料向量缓存失败: %s", exc)

    matrix = VectorMatrix(vectors)
    if force_rebuild or _matrix_cache:
        # 只保留少量版本，防止语料频繁变动导致内存无限增长
        while len(_matrix_cache) >= _MATRIX_CACHE_MAX:
            _matrix_cache.pop(next(iter(_matrix_cache)))
    _matrix_cache[fingerprint] = matrix
    return matrix


async def invalidate_corpus_index() -> None:
    """清掉语料向量缓存（Redis + 进程内），用于换了模型或语料结构变化时"""
    _matrix_cache.clear()
    try:
        await redis_client.delete(_CORPUS_VECTORS_CACHE_KEY)
        logger.info("语料向量缓存已清除")
    except Exception as exc:
        logger.warning("清除语料向量缓存失败: %s", exc)


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = RRF_K,
    single_path_weight: float = SINGLE_PATH_WEIGHT,
) -> dict[int, float]:
    """
    RRF 融合：score(d) = Σ 1 / (k + rank_i(d))，rank 从 1 开始。

    输入是若干「按相关性降序排列的新闻下标列表」。

    single_path_weight 的作用：只有**一路**召回的文档，证据强度天然弱于
    两路都召回的文档。
    单路命中的最高分只有 1/(k+1) ≈ 0.0164，而两路都命中的最高分可达 0.033，
    两者差一个量级。如果不区分，只剩一路可用时（向量服务故障降级），
    几乎任何弱相关命中都能越过置信度阈值，然后被送去让模型编造。
    所以对单路命中的分数乘一个惩罚系数，把「一路有点相关」和
    「两路都相关」区分开，置信度阈值就能卡在两者之间。
    """
    total_lists = len(ranked_lists)
    scores: dict[int, float] = {}
    hits: dict[int, int] = {}

    for ranking in ranked_lists:
        for rank, doc_index in enumerate(ranking, start=1):
            scores[doc_index] = scores.get(doc_index, 0.0) + 1.0 / (k + rank)
            hits[doc_index] = hits.get(doc_index, 0) + 1

    if single_path_weight < 1.0 and total_lists > 1:
        for doc_index, count in hits.items():
            if count < total_lists:
                scores[doc_index] *= single_path_weight

    return scores


class RetrievalTrace:
    """
    单次检索的链路追踪。

    排查「为什么这个查询返回了这个结果」时，需要能看到：
    1. 各阶段耗时（慢在哪）
    2. 每路召回了多少候选
    3. 最终结果的分数构成（哪路贡献的、是不是单路命中被降权）
    4. 置信度判定依据

    这些数据写结构化日志，同时在开启 trace 开关时落到 Redis，
    便于事后按 request_id 回查。
    """

    __slots__ = (
        "query", "corpus_size", "bm25_ms", "vector_ms", "fuse_ms",
        "bm25_hits", "vector_hits", "vector_available", "top_score",
        "threshold", "confident", "results", "fingerprint",
    "gate_mode",
    )

    def __init__(self, query: str):
        self.query = query
        self.corpus_size = 0
        self.bm25_ms = 0.0
        self.vector_ms = 0.0
        self.fuse_ms = 0.0
        self.bm25_hits = 0
        self.vector_hits = 0
        self.vector_available = False
        self.top_score = 0.0
        self.threshold = MIN_FUSION_SCORE
        # 记录这次用的是哪种置信判据。降级到 RRF 时必须能看出来，
        # 否则「向量服务故障导致判据变弱」会被误读成「检索质量下降」。
        self.gate_mode = "rrf"
        self.confident = False
        self.fingerprint = ""
        self.results: list[dict] = []

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "corpusSize": self.corpus_size,
            "corpusFingerprint": self.fingerprint,
            "timing": {
                "bm25Ms": round(self.bm25_ms, 2),
                "vectorMs": round(self.vector_ms, 2),
                "fuseMs": round(self.fuse_ms, 2),
                "totalMs": round(self.bm25_ms + self.vector_ms + self.fuse_ms, 2),
            },
            "recall": {
                "bm25Hits": self.bm25_hits,
                "vectorHits": self.vector_hits,
                "vectorAvailable": self.vector_available,
                "degradedToBm25": not self.vector_available,
            },
            "decision": {
                "topScore": round(self.top_score, 6),
                "threshold": self.threshold,
                "confident": self.confident,
                "willRefuse": not self.confident,
        "gateMode": self.gate_mode,
            },
            "results": self.results,
        }

    def log_line(self) -> str:
        return (
            f"检索 '{self.query}' 语料{self.corpus_size}条 "
            f"| BM25 {self.bm25_hits}档 {self.bm25_ms:.1f}ms "
            f"| 向量 {self.vector_hits}档 {self.vector_ms:.1f}ms"
            f"{'' if self.vector_available else '(降级)'}"
            f" | 融合 {self.fuse_ms:.1f}ms "
            f"| 最高分 {self.top_score:.4f} vs 阈值 {self.threshold} "
            f"-> {'作答' if self.confident else '拒答'}"
        )


async def _write_trace(trace: RetrievalTrace) -> None:
    """把 trace 写结构化日志；开关打开时同时落 Redis"""
    payload = trace.to_dict()
    logger.info("rag_trace %s", json.dumps(payload, ensure_ascii=False))

    if not TRACE_ENABLED:
        return
    try:
        from utils.logging_conf import get_request_id

        key = f"ai:trace:{get_request_id()}"
        # 必须 await：redis-py 的命令返回协程，不 await 等于什么都没做
        # （这个 bug 曾导致 trace 静默丢失：日志正常但 Redis 里没有任何数据）
        await redis_client.setex(key, TRACE_TTL, json.dumps(payload, ensure_ascii=False))
    except Exception as exc:
        # trace 写不进去不能影响主流程 —— 它是可观测性手段，不是业务逻辑
        logger.debug("写检索 trace 失败: %s", exc)


async def retrieve(
    db: AsyncSession,
    query: str,
    top_k: int = TOP_K_FINAL,
    use_cache: bool = True,
    trace: RetrievalTrace | None = None,
    history: list[tuple[str, str]] | None = None,
) -> tuple[list[RetrievedNews], bool]:
    """
    对新闻语料执行混合检索。

    返回 (召回列表, 是否达到置信度阈值)。
    第二个返回值供调用方判断：没达到就该直接拒答，不要调 LLM 去编。

    history 传入上一轮问答后，检索词会带上轮的关键词。
    这一步是必需的：多轮追问（「那人均呢」）本身没有检索词，
    只靠原句检索必然召回不到任何东西 —— 上下文只喂给生成层是不够的。
    """
    trace = trace or RetrievalTrace(query)
    import time as _time

    t_all = _time.perf_counter()
    corpus = await load_corpus(db, use_cache=use_cache)
    if not corpus:
        logger.warning("新闻语料为空，跳过检索: %s", query)
        trace.corpus_size = 0
        trace.confident = False
        await _write_trace(trace)
        return [], False

    trace.corpus_size = len(corpus)
    trace.fingerprint = _corpus_fingerprint(corpus)
    trace.threshold = MIN_FUSION_SCORE

    # 检索词改写：把上一轮的问题并进当前 query。
    #
    # 为什么必须做：追问句本身几乎没有检索词。「那人均呢」拿去搜 BM25 和
    # 向量都搜不到 GDP 相关内容 —— 实测会直接触发拒答，而用户看到的是
    # 「这个系统听不懂追问」。上下文只喂给生成层是不够的，检索层也要用。
    search_query = query
    if history:
        previous_question = history[-1][0]
        # 只带上一轮的问题，不带回答：回答里的字会大量稀释本轮的检索意图
        # （实测回答里有 [1][2] 和整段叙述，拼进去会把召回带偏）
        if previous_question and previous_question.strip():
            search_query = f"{previous_question} {query}"
            logger.info("多轮检索词改写: %r -> %r", query, search_query)

    t0 = _time.perf_counter()
    bm25_hits = bm25_search(corpus, search_query)
    trace.bm25_ms = (_time.perf_counter() - t0) * 1000
    trace.bm25_hits = len(bm25_hits)

    t0 = _time.perf_counter()
    vector_hits = await vector_search(corpus, search_query)
    trace.vector_ms = (_time.perf_counter() - t0) * 1000
    trace.vector_hits = len(vector_hits)
    # 有 BM25 命中但向量一路空 = 向量服务不可用，已降级
    trace.vector_available = bool(vector_hits)

    if not bm25_hits and not vector_hits:
        logger.info("BM25 与向量检索均无命中: %s", query)
        await _write_trace(trace)
        return [], False

    t0 = _time.perf_counter()
    bm25_rank = {idx: i + 1 for i, (idx, _, _) in enumerate(bm25_hits)}
    bm25_meta = {idx: (score, terms) for idx, score, terms in bm25_hits}
    vector_rank = {idx: i + 1 for i, (idx, _) in enumerate(vector_hits)}
    vector_score = {idx: score for idx, score in vector_hits}

    fused = reciprocal_rank_fusion([list(bm25_rank.keys()), list(vector_rank.keys())])

    # 召回后按内容去重。
    #
    # 真实语料里有 15 组共 33 篇标题/描述/正文完全相同的新闻。索引侧
    # 已经把 content_hash 改成一对多映射，33 篇都能进索引；但它们在检索
    # 侧是同一份知识，重复召回会挤占 top_k 名额 —— 实测出现过拒答消息里
    # 连续列出三条《中国数字阅读用户规模超5.3亿》，既浪费额度又难看。
    #
    # 保留每组里融合分最高的那条（也就是最相关的），其余丢弃。
    results: list[RetrievedNews] = []
    seen_hashes: set[str] = set()
    for idx, fusion_score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
        if len(results) >= top_k:
            break
        news = corpus[idx]
        content_hash = document_content_hash(news.title, news.description, news.content)
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        _, terms = bm25_meta.get(idx, (0.0, []))
        excerpt = (news.content or news.description or "")[:MAX_DOC_CHARS]
        results.append(
            RetrievedNews(
                news_id=news.id,
                title=news.title,
                excerpt=excerpt,
                category_id=news.category_id,
                publish_time=news.publish_time,
                bm25_rank=bm25_rank.get(idx),
                vector_rank=vector_rank.get(idx),
                vector_score=vector_score.get(idx),
                fusion_score=fusion_score,
                matched_terms=terms,
            )
        )
    trace.fuse_ms = (_time.perf_counter() - t0) * 1000

    best_score = results[0].fusion_score if results else 0.0

# 置信门控：优先向量余弦，向量路不可用时退回 BM25 分数，RRF 只作最后兜底。
    #
    # 三级判据的原因：
    #  1. 向量余弦（主）：负例中位 0.48 / 正例中位 0.80，有区分度。
    #     原来只看 RRF 分数，实测负例误召回率 100% —— 阈值形同虚设，
    #     因为 RRF 只用排名、丢掉分数幅度，403篇小语料里所有查询都顶到上限 0.0328。
    #  2. BM25 分数（降级）：向量服务不可用时 RRF 恒为 0.0082，会把全部用例拒答掉，
    #     所以降级时改用有真实幅度的 BM25 分数判断。
    #  3. RRF 分数（兜底）：两者都拿不到时才用，语义最弱。
    best_vector_sim = results[0].vector_score if results else None
    top_bm25 = max(bm25_meta.values())[0] if bm25_meta else None

    if best_vector_sim is not None:
        confident = best_vector_sim >= MIN_VECTOR_SIM
        trace.gate_mode = "vector_cosine"
        trace.top_score = best_vector_sim
        trace.threshold = MIN_VECTOR_SIM
    elif top_bm25 is not None:
        confident = top_bm25 >= MIN_BM25_SCORE
        trace.gate_mode = "bm25_score"
        trace.top_score = top_bm25
        trace.threshold = MIN_BM25_SCORE
    else:
        confident = False
        trace.gate_mode = "none"
        trace.top_score = 0.0
        trace.threshold = 0.0

    trace.confident = confident
    trace.results = [
        {
            "newsId": r.news_id,
            "title": r.title,
            "fusionScore": round(r.fusion_score, 6),
            "bm25Rank": r.bm25_rank,
            "vectorRank": r.vector_rank,
            "vectorScore": round(r.vector_score, 4) if r.vector_score is not None else None,
            "matchedTerms": r.matched_terms,
            # 只被一路命中 -> 被降权过，标记出来便于排查「为什么分数这么低」
            "singlePath": (r.bm25_rank is None) != (r.vector_rank is None),
        }
        for r in results
    ]

    logger.info("rag_trace %s", trace.log_line())
    await _write_trace(trace)
    return results, confident


def invalidate_corpus_cache() -> None:
    """
    语料变更后手动失效缓存。

    目前新闻没有写接口，所以只在文档里说明用法；
    将来加了新闻管理接口就应该在写入成功后调用。
    """
    logger.info("如需强制刷新语料缓存，请手动删除 Redis key: %s", _CORPUS_CACHE_KEY)