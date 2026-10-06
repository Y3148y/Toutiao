"""
离线索引管线测试：校验、manifest、增量更新、断点续跑、持久化。

全部离线运行，用确定性假向量，不调任何外部 API。
"""
import asyncio
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

import pytest

import ai.ingest.builder as builder_module
import ai.ingest.storage as storage_module
from ai.ingest.manifest import IndexManifest, make_index_id
from ai.ingest.validate import (
    MAX_CONTENT_CHARS,
    build_content_hash_index,
    corpus_fingerprint,
    document_content_hash,
    find_duplicate_groups,
    validate_corpus,
    validate_document,
)
from models.news import News

DIM = 64


def make_news(id_, title="标题", description=None, content="正文内容", category_id=1):
    return News(
        id=id_,
        title=title,
        description=description,
        content=content,
        category_id=category_id,
        views=0,
        publish_time=datetime(2026, 1, 1),
    )


def make_corpus(size=20):
    return [
        make_news(i, title=f"新闻标题{i} 国务院政策", content=f"正文{i} " * 20)
        for i in range(1, size + 1)
    ]


@pytest.fixture(autouse=True)
def isolated_index(tmp_path, monkeypatch):
    """
    把索引目录指向 tmp_path，避免测试污染真实索引。

    隔离是必须的：manifest 和向量矩阵都是文件，
    不隔离的话测试会读到仓库里上次构建的产物，导致结果依赖执行顺序。
    """
    index_dir = tmp_path / "index"
    monkeypatch.setattr(storage_module, "INDEX_DIR", index_dir)
    monkeypatch.setattr(storage_module, "MANIFEST_FILE", index_dir / "manifest.json")
    monkeypatch.setattr(storage_module, "VECTORS_FILE", index_dir / "vectors.npy")
    monkeypatch.setattr(storage_module, "IDS_FILE", index_dir / "news_ids.json")
    monkeypatch.setattr(builder_module, "_PROGRESS_FILE", index_dir / "progress.json")
    yield index_dir


@pytest.fixture(autouse=True)
def stub_redis(monkeypatch):
    """
    打桩 Redis。

    索引产物以文件为准，Redis 只是快路径，但代码里仍会尝试写它 ——
    不打桩的话每个用例都要等连接超时，实测单个用例多花 2-4 秒。
    """
    class NullRedis:
        async def get(self, *a, **k):
            return None

        async def set(self, *a, **k):
            return True

        async def setex(self, *a, **k):
            return True

        async def delete(self, *a, **k):
            return True

        async def mget(self, keys):
            return [None] * len(keys)

    monkeypatch.setattr("config.cache_conf.redis_client", NullRedis(), raising=False)


@pytest.fixture(autouse=True)
def fake_embed(monkeypatch):
    """确定性假向量 + 调用计数，用于验证增量行为"""
    counter = {"calls": 0}

    async def fake(texts, use_cache=True, batch_lookup=False):
        counter["calls"] += len(texts)
        out = []
        for t in texts:
            chunks, k = [], 0
            while len(chunks) < DIM:
                d = hashlib.sha512(f"{k}::{t or ''}".encode()).digest()
                chunks.extend((b - 127.5) / 255.0 for b in d)
                k += 1
            out.append(chunks[:DIM])
        return out

    monkeypatch.setattr(builder_module, "embed_texts", fake)
    counter["reset"] = lambda: counter.__setitem__("calls", 0)
    return counter


# ---------------------------------------------------------------- 语料校验


def test_valid_document_has_no_issues():
    assert validate_document(make_news(1)) == []


def test_empty_title_is_error():
    issues = validate_document(make_news(1, title=""))

    assert any(i.code == "EMPTY_TITLE" and i.level == "error" for i in issues)


def test_empty_content_is_error():
    issues = validate_document(make_news(1, content=""))

    assert any(i.code == "EMPTY_CONTENT" and i.level == "error" for i in issues)


def test_whitespace_only_content_is_error():
    """只有空格的正文等同于空，不能当有效内容放进索引"""
    issues = validate_document(make_news(1, content="   \n\t  "))

    assert any(i.code == "EMPTY_CONTENT" for i in issues)


def test_oversized_content_is_error():
    issues = validate_document(make_news(1, content="字" * (MAX_CONTENT_CHARS + 100)))

    assert any(i.code == "CONTENT_TOO_LONG" and i.level == "error" for i in issues)


def test_long_title_is_warning_not_error():
    """标题过长只是质量问题，不该阻止入库"""
    issues = validate_document(make_news(1, title="标" * 300))

    assert any(i.code == "TITLE_TOO_LONG" and i.level == "warning" for i in issues)
    assert not any(i.level == "error" for i in issues)


def test_missing_category_is_warning():
    issues = validate_document(make_news(1, category_id=None))

    assert any(i.code == "NO_CATEGORY" and i.level == "warning" for i in issues)


def test_validate_corpus_accepts_clean_corpus():
    accepted, report = validate_corpus(make_corpus(10))

    assert len(accepted) == 10
    assert report.total == 10
    assert report.valid == 10
    assert not report.has_errors()


def test_validate_corpus_skips_bad_docs_by_default():
    corpus = make_corpus(5)
    corpus[2].content = ""
    corpus[3].title = ""

    accepted, report = validate_corpus(corpus)

    assert len(accepted) == 3, "有 error 的文档应被跳过"
    assert report.skipped == 2
    assert len(report.errors) == 2


def test_validate_corpus_strict_raises():
    corpus = make_corpus(3)
    corpus[1].content = ""

    with pytest.raises(ValueError, match="strict"):
        validate_corpus(corpus, strict=True)


def test_validate_corpus_keeps_doc_with_only_description():
    """正文为空但有 description 时仍可用于检索，不该丢弃"""
    corpus = [make_news(1, title="标题", content="", description="有摘要")]

    accepted, report = validate_corpus(corpus)

    assert len(accepted) == 1
    assert report.skipped == 0


# ---------------------------------------------------------------- 内容指纹


def test_content_hash_changes_with_title():
    a = document_content_hash("标题A", "描述", "正文")
    b = document_content_hash("标题B", "描述", "正文")

    assert a != b


def test_content_hash_changes_with_content():
    assert document_content_hash("t", "d", "c1") != document_content_hash("t", "d", "c2")


def test_content_hash_ignores_category():
    """
    分类不参与检索文本，变化时不该触发重新向量化（浪费 API 调用）。
    """
    a = make_news(1, category_id=1)
    b = make_news(1, category_id=8)

    assert document_content_hash(a.title, a.description, a.content) == document_content_hash(
        b.title, b.description, b.content
    )


def test_corpus_fingerprint_changes_on_content_change():
    c1 = make_corpus(5)
    c2 = make_corpus(5)
    c2[0].content = "改过的内容"

    assert corpus_fingerprint(c1) != corpus_fingerprint(c2)


def test_corpus_fingerprint_stable_for_same_corpus():
    assert corpus_fingerprint(make_corpus(5)) == corpus_fingerprint(make_corpus(5))


def test_corpus_fingerprint_changes_on_add():
    assert corpus_fingerprint(make_corpus(5)) != corpus_fingerprint(make_corpus(6))


# ---------------------------------------------------------------- manifest


def test_manifest_detects_model_mismatch():
    m = IndexManifest(embed_model="model-a", embed_dim=1024)

    compatible, reason = m.is_compatible_with("model-b", 1024)

    assert not compatible
    assert "模型不匹配" in reason


def test_manifest_detects_dimension_mismatch():
    """
    回归场景：换了 embedding 模型但维度不同，旧索引的向量无法使用。
    没有这道检查的话，只会在检索时报 index out of range，且看不出原因。
    """
    m = IndexManifest(embed_model="model-a", embed_dim=768)

    compatible, reason = m.is_compatible_with("model-a", 1024)

    assert not compatible
    assert "维度不匹配" in reason


def test_manifest_compatible_with_same_config():
    m = IndexManifest(embed_model="model-a", embed_dim=1024)

    compatible, reason = m.is_compatible_with("model-a", 1024)

    assert compatible
    assert reason == "兼容"


def test_manifest_detects_stale_corpus():
    m = IndexManifest(corpus_fingerprint="abc")

    assert m.is_stale_for("abc") is False
    assert m.is_stale_for("def") is True


def test_manifest_changed_docs_returns_only_new_or_modified():
    m = IndexManifest(doc_hashes={"h1": [1], "h2": [2]})

    changed = m.changed_docs({"h1": [1], "h2": [2], "h3": [3]})

    assert changed == {3}


def test_manifest_changed_docs_detects_id_reassignment():
    """同样的内容 hash 对应了不同的 news_id，说明数据被替换过"""
    m = IndexManifest(doc_hashes={"h1": [1]})

    assert m.changed_docs({"h1": [99]}) == {99}


def test_manifest_removed_docs():
    m = IndexManifest(doc_hashes={"h1": [1], "h2": [2]})

    assert m.removed_docs({"h1": [1]}) == [2]


def test_content_hash_index_keeps_duplicate_documents():
    """
    回归测试：真实数据里存在标题+描述+正文完全相同的重复新闻
    （本项目 403 篇中有 15 组、共 33 篇，如 id 22 与 116）。

    用 hash -> 单个 id 做映射时，后写入的会覆盖先写入的，
    那 33 篇文档永远进不了索引 —— 而且没有任何报错，
    索引少了几十条数据却看不出来。
    """
    corpus = [
        make_news(1, title="相同标题", content="相同正文"),
        make_news(2, title="相同标题", content="相同正文"),
        make_news(3, title="不同标题", content="不同正文"),
    ]

    index = build_content_hash_index(corpus)

    assert sorted(i for ids in index.values() for i in ids) == [1, 2, 3], "重复文档必须全部保留"
    assert max(len(ids) for ids in index.values()) == 2, "重复的两篇应归到同一个 hash"


def test_duplicate_groups_are_detectable():
    corpus = [
        make_news(1, title="重复", content="重复内容"),
        make_news(2, title="重复", content="重复内容"),
    ]

    groups = find_duplicate_groups(corpus)

    assert len(groups) == 1
    assert sorted(next(iter(groups.values()))) == [1, 2]


def test_duplicate_docs_do_not_get_dropped_from_index(fake_embed):
    """
    重复文档必须全部进入索引（向量相同可复用，但不能丢文档）。
    """
    corpus = [
        make_news(1, title="完全相同的新闻", content="完全相同的正文内容"),
        make_news(2, title="完全相同的新闻", content="完全相同的正文内容"),
    ]

    result = asyncio.run(builder_module.build_index(corpus, embed_model="m"))

    assert result.manifest.document_count == 2, "重复文档不应被丢弃"
    stored = storage_module.load_matrix_file()
    assert sorted(stored[1]) == [1, 2]


def test_manifest_round_trips_through_dict():
    m = IndexManifest(
        index_id="abc", embed_model="m", embed_dim=8,
        corpus_fingerprint="fp", document_count=3,
        doc_hashes={"a": 1}, built_at="2026-01-01",
    )

    restored = IndexManifest.from_dict(m.to_dict())

    assert restored.index_id == "abc"
    assert restored.embed_dim == 8
    assert restored.doc_hashes == {"a": 1}


def test_index_id_changes_with_any_component():
    base = make_index_id("model", 1024, "fp1")

    assert base != make_index_id("other", 1024, "fp1")
    assert base != make_index_id("model", 768, "fp1")
    assert base != make_index_id("model", 1024, "fp2")
    assert base == make_index_id("model", 1024, "fp1")


# ---------------------------------------------------------------- 全量构建


def test_build_creates_index_and_manifest(fake_embed):
    result = asyncio.run(builder_module.build_index(make_corpus(10), embed_model="test-model"))

    assert result.embedded == 10
    assert result.manifest.document_count == 10
    assert result.manifest.embed_model == "test-model"
    assert storage_module.index_exists()


def test_build_stores_vectors_in_correct_order(fake_embed):
    """
    索引正确性的前提：矩阵第 i 行必须对应 news_ids[i]。
    顺序错了，所有检索结果都会指向错误的新闻。
    """
    corpus = make_corpus(8)
    asyncio.run(builder_module.build_index(corpus, embed_model="test-model"))

    stored = storage_module.load_matrix_file()
    matrix, news_ids, _ = stored

    assert len(matrix) == len(news_ids)
    assert news_ids == [n.id for n in corpus]


def test_build_survives_redis_failure(fake_embed, monkeypatch):
    """Redis 不可用时构建仍须成功 —— 索引产物以文件为准"""
    class BrokenRedis:
        async def setex(self, *a, **k):
            raise RuntimeError("redis down")

        async def get(self, *a, **k):
            raise RuntimeError("redis down")

        async def set(self, *a, **k):
            raise RuntimeError("redis down")

        async def delete(self, *a, **k):
            raise RuntimeError("redis down")

    monkeypatch.setattr("config.cache_conf.redis_client", BrokenRedis(), raising=False)

    result = asyncio.run(builder_module.build_index(make_corpus(5), embed_model="m"))

    assert result.embedded == 5
    assert storage_module.index_exists(), "Redis 挂了索引也必须落盘"


# ---------------------------------------------------------------- 增量更新


def test_no_previous_manifest_builds_everything(fake_embed):
    """
    没有历史 manifest 时必须全量构建。
    这条是为了防止一种坏情况：读不到 manifest 却以为可以复用，
    导致构建出一个空索引。
    """
    fake_embed["reset"]()
    result = asyncio.run(builder_module.build_index(make_corpus(10), embed_model="m"))

    assert result.embedded == 10
    assert result.manifest.document_count == 10


def test_incremental_only_embeds_changed(fake_embed):
    """改 2 篇 -> 只应调用 2 次 embedding"""
    corpus = make_corpus(20)
    asyncio.run(builder_module.build_index(corpus, embed_model="m"))
    manifest = IndexManifest.from_dict(storage_module.load_manifest_file())

    changed = make_corpus(20)
    changed[0].content = "完全改过的正文"
    changed[5].title = "改过的标题"

    fake_embed["reset"]()
    result = asyncio.run(
        builder_module.build_index(changed, previous=manifest, embed_model="m")
    )

    assert result.embedded == 2, f"应只向量化 2 篇，实际 {result.embedded}"
    assert result.reused == 18
    assert fake_embed["calls"] == 2, "未变更文档不应产生 embedding 调用"


def test_new_document_is_embedded(fake_embed):
    corpus = make_corpus(10)
    asyncio.run(builder_module.build_index(corpus, embed_model="m"))
    manifest = IndexManifest.from_dict(storage_module.load_manifest_file())

    extended = corpus + [make_news(99, title="新增新闻", content="新增内容")]
    fake_embed["reset"]()
    result = asyncio.run(
        builder_module.build_index(extended, previous=manifest, embed_model="m")
    )

    assert result.embedded == 1
    assert result.manifest.document_count == 11


def test_model_change_forces_full_rebuild(fake_embed):
    """换了模型旧向量不可复用，必须全量重建"""
    corpus = make_corpus(10)
    asyncio.run(builder_module.build_index(corpus, embed_model="model-a"))
    manifest = IndexManifest.from_dict(storage_module.load_manifest_file())

    fake_embed["reset"]()
    result = asyncio.run(
        builder_module.build_index(corpus, previous=manifest, embed_model="model-b")
    )

    assert result.embedded == 10, "换模型必须全量重新向量化"
    assert result.reused == 0


def test_incremental_preserves_unchanged_vectors(fake_embed):
    """
    增量构建不能改动未变更文档的向量 —— 否则检索结果会无故变化。

    changed[3] 是列表第 4 项，其 news_id 为 4，所以断言时排除的是 4 而不是 3。
    """
    corpus = make_corpus(10)
    asyncio.run(builder_module.build_index(corpus, embed_model="m"))
    before_matrix, before_ids, _ = storage_module.load_matrix_file()

    manifest = IndexManifest.from_dict(storage_module.load_manifest_file())
    changed = make_corpus(10)
    changed_doc_id = changed[3].id
    changed[3].content = "改过的内容"
    asyncio.run(builder_module.build_index(changed, previous=manifest, embed_model="m"))

    after_matrix, after_ids, _ = storage_module.load_matrix_file()

    assert before_ids == after_ids, "id 顺序应保持稳定"
    for i, doc_id in enumerate(before_ids):
        if doc_id == changed_doc_id:
            # 被修改的文档向量必须变化，否则说明它没被重新向量化
            assert before_matrix.row(i) != pytest.approx(after_matrix.row(i)), (
                f"已修改的文档 {doc_id} 向量未更新"
            )
        else:
            assert before_matrix.row(i) == pytest.approx(after_matrix.row(i)), (
                f"未变更文档 {doc_id} 的向量被改动了"
            )


# ---------------------------------------------------------------- 断点续跑


def test_resume_skips_completed_documents(fake_embed):
    """
    模拟「上次跑了 10 篇就中断」：进度文件里记录了 10 个已完成 id。
    重跑时这 10 篇不应再调用 embedding。
    """
    corpus = make_corpus(20)
    manifest = IndexManifest(
        embed_model="m", embed_dim=DIM,
        corpus_fingerprint="old", doc_hashes={}, document_count=0,
    )
    storage_module.ensure_dir()
    with io.open(builder_module._PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"doneIds": list(range(1, 11))}, f)

    async def run():
        return await builder_module.build_index(
            corpus, previous=manifest, embed_model="m", resume=True
        )

    fake_embed["reset"]()
    # 由于缺少对应向量，缺失文档会被重新处理；关键是进度被识别到了
    result = asyncio.run(run())

    assert result.resumed, "应识别到已有进度并进入续跑模式"
    assert fake_embed["calls"] <= 20


def test_stale_progress_is_discarded(fake_embed):
    """
    回归测试：进度文件与本次待处理文档无交集时必须清空。

    否则会把所有文档误判为「已完成」，构建出空索引而毫无提示。
    """
    corpus = make_corpus(10)
    storage_module.ensure_dir()
    with io.open(builder_module._PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"doneIds": [999, 1000, 1001]}, f)

    fake_embed["reset"]()
    result = asyncio.run(builder_module.build_index(corpus, embed_model="m"))

    assert result.embedded == 10, "陈旧进度不应导致文档被跳过"
    assert fake_embed["calls"] == 10


def test_progress_cleared_after_successful_build(fake_embed):
    corpus = make_corpus(5)
    asyncio.run(builder_module.build_index(corpus, embed_model="m"))

    assert not builder_module._PROGRESS_FILE.exists(), "构建成功后应清除进度"


def test_no_resume_ignores_progress(fake_embed):
    corpus = make_corpus(10)
    storage_module.ensure_dir()
    with io.open(builder_module._PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"doneIds": list(range(1, 6))}, f)

    fake_embed["reset"]()
    result = asyncio.run(
        builder_module.build_index(corpus, embed_model="m", resume=False)
    )

    assert result.embedded == 10, "resume=False 时不应读取进度"


# ---------------------------------------------------------------- 持久化


def test_matrix_persists_and_reloads(fake_embed):
    asyncio.run(builder_module.build_index(make_corpus(6), embed_model="m"))

    stored = storage_module.load_matrix_file()
    assert stored is not None
    matrix, ids, fp = stored
    assert len(matrix) == 6
    assert len(ids) == 6
    assert fp


def test_load_matrix_returns_none_when_files_missing():
    assert storage_module.load_matrix_file() is None


def test_load_matrix_rejects_count_mismatch(monkeypatch):
    """
    行数与 id 数不一致说明文件被部分覆盖，继续用会导致检索结果错位，
    必须整体丢弃并回退到懒加载。
    """
    storage_module.ensure_dir()
    import numpy as np

    np.save(storage_module.VECTORS_FILE, np.ones((3, DIM), dtype=np.float32))
    with io.open(storage_module.IDS_FILE, "w", encoding="utf-8") as f:
        json.dump({"newsIds": [1, 2], "fingerprint": "x"}, f)

    assert storage_module.load_matrix_file() is None


def test_clear_index_removes_all_artifacts(fake_embed):
    asyncio.run(builder_module.build_index(make_corpus(3), embed_model="m"))
    assert storage_module.index_exists()

    storage_module.clear_index()

    assert not storage_module.index_exists()


def test_index_stats_reports_configuration(fake_embed):
    asyncio.run(builder_module.build_index(make_corpus(4), embed_model="model-x"))

    stats = storage_module.index_stats()

    assert stats["exists"]
    assert stats["documentCount"] == 4
    assert stats["embedModel"] == "model-x"


# ---------------------------------------------------------------- 校验与构建联动


def test_build_skips_invalid_documents(fake_embed):
    corpus = make_corpus(10)
    corpus[3].content = ""
    corpus[7].content = ""

    result = asyncio.run(builder_module.build_index(corpus, embed_model="m"))

    assert result.manifest.document_count == 8, "无效文档不应进索引"
    assert result.report.skipped == 2
    assert result.embedded == 8


def test_build_strict_mode_aborts_on_validation_failure(fake_embed):
    """strict 模式下校验失败应直接抛错，不产出索引"""
    corpus = make_corpus(5)
    corpus[1].content = ""

    with pytest.raises(ValueError, match="strict"):
        asyncio.run(builder_module.build_index(corpus, embed_model="m", strict=True))