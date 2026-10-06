"""
检索性能与链路追踪测试。

这两个模块此前没有测试保护，导致性能回退（N+1、索引重建）只能在生产环境发现。
这里用断言把性能钉住：一旦退回到旧实现，CI 会红。
"""
import asyncio
import time
from datetime import datetime

import pytest

import ai.retriever as retriever_module
from ai.config import INDEX_CACHE_TTL, MIN_FUSION_SCORE, MIN_VECTOR_SIM
from ai.embeddings import VectorMatrix, cosine_similarity
from ai.retriever import (
    RetrievalTrace,
    _corpus_fingerprint,
    clear_index_cache,
    get_bm25_index,
    invalidate_corpus_cache,
    retrieve,
)
from models.news import News
from tests.fakes import FakeResult, FakeSession

DIM = 1024
CORPUS_SIZE = 403  # 与真实语料规模一致


def make_corpus(size: int = CORPUS_SIZE) -> list[News]:
    """构造与真实规模一致的语料"""
    return [
        News(
            id=i,
            title=f"新闻标题{i} 国务院发布政策 半导体产业",
            description=f"摘要{i}",
            content=f"正文内容{i} " * 40,
            category_id=i % 8 + 1,
            views=0,
            publish_time=datetime(2026, 1, 1),
        )
        for i in range(1, size + 1)
    ]


@pytest.fixture
def corpus():
    return make_corpus()


@pytest.fixture(autouse=True)
def clean_caches():
    clear_index_cache()
    retriever_module._matrix_cache.clear()
    yield
    clear_index_cache()
    retriever_module._matrix_cache.clear()


@pytest.fixture(autouse=True)
def stub_vectors(monkeypatch):
    """打桩向量化，避免真调 API"""

    async def fake_embed_texts(texts, use_cache=True, batch_lookup=False):
        """
        由文本派生确定性向量。

        sha256 只产出 32 字节（64 维以下），所以反复摘要填满 DIM 维 ——
        否则维度不足会让 VectorMatrix.search 抛 index out of range，
        掩盖真正要测的东西。
        """
        import hashlib

        out = []
        for t in texts:
            chunks: list[float] = []
            counter = 0
            while len(chunks) < DIM:
                digest = hashlib.sha512(f"{counter}::{t or ''}".encode()).digest()
                chunks.extend((b - 127.5) / 255.0 for b in digest)
                counter += 1
            out.append(chunks[:DIM])
        return out

    async def fake_embed_query(text):
        return (await fake_embed_texts([text]))[0]

    monkeypatch.setattr(retriever_module, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(retriever_module, "embed_query", fake_embed_query)


# ---------------------------------------------------------------- VectorMatrix


def test_vector_matrix_returns_all_documents():
    m = VectorMatrix([[1.0] * DIM] * CORPUS_SIZE)

    assert len(m) == CORPUS_SIZE
    assert m.dim == DIM


def test_vector_matrix_search_finds_identical_vector():
    query = [1.0] * DIM
    m = VectorMatrix([[0.0] * DIM, [0.0] * DIM, query])

    hits = m.search(query, top_k=3)

    assert hits[0][0] == 2, "完全相同的向量应排第一"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)


def test_vector_matrix_respects_top_k():
    m = VectorMatrix([[0.0] * DIM] * 50)

    assert len(m.search([1.0] * DIM, top_k=5)) == 5
    assert len(m.search([1.0] * DIM, top_k=100)) == 50


def test_vector_matrix_scores_descending():
    m = VectorMatrix([[1.0] * DIM, [0.0] * DIM, [0.0] * DIM])

    hits = m.search([1.0] * DIM, top_k=3)

    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_vector_matrix_handles_empty():
    m = VectorMatrix([])

    assert len(m) == 0
    assert m.search([1.0] * DIM, top_k=5) == []


def test_vector_matrix_handles_zero_vector():
    """
    回归测试：全 0 向量除零会产生 nan，
    而 nan 的所有比较都是 False，会让排序结果无法预测。
    """
    m = VectorMatrix([[0.0] * DIM, [1.0] * DIM])

    hits = m.search([1.0] * DIM, top_k=2)

    assert all(score == score for score in (s for _, s in hits)), "不应出现 nan"


def test_vector_matrix_orthogonal_scores_zero():
    """
    正交向量余弦应为 0。

    这里查的是「与查询正交的那个文档」的分数，而不是 hits[0]——
    另一个文档与查询完全相同得 1.0，排序把它放在前面才是对的。
    """
    a = [1.0] + [0.0] * (DIM - 1)
    b = [0.0, 1.0] + [0.0] * (DIM - 2)

    m = VectorMatrix([b, a])
    hits = dict(m.search(a, top_k=2))

    assert hits[0] == pytest.approx(0.0, abs=1e-6)
    assert hits[1] == pytest.approx(1.0, abs=1e-5)


def test_vector_matrix_rejects_dimension_mismatch():
    """
    维度不匹配必须明确报出「谁和谁」以及可能原因。

    之前只说「维度不一致」，而调用方在 except 里会把它降级成
    「向量检索不可用」，日志上看像网络或服务故障，实际是模型换了。
    """
    m = VectorMatrix([[1.0] * DIM])

    with pytest.raises(ValueError) as exc:
        m.search([1.0] * (DIM * 2), top_k=1)

    message = str(exc.value)
    assert str(DIM * 2) in message, "应报出查询向量的实际维度"
    assert str(DIM) in message, "应报出索引的实际维度"
    assert "模型" in message, "应提示可能更换了 embedding 模型"


def test_vector_matrix_normalizes_non_unit_input():
    """传入未归一化向量也应得到正确余弦"""
    a = [3.0, 4.0] * (DIM // 2)
    m = VectorMatrix([a])

    hits = m.search(a, top_k=1)

    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------- BM25 索引缓存


def test_bm25_index_is_cached(corpus):
    first = get_bm25_index(corpus)
    second = get_bm25_index(corpus)

    assert first[0] is second[0], "同一语料应复用同一索引对象"


def test_bm25_index_rebuilds_when_corpus_changes(corpus):
    first = get_bm25_index(corpus)[0]

    changed = list(corpus)
    changed[0] = News(
        id=changed[0].id, title="完全不同的标题", description=None,
        content="内容", category_id=1, views=0, publish_time=datetime(2026, 1, 1),
    )
    second = get_bm25_index(changed)[0]

    assert first is not second, "语料变化后应重建索引"


def test_bm25_cache_does_not_grow_unbounded(corpus):
    """语料频繁变动时缓存不能无限增长，否则内存泄漏"""
    for i in range(8):
        changed = make_corpus()
        changed[0] = News(
            id=changed[0].id, title=f"标题版本{i}", description=None,
            content="内容", category_id=1, views=0, publish_time=datetime(2026, 1, 1),
        )
        get_bm25_index(changed)

    assert len(retriever_module._bm25_cache) <= retriever_module._INDEX_CACHE_MAX


def test_bm25_second_lookup_is_faster(corpus):
    """第二次查索引应显著快于第一次 —— 缓存生效的直接证据"""
    first_start = time.perf_counter()
    get_bm25_index(corpus)
    first_ms = (time.perf_counter() - first_start) * 1000

    second_start = time.perf_counter()
    get_bm25_index(corpus)
    second_ms = (time.perf_counter() - second_start) * 1000

    assert second_ms < first_ms, f"缓存未生效: 首查 {first_ms:.2f}ms 次查 {second_ms:.2f}ms"


def test_bm25_empty_corpus_returns_none():
    bm25, docs = get_bm25_index([])

    assert bm25 is None


# ---------------------------------------------------------------- 语料指纹


def test_fingerprint_changes_with_size():
    assert _corpus_fingerprint(make_corpus(10)) != _corpus_fingerprint(make_corpus(11))


def test_fingerprint_changes_with_title():
    c = make_corpus()
    before = _corpus_fingerprint(c)
    c[0].title = "改过的标题"
    assert _corpus_fingerprint(c) != before


def test_fingerprint_stable_for_same_corpus():
    assert _corpus_fingerprint(make_corpus()) == _corpus_fingerprint(make_corpus())


def test_fingerprint_handles_empty():
    """
    空语料的指纹应稳定（同样输入得到同样输出），且与非空语料不同。

    具体格式由 corpus_fingerprint 决定，这里不硬编码字符串 ——
    格式变了这个测试会假失败，而它真正要守的是「稳定性」和「可区分」。
    """
    from ai.ingest.validate import corpus_fingerprint

    empty_fp = _corpus_fingerprint([])
    assert empty_fp == corpus_fingerprint([])
    assert empty_fp != _corpus_fingerprint(make_corpus(5))


# ---------------------------------------------------------------- 向量缓存


def test_corpus_vector_matrix_is_cached(corpus):
    async def run():
        first = await retriever_module.get_corpus_vector_matrix(corpus)
        second = await retriever_module.get_corpus_vector_matrix(corpus)
        return first, second

    first, second = asyncio.run(run())

    assert first is second


def test_invalidate_corpus_index_clears_memory_cache(corpus):
    async def run():
        await retriever_module.get_corpus_vector_matrix(corpus)
        assert retriever_module._matrix_cache, "构造后应有进程内缓存"
        await retriever_module.invalidate_corpus_index()

    asyncio.run(run())

    assert not retriever_module._matrix_cache, "失效后进程内缓存应清空"


# ---------------------------------------------------------------- 性能断言


def test_bm25_search_is_fast_enough(corpus):
    """
    性能回归哨兵。

    优化前每次查询都要重建索引（403 条语料实测 74.3ms）。
    这里给 25ms 上限，超出说明缓存失效了。
    """
    get_bm25_index(corpus)  # 预热

    start = time.perf_counter()
    for _ in range(5):
        retriever_module.bm25_search(corpus, "高铁新增班次")
    avg_ms = (time.perf_counter() - start) * 1000 / 5

    assert avg_ms < 25, f"BM25 检索退化到 {avg_ms:.1f}ms/次（上限 25ms）"


def test_vector_matrix_search_is_fast(corpus):
    """全量扫描 403 条应在 5ms 内完成"""
    m = VectorMatrix([[1.0] * DIM] * CORPUS_SIZE)
    query = [0.5] * DIM
    m.search(query, top_k=20)  # 预热

    start = time.perf_counter()
    for _ in range(50):
        m.search(query, top_k=20)
    avg_ms = (time.perf_counter() - start) * 1000 / 50

    assert avg_ms < 5, f"向量检索退化到 {avg_ms:.2f}ms/次（上限 5ms）"


def test_full_retrieval_reuses_caches(corpus):
    """连续两次完整检索，第二次应远快于第一次"""
    session = FakeSession(result=FakeResult(rows=list(corpus)))

    async def run():
        t0 = time.perf_counter()
        await retrieve(session, "GDP", top_k=5, use_cache=False)
        first = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        await retrieve(session, "GDP", top_k=5, use_cache=False)
        second = (time.perf_counter() - t0) * 1000
        return first, second

    first, second = asyncio.run(run())

    assert second < first, f"缓存未生效: 首查 {first:.1f}ms 次查 {second:.1f}ms"


# ---------------------------------------------------------------- trace


def test_trace_records_all_stages(corpus):
    """trace 必须覆盖每个阶段，否则排查问题时缺关键信息"""
    session = FakeSession(result=FakeResult(rows=list(corpus)))
    trace = RetrievalTrace("测试查询")

    # 查「半导体」—— 合成语料的标题里含该词，保证 BM25 一定有命中
    results, _ = asyncio.run(
        retrieve(session, "半导体产业政策", top_k=5, use_cache=False, trace=trace)
    )

    assert trace.corpus_size == CORPUS_SIZE
    assert results, "查询词在合成语料标题中，BM25 应有命中"
    assert trace.bm25_hits > 0
    assert trace.bm25_ms > 0
    assert trace.fuse_ms >= 0
    assert trace.fingerprint

    # 阈值跟着实际使用的判据走：有向量余弦就用 MIN_VECTOR_SIM，
    # 向量路不可用才退回 MIN_FUSION_SCORE。gateMode 记录了用的是哪一种，
    # 排查「为什么这次拒答了」时必须能看出判据本身有没有被降级。
    assert trace.gate_mode in {"vector_cosine", "rrf"}
    if trace.gate_mode == "vector_cosine":
        assert trace.threshold == MIN_VECTOR_SIM
    else:
        assert trace.threshold == MIN_FUSION_SCORE


def test_trace_is_json_serializable(corpus):
    """trace 要能写日志和落Redis，必须可 json 序列化"""
    import json

    session = FakeSession(result=FakeResult(rows=list(corpus)))
    trace = RetrievalTrace("测试查询")

    asyncio.run(retrieve(session, "GDP", top_k=5, use_cache=False, trace=trace))

    payload = json.dumps(trace.to_dict(), ensure_ascii=False)
    assert json.loads(payload)["query"] == "测试查询"


def test_trace_marks_degraded_mode(corpus):
    """
    回归测试：向量路不可用时 trace 必须标记降级。

    线上曾出现过「以为是混合检索，实际一路都没生效」而无法察觉的情况，
    degradedToBm25 这个标记就是为此存在。
    """
    import ai.retriever as rm

    async def failing_embed(texts, use_cache=True, batch_lookup=False):
        raise RuntimeError("向量服务不可用")

    original = rm.embed_texts
    rm.embed_texts = failing_embed
    try:
        session = FakeSession(result=FakeResult(rows=list(corpus)))
        trace = RetrievalTrace("测试")
        asyncio.run(retrieve(session, "高铁", top_k=5, use_cache=False, trace=trace))
    finally:
        rm.embed_texts = original

    assert trace.vector_hits == 0
    assert trace.vector_available is False
    assert trace.to_dict()["recall"]["degradedToBm25"] is True


def test_trace_flags_single_path_results(corpus):
    """单路命中被降权的文档应被标记，便于定位「为什么分数这么低」"""
    session = FakeSession(result=FakeResult(rows=list(corpus)))
    trace = RetrievalTrace("GDP")

    asyncio.run(retrieve(session, "GDP", top_k=5, use_cache=False, trace=trace))

    # 只开了打桩向量化，向量路的文档通常是 hash 向量，可能与 BM25 重合
    for item in trace.results:
        assert "singlePath" in item
        assert isinstance(item["singlePath"], bool)


def test_trace_log_line_is_readable():
    t = RetrievalTrace("高铁")
    t.corpus_size = 403
    t.bm25_hits = 12
    t.vector_hits = 8
    t.top_score = 0.0328
    t.threshold = 0.02
    t.confident = True

    line = t.log_line()

    assert "高铁" in line and "403" in line and "作答" in line


def test_trace_refusal_reason_recorded(corpus):
    """拒答时 trace 要记录判定依据，否则无法排查「为什么不给答」"""
    session = FakeSession(result=FakeResult(rows=list(corpus)))
    trace = RetrievalTrace("量子计算机求解蛋白质折叠")

    _, confident = asyncio.run(
        retrieve(session, "量子计算机求解蛋白质折叠", top_k=5, use_cache=False, trace=trace)
    )

    decision = trace.to_dict()["decision"]
    assert decision["willRefuse"] == (not confident)
    assert decision["topScore"] >= 0