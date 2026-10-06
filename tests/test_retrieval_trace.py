"""
检索链路 trace 的写入测试。

重点覆盖一个已修复的真实缺陷：redis-py 的 setex 返回协程，
忘记 await 时 trace 静默丢失——日志正常、Redis 里什么都没有，
排查时完全看不出问题。
"""
import asyncio
import json
from datetime import datetime

import pytest

import ai.retriever as retriever_module
from ai import config as ai_config
from ai.retriever import RetrievalTrace, retrieve
from models.news import News
from tests.fakes import FakeResult, FakeSession


class RecordingRedis:
    """记录所有写入，用于断言 setex 确实被 await 并执行了"""

    def __init__(self):
        self.writes: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.deleted: list[str] = []

    async def setex(self, key, ttl, value):
        self.writes[key] = value
        self.ttls[key] = ttl
        return True

    async def get(self, key):
        return None

    async def mget(self, keys):
        return [None] * len(keys)

    async def delete(self, key):
        self.deleted.append(key)
        return True


def make_corpus(size: int = 20) -> list[News]:
    return [
        News(
            id=i,
            title=f"新闻标题{i} 国务院政策 半导体产业",
            description=f"摘要{i}",
            content=f"正文{i} " * 30,
            category_id=1,
            views=0,
            publish_time=datetime(2026, 1, 1),
        )
        for i in range(1, size + 1)
    ]


@pytest.fixture(autouse=True)
def enable_trace(monkeypatch):
    """
    默认开启 trace 并注入记录型 Redis。

    必须同时改 ai_config 和 retriever_module 上的名字：
    retriever 在 import 时就把 TRACE_ENABLED 拷进了自己的模块作用域，
    只改配置对象不会影响已经导入的引用。
    """
    monkeypatch.setattr(ai_config, "TRACE_ENABLED", True, raising=False)
    monkeypatch.setattr(retriever_module, "TRACE_ENABLED", True, raising=False)
    redis = RecordingRedis()
    monkeypatch.setattr(retriever_module, "redis_client", redis, raising=False)
    return redis


@pytest.fixture(autouse=True)
def stub_embeddings(monkeypatch):
    import hashlib

    dim = 64

    async def fake_embed_texts(texts, use_cache=True, batch_lookup=False):
        out = []
        for t in texts:
            chunks, k = [], 0
            while len(chunks) < dim:
                d = hashlib.sha512(f"{k}::{t or ''}".encode()).digest()
                chunks.extend((b - 127.5) / 255.0 for b in d)
                k += 1
            out.append(chunks[:dim])
        return out

    async def fake_embed_query(text):
        return (await fake_embed_texts([text]))[0]

    monkeypatch.setattr(retriever_module, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(retriever_module, "embed_query", fake_embed_query)


def run_retrieve(question: str, **kwargs):
    session = FakeSession(result=FakeResult(rows=make_corpus()))
    trace = RetrievalTrace(question)
    results, confident = asyncio.run(
        retrieve(session, question, top_k=5, use_cache=False, trace=trace, **kwargs)
    )
    return results, confident, trace


def test_trace_is_written_to_redis(enable_trace):
    """
    回归测试：曾因 redis_client.setex() 没有 await，
    trace 静默丢失 —— 日志正常但 Redis 里什么都没有。
    """
    run_retrieve("半导体产业政策")

    # 语料向量缓存也会走同一个 Redis，所以只筛 trace 的 key
    trace_keys = [k for k in enable_trace.writes if k.startswith("ai:trace:")]
    assert trace_keys, f"trace 未写入 Redis，实际写入: {list(enable_trace.writes)}"


def test_trace_uses_request_id_in_key(enable_trace, monkeypatch):
    """key 必须含 request_id，这样用户报障时给的 ID 能直接查到"""
    monkeypatch.setattr(
        "utils.logging_conf.get_request_id", lambda: "test-req-123", raising=False
    )

    run_retrieve("半导体产业政策")

    assert any("test-req-123" in k for k in enable_trace.writes), (
        f"key 未包含 request_id: {list(enable_trace.writes)}"
    )


def test_trace_payload_is_valid_json(enable_trace):
    run_retrieve("半导体产业政策")

    payload = json.loads(list(enable_trace.writes.values())[0])
    assert payload["query"] == "半导体产业政策"
    assert "timing" in payload and "recall" in payload and "decision" in payload


def test_trace_respects_ttl(enable_trace):
    run_retrieve("半导体产业政策")

    ttl = [
        v for k, v in enable_trace.ttls.items() if k.startswith("ai:trace:")
    ]
    assert ttl == [ai_config.TRACE_TTL]


def test_trace_not_written_when_disabled(monkeypatch, enable_trace):
    """开关关闭时不应写入，避免无谓的 Redis 开销"""
    monkeypatch.setattr(ai_config, "TRACE_ENABLED", False, raising=False)
    monkeypatch.setattr(retriever_module, "TRACE_ENABLED", False, raising=False)

    run_retrieve("半导体产业政策")

    assert not enable_trace.writes


def test_trace_write_failure_does_not_break_retrieval(monkeypatch):
    """
    trace 写失败不能影响主流程 —— trace 是可观测性手段，
    不是业务逻辑。
    """

    class BrokenRedis:
        async def setex(self, *args, **kwargs):
            raise RuntimeError("Redis 挂了")

    monkeypatch.setattr(retriever_module, "redis_client", BrokenRedis(), raising=False)

    results, confident, trace = run_retrieve("半导体产业政策")

    assert results, "trace 写失败时检索仍应返回结果"
    assert trace.confident == confident


def test_trace_records_degraded_mode(enable_trace):
    """
    向量路不可用时 trace 必须标记降级。

    打桩必须替换 get_corpus_vector_matrix 而不是 embed_texts：
    向量矩阵有进程内缓存，前一个用例已经把它建好，
    再让 embed_texts 抛异常也不会被调用，降级路径根本走不到。
    """
    async def failing_matrix(corpus, force_rebuild=False):
        raise RuntimeError("向量服务不可用")

    original = retriever_module.get_corpus_vector_matrix
    retriever_module.get_corpus_vector_matrix = failing_matrix
    try:
        retriever_module._matrix_cache.clear()
        _, _, trace = run_retrieve("半导体产业政策")
    finally:
        retriever_module.get_corpus_vector_matrix = original
        retriever_module._matrix_cache.clear()

    assert trace.vector_available is False, "向量路挂了必须标记降级"
    assert trace.to_dict()["recall"]["degradedToBm25"] is True
    assert trace.vector_hits == 0


def test_trace_records_confident_decision(enable_trace):
    _, confident, trace = run_retrieve("半导体产业政策")

    decision = trace.to_dict()["decision"]
    assert decision["confident"] == confident
    assert decision["willRefuse"] == (not confident)


def test_trace_per_result_carries_scores(enable_trace):
    _, _, trace = run_retrieve("半导体产业政策")

    assert trace.results
    for item in trace.results:
        assert "newsId" in item
        assert "fusionScore" in item
        assert "bm25Rank" in item or item["bm25Rank"] is None
        assert "singlePath" in item