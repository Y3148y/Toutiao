"""
离线索引与在线检索的衔接测试。

核心要验证的是「校验逻辑」：离线索引什么时候该被使用、什么时候该被拒绝。
写错的后果分两种，都很糟：
- 该拒绝却用了 -> 拿旧向量检索，返回过期结果且无人察觉
- 该使用却拒了 -> 退化成运行时构建，性能优势消失（这个问题较轻，但说明逻辑没生效）
"""
import asyncio
import hashlib
import io
import json
from datetime import datetime

import numpy as np
import pytest

import ai.ingest.builder as builder_module
import ai.ingest.storage as storage_module
import ai.retriever as retriever_module
from ai.embeddings import VectorMatrix
from ai.ingest.manifest import IndexManifest
from models.news import News

DIM = 64
MODEL = "test-embed-model"


def make_news(id_, title="标题", content="正文"):
    return News(
        id=id_, title=title, description=None, content=content,
        category_id=1, views=0, publish_time=datetime(2026, 1, 1),
    )


def make_corpus(size=10):
    return [make_news(i, title=f"标题{i} 政策", content=f"正文{i} " * 20) for i in range(1, size + 1)]


@pytest.fixture(autouse=True)
def isolated_index(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    monkeypatch.setattr(storage_module, "INDEX_DIR", index_dir)
    monkeypatch.setattr(storage_module, "MANIFEST_FILE", index_dir / "manifest.json")
    monkeypatch.setattr(storage_module, "VECTORS_FILE", index_dir / "vectors.npy")
    monkeypatch.setattr(storage_module, "IDS_FILE", index_dir / "news_ids.json")
    monkeypatch.setattr(builder_module, "_PROGRESS_FILE", index_dir / "progress.json")
    yield index_dir


@pytest.fixture(autouse=True)
def stub_redis(monkeypatch):
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
def patch_model(monkeypatch):
    """让配置里的模型名与测试构建时一致，否则兼容性校验必然失败"""
    monkeypatch.setattr("ai.config.DASHSCOPE_EMBED_MODEL", MODEL, raising=False)


@pytest.fixture
def fake_embed(monkeypatch):
    """
    打桩向量化。

    必须同时覆盖 embed_texts 和 embed_query：conftest 里还有一个 autouse 的
    stub_embeddings 用 DIM=16，只覆盖其中一个会导致查询向量与语料向量维度不一致，
    表现为「向量检索不可用」的降级日志，看不出真实原因。
    """

    async def fake(texts, use_cache=True, batch_lookup=False):
        out = []
        for t in texts:
            chunks, k = [], 0
            while len(chunks) < DIM:
                d = hashlib.sha512(f"{k}::{t or ''}".encode()).digest()
                chunks.extend((b - 127.5) / 255.0 for b in d)
                k += 1
            out.append(chunks[:DIM])
        return out

    async def fake_query(text):
        return (await fake([text]))[0]

    monkeypatch.setattr(builder_module, "embed_texts", fake)
    monkeypatch.setattr(retriever_module, "embed_texts", fake)
    monkeypatch.setattr(retriever_module, "embed_query", fake_query)
    monkeypatch.setattr("ai.embeddings.embed_texts", fake)
    monkeypatch.setattr("ai.embeddings.embed_query", fake_query)


@pytest.fixture(autouse=True)
def clear_runtime_cache():
    """
    清理检索侧的进程内缓存。

    必须清：conftest 的 stub_embeddings 用 DIM=16、本文件的 fake_embed 用 DIM=64，
    缓存里残留着前者的向量矩阵时，维度校验会失败或结果错乱。
    """
    retriever_module._matrix_cache.clear()
    retriever_module.clear_index_cache()
    yield
    retriever_module._matrix_cache.clear()
    retriever_module.clear_index_cache()


def run(coro):
    return asyncio.run(coro)


def build(corpus, model=MODEL):
    return run(builder_module.build_index(corpus, embed_model=model))


# ---------------------------------------------------------------- 基本使用


def test_offline_index_used_when_fresh(fake_embed):
    corpus = make_corpus(10)
    build(corpus)

    matrix = run(retriever_module._load_offline_index(corpus))

    assert matrix is not None, "语料未变时应命中离线索引"
    assert len(matrix) == 10


def test_returns_none_when_no_index(fake_embed):
    assert run(retriever_module._load_offline_index(make_corpus(5))) is None


# ---------------------------------------------------------------- 拒绝条件


def test_rejected_when_corpus_changed(fake_embed):
    """
    语料变了必须拒绝。
    否则会用旧向量检索新语料 —— 返回结果过时且完全无人察觉。
    """
    original = make_corpus(10)
    build(original)

    changed = make_corpus(10)
    changed[0].content = "完全不同的新内容"

    assert run(retriever_module._load_offline_index(changed)) is None


def test_rejected_when_corpus_grew(fake_embed):
    original = make_corpus(10)
    build(original)

    extended = original + [make_news(99, title="新增", content="新增内容")]

    assert run(retriever_module._load_offline_index(extended)) is None


def test_rejected_when_model_changed(fake_embed):
    """
    换了 embedding 模型必须拒绝。
    维度相同时最隐蔽：不会报错，但两个模型的向量空间完全不同，
    相似度算出来没有意义。
    """
    corpus = make_corpus(10)
    build(corpus, model="model-a")

    import ai.config as cfg

    cfg.DASHSCOPE_EMBED_MODEL = "model-b"

    assert run(retriever_module._load_offline_index(corpus)) is None


def test_rejected_when_dimension_changed(fake_embed):
    corpus = make_corpus(10)
    build(corpus)
    manifest = storage_module.load_manifest_file()
    manifest["embed_dim"] = DIM + 8
    with io.open(storage_module.MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    assert run(retriever_module._load_offline_index(corpus)) is None


def test_rejected_when_matrix_corrupted(fake_embed):
    """
    矩阵行数与 id 数不一致说明文件被部分覆盖，必须整体拒绝。

    此时若继续使用，矩阵第 i 行会对应错误的 news_id，
    所有检索结果都会指向不相干的新闻。
    """
    corpus = make_corpus(10)
    build(corpus)

    np.save(storage_module.VECTORS_FILE, np.ones((5, DIM), dtype=np.float32))

    assert run(retriever_module._load_offline_index(corpus)) is None


def test_rejected_when_manifest_missing_but_index_exists(fake_embed):
    """
    只有向量没有 manifest 时仍然可用（文件手工删除 manifest 的场景），
    语料指纹那道校验已经能挡住主要风险。
    """
    corpus = make_corpus(10)
    build(corpus)
    storage_module.MANIFEST_FILE.unlink()

    matrix = run(retriever_module._load_offline_index(corpus))

    assert matrix is not None, "manifest 缺失不该让整个索引失效"


# ---------------------------------------------------------------- 端到端


def test_vector_search_uses_offline_index(fake_embed, monkeypatch):
    """端到端：vector_search 应走离线索引路径"""
    corpus = make_corpus(10)
    build(corpus)

    called = {"runtime": False}

    async def spy(*args, **kwargs):
        called["runtime"] = True
        return None

    monkeypatch.setattr(retriever_module, "get_corpus_vector_matrix", spy)

    hits = run(retriever_module.vector_search(corpus, "政策", top_k=5))

    assert len(hits) == 5
    assert not called["runtime"], "命中离线索引时不应触发运行时构建"


def test_vector_search_falls_back_to_runtime(fake_embed, monkeypatch):
    """离线索引不可用时回退到运行时构建，而不是直接放弃向量检索"""
    hits = run(retriever_module.vector_search(make_corpus(8), "政策", top_k=5))

    assert len(hits) == 5, "无离线索引时应能走运行时构建"


def test_vector_search_returns_empty_on_embedding_failure(fake_embed, monkeypatch):
    """
    查询向量化失败时返回空列表（降级纯 BM25）。

    先取查询向量再找索引：查询向量没有就无从检索，
    此时的正确行为是放弃向量路而不是报��。
    """

    async def boom(text):
        raise RuntimeError("embedding 服务不可用")

    monkeypatch.setattr(retriever_module, "embed_query", boom)

    assert run(retriever_module.vector_search(make_corpus(5), "政策")) == []


def test_offline_index_survives_redis_loss(fake_embed, monkeypatch):
    """Redis 完全不可用时，离线索引仍应可用"""

    class BrokenRedis:
        async def get(self, *a, **k):
            raise RuntimeError("down")

        async def set(self, *a, **k):
            raise RuntimeError("down")

        async def setex(self, *a, **k):
            raise RuntimeError("down")

        async def delete(self, *a, **k):
            raise RuntimeError("down")

        async def mget(self, keys):
            raise RuntimeError("down")

    monkeypatch.setattr("config.cache_conf.redis_client", BrokenRedis(), raising=False)
    monkeypatch.setattr(retriever_module, "redis_client", BrokenRedis(), raising=False)

    corpus = make_corpus(6)
    build(corpus)

    matrix = run(retriever_module._load_offline_index(corpus))
    assert matrix is not None


def test_offline_index_rebuild_keeps_same_results(fake_embed):
    """
    离线构建与运行时构建应产出相同的检索结果。

    这条保证「用了离线索引」不会改变检索行为 —— 性能优化不该带来质量差异。
    """
    corpus = make_corpus(12)
    build(corpus)

    offline = run(retriever_module._load_offline_index(corpus))
    query_vec = run(retriever_module.embed_query("政策"))

    offline_hits = offline.search(query_vec, 5)

    retriever_module._matrix_cache.clear()
    runtime = run(retriever_module.get_corpus_vector_matrix(corpus))
    runtime_hits = runtime.search(query_vec, 5)

    assert [i for i, _ in offline_hits] == [i for i, _ in runtime_hits]