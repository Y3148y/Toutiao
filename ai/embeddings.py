"""
文本向量化。

用 DashScope 的 text-embedding（OpenAI 兼容协议 /embeddings 接口），
把新闻文本转成稠密向量，供语义检索使用。

为什么要有这一路检索
------------------
BM25 是词面匹配，只认字面。「高铁延长运行时间」和「动车组新增班次」没有任何
字面重叠，BM25 分数是 0，但语义其实相近。向量化能把这类表述统一到相近的向量
空间，从而召回。所以它和 BM25 是互补关系，不是替代关系。

为什么要缓存
------------
新闻语料是静态的（50 条左右），但每次查询都要重新向量化语料就是浪费。
embedding 有成本（计费 + 耗时），所以按文本内容 hash 缓存到 Redis，
语料不变时只算一次。新闻改了内容 hash 就变了，自动失效。
"""
import hashlib
import json
from typing import Iterable

import httpx

from ai.config import (
    DASHSCOPE_BASE_URL,
    DASHSCOPE_EMBED_MODEL,
    EMBEDDING_CACHE_TTL,
    REQUEST_TIMEOUT,
    auth_headers,
)
from utils.logging_conf import get_logger

logger = get_logger(__name__)

_EMBEDDING_CACHE_PREFIX = "ai:embed:"
# DashScope 单次最多接收的文本条数
_BATCH_SIZE = 10


def _cache_key(text: str, model: str) -> str:
    """按内容 hash 生成缓存 key。文本或模型变了 key 就会变。"""
    digest = hashlib.sha256(f"{model}::{text}".encode("utf-8")).hexdigest()
    return f"{_EMBEDDING_CACHE_PREFIX}{digest}"


async def _get_cached_embedding(text: str) -> list[float] | None:
    """从 Redis 读缓存。读失败返回 None，不影响主流程。"""
    from config.cache_conf import redis_client

    key = _cache_key(text, DASHSCOPE_EMBED_MODEL)
    try:
        raw = await redis_client.get(key)
        return json.loads(raw) if raw else None
    except Exception as exc:
        logger.warning("读取 embedding 缓存失败: %s", exc)
        return None


async def _set_cached_embedding(text: str, vector: list[float]) -> None:
    from config.cache_conf import redis_client

    key = _cache_key(text, DASHSCOPE_EMBED_MODEL)
    try:
        await redis_client.setex(key, EMBEDDING_CACHE_TTL, json.dumps(vector))
    except Exception as exc:
        logger.warning("写入 embedding 缓存失败: %s", exc)


async def _call_embedding_api(texts: list[str]) -> list[list[float]]:
    """调用 DashScope /embeddings 接口。分批发送避免超长请求。"""
    vectors: list[list[float]] = []
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            resp = await client.post(
                f"{DASHSCOPE_BASE_URL}/embeddings",
                headers=auth_headers(),
                json={
                    "model": DASHSCOPE_EMBED_MODEL,
                    "input": batch,
                    "encoding_format": "float",
                },
            )
            resp.raise_for_status()
            payload = resp.json()
            # 返回顺序按 index 排，data 里带 index 字段，不能直接 append
            items = sorted(payload["data"], key=lambda item: item["index"])
            vectors.extend(item["embedding"] for item in items)
    return vectors


async def embed_texts(
    texts: list[str], use_cache: bool = True, batch_lookup: bool = False
) -> list[list[float]]:
    """
    批量向量化。先查缓存，只对未命中的文本调 API。

    返回顺序与入参严格一致 —— 检索时要靠下标把向量映射回新闻，
    顺序错了整个检索结果就是错的。

    batch_lookup=True 时用 MGET 一次取回所有缓存。
    逐条 GET 在 403 条语料上会产生 404 次网络往返（实测 1497ms），
    批量查能把往返降到 1 次。语料检索走这个模式。
    """
    if not texts:
        return []

    vectors: list[list[float] | None] = [None] * len(texts)

    if not use_cache:
        pending_index = list(range(len(texts)))
    elif batch_lookup:
        cached_list = await _get_cached_embeddings_batch(texts)
        pending_index = []
        for i, cached in enumerate(cached_list):
            if cached is not None:
                vectors[i] = cached
            else:
                pending_index.append(i)
    else:
        pending_index = []
        for i, text in enumerate(texts):
            cached = await _get_cached_embedding(text)
            if cached is not None:
                vectors[i] = cached
            else:
                pending_index.append(i)

    if pending_index:
        pending_texts = [texts[i] for i in pending_index]
        logger.info("需向量化 %s 条文本（模型 %s）", len(pending_texts), DASHSCOPE_EMBED_MODEL)
        fresh = await _call_embedding_api(pending_texts)
        if len(fresh) != len(pending_index):
            raise ValueError(
                f"embedding 返回数量不匹配：期望 {len(pending_index)}，实际 {len(fresh)}"
            )
        if use_cache:
            await _set_cached_embeddings_batch(pending_texts, fresh)
        for idx, vector in zip(pending_index, fresh):
            vectors[idx] = vector

    if any(v is None for v in vectors):
        raise ValueError("部分文本向量化失败")
    return vectors  # type: ignore[return-value]


async def _get_cached_embeddings_batch(texts: list[str]) -> list[list[float] | None]:
    """
    用 MGET 一次取回所有缓存。

    MGET 的返回顺序与 key 顺序一致，可以安全按下标映射回原文。
    单条读取失败时退化成逐条 GET，保证不会因为批量能力缺失就整体失效。
    """
    from config.cache_conf import redis_client

    if not texts:
        return []

    keys = [_cache_key(t, DASHSCOPE_EMBED_MODEL) for t in texts]
    try:
        raws = await redis_client.mget(keys)
    except Exception as exc:
        logger.warning("批量读取 embedding 缓存失败，退化为逐条读取: %s", exc)
        return [await _get_cached_embedding(t) for t in texts]

    out: list[list[float] | None] = []
    for raw in raws:
        if not raw:
            out.append(None)
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            logger.warning("embedding 缓存内容非法，按未命中处理")
            out.append(None)
    return out


async def _set_cached_embeddings_batch(
    texts: list[str], vectors: list[list[float]]
) -> None:
    """用 pipeline 批量写入，避免 N 次往返"""
    from config.cache_conf import redis_client

    if not texts:
        return
    try:
        async with redis_client.pipeline(transaction=False) as pipe:
            for text, vector in zip(texts, vectors):
                pipe.setex(
                    _cache_key(text, DASHSCOPE_EMBED_MODEL),
                    EMBEDDING_CACHE_TTL,
                    json.dumps(vector),
                )
            await pipe.execute()
    except Exception as exc:
        logger.warning("批量写入 embedding 缓存失败: %s", exc)


async def embed_query(text: str) -> list[float]:
    """向量化用户查询"""
    vectors = await embed_texts([text])
    return vectors[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    单对余弦相似度（纯 Python，保留给测试和非批量场景）。

    批量检索请用 vector_matrix.search()，走 numpy 矩阵乘。
    做了零向量保护：全 0 向量除零会得到 nan，而 nan 的所有比较都是 False，
    会让排序行为变得难以预测。
    """
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorMatrix:
    """
    语料向量矩阵：一次建表，之后每次查询只做一次矩阵乘。

    为什么需要它
    ------------
    逐条`cosine_similarity` 是纯 Python 的 zip 循环，403×1024 维实测 72.7ms；
    而归一化之后 M @ q 一次矩阵乘只要 0.073ms，**快约 1000 倍**。

    向量已经 L2 归一化，所以点积就等于余弦相似度，不用再算模长。
    （embedding 模型输出本身已归一化，这里显式归一化一次以防第三方返回非归一化向量）

    为什么不用向量数据库
    ------------------
    403 条语料、全量扫描 0.073ms，引FAISS/Milvus 增加部署和运维成本却没有收益。
    真正需要 ANN 索引（HNSW/IVF）时是百万级以上，这个模块的 `search()` 接口
    可以直接换掉内部实现，调用方无需改动。
    """

    def __init__(self, vectors: list[list[float]]):
        import numpy as np

        self._np = np
        if not vectors:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            self._norms = np.zeros((0,), dtype=np.float32)
            return

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"向量维度不正确: shape={matrix.shape}")

        # 显式归一化：让点积等于余弦。零向量用 1.0 兜底避免除零产生 nan
        norms = np.linalg.norm(matrix, axis=1)
        safe_norms = np.where(norms > 0, norms, 1.0)
        self._matrix = matrix / safe_norms[:, None]
        self._norms = norms

    def __len__(self) -> int:
        return int(self._matrix.shape[0])

    @property
    def dim(self) -> int:
        return int(self._matrix.shape[1]) if len(self) else 0

    def search(self, query: list[float], top_k: int) -> list[tuple[int, float]]:
        """
        检索最相似的 top_k 个，返回 [(下标, 余弦相似度)]，按相似度降序。
        """
        if not len(self) or not query:
            return []

        q = self._np.asarray(query, dtype=self._matrix.dtype)
        if q.shape[0] != self.dim:
            raise ValueError(f"查询向量维度 {q.shape[0]} 与语料维度 {self.dim} 不一致")

        q_norm = float(self._np.linalg.norm(q))
        if q_norm > 0:
            q = q / q_norm

        # 一次矩阵乘得到全部相似度，复杂度 O(N*d) 但在 BLAS 下极快
        scores = self._matrix @ q

        k = min(top_k, scores.shape[0])
        if k <= 0:
            return []
        # argpartition 只做部分排序，比全排序快
        top_idx = self._np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[self._np.argsort(-scores[top_idx])]
        return [(int(i), float(scores[i])) for i in top_idx]