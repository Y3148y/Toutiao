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


async def embed_texts(texts: list[str], use_cache: bool = True) -> list[list[float]]:
    """
    批量向量化。先查缓存，只对未命中的文本调 API。

    返回顺序与入参严格一致 —— 检索时要靠下标把向量映射回新闻，
    顺序错了整个检索结果就是错的。
    """
    if not texts:
        return []

    vectors: list[list[float] | None] = [None] * len(texts)
    pending_index: list[int] = []

    if use_cache:
        for i, text in enumerate(texts):
            cached = await _get_cached_embedding(text)
            if cached is not None:
                vectors[i] = cached
            else:
                pending_index.append(i)
    else:
        pending_index = list(range(len(texts)))

    if pending_index:
        logger.info("需向量化 %s 条文本（模型 %s）", len(pending_index), DASHSCOPE_EMBED_MODEL)
        fresh = await _call_embedding_api([texts[i] for i in pending_index])
        if len(fresh) != len(pending_index):
            raise ValueError(
                f"embedding 返回数量不匹配：期望 {len(pending_index)}，实际 {len(fresh)}"
            )
        for idx, vector, text in zip(pending_index, fresh, [texts[i] for i in pending_index]):
            vectors[idx] = vector
            if use_cache:
                await _set_cached_embedding(text, vector)

    # 全部成功才算正常返回
    if any(v is None for v in vectors):
        raise ValueError("部分文本向量化失败")
    return vectors  # type: ignore[return-value]


async def embed_query(text: str) -> list[float]:
    """向量化用户查询"""
    vectors = await embed_texts([text])
    return vectors[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    余弦相似度。

    语料规模不大（几十条），直接在 Python 里算即可，不需要引入向量数据库。
    生产上量大了再换 FAISS / Milvus / pgvector 才有意义。

    做了零向量保护：向量全 0 时除零会得到 nan，nan 比较全是 False，
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