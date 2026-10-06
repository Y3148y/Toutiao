"""
评估用的确定性假 embedding 服务。

用途
----
线上向量检索依赖 DashScope（要API Key、要联网、有计费），
这让「向量路对召回到底有多大帮助」无法在本地和 CI 里量化。
本模块用确定性哈希向量替代，把向量路的真实行为接进评估闭环：

- 评估可离线跑（CI 无需 Secret）
- 结果完全可复现（同样输入必然同样向量）
- 但它**不模拟语义相似性** —— 哈希向量之间没有语义关系，
  只能验证「双路融合机制是否正常工作、分数量级是否符合预期」，
  不能用来论证「向量检索提升了召回率」。后者必须用真实 embedding 测。

局限在 README 里明确标注，不拿它冒充真实结论。
"""
import hashlib
from typing import Iterable


DIM = 1024  # 与 text-embedding-v4 维度一致，避免因维度不同影响相似度量级


def fake_vector(text: str, dim: int = DIM) -> list[float]:
    """
    由文本内容派生的确定性单位向量。

    用 SHA-512 反复摘要填满 dim 维，比单次摘要拼接更均匀。
    归一化到单位长度，这样余弦相似度直接等于点积。
    """
    if not text:
        return [0.0] * dim

    needed = dim
    chunks: list[float] = []
    counter = 0
    while len(chunks) < needed:
        digest = hashlib.sha512(f"{counter}::{text}".encode("utf-8")).digest()
        # 每个字节映射到 [-0.5, 0.5]
        chunks.extend((b - 127.5) / 255.0 for b in digest)
        counter += 1

    vector = chunks[:dim]
    norm = sum(x * x for x in vector) ** 0.5
    if norm == 0:
        return [0.0] * dim
    return [x / norm for x in vector]


def fake_vectors(texts: Iterable[str], dim: int = DIM) -> list[list[float]]:
    return [fake_vector(t, dim) for t in texts]
