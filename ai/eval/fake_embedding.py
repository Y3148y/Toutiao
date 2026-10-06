"""
确定性假 embedding —— 只用于 CI 和离线自测。

为什么需要它
------------
真实 embedding 需要 API Key、要花钱、有网络依赖。CI 上不能有这些。
假向量让「双路融合的代码路径能否跑通、会不会抛异常」这件事在 CI 里可验证。

它现在能做什么（重要）
--------------------
旧版实现是「对文本做 SHA-512 摘要再摊成向量」—— 纯哈希，**词面毫无关联**：
两段完全相同的文字算出的相似度接近 0，两段毫不相干的文字相似度也一样。
那样的向量对 MRR、置信门控这类指标完全没有代表性，CI 门禁就是个摆设。

现在改成 **字符 n-gram 哈希（hashing trick）**：
把文本拆成单字与相邻二元组，各自哈希到固定维度的桶里累加，最后 L2 归一化。
于是「字面重合度高」的两段文本会得到更高的余弦相似度。

这样做的边界，必须说清楚：
- 它捕捉的是**词面重合**，不是语义相似。「股市下跌」和「A股走低」在这里是相近的，
  真实 embedding 认为它们高度相似，而假向量只能看到「股市/下跌」这些字的共现。
- 所以 CI 里的 MRR、拒答阈值**不能当成线上指标**，只能作为回归哨兵：
  用来发现「改了排序逻辑导致结果大幅劣化」这类回归，
  不能用来验证「检索质量达标」。

线上真实数字请用 `python -m ai.eval.runner --baseline --vector-real` 单独测，
结果记录在 ai/eval/README.md。
"""
import hashlib
import math
from typing import Iterable

DIM = 1024


def _ngrams(text: str) -> Iterable[str]:
    """
    产出单字与相邻二元组。

    中文没有空格分词，字符级的 n-gram 是不依赖词典的最省事做法。
    单字保证「部分重合」也有信号，二元组提供一点词序信息。
    """
    compact = "".join(text.split())
    for i, ch in enumerate(compact):
        yield ch
        if i + 1 < len(compact):
            yield compact[i : i + 2]


def fake_vector(text: str, dim: int = DIM) -> list[float]:
    """
    把文本映射成固定维度的确定性向量。

    用 sublinear TF（1+log tf）抑制长文档里高频字的权重 —— 不然一篇长新闻
    会被重复出现的常用字撑出一个大向量，跟什么都「有点像」。
    """
    counts: dict[int, float] = {}
    for gram in _ngrams(text):
        h = int.from_bytes(
            hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big"
        )
        idx = h % dim
        counts[idx] = counts.get(idx, 0.0) + 1.0

    vector = [0.0] * dim
    for idx, tf in counts.items():
        vector[idx] = 1.0 + math.log(tf)

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


def fake_vectors(texts: list[str], dim: int = DIM) -> list[list[float]]:
    """批量版本"""
    return [fake_vector(t, dim) for t in texts]