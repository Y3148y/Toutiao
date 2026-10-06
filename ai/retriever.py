"""
新闻语料混合检索：BM25（词面）+ 向量（语义）+ RRF 融合。

为什么要两路
------------
两路检索的失败模式互补，这是混合检索（Hybrid Search）的核心价值：

- **BM25 词面**：对专有名词、数字、缩写特别准。「GDP 增长5.2%」这种查询，
  只有词面能精确命中，向量反而会把它和「经济形势」泛泛地拉近。
- **向量语义**：能跨越表述差异。「高铁延长运行时间」和「动车组新增班次」
  零字面重叠，BM25 分数是 0，但向量能判定它们语义相近。

实测中常见的结论是：只用向���会漏掉专名查询，只用 BM25 会漏掉同义表述，
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
本项目语料只有几十条新闻，逐条算余弦相似度是几十次乘加，微秒级。
引入 FAISS/Milvus 增加运维和依赖成本却没有实际收益。
真到了百万级再换，届时接口不用改。
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
    MIN_FUSION_SCORE,
    RRF_K,
    SINGLE_PATH_WEIGHT,
    TOP_K_BM25,
    TOP_K_FINAL,
    TOP_K_VECTOR,
)
from ai.embeddings import cosine_similarity, embed_query, embed_texts
from models.news import News
from utils.logging_conf import get_logger

logger = get_logger(__name__)

_CORPUS_CACHE_KEY = "ai:corpus:news"

# 连续的中日韩统一表意文字
_CJK = r"\u4e00-\u9fff\u3400-\u4dbf"
_CJK_RUN = re.compile(f"[{_CJK}]+")
_ASCII_WORD = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """
    分词。

    刻意不引入 jieba：为了中文分词多装一个依赖不划算，而且语料是新闻这种
    规整文本。用「单字 + 相邻二元组」的中文检索常用基线就能拿到不错的效果：
        「中国高铁」 -> 中 国 高 铁 / 中国 国高 ��铁
    这样单字查询（「高铁」）和组合查询（「中国高铁」）都能命中。
    ASCII 单词和数字整体保留，因为「GDP」「5.2%」这类专名必须当成完整 token。
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
    from config.cache_conf import redis_client

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
    """
    from rank_bm25 import BM25Okapi

    if not corpus:
        return []

    docs = [tokenize(build_document(n)) for n in corpus]
    if not any(docs):
        return []

    bm25 = BM25Okapi(docs)
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

    向量化失败时返回空列表并降级为纯 BM25 —— 语义检索是增强项，
    不该因为它挂了就让整个问答不可用。
    """
    if not corpus:
        return []

    try:
        documents = [build_document(n) for n in corpus]
        all_vectors = await embed_texts(documents)
        query_vector = await embed_query(query)
    except Exception as exc:
        logger.warning("向量检索不可用，降级为纯 BM25: %s", exc)
        return []

    scored = [
        (idx, cosine_similarity(query_vector, vec)) for idx, vec in enumerate(all_vectors)
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


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


async def retrieve(
    db: AsyncSession,
    query: str,
    top_k: int = TOP_K_FINAL,
    use_cache: bool = True,
) -> tuple[list[RetrievedNews], bool]:
    """
    对新闻语料执行混合检索。

    返回 (召回列表, 是否达到置信度阈值)。
    第二个返回值供调用方判断：没达到就该直接拒答，不要调 LLM 去编。
    """
    corpus = await load_corpus(db, use_cache=use_cache)
    if not corpus:
        logger.warning("新闻语料为空，跳过检索")
        return [], False

    bm25_hits = bm25_search(corpus, query)
    vector_hits = await vector_search(corpus, query)

    if not bm25_hits and not vector_hits:
        logger.info("BM25 与向量检索均无命中: %s", query)
        return [], False

    bm25_rank = {idx: i + 1 for i, (idx, _, _) in enumerate(bm25_hits)}
    bm25_meta = {idx: (score, terms) for idx, score, terms in bm25_hits}
    vector_rank = {idx: i + 1 for i, (idx, _) in enumerate(vector_hits)}
    vector_score = {idx: score for idx, score in vector_hits}

    fused = reciprocal_rank_fusion([list(bm25_rank.keys()), list(vector_rank.keys())])

    results: list[RetrievedNews] = []
    for idx, fusion_score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
        news = corpus[idx]
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

    best_score = results[0].fusion_score if results else 0.0
    confident = best_score >= MIN_FUSION_SCORE

    logger.info(
        "检索 '%s' -> %s 条（BM25 %s / 向量 %s），最高融合分 %.4f，置信%s",
        query,
        len(results),
        len(bm25_hits),
        len(vector_hits),
        best_score,
        "足" if confident else "不足",
    )
    return results, confident


def invalidate_corpus_cache() -> None:
    """
    语料变更后手动失效缓存。

    目前新闻没有写接口，所以只在文档里说明用法；
    将来加了新闻管理接口就应该在写入成功后调用。
    """
    logger.info("如需强制刷新语料缓存，请手动删除 Redis key: %s", _CORPUS_CACHE_KEY)