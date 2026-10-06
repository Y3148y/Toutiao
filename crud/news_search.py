"""
新闻搜索。

为什么不自己写 LIKE 查询
------------------------
`WHERE title LIKE '%关键词%'` 有三个问题：
1. 前置通配符用不上索引，数据量一大就是全表扫描
2. 中文分词要自己做，"人工智能"搜不到"AI 芯片"这类同义表达
3. 无法排序相关性，只能按时间或 id 排

项目里已经有一套经过评估验证的 BM25 实现（`ai/retriever.py`，
Recall@5 = 93.9%），直接复用而不是重写一套更差的。

实现取舍
--------
全量取到内存再检索：403 篇规模下 BM25 索引本身就在进程内缓存，
实测检索 < 10ms。要换成 MySQL 全文索引或 ES，接口签名不用改，
只替换 `search_news` 内部的实现。
"""
import time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.retriever import bm25_search
from models.news import News

# 一次搜索最多返回多少条候选。
# 用户翻页不太可能翻过 200 条，超出部分不返回，避免把全部结果塞给前端。
SEARCH_MAX_CANDIDATES = 200


async def search_news(
    db: AsyncSession,
    keyword: str,
    *,
    category_id: int | None = None,
    page: int = 1,
    page_size: int = 10,
) -> tuple[list[tuple[News, float, list[str]]], int, float]:
    """
    关键词搜索。

    返回 (当前页结果, 总命中数, 检索耗时毫秒)。
    结果项是 (新闻, 相关性得分, 命中词列表)。

    keyword 为空白时直接返回空，不去跑一次全量检索 ——
    前端搜索框清空的瞬间会发请求，不挡掉就是白烧 CPU。
    """
    started = time.perf_counter()

    if not keyword or not keyword.strip():
        return [], 0, (time.perf_counter() - started) * 1000

    keyword = keyword.strip()

    stmt = select(News)
    if category_id is not None:
        # 分类过滤在 BM25 之前做掉：先缩小候选集再排序，
        # 比先对全库排序再过滤要快，而且语义上更对
        # （"这个分类里最相关的" 而不是 "全库最相关的里属于这个分类的"）
        stmt = stmt.where(News.category_id == category_id)
    rows = await db.execute(stmt)
    corpus = list(rows.scalars().all())

    if not corpus:
        return [], 0, (time.perf_counter() - started) * 1000

    # top_k 传 SEARCH_MAX_CANDIDATES 而不是分页偏移量，
    # 因为 BM25 必须先算完全部相关性才能知道正确的顺序，
    # 只取一页会让翻页结果错乱。
    hits = bm25_search(corpus, keyword, top_k=SEARCH_MAX_CANDIDATES)

    total = len(hits)
    offset = max(0, (page - 1) * page_size)
    window = hits[offset : offset + page_size]

    results = [
        (corpus[idx], score, terms) for idx, score, terms in window
    ]
    return results, total, (time.perf_counter() - started) * 1000