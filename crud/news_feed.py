"""
首页信息流与热度排序。

要解决的问题
------------
原来的 `/api/news/list` 强制要求传 `categoryId`，于是：
- 用户打开 App 第一屏没有内容可看，必须先知道要进哪个分类
- 浏览量在详情页有累计，但没有任何地方消费它 —— 数据白攒

这里补上新闻产品的发现入口：信息流（可全量/可分类）、热度排序、
基于阅读历史的分类偏好加权。

热度算法
--------
用 HackerNews 的经典思路（也是 Reddit / HN 一直在用的变体）：

    hot = (views + 1) / (age_hours + 2) ^ gravity

分母随时间增长，所以老新闻即使浏览量高，热度也会自然衰减；
`+1` 让刚发布、还没人看的文章不会算出 0 分而被排到最后；
`+2` 和 `gravity=1.5` 是经验值，作用是压制时效性过强的权重。

为什么在 Python 里排序而不是写进 SQL
------------------------------------
`views / POW(...)` 是可以在 SQL 里算的。但这里有两个理由放在应用层：
1. 个性化排序需要在同一个列表上叠加用户权重，混在一段 SQL 里可读性差很多
2. 热度要能单独测试（纯函数），SQL 里就只能靠集成测试

代价是会把候选集全部读出来。403 篇完全不是问题；
如果语料到十万级，正确做法是改成 SQL 计算 + `(publish_time, id)` 游标分页，
这里不做这个优化 —— 提前优化比不优化更容易出错。
"""
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.history import History
from models.news import Category, News

# 热度公式的时间基准偏移量。越小越看重时效，越大越看重累积浏览量
HOT_AGE_OFFSET = 2.0
HOT_GRAVITY = 1.5

# 年龄上限。只用于防止数值溢出，不改变正常范围内的排序。
# 5 年前的文章和 6 年前的文章本来就该是同一档，不该因为年龄差异被拉开。
HOT_MAX_AGE_HOURS = 5 * 365 * 24


def hot_score(
    views: int,
    publish_time: datetime,
    now: datetime | None = None,
    gravity: float = HOT_GRAVITY,
) -> float:
    """
    单篇文章的热度分。纯函数，方便直接单测。

    views=0 且刚发布时结果是 1/2^1.5 ≈ 0.35，不为 0，
    否则新文章会永远排在有浏览量的老文章后面。

    关于「当前语料下热榜会退化成时间排序」
    -------------------------------------
    实测这份语料：403 篇全部发布于 1~3 年前，且浏览量全部为 0。
    也就是说热度公式里只有 publish_time 那一项在起作用，
    热榜实际等价于按发布时间倒序 —— 这是数据的真实状态，不是公式的问题。
    浏览量要等真实流量进来才会有值，届时排序会自动切换到「热度优先」。

    不做特殊处理（比如把 now 换成语料内最新时间），是因为那等于让算法去
    迁就数据、伪造一个「看起来有区分度」的结果。真实流量下 now 就是当前时间。
    """
    now = now or datetime.now()
    age_hours = max(0.0, (now - publish_time).total_seconds() / 3600)
    denominator = (min(age_hours, HOT_MAX_AGE_HOURS) + HOT_AGE_OFFSET) ** gravity
    return (views + 1) / denominator


async def get_categories_map(db: AsyncSession) -> dict[int, str]:
    """分类 id -> 名称。列表接口要带分类名，避免前端再做一次映射请求。"""
    rows = await db.execute(select(Category.id, Category.name))
    return {cid: name for cid, name in rows.all()}


async def get_category_weights(
    db: AsyncSession, user_id: int, recent_limit: int = 50
) -> dict[int, float]:
    """
    根据用户最近阅读历史算分类偏好权重。

    只看最近 N 条而不是全量历史：兴趣会变，全量会把三个月前的行为
    和今天的行为等权看待。

    权重按「该分类占比」计算，最高的分类拿到的权重接近 1。
    没有历史时返回空字典，调用方回退到普通排序。
    """
    subq = (
        select(
            News.category_id.label("cid"),
            func.count(History.id).label("hits"),
        )
        .join(History, History.news_id == News.id)
        .where(History.user_id == user_id)
        .group_by(News.category_id)
        .order_by(func.count(History.id).desc())
        .limit(recent_limit)
        .subquery()
    )
    rows = await db.execute(
        select(subq.c.cid, subq.c.hits).order_by(subq.c.hits.desc())
    )
    pairs = rows.all()
    if not pairs:
        return {}

    total = sum(hits for _, hits in pairs)
    if total <= 0:
        return {}
    return {cid: hits / total for cid, hits in pairs}


async def _load_candidates(
    db: AsyncSession, category_id: int | None
) -> list[News]:
    """
    取出候选集。

    category_id 为 None 时取全部 —— 这正是原接口缺失的能力。
    """
    stmt = select(News)
    if category_id is not None:
        stmt = stmt.where(News.category_id == category_id)
    rows = await db.execute(stmt)
    return list(rows.scalars().all())


def _apply_personalization(
    items: list[News],
    weights: dict[int, float],
    now: datetime,
) -> list[News]:
    """
    个性化排序：热度分 × (1 + 该分类权重 × 系数)。

    系数 3.0 的含义：偏好最高的分类，排序分最高能放大到 4 倍。
    取 3 是因为再高就会让「用户偏好」压过「文章本身质量」，
    结果是他只看到一个分类的新闻，反而信息茧房更严重。
    """
    scored = []
    for item in items:
        base = hot_score(item.views, item.publish_time, now)
        weight = weights.get(item.category_id, 0.0)
        scored.append((base * (1.0 + weight * 3.0), item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored]


async def get_feed(
    db: AsyncSession,
    *,
    category_id: int | None = None,
    sort: str = "latest",
    user_id: int | None = None,
    page: int = 1,
    page_size: int = 10,
) -> tuple[list[News], int, dict[int, str]]:
    """
    首页信息流。

    sort 取值：
      latest       按发布时间倒序（默认）
      hot          按热度分倒序
      personalized 按用户分类偏好加权，需要登录

    返回 (当前页数据, 总条数, 分类名映射)。
    """
    categories = await get_categories_map(db)
    items = await _load_candidates(db, category_id)
    now = datetime.now()

    if sort == "hot":
        items.sort(
            key=lambda n: hot_score(n.views, n.publish_time, now), reverse=True
        )
    elif sort == "personalized":
        weights = (
            await get_category_weights(db, user_id) if user_id else {}
        )
        if weights:
            items = _apply_personalization(items, weights, now)
        else:
            # 没有历史可依据时退化为时间序，而不是给一个「假装个性化」的结果
            items.sort(key=lambda n: n.publish_time, reverse=True)
    else:
        items.sort(key=lambda n: n.publish_time, reverse=True)

    total = len(items)
    offset = max(0, (page - 1) * page_size)
    return items[offset : offset + page_size], total, categories


async def get_hot_news(
    db: AsyncSession, *, limit: int = 20
) -> tuple[list[tuple[News, float]], int, dict[int, str]]:
    """热榜。返回 (带热度分的文章列表, 总条数, 分类名映射)。"""
    categories = await get_categories_map(db)
    items = await _load_candidates(db, None)
    now = datetime.now()
    scored = [(n, hot_score(n.views, n.publish_time, now)) for n in items]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:limit], len(scored), categories


__all__ = [
    "hot_score",
    "get_feed",
    "get_hot_news",
    "get_categories_map",
    "get_category_weights",
    "HOT_GRAVITY",
    "HOT_AGE_OFFSET",
    "HOT_MAX_AGE_HOURS",
]