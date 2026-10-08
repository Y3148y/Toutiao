"""
相关推荐的读写。

读取策略：手动关联优先，实时查询兜底
------------------------------------
`related_news` 表之前建好但一直没接代码，详情页走的是「同分类 + 浏览量」
的实时查询 —— 排序逻辑死板，运营无法控制推荐什么。

现在改成两级：
  1. 先读 related_news 表（运营手工配的白名单，优先级最高）
  2. 表里没有才回落到实时查询

**兜底必须有。** 这张表现在是空的，纯读表会让推荐位空掉，
比推荐得不好更糟。所以是有表用表、没表用算法。

缓存
----
沿用原有的 related cache（key 只按 news_id）。手工增删关联时必须主动
失效对应 key，否则运营改了配置、页面半小时内还是旧的。
"""
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from cache.news_cache import (
    delete_cache_related_news,
    get_cache_related_news,
    set_cache_related_news,
)
from models.news import News
from models.related_news import RelatedNews
from utils.logging_conf import get_logger

logger = get_logger(__name__)


def _to_citation_dict(row: News) -> dict:
    """News ORM -> 详情页用的推荐条目"""
    return {
        "id": row.id,
        "title": row.title,
        "content": row.content,
        "image": row.image,
        "author": row.author,
        "publishTime": row.publish_time,
        "categoryId": row.category_id,
        "views": row.views,
    }


def _extract_ids(result) -> list[int]:
    """从单列查询结果里取出 id 列表。

    真实驱动返回 Row，取 row[0] 即可。
    """
    return [row[0] for row in result.all()]


async def _fetch_manual(db: AsyncSession, news_id: int, limit: int) -> list[dict]:
    """
    读 related_news 表里的手工关联，附带新闻本体。

    分两次查询而不是 `select(RelatedNews, News).join(...)`：
    两步写法返回的都是单实体行，行为与项目里其他 crud 一致，
    「关联不存在 → 返回空 → 回落实时查询」这条路径也更容易推理。

    任何异常都吞掉并返回空列表 —— 手工关联是增强手段，
    读不出来不该让详情页整个挂掉，回落到实时查询即可。
    """
    try:
        ids_result = await db.execute(
            select(RelatedNews.related_news_id)
            .where(RelatedNews.news_id == news_id)
            # created_at 升序 = 运营录入的先后顺序，保留运营的编排意图
            .order_by(RelatedNews.created_at.asc(), RelatedNews.id.asc())
            .limit(limit)
        )
        related_ids = _extract_ids(ids_result)
        if not related_ids:
            return []

        news_result = await db.execute(select(News).where(News.id.in_(related_ids)))
        # IN 查询不保证顺序，这里按关联表里的顺序还原，否则运营编排的先后会丢失
        by_id = {n.id: n for n in news_result.scalars().all()}
        return [_to_citation_dict(by_id[rid]) for rid in related_ids if rid in by_id]
    except Exception as exc:
        logger.warning("读取手工关联失败，回落到实时查询: %s", exc)
        return []


async def _fetch_fallback(
    db: AsyncSession, news_id: int, category_id: int, limit: int
) -> list[dict]:
    """同分类热门。手工没配时的兜底"""
    result = await db.execute(
        select(News)
        .where(News.id != news_id, News.category_id == category_id)
        .order_by(News.views.desc(), News.publish_time.desc())
        .limit(limit)
    )
    return [_to_citation_dict(n) for n in result.scalars().all()]


async def get_related_news(
    db: AsyncSession, news_id: int, category_id: int, limit: int = 5
) -> list[dict]:
    """详情页的相关推荐：手工关联优先，无配置时回落实时查询"""
    cached = await get_cache_related_news(news_id)
    if cached:
        return cached

    items = await _fetch_manual(db, news_id, limit)
    if not items:
        # 没配手工关联（或关联查询出问题）都回落到实时查询。
        # 推荐位空着比推荐得不好更糟。
        items = await _fetch_fallback(db, news_id, category_id, limit)

    # jsonable_encoder 处理 datetime 字段
    from fastapi.encoders import jsonable_encoder

    await set_cache_related_news(news_id, jsonable_encoder(items))
    return items


async def add_related(
    db: AsyncSession, news_id: int | None, related_news_id: int
) -> tuple[bool, int]:
    """
    新增关联，返回 (是否真的新增, 实际生效的 news_id)。

    news_id 为 None 时反查：库里如果已经有 B -> A 这条，就把它解释成
    同一个意图（A 关联到 B），而不是再插一条重复的记录。
    这样前端只需要传一个方向，不必关心谁先谁后。

    重复添加当幂等处理：运营后台反复点同一个不会攒出一堆重复行，
    接口也可以安全重试。
    """
    if news_id is not None and news_id == related_news_id:
        # 自关联没有意义，直接拒绝而不是静默写一条
        raise ValueError("不能把新闻关联到自身")

    # 校验两端都存在，否则会写出指向不存在新闻的脏关联
    for target in (related_news_id,) if news_id is None else (news_id, related_news_id):
        found = await db.execute(
            select(func.count()).select_from(News).where(News.id == target)
        )
        if (found.scalar_one() or 0) == 0:
            raise ValueError(f"新闻不存在: {target}")

    if news_id is None:
        reversed_rows = await db.execute(
            select(RelatedNews).where(
                RelatedNews.news_id == related_news_id,
                RelatedNews.related_news_id != related_news_id,
            )
        )
        for existing in reversed_rows.scalars().all():
            # 已有反向记录，把当前这条理解成同一意图
            return False, existing.related_news_id

        target_id = related_news_id
    else:
        exists = await db.execute(
            select(func.count())
            .select_from(RelatedNews)
            .where(
                RelatedNews.news_id == news_id,
                RelatedNews.related_news_id == related_news_id,
            )
        )
        if (exists.scalar_one() or 0) > 0:
            return False, news_id
        target_id = news_id

    db.add(RelatedNews(news_id=target_id, related_news_id=related_news_id))
    await db.commit()
    await invalidate_related_cache(target_id)
    return True, target_id


async def remove_related(
    db: AsyncSession, news_id: int, related_news_id: int
) -> bool:
    """删除关联，返回是否真的删掉了"""
    result = await db.execute(
        delete(RelatedNews).where(
            RelatedNews.news_id == news_id,
            RelatedNews.related_news_id == related_news_id,
        )
    )
    await db.commit()
    if (result.rowcount or 0) > 0:
        await invalidate_related_cache(news_id)
        return True
    return False


async def list_related_ids(db: AsyncSession, news_id: int) -> list[int]:
    """列出手工关联的新闻 id，供运营后台编辑页回显"""
    result = await db.execute(
        select(RelatedNews.related_news_id)
        .where(RelatedNews.news_id == news_id)
        .order_by(RelatedNews.created_at.asc(), RelatedNews.id.asc())
    )
    return _extract_ids(result)


async def invalidate_related_cache(news_id: int) -> None:
    """删掉该新闻的推荐缓存。

    关联改了却不清缓存，运营在后台看到的和用户看到的会不一致，
    而且要等 TTL 到期才恢复 —— 这种 bug 极难定位。
    """
    try:
        await delete_cache_related_news(news_id)
    except Exception as exc:
        # 缓存失效失败不该让增删关联失败：数据已经写进库了
        logger.warning("推荐缓存失效失败（数据已保存）: %s", exc)
