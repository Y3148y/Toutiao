from fastapi.encoders import jsonable_encoder
from sqlalchemy.ext.asyncio import AsyncSession

from cache.news_cache import get_cached_categories, set_cache_categories, get_cache_news_lists, set_cache_news_list, \
    get_cache_news_detail, set_cache_news_detail, get_news_views, init_news_views, increment_news_views, \
    get_cache_related_news, set_cache_related_news
from models.news import Category
from models.news import News
from sqlalchemy import select
from sqlalchemy import func
from sqlalchemy import update
from models.base import NewsItemBase, NewsItemFull
from crud import related_news as related_crud


# 返回dict
async def get_categories(db: AsyncSession, skip: int = 0, limit: int = 100):
    # 查缓存
    cached_categories = await get_cached_categories()
    if cached_categories:
        return cached_categories # dict
    # 查库
    stmt = select(Category).offset(skip).limit(limit)
    result = await db.execute(stmt)
    categories = result.scalars().all() # ORM
    # 写入缓存
    if categories:
        categories = jsonable_encoder(categories) # 转dict
        await set_cache_categories(categories)

    # 返回
    return categories # dict

# 返回orm
async def get_news_list(db: AsyncSession, category_id: int,
                        skip: int = 0,
                        limit: int = 10):
    # 缓存
    # skip = (页码-1)*limit
    page = skip // limit + 1
    cached_news_lists = await get_cache_news_lists(category_id, page, limit)
    if cached_news_lists:
        # dict转orm
        return [News(**item) for item in cached_news_lists]

    stmt = select(News).where(News.category_id == category_id).offset(skip).limit(limit)
    result = await db.execute(stmt)
    cached_news_lists = result.scalars().all()
    if cached_news_lists:
        # cached_news_lists = jsonable_encoder(cached_news_lists)
        # ORM->pydantic->字典 (->jsonDumps字符串->缓存)
        # by_alias不使用别名，后端使用
        news_data = [NewsItemBase.model_validate(item).model_dump(mode="json", by_alias=False) for item in cached_news_lists]
        await set_cache_news_list(category_id, page, limit, news_data)
    return cached_news_lists


async def get_news_count(db: AsyncSession, category_id: int):
    stmt = select(func.count(News.id)).where(News.category_id == category_id)
    result = await db.execute(stmt)
    # return result.rowcount
    return result.scalar_one()


"""
    当前新闻详情 + 增加1次浏览量 + 相关新闻（同分类id的新闻）
"""
async def get_news_detail(db: AsyncSession, news_id: int):
    cache_news_detail = await get_cache_news_detail(news_id)
    if cache_news_detail:
        # 2. 从 Redis 获取浏览量
        views = await get_news_views(news_id)
        if views == 0:
            # 如果缓存中没有 views，从数据库补一次（可能第一次访问）
            stmt = select(News.views).where(News.id == news_id)
            views = await db.scalar(stmt) or 0
            await init_news_views(news_id, views)
        cache_news_detail['views']=views
        return News(**cache_news_detail) # dict转orm

    stmt = select(News).where(News.id == news_id)
    result = await db.execute(stmt)
    cache_news_detail = result.scalar_one_or_none()
    # 新闻不存在：交回路由层抛 404，不要往下走 model_validate
    if cache_news_detail is None:
        return None

    # ORM -> Pydantic -> dict (-> json.dumps -> Redis)
    # by_alias=False：缓存里存snake_case，回读时可直接 News(**dict)
    await set_cache_news_detail(news_id, NewsItemFull.model_validate(cache_news_detail).model_dump(mode="json", by_alias=False))
    # 初始化浏览量到 Redis
    await init_news_views(news_id, cache_news_detail.views)
    return cache_news_detail

async def increase_news_news(db: AsyncSession, news_id: int):
    # 1. Redis 原子递增
    new_views = await increment_news_views(news_id)

    stmt = update(News).where(News.id == news_id).values(views = News.views + 1)
    result = await db.execute(stmt)
    await db.commit()

    # 更新成功
    return result.rowcount > 0


async def get_related_news(db: AsyncSession, news_id: int, category_id: int, limit: int = 5):
    """
    相关推荐。转发到 crud/related_news 的实现。

    真正的实现在那边（手工关联优先、实时查询兜底）。这里保留转发是为了
    不改动 routers/news.py 的调用方式 —— 外部代码 import 的是这个路径。
    """
    return await related_crud.get_related_news(db, news_id, category_id, limit)