from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func

from models.favorite import Favorite
from models.news import News


# 是否收藏
async def is_news_favorite(db: AsyncSession, user_id: int, news_id: int):
    query = select(Favorite).where(Favorite.user_id == user_id, Favorite.news_id == news_id)
    result = await db.execute(query)
    return result.scalar_one_or_none() is not None


async def add_news_favorite(db: AsyncSession, user_id: int, news_id: int):
    favorite = Favorite(user_id=user_id, news_id=news_id)
    db.add(favorite)
    await db.commit()
    await db.refresh(favorite)
    return favorite


async def remove_news_favorite(db: AsyncSession, user_id: int, news_id: int):
    stmt = delete(Favorite).where(Favorite.user_id==user_id, Favorite.news_id==news_id)
    res = await db.execute(stmt)
    await db.commit()
    return res.rowcount > 0

# 联表查询
async def get_favorite_list(db: AsyncSession, user_id: int, page: int = 1, page_size:int = 10):
    count_query = select(func.count()).where(Favorite.user_id == user_id)
    count_result = await db.execute(count_query)
    total = count_result.scalar_one()
    offset = (page-1)*page_size
    # 获取收藏列表
    query = (select(News, Favorite.created_at.label("favorite_time"), Favorite.id.label("favorite_id"))
             .join(Favorite, Favorite.news_id == News.id)
             .where(Favorite.user_id==user_id)
             .order_by(Favorite.created_at.desc())
             .offset(offset)
             .limit(page_size))
    result = await db.execute(query)
    rows = result.all()
    return rows, total

async def remove_favorite_list(db: AsyncSession, user_id: int):
    stmt = delete(Favorite).where(Favorite.user_id==user_id)
    result = await db.execute(stmt)
    return result.rowcount










