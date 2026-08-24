from datetime import datetime
from idlelib.history import History

from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession
from models.history import History
from models.news import News

# 2. 新增：专门用于更新浏览时间的 CRUD 函数
async def update_history_view_time(db: AsyncSession, history: History):
    history.view_time = datetime.now()  # 或者用 func.now()
    await db.commit()
    await db.refresh(history)
    return history

# 1. 查询是否存在
async def get_news_history(db: AsyncSession, user_id: int, news_id: int):
    query = select(History).where(History.user_id == user_id, History.news_id == news_id)
    result = await db.execute(query)
    return result.scalar_one_or_none()

async def add_news_history(db: AsyncSession, user_id: int, news_id: int):
    history = History(user_id=user_id, news_id=news_id)
    db.add(history)
    await db.commit()
    await db.refresh(history)
    return history


async def get_history_list(db: AsyncSession, user_id: int, page: int = 1, page_size:int = 10):
    count_query = select(func.count()).where(History.user_id == user_id)
    count_result = await db.execute(count_query)
    total = count_result.scalar_one()
    offset = (page - 1) * page_size
    # 获取收藏列表
    query = (select(News, History.view_time, History.id.label("history_id"))
             .join(History, History.news_id == News.id)
             .where(History.user_id == user_id)
             .order_by(History.view_time.desc())
             .offset(offset)
             .limit(page_size))
    result = await db.execute(query)
    rows = result.all()
    return rows, total

async def remove_news_history(db: AsyncSession, user_id: int, news_id: int):
    stmt = delete(History).where(History.user_id==user_id, History.news_id==news_id)
    res = await db.execute(stmt)
    await db.commit()
    return res.rowcount > 0


async def remove_favorite_list(db: AsyncSession, user_id: int):
    stmt = delete(History).where(History.user_id==user_id)
    result = await db.execute(stmt)
    return result.rowcount














