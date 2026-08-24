from sqlalchemy.ext.asyncio import AsyncSession
from models.news import Category
from models.news import News
from sqlalchemy import select
from sqlalchemy import func
from sqlalchemy import update


async def get_categories(db: AsyncSession, skip: int = 0, limit: int = 100):
     stmt = select(Category).offset(skip).limit(limit)
     result = await db.execute(stmt)
     return result.scalars().all()

 
async def get_news_list(db: AsyncSession, category_id: int,
                        skip: int = 0,
                        limit: int = 10):
    stmt = select(News).where(News.category_id == category_id).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()

async def get_news_count(db: AsyncSession, category_id: int):
    stmt = select(func.count(News.id)).where(News.category_id == category_id)
    result = await db.execute(stmt)
    # return result.rowcount
    return result.scalar_one()


"""
    当前新闻详情 + 增加1次浏览量 + 相关新闻（同分类id的新闻）
"""
async def get_news_detail(db: AsyncSession, news_id: int):
    stmt = select(News).where(News.id == news_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()

async def increase_news_news(db: AsyncSession, news_id: int):
    stmt = update(News).where(News.id == news_id).values(views = News.views + 1)
    result = await db.execute(stmt)
    await db.commit()

    # 更新成功
    return result.rowcount > 0


async def get_related_news(db: AsyncSession, news_id: int, category_id: int, limit: int = 5):
    stmt = select(News).where(
        News.id != news_id,
        News.category_id == category_id
    ).order_by(
        News.views.desc(), # 降序
        News.publish_time.desc()
    ).limit(limit)
    result = await db.execute(stmt)
    # return result.scalars().all()
    related_news = result.scalars().all()
    return [{
            "id": news.id,
            "title": news.title,
            "content": news.content,
            "image": news.image,
            "author": news.author,
            "publishTime": news.publish_time,
            "categoryId": news.category_id,
            "views": news.views,
    } for news in related_news]