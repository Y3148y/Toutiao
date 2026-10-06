from fastapi import APIRouter, Query, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from config.db_conf import get_db
from crud import news
from crud import news_cache
from utils.response import success_response

router = APIRouter(prefix="/api/news", tags=["news"])

@router.get("/categories")
async def get_categories(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    categories = await news_cache.get_categories(db, skip, limit)
    return success_response(data=categories)

@router.get("/list")
async def get_news_list(category_id: int = Query(..., alias="categoryId"),
                        page: int = 1,
                        page_size: int = Query(10, alias="pageSize", le=100),
                        db: AsyncSession=Depends(get_db)):
    # 查询新闻列表->计算总量->计算是否还有更多
    offset = (page - 1) * page_size
    news_list = await news_cache.get_news_list(db, category_id, offset, page_size)
    total = await news.get_news_count(db, category_id)
    # (跳过的 + 当前列表的数量) < 总数 -> 还有更多
    has_more = (offset + len(news_list)) < total
    return success_response(data={
        "list": news_list,
        "total": total,
        "hasMore": has_more
    })

@router.get("/detail")
async def read_news_detail(news_id: int=Query(..., alias="id"),
                           db: AsyncSession=Depends(get_db)):
    news_detail = await news_cache.get_news_detail(db, news_id)
    if not news_detail:
        raise HTTPException(status_code=404, detail="新闻不存在")

    news_res = await news_cache.increase_news_news(db, news_detail.id)
    if not news_res:
        raise HTTPException(status_code=404, detail="新闻不存在")

    # 浏览量已在 Redis 原子自增，返回自增后的值，保证与 Redis 计数一致
    views = await news_cache.get_news_views(news_detail.id)
    related_news = await news_cache.get_related_news(db, news_id, news_detail.category_id)

    return success_response(data={
        "id": news_detail.id,
        "title": news_detail.title,
        "content": news_detail.content,
        "image": news_detail.image,
        "author": news_detail.author,
        "publishTime": news_detail.publish_time,
        "categoryId": news_detail.category_id,
        "views": views,
        "relatedNews": related_news
    })
