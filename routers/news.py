from fastapi import APIRouter, Query, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from config.db_conf import get_db
from crud import news
from crud import news_cache
from crud import news_feed
from crud import news_search
from models.users import User
from schemas.news import (
    HotNewsItem,
    HotNewsResponse,
    NewsListItem,
    NewsListResponse,
    NewsSearchItem,
    NewsSearchResponse,
)
from utils.auth import get_optional_user
from utils.response import success_response

router = APIRouter(prefix="/api/news", tags=["news"])


def _to_item(news_row, categories: dict[int, str]) -> dict:
    """ORM 对象 -> 列表项 dict。分类名顺手带上，省前端一次请求。"""
    return {
        "id": news_row.id,
        "title": news_row.title,
        "description": news_row.description,
        "image": news_row.image,
        "author": news_row.author,
        "categoryId": news_row.category_id,
        "categoryName": categories.get(news_row.category_id, ""),
        "publishTime": news_row.publish_time,
        "views": news_row.views,
    }


@router.get("/categories")
async def get_categories(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    categories = await news_cache.get_categories(db, skip, limit)
    return success_response(data=categories)


@router.get("/feed", response_model=NewsListResponse)
async def get_feed(
    category_id: int | None = Query(None, alias="categoryId"),
    sort: str = Query("latest", pattern="^(latest|hot|personalized)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, alias="pageSize", ge=1, le=50),
    user: User | None = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """
    首页信息流。

    与 `/list` 的区别：`/list` 要求必须传 categoryId（分类列表页用），
    这个接口不传就返回全部（首页用），另外支持按热度排序和个性化排序。

    sort=personalized 需要登录；未登录时自动退化为时间序，
    而不是返回 401 —— 首页不该因为没登录就白屏。
    """
    items, total, categories = await news_feed.get_feed(
        db,
        category_id=category_id,
        sort=sort,
        user_id=user.id if user else None,
        page=page,
        page_size=page_size,
    )
    offset = (page - 1) * page_size
    payload = [_to_item(n, categories) for n in items]
    return success_response(
        data=NewsListResponse(
            list=[NewsListItem(**p) for p in payload],
            total=total,
            hasMore=(offset + len(payload)) < total,
        )
    )


@router.get("/hot", response_model=HotNewsResponse)
async def get_hot(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """
    热度榜。

    热度 = (浏览量+1) / (发布时长小时+2)^1.5，见 crud/news_feed.hot_score。
    带出 hotScore 是为了让前端能展示「为什么它排第一」，
    纯 opaque 的排名不利于排查排序问题。
    """
    from datetime import datetime

    scored, total, categories = await news_feed.get_hot_news(db, limit=limit)
    items = [
        HotNewsItem(**_to_item(n, categories), hotScore=round(score, 6))
        for n, score in scored
    ]
    return success_response(data=HotNewsResponse(list=items, total=total, updatedAt=datetime.now()))


@router.get("/search", response_model=NewsSearchResponse)
async def search(
    keyword: str = Query(..., min_length=1, max_length=100),
    category_id: int | None = Query(None, alias="categoryId"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, alias="pageSize", ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """
    关键词搜索，走项目里已有的 BM25 实现。

    响应带 tookMs，方便前端和监控观察检索性能是否退化。
    """
    results, total, took = await news_search.search_news(
        db,
        keyword,
        category_id=category_id,
        page=page,
        page_size=page_size,
    )
    categories = await news_feed.get_categories_map(db)
    offset = (page - 1) * page_size
    items = [
        NewsSearchItem(
            **_to_item(row, categories),
            score=round(score, 4),
            matchedTerms=terms[:8],
        )
        for row, score, terms in results
    ]
    return success_response(
        data=NewsSearchResponse(
            keyword=keyword.strip(),
            list=items,
            total=total,
            hasMore=(offset + len(items)) < total,
            tookMs=round(took, 2),
        )
    )

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
