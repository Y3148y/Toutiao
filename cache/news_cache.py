from typing import List, Dict, Any, Optional

from config.cache_conf import get_json_cache, set_cache, get_cache, delete_cache, icr, exp

CATEGORIES_KEY = "news:categories"
NEWS_LIST_PREFIX = "news_list:"
NEWS_DETAIL_PREFIX = "news_detail:"
NEWS_RELATED_PREFIX = "news_related:"

# 读取
async def get_cached_categories():
    return await get_json_cache(CATEGORIES_KEY)
# 写入
async def set_cache_categories(data: List[Dict[str, Any]], expire: int = 7200):
    return await set_cache(CATEGORIES_KEY, data, expire)


# 设计新闻列表缓存
# 唯一key: news:list:分类ID:页码:每页数量 + 列表数量 + 过期时间

async def set_cache_news_list(category_id: Optional[int], page: int, size: int, news_list: list[Dict[str, Any]], expire: int = 600):
    category_part = category_id if category_id is not None else "all"
    key = f"{NEWS_LIST_PREFIX}{category_part}:{page}:{size}"
    return await set_cache(key, news_list, expire)

async def get_cache_news_lists(category_id: Optional[int], page: int, size: int):
    category_part = category_id if category_id is not None else "all"
    key = f"{NEWS_LIST_PREFIX}{category_part}:{page}:{size}"
    return await get_json_cache(key)

# detail:（同分类id的新闻）
async def set_cache_news_detail(news_id: Optional[int], news_detail: dict, expire: int = 600):
    key = f"{NEWS_DETAIL_PREFIX}{news_id}"
    return await set_cache(key, news_detail, expire)

async def get_cache_news_detail(news_id: Optional[int]):
    key = f"{NEWS_DETAIL_PREFIX}{news_id}"
    return await get_json_cache(key)

async def set_cache_related_news(news_id: int, related_news: list, expire: int = 600):
    key = f"{NEWS_RELATED_PREFIX}{news_id}"
    return await set_cache(key, related_news, expire)

async def get_cache_related_news(news_id: int):
    key = f"{NEWS_RELATED_PREFIX}{news_id}"
    return await get_json_cache(key)

async def delete_cache_related_news(news_id: int):
    """手工关联变更后必须调用，否则用户看到的还是旧推荐，直到 TTL 到期"""
    key = f"{NEWS_RELATED_PREFIX}{news_id}"
    return await delete_cache(key)

async def get_news_views(news_id: int) -> int:
    key = f"news:views:{news_id}"
    views = await get_cache(key)
    if views is None:
        # 如果缓存中没有浏览量，从数据库加载并初始化（避免第一次请求查库）
        # 但此处我们可以在第一次查询时从数据库加载，也可以让数据库初始化
        # 这里我们简单返回 0，由调用方决定是否从数据库补齐
        return 0
    return int(views)

async def init_news_views(news_id: int, db_views: int):
    """从数据库初始化缓存浏览量（首次查询时调用）"""
    key = f"news:views:{news_id}"
    await set_cache(key, db_views)  # 设置过期时间，可选

async def increment_news_views(news_id: int) -> int:
    """Redis 原子递增浏览量，并返回新值"""
    key = f"news:views:{news_id}"
    new_views = await icr(key)
    await exp(key)
    return new_views




