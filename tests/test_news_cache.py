"""Redis 缓存相关测试，重点覆盖缓存命中 / 未命中两条分支"""
import json
from datetime import datetime
from unittest.mock import AsyncMock

from crud import news_cache
from models.news import Category, News
from tests.fakes import FakeResult, FakeSession


def make_news(**overrides):
    data = dict(
        id=1,
        title="标题",
        description="摘要",
        image=None,
        author="作者",
        category_id=2,
        views=0,
        publish_time=datetime(2026, 1, 1, 12, 0, 0),
        content="正文",
    )
    data.update(overrides)
    return News(**data)


# ---------------- 分类缓存 ----------------

async def test_categories_cache_hit_skips_db(monkeypatch):
    """缓存命中：直接返回缓存，绝不查数据库，也不回写缓存"""
    monkeypatch.setattr(
        news_cache, "get_cached_categories", AsyncMock(return_value=[{"id": 1, "name": "科技"}])
    )
    set_mock = AsyncMock()
    monkeypatch.setattr(news_cache, "set_cache_categories", set_mock)

    db = FakeSession(result=FakeResult(rows=[Category(id=1, name="科技")]))
    result = await news_cache.get_categories(db)

    assert result == [{"id": 1, "name": "科技"}]
    assert db.execute_calls == 0, "缓存命中时不应访问数据库"
    set_mock.assert_not_awaited()


async def test_categories_cache_miss_queries_db_and_writes_cache(monkeypatch):
    """缓存未命中：查库并回写缓存"""
    monkeypatch.setattr(news_cache, "get_cached_categories", AsyncMock(return_value=None))
    set_mock = AsyncMock()
    monkeypatch.setattr(news_cache, "set_cache_categories", set_mock)

    db = FakeSession(result=FakeResult(rows=[Category(id=1, name="科技"), Category(id=2, name="财经")]))
    result = await news_cache.get_categories(db)

    assert db.execute_calls == 1, "缓存未命中应查一次数据库"
    set_mock.assert_awaited_once()
    cached = set_mock.await_args.args[0]
    assert cached == [{"id": 1, "name": "科技"}, {"id": 2, "name": "财经"}]
    assert result == cached


async def test_categories_empty_result_does_not_write_cache(monkeypatch):
    """查库结果为空时不写缓存，避免缓存穿透"""
    monkeypatch.setattr(news_cache, "get_cached_categories", AsyncMock(return_value=None))
    set_mock = AsyncMock()
    monkeypatch.setattr(news_cache, "set_cache_categories", set_mock)

    db = FakeSession(result=FakeResult(rows=[]))
    result = await news_cache.get_categories(db)

    assert result == []
    assert db.execute_calls == 1
    set_mock.assert_not_awaited()


# ---------------- 新闻列表缓存 ----------------

async def test_news_list_cache_hit_rebuilds_orm(monkeypatch):
    """列表缓存命中：把 dict 还原成 News ORM 对象返回，且不查库"""
    payload = [
        {
            "id": 1,
            "title": "缓存里的标题",
            "description": None,
            "image": None,
            "author": "作者",
            "category_id": 2,
            "views": 7,
            "publish_time": "2026-01-01T12:00:00",
        }
    ]
    get_mock = AsyncMock(return_value=payload)
    monkeypatch.setattr(news_cache, "get_cache_news_lists", get_mock)
    monkeypatch.setattr(news_cache, "set_cache_news_list", AsyncMock())

    db = FakeSession()
    rows = await news_cache.get_news_list(db, category_id=2, skip=0, limit=10)

    assert db.execute_calls == 0, "列表缓存命中时不应访问数据库"
    assert len(rows) == 1
    assert isinstance(rows[0], News)
    assert rows[0].title == "缓存里的标题"
    assert rows[0].views == 7
    # 缓存 key 由分类ID + 页码 + 每页条数 拼成
    get_mock.assert_awaited_once_with(2, 1, 10)


async def test_news_list_cache_key_uses_page_number(monkeypatch):
    """页码由 offset/limit 推导，缓存 key 要跟着变"""
    get_mock = AsyncMock(return_value=[{"id": 1, "title": "t", "category_id": 2, "views": 0}])
    monkeypatch.setattr(news_cache, "get_cache_news_lists", get_mock)
    monkeypatch.setattr(news_cache, "set_cache_news_list", AsyncMock())

    db = FakeSession()
    await news_cache.get_news_list(db, category_id=2, skip=20, limit=10)

    get_mock.assert_awaited_once_with(2, 3, 10)  # 第 3 页


async def test_news_list_cache_miss_writes_serialized_cache(monkeypatch):
    """列表缓存未命中：回源查库并写缓存"""
    monkeypatch.setattr(news_cache, "get_cache_news_lists", AsyncMock(return_value=None))
    set_mock = AsyncMock()
    monkeypatch.setattr(news_cache, "set_cache_news_list", set_mock)

    db = FakeSession(result=FakeResult(rows=[make_news()]))
    rows = await news_cache.get_news_list(db, category_id=2, skip=0, limit=10)

    assert db.execute_calls == 1
    set_mock.assert_awaited_once()
    cached = set_mock.await_args.args[3]
    assert cached[0]["title"] == "标题"
    # datetime 必须被 model_dump(mode="json") 转成 ISO 字符串，
    # 否则 json.dumps 会抛 TypeError 且被 except 吞掉，导致缓存永远写不进去
    assert isinstance(cached[0]["publish_time"], str)
    json.dumps(cached)  # 不抛异常即通过


# ---------------- 浏览量 ----------------

async def test_increase_views_updates_redis_and_db(monkeypatch):
    """浏览量自增：Redis INCR 一次 + MySQL UPDATE 一次"""
    incr_mock = AsyncMock(return_value=11)
    monkeypatch.setattr(news_cache, "increment_news_views", incr_mock)

    db = FakeSession(result=FakeResult(rowcount=1))
    ok = await news_cache.increase_news_news(db, news_id=1)

    assert ok is True
    incr_mock.assert_awaited_once_with(1)
    assert db.execute_calls == 1, "应同步持久化到 MySQL"
    assert db.commits == 1


async def test_increase_views_returns_false_when_row_missing(monkeypatch):
    monkeypatch.setattr(news_cache, "increment_news_views", AsyncMock(return_value=1))

    db = FakeSession(result=FakeResult(rowcount=0))
    ok = await news_cache.increase_news_news(db, news_id=1)

    assert ok is False


async def test_news_detail_cache_hit_reads_views_from_redis(monkeypatch):
    """详情缓存命中：浏览量从 Redis 取，并重建为 News 对象"""
    detail_payload = {
        "id": 1,
        "title": "标题",
        "description": None,
        "image": None,
        "author": "作者",
        "category_id": 2,
        "views": 0,
        "publish_time": "2026-01-01T12:00:00",
        "content": "正文",
    }
    monkeypatch.setattr(news_cache, "get_cache_news_detail", AsyncMock(return_value=detail_payload))
    monkeypatch.setattr(news_cache, "get_news_views", AsyncMock(return_value=42))

    db = FakeSession()
    result = await news_cache.get_news_detail(db, news_id=1)

    assert isinstance(result, News)
    assert result.views == 42, "浏览量应取Redis 中的实时值"
    assert db.execute_calls == 0, "详情缓存命中时不应访问数据库"


async def test_news_detail_cache_miss_inits_views(monkeypatch):
    """详情缓存未命中：查库并把浏览量初始化进 Redis"""
    monkeypatch.setattr(news_cache, "get_cache_news_detail", AsyncMock(return_value=None))
    set_detail_mock = AsyncMock()
    monkeypatch.setattr(news_cache, "set_cache_news_detail", set_detail_mock)
    init_views_mock = AsyncMock()
    monkeypatch.setattr(news_cache, "init_news_views", init_views_mock)

    row = make_news(views=5)
    db = FakeSession(result=FakeResult(one=row))
    result = await news_cache.get_news_detail(db, news_id=1)

    assert result is row
    assert db.execute_calls == 1
    set_detail_mock.assert_awaited_once()
    # 缓存里的 content 必须是字符串，不能是 datetime 之类的不可序列化对象
    json.dumps(set_detail_mock.await_args.args[1])
    init_views_mock.assert_awaited_once_with(1, 5)