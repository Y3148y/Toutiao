"""统一响应体测试：所有接口都必须是 code/message/data，不能再有 msg"""
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from models.news import News
from tests.fakes import FakeResult

NEWS_ENDPOINTS = [
    ("GET", "/api/news/categories"),
    ("GET", "/api/news/list?categoryId=2"),
]


def seed_news_row(fake_session, stub_redis):
    """给假 DB 放一条新闻，并让详情缓存未命中，模拟真实的首次访问"""
    row = News(
        id=1,
        title="测试新闻",
        description="摘要",
        image=None,
        author="作者",
        category_id=2,
        views=0,
        publish_time=datetime(2026, 1, 1, 12, 0, 0),
        content="正文",
    )
    fake_session.result = FakeResult(rows=[row], one=row, scalar=1, rowcount=1)
    stub_redis["get_cache_news_detail"].return_value = None
    # 相关推荐走缓存命中分支（非空才命中，[] 会被当未命中而回源查库）
    stub_redis["get_cache_related_news"].return_value = [
        {"id": 2, "title": "相关推荐", "categoryId": 2}
    ]
    stub_redis["increment_news_views"].return_value = 1
    stub_redis["get_news_views"].return_value = 1
    return row


@pytest.mark.parametrize("method,url", NEWS_ENDPOINTS)
def test_news_endpoints_use_message_key(client_authed, method, url):
    """
    回归测试：routers/news.py 曾返回 "msg"，而其他模块返回 "message"，
    前端需要兼容两种字段名。统一后必须只有 message。
    """
    resp = client_authed.request(method, url)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "msg" not in body, "响应体不应再出现 msg 字段"
    assert set(body) == {"code", "message", "data"}
    assert body["code"] == 200


def test_news_detail_uses_message_key(client_authed, fake_session, stub_redis):
    """详情接口同样走统一响应体，且 views 取自 Redis 自增后的值"""
    seed_news_row(fake_session, stub_redis)

    resp = client_authed.get("/api/news/detail?id=1")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "msg" not in body
    assert set(body) == {"code", "message", "data"}
    data = body["data"]
    assert data["id"] == 1
    assert data["title"] == "测试新闻"
    assert data["views"] == 1, "views 应对齐 Redis 自增后的值，而不是少 1"
    assert data["relatedNews"] == [{"id": 2, "title": "相关推荐", "categoryId": 2}]


def test_news_list_returns_pagination_shape(client_authed, fake_session):
    """列表接口 data 里是 list/total/hasMore"""
    resp = client_authed.get("/api/news/list?categoryId=2&page=1&pageSize=10")

    data = resp.json()["data"]
    assert set(data) == {"list", "total", "hasMore"}
    assert data["list"] == []
    assert data["total"] == 0
    assert data["hasMore"] is False


def test_user_info_uses_message_key(client_authed):
    """user 模块本来就是 message，改动后仍要保持一致"""
    resp = client_authed.get("/api/user/info")

    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"code", "message", "data"}
    assert body["data"]["username"] == "tester"


def test_favorite_check_uses_message_key(client_authed):
    """已登录用户访问收藏接口，返回体同样统一"""
    resp = client_authed.get("/api/favorite/check?newsId=1")

    assert resp.status_code == 200
    body = resp.json()
    assert "msg" not in body
    assert set(body) == {"code", "message", "data"}


def test_news_detail_404_uses_unified_error_shape(client_authed, fake_session, stub_redis):
    """
    回归测试：routers/news.py 曾从 http.client 导入 HTTPException，
    抛出的异常 FastAPI 捕获不到，新闻不存在时应返回 404 而非 500。
    """
    # 缓存未命中 + 查库查不到该新闻
    stub_redis["get_cache_news_detail"].return_value = None
    fake_session.result = FakeResult(one=None)

    resp = client_authed.get("/api/news/detail?id=999")

    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == 404
    assert body["message"] == "新闻不存在"


def test_openapi_documents_all_endpoints():
    """
    确认全部业务接口都在。

    新闻 6（分类/列表/详情/信息流/搜索/热榜）
    收藏 5 / 历史 4 / 用户 5 / AI 4 + 根路径 = 25
    """
    from main import app

    paths = app.openapi()["paths"]
    assert len(paths) == 25  # 24 个业务接口 + 根路径 /

    assert set(paths["/api/news/detail"]) == {"get"}
    assert set(paths["/api/user/register"]) == {"post"}
    assert set(paths["/api/user/password"]) == {"put"}
    assert set(paths["/api/favorite/remove"]) == {"delete"}
    assert set(paths["/api/history/delete/{history_id}"]) == {"delete"}
    assert set(paths["/api/ai/chat"]) == {"post"}
    assert set(paths["/api/ai/news-qa"]) == {"post"}
    assert set(paths["/api/ai/news-qa/sync"]) == {"post"}