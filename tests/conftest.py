"""
测试公共依赖。

测试策略：纯 mock，不连真实的 MySQL / Redis。
- 数据库用FakeSession 顶替，只需实现被测代码真正调用的几个方法
- Redis 用 monkeypatch 打桩crud.news_cache 模块里的缓存函数
- 接口测试用 app.dependency_overrides 覆盖 get_db / get_current_user
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# 保证能import 到项目根目录下的模块
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from config.db_conf import get_db  # noqa: E402
from crud import news_cache  # noqa: E402
from main import app  # noqa: E402
from tests.fakes import FakeResult, FakeSession  # noqa: E402,F401
from utils.auth import get_current_user  # noqa: E402


@pytest.fixture
def fake_session():
    # 默认：查库无结果、聚合计数为 0
    return FakeSession(result=FakeResult(rows=[], scalar=0))


@pytest.fixture
def stub_redis(monkeypatch):
    """
    打桩 crud.news_cache 里所有 Redis 读写函数，避免测试真去连 Redis。

    注意：crud/news_cache.py 用的是 `from cache.news_cache import get_xxx`，
    函数名已经被复制到 crud.news_cache 命名空间，所以必须打在 crud.news_cache 上。
    """
    mocks = {}
    for name in (
        "get_cached_categories",
        "set_cache_categories",
        "get_cache_news_lists",
        "set_cache_news_list",
        "get_cache_news_detail",
        "set_cache_news_detail",
        "get_cache_related_news",
        "set_cache_related_news",
        "get_news_views",
        "init_news_views",
        "increment_news_views",
    ):
        mock = AsyncMock(return_value=None)
        monkeypatch.setattr(news_cache, name, mock)
        mocks[name] = mock
    return mocks


@pytest.fixture(autouse=True)
def stub_embeddings(monkeypatch):
    """
    默认把向量化打桩成确定性的本地实现。

    不打桩的话凡是走到 retrieve() 的用例会真的去请求 DashScope，
    没配 key 时每个用例要等 4 秒网络超时。测试里用固定 hash 派生向量，
    保证可重复；需要断言具体融合行为的用例自行 monkeypatch 覆盖。
    """
    import hashlib

    import ai.retriever as retriever

    def fake_vector(text, dim=16):
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        return [digest[i] / 255.0 for i in range(dim)]

    async def fake_embed_texts(texts, use_cache=True):
        return [fake_vector(t) for t in texts]

    async def fake_embed_query(text):
        return fake_vector(text)

    monkeypatch.setattr(retriever, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(retriever, "embed_query", fake_embed_query)


@pytest.fixture(autouse=True)
def stub_rate_limit_redis(monkeypatch):
    """
    默认给限流中间件装一个内存版 Redis。

    不打桩的话每个请求都会真去连 Redis，连不上要等 TCP 超时，
    整套测试会从 9 秒拖到 27 秒。需要断言具体限流行为的用例
    自行用 monkeypatch.setattr 换成自己的 FakeRedis（见 test_rate_limit.py）。
    """
    import utils.rate_limit as rate_limit

    class FakeRedis:
        async def eval(self, script, numkeys, key, capacity, rate, cost, now_ms):
            return [1, str(float(capacity) - 1)]

    monkeypatch.setattr(rate_limit, "redis_client", FakeRedis())


@pytest.fixture
def client_authed(fake_session, stub_redis):
    """已登录用户 + 假数据库 + Redis 打桩，用于验证正常业务响应"""
    from models.users import User

    async def _override_db():
        yield fake_session

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = lambda: User(id=1, username="tester")
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def client_unauthenticated(stub_redis):
    """
    只覆盖 get_db、不覆盖鉴权，用于验证鉴权失败链路。
    第一次查询（查 user_token 表）返回 None，模拟 token 无效/已过期。
    """
    session = FakeSession(result=FakeResult(one=None))

    async def _override_db():
        yield session

    app.dependency_overrides[get_db] = _override_db
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()