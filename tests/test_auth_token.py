"""Token 认证相关测试，重点覆盖 token 过期分支与 401 链路"""
from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException, status

from crud import users
from models.users import User, UserToken
from tests.fakes import FakeResult, FakeSession
from utils.auth import get_current_user


def make_token_row(expires_at: datetime) -> UserToken:
    return UserToken(id=1, user_id=7, token="a-token", expires_at=expires_at)


class TokenSession:
    """get_user_by_token 会执行两次查询：先查 token 表，再查 user 表"""

    def __init__(self, token_row):
        self.token_row = token_row
        self.execute_calls = 0

    async def execute(self, stmt):
        self.execute_calls += 1
        if self.execute_calls == 1:
            return FakeResult(one=self.token_row)
        return FakeResult(one=User(id=7, username="tester"))


async def test_valid_token_returns_user():
    session = TokenSession(make_token_row(datetime.now() + timedelta(days=7)))

    user = await users.get_user_by_token(session, "a-token")

    assert user is not None
    assert user.id == 7
    assert session.execute_calls == 2, "token 有效时应先查 token 表再查 user 表"


async def test_expired_token_returns_none():
    """token 已过期：只查一次 token 表就短路返回，不再查 user 表"""
    session = TokenSession(make_token_row(datetime.now() - timedelta(seconds=1)))

    user = await users.get_user_by_token(session, "a-token")

    assert user is None
    assert session.execute_calls == 1, "token 过期后不应继续查 user 表"


async def test_missing_token_returns_none():
    session = TokenSession(None)

    user = await users.get_user_by_token(session, "not-exist")

    assert user is None
    assert session.execute_calls == 1


async def test_get_current_user_raises_401_on_invalid_token(monkeypatch):
    """无效 token 时 get_current_user 必须抛 FastAPI 的 401"""

    async def fake_get_user_by_token(db, token):
        return None

    monkeypatch.setattr(users, "get_user_by_token", fake_get_user_by_token)

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(authorization="Bearer invalid-token", db=FakeSession())

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


async def test_get_current_user_returns_user_on_valid_token(monkeypatch):
    """有效 token 时 get_current_user 直接返回用户对象"""

    async def fake_get_user_by_token(db, token):
        return User(id=7, username="tester")

    monkeypatch.setattr(users, "get_user_by_token", fake_get_user_by_token)

    user = await get_current_user(authorization="Bearer good-token", db=FakeSession())

    assert user.username == "tester"


def test_protected_endpoint_returns_401_not_500(client_unauthenticated):
    """
    回归测试：utils/auth.py 曾从 http.client 导入 HTTPException，
    导致 token 无效时抛标准库异常、FastAPI 捕获不到、接口返回 500。
    修复后必须返回 401。
    """
    resp = client_unauthenticated.get(
        "/api/user/info", headers={"Authorization": "Bearer expired-token"}
    )

    assert resp.status_code == 401
    body = resp.json()
    assert body["code"] == 401
    assert body["message"] == "无效的令牌或者已过期"
    assert body["data"] is None


def test_protected_endpoint_without_header_is_rejected(client_unauthenticated):
    """缺少 Authorization 头时由 Header(...) 校验拦截，返回 422"""
    resp = client_unauthenticated.get("/api/user/info")

    assert resp.status_code == 422


def test_401_body_uses_unified_response_format(client_unauthenticated):
    """异常处理器输出的也是统一响应体，并带上 request_id 便于对日志"""
    resp = client_unauthenticated.get(
        "/api/favorite/list", headers={"Authorization": "Bearer expired-token"}
    )

    assert resp.status_code == 401
    assert set(resp.json()) == {"code", "message", "data", "request_id"}