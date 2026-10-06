"""用户 CRUD 测试：密码哈希、Token 生命周期、部分字段更新"""
import pytest
from fastapi import HTTPException, status

from crud import users
from models.users import User
from schemas.users import UserRequest, UserUpdateRequest
from tests.fakes import FakeResult, FakeSession
from utils import security


class FakeUpdateStmt:
    """记录 update() 语句 .values() 里实际传了哪些字段"""

    def __init__(self):
        self.captured = {}

    def where(self, *args):
        return self

    def values(self, **kwargs):
        self.captured.update(kwargs)
        return self


# ---------------- 密码安全 ----------------

def test_password_is_hashed_not_stored_plaintext():
    hashed = security.get_hash_password("my-password")

    assert hashed != "my-password"
    assert hashed.startswith("$2")  # bcrypt 哈希
    assert security.verify_password("my-password", hashed)
    assert not security.verify_password("wrong-password", hashed)


def test_bcrypt_salt_makes_same_password_different_hash():
    a = security.get_hash_password("same-password")
    b = security.get_hash_password("same-password")

    assert a != b, "bcrypt 每次加随机盐，同一密码的哈希应不同"
    assert security.verify_password("same-password", a)
    assert security.verify_password("same-password", b)


def test_password_longer_than_72_bytes_is_supported():
    """
    bcrypt 原生拒绝超过 72 字节的密码。
    这里的实现先做 sha256 摘要再 base64，而不是截断——
    截断会让前 72 位相同的长密码得到相同哈希，形成安全隐患。
    """
    long_password = "x" * 200
    hashed = security.get_hash_password(long_password)

    assert security.verify_password(long_password, hashed)
    # 只差最后一个字符的长密码必须被拒绝（截断实现会漏掉这一条）
    assert not security.verify_password("x" * 199 + "y", hashed)


def test_chinese_password_round_trips():
    """中文密码按 utf-8 编码后可能超 72 字节，也要能正常校验"""
    password = "密码" * 30
    hashed = security.get_hash_password(password)

    assert security.verify_password(password, hashed)
    assert not security.verify_password("密码" * 29 + "错", hashed)


def test_verify_password_tolerates_corrupt_hash():
    """
    回归测试：库里出现脏数据（空值或格式非法的哈希）时，
    登录接口应返回验证失败，而不是抛异常变成 500。
    """
    assert security.verify_password("any", "") is False
    assert security.verify_password("any", "not-a-valid-bcrypt-hash") is False


async def test_create_user_stores_hashed_password():
    db = FakeSession()
    user_data = UserRequest(username="alice", password="secret123")

    await users.create_user(db, user_data)

    created = db.added[0]
    assert created.username == "alice"
    assert created.password != "secret123"
    assert security.verify_password("secret123", created.password)


async def test_authenticate_user_rejects_wrong_password():
    class OneUserSession(FakeSession):
        async def execute(self, stmt):
            self.execute_calls += 1
            return FakeResult(one=User(id=1, username="alice", password=security.get_hash_password("right")))

    assert await users.authenticate_user(OneUserSession(), "alice", "right") is not None
    assert await users.authenticate_user(OneUserSession(), "alice", "wrong") is None


async def test_authenticate_user_returns_none_for_missing_user():
    class EmptySession(FakeSession):
        async def execute(self, stmt):
            self.execute_calls += 1
            return FakeResult(one=None)

    assert await users.authenticate_user(EmptySession(), "ghost", "any") is None


# ---------------- Token ----------------

async def test_create_token_generates_uuid_and_7_day_expiry():
    from datetime import datetime, timedelta

    class NoExistingTokenSession(FakeSession):
        async def execute(self, stmt):
            self.execute_calls += 1
            return FakeResult(one=None)

    db = NoExistingTokenSession()
    before = datetime.now()
    token = await users.create_token(db, user_id=7)

    assert isinstance(token, str) and len(token) == 36  # uuid4 字符串
    created = db.added[0]
    assert created.user_id == 7
    assert created.token == token
    # 有效期约 7 天
    assert before + timedelta(days=6) < created.expires_at <= before + timedelta(days=8)


async def test_create_token_commits_when_replacing_existing_token():
    """
    回归测试：create_token 原本只在「首次创建」分支里 commit，
    重新登录（覆盖已有 token）时不提交。那样登录接口已经把新 token
    返回给客户端、但数据库里还是旧 token，客户端立刻用它会偶发 401。
    """
    from datetime import datetime, timedelta

    from models.users import UserToken

    existing = UserToken(id=1, user_id=7, token="old-token", expires_at=datetime.now())

    class ExistingTokenSession(FakeSession):
        async def execute(self, stmt):
            self.execute_calls += 1
            return FakeResult(one=existing)

    db = ExistingTokenSession()
    before = datetime.now()
    token = await users.create_token(db, user_id=7)

    assert token != "old-token"
    assert existing.token == token, "应复用并覆盖已有记录"
    assert db.added == [], "已有 token 时不应再 insert"
    assert db.commits == 1, "覆盖已有 token 后必须立即提交，否则客户端拿到的 token 尚未落库"
    assert existing.expires_at > before + timedelta(days=6)


# ---------------- 用户信息更新 ----------------

async def test_update_user_raises_fastapi_404(monkeypatch):
    """
    回归测试：crud/users.py 曾从 http.client 导入 HTTPException，
    用户不存在时抛出的异常 FastAPI 捕获不到，应返回 404 而非 500。
    """
    stmt = FakeUpdateStmt()
    monkeypatch.setattr(users, "update", lambda table: stmt)

    with pytest.raises(HTTPException) as exc_info:
        await users.update_user(FakeSession(result=FakeResult(rowcount=0)), "ghost",
                                UserUpdateRequest(nickname="新昵称"))

    assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
    assert exc_info.value.detail == "用户不存在"


async def test_update_user_only_writes_provided_non_null_fields(monkeypatch):
    """靠 Pydantic 的 exclude_unset/exclude_none 实现部分字段更新"""
    stmt = FakeUpdateStmt()
    monkeypatch.setattr(users, "update", lambda table: stmt)

    updated = User(id=1, username="tester", nickname="新昵称")

    async def fake_get_user_by_username(db, username):
        return updated

    monkeypatch.setattr(users, "get_user_by_username", fake_get_user_by_username)

    result = await users.update_user(
        FakeSession(result=FakeResult(rowcount=1)), "tester",
        UserUpdateRequest(nickname="新昵称")
    )

    assert stmt.captured == {"nickname": "新昵称"}, "未传的字段不应出现在 SET 子句里"
    assert result is updated


# ---------------- 密码修改 ----------------

async def test_update_password_rejects_wrong_old_password():
    db = FakeSession()
    user = User(id=1, username="tester", password=security.get_hash_password("old-pass"))
    old_hash = user.password

    ok = await users.update_password(db, user, "wrong-pass", "new-pass")

    assert ok is False
    assert user.password == old_hash, "旧密码错误时不应改动密码字段"
    assert db.commits == 0
    assert db.added == []


async def test_update_password_rehashes_on_success():
    db = FakeSession()
    user = User(id=1, username="tester", password=security.get_hash_password("old-pass"))
    old_hash = user.password

    ok = await users.update_password(db, user, "old-pass", "new-pass")

    assert ok is True
    assert user.password != old_hash, "密码应被重新哈希"
    assert security.verify_password("new-pass", user.password)
    assert db.commits == 1
    assert user in db.added, "应把 ORM 对象交回 session 托管以确保提交"