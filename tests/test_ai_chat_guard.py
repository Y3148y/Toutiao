"""
`/api/ai/chat` 的防护测试。

这个端点是全项目**花钱最直接**的一处：没有检索护栏，用户传什么转发什么。
所以锁四类防线：

1. 默认要求登录 —— 公开等于白送模型额度
2. 入参上限 —— 否则一次请求就能把上下文打满
3. 成本计量 —— 不计量就是成本黑洞
4. 上游错误要能在**响应头发出去之前**拿到状态码

不连接真实大模型，用桩替代 HTTP 响应。
"""
import json

import httpx  # noqa: F401
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import ai.ai_chat as ai_chat


# --------------------------------------------------------------------------
# 桩
# --------------------------------------------------------------------------
class FakeResponse:
    """模拟 httpx 的流式响应"""

    def __init__(self, lines, status_code=200, body=b""):
        self._lines = lines
        self.status_code = status_code
        self._body = body
        self.closed = False

    async def aread(self):
        return self._body

    async def aclose(self):
        self.closed = True

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class FakeClient:
    """模拟 httpx.AsyncClient"""

    def __init__(self, lines=(), status_code=200, body=b"", recorder=None, **kwargs):
        self._lines = lines
        self._status = status_code
        self._body = body
        self._recorder = recorder
        self.closed = False

    async def aclose(self):
        self.closed = True

    def build_request(self, method, url, headers=None, json=None):
        self.sent = {"method": method, "url": url, "headers": headers, "json": json}
        return "REQ"

    async def send(self, req, stream=False):
        if self._recorder is not None:
            self._recorder.append(self)
        return FakeResponse(self._lines, self._status, self._body)


def _sse(obj) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False)


def _token(text: str) -> str:
    return _sse({"choices": [{"delta": {"content": text}, "finish_reason": None}]})


def _usage(inp: int, out: int) -> str:
    return _sse(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": inp,
                "completion_tokens": out,
                "total_tokens": inp + out,
            },
        }
    )


def _patch_client(monkeypatch, **kwargs):
    made: list = []

    def factory(**_kw):
        made.append(FakeClient(**kwargs))
        return made[-1]

    monkeypatch.setattr(ai_chat.httpx, "AsyncClient", factory)
    monkeypatch.setattr(ai_chat, "is_configured", lambda: True)
    return made


def _app():
    """只挂 ai_chat 路由的小应用 + TestClient，避免拉起整个 main"""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(ai_chat.router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _auth_off_by_default(request, monkeypatch):
    """
    除鉴权专项用例外，其余用例都跑在「已关闭鉴权」模式下，
    这样用例聚焦在被测的防护项，不会互相干扰。

    用 request.node.name 白名单，让鉴权相关用例自己控制开关。
    """
    if "auth" in request.node.name:
        return
    monkeypatch.setattr(ai_chat, "REQUIRE_AUTH", False)


# --------------------------------------------------------------------------
# 1. 鉴权
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_requires_auth_by_default():
    """默认 REQUIRE_AUTH 为真 —— 这是防白嫖的第一道闸"""
    assert ai_chat.REQUIRE_AUTH is True, (
        "默认必须要求登录；确需关闭请显式设置 AI_CHAT_REQUIRE_AUTH=false"
    )


@pytest.mark.asyncio
async def test_auth_enabled_rejects_missing_token(monkeypatch):
    """开启鉴权时缺 Token 直接 401，且不会调用大模型"""
    monkeypatch.setattr(ai_chat, "REQUIRE_AUTH", True)
    made = _patch_client(monkeypatch, lines=["data: [DONE]"])

    resp = _app().post(
        "/api/ai/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 401, resp.text
    assert made == [], "鉴权失败时绝不能建立上游连接（会白花钱）"


@pytest.mark.asyncio
async def test_auth_enabled_rejects_invalid_token(monkeypatch):
    """
    带了一个无效 Token 应被拒绝。

    这条会真的去查 user_token 表，所以把 get_current_user 换成桩 ——
    鉴权逻辑本身由 tests/test_auth_token.py 覆盖，这里只验证
    「开启了鉴权时确实会走到校验」。
    """
    monkeypatch.setattr(ai_chat, "REQUIRE_AUTH", True)

    called = []

    async def fake_auth(authorization=None, db=None):
        called.append(authorization)
        raise HTTPException(status_code=401, detail="无效的令牌或者已过期")

    monkeypatch.setattr(ai_chat, "get_current_user", fake_auth)
    _patch_client(monkeypatch, lines=["data: [DONE]"])

    resp = _app().post(
        "/api/ai/chat",
        headers={"Authorization": "Bearer totally-invalid-token"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 401, resp.text
    assert called == ["Bearer totally-invalid-token"], "Token 必须原样传给校验逻辑"


@pytest.mark.asyncio
async def test_auth_can_be_disabled_for_local_dev(monkeypatch):
    """显式关闭后放行，且能正常建立上游连接"""
    monkeypatch.setattr(ai_chat, "REQUIRE_AUTH", False)
    made = _patch_client(monkeypatch, lines=["data: [DONE]"])

    resp = _app().post(
        "/api/ai/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 200, resp.text
    assert len(made) == 1


# --------------------------------------------------------------------------
# 2. 入参上限
# --------------------------------------------------------------------------


def test_messages_count_is_capped():
    """messages 条数有硬上限，否则一次请求就能把上下文打满"""
    schema = ai_chat.ChatRequest.model_json_schema()
    max_items = schema["properties"]["messages"]["maxItems"]
    assert max_items == ai_chat.MAX_MESSAGES


def test_content_length_is_capped():
    """单条 content 长度有硬上限"""
    schema = ai_chat.ChatMessage.model_json_schema()
    assert schema["properties"]["content"]["maxLength"] == ai_chat.MAX_CONTENT_CHARS


def test_role_is_restricted():
    """role 必须来自合法集合，否则可以伪造 system 注入"""
    schema = ai_chat.ChatMessage.model_json_schema()
    pattern = schema["properties"]["role"]["pattern"]
    assert "system" in pattern and "user" in pattern and "assistant" in pattern


def test_rejects_empty_messages():
    with pytest.raises(Exception):
        ai_chat.ChatRequest(messages=[])


def test_rejects_oversized_content():
    with pytest.raises(Exception):
        ai_chat.ChatMessage(role="user", content="x" * (ai_chat.MAX_CONTENT_CHARS + 1))


def test_rejects_illegal_role():
    with pytest.raises(Exception):
        ai_chat.ChatMessage(role="developer", content="hi")


# --------------------------------------------------------------------------
# 3. 成本计量
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_usage_frame_after_finish_reason_is_captured(monkeypatch):
    """
    usage 帧排在 finish_reason 之后，提前返回就会漏计 —— 这个坑在
    news_qa 流式路径上真实漏过一次，这里必须同样覆盖。
    """
    _patch_client(
        monkeypatch,
        lines=[_token("你好"), _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}), _usage(100, 20), "data: [DONE]"],
    )

    recorded: list = []

    async def fake_record(model, usage):
        recorded.append((model, usage))

    monkeypatch.setattr("ai.cost.record_chat_usage", fake_record)

    _app().post(
        "/api/ai/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert len(recorded) == 1, f"usage 未被计量：{recorded}"
    assert recorded[0][1]["prompt_tokens"] == 100


@pytest.mark.asyncio
async def test_requests_include_usage_option(monkeypatch):
    """必须显式要求 usage，否则拿不到 token 数"""
    made = _patch_client(monkeypatch, lines=["data: [DONE]"])

    _app().post(
        "/api/ai/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    sent = made[0].sent
    assert sent["json"]["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_model_comes_from_shared_config(monkeypatch):
    """模型名必须统一走 ai/config，不能在这个文件里另读环境变量"""
    from ai.config import DASHSCOPE_CHAT_MODEL

    made = _patch_client(monkeypatch, lines=["data: [DONE]"])

    _app().post(
        "/api/ai/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert made[0].sent["json"]["model"] == DASHSCOPE_CHAT_MODEL


@pytest.mark.asyncio
async def test_missing_usage_is_logged_not_silently_dropped(monkeypatch, caplog):
    """没拿到 usage 必须留痕，否则账单凭空少一段事后无从发现"""
    _patch_client(monkeypatch, lines=[_token("hi"), "data: [DONE]"])

    with caplog.at_level("WARNING"):
        _app().post(
            "/api/ai/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )
    assert any("usage" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# 4. 上游错误处理
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upstream_error_status_is_mapped_not_swallowed(monkeypatch):
    """
    上游返回 402(欠费) 时，本服务应返回 503 并保留上游错误体。

    原实现在响应头发出后才 raise_for_status()，客户端只能看到断流。
    """
    _patch_client(
        monkeypatch,
        lines=(),
        status_code=402,
        body=b'{"error":{"code":"Arrearage","message":"balance is insufficient"}}',
    )

    resp = _app().post(
        "/api/ai/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert resp.status_code == 503, resp.text
    assert "Arrearage" in resp.text, "上游错误码必须透出，否则无法排查欠费"


@pytest.mark.asyncio
async def test_missing_api_key_returns_500(monkeypatch):
    monkeypatch.setattr(ai_chat, "is_configured", lambda: False)

    resp = _app().post(
        "/api/ai/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 500
    assert "DASHSCOPE_API_KEY" in resp.text


@pytest.mark.asyncio
async def test_sse_headers_disable_proxy_buffering(monkeypatch):
    """nginx 缓冲会把 SSE 攒成一坨最后一次性吐出"""
    _patch_client(monkeypatch, lines=["data: [DONE]"])

    resp = _app().post(
        "/api/ai/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.headers.get("x-accel-buffering") == "no"
    assert resp.headers.get("cache-control") == "no-cache"