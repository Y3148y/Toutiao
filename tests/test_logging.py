"""
日志与请求 ID 中间件测试。

重点验证三件事：
1. 每个响应都带 X-Request-ID
2. 传入自己的 X-Request-ID 时会被复用（便于网关/上游链路串联）
3. contextvars 在请求结束后被还原，不会污染下一个请求
"""
import pytest

from utils.logging_conf import get_request_id, request_id_var, setup_logging


def test_response_has_request_id_header(client_authed):
    resp = client_authed.get("/api/news/categories")

    assert "X-Request-ID" in resp.headers
    assert len(resp.headers["X-Request-ID"]) == 8


def test_request_id_is_unique_per_request(client_authed):
    first = client_authed.get("/api/news/categories").headers["X-Request-ID"]
    second = client_authed.get("/api/news/categories").headers["X-Request-ID"]

    assert first != second, "不同请求应生成不同的 ID"


def test_incoming_request_id_is_reused(client_authed):
    """上游已经带了 ID 就复用它，这样网关-应用-日志能串成一条链路"""
    resp = client_authed.get(
        "/api/news/categories", headers={"X-Request-ID": "trace-abc"}
    )

    assert resp.headers["X-Request-ID"] == "trace-abc"


def test_request_id_context_is_reset_after_request(client_authed):
    """
    请求结束后 contextvar 必须还原。
    否则同协程复用（比如连接池复用）时，下一个请求的日志会带上上一次的 ID。
    """
    assert get_request_id() == "-", "请求结束后应回落到默认值"

    client_authed.get("/api/news/categories", headers={"X-Request-ID": "first-id"})

    assert get_request_id() == "-"


def test_contextvar_default_is_placeholder():
    """没有中间件时取到占位符而不是 None，避免日志格式化报错"""
    assert request_id_var.get() == "-"
    assert get_request_id() == "-"


def test_404_response_also_carries_request_id(client_unauthenticated):
    """异常响应同样要带 ID，否则用户报障时无法定位日志"""
    resp = client_unauthenticated.get("/api/not-exist")

    assert resp.status_code == 404
    assert "X-Request-ID" in resp.headers


def test_setup_logging_is_idempotent():
    """重复调用不能叠加 handler，否则日志会重复输出多次"""
    import logging

    root = logging.getLogger()
    setup_logging()
    before = len(root.handlers)
    setup_logging()
    after = len(root.handlers)

    assert before == after


def test_debug_mode_defaults_to_disabled():
    """
    回归测试：utils/exception.py 曾硬编码 DEBUG_MODE = True，
    等于生产环境会把异常堆栈返回给前端。现在默认必须是关闭的。
    """
    from utils import exception

    assert exception.DEBUG_MODE is False, "DEBUG_MODE 默认必须为 False，避免泄露堆栈"