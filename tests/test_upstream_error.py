"""
上游错误透传测试。

背景：以前所有 httpx.HTTPError 都统一报「连接大模型服务失败」。
实际排查时上游返回的是 400 Arrearage（账号欠费），客户端看到的却是
「连接失败」—— 于是去查网络、代理、防火墙，全是死路。
真实原因是账单。

这些用例锁住「上游错误码必须透出来」这个行为。
"""
import httpx
import pytest

from ai.news_qa import _upstream_error, upstream_status_to_http


def _resp(status: int, body: dict | None = None, text: str = "") -> httpx.Response:
    if body is not None:
        return httpx.Response(status, json=body)
    return httpx.Response(status, text=text)


def test_upstream_error_extracts_code_and_message():
    resp = _resp(
        400,
        {
            "error": {
                "code": "Arrearage",
                "message": "please make sure your account is in good standing",
            }
        },
    )
    got = _upstream_error(resp)
    assert "Arrearage" in got
    assert "good standing" in got


def test_upstream_error_handles_non_json():
    """上游返回 HTML 错误页时不能抛异常，要有兜底"""
    resp = _resp(502, text="<html>Bad Gateway</html>")
    assert "Bad Gateway" in _upstream_error(resp)


def test_upstream_error_handles_empty_body():
    resp = _resp(500, body={})
    assert _upstream_error(resp) == ""


@pytest.mark.parametrize(
    "upstream_code,expected",
    [
        (401, 502),  # 密钥无效：服务端配置问题，不该报客户端 401
        (402, 503),  # 欠费：服务不可用
        (429, 429),  # 上游限流：调用方该退避
        (400, 502),  # 其他上游错误
        (500, 502),
    ],
)
def test_upstream_status_mapping(upstream_code, expected):
    assert upstream_status_to_http(upstream_code) == expected


def test_arrearage_is_not_reported_as_connection_failure():
    """
    回归测试：欠费不能被报成「连接失败」。
    这条正是之前浪费排查时间的原因。
    """
    resp = _resp(400, {"error": {"code": "Arrearage", "message": "overdue"}})
    msg = _upstream_error(resp)
    assert msg != ""
    assert "连接" not in msg
    assert "Arrearage" in msg