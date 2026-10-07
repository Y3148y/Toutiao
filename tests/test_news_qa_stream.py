"""
SSE 流式问答的帧序列测试。

锁的是一个真实漏过的坑：**流式请求完全不被成本计量**。

现象：sync 版 usage 正常（781 token），流式版 done 事件里 usage 恒为 null。
原因：DashScope 把 usage 放在 finish_reason 帧**之后**的单独一帧里，
而代码在 finish_reason 处就 return 了，那个帧永远读不到。
结果是钱照扣、账上查不到 —— 流式接口成了成本黑洞。

不连接真实大模型，用桩替代 HTTP 响应。
"""
import json
from datetime import datetime

import pytest

from ai.retriever import RetrievedNews


def _sse_line(obj) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False)


def _token(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}, "finish_reason": None}]}


def _finish(reason: str = "stop") -> dict:
    return {"choices": [{"delta": {}, "finish_reason": reason}], "usage": None}


def _usage_frame(inp: int, out: int) -> dict:
    """DashScope 在 include_usage 下追加的只含 usage 的帧，choices 为空"""
    return {
        "choices": [],
        "usage": {
            "prompt_tokens": inp,
            "completion_tokens": out,
            "total_tokens": inp + out,
        },
    }


def _item() -> RetrievedNews:
    return RetrievedNews(
        news_id=1,
        title="测试新闻",
        excerpt="摘要",
        category_id=1,
        publish_time=datetime.now(),
    )


class FakeStream:
    """模拟 httpx 的 client.stream() 上下文"""

    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def status_code(self):
        return 200

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class FakeClient:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, *args, **kwargs):
        return FakeStream(self._lines)


def _patch(monkeypatch, lines, recorder):
    """把 httpx.AsyncClient 换成桩，并把计量结果记到 recorder"""
    import ai.news_qa as news_qa

    async def fake_record(usage):
        recorder.append(usage)

    monkeypatch.setattr(news_qa, "_record_chat_usage", fake_record)
    monkeypatch.setattr(news_qa.httpx, "AsyncClient", lambda **kw: FakeClient(lines))


async def _collect(monkeypatch, lines, question="问题", confident=True):
    recorder: list = []
    _patch(monkeypatch, lines, recorder)

    from ai.news_qa import _stream_answer

    events = []
    async for chunk in _stream_answer(question, [_item()], confident):
        events.append(chunk)
    return events, recorder


@pytest.mark.asyncio
async def test_usage_frame_after_finish_reason_is_captured(monkeypatch):
    """
    核心回归：usage 帧排在 finish_reason 之后时也必须被读到。

    帧顺序照抄 DashScope 的真实行为：
        token -> finish_reason(usage=null) -> usage 帧(choices=[]) -> [DONE]
    """
    lines = [
        _sse_line(_token("你好")),
        _sse_line(_finish()),
        _sse_line(_usage_frame(700, 40)),
        "data: [DONE]",
    ]
    events, recorder = await _collect(monkeypatch, lines)

    assert recorder, "usage 帧没被读到，流式请求不会被计量"
    assert recorder[0]["prompt_tokens"] == 700
    assert recorder[0]["completion_tokens"] == 40
    assert sum("event: done" in c for c in events) == 1


@pytest.mark.asyncio
async def test_done_event_is_emitted_exactly_once(monkeypatch):
    """
    done 只能发一次。

    修 usage 时删掉了提前的 return，结果末尾原有的兜底 yield 又执行了，
    导致每次正常请求都发两个 done。前端按 done 收尾会被触发两次。
    """
    lines = [
        _sse_line(_token("a")),
        _sse_line(_token("b")),
        _sse_line(_finish()),
        _sse_line(_usage_frame(10, 2)),
        "data: [DONE]",
    ]
    events, _ = await _collect(monkeypatch, lines)

    assert sum("event: done" in c for c in events) == 1
    assert sum("event: token" in c for c in events) == 2


@pytest.mark.asyncio
async def test_missing_usage_is_still_reported_as_done(monkeypatch):
    """上游没给 usage 时也要正常收尾，且不伪造数据"""
    lines = [
        _sse_line(_token("x")),
        _sse_line(_finish()),
        "data: [DONE]",
    ]
    events, recorder = await _collect(monkeypatch, lines)

    assert not recorder, "没有 usage 就不能记账"
    done = [c for c in events if "event: done" in c]
    assert len(done) == 1
    assert '"usage": null' in done[0]


@pytest.mark.asyncio
async def test_refused_path_never_calls_llm(monkeypatch):
    """
    拒答时不能调大模型。
    防幻觉的关键：不给模型编造的机会，也省一次调用。
    """
    import ai.news_qa as news_qa

    def explode(**kwargs):
        raise AssertionError("拒答路径不该创建 HTTP 客户端")

    monkeypatch.setattr(news_qa.httpx, "AsyncClient", explode)

    events = []
    async for chunk in news_qa._stream_answer("问题", [_item()], False):
        events.append(chunk)

    assert sum("event: token" in c for c in events) == 1
    done = [c for c in events if "event: done" in c]
    assert len(done) == 1
    assert "refused" in done[0]
    assert "sources" in events[0], "溯源信息必须是第一个事件"


@pytest.mark.asyncio
async def test_stream_requests_include_usage_option(monkeypatch):
    """
    必须带 stream_options.include_usage。

    不带的话上游根本不会返回 usage，计量无从谈起 ——
    这是「流式请求不被计量」的直接原因。
    """
    import ai.news_qa as news_qa

    captured = {}

    class RecordingClient(FakeClient):
        def stream(self, method, url, headers=None, json=None, **kw):
            captured.update(json or {})
            return super().stream(method, url)

    monkeypatch.setattr(news_qa, "_record_chat_usage", lambda u: None)
    monkeypatch.setattr(
        news_qa.httpx,
        "AsyncClient",
        lambda **kw: RecordingClient([_sse_line(_finish()), "data: [DONE]"]),
    )

    async for _ in news_qa._stream_answer("问题", [_item()], True):
        pass

    assert captured.get("stream") is True
    assert captured.get("stream_options") == {"include_usage": True}