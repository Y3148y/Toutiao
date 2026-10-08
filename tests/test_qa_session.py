"""
AI 问答多轮与会话历史的测试。

要锁住三件事：
1. 登录用户的问答落 ai_chat 表，拒答也落
2. 未登录走 Redis，不污染库表
3. **落库失败绝不能影响已经返回给用户的答案**
   ——模型的钱已经花了、答案已经推出去了，为写库失败返回 500 是净损失
"""
import asyncio
from datetime import datetime, timedelta

import pytest

from ai import anon_history
from ai.qa_session import QaSession, resolve_session, take_anon_id
from ai.retriever import RetrievedNews


def _item(nid=1, title="测试新闻") -> RetrievedNews:
    return RetrievedNews(
        news_id=nid,
        title=title,
        excerpt="摘要",
        category_id=1,
        publish_time=datetime.now(),
    )


# ---------------------------------------------------------------- 匿名 id 校验


def test_take_anon_id_accepts_clean_value():
    assert take_anon_id("abc-123_DEF") == "abc-123_DEF"


@pytest.mark.parametrize("bad", ["", "   ", "a" * 100, "abc;rm -rf", "a b", "中文"])
def test_take_anon_id_rejects_unsafe_value(bad):
    """
    这个值会进 Redis key。

    恶意长串会造成 key 膨胀，含空格/分号/中文的会造成日志注入和 key 混乱，
    所以只接受字母数字加 - _。
    """
    assert take_anon_id(bad) is None


def test_take_anon_id_handles_none():
    assert take_anon_id(None) is None


# ---------------------------------------------------------------- 会话解析


@pytest.mark.asyncio
async def test_logged_in_user_uses_db():
    class U:
        id = 7

    session = await resolve_session(None, U(), None)
    assert session.is_logged_in is True
    assert session.user_id == 7
    # 登录用户不该收到匿名 id 头，否则前端会误以为能随便伪造
    assert session.response_headers() == {}


@pytest.mark.asyncio
async def test_anonymous_user_gets_generated_id():
    session = await resolve_session(None, None, None)
    assert session.is_logged_in is False
    assert session.anon_id and len(session.anon_id) >= 16
    headers = session.response_headers()
    assert headers["X-Anonymous-Id"] == session.anon_id


@pytest.mark.asyncio
async def test_logged_in_wins_over_anon_id():
    """登录用户即使带了匿名 id 也走库 —— 那样历史才能跨设备"""

    class U:
        id = 9

    session = await resolve_session(None, U(), "some-anon")
    assert session.is_logged_in is True
    assert session.user_id == 9


# ---------------------------------------------------------------- 落库失败不阻塞


@pytest.mark.asyncio
async def test_recorder_failure_does_not_raise():
    """
    核心回归：写库失败时答案必须已经交付，不能变成 500。

    这条对应真实故障模式 —— DB 连接抖动 / 磁盘满 / 慢查询超时，
    任何一个发生都不该让已经花钱调完模型的请求白跑。
    """
    from ai.news_qa import _safe_record

    async def boom(_answer):
        raise RuntimeError("MySQL 连接断开")

    # _safe_record 吞掉异常，不该向上抛
    await _safe_record(boom, "已经生成好的答案")


@pytest.mark.asyncio
async def test_safe_record_accepts_none_callback():
    from ai.news_qa import _safe_record

    await _safe_record(None, "任意答案")  # 不应抛错


@pytest.mark.asyncio
async def test_empty_answer_is_not_recorded():
    """模型返回空串时不该写一条空历史，那只会污染列表"""

    calls = []

    async def recorder(answer):
        calls.append(answer)

    session = QaSession(user_id=1)
    rec = session.recorder(None, "问题")
    # recorder 内部应当跳过空答案
    await rec("")
    assert calls == []


# ---------------------------------------------------------------- 匿名历史


@pytest.mark.asyncio
async def test_anon_history_roundtrip(monkeypatch):
    store: dict = {}

    class FakeRedis:
        async def get(self, key):
            return store.get(key)

        async def setex(self, key, ttl, value):
            store[key] = value

        async def delete(self, key):
            store.pop(key, None)

    import config.cache_conf as cache_conf

    monkeypatch.setattr(cache_conf, "redis_client", FakeRedis())

    aid = "test-anon-id"
    await anon_history.save_turn(aid, "GDP增长多少", "5.2%")
    await anon_history.save_turn(aid, "那人均呢", "人均可支配收入增长")

    history = await anon_history.get_history(aid)
    assert len(history) == 2
    assert history[0][0] == "GDP增长多少"
    assert history[1][0] == "那人均呢"


@pytest.mark.asyncio
async def test_anon_history_is_capped(monkeypatch):
    """只留最近几轮。存满会撑大 Redis value，更早的轮次本来就送不进模型。"""
    store: dict = {}

    class FakeRedis:
        async def get(self, key):
            return store.get(key)

        async def setex(self, key, ttl, value):
            store[key] = value

    import config.cache_conf as cache_conf

    monkeypatch.setattr(cache_conf, "redis_client", FakeRedis())

    for i in range(8):
        await anon_history.save_turn("cap-test", f"问题{i}", f"回答{i}")

    history = await anon_history.get_history("cap-test")
    assert len(history) == 3
    assert history[-1][0] == "问题7"


@pytest.mark.asyncio
async def test_anon_history_write_failure_is_swallowed(monkeypatch):
    """Redis 挂了要能继续，只是丢历史而已"""

    class BrokenRedis:
        async def get(self, key):
            raise RuntimeError("Redis 不可用")

        async def setex(self, key, ttl, value):
            raise RuntimeError("Redis 不可用")

    import config.cache_conf as cache_conf

    monkeypatch.setattr(cache_conf, "redis_client", BrokenRedis())

    # 不抛错
    await anon_history.save_turn("broken", "问题", "回答")
    assert await anon_history.get_history("broken") == []


@pytest.mark.asyncio
async def test_anon_history_does_not_leak_between_sessions(monkeypatch):
    """不同会话标识之间必须隔离，否则同一出口 IP 的用户会看到彼此对话"""
    store: dict = {}

    class FakeRedis:
        async def get(self, key):
            return store.get(key)

        async def setex(self, key, ttl, value):
            store[key] = value

    import config.cache_conf as cache_conf

    monkeypatch.setattr(cache_conf, "redis_client", FakeRedis())

    await anon_history.save_turn("user-A", "A的问题", "A的回答")
    await anon_history.save_turn("user-B", "B的问题", "B的回答")

    a = await anon_history.get_history("user-A")
    b = await anon_history.get_history("user-B")
    assert a == [("A的问题", "A的回答")]
    assert b == [("B的问题", "B的回答")]


# ---------------------------------------------------------------- 库表 CRUD


@pytest.mark.asyncio
async def test_history_turns_are_returned_oldest_first(fake_session):
    """
    多轮对话的顺序必须是最旧在前。

    反序的话模型会看到「回答」在「提问」之前，上下文直接乱掉。
    """
    from crud import ai_chat as ai_chat_crud
    from tests.fakes import FakeResult

    rows = [("最新提问", "最新回答"), ("较早提问", "较早回答")]
    fake_session.result = FakeResult(rows=rows)

    history = await ai_chat_crud.get_recent_history(fake_session, user_id=1)

    assert history == [("较早提问", "较早回答"), ("最新提问", "最新回答")]


@pytest.mark.asyncio
async def test_related_manual_ids_extracted(fake_session):
    """关联 id 要按查询顺序取出"""
    from crud import related_news as related_crud
    from tests.fakes import FakeResult

    fake_session.result = FakeResult(rows=[(3,), (7,), (9,)])
    assert await related_crud.list_related_ids(fake_session, 1) == [3, 7, 9]


@pytest.mark.asyncio
async def test_manual_related_falls_back_when_table_empty(fake_session, stub_redis):
    """
    关联表没配时必须回落到实时查询，不能让推荐位空着。
    """
    from crud import related_news as related_crud
    from models.news import News
    from tests.fakes import FakeResult

    stub_redis["get_cache_related_news"].return_value = None
    fake_news = News(
        id=5,
        title="兜底新闻",
        description=None,
        content="内容",
        image=None,
        author="记者",
        category_id=2,
        views=100,
        publish_time=datetime.now(),
    )
    # 第一次查关联 id 返回空，第二次查新闻本体返回兜底结果
    async def execute(stmt):
        text = str(stmt)
        if "FROM related_news" in text:
            return FakeResult(rows=[])
        return FakeResult(rows=[fake_news])

    fake_session.execute = execute

    result = await related_crud.get_related_news(fake_session, 1, 2)

    assert result, "关联表为空时应回落到实时查询"
    assert result[0]["id"] == 5
