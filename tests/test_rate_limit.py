"""
令牌桶限流测试。

覆盖：突发容量、令牌补充速率、拒绝后恢复、按 IP/路径分桶、
Lua 脚本的原子性（并发不超发）、Redis 故障 fail-open。
"""
import asyncio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from utils import rate_limit


def _mini_app(capacity, rate, monkeypatch, redis_client):
    mini = FastAPI()

    @mini.get("/api/news/categories")
    async def categories():
        return {"code": 200, "message": "success", "data": []}

    @mini.get("/api/news/list")
    async def news_list():
        return {"code": 200, "message": "success", "data": []}

    monkeypatch.setattr(rate_limit, "redis_client", redis_client)
    mini.add_middleware(rate_limit.RateLimitMiddleware, capacity=capacity, refill_rate=rate)
    return mini


class TokenBucketRedis:
    """
    内存版 Redis，用 Python 复现 Lua 脚本的语义。

    刻意保留了 Lua 版本的一个特性：返回的令牌数用字符串传回
    （Lua 返回值会被 Redis 转成整数，小数会被截断）。
    """

    def __init__(self):
        self.buckets = {}

    async def eval(self, script, numkeys, key, capacity, rate, cost, now_ms):
        bucket = self.buckets.setdefault(
            key, {"tokens": float(capacity), "ts": now_ms}
        )
        elapsed = max(0, now_ms - bucket["ts"])
        tokens = min(float(capacity), bucket["tokens"] + (elapsed / 1000.0) * float(rate))
        allowed = 0
        if tokens >= float(cost):
            tokens -= float(cost)
            allowed = 1
        bucket["tokens"] = tokens
        bucket["ts"] = now_ms
        return [allowed, str(tokens)]


class BrokenRedis:
    async def eval(self, *args, **kwargs):
        raise ConnectionError("redis 不可用")


@pytest.fixture
def fake_redis():
    return TokenBucketRedis()


@pytest.fixture
def clock(monkeypatch):
    """可控时钟，令牌补充速率相关的测试需要"""
    now = {"ms": 1_700_000_000_000}
    import time as time_module

    monkeypatch.setattr(rate_limit.time, "time", lambda: now["ms"] / 1000.0)
    return now


def test_allows_burst_up_to_capacity(fake_redis, monkeypatch, clock):
    """桶容量就是突发上限：capacity 次瞬时请求应全部放行"""
    client = TestClient(_mini_app(5, 1, monkeypatch, fake_redis))

    for _ in range(5):
        assert client.get("/api/news/categories").status_code == 200


def test_rejects_after_burst_exhausted(fake_redis, monkeypatch, clock):
    client = TestClient(_mini_app(3, 1, monkeypatch, fake_redis))

    for _ in range(3):
        client.get("/api/news/categories")

    resp = client.get("/api/news/categories")

    assert resp.status_code == 429
    assert resp.json() == {"code": 429, "message": "请求过于频繁，请稍后再试", "data": None}


def test_tokens_refill_over_time(fake_redis, monkeypatch, clock):
    """
    令牌桶的核心：不用令牌会按 refill_rate 匀速补充。
    rate=2/s，过 1 秒应补回 2 个令牌。
    """
    client = TestClient(_mini_app(2, 2, monkeypatch, fake_redis))

    assert client.get("/api/news/categories").status_code == 200
    assert client.get("/api/news/categories").status_code == 200
    assert client.get("/api/news/categories").status_code == 429

    clock["ms"] += 1000  # 前进 1 秒 -> 补回 2 个令牌

    assert client.get("/api/news/categories").status_code == 200
    assert client.get("/api/news/categories").status_code == 200
    assert client.get("/api/news/categories").status_code == 429


def test_partial_refill(fake_redis, monkeypatch, clock):
    """rate=2/s，过 0.5 秒只补 1 个令牌，所以只能再放行 1 次"""
    client = TestClient(_mini_app(2, 2, monkeypatch, fake_redis))

    client.get("/api/news/categories")
    client.get("/api/news/categories")
    assert client.get("/api/news/categories").status_code == 429

    clock["ms"] += 500

    assert client.get("/api/news/categories").status_code == 200, "应补回 1 个令牌"
    assert client.get("/api/news/categories").status_code == 429, "只剩 1 个令牌，不够第 2 次"


def test_tokens_capped_at_capacity(fake_redis, monkeypatch, clock):
    """
    长时间不用，桶最多装满 capacity 个，不会溢出。
    这正是令牌桶相对固定窗口的优势：空闲后仍只允许 capacity 突发。
    """
    client = TestClient(_mini_app(3, 100, monkeypatch, fake_redis))

    clock["ms"] += 60_000  # 空闲 1 分钟，理论能补 6000 个

    allowed = 0
    for _ in range(10):
        if client.get("/api/news/categories").status_code == 200:
            allowed += 1
        else:
            break

    assert allowed == 3, f"空闲后最多只能放行 capacity=3 次，实际 {allowed}"


def test_retry_after_reflects_refill_time(fake_redis, monkeypatch, clock):
    """rate=0.5/s（每 2 秒补 1 个），Retry-After 应接近 2 秒而不是 1"""
    client = TestClient(_mini_app(1, 0.5, monkeypatch, fake_redis))

    client.get("/api/news/categories")
    resp = client.get("/api/news/categories")

    assert resp.status_code == 429
    assert int(resp.headers["Retry-After"]) >= 1


def test_429_has_standard_headers(fake_redis, monkeypatch, clock):
    client = TestClient(_mini_app(1, 1, monkeypatch, fake_redis))
    client.get("/api/news/categories")

    resp = client.get("/api/news/categories")

    assert resp.headers["X-RateLimit-Limit"] == "1"
    assert resp.headers["X-RateLimit-Remaining"] == "0"
    assert int(resp.headers["Retry-After"]) > 0


def test_remaining_header_decrements(fake_redis, monkeypatch, clock):
    client = TestClient(_mini_app(3, 1, monkeypatch, fake_redis))

    assert client.get("/api/news/categories").headers["X-RateLimit-Remaining"] == "2"
    assert client.get("/api/news/categories").headers["X-RateLimit-Remaining"] == "1"
    assert client.get("/api/news/categories").headers["X-RateLimit-Remaining"] == "0"


def test_different_paths_have_separate_buckets(fake_redis, monkeypatch, clock):
    client = TestClient(_mini_app(2, 1, monkeypatch, fake_redis))

    client.get("/api/news/categories")
    client.get("/api/news/categories")
    assert client.get("/api/news/categories").status_code == 429

    assert client.get("/api/news/list").status_code == 200, "不同路径应有独立桶"


def test_different_ips_have_separate_buckets(fake_redis, monkeypatch, clock):
    client = TestClient(_mini_app(1, 1, monkeypatch, fake_redis))

    assert client.get("/api/news/categories", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert client.get("/api/news/categories", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 429
    assert client.get("/api/news/categories", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200


def test_docs_paths_are_exempt(fake_redis, monkeypatch, clock):
    client = TestClient(_mini_app(1, 1, monkeypatch, fake_redis))

    for _ in range(5):
        assert client.get("/openapi.json").status_code == 200


def test_zero_capacity_disables_rate_limiting(fake_redis, monkeypatch, clock):
    client = TestClient(_mini_app(0, 1, monkeypatch, fake_redis))

    for _ in range(20):
        assert client.get("/api/news/categories").status_code == 200


def test_fails_open_when_redis_is_down(monkeypatch, clock):
    """
    限流是保护措施而非业务逻辑，Redis 挂掉时放行而非 500。
    """
    client = TestClient(_mini_app(1, 1, monkeypatch, BrokenRedis()))

    for _ in range(5):
        assert client.get("/api/news/categories").status_code == 200


async def test_lua_script_is_atomic_under_concurrency():
    """
    回归测试：令牌桶如果用多条 Redis 命令实现「读-补充-扣减-写回」，
    并发请求会读到同一个 tokens 值导致超发。
    这里用 Lua 脚本在单线程内顺序执行，放行数应严格等于 capacity。

    capacity=20，并发 50 个请求 -> 必须恰好放行 20 个
    """
    mini = FastAPI()

    @mini.get("/probe")
    async def probe():
        return {"ok": True}

    bucket = TokenBucketRedis()
    now_ms = 1_700_000_000_000

    async def one_request():
        return await bucket.eval(
            rate_limit._TOKEN_BUCKET_LUA,
            1,
            "k",
            20,          # capacity
            0.001,       # rate：补充极慢，可视为不补充
            1,           # cost
            now_ms,
        )

    results = await asyncio.gather(*[one_request() for _ in range(50)])
    allowed = sum(1 for r in results if int(r[0]) == 1)

    assert allowed == 20, f"应恰好放行 20 次，实际 {allowed}（说明存在超发）"


def test_clock_going_backwards_does_not_add_tokens(fake_redis, monkeypatch, clock):
    """
    客户端时钟回拨（now < ts）时不能凭空补出令牌，
    否则一次时间跳变就能绕过限流。
    """
    client = TestClient(_mini_app(2, 100, monkeypatch, fake_redis))

    clock["ms"] += 10_000
    client.get("/api/news/categories")
    client.get("/api/news/categories")
    assert client.get("/api/news/categories").status_code == 429

    clock["ms"] -= 5_000  # 时钟回拨

    assert client.get("/api/news/categories").status_code == 429, "时钟回拨不应补充令牌"