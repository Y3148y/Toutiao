"""
接口限流：Redis 令牌桶（Token Bucket）。

算法
----
桶里存两样东西：当前令牌数 tokens、上次更新时间 ts。请求进来时：
    1. 按流逝时间补充令牌：tokens += (now - ts) * refill_rate
    2. tokens 封顶为桶容量 capacity
    3. 若 tokens >= 1，扣 1 并放行；否则返回 429

为什么是令牌桶而不是固定窗口
--------------------------
固定窗口把时间切成互不相干的段，窗口交界处可以打满两次：
1 秒内最多能放 2N 个请求。令牌桶是「按速率匀速放行 + 允许有限突发」，
既限制长期平均速率，又允许短时突发，更符合 API 网关的实际需求。

突发能力的量化：桶容量 capacity 就是突发上限。
capacity=100、refill_rate=100/s 表示任意时刻最多攒 100 个令牌
（即允许 100 次瞬时突发），之后严格按每秒 100 个的速度补充。

为什么必须用 Lua 脚本
------------------
令牌桶是「读-补充-扣减-写回」四步。如果拆成多条 Redis 命令（MGET / HSET ...），
两个并发请求可能读到同一个 tokens=0，然后都通过校验，导致实际放行量翻倍。
Redis 的 Lua 脚本在执行期间是原子的，这一步由服务端保证，不需要分布式锁。

时间源说明
---------
时间戳由客户端传入而不是在脚本里调TIME：
- 好处：避免 Redis 主从切换时 TIME 返回值跳变导致计数异常
- 代价：依赖各机器时钟大致同步（NTP）。对本项目够用；
  如果要严格保证，应改用 Redis TIME 或接入统一时钟服务。
"""
import os
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from config.cache_conf import redis_client
from utils.logging_conf import get_logger

logger = get_logger(__name__)

# 桶容量：允许的瞬时突发上限
DEFAULT_CAPACITY = int(os.getenv("RATE_LIMIT_CAPACITY", "100"))
# 补充速率：每秒补多少个令牌（长期平均速率上限）
DEFAULT_REFILL_RATE = float(os.getenv("RATE_LIMIT_RATE", "100"))

# 文档与健康检查不限流，避免调试时被自己的请求挡在门外
EXCLUDED_PREFIXES = ("/docs", "/redoc", "/openapi.json")

# 令牌桶 Lua 脚本。返回 {是否放行, 剩余令牌数}
# 注意：Redis 会把 Lua 的返回值转成整数，所以令牌数用 tostring 转字符串传回，
# 否则小数部分会被截断（表现为限流比配置的更严）。
_TOKEN_BUCKET_LUA = """
local key          = KEYS[1]
local capacity     = tonumber(ARGV[1])   -- 桶容量
local refill_rate  = tonumber(ARGV[2])   -- 每秒补充令牌数
local cost         = tonumber(ARGV[3])   -- 本次请求消耗
local now_ms       = tonumber(ARGV[4])   -- 客户端时间戳（毫秒）

local tokens = redis.call('HGET', key, 'tokens')
local ts     = redis.call('HGET', key, 'ts')

if tokens == false or ts == false then
    -- 桶不存在：初始装满
    tokens = capacity
    ts = now_ms
end

tokens = tonumber(tokens)
ts = tonumber(ts)

-- 按流逝的时间补充令牌，负数（时钟回拨）按 0 处理
local elapsed_ms = now_ms - ts
if elapsed_ms < 0 then elapsed_ms = 0 end
tokens = tokens + (elapsed_ms / 1000.0) * refill_rate
if tokens > capacity then tokens = capacity end

local allowed = 0
if tokens >= cost then
    tokens = tokens - cost
    allowed = 1
end

redis.call('HSET', key, 'tokens', tokens, 'ts', now_ms)

-- 桶空闲多久后自动清理：装满所需时间 + 1 秒余量
local ttl_ms = math.ceil((capacity / refill_rate) * 1000) + 1000
redis.call('PEXPIRE', key, ttl_ms)

return {allowed, tostring(tokens)}
"""


class RateLimitMiddleware(BaseHTTPMiddleware):
    """按「客户端 IP + 请求路径」做令牌桶限流"""

    def __init__(
        self,
        app,
        capacity: int | None = None,
        refill_rate: float | None = None,
        cost: float = 1.0,
    ):
        super().__init__(app)
        self.capacity = capacity if capacity is not None else DEFAULT_CAPACITY
        self.refill_rate = refill_rate if refill_rate is not None else DEFAULT_REFILL_RATE
        self.cost = cost

    def _get_client_ip(self, request: Request) -> str:
        """
        取真实客户端 IP。

        只有在可信反向代理后面才应该信任 X-Forwarded-For，
        否则客户端可以随便伪造这个头绕过限流。
        部署时需保证前置代理清洗这些头。
        """
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        if self.capacity <= 0 or request.url.path.startswith(EXCLUDED_PREFIXES):
            return await call_next(request)

        now_ms = int(time.time() * 1000)
        key = f"ratelimit:{request.url.path}:{self._get_client_ip(request)}"

        try:
            allowed, remaining = await redis_client.eval(
                _TOKEN_BUCKET_LUA,
                1,
                key,
                self.capacity,
                self.refill_rate,
                self.cost,
                now_ms,
            )
        except Exception as exc:
            # fail-open：限流是保护措施而非业务逻辑，
            # 挡不住流量时让请求继续走，比整个服务不可用更合理
            logger.warning("限流检查失败，本次放行: %s", exc)
            return await call_next(request)

        if int(allowed) == 0:
            # 算出还需等待多久才能再拿到一个令牌
            retry_after = int((self.cost - float(remaining)) / self.refill_rate) + 1
            logger.warning(
                "触发限流: %s %s client=%s",
                request.method, request.url.path, self._get_client_ip(request),
            )
            return JSONResponse(
                status_code=429,
                content={
                    "code": 429,
                    "message": "请求过于频繁，请稍后再试",
                    "data": None,
                },
                headers={
                    "Retry-After": str(max(1, retry_after)),
                    "X-RateLimit-Limit": str(self.capacity),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.capacity)
        response.headers["X-RateLimit-Remaining"] = str(int(float(remaining)))
        return response