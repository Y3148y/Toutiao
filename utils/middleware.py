"""
自定义中间件。

RequestIdMiddleware 为每个请求生成唯一 ID，并通过响应头返回给前端。
用途：用户报障时说「我刷新了一下就报错了」，这时让他把X-Request-ID 发过来，
就能在日志里 grep 出那一次请求的全部日志（含耗时、状态码、异常栈）。

为什么用 contextvars 传递：
asyncio 是单线程协作式调度，多个请求的协程交替跑在同一线程里。用普通的
局部变量传request_id 只有当前函数栈能看到；用threading.local 也不行，
因为线程只有一个。contextvars 是per-Task 存储，asyncio 每创建一个 Task
（每个请求就是一个 Task）都会复制一份上下文，所以各请求互不干扰，
又能在任意深的调用栈里通过 get_request_id() 取到。
"""
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from utils.logging_conf import get_logger, request_id_var

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """生成请求 ID、记录访问日志、统计耗时"""

    # 慢请求阈值（秒），超过则打 WARNING 便于发现性能问题
    SLOW_REQUEST_SECONDS = 1.0

    async def dispatch(self, request: Request, call_next) -> Response:
        # 优先复用上游（如网关/nginx）已经生成的 ID，没有再自己生成
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:8]

        # token 用于请求结束时恢复默认值，避免污染后续请求的上下文
        token = request_id_var.set(request_id)

        started_at = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # 这里不吞异常：交给全局异常处理器统一转成统一响应体。
            # 只补一条 ERROR 日志，方便直接拿到耗时和 request_id。
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.exception(
                "%s %s 未处理异常，耗时 %.1fms", request.method, request.url.path, duration_ms
            )
            raise
        else:
            duration_ms = (time.perf_counter() - started_at) * 1000
            # /docs /openapi.json 是文档访问，不入业务日志，避免刷屏
            if not request.url.path.startswith(("/docs", "/redoc", "/openapi.json")):
                log = logger.warning if duration_ms >= self.SLOW_REQUEST_SECONDS * 1000 else logger.info
                log(
                    "%s %s -> %s，耗时 %.1fms",
                    request.method,
                    request.url.path,
                    response.status_code,
                    duration_ms,
                )
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            # 无论成功失败都要还原，否则请求复用协程时会带上上一次的 ID
            request_id_var.reset(token)