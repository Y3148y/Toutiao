"""
日志配置。

要点：
- request_id 通过 contextvars 注入到每条日志，用户反馈「某次请求有问题」时
  可以直接用响应头里的 X-Request-ID 在日志里grep 出整条链路。
- LOG_LEVEL / LOG_FILE 都从环境变量读取，生产环境把 LOG_LEVEL 调成 INFO 或
  更高即可自动丢掉 DEBUG 噪声，无需改代码。
"""
import logging
import os
import sys
from contextvars import ContextVar

# 当前请求的唯一标识，由 RequestIdMiddleware 写入。
# 用 contextvars 而不是 threading.local：asyncio 下所有协程跑在同一线程，
# threading.local 无法区分不同请求；contextvars 每次请求（每个 Task）有独立副本。
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# 方便调用方取当前请求 ID
def get_request_id() -> str:
    return request_id_var.get()


class RequestIdFilter(logging.Filter):
    """把 contextvars 里的 request_id 塞进每条 LogRecord"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def setup_logging() -> None:
    """配置根 logger，重复调用不会叠加 handler"""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # 已经初始化过就不再重复添加
    if getattr(root, "_toutiao_configured", False):
        return

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | req=%(request_id)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    request_id_filter = RequestIdFilter()

    # uvicorn 会自带 access 日志 handler，保留但去掉它的重复输出
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.addFilter(request_id_filter)

    handlers = [console]

    # 可选：同时写入文件（带轮转，避免日志撑爆磁盘）
    log_file = os.getenv("LOG_FILE")
    if log_file:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024))),
            backupCount=int(os.getenv("LOG_BACKUP_COUNT", "5")),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(request_id_filter)
        handlers.append(file_handler)

    for handler in handlers:
        root.addHandler(handler)

    # uvicorn 自带的 handler 不带我们的格式，关掉避免日志格式混杂
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    root._toutiao_configured = True


def get_logger(name: str) -> logging.Logger:
    """业务代码统一用这个拿logger"""
    return logging.getLogger(name)