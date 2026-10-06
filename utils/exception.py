"""
全局异常处理器。

四个 handler 对应 utils/exception_handlers.py 里注册的异常类型，
顺序按「子类在前、父类在后」：否则 IntegrityError 会被 SQLAlchemyError
抢先接管，友好提示就丢了。

DEBUG_MODE 从环境变量读取（默认关闭）：
以前这里硬编码 True，等于生产环境也会把异常堆栈返回给前端，属于信息泄露。
"""
import os
import traceback

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from starlette.requests import Request
from starlette.responses import JSONResponse

from utils.logging_conf import get_logger

logger = get_logger(__name__)

# 开发模式：返回详细错误信息与堆栈；生产模式：只返回简化信息
# 默认 False —— 生产环境必须为 False
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


def _build_error_data(request: Request, exc: Exception, include_traceback: bool = True) -> dict | None:
    """开发模式下构造详细错误信息，生产模式返回 None"""
    if not DEBUG_MODE:
        return None

    data = {
        "error_type": type(exc).__name__,
        "error_detail": str(exc),
        "path": str(request.url),
        "request_id": request.headers.get("X-Request-ID", "-"),
    }
    if include_traceback:
        data["traceback"] = traceback.format_exc()
    return data


async def http_exception_handler(request: Request, exc: HTTPException):
    logger.info("HTTP %s %s -> %s", request.method, request.url.path, exc.status_code)

    error_data = None
    if DEBUG_MODE:
        error_data = {
            "path": str(request.url),
            "request_id": request.headers.get("X-Request-ID", "-"),
        }

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.status_code,
            "message": exc.detail,
            "data": error_data,
            "request_id": request.headers.get("X-Request-ID", "-"),
        }
    )


async def integrity_error_handler(request: Request, exc: IntegrityError):
    """
    数据库完整性约束：把 MySQL 的英文报错翻译成用户看得懂的中文提示
    """
    error_msg = str(exc.orig) if exc.orig is not None else str(exc)

    if "username_UNIQUE" in error_msg or "Duplicate entry" in error_msg:
        detail = "用户名已存在"
    elif "FOREIGN KEY" in error_msg:
        detail = "关联的数据不存在"
    else:
        detail = "数据约束冲突，请检查输入"

    logger.warning("数据完整性冲突: %s", detail)

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "code": 400,
            "message": detail,
            "data": _build_error_data(request, exc, include_traceback=False),
            "request_id": request.headers.get("X-Request-ID", "-"),
        }
    )


async def sqlalchemy_error_handler(request: Request, exc: SQLAlchemyError):
    """
    SQLAlchemy 数据库错误
    """
    logger.error("数据库操作失败", exc_info=exc)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "code": 500,
            "message": "数据库操作失败，请稍后重试",
            "data": _build_error_data(request, exc),
            "request_id": request.headers.get("X-Request-ID", "-"),
        }
    )


async def general_error_handler(request: Request, exc: Exception):
    """
    所有未捕获异常的兜底处理，保证任何情况下都返回统一响应格式
    """
    logger.error("未捕获异常: %s", exc, exc_info=exc)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "code": 500,
            "message": "服务器内部错误",
            "data": _build_error_data(request, exc),
            "request_id": request.headers.get("X-Request-ID", "-"),
        }
    )