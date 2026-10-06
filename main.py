import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

from ai import ai_chat, news_qa
from routers import favorite, history, news, users
from utils.exception_handlers import register_exception_handler
from utils.logging_conf import get_logger, setup_logging
from utils.middleware import RequestIdMiddleware
from utils.rate_limit import RateLimitMiddleware

# 日志要在其他模块之前初始化
setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Toutiao Backend",
    description="新闻资讯后端服务",
    version="1.0.0",
)

# 全局异常处理器
register_exception_handler(app)

# CORS：开发环境放开所有来源，生产环境务必用 CORS_ORIGINS 指定具体域名
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Request-ID",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "Retry-After",
    ],
)

# 限流放最外层，被限流的请求就不必再走日志、压缩等后续处理。
# 注意 add_middleware 是「后添加的在外层」，所以限流写在最后
app.add_middleware(RequestIdMiddleware)  # 请求 ID + 访问日志 + 耗时统计
app.add_middleware(GZipMiddleware)  # 响应压缩
# 令牌桶限流：capacity 是突发上限，rate 是每秒补充速率
app.add_middleware(
    RateLimitMiddleware,
    capacity=int(os.getenv("RATE_LIMIT_CAPACITY", "100")),
    refill_rate=float(os.getenv("RATE_LIMIT_RATE", "100")),
)


app.include_router(news.router)
app.include_router(favorite.router)
app.include_router(history.router)
app.include_router(users.router)
app.include_router(ai_chat.router)
app.include_router(news_qa.router)


@app.get("/")
def read_root():
    return {"Hello": "welcome toutiao_backend"}


@app.on_event("startup")
async def on_startup():
    logger.info("服务启动完成")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "3001")),
        reload=os.getenv("RELOAD", "true").lower() == "true",
    )