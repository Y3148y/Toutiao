from fastapi import FastAPI
# from routers.news import router as news_router
from routers import news, favorite, history, users
from fastapi.middleware.cors import CORSMiddleware

from routers.users import register
from utils.exception_handlers import register_exception_handler

app = FastAPI()

# 异常处理器
register_exception_handler(app)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 设置允许的源，开发环境下设置为["*"]允许所有源，生产环境需要指定源
    allow_credentials=True, # 允许携带cookie
    allow_methods=["*"], # 允许的请求方法
    allow_headers=["*"], # 允许的请求头
)


# app.include_router(news_router)
app.include_router(news.router)
app.include_router(favorite.router)
app.include_router(history.router)
app.include_router(users.router)

@app.get("/")
def read_root():
    return {"Hello": "welcome toutiao_backend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3001, reload=True)