from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


def success_response(message: str = "success", data=None, headers: dict | None = None):
    """
    统一响应体：{code, message, data}。

    headers 用来回传会话标识（匿名用户的 X-Anonymous-Id），
    让调用方下次带上就能接上多轮上下文。默认 None 即不改动原有行为。
    """
    content = {
        "code": 200,
        "message": message,
        "data": data,
    }
    # 把任何 FastAPI、Pydantic、ORM 对象都正常序列化
    return JSONResponse(content=jsonable_encoder(content), headers=headers)
