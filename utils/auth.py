# 整合工具，根据token查用户
from fastapi import Header, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from crud import users

from config.db_conf import get_db

# Header()：从HTTP 请求头拿数据
async def get_current_user(
        authorization: str = Header(..., alias="Authorization"),
        db: AsyncSession = Depends(get_db)
):
    # token = authorization.strip(" ")[1]
    token = authorization.replace("Bearer ", "")
    user = await users.get_user_by_token(db, token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的令牌或者已过期")
    return user


async def get_optional_user(
        authorization: str | None = Header(None, alias="Authorization"),
        db: AsyncSession = Depends(get_db)
):
    """
    可选鉴权：有合法 token 就返回用户，没有或无效就返回 None。

    给「登录后体验更好、但不给登录也能用」的接口用，比如首页信息流：
    未登录返回时间序，登录了返回个性化排序。

    **这里必须吞掉 token 解析异常而不是让它冒泡。** 前端带着一个过期的
    token 打开首页是很常见的（token 过期但客户端还没清理），
    此时报 401 会让首页直接白屏，用户只能手动清缓存或重新登录。
    """
    if not authorization:
        return None
    token = authorization.replace("Bearer ", "").strip()
    if not token:
        return None
    try:
        return await users.get_user_by_token(db, token)
    except Exception:
        # 库连接异常等不该被当成「token 无效」吞掉，但也不该让首页 500。
        # 记一条 warning 便于排查，然后按未登录处理。
        import logging

        logging.getLogger(__name__).warning(
            "可选鉴权失败，按未登录处理", exc_info=True
        )
        return None

