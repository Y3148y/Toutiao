# 整合工具，根据token查用户
from http.client import HTTPException

from fastapi import Header, Depends, status
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

