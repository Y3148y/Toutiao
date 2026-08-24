import uuid
from datetime import timedelta, datetime
from http.client import HTTPException

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from models.users import User, UserToken
from schemas.users import UserRequest, UserUpdateRequest
from utils import security


# 根据用户名查询用户
async def get_user_by_username(db: AsyncSession, username: str):
    query = select(User).where(User.username == username)
    result = await db.execute(query)
    return result.scalar_one_or_none()

# 创建用户
async def create_user(db: AsyncSession, user_data: UserRequest):
    # 密码加密
    hashed_password = security.get_hash_password(user_data.password)
    user = User(username=user_data.username, password=hashed_password)
    db.add(user)
    await db.commit()
    await db.refresh(user) # 从数据库读回最新的 user
    return user

# 生成token
async def create_token(db: AsyncSession, user_id: int):
    token = str(uuid.uuid4())
    expires_at = datetime.now() + timedelta(days=7)
    query = select(UserToken).where(UserToken.user_id == user_id)
    res = await db.execute(query)
    user_token = res.scalar_one_or_none()
    if user_token:
        user_token.token = token
        user_token.expires_at = expires_at
    else:
        user_token = UserToken(user_id=user_id, token=token, expires_at=expires_at)
        db.add(user_token)
        await db.commit()

    return token

# 验证用户
async def authenticate_user(db: AsyncSession, username: str, password: str):
    user = await get_user_by_username(db, username)
    if not user:
        return None
    if not security.verify_password(password, user.password):
        return None

    return user

# 根据token查询用户：验证token->查询用户
async def get_user_by_token(db: AsyncSession, token: str):
    query = select(UserToken).where(UserToken.token == token)
    result = await db.execute(query)
    db_token = result.scalar_one_or_none()

    if not db_token or db_token.expires_at < datetime.now():
        return None

    query = select(User).where(User.id == db_token.user_id)
    result = await db.execute(query)
    return result.scalar_one_or_none()


# 更新用户信息
async def update_user(db: AsyncSession, username: str, user_data: UserUpdateRequest):
    query = update(User).where(User.username == username).values(**user_data.model_dump(exclude_unset=True, exclude_none=True))
    result = await db.execute(query)
    await db.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="用户不存在")

    #
    update_user = await get_user_by_username(db, username)
    return update_user

# 更新密码
async def update_password(db: AsyncSession, user: User, old_password: str, new_password: str):
    if not security.verify_password(old_password, user.password):
        return False
    new_hash_password = security.get_hash_password(new_password)
    user.password = new_hash_password
    # user 是从数据库查询出来的 ORM 模型实例对象，已经在 session 托管中
    # 更新：由SQLAlchemy真正接管这个 User 对象，确保可以commit
    # 规避 session 过期或关闭导致的不能提交问题
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return True

