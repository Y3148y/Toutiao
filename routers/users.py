from fastapi import APIRouter
from fastapi.params import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status
from crud import users
from config.db_conf import get_db
from models.users import User
from schemas.users import UserRequest, UserAuthResponse, UserInfoResponse, UserUpdateRequest, UserChangePasswordRequest
from utils.auth import get_current_user
from utils.response import success_response

router = APIRouter(prefix="/api/user", tags=["user"])


# post请求体参数user_data
@router.post("/register")
async def register(user_data: UserRequest, db: AsyncSession = Depends(get_db)):
    """
    检查用户是否存在，创建用户（密码加密处理passlib，添加数据），uuid生成临时Token
    """
    existing_user = await users.get_user_by_username(db, user_data.username)
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户已存在")
    user = await users.create_user(db, user_data)
    token = await users.create_token(db, user.id)
    # return {
    #     "code": 200,
    #     "msg": "注册成功",
    #     "data": {
    #         "token": token,
    #         "userinfo":{
    #             "id": user.id,
    #             "username": user.username,
    #             "bio": user.bio,
    #             "avatar": user.avatar
    #         }
    #     }
    # }
    # response_data = UserAuthResponse(token=token, user_info=UserInfoResponse.model_validate(user))
    response_data = UserAuthResponse(token=token, userInfo=UserInfoResponse.model_validate(user))
    return success_response(message="注册成功", data=response_data)


@router.post("/login")
async def login(user_data: UserRequest, db: AsyncSession = Depends(get_db)):
    # 登录逻辑：验证用户是否存在，验证密码，生成token，响应结果
    user = await users.authenticate_user(db, user_data.username, user_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    token = await users.create_token(db, user.id)
    response_data = UserAuthResponse(token=token, userInfo=UserInfoResponse.model_validate(user))

    return success_response(message="登录成功", data=response_data)


# 认证请求头+校验token+查询用户详细信息->依赖注入
@router.get("/info")
async def get_user_info(user: User=Depends(get_current_user)):
    return success_response(message="获取用户信息成功", data=UserInfoResponse.model_validate(user))

# 根据token查询用户(token->user)，用户名查询用户(db:select user.name)，根据请求体(封装pydantic)更新用户（db:update，空值忽略）
@router.put("/update")
async def update_user_info(user_data: UserUpdateRequest, user: User=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    user = await users.update_user(db, user.username, user_data)
    return success_response(message="更新用户信息成功", data=UserInfoResponse.model_validate(user))

# 验证token，校验旧密码，设置新密码（db:update）
@router.put("/password")
async def update_password(password_data: UserChangePasswordRequest, user: User=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    res_change_password = await users.update_password(db, user, password_data.old_password, password_data.new_password)
    if not res_change_password:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="修改密码失败")
    return success_response(message="更新密码信息成功")









