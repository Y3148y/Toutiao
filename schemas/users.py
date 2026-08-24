from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class UserRequest(BaseModel):
    username: str
    password: str


class UserInfoBase(BaseModel):
    """
    用户基础信息数据模型
    """
    nickname: Optional[str] = Field(None, max_length=50, description="昵称")
    avatar: Optional[str] = Field(None, max_length=255,description="头像URL")
    gender: Optional[str] = Field(None, max_length=10, description="性别")
    bio: Optional[str] = Field(None, max_length=500, description="个人简介")

    # user_info
class UserInfoResponse(UserInfoBase):
    id: int
    username: str

    model_config = ConfigDict(
        from_attributes=True  # 允许从ORM对象属性中取值
    )


class UserAuthResponse(BaseModel):
    token: str
    user_info: UserInfoResponse = Field(..., alias="userInfo")

    model_config = ConfigDict(
        populate_by_name=True, # alias字段名兼容
        from_attributes=True  # 打开这个配置，`model_validate()`就会读取 ORM 对象的属性，自动映射到 pydantic 字段，不用手动写`resp.id = user_orm.id`
    )

# 更新用户信息的模型类
class UserUpdateRequest(BaseModel):
    nickname: str = None
    avatar: str = None
    gender: str = None
    bio: str = None

class UserChangePasswordRequest(BaseModel):
    old_password: str = Field(..., alias="oldPassword", description="旧密码")
    new_password: str = Field(..., min_length=6, alias="newPassword", description="新密码")