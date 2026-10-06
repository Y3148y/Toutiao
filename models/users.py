from typing import Optional

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Index, Integer, String, Enum, DateTime, ForeignKey
from datetime import datetime


class Base(DeclarativeBase):
    pass


class User(Base):

    __tablename__ = 'user'

    __table_args__ = (
        Index('username_UNIQUE', 'username'),
        Index('phone_UNIQUE', 'phone')
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment="用户ID")
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, comment="用户名")
    password: Mapped[str] = mapped_column(String(255), nullable=False, comment="密码（加密存储）")
    nickname: Mapped[Optional[str]] = mapped_column(String(50), comment="昵称")
    avatar: Mapped[Optional[str]] = mapped_column(String(255), comment="头像URL",
                                        default='')
    gender: Mapped[Optional[str]] = mapped_column(Enum('male', 'female', 'unknown'), comment="性别", default='unknown')
    bio: Mapped[Optional[str]] = mapped_column(String(500), comment="个人简介", default='这儿嗯很懒，什么都没留下')
    phone: Mapped[Optional[str]] = mapped_column(String(20), unique=True ,comment="手机号")
    created_at: Mapped[DateTime] = mapped_column(DateTime, default=datetime.now, comment="创建时间")
    updated_at: Mapped[DateTime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now,
                                                 comment="更新时间",)
class UserToken(Base):

    __tablename__ = 'user_token'

    __table_args__ = (
        Index('token_UNIQUE', 'token'),
        Index('fk_user_token_user_idx', 'user_id')
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment='令牌id')
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey(User.id), nullable=False, comment="用户id")
    token: Mapped[str] = mapped_column(String(255), nullable=False, comment="令牌")
    expires_at: Mapped[DateTime] = mapped_column(DateTime, nullable=False, comment="过期时间")
    created_at: Mapped[DateTime] = mapped_column(DateTime, default=datetime.now, comment="创建时间")

    def __repr__(self):
        return f"<UserToken(id={self.id}, user_id={self.user_id}, token='{self.token}')>"
