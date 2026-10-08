from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from models.users import User


class Base(DeclarativeBase):
    pass


class AiChat(Base):
    """
    AI 问答历史。

    表结构是之前就建好的（message / response / user_id / created_at，
    带 fk_ai_chat_user_idx 和 idx_created_at 两个索引），但一直没有对应的
    ORM 模型和读写代码。字段设计刚好就是「一问一答」的结构，
    所以直接映射它，不改 DDL。

    只记录问答，不记录检索过程：检索结果每次都可能因为索引更新而变化，
    存下来反而会误导后续追问。
    """

    __tablename__ = "ai_chat"

    __table_args__ = (
        Index("fk_ai_chat_user_idx", "user_id"),
        Index("idx_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True, comment="记录ID"
    )
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(User.id), nullable=False, comment="用户ID"
    )
    message: Mapped[str] = mapped_column(Text, nullable=False, comment="用户提问")
    response: Mapped[str] = mapped_column(Text, nullable=False, comment="系统回答")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False, comment="提问时间"
    )
