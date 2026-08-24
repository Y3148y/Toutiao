from datetime import datetime

from sqlalchemy import Integer, DateTime, Index, ForeignKey, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped
from sqlalchemy.testing.schema import mapped_column

from models.news import News
from models.users import User

class Base(DeclarativeBase):
    pass


class History(Base):
    __tablename__ = 'history'

    __table_args__ = (
        UniqueConstraint( 'user_id', 'news_id', name='user_news_unique'),
        Index('fk_history_user_idx', 'user_id'),
        Index('fk_history_news_idx', 'news_id'),
        Index('idx_view_time', 'view_time')
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment='浏览ID')
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey(User.id), nullable=False, comment='用户ID')
    news_id: Mapped[int] = mapped_column(Integer, ForeignKey(News.id), nullable=False, comment='新闻ID')
    view_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False, comment='浏览时间')

