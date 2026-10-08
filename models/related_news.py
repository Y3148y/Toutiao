from datetime import datetime

from sqlalchemy import DateTime, Integer, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from models.news import News


class Base(DeclarativeBase):
    pass


class RelatedNews(Base):
    """
    手动关联的相关新闻。

    同样是之前建好表但一直没接代码：详情页的相关推荐走的是
    「同分类 + 按浏览量排序」的实时查询，运营无法控制推荐什么。
    这张表就是给运营用的白名单 —— 配了就优先展示。

    两个方向都存（news_id -> related_news_id），由写入方保证对称；
    读取时不依赖对称性，读到什么算什么。
    """

    __tablename__ = "related_news"

    __table_args__ = (
        UniqueConstraint("news_id", "related_news_id", name="news_related_unique"),
        Index("fk_related_news_news_idx", "news_id"),
        Index("fk_related_news_related_idx", "related_news_id"),
    )

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True, comment="关联ID"
    )
    news_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(News.id), nullable=False, comment="主新闻ID"
    )
    related_news_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(News.id), nullable=False, comment="关联新闻ID"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False, comment="关联创建时间"
    )
