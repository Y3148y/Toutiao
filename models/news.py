from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func
from datetime import datetime
from sqlalchemy import DateTime, ForeignKey
from sqlalchemy import String
from sqlalchemy import Integer
from sqlalchemy import Index
from sqlalchemy import Text
from typing import Optional


# 基类
class Base(DeclarativeBase):
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), comment="创建时间")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), comment="更新时间")

# 表对应的模型类
class Category(Base):
    __tablename__ = "news_category"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment="分类ID")
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, comment="分类名称")
    sort_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False, comment="排序")

    def __repr__(self):
        return f"<Category(id={self.id}, name={self.name}, sort_order={self.sort_order})>"


# 新闻类
class News(Base):
    __tablename__ = "news"

    # 创建索引，提升查询速度
    __table_args__ = (
        Index('fk_news_category_idx', 'category_id'),
        Index('idx_publish_time', 'publish_time')
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment="新闻ID")
    author: Mapped[Optional[str]] = mapped_column(String(50), comment="作者")
    category_id: Mapped[int] = mapped_column(Integer, ForeignKey('news_category.id'),nullable=False, comment="分类ID")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="内容")
    description: Mapped[Optional[str]] = mapped_column(String(255), comment="描述")
    image: Mapped[Optional[str]] = mapped_column(String(255), comment="图片")
    publish_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False, comment="发布时间")
    title: Mapped[str] = mapped_column(String(255), nullable=False, comment="标题")
    views: Mapped[int] = mapped_column(Integer, default=0, nullable=False, comment="浏览量")

    def __repr__(self):
        return f"<News(id={self.id}, author={self.author}, category_id={self.category_id}, content={self.content}, description={self.description}, image={self.image}, publish_time={self.publish_time}, title={self.title}, views={self.views})>"

















