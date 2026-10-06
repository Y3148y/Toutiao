"""新闻模块的响应模型。

之前这个文件是空的，接口直接返回 dict，字段名靠各处手写。
统一到 pydantic 模型后，字段改名能被静态发现，前端也有 OpenAPI 文档可依。
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class NewsListItem(BaseModel):
    """列表项。不返回 content —— 列表页不需要全文，带上会撑大响应体"""

    id: int
    title: str
    description: Optional[str] = None
    image: Optional[str] = None
    author: Optional[str] = None
    categoryId: int
    # 分类名直接带在列表里：前端不用为了显示分类名再发一次请求，
    # 也避免出现「列表已渲染但分类名后到」导致的布局跳动
    categoryName: str = ""
    publishTime: datetime
    views: int = 0


class NewsListResponse(BaseModel):
    list: list[NewsListItem]
    total: int = Field(description="符合条件的总条数，用于前端算页码")
    hasMore: bool


class NewsSearchItem(NewsListItem):
    """搜索结果比列表项多两个字段：命中词和匹配得分"""

    score: float = Field(description="BM25 相关性得分")
    matchedTerms: list[str] = Field(default_factory=list)


class NewsSearchResponse(BaseModel):
    keyword: str
    list: list[NewsSearchItem]
    total: int
    hasMore: bool
    tookMs: float = Field(description="检索耗时，便于观察性能回退")


class HotNewsItem(NewsListItem):
    """热榜项比列表项多一个热度分，用于展示「为什么它排在前面」"""

    hotScore: float


class HotNewsResponse(BaseModel):
    list: list[HotNewsItem]
    total: int
    updatedAt: datetime