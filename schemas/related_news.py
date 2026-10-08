from typing import Optional

from pydantic import BaseModel, Field


class RelatedNewsCreate(BaseModel):
    """
    新增关联。

    newsId 为空时，从 relatedNewsId 反查双向：如果你把 A 关联到 B，
    再把 B 关联到 A 就是同一个意思，这里自动识别。
    """

    news_id: Optional[int] = Field(None, alias="newsId")
    related_news_id: int = Field(..., alias="relatedNewsId")

    model_config = {"populate_by_name": True}


class RelatedNewsResponse(BaseModel):
    relatedNewsIds: list[int] = Field(default_factory=list)


class RelatedNewsCreateResponse(BaseModel):
    created: bool = Field(description="false 表示已存在，未新增")
    relatedNewsIds: list[int] = Field(default_factory=list)
