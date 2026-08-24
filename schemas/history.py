from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict

from models.base import NewsItemBase


class HistoryAddRequest(BaseModel):
    news_id: int = Field(..., alias="newsId", description="新闻ID")

class HistoryItemResponse(NewsItemBase):
    history_id: int = Field(alias="historyId")
    view_time: datetime = Field(alias="viewTime")


class HistoryListResponse(BaseModel):
    list: list[HistoryItemResponse]
    total: int
    has_more: bool = Field(alias="hasMore")

    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True
    )
