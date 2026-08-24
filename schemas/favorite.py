from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict

from models.base import NewsItemBase


class FavoriteCheckResponse(BaseModel):
    is_favorite: bool = Field(..., alias="isFavorite")

class FavoriteAddRequest(BaseModel):
    news_id: int = Field(..., alias="newsId")

# 一个新闻模型类，一个收藏模型类
class FavoriteNewsItemResponse(NewsItemBase):
    favorite_id: int = Field(alias="favoriteId")
    favorite_time: datetime = Field(alias="favoriteTime")

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True
    )


class FavoriteListResponse(BaseModel):
    list: list[FavoriteNewsItemResponse]
    total: int
    has_more: bool = Field(alias="hasMore")

    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True
    )

