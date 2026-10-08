"""
问答会话上下文。

把「这次请求该用哪份历史、回答后该写去哪」收敛到一个地方，
避免两个路由各写一遍、各漏一遍。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Header
from sqlalchemy.ext.asyncio import AsyncSession

from ai import anon_history
from utils.auth import get_optional_user
from utils.logging_conf import get_logger

logger = get_logger(__name__)

ANON_HEADER = "X-Anonymous-Id"


@dataclass
class QaSession:
    """
    一次问答的会话上下文。

    user_id 非空  -> 历史读 ai_chat 表、回答写 ai_chat 表
    user_id 为空  -> 历史读 Redis、回答写 Redis（带 24h TTL）

    anon_id 非空时会被回填（前端没带，服务端生成），用于响应头返回。
    """

    user_id: int | None = None
    anon_id: str | None = None

    @property
    def is_logged_in(self) -> bool:
        return self.user_id is not None

    async def load_history(self, db: AsyncSession) -> list[tuple[str, str]]:
        if self.is_logged_in:
            from crud import ai_chat as ai_chat_crud

            try:
                return await ai_chat_crud.get_recent_history(db, self.user_id)  # type: ignore[arg-type]
            except Exception as exc:
                # 历史读失败降级成单轮。不能因为读历史让整个问答 500。
                logger.warning("读取问答历史失败，降级为单轮: %s", exc)
                return []

        if self.anon_id:
            return await anon_history.get_history(self.anon_id)
        return []

    def recorder(self, db: AsyncSession, question: str) -> Any:
        """返回一个 async 回调，用于在本轮回答结束后落库。"""

        async def _record(answer: str) -> None:
            if not answer:
                # 模型可能返回空串（被内容过滤等），存一条空回答没有意义
                return
            if self.is_logged_in:
                from crud import ai_chat as ai_chat_crud

                await ai_chat_crud.save_qa_pair(db, self.user_id, question, answer)  # type: ignore[arg-type]
            elif self.anon_id:
                await anon_history.save_turn(self.anon_id, question, answer)

        return _record

    def response_headers(self) -> dict[str, str]:
        """
        把会话标识回传给前端。

        登录用户不需要，所以不发这个头 —— 免得前端误以为可以随便伪造。
        """
        if self.is_logged_in or not self.anon_id:
            return {}
        return {ANON_HEADER: self.anon_id}


async def resolve_session(
    db: AsyncSession,
    user: Any | None = None,
    anon_id: str | None = None,
) -> QaSession:
    """
    解析会话。路由里已经依赖注入了 user 的，直接传进来即可。

    优先用登录态。登录用户即使带了匿名 id 也走库 —— 那样历史才能跨设备。
    """
    if user is not None:
        return QaSession(user_id=getattr(user, "id", None), anon_id=anon_id)
    return QaSession(user_id=None, anon_id=anon_id or anon_history.new_anon_id())


def take_anon_id(anon_id: str | None = Header(None, alias=ANON_HEADER)) -> str | None:
    """
    从请求头取匿名会话标识。

    只接受十六进制/字母数字加 - _：
    - 恶意长串会造成 Redis key 膨胀
    - 空格、分号、控制字符会造成日志注入和 key 混乱
    - 非 ASCII（中文等）虽然在 Redis 里合法，但无意义且在不同客户端
      编码下容易不一致，一并拒绝
    """
    if not anon_id:
        return None
    cleaned = anon_id.strip()
    if not cleaned or len(cleaned) > 64:
        return None
    if not cleaned.isascii():
        return None
    if not all(c.isalnum() or c in "-_" for c in cleaned):
        return None
    return cleaned


__all__ = [
    "ANON_HEADER",
    "QaSession",
    "resolve_session",
    "take_anon_id",
    "get_optional_user",
]
