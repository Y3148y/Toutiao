from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.ai_chat import AiChat

# 送进模型的最近轮次上限。
# 和 ai/prompts.py 里 build_news_qa_messages 的 history[-3:] 保持一致 ——
# 这里多取不会让模型看到更多轮次，只是白白多查了几行。
HISTORY_TURNS_FOR_PROMPT = 3


async def get_recent_history(
    db: AsyncSession, user_id: int, limit: int = HISTORY_TURNS_FOR_PROMPT
) -> list[tuple[str, str]]:
    """
    取最近若干轮问答，返回 [(用户提问, 系统回答), ...]，按时间正序。

    先按时间倒序取 limit 条（数据库侧分页），再在 Python 里翻转成正序，
    这样 prompt 里的对话顺序才是对的。
    """
    result = await db.execute(
        select(AiChat.message, AiChat.response)
        .where(AiChat.user_id == user_id)
        .order_by(AiChat.created_at.desc(), AiChat.id.desc())
        .limit(limit)
    )
    rows = result.all()
    return [(msg, resp) for msg, resp in reversed(rows)]


async def list_history(
    db: AsyncSession, user_id: int, limit: int = 20
) -> list[AiChat]:
    """取历史列表，供前端展示「我之前问过什么」。最新在前。"""
    result = await db.execute(
        select(AiChat)
        .where(AiChat.user_id == user_id)
        .order_by(AiChat.created_at.desc(), AiChat.id.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def save_qa_pair(
    db: AsyncSession, user_id: int, message: str, response: str
) -> AiChat:
    """写入一轮问答。

    只记问答本身，不记检索结果：检索结果会随索引更新而变化，
    存下来反而会在后续追问里给出过时的依据。
    """
    record = AiChat(user_id=user_id, message=message, response=response)
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


async def clear_history(db: AsyncSession, user_id: int) -> int:
    """清空某用户的历史，返回删除条数"""
    from sqlalchemy import delete

    result = await db.execute(delete(AiChat).where(AiChat.user_id == user_id))
    await db.commit()
    return result.rowcount or 0
