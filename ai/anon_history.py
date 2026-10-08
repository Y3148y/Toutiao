"""
匿名用户的问答历史。

未登录用户也能用问答，但 `ai_chat.user_id` 是 NOT NULL —— 那张表设计上
就是给登录用户的。硬塞一个 0 或负数当 user_id 会污染表、还会和其他匿名用户
串数据（id=0 的记录会包含所有人的对话）。

所以匿名历史走 Redis，按会话标识隔离，带 TTL 自动过期。

会话标识从哪来
--------------
要求前端带 `X-Anonymous-Id` 请求头。前端没有的话，服务端会生成一个
返回给调用方（响应头 X-Anonymous-Id），调用方下次带上即可。

不在服务端用 IP 当标识：同一个公司/学校出口 IP 相同，会导致多人看到
彼此的对话。
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from utils.logging_conf import get_logger

logger = get_logger(__name__)

_HISTORY_PREFIX = "ai:anon:qa:"
# 和登录用户取 3 轮保持一致，多存没有意义 —— 送进模型的本来就只有 3 轮
_MAX_TURNS = 3
# 匿名历史留 24 小时，隔天再追问时上下文已经断了是合理的
_TTL_SECONDS = 24 * 3600


def new_anon_id() -> str:
    return uuid.uuid4().hex


def _key(anon_id: str) -> str:
    return f"{_HISTORY_PREFIX}{anon_id}"


async def get_history(anon_id: str) -> list[tuple[str, str]]:
    """取匿名历史，返回 [(提问, 回答), ...]，时间正序"""
    from config.cache_conf import redis_client

    try:
        raw = await redis_client.get(_key(anon_id))
        if not raw:
            return []
        payload: list[Any] = json.loads(raw)
        return [(str(item[0]), str(item[1])) for item in payload]
    except Exception as exc:
        # 历史读不到不该影响问答本身，降级成单轮即可
        logger.warning("读取匿名问答历史失败，降级为单轮: %s", exc)
        return []


async def save_turn(anon_id: str, message: str, response: str) -> None:
    """追加一轮。失败只告警 —— 钱已经花了，不能因为存历史让接口 500。"""
    from config.cache_conf import redis_client

    try:
        turns = await get_history(anon_id)
        turns.append((message, response))
        # 只留最近几轮。存满会撑大 value，且更早的轮次本来就送不进模型
        turns = turns[-_MAX_TURNS:]
        await redis_client.setex(
            _key(anon_id), _TTL_SECONDS, json.dumps(turns, ensure_ascii=False)
        )
    except Exception as exc:
        logger.warning("保存匿名问答历史失败（不影响本次回答）: %s", exc)


async def clear_history(anon_id: str) -> None:
    from config.cache_conf import redis_client

    try:
        await redis_client.delete(_key(anon_id))
    except Exception as exc:
        logger.warning("清空匿名问答历史失败: %s", exc)
