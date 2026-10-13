"""
通用大模型 SSE 代理。

这个端点是**直通大模型的通用入口**：没有检索、没有置信判控，
用户传什么就转发什么。所以它是全项目**花钱最直接、防护最薄**的一处，
这里的几项限制都不是可选的优化，而是防止接口被当成免费算力：

1. **默认要求登录**（`AI_CHAT_REQUIRE_AUTH`，默认开）。未登录直接 422/401，
   不给匿名用户白嫖模型额度。
2. **入参上限**。messages 条数与单条 content 长度都设了硬上限 ——
   不设的话，一个匿名脚本塞 500 条超长消息就能把上下文窗口打满、
   让单次调用成本飙升。
3. **成本计量**。走 `ai.cost.record_chat_usage`，与 `/api/ai/news-qa` 同一个闸门；
   超预算直接 429。**不计量等于这个端点是个成本黑洞。**
4. **上游错误要能诊断**。原实现只有 `resp.raise_for_status()`，且它发生在
   **响应头已经发出之后** —— 客户端只会看到一个断掉的空流，
   HTTP 状态码和上游错误码（欠费 / 密钥无效）全部丢失。
   现在改为先 `send(stream=True)` 拿到状态码再决定是否返回 StreamingResponse。
"""
import json
import os
from typing import Any, List

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from config.db_conf import get_db

from ai.config import (
    DASHSCOPE_BASE_URL,
    DASHSCOPE_CHAT_MODEL,
    STREAM_TIMEOUT,
    auth_headers,
    is_configured,
)
from utils.auth import get_current_user
from utils.logging_conf import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/ai", tags=["ai"])

# 默认要求登录。这个端点花钱且无检索护栏，公开等于送额度。
REQUIRE_AUTH = os.getenv("AI_CHAT_REQUIRE_AUTH", "true").lower() not in (
    "0",
    "false",
    "no",
)

# 入参上限。超过就 422，不转发 —— 否则单次请求的 token 量不可控。
MAX_MESSAGES = int(os.getenv("AI_CHAT_MAX_MESSAGES", "50"))
MAX_CONTENT_CHARS = int(os.getenv("AI_CHAT_MAX_CONTENT_CHARS", "8000"))

# 上游错误体最多取多少字回传给客户端
_ERROR_BODY_LIMIT = 300


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str = Field(..., min_length=1, max_length=MAX_CONTENT_CHARS)


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1, max_length=MAX_MESSAGES)
    stream: bool = True


async def _chat_auth(
    authorization: str | None = Header(None, alias="Authorization"),
    db: AsyncSession = Depends(get_db),
) -> Any | None:
    """
    条件鉴权：需要时校验 Token，不需要时放行。

    写成带声明参数的依赖、而不是在函数体里 `await get_current_user()` ——
    `get_current_user` 的默认参数是 `Header(...)` 对象，直接调用会把那个
    Header 实例当成 token 传进去，必然 AttributeError。必须让 FastAPI
    完成参数解析，再把真实字符串传进去。

    与其它需登录接口的唯一差异：这里缺 Token 返回 **401** 而不是 422
    （422 是 `Header(...)` 必填校验的副作用，语义上 401 更准确）。
    """
    if not REQUIRE_AUTH:
        return None
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="通用对话需要登录后才能使用",
        )
    return await get_current_user(authorization=authorization, db=db)


async def _open_upstream_stream(
    payload: dict,
) -> tuple[httpx.AsyncClient, httpx.Response]:
    """
    打开上游流式连接并**在返回 StreamingResponse 之前**确认状态码。

    为什么不能直接 `async with client.stream(...)` 然后 raise_for_status()：
    那行代码执行时 HTTP 响应头已经发给客户端了，此时再抛异常，
    客户端拿到的是一个「200 但内容为空」的断流，真实错误码全部丢失。
    所以这里先 `send(stream=True)` 拿到响应对象，状态码不对就读 body 抛
    HTTPException —— 此时响应头尚未发出，状态码才真正能被客户端看到。
    """
    client = httpx.AsyncClient(timeout=STREAM_TIMEOUT)
    try:
        req = client.build_request(
            "POST",
            f"{DASHSCOPE_BASE_URL}/chat/completions",
            headers=auth_headers(stream=True),
            json=payload,
        )
        resp = await client.send(req, stream=True)
    except Exception:
        await client.aclose()
        raise

    if resp.status_code != 200:
        # 读出上游错误体再关连接：错误码（欠费/密钥无效）才是排查的关键，
        # 只报「连接大模型服务失败」会把人引到网络和代理上查错方向。
        try:
            raw = (await resp.aread()).decode("utf-8", errors="replace")
        except Exception:
            raw = ""
        finally:
            await resp.aclose()
            await client.aclose()

        from ai.news_qa import upstream_status_to_http

        detail = raw[:_ERROR_BODY_LIMIT] or "上游未返回错误详情"
        logger.error("大模型返回 %s: %s", resp.status_code, detail)
        raise HTTPException(
            status_code=upstream_status_to_http(resp.status_code),
            detail=f"大模型服务返回错误({resp.status_code}): {detail}",
        )

    return client, resp


@router.post("/chat")
async def ai_chat_proxy(
    req: ChatRequest,
    user: Any | None = Depends(_chat_auth),
):
    """
    通用大模型流式问答（透传 SSE）。

    需要登录（可用 AI_CHAT_REQUIRE_AUTH=false 关闭，仅限本地调试）。
    帧格式与上游保持一致：`data: {...}` / `data: [DONE]`。
    """
    if not is_configured():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服务器未配置 DASHSCOPE_API_KEY 环境变量",
        )

    # 模型名统一走 ai/config.py，避免这里再读一遍环境变量导致
    # 两个端点用不同模型、计价表也对不上。
    payload = {
        "model": DASHSCOPE_CHAT_MODEL,
        "messages": [m.model_dump() for m in req.messages],
        "stream": True,
        # 不显式要 usage 就拿不到 token 数，成本闸门形同虚设
        "stream_options": {"include_usage": True},
    }

    client, resp = await _open_upstream_stream(payload)

    async def event_generator():
        usage: dict | None = None
        try:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = line[6:]

                # usage 可能单独成帧且排在 finish_reason 之后，
                # 必须在循环里持续捕获，不能遇到结束就返回
                # （这个坑在 news_qa 流式路径上真实漏过一次，见该文件注释）。
                if chunk.strip() != "[DONE]":
                    try:
                        parsed = json.loads(chunk)
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict) and parsed.get("usage"):
                        usage = parsed["usage"]

                yield f"data: {chunk}\n\n"
        except Exception as exc:  # noqa: BLE001 —— 流已开始，只能就地收尾
            # 响应头早已发出，无法改状态码。发一个 error 帧让前端能提示，
            # 然后正常关闭 —— 直接抛出会留下一个没有 [DONE] 的断流。
            logger.error("透传流中断: %s", exc)
            yield (
                "data: "
                + json.dumps(
                    {"error": {"message": f"上游流中断: {exc}"}}, ensure_ascii=False
                )
                + "\n\n"
            )
            yield "data: [DONE]\n\n"
        finally:
            await resp.aclose()
            await client.aclose()

            if usage:
                from ai.cost import BudgetExceeded, record_chat_usage

                try:
                    await record_chat_usage(DASHSCOPE_CHAT_MODEL, usage)
                except BudgetExceeded as exc:
                    # 钱已经花了，但预算闸门要拦住后续调用。流已结束无法改状态码，
                    # 只能记日志 —— 同步的 news-qa 路径是抛 429。
                    logger.error("预算超限（流式已结束，仅记录）: %s", exc)
            else:
                # 没拿到 usage 必须留痕。静默跳过等于账单凭空少一段，
                # 事后根本无从发现。
                logger.warning("通用对话流式未收到 usage 帧，成本可能漏计")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 关掉 nginx 缓冲，否则 SSE 会被攒成一坨最后一次性吐出
            "X-Accel-Buffering": "no",
        },
    )