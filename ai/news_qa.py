"""
新闻问答接口（基于本地新闻语料的 RAG）。

数据流：
    用户提问
      → 混合检索（BM25 + 向量 + RRF）拿到相关新闻
      → 置信度不足？直接拒答，不调 LLM
      → 构造带约束的 prompt（附新闻编号）
      → SSE 流式返回，token 逐字透传
      → 检索元信息（新闻ID溯源）作为首个事件下发

为什么要「先检索后回答」而不是直接问模型
--------------------------------------
模型不知道你的私有语料，而且它会编造看起来合理的新闻内容。
新闻问答场景下这是最不能接受的错误类型。先检索把事实喂进去，
再用 prompt 硬约束它只引用资料，才能让答案可溯源。
"""
import json

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from ai.config import (
    DASHSCOPE_BASE_URL,
    DASHSCOPE_CHAT_MODEL,
    REQUEST_TIMEOUT,
    STREAM_TIMEOUT,
    TOP_K_FINAL,
    auth_headers,
    is_configured,
)
from ai.prompts import build_news_qa_messages, build_refusal_message
from ai.retriever import retrieve
from config.db_conf import get_db
from utils.logging_conf import get_logger
from utils.response import success_response

import httpx

logger = get_logger(__name__)

router = APIRouter(prefix="/api/ai", tags=["ai"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # 关掉 nginx 缓冲，否则 SSE 会被攒成一坨最后一次性吐出
    "X-Accel-Buffering": "no",
}


class NewsQaRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=200, description="用户问题")
    top_k: int = Field(TOP_K_FINAL, ge=1, le=10, description="参与回答的新闻条数")


def _sse(event: str, data: dict) -> str:
    """按 SSE 协议格式化一个事件"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/news-qa")
async def news_qa_stream(req: NewsQaRequest, db: AsyncSession = Depends(get_db)):
    """
    基于新闻语料的问答，SSE 流式返回。

    事件序列：
        event: sources   -> 检索到的新闻溯源信息（首个事件，前端先渲染引用列表）
        event: token     -> 逐字生成的答案内容
        event: done      -> 生成结束
        event: error     -> 出错（仅在检索/请求阶段失败时出现）
    """
    if not is_configured():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服务器未配置 DASHSCOPE_API_KEY 环境变量",
        )

    retrieved, confident = await retrieve(db, req.question, top_k=req.top_k)

    async def event_generator():
        # 先把溯源信息发出去。即使模型随后失败，前端也已经知道引用了哪些新闻
        yield _sse("sources", {"sources": [item.to_citation() for item in retrieved]})

        if not confident:
            # 没有足够相关的内容，直接拒答，不调 LLM。
            # 这是防幻觉最关键的一步：不给模型编造的机会。
            yield _sse("token", {"content": build_refusal_message(req.question, retrieved)})
            yield _sse("done", {"finishReason": "refused", "sourceCount": len(retrieved)})
            return

        messages = build_news_qa_messages(req.question, retrieved)
        payload = {"model": DASHSCOPE_CHAT_MODEL, "messages": messages, "stream": True}

        try:
            async with httpx.AsyncClient(timeout=STREAM_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    f"{DASHSCOPE_BASE_URL}/chat/completions",
                    headers=auth_headers(stream=True),
                    json=payload,
                ) as resp:
                    if resp.status_code != 200:
                        detail = (await resp.aread()).decode("utf-8", errors="replace")[:500]
                        logger.error("大模型返回 %s: %s", resp.status_code, detail)
                        yield _sse("error", {"message": "大模型服务返回异常，请稍后再试"})
                        return

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        chunk = line[6:]
                        # [DONE] 是流结束标记，不是 JSON
                        if chunk.strip() == "[DONE]":
                            break
                        try:
                            parsed = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue

                        choices = parsed.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if content:
                            yield _sse("token", {"content": content})
                        finish_reason = choices[0].get("finish_reason")
                        if finish_reason:
                            yield _sse(
                                "done",
                                {
                                    "finishReason": finish_reason,
                                    "sourceCount": len(retrieved),
                                    "usage": parsed.get("usage"),
                                },
                            )
                            return
        except httpx.HTTPError as exc:
            logger.error("调用大模型失败: %s", exc)
            yield _sse("error", {"message": "连接大模型服务失败，请稍后再试"})
            return

        # 流自然结束但没收到 finish_reason（少数网关会这样）
        yield _sse("done", {"finishReason": "stop", "sourceCount": len(retrieved)})

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/news-qa/sync")
async def news_qa_sync(req: NewsQaRequest, db: AsyncSession = Depends(get_db)):
    """
    同步版本的新闻问答，一次性返回完整答案。

    与流式版共用检索和 prompt 逻辑，方便非流式场景（比如脚本调用、
    不支持 SSE 的客户端）使用，也便于写断言做集成测试。
    """
    if not is_configured():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服务器未配置 DASHSCOPE_API_KEY 环境变量",
        )

    retrieved, confident = await retrieve(db, req.question, top_k=req.top_k)

    if not confident:
        return success_response(
            data={
                "answer": build_refusal_message(req.question, retrieved),
                "refused": True,
                "sources": [item.to_citation() for item in retrieved],
            }
        )

    messages = build_news_qa_messages(req.question, retrieved)
    payload = {"model": DASHSCOPE_CHAT_MODEL, "messages": messages, "stream": False}

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(
                f"{DASHSCOPE_BASE_URL}/chat/completions",
                headers=auth_headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        logger.error("调用大模型失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="连接大模型服务失败"
        ) from exc

    answer = ""
    choices = data.get("choices") or []
    if choices:
        answer = (choices[0].get("message") or {}).get("content", "")

    return success_response(
        data={
            "answer": answer,
            "refused": False,
            "sources": [item.to_citation() for item in retrieved],
            "usage": data.get("usage"),
        }
    )


@router.post("/news-qa/reload")
async def reload_corpus(db: AsyncSession = Depends(get_db)):
    """强制重新载入新闻语料并回写缓存（语料有变更时用）"""
    rows = await retrieve(db, "刷新", top_k=1, use_cache=False)
    return success_response(message="语料缓存已刷新", data={"hit": bool(rows[1])})