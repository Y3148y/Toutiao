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


def _upstream_error(response: httpx.Response) -> str:
    """
    从上游错误响应里取出可读原因。

    上游的错误码（如 Arrearage 欠费、InvalidApiKey）才是排查的关键信息。
    只报「连接大模型服务失败」会把人引到网络和代理上去查方向。
    """
    try:
        err = (response.json().get("error") or {})
        code = err.get("code") or ""
        message = err.get("message") or ""
        return f"{code} {message}".strip()
    except Exception:
        return response.text[:200]


def upstream_status_to_http(status_code: int) -> int:
    """把上游状态码映射成对调用方更有意义的本服务状态码"""
    return {
        401: 502,  # 密钥无效 —— 是服务端配置问题，不是客户端没权限
        402: 503,  # 欠费 —— 服务不可用
        429: 429,  # 上游限流 —— 调用方该退避重试
    }.get(status_code, 502)


async def _record_chat_usage(usage: dict | None) -> None:
    """
    把一次问答的 token 用量计入成本闸门。

    DeepSeek 系模型普通输入 元1/M 是缓存命中价 元0.2/M 的 5 倍，
    而 system prompt 每次请求都完全相同 —— 缓存命中率直接决定成本。
    所以把 cached_tokens 单独记下来，缓存没生效时能立刻看出来。

    实测结论：百炼隐式缓存要求前缀 >= 1024 token（详见 ai/config.py）。
    我们真实请求的前缀（system 约 200 token + 检索上下文约 700 token）
    低于该门槛，cached_tokens 恒为 0。日志里缓存命中率一直是 0% 属于预期，
    不是配置错误 —— 别再花时间调这个。
    """
    if not usage:
        return
    from ai.cost import BudgetExceeded, record_usage

    details = usage.get("prompt_tokens_details") or {}
    try:
        snap = await record_usage(
            DASHSCOPE_CHAT_MODEL,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            cached_input_tokens=int(details.get("cached_tokens") or 0),
        )
    except BudgetExceeded as exc:
        # 已超预算。这次调用的钱已经付了，但继续服务只会越欠越多，
        # 明确返回 429 让调用方知道是预算问题而不是服务故障。
        logger.error("预算超限: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)
        ) from exc

    logger.info(
        "问答计量 %s 缓存命中率 %.0f%%", snap.describe(), snap.cache_hit_ratio * 100
    )


async def _stream_answer(
    question: str, retrieved: list, confident: bool
):
    """
    生成 SSE 事件序列。

    独立成函数是为了能直接测帧序列 —— 这段逻辑里已经漏过两个真实 bug
    （usage 帧读不到、done 事件发两次），都发生在测试覆盖不到的内联闭包里。

    事件顺序：
        sources -> token* -> done
                  -> error（上游异常时）
    """
    # 先把溯源信息发出去。即使模型随后失败，前端也已经知道引用了哪些新闻
    yield _sse("sources", {"sources": [item.to_citation() for item in retrieved]})

    if not confident:
        # 没有足够相关的内容，直接拒答，不调 LLM。
        # 这是防幻觉最关键的一步：不给模型编造的机会。
        yield _sse("token", {"content": build_refusal_message(question, retrieved)})
        yield _sse("done", {"finishReason": "refused", "sourceCount": len(retrieved)})
        return

    messages = build_news_qa_messages(question, retrieved)
    payload = {
        "model": DASHSCOPE_CHAT_MODEL,
        "messages": messages,
        "stream": True,
        # 必须显式要求返回 usage，否则成本闸门拿不到这次调用的 token 数。
        "stream_options": {"include_usage": True},
    }

    try:
        async with httpx.AsyncClient(timeout=STREAM_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{DASHSCOPE_BASE_URL}/chat/completions",
                headers=auth_headers(stream=True),
                json=payload,
            ) as resp:
                if resp.status_code != 200:
                    detail = (
                        (await resp.aread()).decode("utf-8", errors="replace")[:500]
                    )
                    logger.error("大模型返回 %s: %s", resp.status_code, detail)
                    yield _sse("error", {"message": detail or "大模型服务返回异常"})
                    return

                # 不能在 finish_reason 帧就 return。
                #
                # 实测 DashScope 的帧序列是：
                #   ...token... -> finish_reason 帧(usage=null)
                #   -> 单独的 usage 帧(choices 为空) -> [DONE]
                # usage 帧排在 finish_reason 之后，提前 return 就永远读不到，
                # 结果流式请求完全不被计量：钱照扣、账上查不到。
                # 同步版有 usage、流式版没有 —— 同一个功能两条路径行为不一致。
                finish_reason = None
                usage = None

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:]
                    if chunk.strip() == "[DONE]":
                        break
                    try:
                        parsed = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue

                    # usage 可能单独成帧，choices 为空，必须先取再跳过
                    frame_usage = parsed.get("usage")
                    if frame_usage:
                        usage = frame_usage

                    choices = parsed.get("choices") or []
                    if not choices:
                        continue
                    content = (choices[0].get("delta") or {}).get("content")
                    if content:
                        yield _sse("token", {"content": content})
                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]

                if usage:
                    await _record_chat_usage(usage)
                else:
                    # 没拿到 usage 要留痕。静默跳过等于账单凭空少一段，
                    # 事后根本无从发现。
                    logger.warning(
                        "流式响应未返回 usage，本次调用无法计量（模型 %s）",
                        DASHSCOPE_CHAT_MODEL,
                    )

                yield _sse(
                    "done",
                    {
                        "finishReason": finish_reason or "stop",
                        "sourceCount": len(retrieved),
                        "usage": usage,
                    },
                )
    except httpx.HTTPStatusError as exc:
        # 上游的业务错误不能报成「连接失败」。
        upstream = _upstream_error(exc.response)
        logger.error("大模型返回错误 %s: %s", exc.response.status_code, upstream)
        yield _sse(
            "error",
            {
                "message": f"大模型服务返回错误({exc.response.status_code}): {upstream}",
                "upstreamStatus": exc.response.status_code,
            },
        )
    except httpx.HTTPError as exc:
        logger.error("连接大模型服务失败: %s", exc)
        yield _sse("error", {"message": "连接大模型服务失败，请稍后再试"})


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

    return StreamingResponse(
        _stream_answer(req.question, retrieved, confident),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


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
    except httpx.HTTPStatusError as exc:
        # 区分「上游返回了业务错误」和「网络/服务不可达」。
        #
        # 原来所有 httpx.HTTPError 都报「连接大模型服务失败」，结果上游返回
        # 400 Arrearage（账号欠费）时，客户端看到的也是「连接失败」——
        # 排查方向被完全带偏，会去查网络和代理，而真实原因是账单。
        # 状态码和上游的错误码必须透出来。
        upstream = _upstream_error(exc.response)
        logger.error(
            "大模型返回错误 %s: %s", exc.response.status_code, upstream
        )
        raise HTTPException(
            status_code=upstream_status_to_http(exc.response.status_code),
            detail=f"大模型服务返回错误({exc.response.status_code}): {upstream}",
        ) from exc
    except httpx.HTTPError as exc:
        # 真正的连接层问题：DNS、超时、连接被拒
        logger.error("连接大模型服务失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="连接大模型服务失败"
        ) from exc

    answer = ""
    choices = data.get("choices") or []
    if choices:
        answer = (choices[0].get("message") or {}).get("content", "")

    usage = data.get("usage")
    await _record_chat_usage(usage)

    return success_response(
        data={
            "answer": answer,
            "refused": False,
            "sources": [item.to_citation() for item in retrieved],
            "usage": usage,
        }
    )


@router.post("/news-qa/reload")
async def reload_corpus(db: AsyncSession = Depends(get_db)):
    """强制重新载入新闻语料并回写缓存（语料有变更时用）"""
    rows = await retrieve(db, "刷新", top_k=1, use_cache=False)
    return success_response(message="语料缓存已刷新", data={"hit": bool(rows[1])})