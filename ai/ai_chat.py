import os
import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/api/ai", tags=["ai"])

# Key 只放后端环境变量，绝不出现在代码或前端浏览器里
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_API_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DASHSCOPE_MODEL = os.getenv("DASHSCOPE_MODEL", "max-qwen3.7-plus")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = True


@router.post("/chat")
async def ai_chat_proxy(req: ChatRequest):
    if not DASHSCOPE_API_KEY:
        raise HTTPException(status_code=500, detail="服务器未配置 DASHSCOPE_API_KEY 环境变量")

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "X-DashScope-SSE": "enable",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DASHSCOPE_MODEL,
        "messages": [m.model_dump() for m in req.messages],
        "stream": True,
    }

    async def event_generator():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", DASHSCOPE_API_ENDPOINT, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        yield f"data: {data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")