import os
import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Key 放在后端环境变量里（或者 .env 文件），不要写在代码里
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-xxxx-这里放后端能读到的key")
DASHSCOPE_API_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DASHSCOPE_MODEL = os.getenv("DASHSCOPE_MODEL", "max-qwen3.7-plus")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = True

@app.post("/api/ai/chat")
async def ai_chat_proxy(req: ChatRequest):
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "X-DashScope-SSE": "enable",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DASHSCOPE_MODEL,
        "messages": [m.dict() for m in req.messages],
        "stream": True,
    }

    async def event_generator():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", DASHSCOPE_API_ENDPOINT, headers=headers, json=payload) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        yield f"data: {data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")