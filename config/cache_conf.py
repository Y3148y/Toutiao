import json
from typing import Any

import redis.asyncio as redis

REDIS_HOST = "192.168.119.128"
REDIS_PORT = 6379
REDIS_DB = 3

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True # 是否将字节数据解码为字符串
)

# 字符串
async def get_cache(key: str):
    try:
        return await redis_client.get(key)
    except Exception as e:
        print(f"获取缓存失败：{e}")

# list dict
async def get_json_cache(key: str):
    try:
        data = await redis_client.get(key)
        if data:
            return json.loads(data) # 序列化
        return None
    except Exception as e:
        print(f"获取缓存失败：{e}")
        return None

# 设置缓存 setex(key, expire, value)
async def set_cache(key: str, value: Any, expire: int= 3600 ):
    try:
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False) # 转字符串，保留中文不转意
        await redis_client.setex(key, expire, value)
        return True
    except Exception as e:
        print(f"设置缓存失败：{e}")
        return False

async def icr(key):
    return await redis_client.incr(key)

async def exp(key, time=3600):
    await redis_client.expire(key, time)   # 1小时过期，可调


