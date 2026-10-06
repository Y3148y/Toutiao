"""
从数据库导出语料到本地 JSON，供评估器离线使用。

    python -m ai.eval.export_corpus

为什么需要落成本地文件：
评估必须能在 CI 里跑（不连 MySQL），且结果可复现。
语料文件应该与 resourse/database.sql 保持同步 ——
改了种子数据就重新跑一次这个脚本。
"""
import asyncio
import io
import json
import os
import sys
from pathlib import Path

OUTPUT = Path(__file__).with_name("_corpus.json")


async def export() -> None:
    try:
        import aiomysql
    except ImportError:
        print("缺少依赖，执行 pip install aiomysql", file=sys.stderr)
        raise SystemExit(1)

    conn = await aiomysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        db=os.getenv("DB_NAME", "news_app"),
        charset="utf8mb4",
    )
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT id, title, description, content, category_id, publish_time "
                "FROM news ORDER BY id"
            )
            rows = await cur.fetchall()
    finally:
        conn.close()

    payload = {
        "news": [
            {
                "id": r[0],
                "title": r[1],
                "description": r[2],
                "content": r[3],
                "category_id": r[4],
                "publish_time": r[5].isoformat() if r[5] else None,
            }
            for r in rows
        ]
    }
    with io.open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"已导出 {len(payload['news'])} 条 -> {OUTPUT}")


if __name__ == "__main__":
    asyncio.run(export())
