"""
演示数据填充脚本（幂等）。

用途
----
面试演示前跑一次，让 related_news 表有真实的推荐关系、有一个可直接登录的
演示账号。否则详情页只能走「同分类 + 浏览量」的兜底推荐，
演示不出「运营手工配置优先」这个特性。

为什么做成幂等脚本而不是一次性 SQL
--------------------------------
面试前可能反复跑，或者换环境重跑。非幂等脚本第二次执行会撞唯一约束
（user_news_unique / news_related_unique）直接报错，
使用者还得先手动清表。幂等写法让它可以随时执行、结果一致。

用法
----
    python -m scripts.seed_demo

可选：清空演示数据后重跑
    python -m scripts.seed_demo --reset
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import delete, func, select  # noqa: E402

from config.db_conf import async_engine, AsyncSessionLocal  # noqa: E402
from crud.related_news import invalidate_related_cache  # noqa: E402
from models.news import News  # noqa: E402
from models.related_news import RelatedNews  # noqa: E402
from models.users import User  # noqa: E402

DEMO_USERNAME = "demo"
DEMO_PASSWORD = "demo123456"
DEMO_NICKNAME = "演示用户"

# 关联规则：(主新闻标题包含, 关联新闻标题包含)
# 用标题关键词而不是硬编码 id —— 换库导入后 id 会变，标题相对稳定。
RELATED_RULES: list[tuple[str, list[str]]] = [
    ("GDP", ["服务业PMI", "全球债务风险"]),
    ("AI芯片", ["AI监管", "人工智能", "AI编程助手"]),
    ("AI监管", ["AI芯片", "AI伦理"]),
    ("量子计算", ["量子互联网", "量子通信", "量子计算机"]),
    ("SpaceX", ["星舰", "火箭"]),
    ("新能源汽车", ["充电桩", "电池"]),
    ("芯片", ["AI芯片", "半导体", "三星"]),
]


async def _demo_pairs(session) -> set[tuple[int, int]]:
    """按现有规则算出会生成哪些关联对，用于精准清理"""
    all_news = list((await session.execute(select(News))).scalars().all())
    pairs: set[tuple[int, int]] = set()
    for main_kw, related_kws in RELATED_RULES:
        for main in [n for n in all_news if main_kw in (n.title or "")]:
            for kw in related_kws:
                for target in all_news:
                    if kw in (target.title or "") and target.id != main.id:
                        pairs.add((main.id, target.id))
    return pairs


async def reset(session) -> None:
    """
    清掉本脚本此前造的数据，保证可重复执行。

    只删「按规则会生成的那些关联对」，不整表清空 ——
    库里可能有手工配的关联，一刀切掉就丢数据了。
    """
    pairs = await _demo_pairs(session)
    removed = 0
    for main_id, target_id in pairs:
        result = await session.execute(
            delete(RelatedNews).where(
                RelatedNews.news_id == main_id,
                RelatedNews.related_news_id == target_id,
            )
        )
        removed += result.rowcount or 0
    if removed:
        print(f"已清除 {removed} 条演示关联")

    user = await session.execute(select(User).where(User.username == DEMO_USERNAME))
    existing = user.scalar_one_or_none()
    if existing:
        await session.execute(delete(User).where(User.id == existing.id))
        print("已清除演示账号")
    await session.commit()


async def seed_user(session) -> User:
    existing = await session.execute(
        select(User).where(User.username == DEMO_USERNAME)
    )
    user = existing.scalar_one_or_none()
    if user:
        print(f"演示账号已存在: {user.username} (id={user.id})")
        return user

    from utils.security import get_hash_password

    user = User(
        username=DEMO_USERNAME,
        password=get_hash_password(DEMO_PASSWORD),
        nickname=DEMO_NICKNAME,
        bio="用于面试演示的账号",
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    print(f"已创建演示账号: {user.username} / {DEMO_PASSWORD} (id={user.id})")
    return user


async def seed_related(session) -> int:
    """
    按标题关键词建关联。

    已存在的关联跳过，不重复插入 —— 幂等的关键。
    """
    all_news = list((await session.execute(select(News))).scalars().all())
    if not all_news:
        print("news 表为空，跳过关联填充")
        return 0

    existing_pairs = {
        (row[0], row[1])
        for row in (await session.execute(
            select(RelatedNews.news_id, RelatedNews.related_news_id)
        )).all()
    }

    created = 0
    for main_kw, related_kws in RELATED_RULES:
        mains = [n for n in all_news if main_kw in (n.title or "")]
        for main in mains:
            for kw in related_kws:
                for target in all_news:
                    if kw not in (target.title or ""):
                        continue
                    # 不能自关联，也不重复
                    if target.id == main.id:
                        continue
                    if (main.id, target.id) in existing_pairs:
                        continue
                    session.add(
                        RelatedNews(news_id=main.id, related_news_id=target.id)
                    )
                    existing_pairs.add((main.id, target.id))
                    created += 1
                    print(f"  关联 #{main.id} -> #{target.id}")

    if created:
        await session.commit()
        # 关联变了要清推荐缓存，否则用户看到的还是旧的
        touched = {m.id for m in all_news}
        for nid in touched:
            await invalidate_related_cache(nid)
    print(f"新增关联 {created} 条")
    return created


async def main_async(args) -> int:
    try:
        async with AsyncSessionLocal() as session:
            if args.reset:
                await reset(session)
            await seed_user(session)
            created = await seed_related(session)

            total_related = await session.execute(
                select(func.count()).select_from(RelatedNews)
            )
            print()
            print(f"related_news 表当前共 {total_related.scalar()} 条关联")
        return 0
    finally:
        await async_engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description="填充面试演示数据")
    parser.add_argument(
        "--reset", action="store_true", help="先清空演示数据再重建"
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())