"""
离线索引构建的 CLI 入口。

    python -m ai.ingest.cli build          # 构建/增量更新索引
    python -m ai.ingest.cli validate       # 只做语料校验，看脏数据
    python -m ai.ingest.cli status         # 查看当前 manifest
    python -m ai.ingest.cli snapshots      # 列出历史版本，可用于回滚

设计取向：默认**只读本地语料快照**（ai/eval/_corpus.json），不连数据库。
这样离线构建可以在任何环境跑，也能在 CI 里验证。
需要连库时用 `--from-db`。
"""
import argparse
import asyncio
import io
import json
import sys
from pathlib import Path

from ai import config as ai_config
from ai.config import DASHSCOPE_EMBED_MODEL
from ai.ingest.builder import build_index
from ai.ingest.manifest import IndexManifest, load_manifest, list_snapshots
from ai.ingest.validate import corpus_fingerprint, validate_corpus
from config.db_conf import AsyncSessionLocal

DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "eval" / "_corpus.json"


def load_corpus_file(path: Path) -> list:
    """从导出的语料 JSON 构造 News 对象"""
    from datetime import datetime

    from models.news import News

    if not path.exists():
        print(f"找不到语料文件 {path}")
        print("先运行 python -m ai.eval.export_corpus 导出")
        raise SystemExit(1)

    data = json.load(io.open(path, encoding="utf-8"))
    return [
        News(
            id=n["id"],
            title=n["title"],
            description=n.get("description"),
            content=n.get("content") or n["title"],
            category_id=n["category_id"],
            views=0,
            publish_time=datetime.fromisoformat(n["publish_time"])
            if n.get("publish_time")
            else datetime(2026, 1, 1),
        )
        for n in data["news"]
    ]


async def load_corpus_db():
    """
    从数据库读语料。

    必须显式关闭 engine：aiomysql 的连接在 __del__ 里尝试关闭，
    事件循环已经结束时抛「Event loop is closed」，
    那段噪音会淹没 CLI 真正的错误输出。
    """
    from sqlalchemy import select

    from config.db_conf import async_engine
    from models.news import News

    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(News).order_by(News.id))
            return list(result.scalars().all())
    finally:
        await async_engine.dispose()


async def load_docs(args) -> list:
    """
    按参数选择语料来源。

    必须是 async：--from-db 时要连数据库，而调用方已在事件循环内，
    此时用同步的 asyncio.run() 会抛「cannot be called from a running event loop」。
    """
    if args.from_db:
        return await load_corpus_db()
    return load_corpus_file(Path(args.corpus))


async def cmd_validate(args) -> int:
    docs = await load_docs(args)

    accepted, report = validate_corpus(docs, strict=False)

    print(f"语料校验结果：{report.summary()}")
    print(f"语料指纹：{corpus_fingerprint(accepted)}")
    print()

    if report.issues:
        print("发现的问题（最多列 50 条）：")
        for issue in report.issues[:50]:
            print(f"  {issue}")
        if len(report.issues) > 50:
            print(f"  ... 还有 {len(report.issues) - 50} 条")
    else:
        print("未发现问题")

    return 1 if report.has_errors() and args.strict else 0


async def cmd_build(args) -> int:
    docs = await load_docs(args)

    if not docs:
        print("语料为空，终止")
        return 1

    if not args.model:
        print("缺少 --model：embedding 模型名是索引兼容性判定的依据，必须显式传入")
        print("例如 --model text-embedding-v4")
        return 1

    previous = None
    if not args.full:
        previous = await load_manifest()

    result = await build_index(
        docs,
        previous=previous,
        embed_model=args.model,
        # 显式传维度，让「同模型换维度」也能触发失效判定。
        # 测试里用假向量时故意不传，避免固定维度的假数据被误判为变更。
        embed_dim=ai_config.DASHSCOPE_EMBED_DIM,
        resume=not args.no_resume,
        strict=args.strict,
    )

    if result.errors:
        print("构建失败：")
        for err in result.errors:
            print(f"  {err}")
        return 1

    m = result.manifest
    print()
    print("索引构建完成")
    print(f"  索引 ID      {m.index_id}")
    print(f"  模型/维度   {m.embed_model} / {m.embed_dim}")
    print(f"  语料指纹     {m.corpus_fingerprint}")
    print(f"  语料         {m.document_count} 篇")
    print(f"  校验        {result.report.summary()}")
    print(f"  向量化      {result.embedded} 篇")
    print(f"  复用        {result.reused} 篇")
    if result.resumed:
        print("  （本次为断点续跑）")
    print(f"  用时        {result.duration_ms:.0f} ms")

    if result.report.issues:
        print()
        print("校验问题（不阻断构建，但建议处理）：")
        for issue in result.report.issues[:20]:
            print(f"  {issue}")

    return 0


async def cmd_status(args) -> int:
    m = await load_manifest()
    if not m:
        print("当前没有索引 manifest")
        print("执行 python -m ai.ingest.cli build --model <model> 构建")
        return 1

    print("当前索引 manifest")
    print(json.dumps(m.to_dict(), ensure_ascii=False, indent=2))

    print()
    compatible, reason = m.is_compatible_with(args.model or m.embed_model, m.embed_dim)
    print(f"与当前模型({args.model or m.embed_model})兼容：{compatible} — {reason}")
    return 0


def cmd_snapshots(args) -> int:
    files = list_snapshots()
    if not files:
        print("没有历史快照")
        return 1
    print("历史索引版本：")
    for path in files:
        data = json.load(io.open(path, encoding="utf-8"))
        print(
            f"  {path.name}  模型={data.get('embed_model')} "
            f"维度={data.get('embed_dim')} 语料={data.get('document_count')}篇 "
            f"构建于={data.get('built_at')}"
        )
    return 0


async def cmd_index_status(args) -> int:
    """当前索引的落盘状态，判断它还能不能用"""
    from ai.ingest.storage import index_stats, load_manifest_file

    stats = index_stats()
    if not stats.get("exists"):
        print("当前没有离线索引，检索会走运行时构建路径")
        print("执行 python -m ai.ingest.cli build --model <model> 构建")
        return 1

    print("离线索引状态")
    print(f"  文档数      {stats['documentCount']}")
    print(f"  向量维度    {stats['dim']}")
    print(f"  文件大小    {stats['vectorsSizeMB']} MB ({stats['vectorsFile']})")
    print(f"  索引 ID     {stats.get('indexId')}")
    print(f"  模型        {stats.get('embedModel')}")
    print(f"  构建时间    {stats.get('builtAt')}")
    print(f"  语料指纹    {stats.get('fingerprint')}")

    manifest_data = load_manifest_file()
    if manifest_data and args.model:
        manifest = IndexManifest.from_dict(manifest_data)
        compatible, reason = manifest.is_compatible_with(args.model, stats["dim"])
        print()
        print(f"  与指定模型({args.model})兼容性: {compatible} — {reason}")
    return 0


def cmd_clear(args) -> int:
    """清空离线索引，强制下次检索走运行时构建"""
    from ai.ingest.storage import clear_index

    clear_index()
    print("离线索引已清空")
    return 0


async def cmd_cost(args) -> int:
    """打印累计 token 用量与预估花费"""
    from ai.config import BUDGET_ASK_CNY, BUDGET_TOTAL_CNY
    from ai.cost import get_usage, reset_usage

    if args.reset:
        await reset_usage()
        print("计量已清零")
        return 0

    usage = await get_usage()
    if not usage:
        print("尚无计量记录（Redis 不可用或还没有调用过模型）")
        return 0

    total = usage.pop("total_cny", 0.0)
    # 6 位小数：embedding 一次全量重建约 0.04 元，单篇增量约 0.0001 元，
    # 用 4 位小数会把最需要被看见的量显示成 0.0000
    print("累计预估花费: 元%.6f" % total)
    print()
    print(f"{'模型':<32}{'输入':>12}{'输出':>10}{'缓存命中':>12}")

    # 先按模型归组再打印。直接遍历扁平 hash 会串列 ——
    # embedding 只有 in 字段，没有 out，逐行拼接收尾符的写法会错位。
    grouped: dict[str, dict[str, float]] = {}
    for field_name, value in usage.items():
        model, kind = field_name.rsplit(":", 1)
        grouped.setdefault(model, {})[kind] = value

    for model, kinds in sorted(grouped.items()):
        print(
            f"{model:<32}"
            f"{int(kinds.get('in', 0)):>12}"
            f"{int(kinds.get('out', 0)):>10}"
            f"{int(kinds.get('cached', 0)):>12}"
        )

    print()
    limit = BUDGET_TOTAL_CNY
    if limit > 0:
        pct = 100 * total / limit
        print(f"预算上限 元{limit:.2f}，已用 {pct:.1f}%")
        if total > limit:
            print("已超预算，AI 接口会拒绝继续调用")
    else:
        print("未设置预算上限（AI_BUDGET_TOTAL_CNY）。生产环境建议设置。")
    print(f"单次问答上限 元{BUDGET_ASK_CNY:.4f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 离线索引管理")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="语料 JSON 路径")
    common.add_argument("--from-db", action="store_true", help="从数据库读语料而非本地快照")

    p_validate = sub.add_parser("validate", parents=[common], help="只做语料校验")
    p_validate.add_argument("--strict", action="store_true", help="有 error 时返回非零退出码")
    p_validate.set_defaults(func=lambda args: asyncio.run(cmd_validate(args)))

    p_build = sub.add_parser("build", parents=[common], help="构建/增量更新索引")
    p_build.add_argument("--model", default="", help="embedding 模型名（必填）")
    p_build.add_argument("--full", action="store_true", help="忽略已有 manifest，强制全量重建")
    p_build.add_argument("--no-resume", action="store_true", help="不读取上次进度")
    p_build.add_argument("--strict", action="store_true", help="校验失败则中止构建")
    p_build.set_defaults(func=lambda args: asyncio.run(cmd_build(args)))

    p_status = sub.add_parser("status", help="查看当前 manifest")
    p_status.add_argument("--model", default="", help="要校验兼容性的模型名")
    p_status.set_defaults(func=lambda args: asyncio.run(cmd_status(args)))

    p_snapshots = sub.add_parser("snapshots", help="列出历史索引版本")
    p_snapshots.set_defaults(func=cmd_snapshots)

    p_index_status = sub.add_parser("index-status", help="查看离线索引落盘状态")
    p_index_status.add_argument("--model", default="", help="要校验兼容性的模型名")
    p_index_status.set_defaults(func=lambda args: asyncio.run(cmd_index_status(args)))

    p_clear = sub.add_parser("clear", help="清空离线索引")
    p_clear.set_defaults(func=cmd_clear)

    p_cost = sub.add_parser("cost", help="查看累计 token 用量与预估花费")
    p_cost.add_argument(
        "--reset", action="store_true", help="清零计量，重新开始统计"
    )
    p_cost.set_defaults(func=lambda args: asyncio.run(cmd_cost(args)))

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())