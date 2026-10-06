"""
索引清单（manifest）：回答「这份索引还能不能用」。

为什么需要
----------
向量缓存里存着一堆数字，但没人知道它们是什么模型、什么维度、对应哪个版本的语料。
出问题时会出现这种情形：换了 embedding 模型，缓存里旧向量维度对不上，
只有等到检索时报 "index out of range" 才发现，而且此时已经不知道哪些数据是新的。

manifest 把这些元信息显式记录下来：
- 换了模型/维度 -> 立即判定失效，不用等检索报错
- 语料变了 -> 判定为 stale，可以增量更新而不是全量重建
- 支持按 manifest 内容回滚到上一个可用版本

存放位置：Redis（跟随运行时缓存）+ 本地 JSON 快照（便于离线检查与版本管理）。
"""
import io
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1
MANIFEST_KEY = "ai:index:manifest"
SNAPSHOT_DIR = Path(__file__).with_name("snapshots")


@dataclass
class IndexManifest:
    """
    一份索引的身份证明。

    字段设计上有个刻意的取舍：只记录「影响向量可用性」的信息。
    不记录向量本身 —— 向量有几百 KB 到几 MB，放进 manifest 会让它难以读取和比较。
    """

    manifest_version: int = MANIFEST_VERSION
    # 索引唯一标识 = 模型 + 维度 + 语料指纹 的摘要
    index_id: str = ""
    embed_model: str = ""
    embed_dim: int = 0
    # 语料层面的指纹
    corpus_fingerprint: str = ""
    document_count: int = 0
    # 内容 hash -> [news_id, ...] 的映射，增量更新靠它跳过未变更文档。
    # 值是列表而非单个 id：真实数据存在完全重复的新闻（15 组 33 篇），
    # 用单 id 会让同组内后一条覆盖前一条，那些文档就永远进不了索引。
    doc_hashes: dict[str, list[int]] = field(default_factory=dict)
    built_at: str = ""
    build_duration_ms: float = 0.0
    # 统计信息，便于对比不同版本
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "IndexManifest":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    # ---------------- 兼容性判定 ----------------

    def is_compatible_with(self, model: str, dim: int) -> tuple[bool, str]:
        """
        判断当前索引能否用于给定的 embedding 模型。

        返回 (是否兼容, 原因)。原因用于日志，避免排查时看到无信息量的 False。
        """
        if self.embed_model != model:
            return False, f"模型不匹配：索引={self.embed_model} 当前={model}"
        if self.embed_dim and dim and self.embed_dim != dim:
            return False, f"维度不匹配：索引={self.embed_dim} 当前={dim}"
        return True, "兼容"

    def is_stale_for(self, corpus_fingerprint: str) -> bool:
        """语料是否已经变化"""
        return self.corpus_fingerprint != corpus_fingerprint

    def changed_docs(self, current_hashes: dict[str, list[int]]) -> set[int]:
        """
        相比上次构建，哪些文档的内容变了。

        返回受影响的 news_id 集合（新增 + 内容变化 + id 重分配）。

        逐个 hash 比对而不是取差集，是因为映射的值可能是列表
        （内容完全相同的重复文档），且同一 id 可能出现在不同 hash 下。
        """
        changed: set[int] = set()
        for content_hash, news_ids in current_hashes.items():
            previous_ids = set(self.doc_hashes.get(content_hash, []))
            for news_id in news_ids:
                if news_id not in previous_ids:
                    changed.add(news_id)
        return changed

    def removed_docs(self, current_hashes: dict[str, list[int]]) -> list[int]:
        """本次语料里已经消失的文档 id（需要从索引里清理）"""
        current_ids = {i for ids in current_hashes.values() for i in ids}
        previous_ids = {i for ids in self.doc_hashes.values() for i in ids}
        return sorted(previous_ids - current_ids)


# ---------------------------------------------------------------- 持久化


async def save_manifest(manifest: IndexManifest, snapshot: bool = True) -> None:
    """写 Redis，并可选写本地 JSON 快照"""
    payload = json.dumps(manifest.to_dict(), ensure_ascii=False)

    from config.cache_conf import redis_client

    try:
        await redis_client.set(MANIFEST_KEY, payload)
    except Exception as exc:
        from utils.logging_conf import get_logger

        get_logger(__name__).warning("写 manifest 到 Redis 失败: %s", exc)

    if snapshot:
        save_snapshot(manifest)


def save_snapshot(manifest: IndexManifest) -> Path:
    """
    写本地快照。

    存在的意义：Redis 挂了或数据被清空时，仍能从快照判断
    「之前建的索引是什么配置」，而不是完全失去线索。
    文件名带 index_id 前 8 位便于按版本查找。
    """
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"manifest-{manifest.index_id[:8] or 'unknown'}.json"
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)
    return path


async def load_manifest() -> IndexManifest | None:
    """
    读取 manifest。

    读取顺序：Redis -> 本地文件 -> 历史快照。
    Redis 是快路径但不可靠（可被清空、故障），文件才是真相来源。
    """
    from config.cache_conf import redis_client

    try:
        raw = await redis_client.get(MANIFEST_KEY)
        if raw:
            return IndexManifest.from_dict(json.loads(raw))
    except Exception as exc:
        from utils.logging_conf import get_logger

        get_logger(__name__).debug("从 Redis 读 manifest 失败: %s", exc)

    from ai.ingest.storage import load_manifest_file

    data = load_manifest_file()
    if data:
        return IndexManifest.from_dict(data)

    latest = latest_snapshot()
    if latest:
        return IndexManifest.from_dict(json.loads(io.open(latest, encoding="utf-8").read()))
    return None


def latest_snapshot() -> Path | None:
    if not SNAPSHOT_DIR.exists():
        return None
    files = sorted(SNAPSHOT_DIR.glob("manifest-*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def list_snapshots() -> list[Path]:
    """列出所有历史快照，可用于回滚到指定版本"""
    if not SNAPSHOT_DIR.exists():
        return []
    return sorted(SNAPSHOT_DIR.glob("manifest-*.json"))


def make_index_id(model: str, dim: int, corpus_fingerprint: str) -> str:
    """索引 id = 模型与语料的联合摘要，换任一项都会得到不同的 id"""
    import hashlib

    raw = f"{model}|{dim}|{corpus_fingerprint}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")