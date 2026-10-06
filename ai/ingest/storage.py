"""
索引产物的持久化层。

为什么不用 Redis 而以文件为主
----------------------------
最初实现把 manifest 和向量矩阵都写 Redis，测试时立刻暴露出问题：
Redis 不可用时，manifest 存不进去 -> 下次构建读不到历史 -> 增量判定失效 ->
403 篇全部重新向量化（浪费付费 API 调用）。

Redis 作为缓存可以丢，**索引产物是数据不该只存在缓存里**。
所以改成：Redis 作为快路径，文件作为真相来源，Redis 读不到自动回落文件。

向量存成 .npy 而不是 JSON
------------------------
403×1024 维 float32 = 1.6MB。JSON 序列化后约 8MB 且耗时几百毫秒。
numpy 的二进制格式紧凑且零解析开销，用 np.load 直接映射到内存。
"""
import io
import json
from pathlib import Path

import numpy as np

INDEX_DIR = Path(__file__).with_name("index")
MANIFEST_FILE = INDEX_DIR / "manifest.json"
VECTORS_FILE = INDEX_DIR / "vectors.npy"
IDS_FILE = INDEX_DIR / "news_ids.json"


def ensure_dir() -> Path:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return INDEX_DIR


# ---------------------------------------------------------------- manifest


def save_manifest_file(manifest) -> Path:
    ensure_dir()
    with io.open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)
    return MANIFEST_FILE


def load_manifest_file():
    if not MANIFEST_FILE.exists():
        return None
    try:
        return json.load(io.open(MANIFEST_FILE, encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------- 向量矩阵


def save_matrix_file(matrix, news_ids: list[int], fingerprint: str) -> None:
    """
    向量存 .npy，id 列表与指纹存 JSON。

    分开存的原因：npy 必须是连续的二进制数组，没法掺 JSON 元数据；
    而 id 列表是极小的 JSON，放在旁边更便于人工检查顺序是否正确。
    """
    ensure_dir()
    np.save(VECTORS_FILE, matrix.to_numpy(), allow_pickle=False)
    with io.open(IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"newsIds": news_ids, "fingerprint": fingerprint, "count": len(news_ids)},
            f,
            ensure_ascii=False,
        )


def load_matrix_file():
    """返回 (VectorMatrix, news_ids, fingerprint)；文件缺失返回 None"""
    if not VECTORS_FILE.exists() or not IDS_FILE.exists():
        return None
    try:
        meta = json.load(io.open(IDS_FILE, encoding="utf-8"))
        array = np.load(VECTORS_FILE, allow_pickle=False)
        from ai.embeddings import VectorMatrix

        matrix = VectorMatrix(array)
        news_ids = meta["newsIds"]
        if len(matrix) != len(news_ids):
            # 数量对不上说明文件被部分覆盖过，用它会错位，必须丢弃
            return None
        return matrix, news_ids, meta.get("fingerprint", "")
    except Exception:
        return None


def index_exists() -> bool:
    return VECTORS_FILE.exists() and IDS_FILE.exists()


def clear_index() -> None:
    for path in (MANIFEST_FILE, VECTORS_FILE, IDS_FILE):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def index_stats() -> dict:
    """索引概览，用于 CLI status 和排查"""
    result = load_matrix_file()
    if result is None:
        return {"exists": False}
    matrix, news_ids, fingerprint = result
    manifest = load_manifest_file()
    return {
        "exists": True,
        "documentCount": len(matrix),
        "dim": matrix.dim,
        "vectorsFile": str(VECTORS_FILE),
        "vectorsSizeMB": round(VECTORS_FILE.stat().st_size / 1024 / 1024, 2),
        "fingerprint": fingerprint,
        "indexId": manifest.get("index_id") if manifest else None,
        "embedModel": manifest.get("embed_model") if manifest else None,
        "builtAt": manifest.get("built_at") if manifest else None,
    }