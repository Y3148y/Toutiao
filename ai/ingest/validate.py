"""
语料校验。

为什么在建索引前校验，而不是等检索时才发现
----------------------------------------
检索时才发现脏数据，症状是「某条新闻永远召不到」或「报一个看不懂的错」，
排查成本极高。建索引时拦住并给出明确报告，问题在写入侧就暴露。

注意：语料校验默认**只警告不拦截**。丢文档会让召回率无声下降，
比报错更危险 —— 除非显式开启严格模式。
"""
import hashlib
from dataclasses import dataclass, field
from datetime import datetime

# 单篇文档进入 embedding 的长度上限。
# 超过这个长度说明内容异常（可能把整页 HTML 存进了正文），
# 直接送进 embedding 会浪费 token 且结果没有意义。
MAX_CONTENT_CHARS = 4000

# 标题长度下限，低于这个值大概率是数据缺失
MIN_TITLE_CHARS = 2


@dataclass
class DocumentIssue:
    doc_id: int
    level: str  # error / warning
    code: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.level}] doc#{self.doc_id} {self.code}: {self.detail}"


@dataclass
class ValidationReport:
    issues: list[DocumentIssue] = field(default_factory=list)
    total: int = 0
    valid: int = 0
    skipped: int = 0

    @property
    def errors(self) -> list[DocumentIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[DocumentIssue]:
        return [i for i in self.issues if i.level == "warning"]

    def has_errors(self) -> bool:
        return bool(self.errors)

    def summary(self) -> str:
        parts = [
            f"共 {self.total} 篇，合格 {self.valid} 篇，跳过 {self.skipped} 篇",
        ]
        if self.errors:
            parts.append(f"错误 {len(self.errors)} 项")
        if self.warnings:
            parts.append(f"警告 {len(self.warnings)} 项")
        return "，".join(parts)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "valid": self.valid,
            "skipped": self.skipped,
            "errorCount": len(self.errors),
            "warningCount": len(self.warnings),
            "issues": [
                {
                    "docId": i.doc_id,
                    "level": i.level,
                    "code": i.code,
                    "detail": i.detail,
                }
                for i in self.issues[:200]  # 报告里最多列200条，避免刷屏
            ],
        }


def document_content_hash(title: str | None, description: str | None, content: str | None) -> str:
    """
    文档内容指纹，用于增量更新。

    只哈希真正参与检索的字段 —— category_id 变了不影响检索文本，
    不该触发重新向量化（那是浪费 embedding 调用）。
    """
    raw = "\x1f".join([title or "", description or "", content or ""])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_content_hash_index(docs: list) -> dict[str, list[int]]:
    """
    建立 内容指纹 -> [news_id, ...] 的映射。

    为什么值是列表而不是单个 id
    --------------------------
    真实数据里存在标题、描述、正文完全相同的重复新闻
    （本项目 403 篇中有 15 组、共 33 篇，如 id 22 与 116）。
    如果用 hash -> 单个 id 做映射，后写入的会覆盖先写入的，
    导致这些文档永远无法进入索引 —— 而且没有任何报错，
    索引少了几十条数据却看不出来。这是实测踩到的问题。

    重复文档的向量是完全相同的（文本一样），因此只需向量化一次，
    按列表里所有 id 复用同一份向量即可。
    """
    index: dict[str, list[int]] = {}
    for news in docs:
        content_hash = document_content_hash(news.title, news.description, news.content)
        index.setdefault(content_hash, []).append(news.id)
    return index


def find_duplicate_groups(docs: list) -> dict[str, list[int]]:
    """找出内容完全重复的文档组，供校验报告使用"""
    index = build_content_hash_index(docs)
    return {h: ids for h, ids in index.items() if len(ids) > 1}


def validate_document(news, max_content_chars: int = MAX_CONTENT_CHARS) -> list[DocumentIssue]:
    """校验单篇文档，返回问题列表（空列表表示合格）"""
    issues: list[DocumentIssue] = []
    doc_id = news.id

    title = (news.title or "").strip()
    content = (news.content or "").strip()

    if not title:
        issues.append(DocumentIssue(doc_id, "error", "EMPTY_TITLE", "标题为空"))
    elif len(title) < MIN_TITLE_CHARS:
        issues.append(
            DocumentIssue(doc_id, "warning", "TITLE_TOO_SHORT", f"标题仅 {len(title)} 字")
        )

    if len(title) > 200:
        issues.append(
            DocumentIssue(doc_id, "warning", "TITLE_TOO_LONG", f"标题 {len(title)} 字，建议截断")
        )

    if not content:
        issues.append(DocumentIssue(doc_id, "error", "EMPTY_CONTENT", "正文为空，无法检索"))
    elif len(content) > max_content_chars:
        issues.append(
            DocumentIssue(
                doc_id, "error", "CONTENT_TOO_LONG",
                f"正文 {len(content)} 字，超过上限 {max_content_chars}，疑似脏数据",
            )
        )

    if news.category_id is None:
        issues.append(DocumentIssue(doc_id, "warning", "NO_CATEGORY", "缺少分类"))

    return issues


def validate_corpus(
    docs: list,
    strict: bool = False,
    max_content_chars: int = MAX_CONTENT_CHARS,
) -> tuple[list, ValidationReport]:
    """
    校验整个语料。

    返回 (合格文档列表, 报告)。

    strict=False（默认）：有 error 的文档被跳过但流程继续。
        理由是丢文档会让召回率无声下降，而报错会中断整个摄入，
        对 403 条里坏 2 条的场景来说，后者代价更大。
    strict=True：有任何 error 就抛异常，适合 CI 或数据管道。

    有 description 但正文为空时不算 error —— description 本身可作为检索文本，
    但这种情况在当前语料里不存在，出现时提示一下即可。
    """
    report = ValidationReport(total=len(docs))
    accepted = []

    for news in docs:
        issues = validate_document(news, max_content_chars)
        if not issues:
            accepted.append(news)
            report.valid += 1
            continue

        report.issues.extend(issues)
        has_error = any(i.level == "error" for i in issues)
        if has_error:
            report.skipped += 1
            if not strict:
                # 有 description 时仍可用于检索
                if (news.description or "").strip():
                    accepted.append(news)
                    report.valid += 1
                    report.skipped -= 1
        else:
            # 只有 warning 的照常收录
            accepted.append(news)
            report.valid += 1

    if strict and report.has_errors():
        raise ValueError(
            f"语料校验失败（strict 模式）：{report.summary()}\n"
            + "\n".join(str(i) for i in report.errors[:20])
        )

    return accepted, report


def corpus_fingerprint(docs: list) -> str:
    """
    整个语料的指纹：文档 id + 内容 hash 的有序摘要。

    有序是为了让「新增一篇文档」和「顺序变化」都能被识别出来。
    """
    parts = sorted(f"{n.id}:{document_content_hash(n.title, n.description, n.content)[:16]}" for n in docs)
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return f"{len(docs)}:{digest[:32]}"