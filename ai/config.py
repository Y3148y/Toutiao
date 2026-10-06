"""
AI 模块配置。

密钥只从服务端环境变量读取，绝不能出现在代码里或下发到前端浏览器。
未配置密钥时相关接口返回 500 并给出明确提示，而不是静默失败。
"""
import os

# OpenAI 兼容协议（阿里云百炼 DashScope）
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_CHAT_MODEL = os.getenv("DASHSCOPE_MODEL", "qwen-plus")
DASHSCOPE_EMBED_MODEL = os.getenv("DASHSCOPE_EMBED_MODEL", "text-embedding-v4")

# 单次请求超时。流式问答要设长一点，用户思考和生成都需要时间
REQUEST_TIMEOUT = float(os.getenv("AI_REQUEST_TIMEOUT", "60"))
STREAM_TIMEOUT = float(os.getenv("AI_STREAM_TIMEOUT", "300"))

# ---------------- 检索相关 ----------------

# 每路召回的候选条数，取并集后再融合
TOP_K_BM25 = int(os.getenv("AI_TOP_K_BM25", "20"))
TOP_K_VECTOR = int(os.getenv("AI_TOP_K_VECTOR", "20"))

# 最终喂给 LLM 的新闻条数。太多会撑爆上下文且噪声大，太少会丢关键信息
TOP_K_FINAL = int(os.getenv("AI_TOP_K_FINAL", "5"))

# RRF 融合的平滑常数。60 是论文里的经验值：抵消头部排名的过度影响
RRF_K = int(os.getenv("AI_RRF_K", "60"))

# 只有一路检索召回的文档，融合得分乘这个系数作为惩罚。
# 双路召回 = 两路都认为相关；单路召回 = 证据不足，不应被当成可靠答案的依据。
SINGLE_PATH_WEIGHT = float(os.getenv("AI_SINGLE_PATH_WEIGHT", "0.5"))

# 召回置信度阈值：融合得分低于此值认为「没有相关报道」，直接拒答不调 LLM。
# 取值应落在「单路最高分」和「双路最高分」之间：
#   单路最高≈ 0.5 / (60+1) ≈ 0.008
#   双路最高 ≈ 2 / (60+1)     ≈ 0.033
# 设成 0.02 意味着「两路都命中才算有把握」，单路命中会走拒答。
# 这是防幻觉的关键——宁可说不知道，也不要让模型编。
MIN_FUSION_SCORE = float(os.getenv("AI_MIN_FUSION_SCORE", "0.02"))

# 每条新闻最多取多少字进上下文（新闻正文可能很长）
MAX_DOC_CHARS = int(os.getenv("AI_MAX_DOC_CHARS", "400"))

# 语料索引缓存时间。语料变了就等 TTL 到期或手动清 key
INDEX_CACHE_TTL = int(os.getenv("AI_INDEX_CACHE_TTL", "3600"))
EMBEDDING_CACHE_TTL = int(os.getenv("AI_EMBEDDING_CACHE_TTL", "86400"))


def is_configured() -> bool:
    """是否配置了可用的 LLM 密钥"""
    return bool(DASHSCOPE_API_KEY)


def auth_headers(stream: bool = False) -> dict:
    """构造请求头。stream=True 时 DashScope 需要额外的 SSE 标记头"""
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    if stream:
        headers["X-DashScope-SSE"] = "enable"
    return headers