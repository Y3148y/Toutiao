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
DASHSCOPE_CHAT_MODEL = os.getenv("DASHSCOPE_MODEL", "deepseek-v4-flash-0731")
DASHSCOPE_EMBED_MODEL = os.getenv(
    "DASHSCOPE_EMBED_MODEL", "qwen3.7-text-embedding"
)

# 向量维度。qwen3.7-text-embedding 支持 256/512/768/1024/1536/2048/2560，
# 换维度会换掉整套向量空间，所以它必须和模型名一样进缓存 key 和索引 manifest，
# 否则同一模型不同维度的向量会被当成同一种东西复用。
# 512 维时 403 篇的 .npy 从 1.57 MB 降到约 393 KB，检索精度在这个规模上无明显损失。
DASHSCOPE_EMBED_DIM = int(os.getenv("DASHSCOPE_EMBED_DIM", "1024"))

# ---------------- 成本闸门 ----------------

# 累计花费上限（元）。0 = 不限制。超出后 AI 接口会拒绝继续调用。
# 生产环境务必设置 —— 代码里一次循环失误就能把账单刷上去。
BUDGET_TOTAL_CNY = float(os.getenv("AI_BUDGET_TOTAL_CNY", "0") or 0)

# 单次问答的花费上限（元），拦住异常长上下文导致的一次性大额调用
BUDGET_ASK_CNY = float(os.getenv("AI_BUDGET_ASK_CNY", "0.05") or 0)

# 达到预算的这个比例时打告警日志
BUDGET_WARN_RATIO = float(os.getenv("AI_BUDGET_WARN_RATIO", "0.8") or 0.8)

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

# 召回置信度阈值。**这是次要门控**，见下方 MIN_VECTOR_SIM 的说明。
MIN_FUSION_SCORE = float(os.getenv("AI_MIN_FUSION_SCORE", "0.02"))

# 向量 top-1 余弦相似度阈值 —— **主要的拒答判据**。
#
# 为什么不用 RRF 融合分做主要判据：实测（55 条评估集，真实 qwen3.7-text-embedding）
#   正例 top1 余弦：中位 0.8037，最低 0.5143
#   负例 top1 余弦：中位 0.4809，最高 0.6211
#   -> 有区分度，0.55 附近可把负例拒答率做到 83% 而正例作答率仍有 94%
#
# 而 RRF 融合分：
#   正例 0.0313~0.0328（中位 0.0328）
#   负例 0.0311~0.0328（中位 0.0325）
#   -> 完全重叠。RRF 只用排名、丢掉分数幅度，403 篇的小语料里任何查询都能凑出
#      双路命中，分数一律顶到上限 2/(60+1)=0.0328。
#      这就是原来 MIN_FUSION_SCORE=0.02 形同虚设的原因：
#      负例误召回率实测 100%，一条都没拦住。
MIN_VECTOR_SIM = float(os.getenv("AI_MIN_VECTOR_SIM", "0.55"))

# 向量路不可用时的降级判据：BM25 top-1 分数下限。
#
# 降级路径必须另设判据，不能沿用 RRF 分数：向量路关闭时所有文档都是单路命中，
# 融合分一律等于 1/(60+1)×0.5 ≈ 0.0082，低于 MIN_FUSION_SCORE，
# 结果是**对每一个问题都拒答**（实测正例作答率 0%）。
# BM25 分数保留了词面匹配强度，有真实幅度，可以承担置信判断。
#
# 标定依据（55 条评估集，BM25 top-1）：
#   正例 min 26.63 / 中位 87.14 / max 149.54
#   负例 min 22.65 / 中位 47.90 / max  60.36
# 取 40 时负例拒答 50%、正例作答 96%。
MIN_BM25_SCORE = float(os.getenv("AI_MIN_BM25_SCORE", "40.0"))

# 每条新闻最多取多少字进上下文（新闻正文可能很长）
MAX_DOC_CHARS = int(os.getenv("AI_MAX_DOC_CHARS", "400"))

# 语料索引缓存时间。语料变了就等 TTL 到期或手动清 key
INDEX_CACHE_TTL = int(os.getenv("AI_INDEX_CACHE_TTL", "3600"))
EMBEDDING_CACHE_TTL = int(os.getenv("AI_EMBEDDING_CACHE_TTL", "86400"))

# 检索链路 trace：默认写结构化日志；开启后额外按 request_id 落到 Redis，
# 便于事后回查「为什么这次检索返回了这个结果」。默认关闭是因为它有额外写入开销。
TRACE_ENABLED = os.getenv("AI_TRACE_ENABLED", "false").lower() == "true"
TRACE_TTL = int(os.getenv("AI_TRACE_TTL", "3600"))


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