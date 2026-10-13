"""
成本闸门。

**为什么需要这个模块**：embedding 和问答都走按 token 计费的托管 API。
代码里一旦出现「循环里忘了 await」「重试没上限」「评估脚本连跑十遍」，
账单上只会看到一个安静上涨的数字，等发现时已经花掉了。
这个模块把 token 用量和预估花费变成可观测、可设上限的东西。

设计取舍
--------
1. **价格表可被环境变量覆盖。** 官方调价或换模型时不改代码，
   `AI_PRICE_qwen3.7-text-embedding=0.5`（单位：元/百万 token）即可。
2. **Redis 不可用时 fail-open（只告警不阻断）。**
   Redis 抖动不应该让整个 RAG 接口挂掉 —— 成本治理不能比业务本身更脆弱。
   但token 计数会丢失，所以告警日志里会明确说明「本次未能计量」。
3. **未登记价格的模型按 0 计价但仍然计 token。**
   这样「有调用发生」不会因为价格表缺项而不可见，
   同时不会因为猜错价格而给出虚假的花费数字。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_USAGE_KEY = "ai:cost:usage"

# 元 / 百万 token。来源见 README「模型与成本」一节。
# 可用环境变量 AI_PRICE_<模型名> 覆盖。
_DEFAULT_PRICES: dict[str, dict[str, float]] = {
    # embedding 只按输入计费
    "qwen3.7-text-embedding": {"input": 0.5},
    "qwen3.7-text-embedding-flash": {"input": 0.125},
    "text-embedding-v4": {"input": 0.5},
    # 聊天模型：输入 / 输出 / 缓存命中输入
    "deepseek-v4-flash-0731": {"input": 1.0, "output": 2.0, "cached_input": 0.2},
    "deepseek-v4-flash": {"input": 1.0, "output": 2.0, "cached_input": 0.2},
    "qwen-plus": {"input": 0.8, "output": 2.0, "cached_input": 0.1},
}

_WARNED_UNKNOWN: set[str] = set()


class BudgetExceeded(RuntimeError):
    """累计花费超出预算上限。调用方应拒绝服务而不是继续烧钱。"""


@dataclass
class CostSnapshot:
    """一次调用产生的用量。"""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost_cny: float = 0.0
    # 缓存命中占输入的比例，用于验证 prompt 缓存是否真的生效
    cache_hit_ratio: float = field(default=0.0)

    def describe(self) -> str:
        parts = [f"{self.model}", f"in={self.input_tokens}"]
        if self.cached_input_tokens:
            parts.append(f"cached={self.cached_input_tokens}")
        if self.output_tokens:
            parts.append(f"out={self.output_tokens}")
        if self.cost_cny:
            parts.append(f"元{self.cost_cny:.6f}")
        return " ".join(parts)


def _prices_for(model: str) -> dict[str, float]:
    """取某模型的价格，优先用环境变量覆盖。"""
    env_key = f"AI_PRICE_{model}"
    override = os.getenv(env_key)
    if override is not None:
        try:
            return {"input": float(override)}
        except ValueError:
            logger.warning("%s 不是合法数字，忽略该覆盖", env_key)

    prices = _DEFAULT_PRICES.get(model)
    if prices is None:
        if model not in _WARNED_UNKNOWN:
            _WARNED_UNKNOWN.add(model)
            logger.warning(
                "模型 %s 未登记价格，无法估算花费（token 仍会计数）。"
                "登记方式：环境变量 AI_PRICE_%s=元/百万token",
                model,
                model,
            )
        return {"input": 0.0, "output": 0.0, "cached_input": 0.0}
    return prices


def estimate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
) -> float:
    """
    预估花费（元）。

    缓存命中的输入按缓存价计费，从普通输入里扣除 —— 不能重复计费。
    """
    prices = _prices_for(model)
    billable_input = max(0, input_tokens - cached_input_tokens)
    cost = (
        billable_input * prices.get("input", 0.0)
        + output_tokens * prices.get("output", 0.0)
        + cached_input_tokens * prices.get("cached_input", prices.get("input", 0.0))
    ) / 1_000_000
    return round(cost, 8)


def _budget_total() -> float:
    return float(os.getenv("AI_BUDGET_TOTAL_CNY", "0") or 0)


def _budget_single_request() -> float:
    """单次问答的花费上限，防止异常长上下文把一次调用烧穿。"""
    return float(os.getenv("AI_BUDGET_ASK_CNY", "0.05") or 0)


def _warn_ratio() -> float:
    return float(os.getenv("AI_BUDGET_WARN_RATIO", "0.8") or 0.8)


def check_single_request(cost_cny: float) -> None:
    """单次调用闸门。同步检查，不需要 Redis。"""
    limit = _budget_single_request()
    if limit > 0 and cost_cny > limit:
        raise BudgetExceeded(
            f"单次请求预估花费 元{cost_cny:.6f} 超过上限 元{limit}，已拒绝。"
            "上下文可能异常变长，或模型被改成了更贵的档位。"
        )


async def record_usage(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
) -> CostSnapshot:
    """
    记录一次调用的 token 用量，累计到 Redis，返回本次快照。

    累计后立即检查总预算：超了就抛 `BudgetExceeded`，
    让调用方在**已经发生的这次调用之后**停止后续调用。
    事前拦截由 `check_single_request` 负责。
    """
    cost = estimate_cost(model, input_tokens, output_tokens, cached_input_tokens)

    # 单次闸门必须在碰 Redis 之前跑。
    # 累计预算依赖 Redis 里的历史累计值，Redis 不可用时会走下面的 fail-open
    # 分支直接返回 —— 那样连累计预算都不会检查，成本治理等于形同虚设。
    # 单次预算是纯计算，不依赖任何外部状态，是唯一在 Redis 挂掉时仍然有效的闸。
    check_single_request(cost)

    ratio = cached_input_tokens / input_tokens if input_tokens else 0.0
    snap = CostSnapshot(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        cost_cny=cost,
        cache_hit_ratio=ratio,
    )

    try:
        from config.cache_conf import redis_client

        # HINCRBYFLOAT 是原子的，多个 worker 并发累计不会丢账
        await redis_client.hincrbyfloat(_USAGE_KEY, "total_cny", cost)
        await redis_client.hincrby(_USAGE_KEY, f"{model}:in", input_tokens)
        if output_tokens:
            await redis_client.hincrby(_USAGE_KEY, f"{model}:out", output_tokens)
        if cached_input_tokens:
            await redis_client.hincrby(_USAGE_KEY, f"{model}:cached", cached_input_tokens)

        total = float(await redis_client.hget(_USAGE_KEY, "total_cny") or 0.0)
    except Exception as exc:
        # fail-open：计量失败只告警，不能让成本治理把业务打挂。
        # 已知局限 —— 此时总预算不会被检查（没有累计值可比）。
        # 单次请求预算仍然有效，见上面的 check_single_request。
        logger.warning("成本计量失败，本次用量未能累计（调用本身已成功）: %s", exc)
        return snap

    limit = _budget_total()
    if limit > 0 and total > limit:
        raise BudgetExceeded(
            f"累计花费 元{total:.4f} 已超过预算上限 元{limit}。"
            "请清空计量（ai.ingest.cli cost --reset）或调高 AI_BUDGET_TOTAL_CNY。"
        )
    if limit > 0 and total > limit * _warn_ratio():
        logger.warning(
            "累计花费 元%.4f，已达预算 元%.2f 的 %.0f%%",
            total,
            limit,
            100 * total / limit,
        )
    return snap


async def get_usage() -> dict[str, float]:
    """读取累计用量。Redis 不可用时返回空字典而不是抛错。"""
    try:
        from config.cache_conf import redis_client

        raw = await redis_client.hgetall(_USAGE_KEY)
        return {k: float(v) for k, v in (raw or {}).items()}
    except Exception as exc:
        logger.warning("读取累计用量失败: %s", exc)
        return {}


async def reset_usage() -> None:
    """清空计量。测试和重新开始一轮成本统计时用。"""
    try:
        from config.cache_conf import redis_client

        await redis_client.delete(_USAGE_KEY)
    except Exception as exc:
        logger.warning("清空用量失败: %s", exc)


async def record_chat_usage(model: str, usage: dict | None) -> None:
    """
    把一次大模型调用的 token 用量计入成本闸门。

    抽出来是因为**所有**直连大模型的入口都必须走这里：
    `/api/ai/news-qa`（同步 + 流式）和 `/api/ai/chat` 都会真实产生费用，
    任何一个入口绕过计量，成本就会少算一段，而账单还在照扣。

    DeepSeek 系模型普通输入 元1/M 是缓存命中价 元0.2/M 的 5 倍，
    而 system prompt 每次请求都完全相同 —— 缓存命中率直接决定成本。
    所以把 cached_tokens 单独记下来，缓存没生效时能立刻看出来。

    实测结论：百炼隐式缓存要求前缀 >= 1024 token（详见 ai/config.py）。
    news-qa 真实请求的前缀（system 约 200 token + 检索上下文约 700 token）
    低于该门槛，cached_tokens 恒为 0，日志里缓存命中率一直是 0% 属于预期，
    不是配置错误 —— 别再花时间调这个。

    预算超限时抛 BudgetExceeded，由调用方决定映射成什么状态码
    （news_qa 返回 429 告诉调用方是预算问题而非服务故障）。
    """
    if not usage:
        return

    details = usage.get("prompt_tokens_details") or {}
    try:
        snap = await record_usage(
            model,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            cached_input_tokens=int(details.get("cached_tokens") or 0),
        )
    except BudgetExceeded as exc:
        # 已超预算。这次调用的钱已经付了，但继续服务只会越欠越多。
        logger.error("预算超限: %s", exc)
        raise

    logger.info(
        "%s 计量 %s 缓存命中率 %.0f%%",
        model,
        snap.describe(),
        snap.cache_hit_ratio * 100,
    )