"""
成本闸门测试。

这些用例的价值在于锁住三件事：
1. 价格换算算得对（算错价会让整个预算体系失去意义）
2. 超额真的会被拦住（而不是只打个日志）
3. Redis 挂掉时闸门的行为是**已知且明确**的，而不是碰运气
"""
import asyncio
import os

import pytest

from ai import cost


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "AI_BUDGET_TOTAL_CNY",
        "AI_BUDGET_ASK_CNY",
        "AI_BUDGET_WARN_RATIO",
    ):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------- 价格换算


def test_embedding_price_per_million_tokens():
    assert cost.estimate_cost("qwen3.7-text-embedding", input_tokens=1_000_000) == 0.5


def test_chat_price_separates_input_and_output():
    got = cost.estimate_cost(
        "deepseek-v4-flash-0731", input_tokens=1_000_000, output_tokens=1_000_000
    )
    assert got == pytest.approx(3.0)  # 输入 1 + 输出 2


def test_cached_input_is_cheaper_than_normal_input():
    normal = cost.estimate_cost("deepseek-v4-flash-0731", input_tokens=1_000_000)
    cached = cost.estimate_cost(
        "deepseek-v4-flash-0731",
        input_tokens=1_000_000,
        cached_input_tokens=1_000_000,
    )
    assert cached == pytest.approx(0.2)
    assert cached < normal


def test_cached_tokens_are_not_double_billed():
    """
    缓存命中的 token 已经在 input 里报过一次了。
    如果计算时既按 input 又按 cached_input 各算一次，
    实际账单会是 1.2 元而我们记成 1.2 元 —— 看似对，但对「部分命中」就错了。
    """
    half = cost.estimate_cost(
        "deepseek-v4-flash-0731",
        input_tokens=1_000_000,
        cached_input_tokens=500_000,
    )
    # 50万普通输入(1.0) + 50万缓存输入(0.2) = 0.6
    assert half == pytest.approx(0.6)


def test_unknown_model_counts_tokens_but_costs_nothing():
    """
    价格表缺项时不能报错（那会让新模型直接跑不起来），
    但也不能悄悄不计量 —— token 仍然要能看到。
    """
    assert cost.estimate_cost("brand-new-model", input_tokens=1_000_000) == 0.0


def test_price_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("AI_PRICE_my-model", "2.5")
    assert cost.estimate_cost("my-model", input_tokens=1_000_000) == 2.5


# ---------------------------------------------------------------- 闸门行为


def test_single_request_gate_blocks_overspend(monkeypatch):
    monkeypatch.setenv("AI_BUDGET_ASK_CNY", "0.001")
    with pytest.raises(cost.BudgetExceeded):
        cost.check_single_request(0.005)


def test_single_request_gate_allows_normal_call(monkeypatch):
    monkeypatch.setenv("AI_BUDGET_ASK_CNY", "0.05")
    cost.check_single_request(0.002)  # 不应抛错


def test_single_request_gate_defaults_to_a_real_limit():
    """
    默认必须有闸（0.05 元/次）。默认关闭等于没有成本治理，
    而大多数部署根本不会记得去设置这个环境变量。
    """
    assert cost._budget_single_request() == pytest.approx(0.05)


def test_gate_is_disabled_when_limit_is_zero(monkeypatch):
    monkeypatch.setenv("AI_BUDGET_ASK_CNY", "0")
    cost.check_single_request(999.0)  # 显式关闭时不应拦截


def test_single_request_gate_works_without_redis(monkeypatch):
    """
    回归测试：单次闸门曾经被放在 Redis 操作之后，
    Redis 一抖动就走进 fail-open 分支，连单次预算都不检查了。
    """
    monkeypatch.setenv("AI_BUDGET_ASK_CNY", "0.001")
    monkeypatch.setenv("REDIS_PORT", "6399")  # 指向不存在的 Redis

    async def run():
        return await cost.record_usage(
            "deepseek-v4-flash-0731", input_tokens=1_000_000, output_tokens=1_000_000
        )

    with pytest.raises(cost.BudgetExceeded):
        asyncio.run(run())


def test_cache_hit_ratio_is_reported():
    snap = cost.CostSnapshot(
        model="m", input_tokens=1000, cached_input_tokens=250, cache_hit_ratio=0.25
    )
    assert snap.cache_hit_ratio == 0.25
    assert "cached=250" in snap.describe()


# ---------------------------------------------------------------- Redis 行为


def test_usage_fails_open_when_redis_down(monkeypatch):
    """
    Redis 不可用时不能抛错 —— 成本治理不该比业务本身更脆弱。
    但必须留下告警痕迹，否则这次调用等于没被计量，没人知道。
    """
    monkeypatch.setenv("REDIS_PORT", "6399")

    async def run():
        return await cost.record_usage("qwen3.7-text-embedding", input_tokens=1000)

    snap = asyncio.run(run())
    assert snap.input_tokens == 1000
    assert snap.cost_cny == pytest.approx(0.0005)


def test_get_usage_returns_empty_when_redis_down(monkeypatch):
    monkeypatch.setenv("REDIS_PORT", "6399")

    async def run():
        return await cost.get_usage()

    assert asyncio.run(run()) == {}