"""
环境变量解析的回归测试。

起因是一个只在容器里出现的问题：
docker-compose 用 `${VAR:-}` 注入空串，`int("")` 直接 ValueError，
容器起不来；而本地跑得好好的，因为本地根本没有这个环境变量。
"""
import os

import pytest


def test_empty_string_falls_back_to_default(monkeypatch):
    """空串必须按「未设置」处理，不能当值用"""
    from ai.config import _env_int, _env_str

    monkeypatch.setenv("AI_TEST_VAR", "")
    assert _env_str("AI_TEST_VAR", "fallback") == "fallback"
    assert _env_int("AI_TEST_VAR", 5) == 5


def test_whitespace_only_falls_back(monkeypatch):
    from ai.config import _env_int

    monkeypatch.setenv("AI_TEST_VAR", "   ")
    assert _env_int("AI_TEST_VAR", 7) == 7


def test_invalid_value_falls_back(monkeypatch):
    """非法值不该让进程崩掉，回退到默认值并能继续启动"""
    from ai.config import _env_float, _env_int

    monkeypatch.setenv("AI_TEST_VAR", "abc")
    assert _env_int("AI_TEST_VAR", 3) == 3
    assert _env_float("AI_TEST_VAR", 1.5) == 1.5


def test_valid_value_is_used(monkeypatch):
    from ai.config import _env_float, _env_int, _env_str

    monkeypatch.setenv("AI_TEST_VAR", "42")
    assert _env_str("AI_TEST_VAR", "x") == "42"
    assert _env_int("AI_TEST_VAR", 0) == 42
    monkeypatch.setenv("AI_TEST_VAR", "0.55")
    assert _env_float("AI_TEST_VAR", 0.0) == 0.55


def test_config_survives_compose_style_empty_env(monkeypatch):
    """
    复现真实故障：docker-compose 的 ${VAR:-} 会给容器注入一批空串。
    之前这里直接 int(os.getenv(...)) 导致 toutiao-app 启动即崩：
      ValueError: invalid literal for int() with base 10: ''
    """
    compose_vars = [
        "DASHSCOPE_MODEL",
        "DASHSCOPE_EMBED_MODEL",
        "DASHSCOPE_EMBED_DIM",
        "AI_TOP_K_BM25",
        "AI_TOP_K_VECTOR",
        "AI_TOP_K_FINAL",
        "AI_RRF_K",
        "AI_SINGLE_PATH_WEIGHT",
        "AI_MIN_VECTOR_SIM",
        "AI_MIN_BM25_SCORE",
        "AI_MIN_FUSION_SCORE",
        "AI_MAX_DOC_CHARS",
        "AI_INDEX_CACHE_TTL",
        "AI_EMBEDDING_CACHE_TTL",
        "AI_TRACE_ENABLED",
        "AI_TRACE_TTL",
        "AI_BUDGET_TOTAL_CNY",
        "AI_BUDGET_ASK_CNY",
        "AI_BUDGET_WARN_RATIO",
        "AI_REQUEST_TIMEOUT",
        "AI_STREAM_TIMEOUT",
    ]
    for name in compose_vars:
        monkeypatch.setenv(name, "")

    # 重新导入 ai.config 让它在空环境变量下解析一遍
    import importlib

    import ai.config as config_module

    importlib.reload(config_module)

    assert config_module.TOP_K_FINAL == 5
    assert config_module.RRF_K == 60
    assert config_module.DASHSCOPE_EMBED_DIM == 1024
    assert config_module.DASHSCOPE_CHAT_MODEL == "deepseek-v4-flash-0731"
    assert config_module.DASHSCOPE_EMBED_MODEL == "qwen3.7-text-embedding"
    assert config_module.MIN_VECTOR_SIM == 0.55

    # 恢复模块，避免影响后续用例
    monkeypatch.undo()
    importlib.reload(config_module)