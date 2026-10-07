"""
路由自检。

单独放一个文件，是因为它被两处引用：
- GitHub Actions 的 smoke job
- tests/test_response.py

之前两边各自硬编码路由总数（22 -> 25），每加一个接口就要改两个地方，
漏改一个 CI 就红。已经因此失败过两次，不该再靠人肉同步。

用法：
    python -m scripts.check_routes
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 业务接口清单 + 期望的 HTTP 方法。
# 新增接口时改这里，测试和 CI 会一起生效。
REQUIRED_ROUTES: dict[str, tuple[str, ...]] = {
    # 新闻 6
    "/api/news/categories": ("get",),
    "/api/news/list": ("get",),
    "/api/news/detail": ("get",),
    "/api/news/feed": ("get",),
    "/api/news/search": ("get",),
    "/api/news/hot": ("get",),
    # 收藏 5
    "/api/favorite/check": ("get",),
    "/api/favorite/add": ("post",),
    "/api/favorite/remove": ("delete",),
    "/api/favorite/list": ("get",),
    "/api/favorite/clear": ("delete",),
    # 历史 4
    "/api/history/add": ("post",),
    "/api/history/list": ("get",),
    "/api/history/delete/{history_id}": ("delete",),
    "/api/history/clear": ("delete",),
    # 用户 5
    "/api/user/register": ("post",),
    "/api/user/login": ("post",),
    "/api/user/info": ("get",),
    "/api/user/update": ("put",),
    "/api/user/password": ("put",),
    # AI 4
    "/api/ai/chat": ("post",),
    "/api/ai/news-qa": ("post",),
    "/api/ai/news-qa/sync": ("post",),
    "/api/ai/news-qa/reload": ("post",),
    # 根路径
    "/": ("get",),
}

EXPECTED_COUNT = len(REQUIRED_ROUTES)


def check() -> tuple[int, list[str]]:
    """
    返回 (实际路由数, 问题列表)。

    同时校验两件事：
    1. 数量一致 —— 防止接口被误删
    2. 每个清单里的路由都存在且方法符合预期 —— 防止路由被重命名或改错方法

    只查数量不够：删一个再加一个，总数不变但功能已经变了。
    """
    from main import app

    paths = app.openapi()["paths"]
    problems: list[str] = []

    actual = len(paths)
    if actual != EXPECTED_COUNT:
        extra = set(paths) - set(REQUIRED_ROUTES)
        missing = set(REQUIRED_ROUTES) - set(paths)
        detail = []
        if missing:
            detail.append(f"缺失 {sorted(missing)}")
        if extra:
            detail.append(f"多出 {sorted(extra)}")
        problems.append(
            f"路由数不符：期望 {EXPECTED_COUNT}，实际 {actual}"
            + ("（" + "；".join(detail) + "）" if detail else "")
        )

    for route, methods in REQUIRED_ROUTES.items():
        if route not in paths:
            problems.append(f"缺少路由 {route}")
            continue
        got = set(paths[route])
        want = set(methods)
        if got != want:
            problems.append(f"{route} 方法不符：期望 {sorted(want)}，实际 {sorted(got)}")

    return actual, problems


def main() -> int:
    actual, problems = check()
    if problems:
        for p in problems:
            print(f"FAIL {p}")
        return 1
    print(f"OK: {actual} paths")
    return 0


if __name__ == "__main__":
    sys.exit(main())