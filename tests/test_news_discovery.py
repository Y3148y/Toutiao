"""
内容分发接口测试：首页信息流、搜索、热榜。

热度公式和分词是纯函数，直接单测；
接口层复用项目既有的 FakeSession 注入，不连真实 MySQL。
"""
from datetime import datetime, timedelta

import pytest

from crud.news_feed import hot_score
from models.news import Category, News
from tests.fakes import FakeResult

NOW = datetime(2026, 10, 6, 12, 0, 0)


# ---------------------------------------------------------------- 热度公式


def test_new_article_is_not_scored_zero():
    """
    刚发布、还没人看的文章必须有分。
    否则它永远排在任何有浏览量的文章后面，新内容永远出不来 ——
    这是热度算法最常见的实现错误。
    """
    assert hot_score(0, NOW, NOW) > 0


def test_same_views_newer_ranks_higher():
    assert hot_score(100, NOW, NOW) > hot_score(100, NOW - timedelta(days=3), NOW)


def test_same_time_more_views_ranks_higher():
    assert hot_score(500, NOW, NOW) > hot_score(100, NOW, NOW)


def test_old_article_decays():
    assert hot_score(100, NOW - timedelta(days=30), NOW) < hot_score(100, NOW, NOW)


def test_future_publish_time_does_not_explode():
    """
    发布时间晚于当前时间（服务器时钟漂移、数据录入错误）时 age 是负数。
    必须钳到 0，否则负数参与幂运算会算出异常分数。
    """
    assert hot_score(10, NOW + timedelta(days=1), NOW) == hot_score(10, NOW, NOW)


def test_very_old_article_is_clamped():
    """8 年前的文章要落在年龄上限那一档，不该因为年龄无限增大被压到 0"""
    assert hot_score(100, NOW - timedelta(days=365 * 8), NOW) > 0


def test_views_can_outweigh_recency():
    """短时间内的高浏览量应该压过刚发布的小浏览量文章"""
    assert hot_score(1000, NOW - timedelta(hours=2), NOW) > hot_score(10, NOW, NOW)


# ---------------------------------------------------------------- 分词


def test_tokenize_keeps_ascii_words_whole():
    from ai.retriever import tokenize

    tokens = tokenize("2023年GDP增长5.2%")
    assert "gdp" in tokens
    assert "2023" in tokens


def test_tokenize_single_cjk_query_still_works():
    """单字查询要能出结果，否则搜「猫」永远为空"""
    from ai.retriever import tokenize

    assert "猫" in tokenize("猫")


def test_tokenize_handles_empty():
    from ai.retriever import tokenize

    assert tokenize("") == []


# ---------------------------------------------------------------- 测试数据


def _make_news(idx: int, title: str, views: int, category_id: int, days_ago: int):
    return News(
        id=idx,
        title=title,
        description=f"{title}的摘要",
        content=f"{title}的正文内容，包含 GDP 人工智能 芯片等关键词。",
        image=None,
        author="记者",
        category_id=category_id,
        views=views,
        publish_time=NOW - timedelta(days=days_ago),
    )


@pytest.fixture
def corpus_rows():
    """12 篇，覆盖不同分类/热度/时间，用来验证排序和分页"""
    rows = [Category(id=1, name="科技"), Category(id=2, name="财经")]
    base = NOW - timedelta(days=10)
    specs = [
        ("2023年我国GDP同比增长5.2%", 15300, 2, 1),
        ("中国发布AI芯片政策", 7200, 1, 2),
        ("人工智能大模型取得突破", 8900, 1, 3),
        ("全球芯片短缺缓解", 4300, 2, 4),
        ("量子计算云平台上线", 2100, 1, 5),
        ("中国GDP季度增长放缓", 6400, 2, 6),
        ("人工智能与就业影响", 5200, 1, 7),
        ("芯片制造工艺突破", 3100, 1, 8),
        ("全球债务风险上升", 4800, 2, 9),
        ("AI监管法案出台", 2600, 1, 10),
        ("新能源汽车销量增长", 3900, 2, 11),
        ("量子通信试验成功", 1800, 1, 12),
    ]
    for i, (title, views, cat, days) in enumerate(specs, start=1):
        rows.append(_make_news(i, title, views, cat, days))
    return rows


@pytest.fixture
def client_with_news(client_authed, fake_session, corpus_rows):
    """
    把语料灌进 FakeSession。

    FakeSession 对任何 select 都返回同一个 result，这里按语句里的表名分派，
    模拟真实 DB 的返回形状：
      - select(Category.id, Category.name) -> 元组列表
      - select(News)                      -> ORM 对象列表
    形状不一致时 FakeSession 会让用例报出误导性的 TypeError。
    """
    news_rows = [r for r in corpus_rows if isinstance(r, News)]
    cats = [r for r in corpus_rows if isinstance(r, Category)]
    cat_pairs = [(c.id, c.name) for c in cats]

    async def execute(stmt):
        fake_session.execute_calls += 1
        text = str(stmt)
        if "FROM news_category" in text:
            return FakeResult(rows=cat_pairs)
        if "FROM history" in text or "JOIN history" in text:
            return FakeResult(rows=[])
        return FakeResult(rows=news_rows)

    fake_session.execute = execute
    return client_authed


# ---------------------------------------------------------------- 信息流


def test_feed_works_without_category_id(client_with_news):
    """
    核心回归：原 /api/news/list 强制要求 categoryId，
    导致首页没有内容可显示。新接口必须允许不传分类。
    """
    resp = client_with_news.get("/api/news/feed")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["total"] == 12
    assert len(data["list"]) == 10


def test_feed_paginates(client_with_news):
    p1 = client_with_news.get(
        "/api/news/feed", params={"page": 1, "pageSize": 5}
    ).json()["data"]
    p2 = client_with_news.get(
        "/api/news/feed", params={"page": 2, "pageSize": 5}
    ).json()["data"]
    assert p1["hasMore"] is True
    assert not ({i["id"] for i in p1["list"]} & {i["id"] for i in p2["list"]})


def test_feed_returns_category_name(client_with_news):
    """分类名随列表返回，省掉前端一次额外请求"""
    data = client_with_news.get("/api/news/feed").json()["data"]
    assert all(i["categoryName"] for i in data["list"])


def test_feed_latest_sort_is_time_desc(client_with_news):
    """
    最新优先 = 发布时间倒序。
    测试数据里 id 越大代表发布越早（days_ago 递增），
    所以时间倒序对应的期望顺序是 id 升序。
    """
    data = client_with_news.get(
        "/api/news/feed", params={"sort": "latest", "pageSize": 12}
    ).json()["data"]
    ids = [i["id"] for i in data["list"]]
    assert ids == sorted(ids), "最新优先应该把 id=1（最新）排在最前"


def test_feed_hot_sort_differs_from_latest(client_with_news):
    latest = client_with_news.get(
        "/api/news/feed", params={"sort": "latest", "pageSize": 12}
    ).json()["data"]
    hot = client_with_news.get(
        "/api/news/feed", params={"sort": "hot", "pageSize": 12}
    ).json()["data"]
    assert [i["id"] for i in latest["list"]] != [i["id"] for i in hot["list"]]


def test_feed_rejects_unknown_sort(client_with_news):
    assert client_with_news.get("/api/news/feed", params={"sort": "nope"}).status_code == 422


def test_feed_personalized_without_login_does_not_401(client_with_news):
    """
    未登录访问个性化不能 401。
    首页因为「没登录」就白屏是明显的体验事故，退化成时间序才对。
    """
    resp = client_with_news.get("/api/news/feed", params={"sort": "personalized"})
    assert resp.status_code == 200
    assert len(resp.json()["data"]["list"]) > 0


# ---------------------------------------------------------------- 热榜


def test_hot_rank_exposes_descending_score(client_with_news):
    """
    排名要能解释：返回 hotScore 让前端和运维都能看到排序依据。
    """
    data = client_with_news.get("/api/news/hot", params={"limit": 10}).json()["data"]
    scores = [i["hotScore"] for i in data["list"]]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)


def test_hot_returns_at_most_limit(client_with_news):
    data = client_with_news.get("/api/news/hot", params={"limit": 3}).json()["data"]
    assert len(data["list"]) == 3


# ---------------------------------------------------------------- 搜索


def test_search_finds_known_article(client_with_news):
    """搜 GDP 必须能命中语料里那几篇"""
    data = client_with_news.get("/api/news/search", params={"keyword": "GDP"}).json()["data"]
    assert data["total"] > 0
    assert "GDP" in " ".join(i["title"] for i in data["list"])


def test_search_returns_score_and_matched_terms(client_with_news):
    data = client_with_news.get(
        "/api/news/search", params={"keyword": "人工智能"}
    ).json()["data"]
    assert all("score" in i for i in data["list"])
    assert all("matchedTerms" in i for i in data["list"])


def test_search_scores_descend(client_with_news):
    data = client_with_news.get("/api/news/search", params={"keyword": "芯片"}).json()["data"]
    scores = [i["score"] for i in data["list"]]
    assert scores == sorted(scores, reverse=True)


def test_search_blank_keyword_returns_empty(client_with_news):
    """空白关键词不触发一次全量检索"""
    data = client_with_news.get("/api/news/search", params={"keyword": "   "}).json()["data"]
    assert data["total"] == 0
    assert data["list"] == []


def test_search_reports_duration(client_with_news):
    """检索耗时要在响应里，方便发现性能回退"""
    data = client_with_news.get("/api/news/search", params={"keyword": "科技"}).json()["data"]
    assert data["tookMs"] >= 0


def test_search_pagination_no_overlap(client_with_news):
    p1 = client_with_news.get(
        "/api/news/search", params={"keyword": "中国", "page": 1, "pageSize": 3}
    ).json()["data"]
    p2 = client_with_news.get(
        "/api/news/search", params={"keyword": "中国", "page": 2, "pageSize": 3}
    ).json()["data"]
    assert not ({i["id"] for i in p1["list"]} & {i["id"] for i in p2["list"]})