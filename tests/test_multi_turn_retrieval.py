"""
多轮检索改写与召回去重的测试。

这两个都是实测发现的真问题，不是预防性设计：

1. **检索改写**：追问句「那人均呢」几乎没有检索词，直接拿去搜 BM25 和向量
   都搜不到 GDP 内容，实测会直接触发拒答 —— 用户看到的是「这系统听不懂追问」。
   上下文只喂给生成层是不够的，检索层也必须用。

2. **召回去重**：语料里有 15 组共 33 篇内容完全相同的新闻，索引侧已经改成
   一对多映射让它们都能进索引，但检索侧会把同一份知识召回多次，
   挤占 top_k 名额（实测拒答消息里连续出现三条同名新闻）。
"""
from datetime import datetime, timedelta

import pytest

from ai import retriever as retriever_module
from models.news import News


def _news(nid: int, title: str, content: str = "", days_ago: int = 0) -> News:
    return News(
        id=nid,
        title=title,
        description=None,
        content=content or f"{title}的正文内容",
        image=None,
        author="记者",
        category_id=1,
        views=100,
        publish_time=datetime.now() - timedelta(days=days_ago),
    )


@pytest.fixture(autouse=True)
def clean_index_cache():
    """
    清掉 BM25 索引缓存。

    它是按语料指纹缓存的进程级对象，不同用例用不同语料时必须清 ——
    否则会拿到上一个语料的索引，表现为 bm25Hits=0 这种看起来像代码坏了的现象。
    """
    retriever_module.clear_index_cache()
    retriever_module._matrix_cache.clear()
    yield
    retriever_module.clear_index_cache()
    retriever_module._matrix_cache.clear()


@pytest.fixture
def multi_turn_corpus(monkeypatch):
    """
    构造含重复文档的语料，并接管 load_corpus 让 retrieve 不连库。
    """
    corpus = [
        _news(1, "2023年我国GDP同比增长5.2%", "2023年GDP总量1260582亿元，人均未提及"),
        _news(2, "完全重复的新闻A", "这段正文被逐字复制了两次"),
        _news(3, "完全重复的新闻A", "这段正文被逐字复制了两次"),
        _news(4, "完全重复的新闻B", "另一段被复制的正文"),
        _news(5, "完全重复的新闻B", "另一段被复制的正文"),
        _news(6, "完全不相关的科技新闻", "量子计算取得新进展"),
    ]

    import asyncio

    async def fake_load(db, use_cache=True):
        return corpus

    monkeypatch.setattr(retriever_module, "load_corpus", fake_load)
    # 去重后要重新算 top_score 和置信门控，关掉 trace 落库
    return corpus


# ---------------------------------------------------------------- 召回去重


@pytest.mark.asyncio
async def test_identical_documents_are_deduplicated(multi_turn_corpus):
    """
    内容完全相同的文档只保留一条。

    回归测试：曾经出现过拒答消息里连续列出三条同名新闻，
    既浪费 top_k 名额又难看。
    """
    results, _ = await retriever_module.retrieve(
        None, "完全重复的新闻", top_k=10, use_cache=False
    )
    ids = [r.news_id for r in results]

    # 2/3 和 4/5 各留一条，不能同时出现
    assert not (2 in ids and 3 in ids), "id=2/3 内容相同，应只保留一条"
    assert not (4 in ids and 5 in ids), "id=4/5 内容相同，应只保留一条"


@pytest.mark.asyncio
async def test_different_content_with_same_title_is_kept(monkeypatch):
    """
    标题相同但内容不同的文档都要保留 —— 它们是不同的事实。

    这一条很重要：如果按标题去重，会把不同年份的同题材新闻误杀，
    那才是真的信息丢失。
    """
    corpus = [
        _news(10, "中国数字阅读用户规模超5.3亿", "2023年人均电子书阅读量1.6本"),
        _news(11, "中国数字阅读用户规模超5.3亿", "2025年人均电子书阅读量1.6本"),
    ]
    # 同样要补无关文档：只有 2 篇时 IDF 为负，BM25 会把所有结果过滤掉
    for i in range(30):
        corpus.append(_news(200 + i, f"其他报道{i}", f"不同主题的内容{i}"))

    async def fake_load(db, use_cache=True):
        return corpus

    monkeypatch.setattr(retriever_module, "load_corpus", fake_load)

    results, _ = await retriever_module.retrieve(
        None, "数字阅读用户规模", top_k=10, use_cache=False
    )
    ids = {r.news_id for r in results}
    assert 10 in ids and 11 in ids, (
        f"内容不同的同标题新闻都应保留，实际返回 {ids}"
    )


@pytest.mark.asyncio
async def test_dedup_still_fills_up_to_top_k(monkeypatch):
    """
    去重后要继续往后取，直到凑满 top_k。

    如果只是简单跳过重复项就 break，返回条数会少于 top_k，
    用户会看到「明明有相关新闻，怎么只给 2 条」。

    语料要造得够大：rank_bm25 的 IDF 在查询词出现在过半文档时会变负，
    得分 <= 0 的结果会被全部过滤（bm25_search 里就有这个过滤）。
    只有 3~8 篇的语料会让所有查询都得 0 分，看起来像功能坏了。
    """
    corpus = [
        _news(20, "重复文档A", "正文A"),
        _news(21, "重复文档A", "正文A"),
        _news(22, "重复文档B", "正文B"),
        _news(23, "重复文档B", "正文B"),
        _news(24, "重复文档C", "正文C"),
        _news(25, "重复文档C", "正文C"),
        _news(26, "重复文档D", "正文D"),
        _news(27, "重复文档D", "正文D"),
    ]
    # 补一批无关文档，把查询词的文档占比压到一半以下
    for i in range(30):
        corpus.append(_news(100 + i, f"无关新闻编号{i}", f"完全不同的内容主题{i}"))

    async def fake_load(db, use_cache=True):
        return corpus

    monkeypatch.setattr(retriever_module, "load_corpus", fake_load)

    results, _ = await retriever_module.retrieve(
        None, "重复文档", top_k=4, use_cache=False
    )
    assert len(results) == 4, (
        f"8 篇重复对去重后应恰好剩 4 条，实际 {len(results)}"
    )

    # 每组重复只留一条：标题不该出现两次
    titles = [r.title for r in results]
    assert len(titles) == len(set(titles)), f"仍有重复标题: {titles}"


# ---------------------------------------------------------------- 检索改写


@pytest.mark.asyncio
async def test_followup_query_uses_previous_question(monkeypatch):
    """
    核心回归：追问句要带上上一轮的问题去检索。

    没有这一步，「那人均呢」会搜不到任何 GDP 内容并直接拒答。
    """
    seen: list[str] = []

    def fake_bm25(corpus, query, top_k=20):
        seen.append(query)
        return []

    async def fake_vector(corpus, query, top_k=20):
        seen.append(query)
        return []

    monkeypatch.setattr(retriever_module, "bm25_search", fake_bm25)
    monkeypatch.setattr(retriever_module, "vector_search", fake_vector)

    async def fake_load(db, use_cache=True):
        return [_news(1, "测试新闻")]

    monkeypatch.setattr(retriever_module, "load_corpus", fake_load)

    await retriever_module.retrieve(
        None,
        "那人均呢？",
        top_k=3,
        use_cache=False,
        history=[("2023年我国GDP同比增长多少？", "增长了5.2%")],
    )

    assert seen, "BM25/向量都应被调用"
    for query in seen:
        assert "GDP" in query, f"检索词应包含上一轮的关键词，实际 {query!r}"
        assert "那人均呢" in query, f"检索词应保留本轮问题，实际 {query!r}"


@pytest.mark.asyncio
async def test_first_turn_query_is_untouched(monkeypatch):
    """首轮没有历史，检索词必须原样使用，不能拼上任何东西"""
    seen: list[str] = []

    def fake_bm25(corpus, query, top_k=20):
        seen.append(query)
        return []

    async def fake_vector(corpus, query, top_k=20):
        seen.append(query)
        return []

    monkeypatch.setattr(retriever_module, "bm25_search", fake_bm25)
    monkeypatch.setattr(retriever_module, "vector_search", fake_vector)

    async def fake_load(db, use_cache=True):
        return [_news(1, "测试新闻")]

    monkeypatch.setattr(retriever_module, "load_corpus", fake_load)

    await retriever_module.retrieve(None, "人工智能政策", top_k=3, use_cache=False)

    assert all(q == "人工智能政策" for q in seen), f"首轮检索词不该被改写: {seen}"


@pytest.mark.asyncio
async def test_rewrite_does_not_include_previous_answer(monkeypatch):
    """
    只带上一轮的提问，不带回答。

    回答里通常有 [1][2] 引用和整段叙述，拼进去会把本轮的检索意图带偏。
    """
    seen: list[str] = []

    def fake_bm25(corpus, query, top_k=20):
        seen.append(query)
        return []

    async def fake_vector(corpus, query, top_k=20):
        seen.append(query)
        return []

    monkeypatch.setattr(retriever_module, "bm25_search", fake_bm25)
    monkeypatch.setattr(retriever_module, "vector_search", fake_vector)

    async def fake_load(db, use_cache=True):
        return [_news(1, "测试新闻")]

    monkeypatch.setattr(retriever_module, "load_corpus", fake_load)

    await retriever_module.retrieve(
        None,
        "那人均呢？",
        top_k=3,
        use_cache=False,
        history=[
            ("2023年GDP增长多少？", "这段回答里有一堆无关的叙述和[1][2]引用标记")
        ],
    )

    for query in seen:
        assert "[1]" not in query, "上一轮回答的引用标记不该进检索词"
        assert "无关的叙述" not in query, "上一轮回答内容不该进检索词"


@pytest.mark.asyncio
async def test_blank_history_question_does_not_break(monkeypatch):
    """历史里的上一轮提问为空时不能拼出畸形检索词"""

    async def fake_load(db, use_cache=True):
        return [_news(1, "测试新闻")]

    monkeypatch.setattr(retriever_module, "load_corpus", fake_load)
    seen: list[str] = []

    def fake_bm25(corpus, query, top_k=20):
        seen.append(query)
        return []

    monkeypatch.setattr(retriever_module, "bm25_search", fake_bm25)
    monkeypatch.setattr(retriever_module, "vector_search", lambda c, q, top_k=20: _empty())

    await retriever_module.retrieve(
        None, "新问题", top_k=3, use_cache=False, history=[("   ", "回答")]
    )

    for query in seen:
        assert query.strip() == "新问题", f"空历史不该产生畸形检索词: {query!r}"


async def _empty():
    return []
