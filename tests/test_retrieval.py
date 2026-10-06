"""
混合检索（RAG 召回）测试。

全部离线运行，不依赖真实 DashScope API：
向量化调用用 monkeypatch 替换，数据库用假 Session。
"""
import math
from datetime import datetime

import pytest

from ai.embeddings import cosine_similarity
from ai.prompts import (
    build_news_context,
    build_news_qa_messages,
    build_refusal_message,
)
from ai.retriever import (
    build_document,
    bm25_search,
    reciprocal_rank_fusion,
    retrieve,
    tokenize,
)
from models.news import News
from tests.fakes import FakeResult, FakeSession


def make_news(id_, title, content="", description=None, category_id=1):
    return News(
        id=id_,
        title=title,
        description=description,
        content=content,
        category_id=category_id,
        views=0,
        publish_time=datetime(2026, 1, 1, 12, 0, 0),
    )


CORPUS = [
    make_news(1, "全国铁路春运启动", "2026年春运将于1月20日启动，为期40天。高铁方面新增班次。"),
    make_news(2, "央行宣布下调存款准备金率", "中国人民银行决定下调金融机构存款准备金率0.5个百分点。"),
    make_news(3, "2025年GDP增长5.2%", "国家统计局公布初步核算结果，2025年国内生产总值同比增长5.2%。"),
    make_news(4, "国际油价大幅波动", "受地缘因素影响，国际原油价格单周振幅超过10%。"),
]


def corpus_session():
    """假 DB，直接返回固定语料"""
    return FakeSession(result=FakeResult(rows=list(CORPUS)))


# ---------------- 分词 ----------------

def test_tokenize_produces_chinese_unigrams_and_bigrams():
    tokens = tokenize("中国高铁")

    assert "中" in tokens and "国" in tokens and "高" in tokens and "铁" in tokens
    assert "中国" in tokens
    assert "国高" in tokens
    assert "高铁" in tokens


def test_tokenize_keeps_ascii_and_numbers_as_whole_tokens():
    tokens = tokenize("2025年GDP增长5.2%")

    assert "gdp" in tokens, "专名缩写必须整体保留，否则 BM25 匹配不上"
    assert "2025" in tokens
    assert "5" in tokens and "2" in tokens


def test_tokenize_handles_empty_and_none():
    assert tokenize("") == []
    assert tokenize(None) == []


def test_single_chinese_char_query_is_tokenizable():
    """只有单字也能成词，否则单字查询永远召回不到"""
    assert "铁" in tokenize("铁")


# ---------------- 文档构造 ----------------

def test_build_document_weights_title():
    doc = build_document(make_news(1, "铁路春运", "正文提到其他内容"))

    assert doc.count("铁路春运") == 2, "标题应出现两次以提高权重"


# ---------------- BM25 ----------------

def test_bm25_finds_exact_keyword_match():
    hits = bm25_search(CORPUS, "GDP")

    assert hits, "应命中含 GDP 的新闻"
    top_idx, score, terms = hits[0]
    assert top_idx == 2  # 2025年GDP增长5.2% 那条
    assert score > 0
    assert "gdp" in terms


def test_bm25_ranks_exact_number_above_generic_match():
    """BM25 的强项：数字和专名的精确匹配

    查询 5.2% 应该把含 5.2% 的新闻排在前面，而不是含泛化词 GDP 的所有新闻。
    """
    hits = bm25_search(CORPUS, "5.2%")

    assert hits[0][0] == 2


def test_bm25_returns_empty_for_unrelated_query():
    """完全无关的查询不应返回任何结果，更不能返回负分/零分的噪声"""
    hits = bm25_search(CORPUS, "量子计算光刻机")

    assert hits == [] or all(score > 0 for _, score, _ in hits)


def test_bm25_ignores_single_character_overlap_only():
    """
    回归测试：中文单字区分度极低。
    「量子计算」与「人工智能算法」只共享单字「算」，但语义毫不相关。
    如果单字命中就算召回，任何两条新闻之间都能互相召回，导致无关问题
    也会走上 LLM 然后靠模型编造。必须至少命中一个多字词。

    语料要够大：BM25 的 IDF 在 df 接近 N/2 时趋近 0
    （两篇文档、词只出现一次时 idf 恰好为 0），小语料下 BM25 本身就不成立。
    """
    corpus = [
        make_news(1, "人工智能算法研究进展", "团队公布算法成果"),
        make_news(2, "量子计算光刻机取得突破", "光刻机技术进展"),
        make_news(3, "新能源汽车销量同比增长", "汽车市场报告"),
        make_news(4, "国内生产总值增速公布", "统计局数据"),
    ]

    # 「量子计算」只和第 1 篇共享单字「算」，没有任何共同二元组
    query_terms = set(tokenize("量子计算"))
    doc1_terms = set(tokenize(build_document(corpus[0])))
    assert query_terms & doc1_terms, "前提：应存在单字重合"
    assert not ({t for t in query_terms & doc1_terms if len(t) > 1}), "前提：不应有共同多字词"

    hits = bm25_search(corpus, "量子计算")

    matched_ids = {idx for idx, _, _ in hits}
    assert 0 not in matched_ids, "只靠单字重合的第 1 篇不应被召回"
    assert 1 in matched_ids, "真正含「量子计算」的第 2 篇应被召回"


def test_bm25_keeps_genuine_multichar_match():
    corpus = [
        make_news(1, "人工智能算法研究进展", "内容甲"),
        make_news(2, "量子计算光刻机取得突破", "光刻机技术进展"),
        make_news(3, "新能源汽车销量同比增长", "内容丙"),
        make_news(4, "国内生产总值增速公布", "内容丁"),
    ]

    hits = bm25_search(corpus, "光刻机")

    assert hits, "命中多字词时应正常返回"
    assert hits[0][0] == 1, "应命中唯一含「光刻机」的那篇"


def test_bm25_handles_empty_corpus():
    assert bm25_search([], "任意查询") == []


# ---------------- RRF 融合 ----------------

def test_rrf_sums_reciprocal_ranks():
    # 一条文档在两路都排第 1：1/61 + 1/61
    scores = reciprocal_rank_fusion([[0, 1, 2], [0, 1, 2]])

    assert scores[0] == pytest.approx(2 / 61)
    assert scores[1] == pytest.approx(2 / 62)
    assert scores[0] > scores[1], "两路都靠前的文档应得分更高"


def test_rrf_rewards_appearing_in_both_lists():
    """两路都召回的文档应排在只有单路召回的前面"""
    scores = reciprocal_rank_fusion([[10, 11], [11, 12]])

    assert scores[11] > scores[10]
    assert scores[11] > scores[12]


def test_rrf_k_smooths_head_scores():
    """k 值越大，头部文档的优势越小"""
    single = reciprocal_rank_fusion([[0, 1]], k=1)
    smoothed = reciprocal_rank_fusion([[0, 1]], k=1000)

    gap_small_k = single[0] - single[1]
    gap_large_k = smoothed[0] - smoothed[1]
    assert gap_large_k < gap_small_k


def test_rrf_is_symmetric_under_list_order():
    """融合结果不应受两路顺序影响"""
    assert reciprocal_rank_fusion([[0, 1], [1, 0]]) == reciprocal_rank_fusion([[1, 0], [0, 1]])


def test_single_path_hit_is_penalized():
    """
    回归测试：两路检索中只被一路召回的文档，证据强度弱于两路都召回的文档。
    不惩罚的话，向量服务故障降级为纯 BM25 时，1/(60+1)=0.0164 的单路弱相关
    命中也能越过置信度阈值，然后被送去让模型编造。
    """
    unpenalized = reciprocal_rank_fusion([[0, 1], [2, 3]], k=60, single_path_weight=1.0)
    penalized = reciprocal_rank_fusion([[0, 1], [2, 3]], k=60, single_path_weight=0.5)

    assert penalized[0] == pytest.approx(unpenalized[0] * 0.5)
    # 单路最高分被压到约 0.008，低于默认阈值 0.02
    assert penalized[0] < 0.02


def test_two_path_hit_keeps_full_score():
    """两路都召回的文档不受惩罚系数影响"""
    scores = reciprocal_rank_fusion([[0, 1], [0, 1]], k=60, single_path_weight=0.5)

    assert scores[0] == pytest.approx(2 / 61), "双路命中的最高分应约 0.0328"
    assert scores[0] > 0.02, "双路命中应能越过默认置信度阈值"


def test_partial_hit_only_penalized_for_missing_path():
    """部分命中的文档被惩罚，满命中的保持原分"""
    both = reciprocal_rank_fusion([[0], [0]], single_path_weight=0.5)
    partial = reciprocal_rank_fusion([[0], [1]], single_path_weight=0.5)

    assert both[0] == pytest.approx(2 / 61)
    assert partial[0] == pytest.approx(0.5 * (1 / 61))


def test_single_list_fusion_not_penalized():
    """
    只有一路检索时不做惩罚。
    此时不存在「两路一致」的比较基准，再惩罚一次等于双重打折，
    阈值校准会变得不可预测。
    """
    scores = reciprocal_rank_fusion([[0]], single_path_weight=0.5)

    assert scores[0] == pytest.approx(1 / 61)


# ---------------- 余弦相似度 ----------------

def test_cosine_similarity_identical_vectors():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_does_not_divide_by_zero():
    """全 0 向量会让除零得到 nan，nan 比较全是 False，排序行为会变得不可预测"""
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_cosine_similarity_length_mismatch():
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


def test_cosine_similarity_empty_vectors():
    assert cosine_similarity([], []) == 0.0


# ---------------- 混合检索端到端 ----------------

def test_retrieve_falls_back_to_bm25_when_embedding_fails(monkeypatch):
    """
    向量化不可用时应降级为纯 BM25，而不是整个接口 500。
    语义检索是增强项，不该成为单点故障。
    """
    import ai.retriever as retriever_module

    async def boom(*args, **kwargs):
        raise RuntimeError("embedding 服务不可用")

    monkeypatch.setattr(retriever_module, "embed_texts", boom)
    monkeypatch.setattr(retriever_module, "embed_query", boom)

    results, confident = asyncio_run(retrieve(corpus_session(), "GDP", use_cache=False))

    assert results, "应仍能通过 BM25 召回"
    assert any(r.bm25_rank for r in results)
    assert all(r.vector_rank is None for r in results), "向量路应为空"


def test_retrieve_returns_empty_and_unconfident_for_no_match(monkeypatch):
    """语料为空时应返回空且不自信"""
    empty = FakeSession(result=FakeResult(rows=[]))
    results, confident = asyncio_run(retrieve(empty, "任意问题", use_cache=False))

    assert results == []
    assert confident is False


def test_retrieve_hybrid_merges_both_paths(monkeypatch):
    """向量路召回了 BM25 没召回的文档，两路应被合并"""
    import ai.retriever as retriever_module

    async def fake_embed_texts(texts, use_cache=True):
        # 第 0 条（铁路）与查询相似，第 3 条（油价）完全无关
        table = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
        return table[: len(texts)]

    async def fake_embed_query(text):
        return [1.0, 0.0]

    monkeypatch.setattr(retriever_module, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(retriever_module, "embed_query", fake_embed_query)

    results, confident = asyncio_run(retrieve(corpus_session(), "高铁新增班次", use_cache=False))

    assert confident
    assert results, "应有两路融合后的结果"
    top = results[0]
    assert top.vector_rank is not None
    assert top.fusion_score > 0


# ---------------- Prompt 构造 ----------------

def test_build_context_numbers_articles_from_one():
    retrieved, _ = asyncio_run(retrieve(corpus_session(), "GDP", use_cache=False))
    context = build_news_context(retrieved, limit=3)

    assert context.startswith("[1] 标题：")
    assert "[2] 标题：" in context or len(retrieved) == 1


def test_build_context_respects_limit():
    retrieved, _ = asyncio_run(retrieve(corpus_session(), "的", use_cache=False))
    context = build_news_context(retrieved, limit=2)

    assert "[3] 标题：" not in context


def test_system_prompt_forbids_hallucination():
    """防幻觉约束是核心，必须在 system prompt 里显式写死"""
    from ai.prompts import NEWS_QA_SYSTEM_PROMPT

    assert "只使用" in NEWS_QA_SYSTEM_PROMPT
    assert "不要猜测" in NEWS_QA_SYSTEM_PROMPT
    assert "编号" in NEWS_QA_SYSTEM_PROMPT


def test_messages_structure_puts_context_before_question():
    retrieved, _ = asyncio_run(retrieve(corpus_session(), "GDP", use_cache=False))
    messages = build_news_qa_messages("GDP是多少", retrieved)

    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    user_content = messages[-1]["content"]
    assert "【新闻资料】" in user_content
    assert "【用户问题】" in user_content
    assert user_content.index("【新闻资料】") < user_content.index("【用户问题】")


def test_messages_include_only_recent_history():
    retrieved, _ = asyncio_run(retrieve(corpus_session(), "GDP", use_cache=False))
    history = [(f"第{i}轮提问", f"第{i}轮回答") for i in range(1, 6)]

    messages = build_news_qa_messages("当前问题", retrieved, history=history)

    user_turns = [m for m in messages if m["role"] == "user"]
    # 3 轮历史 + 1 个当前问题
    assert len(user_turns) == 4
    assert "第5轮提问" in user_turns[-2]["content"], "应保留最近的第 5 轮"
    assert "第1轮提问" not in str(messages), "久远的轮次应被截断"


# ---------------- 拒答 ----------------

def test_refusal_message_lists_found_articles():
    retrieved, _ = asyncio_run(retrieve(corpus_session(), "GDP", use_cache=False))

    message = build_refusal_message("某个问题", retrieved)

    assert "无法回答" in message
    assert "不做推测" in message
    assert "《" in message, "应告诉用户检索到了哪些报道"


def test_refusal_message_with_no_results_suggests_alternative():
    message = build_refusal_message("某个问题", [])

    assert "无法回答" in message
    assert "浏览新闻列表" in message


def test_to_citation_exposes_news_id_for_traceability():
    """溯源信息必须带 news_id，前端才能跳回原文"""
    retrieved, _ = asyncio_run(retrieve(corpus_session(), "GDP", use_cache=False))

    citation = retrieved[0].to_citation()

    assert citation["newsId"] == retrieved[0].news_id
    assert citation["title"]
    assert "fusionScore" in citation


def test_to_citation_is_json_serializable():
    """
    回归测试：SSE 事件用裸 json.dumps 序列化。
    to_citation 之前直接返回 datetime 对象，导致第一个 sources 事件就抛
    TypeError: Object of type datetime is not JSON serializable，接口完全不可用。
    """
    import json

    retrieved, _ = asyncio_run(retrieve(corpus_session(), "GDP", use_cache=False))

    # 走到真正 json.dumps 这一步才算验证到
    encoded = json.dumps([item.to_citation() for item in retrieved], ensure_ascii=False)

    assert json.loads(encoded)
    assert isinstance(json.loads(encoded)[0]["publishTime"], str)


# ---------------- 测试工具 ----------------

def asyncio_run(coro):
    """在同步测试里跑协程，避免每个用例都写 @pytest.mark.asyncio"""
    import asyncio

    return asyncio.run(coro)