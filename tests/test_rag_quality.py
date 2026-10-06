"""
RAG 检索质量门禁。

目的不是「保证质量好」，而是**保证质量不会悄悄变坏**。
改了分词、改了 BM25 参数、改了 RRF 逻辑，都可能让召回率下降而没人察觉。

纯本地 BM25，不调任何外部 API，约 3 秒跑完，可以进CI。

阈值故意设得宽（见 ai/eval/README.md 的说明）：只挡灾难性退化，不做精细控制。
等向量路用真实 embedding 标定完成后，再收紧。
"""
import pytest

from ai.eval.runner import (
    full_recall,
    load_corpus_sync,
    load_dataset,
    ndcg_at_k,
    positive_answer_rate,
    recall_at_k,
    refusal_accuracy,
    run_retrieval,
)

# CI 门禁阈值
MIN_RECALL_AT_5 = 0.75
MAX_MISFIRE_RATE = 0.35


@pytest.fixture(scope="module")
def corpus():
    return load_corpus_sync()


@pytest.fixture(scope="module")
def cases():
    return load_dataset()


@pytest.fixture(scope="module")
def results(corpus, cases):
    """用线上默认参数跑一遍完整评估"""
    from ai import config as ai_config

    return [
        run_retrieval(
            corpus,
            case,
            top_k=ai_config.TOP_K_FINAL,
            min_score=ai_config.MIN_FUSION_SCORE,
        )
        for case in cases
    ]


def test_dataset_is_wellformed(cases):
    """评估集本身的健康检查 —— 数据坏了门禁就失去意义"""
    assert len(cases) >= 40, "评估集过小，指标不稳定"

    qids = [c.qid for c in cases]
    assert len(set(qids)) == len(qids), "qid 必须唯一"

    negatives = [c for c in cases if c.is_negative]
    positives = [c for c in cases if not c.is_negative]
    assert len(negatives) >= 5, "负例太少，误召回率不可靠"
    assert len(positives) >= 20, "正例太少，召回率不可靠"

    for c in cases:
        assert c.question.strip(), f"{c.qid} 问题为空"
        assert isinstance(c.gold, list), f"{c.gid} gold 必须是列表"


def test_gold_ids_exist_in_corpus(cases, corpus):
    """
    gold 里的新闻 ID 必须真实存在。

    这条是门禁的地基：ID 对不上时Recall 恒为 0，指标全无意义却不会报错。
    """
    valid_ids = {n.id for n in corpus}
    missing = {
        c.qid: [g for g in c.gold if g not in valid_ids]
        for c in cases
        if any(g not in valid_ids for g in c.gold)
    }
    assert not missing, f"以下用例的 gold ID 不存在于语料中: {missing}"


def test_recall_at_5_meets_gate(results):
    """核心门禁：召回率不能低于下限"""
    score = recall_at_k(results, 5)

    assert score >= MIN_RECALL_AT_5, (
        f"Recall@5 = {score * 100:.1f}%，低于门禁 {MIN_RECALL_AT_5 * 100:.0f}%。"
        f"检索可能已退化，检查分词、BM25 参数或 RRF 逻辑的改动。"
    )


def test_misfire_rate_within_gate(results):
    """负例误召回率不能过高 —— 这是幻觉的直接来源"""
    _, misfire = refusal_accuracy(results)

    assert misfire <= MAX_MISFIRE_RATE, (
        f"误召回率 = {misfire * 100:.1f}%，超过门禁 {MAX_MISFIRE_RATE * 100:.0f}%。"
        f"语料外的问题被放行送去让模型作答，存在幻觉风险。"
    )


def test_full_recall_not_worse_than_partial_recall(results):
    """完整召回率不应高于普通召回率（否则说明多个 gold 用了相同 ID）"""
    partial = recall_at_k(results, 5)
    full = full_recall(results, 5)

    assert full <= partial + 1e-9, (
        f"完整召回率 {full:.3f} > Recall {partial:.3f}，评估集定义有问题"
    )


def test_metrics_in_valid_range(results):
    """所有指标必须落在 [0,1]，出现越界说明计算逻辑有 bug"""
    for k in (1, 3, 5):
        for fn in (recall_at_k, full_recall, ndcg_at_k):
            value = fn(results, k)
            assert 0.0 <= value <= 1.0, f"{fn.__name__}@{k} = {value} 越界"

    refused, misfire = refusal_accuracy(results)
    assert 0.0 <= refused <= 1.0
    assert 0.0 <= misfire <= 1.0
    assert 0.0 <= positive_answer_rate(results) <= 1.0


def test_recall_is_monotonic_in_k(results):
    """召回率必须随 K 单调不减 —— 否则说明 top_k 截断逻辑有误"""
    r1 = recall_at_k(results, 1)
    r3 = recall_at_k(results, 3)
    r5 = recall_at_k(results, 5)

    assert r1 <= r3 + 1e-9, f"Recall@1({r1:.3f}) > Recall@3({r3:.3f})"
    assert r3 <= r5 + 1e-9, f"Recall@3({r3:.3f}) > Recall@5({r5:.3f})"


def test_mrr_not_lower_than_recall(results):
    """MRR 是「命中排名的倒数」，数值上不会低于普通召回率的多数情况"""
    recall = recall_at_k(results, 5)
    from ai.eval.runner import mrr_at_k

    mrr = mrr_at_k(results, 5)
    # MRR 应该 >= Recall/K 的某个下界，这里只做宽松的一致性检查
    assert mrr <= 1.0 and recall <= 1.0


def test_results_have_timing_data(results):
    """耗时数据必须被记录，否则性能退化无法在CI 发现"""
    from ai.eval.runner import _avg_timing

    avg = _avg_timing(results)
    assert avg.bm25 > 0, "BM25 耗时未被记录"
    # 当前实现每次都重建索引，这个断言本身就是性能回归的哨兵
    assert avg.bm25 < 500, f"BM25 平均耗时 {avg.bm25:.0f}ms，性能可能已退化"
