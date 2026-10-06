"""
检索质量评估器。

回答一个此前没人能回答的问题：**这套RAG 的召回率到底是多少。**

用法
----
    python -m ai.eval.runner --baseline            # 当前配置的各指标
    python -m ai.eval.runner --baseline --k 3,5,8  # 指定多个 K
    python -m ai.eval.runner --sweep                # 参数网格扫描

设计要点
--------
1. **只跑本地 BM25，不调任何外部 API。**
   向量路默认关闭（`--vector` 开启），因为评估要能进 CI 且秒级完成。
   这意味着默认数字是 **BM25 单路的召回率**，混合检索的真实表现只会更高。
   README 里对此有明确标注，不要拿它当混合检索的最终结论。

2. **指标只用新闻 ID 集合运算，不需要人工标注相关性。**
   gold 是期望命中的新闻 ID，检索结果也是 ID，交集非空即算命中。
   这让评估完全可自动化 —— 这是能把质量门禁放进 CI 的前提。

3. **负例单独统计误召回。**
   幻觉的直接来源是「语料里没有的东西也召回并作答」。
   Recall 只衡量「该找到的找到了」，误召回率衡量「不该找到的却找到了」，
   两者必须一起看：只优化 Recall 很容易靠放宽阈值把幻觉率也放大。

4. **耗时按阶段拆开记录。**
   当前实现每次查询都要重建 BM25 索引（tokenize 403 篇 + 建索引），
   指标里会明确显示，用来定位性能瓶颈。
"""
import argparse
import io
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ai import config as ai_config
from ai.embeddings import cosine_similarity, embed_query, embed_texts
from ai.retriever import bm25_search, build_document, load_corpus
from models.news import News

DATASET_PATH = Path(__file__).with_name("dataset.jsonl")


# ---------------------------------------------------------------- 数据结构


@dataclass
class EvalCase:
    qid: str
    question: str
    gold: list[int]
    category: str
    kind: str
    note: str = ""

    @property
    def is_negative(self) -> bool:
        """负例：期望召回为空"""
        return not self.gold


@dataclass
class StageTiming:
    """各检索阶段的耗时（毫秒）"""

    load_corpus: float = 0.0
    bm25: float = 0.0
    vector: float = 0.0
    fuse: float = 0.0

    @property
    def total(self) -> float:
        return self.load_corpus + self.bm25 + self.vector + self.fuse


@dataclass
class CaseResult:
    case: EvalCase
    retrieved_ids: list[int]
    fusion_scores: dict[int, float] = field(default_factory=dict)
    top_score: float = 0.0
    answered: bool = False
    timing: StageTiming = field(default_factory=StageTiming)

    def rank_of_first_gold(self) -> int | None:
        """首个 gold 出现的名次（1-based）；没召回返回 None"""
        for i, news_id in enumerate(self.retrieved_ids, start=1):
            if news_id in self.case.gold:
                return i
        return None

    def hit_at(self, k: int) -> bool:
        return any(news_id in self.case.gold for news_id in self.retrieved_ids[:k])


# ---------------------------------------------------------------- 指标


def recall_at_k(results: list[CaseResult], k: int) -> float:
    """
    Recall@K：前K 条里召回了多少比例的 gold。

    分母用「有 gold 的用例数」。负例 gold 为空，算进分母会让指标失真，
    负例的判定放在 misfire_rate 里单独看。
    """
    positives = [r for r in results if not r.case.is_negative]
    if not positives:
        return 0.0
    hits = sum(1 for r in positives if r.hit_at(k))
    return hits / len(positives)


def full_recall(results: list[CaseResult], k: int) -> float:
    """完整召回率：前 K 条里找齐了**全部** gold 才算命中。

    比 recall_at_k 严格：多 gold 的用例（如 tech-07 有 2 条重复数据）
    只找到一条不算完整命中。
    """
    positives = [r for r in results if not r.case.is_negative]
    if not positives:
        return 0.0
    hits = sum(
        1
        for r in positives
        if r.case.gold and set(r.case.gold).issubset(set(r.retrieved_ids[:k]))
    )
    return hits / len(positives)


def mrr_at_k(results: list[CaseResult], k: int) -> float:
    """
    MRR@K：首个命中排名的倒数均值。

    只对有 gold 的用例计算。衡量排序质量 —— 命中靠后比不命中更糟，
    因为 top_k 只有 5 个位置，排第 4 名基本等于答不出好答案。
    """
    positives = [r for r in results if not r.case.is_negative]
    if not positives:
        return 0.0
    total = 0.0
    for r in positives:
        rank = r.rank_of_first_gold()
        if rank is not None and rank <= k:
            total += 1.0 / rank
    return total / len(positives)


def ndcg_at_k(results: list[CaseResult], k: int) -> float:
    """
    nDCG@K：位置加权召回。

    用DCG = Σ (命中位置 i 的增益) / log2(i+1)，再除以理想排序的 DCG。
    与 MRR 的区别是它对所有命中位次都计分，不只看第一个。
    """
    positives = [r for r in results if not r.case.is_negative]
    if not positives:
        return 0.0

    scores = []
    for r in positives:
        dcg = sum(
            1.0 / _log2(i + 1)
            for i, news_id in enumerate(r.retrieved_ids[:k], start=1)
            if news_id in r.case.gold
        )
        ideal_hits = min(len(r.case.gold), k)
        idcg = sum(1.0 / _log2(i + 1) for i in range(1, ideal_hits + 1))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(scores) / len(scores)


def _log2(x: float) -> float:
    import math

    return math.log2(x)


def refusal_accuracy(results: list[CaseResult]) -> tuple[float, float]:
    """
    返回 (拒答准确率, 误召回率)，只看负例。

    误召回率 = 负例里被判定为「有把握」的比例，也就是幻觉的直接来源。
    注意这里的「判定」就是线上真实的置信度门控逻辑，
    所以这个数字直接对应线上的幻觉风险。
    """
    negatives = [r for r in results if r.case.is_negative]
    if not negatives:
        return 0.0, 0.0

    correctly_refused = sum(1 for r in negatives if not r.answered)
    misfires = sum(1 for r in negatives if r.answered)
    return correctly_refused / len(negatives), misfires / len(negatives)


def answer_rate(results: list[CaseResult]) -> float:
    """整体作答率：被判为「有把握」的比例。过低说明阈值太保守"""
    if not results:
        return 0.0
    return sum(1 for r in results if r.answered) / len(results)


def positive_answer_rate(results: list[CaseResult]) -> float:
    """正例的作答率：真正该回答的问题里，回答了多少"""
    positives = [r for r in results if not r.case.is_negative]
    if not positives:
        return 0.0
    return sum(1 for r in positives if r.answered) / len(positives)


# ---------------------------------------------------------------- 检索执行


def run_retrieval(
    corpus: list[News],
    case: EvalCase,
    top_k: int,
    min_score: float,
    use_vector: bool = False,
    rrf_k: int | None = None,
    single_weight: float | None = None,
    vector_mode: str = "none",
) -> CaseResult:
    """
    跑一次检索并记录结果与耗时。

    完整复刻 ai.retriever.retrieve 的逻辑（含单路惩罚和置信门控），
    但不经过 Redis 缓存，以便参数扫描时不受缓存状态干扰。
    """
    timing = StageTiming()
    result = CaseResult(case=case, retrieved_ids=[])

    t0 = time.perf_counter()
    timing.load_corpus = (time.perf_counter() - t0) * 1000

    # --- BM25 ---
    t0 = time.perf_counter()
    bm25_hits = bm25_search(corpus, case.question)
    timing.bm25 = (time.perf_counter() - t0) * 1000

    bm25_rank = {idx: i + 1 for i, (idx, _, _) in enumerate(bm25_hits)}

    # --- 向量 ---
    vector_rank: dict[int, int] = {}
    if vector_mode != "none":
        t0 = time.perf_counter()
        try:
            documents = [build_document(n) for n in corpus]
            if vector_mode == "fake":
                from ai.eval.fake_embedding import fake_vectors

                doc_vectors = fake_vectors(documents)
                query_vector = fake_vectors([case.question])[0]
            else:
                doc_vectors = embed_texts_sync(documents)
                query_vector = embed_query_sync(case.question)

            scored = [
                (i, cosine_similarity(query_vector, v)) for i, v in enumerate(doc_vectors)
            ]
            scored.sort(key=lambda pair: pair[1], reverse=True)
            vector_rank = {
                i: r + 1 for r, (i, _) in enumerate(scored[: ai_config.TOP_K_VECTOR])
            }
        except Exception:
            vector_rank = {}
        timing.vector = (time.perf_counter() - t0) * 1000

    # --- RRF 融合（与线上一致，含单路惩罚） ---
    t0 = time.perf_counter()
    effective_rrf_k = rrf_k if rrf_k is not None else ai_config.RRF_K
    effective_weight = (
        single_weight if single_weight is not None else ai_config.SINGLE_PATH_WEIGHT
    )
    scores = _rrf(
        [list(bm25_rank.keys()), list(vector_rank.keys())],
        k=effective_rrf_k,
        single_path_weight=effective_weight,
    )

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    result.retrieved_ids = [corpus[i].id for i, _ in ordered]
    result.fusion_scores = {corpus[i].id: s for i, s in ordered}
    result.top_score = ordered[0][1] if ordered else 0.0
    result.answered = result.top_score >= min_score
    timing.fuse = (time.perf_counter() - t0) * 1000
    result.timing = timing

    return result


def _rrf(ranked_lists: list[list[int]], k: int, single_path_weight: float) -> dict[int, float]:
    """与 ai.retriever.reciprocal_rank_fusion 保持一致的本地实现。

    这里复制一份而不是 import，是为了参数扫描时不污染线上模块的全局配置。
    """
    total = len(ranked_lists)
    scores: dict[int, float] = {}
    hits: dict[int, int] = {}
    for ranking in ranked_lists:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
            hits[idx] = hits.get(idx, 0) + 1
    if single_path_weight < 1.0 and total > 1:
        for idx, count in hits.items():
            if count < total:
                scores[idx] *= single_path_weight
    return scores


def embed_texts_sync(texts: list[str]) -> list[list[float]]:
    import asyncio

    return asyncio.run(embed_texts(texts))


def embed_query_sync(text: str) -> list[float]:
    import asyncio

    return asyncio.run(embed_query(text))


# ---------------------------------------------------------------- 数据加载


def load_dataset(path: Path = DATASET_PATH) -> list[EvalCase]:
    cases = []
    for line_no, raw in enumerate(io.open(path, encoding="utf-8"), start=1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON: {exc}") from exc
        cases.append(
            EvalCase(
                qid=obj["qid"],
                question=obj["question"],
                gold=list(obj.get("gold") or []),
                category=obj.get("category", ""),
                kind=obj.get("kind", ""),
                note=obj.get("note", ""),
            )
        )
    return cases


def load_corpus_sync() -> list[News]:
    """
    从本地导出的语料 JSON 读入，不连数据库。

    这样评估不依赖 MySQL，能在 CI 里跑。
    语料文件由 scripts 导出，且应该与 database.sql 保持同步。
    """
    corpus_path = Path(__file__).with_name("_corpus.json")
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"缺少语料文件 {corpus_path}，"
            f"请先运行 python -m ai.eval.export_corpus 导出"
        )
    data = json.load(io.open(corpus_path, encoding="utf-8"))
    titles = {n["id"]: n for n in data["news"]}
    return [
        News(
            id=n["id"],
            title=n["title"],
            description=n.get("description"),
            content=n.get("content") or n["title"],
            category_id=n["category_id"],
            views=0,
            publish_time=datetime.fromisoformat(n["publish_time"])
            if n.get("publish_time")
            else datetime(2026, 1, 1),
        )
        for n in data["news"]
    ]


# ---------------------------------------------------------------- 报告


def print_baseline(
    results: list[CaseResult],
    ks: Iterable[int],
    meta: dict,
) -> None:
    ks = list(ks)
    print("=" * 78)
    print("检索质量基线")
    print("=" * 78)
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print()

    header = f"{'指标':<18}" + "".join(f"{'@' + str(k):>12}" for k in ks)
    print(header)
    print("-" * len(header))
    rows = [
        ("Recall@K", recall_at_k),
        ("完整召回率@K", full_recall),
        ("MRR@K", mrr_at_k),
        ("nDCG@K", ndcg_at_k),
    ]
    for name, fn in rows:
        print(f"{name:<18}" + "".join(f"{fn(results, k) * 100:>11.1f}%" for k in ks))
    print()

    refused, misfire = refusal_accuracy(results)
    print(f"  拒答准确率 (负例)     {refused * 100:>10.1f}%   {len([r for r in results if r.case.is_negative])} 条负例")
    print(f"  误召回率 (负例)       {misfire * 100:>10.1f}%   <- 幻觉的直接来源")
    print(f"  整体作答率            {answer_rate(results) * 100:>10.1f}%")
    print(f"  正例作答率            {positive_answer_rate(results) * 100:>10.1f}%")
    print()

    avg = _avg_timing(results)
    print("  各阶段平均耗时:")
    print(f"    BM25检索      {avg.bm25:>8.1f} ms   (含 tokenize 全部语料 + 重建索引)")
    if avg.vector > 0:
        print(f"    向量检索      {avg.vector:>8.1f} ms")
    print(f"    融合          {avg.fuse:>8.1f} ms")
    print(f"    合计          {avg.total:>8.1f} ms   <- 不含语料加载")


def _avg_timing(results: list[CaseResult]) -> StageTiming:
    if not results:
        return StageTiming()
    agg = StageTiming()
    for r in results:
        agg.bm25 += r.timing.bm25
        agg.vector += r.timing.vector
        agg.fuse += r.timing.fuse
        agg.load_corpus += r.timing.load_corpus
    n = len(results)
    agg.bm25 /= n
    agg.vector /= n
    agg.fuse /= n
    agg.load_corpus /= n
    return agg


def print_worst(results: list[CaseResult], k: int, limit: int = 12) -> None:
    """列出漏召用例，用于人工分析失败原因"""
    missed = [
        r for r in results
        if not r.case.is_negative and not r.hit_at(k)
    ]
    if not missed:
        print(f"  @{k} 无漏召用例")
        return
    print(f"  @{k} 漏召 {len(missed)} 条：")
    for r in missed[:limit]:
        print(f"    [{r.case.qid}] {r.case.question}")
        print(f"           gold={r.case.gold} 实际={r.retrieved_ids[:k]}")


def print_misfires(results: list[CaseResult], limit: int = 10) -> None:
    """列出负例里被放行的，也就是幻觉风险点"""
    bad = [r for r in results if r.case.is_negative and r.answered]
    if not bad:
        print("  负例无误召回")
        return
    print(f"  负例误召回 {len(bad)} 条（这些会被送去让模型作答，有幻觉风险）：")
    for r in bad[:limit]:
        print(f"    [{r.case.qid}] {r.case.question}")
        print(f"           放行分数={r.top_score:.4f} 命中={r.retrieved_ids[:3]}")


# ---------------------------------------------------------------- 参数扫描


def sweep(
    corpus: list[News],
    cases: list[EvalCase],
    k: int = 5,
    vector_mode: str = "none",
) -> None:
    """
    扫 MIN_FUSION_SCORE × TOP_K × RRF_K，输出权衡表。

    只输出数据不做选择 —— 阈值该定在哪取决于业务对「错答案」和「没答案」
    的相对代价，那是产品决策不是技术决策。
    """
    score_grid = [0.004, 0.008, 0.012, 0.016, 0.020, 0.025, 0.030, 0.040]
    k_grid = [3, 5, 8]
    rrf_k_grid = [ai_config.RRF_K]

    print("=" * 78)
    print("参数扫描")
    print("=" * 78)
    print(f"  语料 {len(corpus)} 条，评估集 {len(cases)} 条，K={k}")
    mode_label = {
        "none": "关闭（仅 BM25 单路）",
        "fake": "开启（确定性假向量，无语义）",
        "real": "开启（真实 DashScope embedding）",
    }.get(vector_mode, vector_mode)
    print(f"  向量路: {mode_label}")
    print()

    for rrf_k in rrf_k_grid:
        print(f"--- RRF_K={rrf_k} ---")
        header = (
            f"{'MIN_FUSION':<12}{'TOP_K':<8}"
            f"{'Recall@K':>10}{'MRR@K':>9}{'拒答率':>9}{'误召回':>9}{'正例作答':>10}"
        )
        print(header)
        print("-" * len(header))

        for top_k in k_grid:
            for min_score in score_grid:
                results = [
                    run_retrieval(
                        corpus, case, top_k=top_k, min_score=min_score,
                        vector_mode=vector_mode, rrf_k=rrf_k,
                    )
                    for case in cases
                ]
                refused, misfire = refusal_accuracy(results)
                print(
                    f"{min_score:<12.3f}{top_k:<8}"
                    f"{recall_at_k(results, top_k) * 100:>9.1f}%"
                    f"{mrr_at_k(results, top_k) * 100:>8.1f}%"
                    f"{(1 - answer_rate(results)) * 100:>8.1f}%"
                    f"{misfire * 100:>8.1f}%"
                    f"{positive_answer_rate(results) * 100:>9.1f}%"
                )
        print()

    print("当前线上配置:")
    print(
        f"  MIN_FUSION_SCORE={ai_config.MIN_FUSION_SCORE}  "
        f"TOP_K_FINAL={ai_config.TOP_K_FINAL}  RRF_K={ai_config.RRF_K}"
    )


# ---------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索质量评估")
    parser.add_argument("--baseline", action="store_true", help="打印当前配置的基线指标")
    parser.add_argument("--sweep", action="store_true", help="参数网格扫描")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="评估集路径")
    parser.add_argument("--k", default="3,5,8", help="逗号分隔的 K 值列表")
    parser.add_argument(
        "--vector",
        action="store_true",
        help="启用向量路，使用确定性假向量（不调API，仅验证融合机制）",
    )
    parser.add_argument(
        "--vector-real",
        action="store_true",
        help="启用向量路，调用真实 DashScope embedding（需要 API Key）",
    )
    parser.add_argument("--min-score", type=float, default=None, help="覆盖置信度阈值")
    parser.add_argument("--top-k", type=int, default=None, help="覆盖 TOP_K")
    args = parser.parse_args()

    ks = [int(x) for x in args.k.split(",") if x.strip()]
    cases = load_dataset(Path(args.dataset))
    corpus = load_corpus_sync()

    min_score = args.min_score if args.min_score is not None else ai_config.MIN_FUSION_SCORE
    top_k = args.top_k if args.top_k is not None else ai_config.TOP_K_FINAL

    vector_mode = "none"
    if args.vector_real:
        vector_mode = "real"
    elif args.vector:
        vector_mode = "fake"

    meta = {
        "语料条数": len(corpus),
        "评估集": f"{len(cases)} 条（正例 {sum(1 for c in cases if not c.is_negative)} / "
        f"负例 {sum(1 for c in cases if c.is_negative)}）",
        "MIN_FUSION_SCORE": min_score,
        "TOP_K_FINAL": top_k,
        "RRF_K": ai_config.RRF_K,
        "单路惩罚系数": ai_config.SINGLE_PATH_WEIGHT,
        "向量路": {
            "none": "关闭（纯 BM25 单路）",
            "fake": "开启（确定性假向量，无语义）",
            "real": "开启（真实 DashScope embedding）",
        }.get(vector_mode, vector_mode),
    }

    if args.sweep:
        sweep(corpus, cases, k=top_k, vector_mode=vector_mode)
        return

    if not args.baseline:
        parser.print_help()
        return

    started = time.perf_counter()
    results = [
        run_retrieval(
            corpus, case, top_k=top_k, min_score=min_score, vector_mode=vector_mode
        )
        for case in cases
    ]
    elapsed = time.perf_counter() - started

    meta["总耗时"] = f"{elapsed:.2f} s"
    print_baseline(results, ks, meta)

    primary_k = top_k
    print()
    print("-" * 78)
    print(f"漏召分析 (K={primary_k})")
    print("-" * 78)
    print_worst(results, primary_k)

    print()
    print("-" * 78)
    print("负例误召回（幻觉风险点）")
    print("-" * 78)
    print_misfires(results)


if __name__ == "__main__":
    main()
