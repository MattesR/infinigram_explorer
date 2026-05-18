"""
Standard IR evaluation metrics using pytrec_eval.

Usage:
    from ir_metrics import compute_metrics

    metrics = compute_metrics(ranked_doc_ids, ranked_scores, qrels, qid, top_k_list=[10, 100, 1000])
"""

import pytrec_eval
import numpy as np


def compute_metrics(ranked_doc_ids, qrels, qid, top_k_list=None, ranked_scores=None):
    """
    Compute IR metrics for a ranked list of doc IDs using pytrec_eval.

    Args:
        ranked_doc_ids: List of doc IDs in ranked order (best first).
        qrels: Full qrels dict {qid: {doc_id: relevance}}.
        qid: Query ID.
        top_k_list: Cutoffs to evaluate at.
        ranked_scores: Optional list of scores (same order as doc_ids).
            If not provided, uses inverse rank as score.

    Returns dict with per-cutoff metrics and overall metrics.
    """
    if top_k_list is None:
        top_k_list = [10, 100, 1000]

    # Build run dict: doc_id -> score
    run = {}
    for i, did in enumerate(ranked_doc_ids):
        if ranked_scores is not None:
            run[did] = float(ranked_scores[i])
        else:
            run[did] = float(len(ranked_doc_ids) - i)  # higher rank = higher score

    # Build qrel for this query
    qrel_single = {qid: {str(did): int(rel) for did, rel in qrels.get(qid, {}).items()}}
    run_single = {qid: {str(did): score for did, score in run.items()}}

    # Build measure set
    measures = set()
    for k in top_k_list:
        measures.add(f"ndcg_cut_{k}")
        measures.add(f"recall_{k}")
        measures.add(f"P_{k}")
    measures.add("recip_rank")
    measures.add("map")

    evaluator = pytrec_eval.RelevanceEvaluator(qrel_single, measures)
    eval_result = evaluator.evaluate(run_single)

    query_metrics = eval_result.get(qid, {})

    # Organize results
    results = {}
    for k in top_k_list:
        results[k] = {
            "ndcg": query_metrics.get(f"ndcg_cut_{k}", 0),
            "recall": query_metrics.get(f"recall_{k}", 0),
            "precision": query_metrics.get(f"P_{k}", 0),
            "found": int(query_metrics.get(f"recall_{k}", 0) *
                        len({d: r for d, r in qrels.get(qid, {}).items() if r > 0})),
        }

    results["mrr"] = query_metrics.get("recip_rank", 0)
    results["map"] = query_metrics.get("map", 0)
    results["n_relevant"] = len({d: r for d, r in qrels.get(qid, {}).items() if r > 0})

    return results


def compute_metrics_batch(runs, qrels, top_k_list=None):
    """
    Compute metrics for multiple queries at once (more efficient).

    Args:
        runs: Dict of qid -> {doc_id: score}
        qrels: Dict of qid -> {doc_id: relevance}
        top_k_list: Cutoffs.

    Returns dict of qid -> metrics dict.
    """
    if top_k_list is None:
        top_k_list = [10, 100, 1000]

    # Convert to string keys for pytrec_eval
    qrel_str = {str(qid): {str(did): int(rel) for did, rel in docs.items()}
                for qid, docs in qrels.items()}
    run_str = {str(qid): {str(did): float(score) for did, score in docs.items()}
               for qid, docs in runs.items()}

    measures = set()
    for k in top_k_list:
        measures.add(f"ndcg_cut_{k}")
        measures.add(f"recall_{k}")
        measures.add(f"P_{k}")
    measures.add("recip_rank")
    measures.add("map")

    evaluator = pytrec_eval.RelevanceEvaluator(qrel_str, measures)
    all_results = evaluator.evaluate(run_str)

    organized = {}
    for qid, query_metrics in all_results.items():
        results = {}
        n_rel = len({d: r for d, r in qrels.get(qid, qrels.get(str(qid), {})).items() if r > 0})
        for k in top_k_list:
            results[k] = {
                "ndcg": query_metrics.get(f"ndcg_cut_{k}", 0),
                "recall": query_metrics.get(f"recall_{k}", 0),
                "precision": query_metrics.get(f"P_{k}", 0),
                "found": int(query_metrics.get(f"recall_{k}", 0) * n_rel),
            }
        results["mrr"] = query_metrics.get("recip_rank", 0)
        results["map"] = query_metrics.get("map", 0)
        results["n_relevant"] = n_rel
        organized[qid] = results

    return organized


def pool_metrics(df):
    for mode in df["mode"].unique():
        mdf = df[df["mode"] == mode]
        recall = mdf["recall"]
        precision = mdf["n_found"] / mdf["n_retrieved"]
        
        # CVaR (Conditional Value at Risk) = mean of bottom 10% recalls
        sorted_recall = recall.sort_values()
        n_tail = max(1, int(len(sorted_recall) * 0.1))
        cvar = sorted_recall.iloc[:n_tail].mean()
        
        print(f"\n{mode} ({len(mdf)} queries):")
        print(f"  Mean Recall:       {recall.mean():.4f}")
        print(f"  Std Recall:        {recall.std():.4f}")
        print(f"  Median Recall:     {recall.median():.4f}")
        print(f"  CVaR (bottom 10%): {cvar:.4f}")
        print(f"  Avg Retrieved:     {mdf['n_retrieved'].mean():.0f}")
        print(f"  Mean Prec@Pool:    {precision.mean():.6f}")
