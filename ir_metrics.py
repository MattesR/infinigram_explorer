"""
Standard IR evaluation metrics.

Usage:
    from ir_metrics import compute_metrics
    
    metrics = compute_metrics(ranked_doc_ids, qrels, qid, top_k_list=[10, 100, 1000])
"""

import math


def compute_metrics(ranked_doc_ids, qrels, qid, top_k_list=None):
    """
    Compute IR metrics for a ranked list of doc IDs.

    Args:
        ranked_doc_ids: List of doc IDs in ranked order (best first).
        qrels: Full qrels dict {qid: {doc_id: relevance}}.
        qid: Query ID.
        top_k_list: Cutoffs to evaluate at.

    Returns dict with per-cutoff metrics and overall metrics.
    """
    if top_k_list is None:
        top_k_list = [10, 100, 1000]

    relevant = {did: rel for did, rel in qrels.get(qid, {}).items() if rel > 0}
    n_relevant = len(relevant)

    results = {}

    for k in top_k_list:
        top_k_ids = ranked_doc_ids[:k]

        # Recall
        found = set(top_k_ids) & set(relevant.keys())
        recall = len(found) / n_relevant if n_relevant else 0

        # Precision
        precision = len(found) / min(k, len(ranked_doc_ids)) if ranked_doc_ids else 0

        # nDCG@k
        dcg = 0.0
        for i, did in enumerate(top_k_ids):
            rel = relevant.get(did, 0)
            dcg += rel / math.log2(i + 2)  # i+2 because rank starts at 1

        # Ideal DCG: sort all relevant by relevance descending
        ideal_rels = sorted(relevant.values(), reverse=True)[:k]
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
        ndcg = dcg / idcg if idcg > 0 else 0

        results[k] = {
            "recall": recall,
            "precision": precision,
            "found": len(found),
            "ndcg": ndcg,
        }

    # MRR (reciprocal rank of first relevant doc)
    mrr = 0.0
    for i, did in enumerate(ranked_doc_ids):
        if did in relevant:
            mrr = 1.0 / (i + 1)
            break

    results["mrr"] = mrr
    results["n_relevant"] = n_relevant

    return results