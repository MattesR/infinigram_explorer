"""
Produce TREC-format run files from pipeline output and evaluate
against qrels using pytrec_eval.

Run file format (one line per result):
    query_id Q0 doc_id rank score run_name

Qrel file format (one line per judgment):
    query_id 0 doc_id relevance

Usage:
    from trec_output import write_run, evaluate_run, full_evaluate

    # Write a run file
    write_run(query_id="123", docs=docs, run_name="infinigram_splade", path="run.txt")

    # Evaluate against qrels
    metrics = evaluate_run("run.txt", "qrels.txt")

    # Or do both in one call
    metrics = full_evaluate(
        query_id="123",
        docs=docs,
        qrels_path="qrels.txt",
        run_name="infinigram_splade",
    )
"""

import os
from collections import defaultdict


def write_run(
    query_id: str,
    docs: list[dict],
    run_name: str = "infinigram_splade",
    path: str = "run.txt",
    score_field: str = "crude_score",
    append: bool = False,
):
    """
    Write a TREC-format run file from ranked documents.

    Args:
        query_id: Query identifier (must match qrels).
        docs: List of document dicts (from resolve_all_queries), must have
              'doc_id' and a score field. Should already be sorted by score.
        run_name: Name of the run (appears in last column).
        path: Output file path.
        score_field: Which field to use as the score. Common options:
            'crude_score' (from crude SPLADE filter) or 'score' (from SPLADE rerank).
        append: If True, append to existing file (for multiple queries).
    """
    mode = "a" if append else "w"
    n_written = 0

    with open(path, mode) as f:
        for rank, doc in enumerate(docs, 1):
            doc_id = doc.get("doc_id")
            if doc_id is None:
                continue
            score = doc.get(score_field, 0.0)
            f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")
            n_written += 1

    action = "Appended" if append else "Wrote"
    print(f"{action} {n_written} results for query {query_id} to {path}")


def write_run_multi(
    results: dict,
    run_name: str = "infinigram_splade",
    path: str = "run.txt",
    score_field: str = "crude_score",
):
    """
    Write a run file for multiple queries.

    Args:
        results: Dict mapping query_id -> list of document dicts (sorted by score).
        run_name: Name of the run.
        path: Output file path.
        score_field: Which field to use as the score.
    """
    total = 0
    with open(path, "w") as f:
        for query_id, docs in results.items():
            for rank, doc in enumerate(docs, 1):
                doc_id = doc.get("doc_id")
                if doc_id is None:
                    continue
                score = doc.get(score_field, 0.0)
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")
                total += 1

    print(f"Wrote {total} results for {len(results)} queries to {path}")


def load_qrels(path: str) -> dict[str, dict[str, int]]:
    """
    Load qrels in TREC format.
    Handles both 3-column (qid docid rel) and 4-column (qid iter docid rel).

    Returns:
        {query_id: {doc_id: relevance}}
    """
    qrels = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qid, _, docid, rel = parts
            elif len(parts) == 3:
                qid, docid, rel = parts
            else:
                continue
            qrels[qid][docid] = int(rel)
    return dict(qrels)


def load_run(path: str) -> dict[str, dict[str, float]]:
    """
    Load a TREC run file.

    Returns:
        {query_id: {doc_id: score}}
    """
    run = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                qid, _, docid, rank, score = parts[:5]
                run[qid][docid] = float(score)
    return dict(run)


def evaluate_run(
    run_path: str,
    qrels_path: str,
    metrics: list[str] = None,
    k_values: list[int] = None,
) -> dict:
    """
    Evaluate a run file against qrels using pytrec_eval.

    Args:
        run_path: Path to TREC run file.
        qrels_path: Path to TREC qrels file.
        metrics: List of metric names. Default: common IR metrics.
        k_values: Cutoff values for metrics. Default: [10, 100, 1000].

    Returns:
        Dict with per-query and aggregated metrics.
    """
    if k_values is None:
        k_values = [10, 100, 1000]

    if metrics is None:
        metrics = []
        for k in k_values:
            metrics.extend([
                f"ndcg_cut_{k}",
                f"recall_{k}",
            ])
        metrics.extend(["recip_rank", "map"])

    try:
        import pytrec_eval
    except ImportError:
        print("pytrec_eval not installed. Install with: pip install pytrec-eval-terrier")
        print("Falling back to manual evaluation...")
        return _evaluate_manual(run_path, qrels_path, k_values)

    qrels = load_qrels(qrels_path)
    run = load_run(run_path)

    # Filter to queries that have both run results and qrels
    common_qids = set(qrels.keys()) & set(run.keys())
    if not common_qids:
        print("No overlapping query IDs between run and qrels!")
        print(f"  Run queries: {list(run.keys())[:5]}...")
        print(f"  Qrel queries: {list(qrels.keys())[:5]}...")
        return {}

    print(f"Evaluating {len(common_qids)} queries with pytrec_eval...")

    # Filter qrels and run to common queries
    qrels_filtered = {qid: qrels[qid] for qid in common_qids}
    run_filtered = {qid: run[qid] for qid in common_qids}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_filtered, set(metrics))
    per_query = evaluator.evaluate(run_filtered)

    # Aggregate
    agg = defaultdict(list)
    for qid, qmetrics in per_query.items():
        for metric, value in qmetrics.items():
            agg[metric].append(value)

    print(f"\nAggregated metrics ({len(common_qids)} queries):")
    print(f"{'Metric':<25s} {'Mean':>10s}")
    print("-" * 37)
    aggregated = {}
    for metric in sorted(agg.keys()):
        values = agg[metric]
        mean = sum(values) / len(values)
        aggregated[metric] = mean
        print(f"  {metric:<23s} {mean:>10.4f}")

    return {
        "per_query": per_query,
        "aggregated": aggregated,
        "n_queries": len(common_qids),
    }


def _evaluate_manual(
    run_path: str,
    qrels_path: str,
    k_values: list[int],
) -> dict:
    """
    Simple manual evaluation when pytrec_eval is not available.
    Computes MRR, Recall@k, and nDCG@k.
    """
    import math

    qrels = load_qrels(qrels_path)
    run_data = defaultdict(list)

    with open(run_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, docid, rank, score, _ = parts[:6]
                run_data[qid].append((int(rank), docid, float(score)))

    common_qids = set(qrels.keys()) & set(run_data.keys())
    if not common_qids:
        print("No overlapping query IDs!")
        return {}

    print(f"Evaluating {len(common_qids)} queries (manual)...")

    aggregated = defaultdict(list)

    for qid in common_qids:
        results = sorted(run_data[qid], key=lambda x: x[0])
        relevant = {did for did, rel in qrels[qid].items() if rel > 0}

        # MRR
        rr = 0.0
        for rank, docid, _ in results:
            if docid in relevant:
                rr = 1.0 / rank
                break
        aggregated["recip_rank"].append(rr)

        for k in k_values:
            top_k = results[:k]
            top_k_ids = [docid for _, docid, _ in top_k]

            # Recall@k
            found = sum(1 for d in top_k_ids if d in relevant)
            recall = found / len(relevant) if relevant else 0.0
            aggregated[f"recall_{k}"].append(recall)

            # nDCG@k
            dcg = sum(
                (1 if docid in relevant else 0) / math.log2(i + 2)
                for i, (_, docid, _) in enumerate(top_k)
            )
            ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
            ndcg = dcg / ideal if ideal > 0 else 0.0
            aggregated[f"ndcg_cut_{k}"].append(ndcg)

    print(f"\nAggregated metrics ({len(common_qids)} queries):")
    print(f"{'Metric':<25s} {'Mean':>10s}")
    print("-" * 37)
    result = {}
    for metric in sorted(aggregated.keys()):
        values = aggregated[metric]
        mean = sum(values) / len(values)
        result[metric] = mean
        print(f"  {metric:<23s} {mean:>10.4f}")

    return {"aggregated": result, "n_queries": len(common_qids)}


def full_evaluate(
    query_id: str,
    docs: list[dict],
    qrels_path: str,
    run_name: str = "infinigram_splade",
    score_field: str = "crude_score",
    run_path: str = "/tmp/run.txt",
    **eval_kwargs,
) -> dict:
    """
    Write run file and evaluate in one call.

    Args:
        query_id: Query ID matching the qrels.
        docs: Ranked document list from the pipeline.
        qrels_path: Path to qrels file.
        run_name: Run identifier.
        score_field: Score field in docs to use for ranking.
        run_path: Temporary run file path.
        **eval_kwargs: Passed to evaluate_run.

    Returns:
        Evaluation metrics dict.
    """
    write_run(query_id, docs, run_name=run_name, path=run_path, score_field=score_field)
    return evaluate_run(run_path, qrels_path, **eval_kwargs)