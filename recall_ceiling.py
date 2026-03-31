"""
Compare raw retrieval recall (before any scoring/filtering) across pipeline modes.

Shows the retrieval ceiling — how many relevant docs are found before ranking.

Usage:
    from recall_ceiling import compare_recall_ceiling

    compare_recall_ceiling(
        topics_path="./topics_rag24_test.txt",
        qrels_path="./qrels.rag24.test-umbrela-all.txt",
        pipeline=pipeline,
        engine=engine,
        tokenizer=tokenizer,
        expansions_path="./keyword_expansions.jsonl",
        max_topics=10,
    )
"""

import time
from pathlib import Path
from tqdm import tqdm
from trec_output import load_qrels
from full_eval import load_topics
from resolve_documents import resolve_all_queries


def retrieval_recall(
    qid: str,
    query_text: str,
    qrels: dict,
    pipeline,
    engine,
    tokenizer,
    index_dir: str = "../msmarco_segmented_index/",
    mode: str = "llm_adaptive",
    expansions_path: str = None,
    max_standalone: int = 5000,
    max_standalone_sup: int = 1000,
    max_refined: int = 50000,
    max_queries: int = 50,
    max_clause_freq: int = 100000,
    filter_mode: str = "stopword",
):
    """
    Run retrieval (no scoring/filtering) and compute raw recall against qrels.

    Returns:
        Dict with retrieval stats, or None if no relevant docs in qrels.
    """
    relevant = set(did for did, rel in qrels.get(qid, {}).items() if rel > 0)
    if not relevant:
        return None

    t0 = time.perf_counter()

    if mode == "llm_adaptive":
        executed, scored, scoring_terms_q = pipeline.build_and_run_llm_adaptive(
            query_text,
            qid=qid,
            engine=engine,
            expansions_path=expansions_path,
            max_standalone=max_standalone,
            max_standalone_sup=max_standalone_sup,
            max_refined=max_refined,
            max_queries=max_queries,
            max_clause_freq=max_clause_freq,
            filter_mode=filter_mode,
            verbose=False,
        )
    elif mode == "splade_adaptive":
        executed, scored = pipeline.build_and_run_adaptive(
            query_text,
            engine=engine,
            max_standalone=max_standalone,
            max_refined=max_refined,
            max_queries=max_queries,
            max_clause_freq=max_clause_freq,
            verbose=False,
        )
    elif mode == "static":
        scored_tokens = pipeline.score_tokens(
            pipeline.encode_query(query_text), min_splade_score=0.3
        )
        if len(scored_tokens) < 2:
            return None
        executed = pipeline.build_and_run(
            query_text,
            engine=engine,
            use_chunking=True,
            max_clause_freq=max_clause_freq,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    t_query = time.perf_counter() - t0

    if not executed:
        return {
            "qid": qid,
            "n_relevant": len(relevant),
            "n_retrieved": 0,
            "n_found": 0,
            "recall": 0.0,
            "time_query": t_query,
            "time_resolve": 0,
            "n_queries": 0,
        }

    # Resolve without filtering
    t0 = time.perf_counter()
    docs = resolve_all_queries(
        queries=executed,
        index_dir=index_dir,
        tokenizer=tokenizer,
        max_doc_len=50,  # minimal, just need doc_id
    )
    t_resolve = time.perf_counter() - t0

    retrieved = set(d["doc_id"] for d in docs if d["doc_id"])
    found = relevant & retrieved

    return {
        "qid": qid,
        "n_relevant": len(relevant),
        "n_retrieved": len(retrieved),
        "n_found": len(found),
        "recall": len(found) / len(relevant) if relevant else 0,
        "time_query": t_query,
        "time_resolve": t_resolve,
        "n_queries": len(executed),
    }


def compare_recall_ceiling(
    topics_path: str,
    qrels_path: str,
    pipeline,
    engine,
    tokenizer,
    index_dir: str = "../msmarco_segmented_index/",
    expansions_paths: dict = None,
    include_splade: bool = True,
    max_topics: int = None,
    max_standalone: int = 5000,
    max_standalone_sup: int = 1000,
    max_refined: int = 50000,
    max_queries: int = 50,
    max_clause_freq: int = 100000,
    filter_mode: str = "stopword",
):
    """
    Compare raw retrieval recall across pipeline modes.

    Args:
        expansions_paths: Dict mapping label -> JSONL path.
            e.g. {"llm_v1": "kw_v1.jsonl", "llm_v2": "kw_v2.jsonl"}
            Each becomes an llm_adaptive mode with that label.
        include_splade: If True, also run splade_adaptive for comparison.
        max_topics: Limit number of topics.
        filter_mode: "stopword" or "noun_phrase" for LLM keyword filtering.
    """
    # Build modes list
    modes = []
    mode_expansions = {}  # mode_name -> expansions_path

    if expansions_paths:
        for label, path in expansions_paths.items():
            modes.append(label)
            mode_expansions[label] = path

    if include_splade:
        modes.append("splade_adaptive")

    if not modes:
        print("No modes to compare!")
        return {}

    topics = load_topics(topics_path)
    if max_topics:
        topics = topics[:max_topics]

    qrels = load_qrels(qrels_path)

    print(f"Comparing retrieval recall ceiling")
    print(f"Modes: {modes}")
    print(f"Topics: {len(topics)}")
    print(f"{'='*80}\n")

    all_results = {mode: [] for mode in modes}

    for qid, query_text in tqdm(topics, desc="Topics"):
        for mode in modes:
            kwargs = {
                "max_standalone": max_standalone,
                "max_standalone_sup": max_standalone_sup,
                "max_refined": max_refined,
                "max_queries": max_queries,
                "max_clause_freq": max_clause_freq,
                "filter_mode": filter_mode,
            }

            if mode in mode_expansions:
                actual_mode = "llm_adaptive"
                kwargs["expansions_path"] = mode_expansions[mode]
            else:
                actual_mode = mode

            try:
                result = retrieval_recall(
                    qid, query_text, qrels, pipeline, engine, tokenizer,
                    index_dir=index_dir, mode=actual_mode, **kwargs,
                )
                if result:
                    all_results[mode].append(result)
            except Exception as e:
                print(f"  [{qid}] {mode} ERROR: {e}")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"{'Mode':<25s} {'Topics':>7s} {'Avg Recall':>12s} {'Avg Retrieved':>15s} "
          f"{'Avg Found':>11s} {'Avg Relevant':>13s} {'Avg Time':>10s}")
    print(f"{'='*80}")

    for mode in modes:
        results = all_results[mode]
        if not results:
            print(f"{mode:<25s}  No results")
            continue

        n = len(results)
        avg_recall = sum(r["recall"] for r in results) / n
        avg_retrieved = sum(r["n_retrieved"] for r in results) / n
        avg_found = sum(r["n_found"] for r in results) / n
        avg_relevant = sum(r["n_relevant"] for r in results) / n
        avg_time = sum(r["time_query"] + r["time_resolve"] for r in results) / n

        print(f"{mode:<25s} {n:>7d} {avg_recall:>12.4f} {avg_retrieved:>15.0f} "
              f"{avg_found:>11.1f} {avg_relevant:>13.1f} {avg_time:>9.2f}s")

    # Per-query comparison
    print(f"\n{'='*80}")
    print(f"Per-query recall comparison:")
    print(f"{'='*80}")
    header = f"{'QID':<15s}"
    for mode in modes:
        header += f" {mode:>20s}"
    header += f" {'Relevant':>9s}"
    print(header)
    print("-" * len(header))

    mode_lookup = {}
    for mode in modes:
        mode_lookup[mode] = {r["qid"]: r for r in all_results[mode]}

    all_qids = set()
    for mode in modes:
        all_qids.update(r["qid"] for r in all_results[mode])

    for qid in sorted(all_qids):
        row = f"{qid:<15s}"
        n_rel = 0
        for mode in modes:
            r = mode_lookup[mode].get(qid)
            if r:
                row += f" {r['n_found']:>6d}/{r['n_retrieved']:<6d}({r['recall']:.2f})"
                n_rel = r["n_relevant"]
            else:
                row += f" {'N/A':>20s}"
        row += f" {n_rel:>9d}"
        print(row)

    # Head-to-head for each pair of modes
    if len(modes) >= 2:
        print(f"\nHead-to-head comparisons:")
        from itertools import combinations
        for mode_a, mode_b in combinations(modes, 2):
            lookup_a = {r["qid"]: r for r in all_results[mode_a]}
            lookup_b = {r["qid"]: r for r in all_results[mode_b]}
            common = set(lookup_a.keys()) & set(lookup_b.keys())
            if not common:
                continue

            a_wins = sum(1 for q in common if lookup_a[q]["recall"] > lookup_b[q]["recall"])
            b_wins = sum(1 for q in common if lookup_b[q]["recall"] > lookup_a[q]["recall"])
            ties = len(common) - a_wins - b_wins
            avg_diff = sum(lookup_a[q]["recall"] - lookup_b[q]["recall"] for q in common) / len(common)

            print(f"  {mode_a} vs {mode_b} ({len(common)} queries):")
            print(f"    {mode_a} wins: {a_wins}, {mode_b} wins: {b_wins}, ties: {ties}")
            print(f"    Avg recall diff: {avg_diff:+.4f}")

    return all_results