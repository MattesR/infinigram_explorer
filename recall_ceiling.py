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

import json
import time
from pathlib import Path
from tqdm import tqdm
from trec_output import load_qrels
from full_eval import load_topics
from resolve_documents import resolve_all_queries
from llm_keyword_filter import STOPWORDS


def retrieval_recall(
    qid: str,
    query_text: str,
    qrels: dict,
    engine,
    tokenizer,
    pipeline=None,
    index_dir: str = "../msmarco_segmented_index/",
    mode: str = "llm_adaptive",
    expansions_path: str = None,
    max_standalone: int = 5000,
    max_standalone_sup: int = 1000,
    max_refined: int = 50000,
    max_queries: int = 50,
    max_clause_freq: int = 100000,
    filter_mode: str = "stopword",
    save_inspection: bool = False,
    inspection_dir: str = None,
    corpus_dir: str = None,
    # Progressive mode params
    max_standalone_key: int = 1000,
    max_standalone_assoc: int = 200,
    prox_peek: int = 10,
    max_docs: int = 10000,
    max_combo_grab: int = 5000,
    prox_cross: int = 50,
    prox_assoc: int = 80,
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

    if mode == "progressive":
        from progressive_queries import build_pieces, peek_and_grab, build_combination_queries
        from adaptive_queries import run_adaptive

        pieces = build_pieces(qid, expansions_path, tokenizer, engine, verbose=False)
        peek = peek_and_grab(pieces, engine, tokenizer, verbose=False,
                             max_standalone_key=max_standalone_key,
                             max_standalone_assoc=max_standalone_assoc,
                             prox_peek=prox_peek,
                             max_clause_freq=max_clause_freq)
        combo = build_combination_queries(peek, engine, tokenizer, verbose=False,
                                           max_docs=max_docs,
                                           max_combo_grab=max_combo_grab,
                                           prox_cross=prox_cross,
                                           prox_assoc=prox_assoc,
                                           max_clause_freq=max_clause_freq)
        all_queries = peek["grabbed"] + combo
        executed = run_adaptive(engine, all_queries, max_clause_freq=max_clause_freq, verbose=False)

    elif mode == "llm_adaptive":
        executed, scored, scoring_terms_q = pipeline.build_and_run_llm_adaptive(
            query_text,
            qid=qid,
            engine=engine,
            expansions_path=expansions_path,
            max_standalone=max_standalone,
            max_standalone_sup=max_standalone_sup,
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
        max_doc_len=200 if save_inspection else 50,
    )
    t_resolve = time.perf_counter() - t0

    retrieved_map = {d["doc_id"]: d for d in docs if d["doc_id"]}
    retrieved = set(retrieved_map.keys())
    found = relevant & retrieved

    result = {
        "qid": qid,
        "n_relevant": len(relevant),
        "n_retrieved": len(retrieved),
        "n_found": len(found),
        "recall": len(found) / len(relevant) if relevant else 0,
        "time_query": t_query,
        "time_resolve": t_resolve,
        "n_queries": len(executed),
        "docs": docs,
        "relevant": relevant,
    }

    # Store peek and executed for progressive mode (needed for scorer)
    if mode == "progressive":
        result["peek"] = peek
        result["executed"] = executed

    # Save inspection files if requested
    if save_inspection and inspection_dir:
        import os
        os.makedirs(inspection_dir, exist_ok=True)

        found_docs = {did: retrieved_map[did] for did in found}
        missed_docs = {did: rel for did, rel in qrels.get(qid, {}).items()
                      if rel > 0 and did not in retrieved}
        irrelevant_docs = {did: d for did, d in retrieved_map.items()
                         if did not in relevant}

        def _to_record(did, doc, rel=None):
            r = {
                "doc_id": did,
                "text": doc.get("text", "")[:2000] if doc.get("text") else "",
                "from_queries": doc.get("from_queries", []),
            }
            if rel is not None:
                r["relevance"] = rel
            return r

        # Found
        path = os.path.join(inspection_dir, f"{qid}_found.jsonl")
        with open(path, "w") as f:
            for did in sorted(found, key=lambda d: qrels.get(qid, {}).get(d, 0), reverse=True):
                f.write(json.dumps(_to_record(did, retrieved_map[did],
                        rel=qrels.get(qid, {}).get(did, 0))) + "\n")

        # Missed — look up text from corpus if available
        missed_texts = {}
        if corpus_dir and missed_docs:
            from retrieval_inspector import _lookup_docs_from_corpus
            missed_texts = _lookup_docs_from_corpus(set(missed_docs.keys()), corpus_dir)

        path = os.path.join(inspection_dir, f"{qid}_missed.jsonl")
        with open(path, "w") as f:
            for did, rel in sorted(missed_docs.items(), key=lambda x: x[1], reverse=True):
                record = {"doc_id": did, "relevance": rel}
                if did in missed_texts:
                    doc_obj = missed_texts[did]
                    text = doc_obj.get("segment", doc_obj.get("body", doc_obj.get("text", "")))
                    record["text"] = text[:2000]
                f.write(json.dumps(record) + "\n")

        # Top irrelevant (random sample since no crude score here)
        path = os.path.join(inspection_dir, f"{qid}_irrelevant.jsonl")
        with open(path, "w") as f:
            for i, (did, doc) in enumerate(irrelevant_docs.items()):
                if i >= 100:
                    break
                f.write(json.dumps(_to_record(did, doc)) + "\n")

        # Info
        path = os.path.join(inspection_dir, f"{qid}_info.json")
        with open(path, "w") as f:
            json.dump({
                "qid": qid,
                "query": query_text,
                "n_relevant": len(relevant),
                "n_retrieved": len(retrieved),
                "n_found": len(found),
                "n_missed": len(missed_docs),
                "recall": result["recall"],
                "queries": [
                    {"description": q.get("description", ""), "count": q.get("cnt", 0)}
                    for q in executed
                ],
            }, f, indent=2)

    return result


def compare_recall_ceiling(
    topics_path: str,
    qrels_path: str,
    engine,
    tokenizer,
    pipeline=None,
    index_dir: str = "../msmarco_segmented_index/",
    expansions_paths: dict = None,
    progressive_paths: dict = None,
    include_splade: bool = False,
    max_topics: int = None,
    max_standalone: int = 5000,
    max_standalone_sup: int = 1000,
    max_refined: int = 50000,
    max_queries: int = 50,
    max_clause_freq: int = 80000000,
    filter_mode: str = "stopword",
    save_inspection: bool = False,
    inspection_dir: str = "./inspection",
    corpus_dir: str = "../data/infinigram_index/msmarco_v2.1_doc_segmented",
    # Progressive mode params
    max_standalone_key: int = 1000,
    max_standalone_assoc: int = 200,
    prox_peek: int = 10,
    max_docs: int = 10000,
    max_combo_grab: int = 5000,
    prox_cross: int = 50,
    prox_assoc: int = 80,
    # Return docs for reranking
    return_docs: bool = False,
):
    """
    Compare raw retrieval recall across pipeline modes.

    Args:
        expansions_paths: Dict mapping label -> JSONL path for llm_adaptive mode.
        progressive_paths: Dict mapping label -> JSONL path for progressive mode.
        include_splade: If True, also run splade_adaptive for comparison.
    """
    # Build modes list
    modes = []
    mode_expansions = {}   # label -> path (llm_adaptive)
    mode_progressive = {}  # label -> path (progressive)

    if expansions_paths:
        for label, path in expansions_paths.items():
            modes.append(label)
            mode_expansions[label] = path

    if progressive_paths:
        for label, path in progressive_paths.items():
            modes.append(label)
            mode_progressive[label] = path

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
                "save_inspection": save_inspection,
                # Progressive params
                "max_standalone_key": max_standalone_key,
                "max_standalone_assoc": max_standalone_assoc,
                "prox_peek": prox_peek,
                "max_docs": max_docs,
                "max_combo_grab": max_combo_grab,
                "prox_cross": prox_cross,
                "prox_assoc": prox_assoc,
            }

            if save_inspection:
                import os
                mode_dir = os.path.join(inspection_dir, mode)
                os.makedirs(mode_dir, exist_ok=True)
                kwargs["inspection_dir"] = mode_dir
                if corpus_dir:
                    kwargs["corpus_dir"] = corpus_dir

            if mode in mode_progressive:
                actual_mode = "progressive"
                kwargs["expansions_path"] = mode_progressive[mode]
            elif mode in mode_expansions:
                actual_mode = "llm_adaptive"
                kwargs["expansions_path"] = mode_expansions[mode]
            else:
                actual_mode = mode

            try:
                result = retrieval_recall(
                    qid, query_text, qrels, engine, tokenizer,
                    index_dir=index_dir, mode=actual_mode, **kwargs,
                )
                if result:
                    if not return_docs:
                        result.pop("docs", None)
                        result.pop("relevant", None)
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