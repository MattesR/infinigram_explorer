"""
End-to-end retrieval pipeline and evaluation.

Usage:
    from pipeline import run_pipeline, run_batch, evaluate

    # Load resources once
    resources = {
        "engine": engine,
        "tokenizer": tokenizer,
        "biencoder_model": bi_model,       # or None to skip
        "crossencoder_model": ce_model,    # or None to skip
    }

    config = DEFAULT_CONFIG.copy()

    # Single query
    result = run_pipeline(
        qid="2024-32912",
        query_text="how did the Vietnam War...",
        expansion=expansions["2024-32912"],
        config=config,
        resources=resources,
    )

    # Batch
    results = run_batch(
        topics_path="./topics_rag24_test.txt",
        expansions_path="./kiss.jsonl",
        config=config,
        resources=resources,
        max_topics=10,
    )

    # Evaluate
    eval_results = evaluate(
        results,
        qrels_path="./qrels.rag24.test-umbrela-all.txt",
    )
"""

import time
from tqdm import tqdm
from trec_output import load_qrels
from full_eval import load_topics
from llm_keyword_filter import load_all_expansions
from ir_metrics import compute_metrics


DEFAULT_CONFIG = {
    # Peek and grab
    "max_standalone_key": 1500,
    "max_standalone_assoc": 750,
    "prox_peek": 10,
    "max_clause_freq": 2000000,

    # Budget and proximity
    "max_budget": 20000,
    "prox_cross": 100,
    "max_assoc_combo": 50000,

    # Resolve
    "max_doc_len": 2000,
    "index_dir": "../msmarco_segmented_index/",

    # Bi-encoder
    "biencoder_top_k": 1000,
    "biencoder_batch_size": 256,
    "biencoder_max_doc_length": 512,

    # Cross-encoder
    "crossencoder_top_k": 100,
    "crossencoder_batch_size": 32,
    "crossencoder_max_length": 512,
}


def _build_pieces_from_expansion(qid, expansion, tokenizer, engine):
    """Build CNF pieces directly from expansion dict (no file needed)."""
    from progressive_queries import _make_term_piece

    key_pieces = {}
    associated = []

    key_entities = expansion.get("KEY_ENTITIES", {})
    for name, terms in key_entities.items():
        if isinstance(terms, dict):
            flat = []
            for level in ["lexical", "conceptual", "referential"]:
                flat.extend(terms.get(level, []))
            terms = flat

        all_terms = [name] + (terms if isinstance(terms, list) else [])
        aspect_pieces = []

        for kw in all_terms:
            p = _make_term_piece(kw, tokenizer, engine)
            if p:
                p["source_aspect"] = name
                aspect_pieces.append(p)

        key_pieces[name] = aspect_pieces

    for kw in expansion.get("ASSOCIATED_TERMS", expansion.get("ASSOCIATED", [])):
        p = _make_term_piece(kw, tokenizer, engine)
        if p:
            associated.append(p)

    return {"key_pieces": key_pieces, "associated": associated}


def run_pipeline(
    qid: str,
    query_text: str,
    expansion: dict,
    config: dict,
    resources: dict,
    verbose: bool = False,
):
    """
    Run full retrieval pipeline for a single query.

    Args:
        qid: Query ID.
        query_text: Original query text.
        expansion: Keyword expansion dict with KEY_ENTITIES, ASSOCIATED_TERMS, VERBS.
        config: Hyperparameter dict (see DEFAULT_CONFIG).
        resources: Shared resources dict with engine, tokenizer, models.
        verbose: Print progress.

    Returns result dict with all intermediate outputs.
    """
    engine = resources["engine"]
    tokenizer = resources["tokenizer"]
    bi_model = resources.get("biencoder_model")
    ce_model = resources.get("crossencoder_model")

    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config)
    timings = {}

    # ================================================================
    # Step 1-2: Two-phase peek and grab
    # ================================================================
    t0 = time.perf_counter()

    from peek_grab_v2 import peek_and_grab_v2

    peek_result = peek_and_grab_v2(
        qid=qid,
        expansions_path=None,  # not used, pieces built below
        tokenizer=tokenizer,
        engine=engine,
        max_standalone_key=cfg["max_standalone_key"],
        max_standalone_assoc=cfg["max_standalone_assoc"],
        max_clause_freq=cfg["max_clause_freq"],
        prox_peek=cfg["prox_peek"],
        prox_cross=cfg["prox_cross"],
        max_budget=cfg["max_budget"],
        max_assoc_combo=cfg.get("max_assoc_combo", 50000),
        verbose=verbose,
        # Pass pre-built pieces
        _pieces=_build_pieces_from_expansion(qid, expansion, tokenizer, engine),
    )

    all_queries = peek_result["grabbed"] + peek_result["combo_queries"]
    timings["peek_and_combo"] = time.perf_counter() - t0

    # ================================================================
    # Step 2: Execute and resolve
    # ================================================================
    t0 = time.perf_counter()

    from adaptive_queries import run_adaptive
    from resolve_documents import resolve_all_queries

    if verbose:
        print(f"  [{qid}] Executing {len(all_queries)} queries...")

    executed = run_adaptive(
        engine, all_queries,
        max_clause_freq=cfg["max_clause_freq"],
        verbose=verbose,
    )

    pool_docs = resolve_all_queries(
        executed,
        index_dir=cfg["index_dir"],
        tokenizer=tokenizer,
        max_doc_len=cfg["max_doc_len"],
    )

    if not pool_docs:
        print(f"  [{qid}] WARNING: 0 docs resolved")
        print(f"    All queries: {len(all_queries)}, Executed: {len(executed)}")
        print(f"    Grabbed: {len(peek.get('grabbed', []))}")
        for q in all_queries[:5]:
            print(f"      {q.get('description', '?')}: cnt={q.get('estimated_count', '?')}")

    # Deduplicate
    seen = set()
    unique_docs = []
    for d in pool_docs:
        did = d.get("doc_id", "")
        if did and did not in seen:
            seen.add(did)
            unique_docs.append(d)
    pool_docs = unique_docs

    timings["resolve"] = time.perf_counter() - t0

    if verbose:
        print(f"  [{qid}] Pool: {len(pool_docs)} docs, "
              f"{len(executed)} queries, {timings['peek']+timings['combo']+timings['resolve']:.1f}s")

    # ================================================================
    # Step 5: Bi-encoder reranking
    # ================================================================
    biencoder_ranked = None
    if bi_model is not None and pool_docs:
        t0 = time.perf_counter()

        from biencoder_rank import rerank_biencoder

        ranked = rerank_biencoder(
            query_text, pool_docs, bi_model,
            top_k=cfg["biencoder_top_k"],
            batch_size=cfg["biencoder_batch_size"],
            max_doc_length=cfg["biencoder_max_doc_length"],
        )

        # Build ranked doc list with texts
        doc_map = {d["doc_id"]: d for d in pool_docs}
        biencoder_ranked = []
        for score, did in ranked:
            doc = doc_map.get(did, {"doc_id": did, "text": ""})
            biencoder_ranked.append({
                "doc_id": did,
                "text": doc.get("text", ""),
                "score": score,
            })

        timings["biencoder"] = time.perf_counter() - t0

        if verbose:
            print(f"  [{qid}] Bi-encoder: {len(biencoder_ranked)} docs, {timings['biencoder']:.1f}s")

    # ================================================================
    # Step 6: Cross-encoder reranking
    # ================================================================
    crossencoder_ranked = None
    if ce_model is not None and biencoder_ranked:
        t0 = time.perf_counter()

        from crossencoder_rank import rerank_crossencoder

        ce_input = biencoder_ranked[:cfg["biencoder_top_k"]]
        ce_doc_ids = [d["doc_id"] for d in ce_input]
        ce_texts = [d["text"] for d in ce_input]

        ce_ranked = rerank_crossencoder(
            query_text, ce_doc_ids, ce_texts, ce_model,
            top_k=cfg["crossencoder_top_k"],
            batch_size=cfg["crossencoder_batch_size"],
            max_length=cfg["crossencoder_max_length"],
        )

        doc_map_ce = {d["doc_id"]: d for d in ce_input}
        crossencoder_ranked = []
        for score, did in ce_ranked:
            doc = doc_map_ce.get(did, {"doc_id": did, "text": ""})
            crossencoder_ranked.append({
                "doc_id": did,
                "text": doc.get("text", ""),
                "score": score,
            })

        timings["crossencoder"] = time.perf_counter() - t0

        if verbose:
            print(f"  [{qid}] Cross-encoder: {len(crossencoder_ranked)} docs, {timings['crossencoder']:.1f}s")

    return {
        "qid": qid,
        "query_text": query_text,
        "peek_result": peek_result,
        "pool_docs": pool_docs,
        "pool_size": len(pool_docs),
        "executed_queries": executed,
        "biencoder_ranked": biencoder_ranked,
        "crossencoder_ranked": crossencoder_ranked,
        "timings": timings,
        "config": cfg,
    }


def run_batch(
    topics_path: str,
    expansions_path: str,
    config: dict,
    resources: dict,
    max_topics: int = None,
    verbose: bool = True,
):
    """
    Run pipeline for multiple queries.

    Returns list of result dicts.
    """
    topics = load_topics(topics_path)
    if max_topics:
        topics = topics[:max_topics]

    expansions = load_all_expansions(expansions_path)

    results = []
    for qid, query_text in tqdm(topics, desc="Pipeline", disable=not verbose):
        expansion = expansions.get(qid, {})
        if not expansion:
            if verbose:
                print(f"  [{qid}] No expansion, skipping")
            continue

        result = run_pipeline(
            qid=qid,
            query_text=query_text,
            expansion=expansion,
            config=config,
            resources=resources,
            verbose=False,
        )
        results.append(result)

        if verbose:
            t_total = sum(result["timings"].values())
            print(f"  [{qid}] pool={result['pool_size']}, "
                  f"bi={len(result['biencoder_ranked'] or [])}, "
                  f"ce={len(result['crossencoder_ranked'] or [])}, "
                  f"time={t_total:.1f}s")

    return results


def evaluate(
    results: list,
    qrels_path: str,
    stages: list = None,
    top_k_list: list = None,
    verbose: bool = True,
):
    """
    Evaluate pipeline results against qrels.

    Evaluates each stage independently:
    - pool: recall ceiling of retrieval pool
    - biencoder: bi-encoder ranked list
    - crossencoder: cross-encoder ranked list

    Args:
        results: List of result dicts from run_pipeline/run_batch.
        qrels_path: Path to qrels file.
        stages: Which stages to evaluate. Default: all available.
        top_k_list: Cutoffs for ranked stages.

    Returns dict of stage -> list of per-query eval dicts.
    """
    if top_k_list is None:
        top_k_list = [10, 100, 1000]

    if stages is None:
        stages = ["pool", "biencoder", "crossencoder"]

    qrels = load_qrels(qrels_path)

    eval_results = {stage: [] for stage in stages}

    for r in results:
        qid = r["qid"]
        relevant = {did: rel for did, rel in qrels.get(qid, {}).items() if rel > 0}
        n_relevant = len(relevant)

        # Pool evaluation
        if "pool" in stages:
            pool_ids = [d["doc_id"] for d in r.get("pool_docs", [])]
            pool_found = len(set(pool_ids) & set(relevant.keys()))
            pool_metrics = compute_metrics(pool_ids, qrels, qid, top_k_list=top_k_list)

            eval_results["pool"].append({
                "qid": qid,
                "n_relevant": n_relevant,
                "pool_size": r.get("pool_size", len(pool_ids)),
                "pool_found": pool_found,
                "pool_recall": pool_found / n_relevant if n_relevant else 0,
                "cutoffs": pool_metrics,
                "n_queries": len(r.get("executed_queries", [])),
                "time": sum(r.get("timings", {}).get(k, 0)
                           for k in ["peek", "combo", "resolve"]),
            })

        # Bi-encoder evaluation
        if "biencoder" in stages and r.get("biencoder_ranked"):
            bi_ids = [d["doc_id"] for d in r["biencoder_ranked"]]
            pool_ids = [d["doc_id"] for d in r.get("pool_docs", [])]
            pool_found = len(set(pool_ids) & set(relevant.keys()))
            bi_metrics = compute_metrics(bi_ids, qrels, qid, top_k_list=top_k_list)

            eval_results["biencoder"].append({
                "qid": qid,
                "n_relevant": n_relevant,
                "pool_found": pool_found,
                "n_input": r.get("pool_size", 0),
                "n_output": len(bi_ids),
                "cutoffs": bi_metrics,
                "time": r.get("timings", {}).get("biencoder", 0),
            })

        # Cross-encoder evaluation
        if "crossencoder" in stages and r.get("crossencoder_ranked"):
            ce_ids = [d["doc_id"] for d in r["crossencoder_ranked"]]
            pool_ids = [d["doc_id"] for d in r.get("pool_docs", [])]
            pool_found = len(set(pool_ids) & set(relevant.keys()))
            ce_metrics = compute_metrics(ce_ids, qrels, qid, top_k_list=top_k_list)

            eval_results["crossencoder"].append({
                "qid": qid,
                "n_relevant": n_relevant,
                "pool_found": pool_found,
                "n_input": len(r.get("biencoder_ranked", [])),
                "n_output": len(ce_ids),
                "cutoffs": ce_metrics,
                "time": r.get("timings", {}).get("crossencoder", 0),
            })

    if verbose:
        _print_eval(eval_results, stages, top_k_list)

    return eval_results


def _print_eval(eval_results, stages, top_k_list):
    """Print evaluation summary."""
    print(f"\n{'='*90}")
    print(f"PIPELINE EVALUATION")
    print(f"{'='*90}")

    for stage in stages:
        results = eval_results.get(stage, [])
        if not results:
            continue

        n = len(results)

        print(f"\n{'─'*90}")
        print(f"Stage: {stage.upper()} ({n} topics)")
        print(f"{'─'*90}")

        if stage == "pool":
            avg_pool_recall = sum(r["pool_recall"] for r in results) / n
            avg_pool_size = sum(r["pool_size"] for r in results) / n
            avg_queries = sum(r["n_queries"] for r in results) / n
            avg_time = sum(r["time"] for r in results) / n
            print(f"  Avg pool recall: {avg_pool_recall:.4f}")
            print(f"  Avg pool size: {avg_pool_size:.0f}")
            print(f"  Avg queries: {avg_queries:.1f}")
            print(f"  Avg time: {avg_time:.1f}s")
        else:
            avg_mrr = sum(r["cutoffs"]["mrr"] for r in results) / n
            avg_time = sum(r["time"] for r in results) / n
            avg_input = sum(r["n_input"] for r in results) / n
            avg_output = sum(r["n_output"] for r in results) / n
            print(f"  MRR: {avg_mrr:.4f}")
            print(f"  Avg input: {avg_input:.0f} → output: {avg_output:.0f}")
            print(f"  Avg time: {avg_time:.1f}s")

        # Cutoff table
        header = f"  {'':>15s}"
        for k in top_k_list:
            header += f" {'R@'+str(k):>8s} {'nDCG@'+str(k):>8s}"
        if stage != "pool":
            header += f" {'%Pool':>6s}"
        print(header)

        print(f"  {'Average':>15s}", end="")
        max_k = max(top_k_list)
        for k in top_k_list:
            recalls = [r["cutoffs"][k]["recall"] for r in results]
            ndcgs = [r["cutoffs"][k]["ndcg"] for r in results]
            print(f" {sum(recalls)/n:>8.4f} {sum(ndcgs)/n:>8.4f}", end="")
        if stage != "pool":
            pool_pcts = [r["cutoffs"][max_k]["found"] / r["pool_found"]
                        if r["pool_found"] > 0 else 0 for r in results]
            print(f" {sum(pool_pcts)/n:>5.0%}", end="")
        print()

        # Per-query
        if stage == "pool":
            print(f"\n  {'QID':<15s} {'Size':>6s} {'Rel':>5s} {'Found':>5s} {'R_pool':>6s} {'Time':>5s}", end="")
        else:
            print(f"\n  {'QID':<15s} {'In':>6s} {'Out':>5s} {'Rel':>5s} {'Pool':>5s} {'MRR':>5s} {'Time':>5s}", end="")
        for k in top_k_list:
            print(f" {'R@'+str(k):>7s} {'nD@'+str(k):>6s}", end="")
        print()
        print(f"  {'-'*85}")

        for r in results:
            if stage == "pool":
                print(f"  {r['qid']:<15s} {r['pool_size']:>6d} {r['n_relevant']:>5d} "
                      f"{r['pool_found']:>5d} {r['pool_recall']:>5.0%} {r['time']:>4.1f}s", end="")
            else:
                print(f"  {r['qid']:<15s} {r['n_input']:>6d} {r['n_output']:>5d} "
                      f"{r['n_relevant']:>5d} {r['pool_found']:>5d} "
                      f"{r['cutoffs']['mrr']:>5.3f} {r['time']:>4.1f}s", end="")
            for k in top_k_list:
                print(f" {r['cutoffs'][k]['recall']:>7.3f} {r['cutoffs'][k]['ndcg']:>6.3f}", end="")
            print()