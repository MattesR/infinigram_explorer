"""
Full evaluation run over a set of TREC topics.

Usage:
    from full_eval import run_full_eval

    results = run_full_eval(
        topics_path="topics_rag24_test.txt",
        qrels_path="qrels_rag24_test-umbrela-all.txt",
        pipeline=pipeline,
        engine=engine,
        tokenizer=tokenizer,
        index_dir="../msmarco_segmented_index/",
    )
"""

import json
import time
from pathlib import Path
from tqdm import tqdm
from resolve_documents import resolve_all_queries
from trec_output import write_run, evaluate_run


def load_topics(path: str) -> list[tuple[str, str]]:
    """
    Load topics file. Supports:
    - TSV: query_id\\tquery_text
    - JSONL: {"id": "...", "title": "..."}

    Returns:
        List of (query_id, query_text) tuples.
    """
    topics = []
    with open(path) as f:
        first_line = f.readline().strip()
        f.seek(0)

        if first_line.startswith("{"):
            # JSONL format
            for line in f:
                obj = json.loads(line.strip())
                qid = str(obj.get("id", obj.get("qid", "")))
                text = obj.get("title", obj.get("text", obj.get("query", "")))
                if qid and text:
                    topics.append((qid, text))
        else:
            # TSV format
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    topics.append((parts[0], parts[1]))

    return topics


def run_full_eval(
    topics_path: str,
    qrels_path: str,
    pipeline,
    engine,
    tokenizer,
    index_dir: str = "../msmarco_segmented_index/",
    run_path: str = "run_full.txt",
    run_name: str = "infinigram_splade",
    max_doc_len: int = 200,
    top_splade_filter: int = 100,
    max_clause_freq: int = 100000,
    use_chunking: bool = True,
    min_splade_score: float = 0.3,
    anchor_score: float = 0.9,
    max_anchor_tup: float = 1e-4,
    min_cluster_score: float = 1.0,
    strategy: str = "anchor",
    max_queries_per_topic: int = 50,
    min_retrieved_docs: int = None,
    lower_query_bound: float = None,
    max_topics: int = None,
    score_field: str = "crude_score",
    adaptive: bool = False,
    max_standalone: int = 5000,
    max_standalone_sup: int = 1000,
    max_refined: int = 20000,
    llm_adaptive: bool = False,
    expansions_path: str = None,
    max_count: int = 500000,
    use_core_only: bool = False,
    filter_mode: str = "stopword",
):
    """
    Run the full pipeline on all topics, write a TREC run file, and evaluate.

    Args:
        topics_path: Path to topics file (TSV or JSONL).
        qrels_path: Path to qrels file.
        pipeline: QueryPipeline instance.
        engine: Infini-gram engine.
        tokenizer: Infini-gram tokenizer.
        index_dir: Path to index directory.
        run_path: Output run file path.
        run_name: Run identifier.
        max_doc_len: Max tokens to read per document.
        top_splade_filter: How many docs to keep after crude scoring.
        max_clause_freq: Passed to engine.find_cnf.
        use_chunking: Use spaCy syntactic linking.
        min_splade_score: Min SPLADE score for token filtering.
        anchor_score: Min SPLADE score for anchor clusters.
        max_anchor_tup: Max TUP for anchor/informative tokens.
        min_cluster_score: Min combined score for non-anchor units.
        strategy: CNF query construction strategy.
        max_queries_per_topic: Max CNF queries per topic.
        min_retrieved_docs: Stop executing queries after this many pointers.
        lower_query_bound: Skip queries with score below this.
        max_topics: If set, only process this many topics.
        score_field: Which score to use for ranking in the run file.

    Returns:
        Dict with evaluation metrics and timing info.
    """
    topics = load_topics(topics_path)
    if max_topics:
        topics = topics[:max_topics]

    print(f"Loaded {len(topics)} topics from {topics_path}")
    print(f"Output: {run_path}")
    mode = "llm_adaptive" if llm_adaptive else ("adaptive" if adaptive else "static")
    print(f"Settings: mode={mode}, top_splade_filter={top_splade_filter}, "
          f"max_clause_freq={max_clause_freq}")
    if llm_adaptive:
        print(f"  expansions_path={expansions_path}")
        print(f"  max_standalone={max_standalone}, max_refined={max_refined}, max_count={max_count}")
    elif adaptive:
        print(f"  max_standalone={max_standalone}, max_refined={max_refined}")
    if min_retrieved_docs:
        print(f"  min_retrieved_docs={min_retrieved_docs}")
    if lower_query_bound:
        print(f"  lower_query_bound={lower_query_bound}")
    print(f"{'='*70}\n")

    timings = []
    total_docs = 0
    failed = []

    # Clear run file
    open(run_path, "w").close()

    for topic_idx, (qid, query_text) in enumerate(tqdm(topics, desc="Topics")):
        t_start = time.perf_counter()
        timing = {"qid": qid}

        try:
            scoring_terms_q = None

            if llm_adaptive:
                # LLM keyword-driven adaptive pipeline
                if not expansions_path:
                    raise ValueError("llm_adaptive requires expansions_path")
                t0 = time.perf_counter()
                queries, scored, scoring_terms_q = pipeline.build_and_run_llm_adaptive(
                    query_text,
                    qid=qid,
                    engine=engine,
                    expansions_path=expansions_path,
                    min_splade_score=min_splade_score,
                    max_standalone=max_standalone,
                    max_standalone_sup=max_standalone_sup,
                    max_count=max_count,
                    max_queries=max_queries_per_topic,
                    max_clause_freq=max_clause_freq,
                    min_retrieved_docs=min_retrieved_docs,
                    use_core_only=use_core_only,
                    filter_mode=filter_mode,
                    verbose=False,
                )
                timing["encode"] = 0
                timing["query_build_run"] = time.perf_counter() - t0

            elif adaptive:
                # Adaptive pipeline: encode + score + build + execute in one call
                t0 = time.perf_counter()
                queries, scored = pipeline.build_and_run_adaptive(
                    query_text,
                    engine=engine,
                    min_splade_score=min_splade_score,
                    anchor_score=anchor_score,
                    max_anchor_tup=max_anchor_tup,
                    min_cluster_score=min_cluster_score,
                    max_standalone=max_standalone,
                    max_refined=max_refined,
                    max_queries=max_queries_per_topic,
                    max_clause_freq=max_clause_freq,
                    min_retrieved_docs=min_retrieved_docs,
                    verbose=False,
                )
                timing["encode"] = 0  # included in query_build_run
                timing["query_build_run"] = time.perf_counter() - t0

            else:
                # Original pipeline
                t0 = time.perf_counter()
                all_tokens = pipeline.encode_query(query_text)
                scored = pipeline.score_tokens(all_tokens, min_splade_score=min_splade_score)
                timing["encode"] = time.perf_counter() - t0

                if len(scored) < 2:
                    print(f"  [{qid}] Skipping: only {len(scored)} tokens after filtering")
                    failed.append((qid, "too_few_tokens"))
                    continue

                t0 = time.perf_counter()
                queries = pipeline.build_and_run(
                    query_text,
                    engine=engine,
                    use_chunking=use_chunking,
                    max_clause_freq=max_clause_freq,
                    min_retrieved_docs=min_retrieved_docs,
                    lower_query_bound=lower_query_bound,
                    min_splade_score=min_splade_score,
                    anchor_score=anchor_score,
                    max_anchor_tup=max_anchor_tup,
                    min_cluster_score=min_cluster_score,
                    strategy=strategy,
                    max_queries=max_queries_per_topic,
                    verbose=False,
                )
                timing["query_build_run"] = time.perf_counter() - t0

            if not queries:
                print(f"  [{qid}] Skipping: no queries generated")
                failed.append((qid, "no_queries"))
                continue

            # Step 3: Resolve documents
            t0 = time.perf_counter()
            resolve_kwargs = {
                "queries": queries,
                "index_dir": index_dir,
                "tokenizer": tokenizer,
                "max_doc_len": max_doc_len,
                "top_splade_filter": top_splade_filter,
            }
            if llm_adaptive and scoring_terms_q:
                resolve_kwargs["scoring_terms"] = scoring_terms_q
            else:
                resolve_kwargs["top_tokens"] = scored

            docs = resolve_all_queries(**resolve_kwargs)
            timing["resolve"] = time.perf_counter() - t0
            timing["n_docs"] = len(docs)
            total_docs += len(docs)

            if not docs:
                print(f"  [{qid}] Skipping: no documents resolved")
                failed.append((qid, "no_docs"))
                continue

            # Step 4: Write to run file
            write_run(
                qid, docs,
                run_name=run_name,
                path=run_path,
                score_field=score_field,
                append=True,
            )

        except Exception as e:
            print(f"  [{qid}] ERROR: {e}")
            failed.append((qid, str(e)))
            continue

        timing["total"] = time.perf_counter() - t_start
        timings.append(timing)

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Topics processed: {len(timings)}/{len(topics)}")
    print(f"Topics failed:    {len(failed)}")
    print(f"Total documents:  {total_docs}")

    if timings:
        avg_total = sum(t["total"] for t in timings) / len(timings)
        avg_encode = sum(t.get("encode", 0) for t in timings) / len(timings)
        avg_query = sum(t.get("query_build_run", 0) for t in timings) / len(timings)
        avg_resolve = sum(t.get("resolve", 0) for t in timings) / len(timings)
        avg_docs = sum(t.get("n_docs", 0) for t in timings) / len(timings)

        print(f"\nTiming (avg per topic):")
        print(f"  Encode:       {avg_encode:.2f}s")
        print(f"  Query+Run:    {avg_query:.2f}s")
        print(f"  Resolve:      {avg_resolve:.2f}s")
        print(f"  Total:        {avg_total:.2f}s")
        print(f"  Docs/topic:   {avg_docs:.0f}")
        print(f"  Est. total:   {avg_total * len(topics) / 60:.1f} min for all {len(topics)} topics")

    if failed:
        print(f"\nFailed topics:")
        for qid, reason in failed[:10]:
            print(f"  {qid}: {reason}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    # Evaluate
    print(f"\n{'='*70}")
    print(f"EVALUATION")
    print(f"{'='*70}")
    metrics = evaluate_run(run_path, qrels_path)

    return {
        "metrics": metrics,
        "timings": timings,
        "failed": failed,
        "run_path": run_path,
    }