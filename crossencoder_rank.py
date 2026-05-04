"""
Cross-encoder reranking of pre-filtered document pools.

Usage:
    from crossencoder_rank import evaluate_batch

    results = evaluate_batch(
        retrieval_results,
        topics_path="./topics_rag24_test.txt",
        qrels_path="./qrels.rag24.test-umbrela-all.txt",
    )

    # Or with pre-ranked input (e.g. from bi-encoder top 1000):
    results = evaluate_batch(
        retrieval_results,
        topics_path="./topics_rag24_test.txt",
        qrels_path="./qrels.rag24.test-umbrela-all.txt",
        max_docs_per_query=1000,
    )
"""

import time
from tqdm import tqdm
from trec_output import load_qrels
from full_eval import load_topics


def _deduplicate_docs(docs):
    """Deduplicate docs by doc_id, return (doc_ids, texts)."""
    seen = set()
    doc_ids = []
    texts = []
    for d in docs:
        did = d.get("doc_id", "")
        if did and did not in seen:
            seen.add(did)
            doc_ids.append(did)
            texts.append(d.get("text", ""))
    return doc_ids, texts


def rerank_crossencoder(
    query_text: str,
    doc_ids: list,
    texts: list,
    model,
    top_k: int = 100,
    batch_size: int = 32,
    max_length: int = 512,
):
    """
    Rerank documents using a cross-encoder model.

    Args:
        query_text: The query string.
        doc_ids: List of document IDs.
        texts: List of document texts (same order as doc_ids).
        model: CrossEncoder model instance.
        top_k: Number of docs to return.
        batch_size: Scoring batch size.
        max_length: Max token length for model input.

    Returns list of (score, doc_id) sorted by score descending.
    """
    if not doc_ids:
        return []

    # Truncate texts
    texts_trunc = [t[:max_length * 4] if t else "" for t in texts]

    # Build pairs
    pairs = [(query_text, t) for t in texts_trunc]

    # Score
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    # Sort and return top_k
    scored = list(zip(scores, doc_ids))
    scored.sort(key=lambda x: x[0], reverse=True)

    return scored[:top_k]


def evaluate_single(
    qid: str,
    query_text: str,
    docs: list,
    model,
    qrels: dict,
    top_k_list: list = None,
    max_docs_per_query: int = None,
    batch_size: int = 32,
):
    """Evaluate cross-encoder reranking for a single query."""
    from ir_metrics import compute_metrics

    if top_k_list is None:
        top_k_list = [10, 20, 50, 100]

    relevant = {did: rel for did, rel in qrels.get(qid, {}).items() if rel > 0}

    doc_ids, texts = _deduplicate_docs(docs)

    # Optionally limit input size
    if max_docs_per_query and len(doc_ids) > max_docs_per_query:
        doc_ids = doc_ids[:max_docs_per_query]
        texts = texts[:max_docs_per_query]

    pool_found = len(set(doc_ids) & set(relevant.keys()))
    max_k = max(top_k_list)

    t0 = time.perf_counter()
    ranked = rerank_crossencoder(query_text, doc_ids, texts, model,
                                  top_k=max_k, batch_size=batch_size)
    elapsed = time.perf_counter() - t0

    ranked_ids = [did for _, did in ranked]
    cutoffs = compute_metrics(ranked_ids, qrels, qid, top_k_list=top_k_list)

    return {
        "qid": qid,
        "n_input_docs": len(doc_ids),
        "n_relevant": len(relevant),
        "pool_found": pool_found,
        "pool_recall": pool_found / len(relevant) if relevant else 0,
        "time_seconds": round(elapsed, 2),
        "cutoffs": cutoffs,
    }


def evaluate_batch(
    retrieval_results: list,
    topics_path: str,
    qrels_path: str,
    model=None,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    top_k_list: list = None,
    max_docs_per_query: int = None,
    batch_size: int = 32,
    verbose: bool = True,
):
    """
    Evaluate cross-encoder reranking across all queries.

    Args:
        retrieval_results: List of result dicts with docs, qid.
            Docs can be pre-filtered (e.g. bi-encoder top 1000).
        model: Pre-loaded CrossEncoder. If None, loads model_name.
        max_docs_per_query: Limit input docs per query (for speed).
    """
    if top_k_list is None:
        top_k_list = [10, 20, 50, 100]

    if model is None:
        from sentence_transformers import CrossEncoder
        if verbose:
            print(f"Loading model: {model_name}")
        model = CrossEncoder(model_name)

    topics = dict(load_topics(topics_path))
    qrels = load_qrels(qrels_path)

    all_results = []
    total_time = 0

    for r in tqdm(retrieval_results, desc="Cross-encoder rerank", disable=not verbose):
        qid = r["qid"]
        docs = r.get("docs", [])
        query_text = topics.get(qid, "")

        if not docs or not query_text:
            continue

        result = evaluate_single(
            qid, query_text, docs, model, qrels,
            top_k_list=top_k_list,
            max_docs_per_query=max_docs_per_query,
            batch_size=batch_size,
        )
        all_results.append(result)
        total_time += result["time_seconds"]

    if verbose and all_results:
        n = len(all_results)
        print(f"\n{'='*80}")
        print(f"Cross-Encoder Reranking ({n} topics, model={model_name})")
        if max_docs_per_query:
            print(f"  Max docs per query: {max_docs_per_query}")
        print(f"{'='*80}")

        avg_pool_recall = sum(r["pool_recall"] for r in all_results) / n
        avg_input = sum(r["n_input_docs"] for r in all_results) / n
        print(f"  Pool avg recall: {avg_pool_recall:.4f}")
        print(f"  Total time: {total_time:.1f}s ({total_time/n:.1f}s per topic)")
        print(f"  Avg input docs: {avg_input:.0f}")
        avg_mrr = sum(r["cutoffs"]["mrr"] for r in all_results) / n
        print(f"  MRR: {avg_mrr:.4f}")

        header = f"  {'':>15s}"
        for k in top_k_list:
            header += f" {'R@'+str(k):>8s} {'nDCG@'+str(k):>8s} {'%Pool':>6s}"
        print(header)

        print(f"  {'Average':>15s}", end="")
        for k in top_k_list:
            recalls = [r["cutoffs"][k]["recall"] for r in all_results]
            ndcgs = [r["cutoffs"][k]["ndcg"] for r in all_results]
            pool_pcts = [r["cutoffs"][k]["found"] / r["pool_found"]
                        if r["pool_found"] > 0 else 0 for r in all_results]
            print(f" {sum(recalls)/n:>8.4f} {sum(ndcgs)/n:>8.4f} {sum(pool_pcts)/n:>5.0%}", end="")
        print()

        # Per-query
        print(f"\n{'QID':<15s} {'Input':>6s} {'Rel':>5s} {'Pool':>5s} {'MRR':>5s} {'Time':>6s}", end="")
        for k in top_k_list:
            print(f" {'R@'+str(k):>7s} {'nD@'+str(k):>6s} {'%P':>5s}", end="")
        print()
        print("-" * (45 + 19 * len(top_k_list)))

        for r in all_results:
            print(f"{r['qid']:<15s} {r['n_input_docs']:>6d} {r['n_relevant']:>5d} "
                  f"{r['pool_found']:>5d} {r['cutoffs']['mrr']:>5.3f} {r['time_seconds']:>5.1f}s", end="")
            for k in top_k_list:
                recall = r["cutoffs"][k]["recall"]
                ndcg = r["cutoffs"][k]["ndcg"]
                pool_pct = r["cutoffs"][k]["found"] / r["pool_found"] if r["pool_found"] > 0 else 0
                print(f" {recall:>7.3f} {ndcg:>6.3f} {pool_pct:>4.0%}", end="")
            print()

    return all_results, model