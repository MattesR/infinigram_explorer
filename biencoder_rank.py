"""
Bi-encoder reranking of retrieval pools.

Usage:
    from biencoder_rank import evaluate_batch, grid_search_model

    results = evaluate_batch(
        retrieval_results,
        topics_path="./topics_rag24_test.txt",
        qrels_path="./qrels.rag24.test-umbrela-all.txt",
    )
"""

import time
import numpy as np
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


def rerank_biencoder(
    query_text: str,
    docs: list,
    model,
    top_k: int = 1000,
    batch_size: int = 256,
    max_doc_length: int = 512,
):
    """
    Rerank documents using a bi-encoder model.

    Args:
        query_text: The query string.
        docs: List of doc dicts with doc_id and text.
        model: SentenceTransformer model instance.
        top_k: Number of docs to return.
        batch_size: Encoding batch size.
        max_doc_length: Truncate docs to this many chars (approx tokens).

    Returns list of (score, doc_id) sorted by score descending.
    """
    doc_ids, texts = _deduplicate_docs(docs)

    if not doc_ids:
        return []

    # Truncate long docs
    texts = [t[:max_doc_length * 4] if t else "" for t in texts]

    # Encode
    query_emb = model.encode([query_text], normalize_embeddings=True,
                              show_progress_bar=False)
    doc_embs = model.encode(texts, batch_size=batch_size,
                             normalize_embeddings=True,
                             show_progress_bar=False)

    # Cosine similarity (already normalized)
    scores = (doc_embs @ query_emb.T).flatten()

    # Sort and return top_k
    top_indices = scores.argsort()[::-1][:top_k]
    ranked = [(float(scores[i]), doc_ids[i]) for i in top_indices]

    return ranked


def evaluate_single(
    qid: str,
    query_text: str,
    docs: list,
    model,
    qrels: dict,
    top_k_list: list = None,
    batch_size: int = 256,
):
    """Evaluate bi-encoder reranking for a single query."""
    if top_k_list is None:
        top_k_list = [10, 100, 1000, 2000]

    max_k = max(top_k_list)
    relevant = {did: rel for did, rel in qrels.get(qid, {}).items() if rel > 0}

    t0 = time.perf_counter()
    ranked = rerank_biencoder(query_text, docs, model, top_k=max_k,
                               batch_size=batch_size)
    elapsed = time.perf_counter() - t0

    # All relevant in pool
    doc_ids_all = {d.get("doc_id") for d in docs if d.get("doc_id")}
    pool_found = len(doc_ids_all & set(relevant.keys()))

    results = {}
    for k in top_k_list:
        top_ids = {did for _, did in ranked[:k]}
        found = top_ids & set(relevant.keys())
        recall = len(found) / len(relevant) if relevant else 0
        precision = len(found) / min(k, len(ranked)) if ranked else 0
        results[k] = {"recall": recall, "precision": precision, "found": len(found)}

    return {
        "qid": qid,
        "n_docs": len(doc_ids_all),
        "n_relevant": len(relevant),
        "pool_found": pool_found,
        "pool_recall": pool_found / len(relevant) if relevant else 0,
        "time_seconds": round(elapsed, 2),
        "cutoffs": results,
    }


def evaluate_batch(
    retrieval_results: list,
    topics_path: str,
    qrels_path: str,
    model=None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k_list: list = None,
    batch_size: int = 256,
    verbose: bool = True,
):
    """
    Evaluate bi-encoder reranking across all queries.

    Args:
        retrieval_results: List of result dicts with docs, qid.
        model: Pre-loaded SentenceTransformer. If None, loads model_name.
    """
    if top_k_list is None:
        top_k_list = [10, 100, 1000, 2000]

    if model is None:
        from sentence_transformers import SentenceTransformer
        if verbose:
            print(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)

    topics = dict(load_topics(topics_path))
    qrels = load_qrels(qrels_path)

    all_results = []
    total_time = 0

    for r in tqdm(retrieval_results, desc="Bi-encoder rerank", disable=not verbose):
        qid = r["qid"]
        docs = r.get("docs", [])
        query_text = topics.get(qid, "")

        if not docs or not query_text:
            continue

        result = evaluate_single(
            qid, query_text, docs, model, qrels,
            top_k_list=top_k_list, batch_size=batch_size,
        )
        all_results.append(result)
        total_time += result["time_seconds"]

    if verbose and all_results:
        n = len(all_results)
        print(f"\n{'='*80}")
        print(f"Bi-Encoder Reranking Summary ({n} topics, model={model_name})")
        print(f"{'='*80}")

        avg_pool_recall = sum(r["pool_recall"] for r in all_results) / n
        print(f"  Pool avg recall: {avg_pool_recall:.4f}")
        print(f"  Total time: {total_time:.1f}s ({total_time/n:.1f}s per topic)")
        print(f"  Avg docs per topic: {sum(r['n_docs'] for r in all_results)/n:.0f}")

        header = f"  {'':>15s}"
        for k in top_k_list:
            header += f" {'R@'+str(k):>10s} {'R/Pool@'+str(k):>10s}"
        print(header)

        print(f"  {'Average':>15s}", end="")
        for k in top_k_list:
            recalls = [r["cutoffs"][k]["recall"] for r in all_results]
            pool_recalls = [r["cutoffs"][k]["found"] / r["pool_found"]
                           if r["pool_found"] > 0 else 0 for r in all_results]
            print(f" {sum(recalls)/n:>10.4f} {sum(pool_recalls)/n:>10.4f}", end="")
        print()

        # Per-query
        print(f"\n{'QID':<15s} {'Retr':>6s} {'Rel':>5s} {'Pool':>5s} {'Ceil':>5s} {'Time':>5s}", end="")
        for k in top_k_list:
            print(f" {'R@'+str(k):>7s} {'%P@'+str(k):>6s}", end="")
        print()
        print("-" * (45 + 14 * len(top_k_list)))

        for r in all_results:
            pool_ceil = r["pool_found"] / r["n_relevant"] if r["n_relevant"] else 0
            print(f"{r['qid']:<15s} {r['n_docs']:>6d} {r['n_relevant']:>5d} "
                  f"{r['pool_found']:>5d} {pool_ceil:>4.0%} {r['time_seconds']:>4.1f}s", end="")
            for k in top_k_list:
                recall = r["cutoffs"][k]["recall"]
                pool_rel = r["cutoffs"][k]["found"] / r["pool_found"] if r["pool_found"] > 0 else 0
                print(f" {recall:>7.3f} {pool_rel:>5.0%}", end="")
            print()

    return all_results, model


def compare_models(
    retrieval_results: list,
    topics_path: str,
    qrels_path: str,
    model_names: list = None,
    top_k_list: list = None,
    batch_size: int = 256,
    verbose: bool = True,
):
    """
    Compare multiple bi-encoder models.

    Args:
        model_names: List of model names to compare.
    """
    if model_names is None:
        model_names = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "BAAI/bge-small-en-v1.5",
        ]

    if top_k_list is None:
        top_k_list = [100, 1000]

    all_model_results = {}

    for model_name in model_names:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Model: {model_name}")
            print(f"{'='*60}")

        results, model = evaluate_batch(
            retrieval_results, topics_path, qrels_path,
            model_name=model_name,
            top_k_list=top_k_list,
            batch_size=batch_size,
            verbose=verbose,
        )
        all_model_results[model_name] = results
        del model  # free memory

    # Comparison table
    if verbose:
        print(f"\n{'='*80}")
        print(f"MODEL COMPARISON")
        print(f"{'='*80}")

        header = f"{'Model':<45s}"
        for k in top_k_list:
            header += f" {'R@'+str(k):>8s}"
        header += f" {'Time':>8s}"
        print(header)
        print("-" * len(header))

        for model_name, results in all_model_results.items():
            n = len(results)
            line = f"{model_name:<45s}"
            for k in top_k_list:
                avg_r = sum(r["cutoffs"][k]["recall"] for r in results) / n
                line += f" {avg_r:>8.4f}"
            avg_t = sum(r["time_seconds"] for r in results) / n
            line += f" {avg_t:>7.1f}s"
            print(line)

    return all_model_results