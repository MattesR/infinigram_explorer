"""
BM25 baseline retrieval using Pyserini Lucene index.

Usage:
    from bm25_baseline import bm25_baseline, bm25_baseline_batch

    # Single query
    result = bm25_baseline(
        qid="2024-32912",
        query_text="how did the Vietnam War devastate the economy in 1968",
        searcher=searcher,
        qrels_path="./qrels.rag24.test-umbrela-all.txt",
    )

    # Batch over topics
    results = bm25_baseline_batch(
        topics_path="./topics_rag24_test.txt",
        qrels_path="./qrels.rag24.test-umbrela-all.txt",
        searcher=searcher,
        max_topics=10,
    )
"""

import json
from tqdm import tqdm
from trec_output import load_qrels
from full_eval import load_topics


def _flatten_keywords(qid, expansions_path):
    """Extract all keywords from expansion file as flat list of terms."""
    from llm_keyword_filter import load_all_expansions
    data = load_all_expansions(expansions_path).get(qid, {})
    terms = []

    key_entities = data.get("KEY_ENTITIES", {})
    for name, aspect in key_entities.items():
        terms.append(name)
        if isinstance(aspect, dict):
            for level in ["lexical", "conceptual", "referential"]:
                terms.extend(aspect.get(level, []))
        elif isinstance(aspect, list):
            terms.extend(aspect)

    for kw in data.get("ASSOCIATED_TERMS", data.get("ASSOCIATED", [])):
        terms.append(kw)

    verbs = data.get("VERBS", [])
    if isinstance(verbs, dict):
        for verb, exps in verbs.items():
            terms.append(verb)
            terms.extend(exps)
    elif isinstance(verbs, list):
        terms.extend(verbs)

    return terms


def bm25_baseline(
    qid: str,
    query_text: str,
    searcher,
    qrels_path: str = None,
    qrels: dict = None,
    k: int = 1000,
    return_docs: bool = False,
    expansions_path: str = None,
    query_weight: int = 1,
    biencoder_model=None,
    biencoder_top_k: int = 1000,
    crossencoder_model=None,
    crossencoder_top_k: int = 100,
):
    """
    Run BM25 search for a single query and evaluate.

    Args:
        searcher: LuceneSearcher instance
        k: number of results to retrieve from BM25
        return_docs: if True, include doc texts in result
        expansions_path: if set, expand query with LLM keywords from this file
        query_weight: how many times to repeat original query vs keywords (1=equal)
        biencoder_model: if set, rerank BM25 pool down to biencoder_top_k
        crossencoder_model: if set, rerank top results with cross-encoder

    Returns dict with recall, found, retrieved stats at multiple cutoffs.
    """
    if qrels is None:
        qrels = load_qrels(qrels_path)

    relevant = {did: rel for did, rel in qrels.get(qid, {}).items() if rel > 0}

    # Build search query
    search_query = query_text
    if expansions_path:
        keywords = _flatten_keywords(qid, expansions_path)
        kw_text = " ".join(keywords)
        if query_weight > 1:
            search_query = " ".join([query_text] * query_weight + [kw_text])
        else:
            search_query = query_text + " " + kw_text

    # Need docs for bi-encoder or cross-encoder
    need_docs = return_docs or biencoder_model is not None or crossencoder_model is not None

    # Search
    hits = searcher.search(search_query, k=k)

    # Collect results
    retrieved_ids = []
    docs = []
    for hit in hits:
        did = hit.docid
        retrieved_ids.append(did)
        if need_docs:
            doc = searcher.doc(did)
            if doc:
                raw = json.loads(doc.raw())
                text = raw.get("segment", raw.get("body", raw.get("text", "")))
            else:
                text = ""
            docs.append({
                "doc_id": did,
                "text": text,
                "score": hit.score,
            })

    retrieved = set(retrieved_ids)
    found = relevant.keys() & retrieved

    # Compute BM25 metrics
    from ir_metrics import compute_metrics
    cutoffs = compute_metrics(retrieved_ids, qrels, qid, top_k_list=[10, 100, 1000])

    result = {
        "qid": qid,
        "n_relevant": len(relevant),
        "n_retrieved": len(retrieved),
        "n_found": len(found),
        "recall": len(found) / len(relevant) if relevant else 0,
        "cutoffs": cutoffs,
    }

    # Bi-encoder reranking: pool -> biencoder_top_k
    if biencoder_model is not None and docs:
        texts = [d["text"] for d in docs]
        query_emb = biencoder_model.encode([query_text], show_progress_bar=False)
        doc_embs = biencoder_model.encode(texts, batch_size=256, show_progress_bar=False)

        from numpy import dot
        from numpy.linalg import norm
        scores = [float(dot(query_emb[0], d) / (norm(query_emb[0]) * norm(d) + 1e-8))
                  for d in doc_embs]

        bi_ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        bi_ranked = bi_ranked[:biencoder_top_k]
        docs = [d for _, d in bi_ranked]

        bi_ranked_ids = [d["doc_id"] for d in docs]
        bi_cutoffs = compute_metrics(bi_ranked_ids, qrels, qid, top_k_list=[10, 100, 1000])
        result["bi_cutoffs"] = bi_cutoffs

    # Cross-encoder reranking of top-k results
    if crossencoder_model is not None and docs:
        top_docs = docs[:crossencoder_top_k]
        pairs = [(query_text, d["text"]) for d in top_docs]
        ce_scores = crossencoder_model.predict(pairs, batch_size=32, show_progress_bar=False)

        ce_ranked = sorted(zip(ce_scores, top_docs), key=lambda x: x[0], reverse=True)
        ce_ranked_ids = [d["doc_id"] for _, d in ce_ranked]

        ce_cutoffs = compute_metrics(ce_ranked_ids, qrels, qid, top_k_list=[10, 100])
        result["ce_cutoffs"] = ce_cutoffs

        if return_docs:
            result["ce_docs"] = [{"doc_id": d["doc_id"], "text": d["text"],
                                   "ce_score": float(s), "bm25_score": d["score"]}
                                  for s, d in ce_ranked]

    if return_docs:
        result["docs"] = docs

    return result


def bm25_baseline_batch(
    topics_path: str,
    qrels_path: str,
    searcher,
    max_topics: int = None,
    k: int = 1000,
    return_docs: bool = False,
    expansions_path: str = None,
    query_weight: int = 1,
    biencoder_rerank: bool = False,
    biencoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    biencoder_top_k: int = 1000,
    crossencoder_rerank: bool = False,
    crossencoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    crossencoder_top_k: int = 100,
    verbose: bool = True,
):
    """
    Run BM25 baseline for multiple topics and print summary.

    Args:
        expansions_path: if set, expand queries with LLM keywords.
        query_weight: repeat original query N times vs keywords 1x.
        biencoder_rerank: if True, filter BM25 pool with bi-encoder.
        crossencoder_rerank: if True, rerank top results with cross-encoder.
    """
    topics = load_topics(topics_path)
    if max_topics:
        topics = topics[:max_topics]
    qrels = load_qrels(qrels_path)

    bi_model = None
    if biencoder_rerank:
        from sentence_transformers import SentenceTransformer
        bi_model = SentenceTransformer(biencoder_model_name)
        if verbose:
            print(f"Loaded bi-encoder: {biencoder_model_name}")

    ce_model = None
    if crossencoder_rerank:
        from sentence_transformers import CrossEncoder
        ce_model = CrossEncoder(crossencoder_model_name)
        if verbose:
            print(f"Loaded cross-encoder: {crossencoder_model_name}")

    # Need docs for reranking
    need_docs = return_docs or biencoder_rerank or crossencoder_rerank

    results = []
    for qid, query_text in tqdm(topics, desc="BM25 baseline", disable=not verbose):
        result = bm25_baseline(
            qid=qid,
            query_text=query_text,
            searcher=searcher,
            qrels=qrels,
            k=k,
            return_docs=need_docs,
            expansions_path=expansions_path,
            query_weight=query_weight,
            biencoder_model=bi_model,
            biencoder_top_k=biencoder_top_k,
            crossencoder_model=ce_model,
            crossencoder_top_k=crossencoder_top_k,
        )
        if result:
            results.append(result)

    if verbose and results:
        n = len(results)
        avg_recall = sum(r["recall"] for r in results) / n
        avg_retrieved = sum(r["n_retrieved"] for r in results) / n
        avg_found = sum(r["n_found"] for r in results) / n
        avg_relevant = sum(r["n_relevant"] for r in results) / n
        avg_mrr = sum(r["cutoffs"]["mrr"] for r in results) / n

        print(f"\n{'='*70}")
        print(f"BM25 Baseline ({n} topics, k={k})")
        print(f"{'='*70}")
        print(f"  Avg Recall@{k}: {avg_recall:.4f}")
        print(f"  Avg Retrieved: {avg_retrieved:.0f}")
        print(f"  Avg Found: {avg_found:.1f}")
        print(f"  Avg Relevant: {avg_relevant:.1f}")
        print(f"  MRR: {avg_mrr:.4f}")

        # Per-cutoff averages
        cutoff_list = [10, 100, 1000]
        header = f"  {'':>15s}"
        for c in cutoff_list:
            header += f" {'R@'+str(c):>8s} {'nDCG@'+str(c):>8s}"
        print(header)

        print(f"  {'BM25':>15s}", end="")
        for c in cutoff_list:
            recalls = [r["cutoffs"].get(c, {}).get("recall", 0) for r in results]
            ndcgs = [r["cutoffs"].get(c, {}).get("ndcg", 0) for r in results]
            print(f" {sum(recalls)/n:>8.4f} {sum(ndcgs)/n:>8.4f}", end="")
        print()

        # Bi-encoder metrics
        if any("bi_cutoffs" in r for r in results):
            bi_results = [r for r in results if "bi_cutoffs" in r]
            nb = len(bi_results)
            print(f"  {'BM25+BI':>15s}", end="")
            for c in cutoff_list:
                recalls = [r["bi_cutoffs"].get(c, {}).get("recall", 0) for r in bi_results]
                ndcgs = [r["bi_cutoffs"].get(c, {}).get("ndcg", 0) for r in bi_results]
                print(f" {sum(recalls)/nb:>8.4f} {sum(ndcgs)/nb:>8.4f}", end="")
            print()

        # Cross-encoder metrics
        if any("ce_cutoffs" in r for r in results):
            ce_results = [r for r in results if "ce_cutoffs" in r]
            nc = len(ce_results)
            print(f"  {'BM25+CE':>15s}", end="")
            for c in [10, 100]:
                recalls = [r["ce_cutoffs"].get(c, {}).get("recall", 0) for r in ce_results]
                ndcgs = [r["ce_cutoffs"].get(c, {}).get("ndcg", 0) for r in ce_results]
                print(f" {sum(recalls)/nc:>8.4f} {sum(ndcgs)/nc:>8.4f}", end="")
            print(f" {'':>8s} {'':>8s}")  # pad for 1000 column

        # Per-query
        print(f"\n{'QID':<15s} {'Found':>6s} {'Rel':>5s} {'MRR':>5s}", end="")
        for c in cutoff_list:
            print(f" {'R@'+str(c):>7s} {'nD@'+str(c):>6s}", end="")
        print()
        print("-" * (35 + 14 * len(cutoff_list)))
        for r in results:
            print(f"{r['qid']:<15s} {r['n_found']:>6d} {r['n_relevant']:>5d} "
                  f"{r['cutoffs']['mrr']:>5.3f}", end="")
            for c in cutoff_list:
                recall = r["cutoffs"].get(c, {}).get("recall", 0)
                ndcg = r["cutoffs"].get(c, {}).get("ndcg", 0)
                print(f" {recall:>7.3f} {ndcg:>6.3f}", end="")
            print()

    return results