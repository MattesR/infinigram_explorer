"""
BM25 reranking of retrieved documents using original query + LLM keywords.

Usage:
    from bm25_rerank import rerank_and_evaluate

    results = rerank_and_evaluate(
        qid="2024-32912",
        query_text="how did the Vietnam War devastate the economy in 1968",
        docs=docs,  # from resolve_all_queries
        qrels_path="./qrels.rag24.test-umbrela-all.txt",
        expansions_path="./kiss.jsonl",
        top_k=1000,
    )
"""

import json
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from llm_keyword_filter import load_all_expansions, STOPWORDS
from trec_output import load_qrels


def _flatten_keywords(qid, expansions_path):
    """Extract all keywords from expansion file as flat list of terms."""
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


def _tokenize(text):
    """Simple whitespace + lowercase tokenization."""
    return text.lower().split()


def rerank_bm25(
    docs: list,
    query_terms: list,
    top_k: int = 1000,
):
    """
    Rerank documents using BM25 against query terms.

    Args:
        docs: List of doc dicts with 'doc_id' and 'text' keys.
        query_terms: List of query term strings.
        top_k: Number of top docs to return.

    Returns:
        List of (score, doc_id, text) tuples, sorted by score descending.
    """
    # Deduplicate docs by doc_id
    seen = set()
    unique_docs = []
    for d in docs:
        did = d.get("doc_id", "")
        if did and did not in seen:
            seen.add(did)
            unique_docs.append(d)

    if not unique_docs:
        return []

    # Tokenize
    doc_texts = [d.get("text", "") for d in unique_docs]
    tokenized_docs = [_tokenize(t) for t in doc_texts]
    tokenized_query = _tokenize(" ".join(query_terms))

    # Build BM25 and score
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)

    # Rank
    ranked = sorted(
        zip(scores, [d["doc_id"] for d in unique_docs], doc_texts),
        reverse=True,
    )

    return ranked[:top_k]


def rerank_and_evaluate(
    qid: str,
    query_text: str,
    docs: list,
    qrels_path: str,
    expansions_path: str = None,
    top_k_list: list = None,
    query_weight: int = 3,
    verbose: bool = True,
):
    """
    Rerank docs with BM25 using three strategies and evaluate against qrels.

    Strategies:
        - query_only: Original query text
        - query_plus_kw: Original query + all LLM keywords (equal weight)
        - query_weighted: Original query repeated N times + keywords once

    Args:
        top_k_list: List of cutoffs to evaluate at. Default [10, 100, 1000].

    Returns dict with per-strategy, per-cutoff results.
    """
    if top_k_list is None:
        top_k_list = [10, 100, 1000]

    max_k = max(top_k_list)

    qrels = load_qrels(qrels_path)
    relevant = {did: rel for did, rel in qrels.get(qid, {}).items() if rel > 0}

    # Get all doc_ids before reranking
    all_doc_ids = {d.get("doc_id") for d in docs if d.get("doc_id")}
    all_found = all_doc_ids & set(relevant.keys())

    if verbose:
        print(f"\nBM25 reranking for {qid}")
        print(f"  Docs to rerank: {len(all_doc_ids)}")
        print(f"  Relevant in pool: {len(all_found)}/{len(relevant)}")
        print(f"  Cutoffs: {top_k_list}")

    # Build query variants
    query_tokens = query_text.lower().split()

    kw_tokens = []
    if expansions_path:
        keywords = _flatten_keywords(qid, expansions_path)
        kw_tokens = " ".join(keywords).lower().split()
        # Remove stopwords from keywords
        kw_tokens = [t for t in kw_tokens if t not in STOPWORDS and len(t) > 2]

    strategies = {}

    # Strategy 1: Query only
    strategies["query_only"] = query_tokens

    # Strategy 2: Query + keywords (equal weight)
    if kw_tokens:
        strategies["query_plus_kw"] = query_tokens + kw_tokens

    # Strategy 3: Query weighted + keywords
    if kw_tokens:
        strategies["query_weighted"] = query_tokens * query_weight + kw_tokens

    results = {}
    for name, terms in strategies.items():
        ranked = rerank_bm25(docs, terms, top_k=max_k)

        strategy_results = {}
        for k in top_k_list:
            top_ids = {did for _, did, _ in ranked[:k]}
            found = top_ids & set(relevant.keys())
            recall = len(found) / len(relevant) if relevant else 0
            precision = len(found) / min(k, len(ranked)) if ranked else 0

            strategy_results[k] = {
                "recall": recall,
                "precision": precision,
                "found": len(found),
            }

        results[name] = strategy_results

        if verbose:
            print(f"\n  {name} ({len(set(terms))} unique terms):")
            for k in top_k_list:
                r = strategy_results[k]
                print(f"    @{k:<5d} Recall: {r['recall']:.3f} ({r['found']}/{len(relevant)})  "
                      f"Precision: {r['precision']:.4f}")

    if verbose:
        best_at_max = max(results[s][max_k]["found"] for s in results)
        lost = len(all_found) - best_at_max
        if lost > 0:
            print(f"\n  WARNING: Best strategy @{max_k} loses {lost} relevant docs vs unreranked pool")

    return results


def rerank_batch(
    retrieval_results: list,
    topics_path: str,
    qrels_path: str,
    expansions_path: str = None,
    top_k_list: list = None,
    query_weight: int = 3,
    verbose: bool = True,
):
    """
    Rerank all retrieval results and print summary report.

    Args:
        retrieval_results: List of result dicts from compare_recall_ceiling
                           (must have 'docs', 'qid' fields).
        topics_path: Path to topics file (for query text).
        qrels_path: Path to qrels file.
        expansions_path: Path to keyword expansions JSONL.
        top_k_list: Cutoffs to evaluate. Default [10, 100, 1000].
        query_weight: Weight multiplier for original query in weighted strategy.

    Returns list of per-query rerank results.
    """
    if top_k_list is None:
        top_k_list = [10, 100, 1000]

    from full_eval import load_topics
    topics = dict(load_topics(topics_path))

    all_rerank = []
    strategies_seen = set()

    for r in tqdm(retrieval_results, desc="Reranking", disable=not verbose):
        qid = r["qid"]
        query_text = topics.get(qid, "")
        docs = r.get("docs", [])

        if not docs:
            continue

        rerank_result = rerank_and_evaluate(
            qid=qid,
            query_text=query_text,
            docs=docs,
            qrels_path=qrels_path,
            expansions_path=expansions_path,
            top_k_list=top_k_list,
            query_weight=query_weight,
            verbose=False,
        )

        rerank_result["qid"] = qid
        rerank_result["n_relevant"] = r.get("n_relevant", 0)
        rerank_result["n_retrieved"] = r.get("n_retrieved", 0)
        rerank_result["raw_recall"] = r.get("recall", 0)
        rerank_result["raw_found"] = r.get("n_found", 0)
        all_rerank.append(rerank_result)
        strategies_seen.update(k for k in rerank_result if k not in
                                {"qid", "n_relevant", "n_retrieved", "raw_recall", "raw_found"})

    if verbose and all_rerank:
        n = len(all_rerank)
        strategies = sorted(strategies_seen)

        # Summary table
        print(f"\n{'='*80}")
        print(f"BM25 Reranking Summary ({n} topics)")
        print(f"{'='*80}")

        # Raw recall
        avg_raw = sum(r["raw_recall"] for r in all_rerank) / n
        avg_retr = sum(r["n_retrieved"] for r in all_rerank) / n
        print(f"\n  Raw retrieval: Avg Recall={avg_raw:.4f}, Avg Retrieved={avg_retr:.0f}")

        # Per strategy, per cutoff
        for strategy in strategies:
            print(f"\n  Strategy: {strategy}")
            header = f"    {'':>15s}"
            for k in top_k_list:
                header += f" {'R@'+str(k):>10s} {'P@'+str(k):>10s}"
            print(header)
            print(f"    {'Average':>15s}", end="")
            for k in top_k_list:
                recalls = [r[strategy][k]["recall"] for r in all_rerank if strategy in r]
                precisions = [r[strategy][k]["precision"] for r in all_rerank if strategy in r]
                avg_r = sum(recalls) / len(recalls) if recalls else 0
                avg_p = sum(precisions) / len(precisions) if precisions else 0
                print(f" {avg_r:>10.4f} {avg_p:>10.4f}", end="")
            print()

        # Per-query table — find actual best strategy by avg recall at max cutoff
        max_k = max(top_k_list)
        best_strategy = None
        best_avg = -1
        for strategy in strategies:
            recalls = [r[strategy][max_k]["recall"] for r in all_rerank if strategy in r]
            avg = sum(recalls) / len(recalls) if recalls else 0
            if avg > best_avg:
                best_avg = avg
                best_strategy = strategy

        print(f"\n{'='*80}")
        print(f"Per-query results (best strategy: {best_strategy})")
        print(f"{'='*80}")
        if best_strategy:
            header = f"{'QID':<15s} {'Retr':>6s} {'Rel':>5s} {'Raw':>6s}"
            for k in top_k_list:
                header += f" {'R@'+str(k):>7s}"
            print(header)
            print("-" * len(header))

            for r in all_rerank:
                row = f"{r['qid']:<15s} {r['n_retrieved']:>6d} {r['n_relevant']:>5d} {r['raw_found']:>6d}"
                for k in top_k_list:
                    recall = r[best_strategy][k]["recall"]
                    row += f" {recall:>7.3f}"
                print(row)

            # Strategy comparison at max cutoff
            max_k = max(top_k_list)
            print(f"\n{'='*80}")
            print(f"Strategy comparison at @{max_k}:")
            print(f"{'='*80}")
            print(f"{'Strategy':<20s} {'Avg Recall':>12s} {'Avg Precision':>14s} {'Avg Found':>10s}")
            for strategy in strategies:
                recalls = [r[strategy][max_k]["recall"] for r in all_rerank if strategy in r]
                precisions = [r[strategy][max_k]["precision"] for r in all_rerank if strategy in r]
                founds = [r[strategy][max_k]["found"] for r in all_rerank if strategy in r]
                print(f"{strategy:<20s} {sum(recalls)/len(recalls):>12.4f} "
                      f"{sum(precisions)/len(precisions):>14.4f} "
                      f"{sum(founds)/len(founds):>10.1f}")

    return all_rerank