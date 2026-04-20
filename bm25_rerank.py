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
    top_k: int = 1000,
    query_weight: int = 1,
    verbose: bool = True,
):
    """
    Rerank docs with BM25 using three strategies and evaluate against qrels.

    Strategies:
        - query_only: Original query text
        - query_plus_kw: Original query + all LLM keywords (equal weight)
        - query_weighted: Original query repeated N times + keywords once

    Returns dict with per-strategy results.
    """
    qrels = load_qrels(qrels_path)
    relevant = {did: rel for did, rel in qrels.get(qid, {}).items() if rel > 0}

    # Get all doc_ids before reranking
    all_doc_ids = {d.get("doc_id") for d in docs if d.get("doc_id")}
    all_found = all_doc_ids & set(relevant.keys())

    if verbose:
        print(f"\nBM25 reranking for {qid}")
        print(f"  Docs to rerank: {len(all_doc_ids)}")
        print(f"  Relevant in pool: {len(all_found)}/{len(relevant)}")
        print(f"  Top-k: {top_k}")

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
        ranked = rerank_bm25(docs, terms, top_k=top_k)

        top_ids = {did for _, did, _ in ranked}
        found = top_ids & set(relevant.keys())

        # Compute recall at top_k
        recall = len(found) / len(relevant) if relevant else 0

        # Also compute precision
        precision = len(found) / len(top_ids) if top_ids else 0

        results[name] = {
            "recall": recall,
            "precision": precision,
            "found": len(found),
            "relevant": len(relevant),
            "top_k": len(ranked),
            "n_terms": len(set(terms)),
        }

        if verbose:
            print(f"\n  {name} ({len(set(terms))} unique terms):")
            print(f"    Recall@{top_k}: {recall:.3f} ({len(found)}/{len(relevant)})")
            print(f"    Precision@{top_k}: {precision:.4f}")

    if verbose:
        lost = len(all_found) - max(r["found"] for r in results.values())
        if lost > 0:
            print(f"\n  WARNING: Best strategy loses {lost} relevant docs vs unreranked pool")

    return results