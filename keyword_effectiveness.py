"""
Analyze keyword effectiveness across all queries.

For each query, checks how many missed documents contain the LLM keywords.
This tells us whether the bottleneck is keyword quality or CNF construction.

Usage:
    from keyword_effectiveness import analyze_keyword_effectiveness

    # With Pyserini for doc lookup
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher('./lucene_index_msmarco_segmented')

    results = analyze_keyword_effectiveness(
        topics_path="./topics_rag24_test.txt",
        qrels_path="./qrels.rag24.test-umbrela-all.txt",
        expansions_path="./keyword_expansions.jsonl",
        found_dir="./inspection/lcr_opus/",
        searcher=searcher,
        max_topics=10,
    )

    # Or with pre-enriched missed JSONL files
    results = analyze_keyword_effectiveness(
        topics_path="./topics_rag24_test.txt",
        qrels_path="./qrels.rag24.test-umbrela-all.txt",
        expansions_path="./keyword_expansions.jsonl",
        missed_dir="./inspection/lcr_opus/",  # dir with {qid}_missed.jsonl files
        max_topics=10,
    )
"""

import json
import os
from pathlib import Path
from tqdm import tqdm


def _check_kw(text, kw):
    """Case-insensitive keyword check."""
    return kw.lower() in text.lower()


def _load_keywords(qid, expansions_path):
    """Load keywords for a query from expansions file."""
    from llm_keyword_filter import load_all_expansions
    expansions = load_all_expansions(expansions_path)
    data = expansions.get(qid, {})

    key_groups = {}
    sup_groups = {}
    verbs = []
    all_terms = []

    for key, values in data.items():
        if not isinstance(values, list):
            continue
        upper = key.upper()
        if upper.startswith("KEY:") or upper.startswith("CORE:"):
            name = key.split(":", 1)[1].strip()
            key_groups[name] = values
            all_terms.extend(values)
        elif upper.startswith("SUP:") or upper.startswith("AUX:"):
            name = key.split(":", 1)[1].strip()
            sup_groups[name] = values
            all_terms.extend(values)
        elif upper == "VERBS":
            verbs = values
            all_terms.extend(values)

    return {
        "key_groups": key_groups,
        "sup_groups": sup_groups,
        "verbs": verbs,
        "all_terms": all_terms,
    }


def _get_missed_texts(qid, qrels, found_ids, searcher=None, missed_dir=None):
    """Get text for missed documents from searcher or pre-enriched files."""
    relevant = {did: rel for did, rel in qrels.get(qid, {}).items() if rel > 0}
    missed_ids = {did: rel for did, rel in relevant.items() if did not in found_ids}

    missed_docs = []

    # Try pre-enriched files first
    if missed_dir:
        missed_path = os.path.join(missed_dir, f"{qid}_missed.jsonl")
        if os.path.exists(missed_path):
            with open(missed_path) as f:
                for line in f:
                    obj = json.loads(line.strip())
                    if obj.get("segment") or obj.get("text"):
                        missed_docs.append(obj)
            return missed_docs

    # Use searcher
    if searcher:
        for did, rel in missed_ids.items():
            doc = searcher.doc(did)
            if doc:
                raw = json.loads(doc.raw())
                text = raw.get("segment", raw.get("body", raw.get("text", "")))
                missed_docs.append({
                    "doc_id": did,
                    "relevance": rel,
                    "segment": text,
                })
            else:
                missed_docs.append({
                    "doc_id": did,
                    "relevance": rel,
                })

    return missed_docs


def _get_found_ids(qid, found_dir):
    """Get set of found doc IDs from inspection files."""
    found_path = os.path.join(found_dir, f"{qid}_found.jsonl")
    found_ids = set()
    if os.path.exists(found_path):
        with open(found_path) as f:
            for line in f:
                obj = json.loads(line.strip())
                found_ids.add(obj.get("doc_id", ""))
    return found_ids


def analyze_query_keywords(qid, missed_docs, keywords):
    """Analyze keyword coverage for a single query's missed docs."""
    docs_with_text = [d for d in missed_docs if d.get("segment") or d.get("text")]
    if not docs_with_text:
        return None

    def get_text(doc):
        return doc.get("segment", doc.get("text", ""))

    n_docs = len(docs_with_text)

    # Per-keyword counts
    kw_hits = {}
    for term in keywords["all_terms"]:
        n = sum(1 for d in docs_with_text if _check_kw(get_text(d), term))
        if n > 0:
            kw_hits[term] = n

    # Per-group coverage
    group_coverage = {}
    for name, terms in keywords["key_groups"].items():
        n = sum(1 for d in docs_with_text if any(_check_kw(get_text(d), t) for t in terms))
        group_coverage[f"KEY: {name}"] = n
    for name, terms in keywords["sup_groups"].items():
        n = sum(1 for d in docs_with_text if any(_check_kw(get_text(d), t) for t in terms))
        group_coverage[f"SUP: {name}"] = n

    # Aggregate stats
    key_terms = [t for terms in keywords["key_groups"].values() for t in terms]
    sup_terms = [t for terms in keywords["sup_groups"].values() for t in terms]
    all_terms = keywords["all_terms"]

    n_any_key = sum(1 for d in docs_with_text
                    if any(_check_kw(get_text(d), t) for t in key_terms))
    n_any_sup = sum(1 for d in docs_with_text
                    if any(_check_kw(get_text(d), t) for t in sup_terms))
    n_any_term = sum(1 for d in docs_with_text
                     if any(_check_kw(get_text(d), t) for t in all_terms))
    n_key_and_sup = sum(1 for d in docs_with_text
                        if any(_check_kw(get_text(d), t) for t in key_terms)
                        and any(_check_kw(get_text(d), t) for t in sup_terms))
    n_all_keys = sum(1 for d in docs_with_text
                     if all(any(_check_kw(get_text(d), t) for t in terms)
                            for terms in keywords["key_groups"].values()))
    n_zero = n_docs - n_any_term

    return {
        "qid": qid,
        "n_missed": n_docs,
        "n_any_key": n_any_key,
        "n_any_sup": n_any_sup,
        "n_any_term": n_any_term,
        "n_key_and_sup": n_key_and_sup,
        "n_all_keys": n_all_keys,
        "n_zero": n_zero,
        "pct_any_key": n_any_key / n_docs if n_docs else 0,
        "pct_any_term": n_any_term / n_docs if n_docs else 0,
        "pct_all_keys": n_all_keys / n_docs if n_docs else 0,
        "pct_zero": n_zero / n_docs if n_docs else 0,
        "kw_hits": kw_hits,
        "group_coverage": group_coverage,
    }


def analyze_keyword_effectiveness(
    topics_path: str,
    qrels_path: str,
    expansions_path: str,
    found_dir: str = None,
    missed_dir: str = None,
    searcher=None,
    max_topics: int = None,
    output_path: str = None,
    verbose: bool = True,
):
    """
    Analyze keyword effectiveness across all queries.

    Needs either:
    - found_dir + searcher: looks up found IDs from files, fetches missed from searcher
    - missed_dir: uses pre-enriched missed JSONL files (with segment text)

    Returns list of per-query results.
    """
    from full_eval import load_topics
    from trec_output import load_qrels

    topics = load_topics(topics_path)
    if max_topics:
        topics = topics[:max_topics]
    qrels = load_qrels(qrels_path)

    results = []

    for qid, query_text in tqdm(topics, desc="Analyzing"):
        # Load keywords
        keywords = _load_keywords(qid, expansions_path)
        if not keywords["all_terms"]:
            continue

        # Get missed docs
        if missed_dir:
            missed_docs = _get_missed_texts(qid, qrels, set(), missed_dir=missed_dir)
        elif found_dir and searcher:
            found_ids = _get_found_ids(qid, found_dir)
            missed_docs = _get_missed_texts(qid, qrels, found_ids, searcher=searcher)
        else:
            print(f"  [{qid}] Need either missed_dir or found_dir+searcher")
            continue

        if not missed_docs:
            continue

        result = analyze_query_keywords(qid, missed_docs, keywords)
        if result:
            result["query"] = query_text
            results.append(result)

    # Print summary
    if verbose and results:
        print(f"\n{'='*90}")
        print(f"KEYWORD EFFECTIVENESS SUMMARY ({len(results)} queries)")
        print(f"{'='*90}")
        print(f"{'QID':<15s} {'Missed':>7s} {'Any KEY':>8s} {'All KEY':>8s} "
              f"{'KEY+SUP':>8s} {'Any Term':>9s} {'Zero':>6s}")
        print("-" * 75)

        for r in results:
            print(f"{r['qid']:<15s} {r['n_missed']:>7d} "
                  f"{r['pct_any_key']:>7.0%} {r['pct_all_keys']:>7.0%} "
                  f"{r['n_key_and_sup']:>7d}  {r['pct_any_term']:>7.0%} "
                  f"{r['pct_zero']:>5.0%}")

        # Averages
        n = len(results)
        avg_any_key = sum(r["pct_any_key"] for r in results) / n
        avg_all_keys = sum(r["pct_all_keys"] for r in results) / n
        avg_any_term = sum(r["pct_any_term"] for r in results) / n
        avg_zero = sum(r["pct_zero"] for r in results) / n

        print("-" * 75)
        print(f"{'AVERAGE':<15s} {'':>7s} "
              f"{avg_any_key:>7.0%} {avg_all_keys:>7.0%} "
              f"{'':>8s} {avg_any_term:>7.0%}  {avg_zero:>5.0%}")

        print(f"\nInterpretation:")
        print(f"  'Any KEY' high + recall low = CNF construction problem (keywords are there but not used)")
        print(f"  'Any KEY' low = keyword quality problem (need better/broader keywords)")
        print(f"  'Zero' high = keywords don't cover the topic vocabulary at all")
        print(f"  'All KEY' low = KEY groups too narrow, need broader terms per group")

    # Save
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved detailed results to {output_path}")

    return results