"""
Aspect-aware document scoring for infini-gram retrieval.

Combines:
1. Aspect coverage: how many KEY_ENTITY aspects are represented in the doc
2. Query specificity: docs from tighter (lower-count) queries rank higher
3. Keyword density: number of keyword matches, weighted by category

Usage:
    from idf_rank import build_scorer, score_and_rank, evaluate_ranking

    scorer = build_scorer(peek)
    ranked = score_and_rank(docs, executed_queries, scorer)
    results = evaluate_ranking(ranked, qrels_path, qid)
"""

import math
from trec_output import load_qrels


def build_scorer(
    peek,
    aspect_coverage_weights: dict = None,
    key_term_weight: float = 1.0,
    assoc_term_weight: float = 0.3,
):
    """
    Build a scorer from peek results.

    Args:
        peek: Output from peek_and_grab.
        aspect_coverage_weights: Custom weights per number of aspects covered.
        key_term_weight: Weight per matched KEY term.
        assoc_term_weight: Weight per matched associated term.
    """
    aspect_keywords = {}  # aspect_name -> [keyword_lower, ...]
    all_keywords = {}     # keyword_lower -> {"category": str, "weight": float, "aspect": str|None}

    # Grabbed KEY terms
    for q in peek.get("grabbed", []):
        kw = q["description"].lower().strip()
        level = q.get("level", "")
        if "key" in level:
            all_keywords[kw] = {"category": "key", "weight": key_term_weight}
        elif "assoc" in level:
            all_keywords[kw] = {"category": "assoc", "weight": assoc_term_weight}

    # Remaining KEY pieces (organized by aspect)
    for aspect_name, pieces in peek.get("remaining_key_pieces", {}).items():
        if aspect_name not in aspect_keywords:
            aspect_keywords[aspect_name] = []
        for piece in pieces:
            kw = piece["keyword"].lower().strip()
            aspect_keywords[aspect_name].append(kw)
            all_keywords[kw] = {
                "aspect": aspect_name,
                "category": "key",
                "weight": key_term_weight,
            }

    # Try to assign grabbed KEY terms to aspects by checking original pieces
    # Use the key_pieces from the pieces structure if available
    for q in peek.get("grabbed", []):
        kw = q["description"].lower().strip()
        level = q.get("level", "")
        if "key" not in level:
            continue
        # Check each aspect for a match
        for aspect_name in aspect_keywords:
            if aspect_name.lower() in kw or kw in aspect_name.lower():
                aspect_keywords[aspect_name].append(kw)
                if kw in all_keywords:
                    all_keywords[kw]["aspect"] = aspect_name
                break

    # Remaining associated pieces
    for piece in peek.get("remaining_assoc_pieces", []):
        kw = piece["keyword"].lower().strip()
        all_keywords[kw] = {"category": "assoc", "weight": assoc_term_weight}

    # Grabbed associated terms
    for q in peek.get("grabbed", []):
        kw = q["description"].lower().strip()
        level = q.get("level", "")
        if "assoc" in level:
            all_keywords[kw] = {"category": "assoc", "weight": assoc_term_weight}

    # Build aspect coverage weights
    n_aspects = len(aspect_keywords)
    if aspect_coverage_weights is None:
        aspect_coverage_weights = {}
        for i in range(1, n_aspects + 1):
            aspect_coverage_weights[i] = float(i ** 2)

    return {
        "aspect_keywords": aspect_keywords,
        "all_keywords": all_keywords,
        "aspect_coverage_weights": aspect_coverage_weights,
        "n_aspects": n_aspects,
        "key_term_weight": key_term_weight,
        "assoc_term_weight": assoc_term_weight,
    }


def score_doc(text, scorer, query_specificity=0.0):
    """Score a single document. Returns (score, details_dict)."""
    text_lower = text.lower()
    aspect_keywords = scorer["aspect_keywords"]
    all_keywords = scorer["all_keywords"]
    coverage_weights = scorer["aspect_coverage_weights"]

    # 1. Aspect coverage
    aspects_matched = set()
    for aspect_name, keywords in aspect_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                aspects_matched.add(aspect_name)
                break

    n_matched = len(aspects_matched)
    coverage_score = coverage_weights.get(n_matched, 0.0)

    # 2. Keyword density
    density_score = 0.0
    matched_terms = []
    for kw, info in all_keywords.items():
        if kw in text_lower:
            density_score += info["weight"]
            matched_terms.append(kw)

    # 3. Combined
    score = coverage_score + query_specificity + density_score * 0.1

    return score, {
        "aspects_matched": aspects_matched,
        "n_aspects": n_matched,
        "coverage_score": coverage_score,
        "query_specificity": query_specificity,
        "density_score": density_score,
        "n_terms_matched": len(matched_terms),
        "matched_terms": matched_terms,
    }


def score_and_rank(docs, executed_queries, scorer, verbose=True):
    """Score and rank all documents."""
    # Build query lookup for specificity
    query_lookup = {}
    for i, q in enumerate(executed_queries):
        desc = q.get("description", f"query_{i}")
        count = q.get("estimated_count", q.get("cnt", 999999))
        query_lookup[desc] = count
        query_lookup[i] = count

    # Deduplicate docs
    doc_map = {}
    for d in docs:
        did = d.get("doc_id", "")
        if not did:
            continue
        if did not in doc_map:
            doc_map[did] = dict(d)
        else:
            existing_fq = set(doc_map[did].get("from_queries", []))
            new_fq = set(d.get("from_queries", []))
            doc_map[did]["from_queries"] = list(existing_fq | new_fq)

    scored = []
    for did, doc in doc_map.items():
        text = doc.get("text", "")

        # Query specificity from tightest query
        from_queries = doc.get("from_queries", [])
        min_count = 999999
        for fq in from_queries:
            count = query_lookup.get(fq, 999999)
            min_count = min(min_count, count)

        query_spec = 1.0 / math.log(2 + min_count) if min_count < 999999 else 0.0

        if not text:
            scored.append((0.0, did, doc, {"n_aspects": 0}))
            continue

        score, details = score_doc(text, scorer, query_specificity=query_spec)
        scored.append((score, did, doc, details))

    scored.sort(key=lambda x: x[0], reverse=True)

    if verbose:
        print(f"\nAspect-aware scoring: {len(scored)} docs")
        coverage_dist = {}
        for _, _, _, details in scored:
            n = details.get("n_aspects", 0)
            coverage_dist[n] = coverage_dist.get(n, 0) + 1
        print(f"  Aspect coverage distribution:")
        for n in sorted(coverage_dist.keys()):
            print(f"    {n}/{scorer['n_aspects']} aspects: {coverage_dist[n]} docs")
        print(f"  Top 5:")
        for score, did, doc, details in scored[:5]:
            print(f"    {score:.2f}  aspects={details['n_aspects']}  "
                  f"terms={details['n_terms_matched']}  "
                  f"spec={details['query_specificity']:.3f}  {did}")

    return scored


def evaluate_ranking(
    ranked,
    qrels_path: str = None,
    qid: str = None,
    qrels: dict = None,
    top_k_list: list = None,
    verbose: bool = True,
):
    """Evaluate ranked list against qrels at multiple cutoffs."""
    if top_k_list is None:
        top_k_list = [10, 100, 1000]

    if qrels is None:
        qrels = load_qrels(qrels_path)

    relevant = {did: rel for did, rel in qrels.get(qid, {}).items() if rel > 0}
    all_ids = {did for _, did, _, _ in ranked}
    total_found = len(all_ids & set(relevant.keys()))

    results = {}
    for k in top_k_list:
        top_ids = {did for _, did, _, _ in ranked[:k]}
        found = top_ids & set(relevant.keys())
        recall = len(found) / len(relevant) if relevant else 0
        precision = len(found) / min(k, len(ranked)) if ranked else 0
        results[k] = {"recall": recall, "precision": precision, "found": len(found)}

    if verbose:
        print(f"\nRanking evaluation for {qid}")
        print(f"  Total docs: {len(ranked)}")
        print(f"  Relevant in pool: {total_found}/{len(relevant)}")
        for k in top_k_list:
            r = results[k]
            print(f"  @{k:<5d} Recall: {r['recall']:.3f} ({r['found']}/{len(relevant)})  "
                  f"Precision: {r['precision']:.4f}")

    results["total_found"] = total_found
    results["n_relevant"] = len(relevant)
    return results


def evaluate_batch(
    retrieval_results: list,
    peek_per_qid: dict,
    executed_per_qid: dict,
    qrels_path: str,
    top_k_list: list = None,
    aspect_coverage_weights: dict = None,
    key_term_weight: float = 1.0,
    assoc_term_weight: float = 0.3,
    verbose: bool = True,
):
    """Evaluate aspect-aware ranking for multiple queries."""
    if top_k_list is None:
        top_k_list = [10, 100, 1000]

    qrels = load_qrels(qrels_path)
    all_results = []

    for r in retrieval_results:
        qid = r["qid"]
        docs = r.get("docs", [])
        if not docs:
            continue

        peek = peek_per_qid.get(qid)
        executed = executed_per_qid.get(qid, [])
        if peek is None:
            continue

        scorer = build_scorer(
            peek,
            aspect_coverage_weights=aspect_coverage_weights,
            key_term_weight=key_term_weight,
            assoc_term_weight=assoc_term_weight,
        )
        ranked = score_and_rank(docs, executed, scorer, verbose=False)
        result = evaluate_ranking(ranked, qid=qid, qrels=qrels,
                                   top_k_list=top_k_list, verbose=False)
        result["qid"] = qid
        result["n_retrieved"] = r.get("n_retrieved", len(docs))
        result["raw_recall"] = r.get("recall", 0)
        result["raw_found"] = r.get("n_found", 0)
        all_results.append(result)

    if verbose and all_results:
        n = len(all_results)
        print(f"\n{'='*70}")
        print(f"Aspect-Aware Ranking Summary ({n} topics)")
        print(f"  Weights: key={key_term_weight}, assoc={assoc_term_weight}")
        print(f"{'='*70}")

        avg_raw = sum(r["raw_recall"] for r in all_results) / n
        print(f"  Raw retrieval avg recall: {avg_raw:.4f}")

        header = f"  {'':>15s}"
        for k in top_k_list:
            header += f" {'R@'+str(k):>10s} {'P@'+str(k):>10s}"
        print(header)

        print(f"  {'Average':>15s}", end="")
        for k in top_k_list:
            recalls = [r[k]["recall"] for r in all_results]
            precisions = [r[k]["precision"] for r in all_results]
            print(f" {sum(recalls)/n:>10.4f} {sum(precisions)/n:>10.4f}", end="")
        print()

        print(f"\n{'QID':<15s} {'Retr':>6s} {'Rel':>5s} {'Raw':>6s}", end="")
        for k in top_k_list:
            print(f" {'R@'+str(k):>7s}", end="")
        print()
        print("-" * (35 + 8 * len(top_k_list)))

        for r in all_results:
            print(f"{r['qid']:<15s} {r['n_retrieved']:>6d} {r['n_relevant']:>5d} "
                  f"{r['raw_found']:>6d}", end="")
            for k in top_k_list:
                print(f" {r[k]['recall']:>7.3f}", end="")
            print()

    return all_results