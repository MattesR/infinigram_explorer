"""
TF-based keyword scorer for document reranking.

Combines:
1. Aspect coverage (squared): docs matching more KEY aspects rank higher
2. Keyword TF: count of keyword occurrences, weighted by category
3. Verb bonus: presence of query-relevant verbs
4. Query specificity: docs from tighter retrieval queries rank higher

Usage:
    from idf_rank import build_scorer, score_and_rank, evaluate_ranking, evaluate_batch

    scorer = build_scorer(peek, query_data)
    ranked = score_and_rank(docs, executed_queries, scorer)
    results = evaluate_ranking(ranked, qrels_path, qid)
"""

import math
import re
from collections import defaultdict
from trec_output import load_qrels
from llm_keyword_filter import load_all_expansions


def build_scorer(
    peek,
    query_data: dict = None,
    aspect_weight: float = 10.0,
    key_tf_weight: float = 1.0,
    assoc_tf_weight: float = 0.5,
    verb_weight: float = 0.3,
    specificity_weight: float = 1.0,
):
    """
    Build a scorer from peek results and keyword data.

    Args:
        peek: Output from peek_and_grab.
        query_data: Raw keyword data dict (from kiss.jsonl) with KEY_ENTITIES,
                    ASSOCIATED_TERMS, VERBS.
        aspect_weight: Base weight for aspect coverage (squared).
        key_tf_weight: Weight per TF count for key terms.
        assoc_tf_weight: Weight per TF count for associated terms.
        verb_weight: Weight per verb match.
        specificity_weight: Weight for query specificity signal.
    """
    # Build aspect keyword lists from peek
    aspect_keywords = {}  # aspect_name -> [keyword_lower, ...]
    all_keywords = {}     # keyword_lower -> {"category": str, "weight": float, "aspect": str|None}

    # From grabbed and remaining KEY pieces
    for q in peek.get("grabbed", []):
        kw = q["description"].lower().strip()
        level = q.get("level", "")
        if "key" in level:
            all_keywords[kw] = {"category": "key", "weight": key_tf_weight}
        elif "assoc" in level:
            all_keywords[kw] = {"category": "assoc", "weight": assoc_tf_weight}

    for aspect_name, pieces in peek.get("remaining_key_pieces", {}).items():
        if aspect_name not in aspect_keywords:
            aspect_keywords[aspect_name] = []
        for piece in pieces:
            kw = piece["keyword"].lower().strip()
            aspect_keywords[aspect_name].append(kw)
            all_keywords[kw] = {
                "aspect": aspect_name,
                "category": "key",
                "weight": key_tf_weight,
            }

    # Try to assign grabbed KEY terms to aspects
    for q in peek.get("grabbed", []):
        kw = q["description"].lower().strip()
        level = q.get("level", "")
        if "key" not in level:
            continue
        for aspect_name in aspect_keywords:
            if aspect_name.lower() in kw or kw in aspect_name.lower():
                aspect_keywords[aspect_name].append(kw)
                if kw in all_keywords:
                    all_keywords[kw]["aspect"] = aspect_name
                break

    # Associated terms
    for piece in peek.get("remaining_assoc_pieces", []):
        kw = piece["keyword"].lower().strip()
        all_keywords[kw] = {"category": "assoc", "weight": assoc_tf_weight}

    for q in peek.get("grabbed", []):
        kw = q["description"].lower().strip()
        level = q.get("level", "")
        if "assoc" in level:
            all_keywords[kw] = {"category": "assoc", "weight": assoc_tf_weight}

    # Verbs from query_data
    verbs = []
    if query_data:
        verb_data = query_data.get("VERBS", {})
        if isinstance(verb_data, dict):
            for verb, expansions in verb_data.items():
                verbs.append(verb.lower())
                if isinstance(expansions, list):
                    verbs.extend([v.lower() for v in expansions])
        elif isinstance(verb_data, list):
            verbs.extend([v.lower() for v in verb_data])

    n_aspects = len(aspect_keywords)

    return {
        "aspect_keywords": aspect_keywords,
        "all_keywords": all_keywords,
        "verbs": verbs,
        "n_aspects": n_aspects,
        "aspect_weight": aspect_weight,
        "key_tf_weight": key_tf_weight,
        "assoc_tf_weight": assoc_tf_weight,
        "verb_weight": verb_weight,
        "specificity_weight": specificity_weight,
    }


def score_doc(text, scorer, query_specificity=0.0):
    """Score a single document. Returns (score, details_dict)."""
    text_lower = text.lower()
    aspect_keywords = scorer["aspect_keywords"]
    all_keywords = scorer["all_keywords"]

    # 1. Aspect coverage (squared)
    aspects_matched = set()
    for aspect_name, keywords in aspect_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                aspects_matched.add(aspect_name)
                break

    n_matched = len(aspects_matched)
    coverage_score = (n_matched ** 2) * scorer["aspect_weight"]

    # 2. Keyword TF scoring
    tf_score = 0.0
    n_key_matches = 0
    n_assoc_matches = 0
    matched_keywords = []

    for kw, info in all_keywords.items():
        count = text_lower.count(kw)
        if count > 0:
            # Log-dampened TF
            tf = 1 + math.log(count) if count > 1 else 1.0
            tf_score += tf * info["weight"]
            matched_keywords.append((kw, count))
            if info["category"] == "key":
                n_key_matches += 1
            else:
                n_assoc_matches += 1

    # 3. Verb bonus
    verb_score = 0.0
    verbs_found = []
    for verb in scorer["verbs"]:
        if verb in text_lower:
            verb_score += scorer["verb_weight"]
            verbs_found.append(verb)

    # 4. Query specificity
    spec_score = query_specificity * scorer["specificity_weight"]

    # Combined
    score = coverage_score + tf_score + verb_score + spec_score

    return score, {
        "aspects_matched": aspects_matched,
        "n_aspects": n_matched,
        "coverage_score": coverage_score,
        "tf_score": tf_score,
        "verb_score": verb_score,
        "spec_score": spec_score,
        "n_key_matches": n_key_matches,
        "n_assoc_matches": n_assoc_matches,
        "n_verbs": len(verbs_found),
        "matched_keywords": matched_keywords,
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
        print(f"\nTF scoring: {len(scored)} docs")
        coverage_dist = {}
        for _, _, _, details in scored:
            n = details.get("n_aspects", 0)
            coverage_dist[n] = coverage_dist.get(n, 0) + 1
        print(f"  Aspect coverage distribution:")
        for n in sorted(coverage_dist.keys()):
            print(f"    {n}/{scorer['n_aspects']} aspects: {coverage_dist[n]} docs")
        print(f"  Top 5:")
        for score, did, doc, details in scored[:5]:
            print(f"    {score:.2f}  asp={details['n_aspects']}  "
                  f"kw={details['n_key_matches']}  "
                  f"assoc={details['n_assoc_matches']}  "
                  f"verbs={details['n_verbs']}  "
                  f"spec={details['spec_score']:.3f}  {did}")

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
        top_k_list = [10, 100, 1000, 2000]

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
    expansions_path: str,
    top_k_list: list = None,
    aspect_weight: float = 10.0,
    key_tf_weight: float = 1.0,
    assoc_tf_weight: float = 0.5,
    verb_weight: float = 0.3,
    specificity_weight: float = 1.0,
    verbose: bool = True,
):
    """Evaluate TF scoring for multiple queries."""
    if top_k_list is None:
        top_k_list = [10, 100, 1000, 2000]

    qrels = load_qrels(qrels_path)
    all_expansions = load_all_expansions(expansions_path)
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

        query_data = all_expansions.get(qid, {})

        scorer = build_scorer(
            peek, query_data=query_data,
            aspect_weight=aspect_weight,
            key_tf_weight=key_tf_weight,
            assoc_tf_weight=assoc_tf_weight,
            verb_weight=verb_weight,
            specificity_weight=specificity_weight,
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
        print(f"\n{'='*80}")
        print(f"TF Scoring Summary ({n} topics)")
        print(f"  Weights: aspect={aspect_weight}, key_tf={key_tf_weight}, "
              f"assoc_tf={assoc_tf_weight}, verb={verb_weight}, spec={specificity_weight}")
        print(f"{'='*80}")

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


def grid_search_scorer(
    retrieval_results: list,
    peek_per_qid: dict,
    executed_per_qid: dict,
    qrels_path: str,
    expansions_path: str,
    param_grid: dict = None,
    top_k_list: list = None,
    verbose: bool = True,
):
    """
    Grid search over scorer hyperparameters.

    Args:
        param_grid: Dict of param_name -> list of values.
            Default tests aspect_weight, key_tf_weight, verb_weight, specificity_weight.
    """
    from itertools import product

    if top_k_list is None:
        top_k_list = [100, 1000]

    if param_grid is None:
        param_grid = {
            "aspect_weight": [5.0, 10.0, 20.0],
            "key_tf_weight": [0.5, 1.0, 2.0],
            "verb_weight": [0.0, 0.3, 1.0],
            "specificity_weight": [0.0, 1.0, 3.0],
        }

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    configs = []
    for combo in product(*param_values):
        config = dict(zip(param_names, combo))
        configs.append(config)

    if verbose:
        print(f"Grid search: {len(configs)} configurations")
        print(f"  Parameters: {param_names}")

    results_all = []

    for i, config in enumerate(configs):
        label = " | ".join(f"{k}={v}" for k, v in config.items())

        eval_results = evaluate_batch(
            retrieval_results, peek_per_qid, executed_per_qid,
            qrels_path, expansions_path,
            top_k_list=top_k_list,
            verbose=False,
            **config,
        )

        if not eval_results:
            continue

        n = len(eval_results)
        row = {"config": label}
        row.update(config)
        for k in top_k_list:
            recalls = [r[k]["recall"] for r in eval_results]
            row[f"R@{k}"] = round(sum(recalls) / n, 4)

        results_all.append(row)

    df = pd.DataFrame(results_all)

    if verbose and not df.empty:
        # Sort by R@1000 (or largest cutoff)
        sort_col = f"R@{max(top_k_list)}"
        df = df.sort_values(sort_col, ascending=False)

        print(f"\n{'='*100}")
        print(f"SCORER GRID SEARCH ({len(df)} configs)")
        print(f"{'='*100}")

        header = f"{'Config':<55s}"
        for k in top_k_list:
            header += f" {'R@'+str(k):>8s}"
        print(header)
        print("-" * len(header))

        for _, r in df.head(20).iterrows():
            line = f"{r['config']:<55s}"
            for k in top_k_list:
                line += f" {r[f'R@{k}']:>8.4f}"
            print(line)

        print(f"\n  Best: {df.iloc[0]['config']}")

    return df


# Need pandas for grid search
import pandas as pd