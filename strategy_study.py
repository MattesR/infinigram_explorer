"""
Strategy coverage analysis - pure recall, no engine calls.

For each document, check which keywords match. Then for each strategy,
count how many docs would be covered and how many queries would be fired.

Usage:
    from strategy_study import compare_strategies

    df = compare_strategies(
        found_dir="./inspection/prog/",
        expansions_path="./kiss.jsonl",
        tokenizer=tokenizer,
        engine=engine,
    )
"""

import os
import json
import pandas as pd
from glob import glob
from collections import defaultdict
from itertools import combinations, product
from tqdm import tqdm
from progressive_queries import build_pieces
from tightest_queries import collect_matches, group_by_aspect, cnf_match_positions


def _analyze_doc_matches(doc_text, pieces, tokenizer):
    """
    Check which key and associated pieces match in a document.
    Returns (grouped_key_matches, assoc_matches).
    """
    # Flatten key pieces
    key_pieces_flat = []
    for aspect_name, piece_list in pieces["key_pieces"].items():
        for p in piece_list:
            p_copy = dict(p)
            p_copy["source_aspect"] = aspect_name
            key_pieces_flat.append(p_copy)

    tokens = list(tokenizer.encode(doc_text, add_special_tokens=False))
    key_matches = collect_matches(key_pieces_flat, tokens)
    assoc_matches = collect_matches(pieces["associated"], tokens)
    grouped = group_by_aspect(key_matches)

    return grouped, assoc_matches, tokens


def _count_strategy_queries(pieces):
    """
    Count how many queries each strategy would fire (worst case).
    Based on the number of keywords per aspect and associated terms.
    """
    aspect_sizes = {name: len(kws) for name, kws in pieces["key_pieces"].items()}
    n_assoc = len(pieces["associated"])
    aspect_names = list(aspect_sizes.keys())
    n_aspects = len(aspect_names)

    counts = {}

    # Standalone key: one query per keyword per aspect
    counts["standalone_key"] = sum(aspect_sizes.values())

    # Standalone assoc: one query per associated term
    counts["standalone_assoc"] = n_assoc

    # Pairwise key: for each pair of aspects, product of their sizes
    pairwise = 0
    for a, b in combinations(aspect_names, 2):
        pairwise += aspect_sizes[a] * aspect_sizes[b]
    counts["pairwise_key"] = pairwise

    # All-aspects key: product of all aspect sizes
    if n_aspects >= 2:
        all_prod = 1
        for s in aspect_sizes.values():
            all_prod *= s
        counts["all_aspects_key"] = all_prod
    else:
        counts["all_aspects_key"] = sum(aspect_sizes.values())

    # Key + assoc: each key keyword × each associated term
    counts["key_plus_assoc"] = sum(aspect_sizes.values()) * n_assoc

    # Pairwise key + assoc
    counts["pairwise_key_plus_assoc"] = pairwise * n_assoc

    # All-aspects + assoc
    counts["all_aspects_plus_assoc"] = counts.get("all_aspects_key", 0) * n_assoc

    return counts


def _check_coverage(grouped, assoc_matches, aspect_names):
    """
    For each strategy, check if this document would be covered.
    Returns dict of strategy -> bool.
    """
    has_key = {name: len(matches) > 0 for name, matches in grouped.items()}
    has_any_key = any(has_key.values())
    has_any_assoc = len(assoc_matches) > 0
    n_aspects_matched = sum(1 for v in has_key.values() if v)

    results = {}

    # Standalone key: any key keyword matches
    results["standalone_key"] = has_any_key

    # Standalone assoc: any associated keyword matches
    results["standalone_assoc"] = has_any_assoc

    # Any standalone: key or assoc
    results["any_standalone"] = has_any_key or has_any_assoc

    # Pairwise key: at least 2 aspects have matches
    results["pairwise_key"] = n_aspects_matched >= 2

    # All-aspects key: all aspects have matches
    results["all_aspects_key"] = n_aspects_matched == len(aspect_names) and len(aspect_names) >= 2

    # Key + assoc: at least one key AND at least one assoc
    results["key_plus_assoc"] = has_any_key and has_any_assoc

    # Pairwise key + assoc: 2+ aspects AND assoc
    results["pairwise_key_plus_assoc"] = n_aspects_matched >= 2 and has_any_assoc

    # All-aspects + assoc: all aspects AND assoc
    results["all_aspects_plus_assoc"] = (
        n_aspects_matched == len(aspect_names) and len(aspect_names) >= 2 and has_any_assoc
    )

    return results, n_aspects_matched


def compare_strategies(
    found_dir: str,
    expansions_path: str,
    tokenizer,
    engine,
    pattern: str = "*_found.jsonl",
    verbose: bool = True,
):
    """
    Compare query construction strategies across all queries.
    No engine calls - just measures recall from keyword presence.

    Returns DataFrame with per-strategy, per-query coverage.
    """
    files = sorted(glob(os.path.join(found_dir, pattern)))
    if verbose:
        print(f"Analyzing {len(files)} queries for strategy coverage")

    rows = []
    all_query_counts = []

    for fpath in tqdm(files, desc="Analyzing", disable=not verbose):
        fname = os.path.basename(fpath)
        qid = fname.replace("_found.jsonl", "")

        # Load docs
        docs = []
        with open(fpath) as f:
            for line in f:
                obj = json.loads(line.strip())
                if obj.get("text"):
                    docs.append(obj)

        if not docs:
            continue

        pieces = build_pieces(qid, expansions_path, tokenizer, engine, verbose=False)
        aspect_names = list(pieces["key_pieces"].keys())
        n_docs = len(docs)

        # Count queries per strategy
        query_counts = _count_strategy_queries(pieces)
        query_counts["qid"] = qid
        query_counts["n_aspects"] = len(aspect_names)
        all_query_counts.append(query_counts)

        # Check coverage per doc per strategy
        strategy_covered = defaultdict(int)
        aspect_dist = defaultdict(int)

        for doc in docs:
            grouped, assoc_matches, tokens = _analyze_doc_matches(
                doc["text"], pieces, tokenizer)

            coverage, n_aspects_matched = _check_coverage(grouped, assoc_matches, aspect_names)
            aspect_dist[n_aspects_matched] += 1

            for strategy, covered in coverage.items():
                if covered:
                    strategy_covered[strategy] += 1

        # Build rows
        for strategy in ["standalone_key", "standalone_assoc", "any_standalone",
                          "pairwise_key", "all_aspects_key",
                          "key_plus_assoc", "pairwise_key_plus_assoc",
                          "all_aspects_plus_assoc"]:
            covered = strategy_covered.get(strategy, 0)
            n_queries = query_counts.get(strategy, 0)
            rows.append({
                "qid": qid,
                "strategy": strategy,
                "n_docs": n_docs,
                "docs_covered": covered,
                "coverage_pct": round(covered / n_docs, 4) if n_docs else 0,
                "n_queries": n_queries,
                "n_aspects": len(aspect_names),
            })

        # Also add aspect distribution
        for n_asp in range(len(aspect_names) + 1):
            rows.append({
                "qid": qid,
                "strategy": f"aspects_{n_asp}",
                "n_docs": n_docs,
                "docs_covered": aspect_dist.get(n_asp, 0),
                "coverage_pct": round(aspect_dist.get(n_asp, 0) / n_docs, 4) if n_docs else 0,
                "n_queries": 0,
                "n_aspects": len(aspect_names),
            })

    df = pd.DataFrame(rows)
    df_queries = pd.DataFrame(all_query_counts)

    if verbose and not df.empty:
        print(f"\n{'='*100}")
        print(f"STRATEGY COVERAGE ANALYSIS ({len(files)} topics)")
        print(f"{'='*100}")

        # Strategy summary
        strategies = ["standalone_key", "standalone_assoc", "any_standalone",
                       "pairwise_key", "all_aspects_key",
                       "key_plus_assoc", "pairwise_key_plus_assoc",
                       "all_aspects_plus_assoc"]

        print(f"\n  {'Strategy':<28s} {'Avg Cov%':>9s} {'Min Cov%':>9s} {'Max Cov%':>9s} {'Avg #Q':>8s}")
        print(f"  {'-'*65}")

        for strategy in strategies:
            sub = df[df["strategy"] == strategy]
            if sub.empty:
                continue
            qc = df_queries[strategy].mean() if strategy in df_queries.columns else 0
            print(f"  {strategy:<28s} "
                  f"{sub['coverage_pct'].mean():>8.1%} "
                  f"{sub['coverage_pct'].min():>8.1%} "
                  f"{sub['coverage_pct'].max():>8.1%} "
                  f"{qc:>8.1f}")

        # Aspect distribution
        print(f"\n  Aspect coverage distribution (avg across topics):")
        for n_asp in range(5):
            sub = df[df["strategy"] == f"aspects_{n_asp}"]
            if not sub.empty and sub["coverage_pct"].mean() > 0:
                print(f"    {n_asp} aspects: {sub['coverage_pct'].mean():>6.1%}")

        # Per-query table
        print(f"\n  Per-query coverage:")
        header = f"  {'QID':<15s} {'Docs':>5s} {'#Asp':>4s}"
        for s in strategies:
            label = s.replace("_", "")[:8]
            header += f" {label:>8s}"
        print(header)
        print(f"  {'-'*len(header)}")

        for qid in df["qid"].unique():
            if f"aspects_0" in df[df["qid"] == qid]["strategy"].values:
                qdf = df[df["qid"] == qid]
                n_docs = qdf["n_docs"].iloc[0]
                n_asp = qdf["n_aspects"].iloc[0]
                print(f"  {qid:<15s} {n_docs:>5d} {n_asp:>4d}", end="")
                for s in strategies:
                    row = qdf[qdf["strategy"] == s]
                    if not row.empty:
                        print(f" {row['coverage_pct'].iloc[0]:>7.1%}", end="")
                    else:
                        print(f" {'N/A':>8s}", end="")
                print()

        # Query count table
        print(f"\n  Queries per strategy (worst case):")
        print(f"  {'QID':<15s}", end="")
        for s in strategies:
            if s in df_queries.columns:
                label = s.replace("_", "")[:8]
                print(f" {label:>8s}", end="")
        print()
        for _, r in df_queries.iterrows():
            print(f"  {r['qid']:<15s}", end="")
            for s in strategies:
                if s in r:
                    print(f" {r[s]:>8.0f}", end="")
            print()

    return df, df_queries