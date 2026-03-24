"""
Construct CNF queries by combining WordPiece clusters with
SPLADE × IDF scoring.

Pipeline:
1. Start with scored SPLADE tokens: (token, splade_score, tup, combined_score)
2. Split into informative (low TUP) and common pool (high TUP)
3. Cluster informative tokens by WordPiece stems (OR groups)
4. Form CNF queries: anchor AND informative, optionally with common pool
5. Score each pair and rank them

Usage:
    from query_construction import build_cnf_queries
    queries = build_cnf_queries(top_tokens, tokenizer_infini)
"""

import numpy as np
from itertools import combinations
from wordpiece_cluster import cluster_tokens, clusters_to_or_clauses


def score_clusters(clusters, scored_tokens):
    """
    Assign each cluster a score based on its tokens' combined scores.

    Args:
        clusters: List of cluster dicts from cluster_tokens().
        scored_tokens: Dict mapping token_string -> combined_score.

    Returns:
        List of cluster dicts with added 'combined_score' field.
    """
    for c in clusters:
        token_scores = []
        for token, splade_score in c["tokens"]:
            if token in scored_tokens:
                token_scores.append(scored_tokens[token])
            else:
                token_scores.append(0.0)
        c["combined_score"] = max(token_scores) if token_scores else 0.0
    return clusters


def build_cnf_queries(
    top_tokens: list[tuple],
    tokenizer,
    min_stem_len: int = 5,
    max_queries: int = 50,
    min_cluster_score: float = 0.0,
    anchor_score: float = 0.9,
    max_anchor_tup: float = 1e-4,
    strategy: str = "anchor",
):
    """
    Build ranked CNF queries from scored SPLADE tokens.

    Tokens are split into three groups:
    1. Anchors: clusters where at least one token has splade >= anchor_score
       AND tup <= max_anchor_tup. Always included.
    2. Informative: clusters with tup <= max_anchor_tup that aren't anchors.
       Filtered by min_cluster_score.
    3. Common pool: all tokens with tup > max_anchor_tup are merged into one
       big OR clause. Can be used as an optional extra AND filter.

    Args:
        top_tokens: List of (token, splade_score, tup, combined_score) tuples.
        tokenizer: Infini-gram tokenizer for encoding to token IDs.
        min_stem_len: For WordPiece clustering.
        max_queries: Maximum number of CNF queries to return.
        min_cluster_score: Drop non-anchor clusters below this combined score.
        anchor_score: Minimum raw SPLADE score for a cluster to be an anchor.
        max_anchor_tup: Maximum TUP for a token to qualify as anchor or informative.
            Tokens above this go into the common pool.
        strategy: How to form pairs:
            - "anchor": anchor AND informative (+ anchor AND anchor)
            - "anchor_plus_common": same but adds common pool as extra AND clause
            - "all_pairs": all cluster pairs from anchors + informative

    Returns:
        List of query dicts sorted by score, each with:
            - 'cnf': ready-to-use CNF query for engine.find_cnf()
            - 'description': human-readable description
            - 'score': combined score of the pair
            - 'clusters': the cluster dicts involved
    """
    # Build lookups
    scored_lookup = {t[0]: t[3] for t in top_tokens}
    splade_lookup = {t[0]: t[1] for t in top_tokens}
    tup_lookup = {t[0]: t[2] for t in top_tokens}

    # Split into common (high TUP) and informative (low TUP) tokens
    common_tokens = [(t[0], t[1]) for t in top_tokens if t[2] > max_anchor_tup]
    informative_tokens = [(t[0], t[1]) for t in top_tokens if t[2] <= max_anchor_tup]

    print(f"\nToken split (max_anchor_tup={max_anchor_tup:.0e}):")
    print(f"  Informative (tup <= threshold): {len(informative_tokens)}")
    print(f"  Common pool (tup > threshold):  {len(common_tokens)}")
    if common_tokens:
        print(f"    Common: {', '.join(t for t, s in common_tokens)}")

    if len(informative_tokens) < 2:
        print("  Not enough informative tokens to cluster")
        return []

    # Cluster only informative tokens
    clusters = cluster_tokens(informative_tokens, min_stem_len=min_stem_len, show_steps=False)
    clusters = score_clusters(clusters, scored_lookup)

    # Determine anchor status: high SPLADE + low TUP
    for c in clusters:
        max_splade = max(splade_lookup.get(t, 0.0) for t, s in c["tokens"])
        min_tup = min(tup_lookup.get(t, 1.0) for t, s in c["tokens"])
        c["max_splade"] = max_splade
        c["min_tup"] = min_tup
        c["is_anchor"] = max_splade >= anchor_score and min_tup <= max_anchor_tup

    anchors = [c for c in clusters if c["is_anchor"]]
    others = [c for c in clusters if not c["is_anchor"] and c["combined_score"] >= min_cluster_score]

    anchors.sort(key=lambda x: x["max_splade"], reverse=True)
    others.sort(key=lambda x: x["combined_score"], reverse=True)

    print(f"\nAnchors ({len(anchors)} clusters, splade >= {anchor_score}, tup <= {max_anchor_tup:.0e}):")
    for c in anchors:
        tokens_str = " | ".join(f"{t}" for t, s in c["tokens"])
        print(f"  [{c['max_splade']:.2f} splade, {c['combined_score']:.2f} combined] "
              f"{c['stem']:<15s} -> {tokens_str}")

    print(f"\nInformative clusters ({len(others)}, combined >= {min_cluster_score}):")
    for c in others:
        tokens_str = " | ".join(f"{t}" for t, s in c["tokens"])
        print(f"  [{c['max_splade']:.2f} splade, {c['combined_score']:.2f} combined] "
              f"{c['stem']:<15s} -> {tokens_str}")

    # Convert to OR clauses
    anchor_clauses = clusters_to_or_clauses(anchors, tokenizer)
    other_clauses = clusters_to_or_clauses(others, tokenizer)
    all_informative_clauses = clusters_to_or_clauses(anchors + others, tokenizer)

    # Build common pool OR clause
    common_clause = None
    if common_tokens:
        common_ids = []
        common_names = []
        for token, score in common_tokens:
            clean = token.lstrip("#")
            ids = tokenizer.encode(clean, add_special_tokens=False)
            if ids:
                common_ids.append(ids)
                common_names.append(token)
        if common_ids:
            common_clause = {
                "clause": common_ids,
                "tokens": common_names,
                "max_score": max(splade_lookup.get(t, 0.0) for t in common_names),
            }
            print(f"\nCommon pool OR clause: ({' OR '.join(common_names)})")

    # Build CNF query pairs
    queries = []
    use_common = strategy == "anchor_plus_common" and common_clause is not None

    if strategy in ("anchor", "anchor_plus_common"):
        # Anchor AND informative
        for a in anchor_clauses:
            for b in other_clauses:
                cnf = [a["clause"], b["clause"]]
                desc = f"({' OR '.join(a['tokens'])}) AND ({' OR '.join(b['tokens'])})"
                if use_common:
                    cnf.append(common_clause["clause"])
                    desc += f" AND ({' OR '.join(common_clause['tokens'])})"

                pair_score = a["max_score"] * b["max_score"]
                queries.append({
                    "cnf": cnf,
                    "description": desc,
                    "score": pair_score,
                    "clusters": [a, b] + ([common_clause] if use_common else []),
                })

        # Anchor AND anchor
        for i, j in combinations(range(len(anchor_clauses)), 2):
            a = anchor_clauses[i]
            b = anchor_clauses[j]
            cnf = [a["clause"], b["clause"]]
            desc = f"({' OR '.join(a['tokens'])}) AND ({' OR '.join(b['tokens'])})"
            if use_common:
                cnf.append(common_clause["clause"])
                desc += f" AND ({' OR '.join(common_clause['tokens'])})"

            pair_score = a["max_score"] * b["max_score"]
            queries.append({
                "cnf": cnf,
                "description": desc,
                "score": pair_score,
                "clusters": [a, b] + ([common_clause] if use_common else []),
            })

    elif strategy == "all_pairs":
        for i, j in combinations(range(len(all_informative_clauses)), 2):
            a = all_informative_clauses[i]
            b = all_informative_clauses[j]
            cnf = [a["clause"], b["clause"]]
            pair_score = a["max_score"] * b["max_score"]

            queries.append({
                "cnf": cnf,
                "description": f"({' OR '.join(a['tokens'])}) AND ({' OR '.join(b['tokens'])})",
                "score": pair_score,
                "clusters": [a, b],
            })

    # Sort by score descending
    queries.sort(key=lambda x: x["score"], reverse=True)
    queries = queries[:max_queries]

    print(f"\nTop {len(queries)} CNF queries:")
    for i, q in enumerate(queries):
        print(f"  {i+1:3d}. [{q['score']:8.2f}] {q['description']}")

    return queries


def run_queries(engine, queries, max_clause_freq=None):
    """
    Execute the CNF queries against the engine and add hit counts.

    Args:
        engine: Infini-gram engine instance.
        queries: List of query dicts from build_cnf_queries().
        max_clause_freq: If set, passed to engine.find_cnf.

    Returns:
        Same list with added 'cnt', 'approx', 'ptrs_by_shard' fields.
    """
    from tqdm import tqdm
    import time

    print(f"Running {len(queries)} CNF queries...")
    for q in tqdm(queries, desc="Querying"):
        kwargs = {"cnf": q["cnf"]}
        if max_clause_freq is not None:
            kwargs["max_clause_freq"] = max_clause_freq

        t0 = time.perf_counter()
        result = engine.find_cnf(**kwargs)
        q["time"] = time.perf_counter() - t0

        if "error" in result:
            q["cnt"] = 0
            q["approx"] = False
            q["error"] = result["error"]
        else:
            q["cnt"] = result["cnt"]
            q["approx"] = result.get("approx", False)
            q["ptrs_by_shard"] = result.get("ptrs_by_shard", [])

    # Print summary sorted by count
    queries.sort(key=lambda x: x["cnt"], reverse=True)
    print(f"\nResults (sorted by count):")
    print(f"{'#':>4s} {'Count':>10s} {'Approx':>6s} {'Score':>8s} {'Query'}")
    print("-" * 80)
    for i, q in enumerate(queries):
        approx = "~" if q["approx"] else ""
        print(f"{i+1:4d} {q['cnt']:>10,d}{approx:>6s} {q['score']:>8.2f} {q['description']}")

    return queries