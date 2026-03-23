"""
Construct CNF queries by combining WordPiece clusters with
SPLADE × IDF scoring.

Pipeline:
1. Start with scored SPLADE tokens: (token, splade_score, tup, combined_score)
2. Cluster by WordPiece stems (OR groups)
3. Score each cluster by its best token's combined score
4. Form CNF queries: pairs of clusters (AND between clusters, OR within)
5. Score each pair and rank them

Usage:
    from query_construction import build_cnf_queries
    queries = build_cnf_queries(top_tokens, tokenizer_infini, clusters=None, min_stem_len=5)
"""

import numpy as np
from itertools import combinations
from wordpiece_cluster import cluster_tokens, clusters_to_or_clauses


def score_clusters(clusters, scored_tokens):
    """
    Assign each cluster a score based on its tokens' combined scores.

    Args:
        clusters: List of cluster dicts from cluster_tokens().
        scored_tokens: Dict mapping token_string -> (splade_score, tup, combined_score).

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
        # Cluster score = max combined score among its tokens
        c["combined_score"] = max(token_scores) if token_scores else 0.0
    return clusters


def build_cnf_queries(
    top_tokens: list[tuple],
    tokenizer,
    min_stem_len: int = 5,
    max_queries: int = 50,
    min_cluster_score: float = 0.0,
    strategy: str = "all_pairs",
):
    """
    Build ranked CNF queries from scored SPLADE tokens.

    Args:
        top_tokens: List of (token, splade_score, tup, combined_score) tuples.
        tokenizer: Infini-gram tokenizer for encoding to token IDs.
        min_stem_len: For WordPiece clustering.
        max_queries: Maximum number of CNF queries to return.
        min_cluster_score: Drop clusters below this combined score.
        strategy: How to form pairs:
            - "all_pairs": all cluster pairs, ranked by combined score
            - "anchor": highest-scoring cluster AND each other cluster

    Returns:
        List of query dicts sorted by score, each with:
            - 'cnf': ready-to-use CNF query for engine.find_cnf()
            - 'description': human-readable description
            - 'score': combined score of the pair
            - 'clusters': the two cluster dicts
    """
    # Build lookup: token -> combined_score
    scored_lookup = {t[0]: t[3] for t in top_tokens}

    # Cluster tokens (using just token, splade_score for clustering)
    splade_pairs = [(t[0], t[1]) for t in top_tokens]
    clusters = cluster_tokens(splade_pairs, min_stem_len=min_stem_len, show_steps=False)

    # Score clusters
    clusters = score_clusters(clusters, scored_lookup)

    # Filter by minimum score
    clusters = [c for c in clusters if c["combined_score"] >= min_cluster_score]
    clusters.sort(key=lambda x: x["combined_score"], reverse=True)

    print(f"\n{len(clusters)} clusters after filtering (min_score={min_cluster_score}):")
    for c in clusters:
        tokens_str = " | ".join(f"{t}" for t, s in c["tokens"])
        print(f"  [{c['combined_score']:6.2f}] {c['stem']:<15s} -> {tokens_str}")

    # Convert clusters to OR clauses with token IDs
    or_clauses = clusters_to_or_clauses(clusters, tokenizer)

    # Build CNF query pairs
    queries = []

    if strategy == "anchor":
        # Anchor = highest scoring cluster, AND with each other
        if len(or_clauses) < 2:
            print("Not enough clusters for anchor strategy")
            return []

        anchor = or_clauses[0]
        for other in or_clauses[1:]:
            cnf = [anchor["clause"], other["clause"]]
            pair_score = anchor["max_score"] + other["max_score"]
            # Boost: prefer pairs where both clusters are strong
            pair_score *= min(anchor["max_score"], other["max_score"])

            queries.append({
                "cnf": cnf,
                "description": f"({' OR '.join(anchor['tokens'])}) AND ({' OR '.join(other['tokens'])})",
                "score": pair_score,
                "clusters": [anchor, other],
            })

    elif strategy == "all_pairs":
        for i, j in combinations(range(len(or_clauses)), 2):
            a = or_clauses[i]
            b = or_clauses[j]
            cnf = [a["clause"], b["clause"]]

            # Score: product of combined scores (rewards pairs where both are strong)
            pair_score = a["max_score"] * b["max_score"]

            queries.append({
                "cnf": cnf,
                "description": f"({' OR '.join(a['tokens'])}) AND ({' OR '.join(b['tokens'])})",
                "score": pair_score,
                "clusters": [a, b],
            })

    # Sort by score descending
    queries.sort(key=lambda x: x["score"], reverse=True)

    # Truncate
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