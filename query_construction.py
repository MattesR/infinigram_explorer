"""
Construct CNF queries from SPLADE token clusters and optional syntactic links.

When syntactic links are provided (from spaCy), clusters that are linked
form multi-clause "units". Queries combine units, not individual clusters.

E.g.: spaCy: "civilized" modifies "community"
      SPLADE clusters: civilized|civilization, community|communities
      Unit: (civilized OR civilization) AND (community OR communities)
      Query: Unit AND (polite OR political)

Usage:
    from query_construction import build_cnf_queries, run_queries
"""

import numpy as np
from itertools import combinations
from wordpiece_cluster import cluster_tokens, clusters_to_or_clauses


def score_clusters(clusters, scored_tokens):
    """Assign each cluster a score from its tokens' combined scores."""
    for c in clusters:
        token_scores = [scored_tokens.get(t, 0.0) for t, s in c["tokens"]]
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
    syntactic_links: list[dict] = None,
):
    """
    Build ranked CNF queries from scored SPLADE tokens.

    If syntactic_links are provided, clusters linked by syntax form
    multi-clause units. Queries combine units rather than individual clusters.

    Args:
        top_tokens: List of (token, splade_score, tup, combined_score) tuples.
        tokenizer: Infini-gram tokenizer.
        min_stem_len: For WordPiece clustering.
        max_queries: Max CNF queries to return.
        min_cluster_score: Min combined score for non-anchor units.
        anchor_score: Min SPLADE score for anchor status.
        max_anchor_tup: Max TUP for anchor/informative tokens.
        strategy: "anchor", "anchor_plus_common", or "all_pairs".
        syntactic_links: From extract_syntactic_links(). If None, no linking.

    Returns:
        List of query dicts sorted by score.
    """
    scored_lookup = {t[0]: t[3] for t in top_tokens}
    splade_lookup = {t[0]: t[1] for t in top_tokens}
    tup_lookup = {t[0]: t[2] for t in top_tokens}

    # Split by TUP
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

    # Cluster informative tokens
    clusters = cluster_tokens(informative_tokens, min_stem_len=min_stem_len, show_steps=False)
    clusters = score_clusters(clusters, scored_lookup)

    # Mark anchor status on clusters
    for c in clusters:
        max_splade = max(splade_lookup.get(t, 0.0) for t, s in c["tokens"])
        min_tup = min(tup_lookup.get(t, 1.0) for t, s in c["tokens"])
        c["max_splade"] = max_splade
        c["min_tup"] = min_tup
        c["is_anchor"] = max_splade >= anchor_score and min_tup <= max_anchor_tup

    # Build query units
    if syntactic_links:
        from phrase_extraction import build_query_units
        units = build_query_units(syntactic_links, clusters, verbose=True)
    else:
        units = []
        for c in clusters:
            tokens_str = " OR ".join(t for t, s in c["tokens"])
            units.append({
                "clusters": [c],
                "type": "standalone",
                "description": f"({tokens_str})",
                "score": c.get("combined_score", 0),
            })
        units.sort(key=lambda x: x["score"], reverse=True)

    # Classify units
    anchor_units = []
    other_units = []
    for u in units:
        is_anchor = any(c.get("is_anchor", False) for c in u["clusters"])
        if is_anchor:
            anchor_units.append(u)
        elif u["score"] >= min_cluster_score:
            other_units.append(u)

    print(f"\nAnchor units ({len(anchor_units)}):")
    for u in anchor_units:
        print(f"  [{u['score']:6.2f}] {u['description']}")

    print(f"\nInformative units ({len(other_units)}, score >= {min_cluster_score}):")
    for u in other_units:
        print(f"  [{u['score']:6.2f}] {u['description']}")

    # Helper: convert a unit to CNF clauses
    def unit_to_cnf_clauses(unit):
        clauses = []
        for c in unit["clusters"]:
            or_ids = []
            for token, score in c["tokens"]:
                clean = token.lstrip("#")
                ids = tokenizer.encode(clean, add_special_tokens=False)
                if ids:
                    or_ids.append(ids)
            if or_ids:
                clauses.append(or_ids)
        return clauses

    # Common pool
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
            common_clause = {"clause": common_ids, "tokens": common_names}
            print(f"\nCommon pool: ({' OR '.join(common_names)})")

    # Build queries by combining units
    queries = []
    use_common = strategy == "anchor_plus_common" and common_clause is not None

    if strategy in ("anchor", "anchor_plus_common"):
        # Anchor AND informative
        for a in anchor_units:
            a_clauses = unit_to_cnf_clauses(a)
            for b in other_units:
                b_clauses = unit_to_cnf_clauses(b)
                cnf = a_clauses + b_clauses
                desc = f"{a['description']} AND {b['description']}"
                if use_common:
                    cnf.append(common_clause["clause"])
                    desc += f" AND ({' OR '.join(common_clause['tokens'])})"

                queries.append({
                    "cnf": cnf,
                    "description": desc,
                    "score": a["score"] * b["score"],
                    "units": [a, b],
                })

        # Anchor AND anchor
        for i, j in combinations(range(len(anchor_units)), 2):
            a, b = anchor_units[i], anchor_units[j]
            cnf = unit_to_cnf_clauses(a) + unit_to_cnf_clauses(b)
            desc = f"{a['description']} AND {b['description']}"
            if use_common:
                cnf.append(common_clause["clause"])
                desc += f" AND ({' OR '.join(common_clause['tokens'])})"

            queries.append({
                "cnf": cnf,
                "description": desc,
                "score": a["score"] * b["score"],
                "units": [a, b],
            })

    elif strategy == "all_pairs":
        all_units = anchor_units + other_units
        for i, j in combinations(range(len(all_units)), 2):
            a, b = all_units[i], all_units[j]
            cnf = unit_to_cnf_clauses(a) + unit_to_cnf_clauses(b)
            queries.append({
                "cnf": cnf,
                "description": f"{a['description']} AND {b['description']}",
                "score": a["score"] * b["score"],
                "units": [a, b],
            })

    # Sort and truncate
    queries.sort(key=lambda x: x["score"], reverse=True)
    queries = queries[:max_queries]

    print(f"\nTop {len(queries)} CNF queries:")
    for i, q in enumerate(queries):
        print(f"  {i+1:3d}. [{q['score']:8.2f}] {q['description']}")

    return queries


def run_queries(
    engine,
    queries,
    max_clause_freq=None,
    min_retrieved_docs: int = None,
    lower_query_bound: float = None,
):
    """
    Execute CNF queries, optionally stopping early.

    Args:
        engine: Infini-gram engine.
        queries: From build_cnf_queries().
        max_clause_freq: Passed to engine.find_cnf.
        min_retrieved_docs: Stop after accumulating this many pointers.
        lower_query_bound: Skip queries below this score.

    Returns:
        List of executed query dicts with cnt, approx, ptrs_by_shard.
    """
    from tqdm import tqdm
    import time

    if lower_query_bound is not None:
        before = len(queries)
        queries = [q for q in queries if q["score"] >= lower_query_bound]
        if len(queries) < before:
            print(f"  Skipping {before - len(queries)} queries below score {lower_query_bound}")

    print(f"Running {len(queries)} CNF queries...")
    total_ptrs = 0
    executed = []

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
            total_ptrs += sum(len(p) for p in q["ptrs_by_shard"])

        executed.append(q)

        if min_retrieved_docs and total_ptrs >= min_retrieved_docs:
            print(f"\n  Reached {total_ptrs} pointers after {len(executed)} queries, stopping.")
            break

    # Print summary
    executed.sort(key=lambda x: x.get("cnt", 0), reverse=True)
    print(f"\nResults (sorted by count):")
    print(f"{'#':>4s} {'Count':>10s} {'Approx':>6s} {'Score':>8s} {'Query'}")
    print("-" * 80)
    for i, q in enumerate(executed):
        approx = "~" if q.get("approx") else ""
        print(f"{i+1:4d} {q.get('cnt', 0):>10,d}{approx:>6s} {q['score']:>8.2f} {q['description']}")

    return executed