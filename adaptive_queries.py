"""
Count-adaptive query construction.

Instead of statically building pairwise CNF queries, this module
peeks at index counts to decide query granularity dynamically:

- Low-count terms (< max_standalone): use as simple find() queries
- Medium-count: AND with spaCy context or cluster expansions
- High-count: AND with multiple clauses, expand ORs from clusters

Algorithm:
    For each anchor (splade >= threshold):
        1. Check standalone count → if low enough, direct find()
        2. Check with spaCy context (e.g. "civilized" AND "community")
        3. Try n-1 prefix + cluster OR expansions
        4. AND with other informative tokens if still too many

Usage:
    from adaptive_queries import build_adaptive_queries, run_adaptive
"""

import time
from tqdm import tqdm
from wordpiece_cluster import cluster_tokens, clusters_to_or_clauses
from phrase_extraction import extract_syntactic_links


def _encode(token: str, tokenizer) -> list[int]:
    """Encode a token/phrase to token IDs."""
    clean = token.lstrip("#")
    return tokenizer.encode(clean, add_special_tokens=False)


def _count(engine, input_ids: list[int]) -> tuple[int, bool]:
    """Get count for a token sequence. Returns (count, approx)."""
    result = engine.count(input_ids=input_ids)
    return result.get("count", 0), result.get("approx", False)


def _count_cnf(engine, cnf: list, max_clause_freq: int = None) -> tuple[int, bool]:
    """Get count for a CNF query. Returns (count, approx)."""
    kwargs = {"cnf": cnf}
    if max_clause_freq:
        kwargs["max_clause_freq"] = max_clause_freq
    result = engine.find_cnf(**kwargs)
    return result.get("cnt", 0), result.get("approx", False)


def _find_simple(engine, input_ids: list[int]):
    """Run a simple find() query, return result dict."""
    return engine.find(input_ids=input_ids)


def _find_cnf(engine, cnf: list, max_clause_freq: int = None):
    """Run a CNF query, return result dict."""
    kwargs = {"cnf": cnf}
    if max_clause_freq:
        kwargs["max_clause_freq"] = max_clause_freq
    return engine.find_cnf(**kwargs)


def build_adaptive_queries(
    top_tokens: list[tuple],
    tokenizer,
    engine,
    query_text: str = None,
    min_stem_len: int = 5,
    anchor_score: float = 0.9,
    max_anchor_tup: float = 1e-4,
    max_standalone: int = 10000,
    max_refined: int = 20000,
    min_cluster_score: float = 1.0,
    max_queries: int = 50,
    max_clause_freq: int = 100000,
    verbose: bool = True,
) -> list[dict]:
    """
    Build queries adaptively using index counts.

    Args:
        top_tokens: List of (token, splade_score, tup, combined_score).
        tokenizer: Infini-gram tokenizer.
        engine: Infini-gram engine (for count lookups).
        query_text: Original query (for spaCy extraction).
        min_stem_len: For WordPiece clustering.
        anchor_score: Min SPLADE score for anchors.
        max_anchor_tup: Max TUP for anchor/informative.
        max_standalone: Max count for standalone queries.
        max_refined: Max count before adding more AND clauses.
        min_cluster_score: Min combined score for informative clusters.
        max_queries: Max total queries to generate.
        max_clause_freq: For engine.find_cnf sampling.
        verbose: Print decisions.

    Returns:
        List of query dicts with 'type' ('simple' or 'cnf'),
        'input_ids' or 'cnf', 'description', 'score', 'estimated_count'.
    """
    scored_lookup = {t[0]: t[3] for t in top_tokens}
    splade_lookup = {t[0]: t[1] for t in top_tokens}
    tup_lookup = {t[0]: t[2] for t in top_tokens}

    # Split by TUP
    common_tokens = [(t[0], t[1]) for t in top_tokens if t[2] > max_anchor_tup]
    informative_tokens = [(t[0], t[1]) for t in top_tokens if t[2] <= max_anchor_tup]

    if verbose:
        print(f"\nToken split (max_anchor_tup={max_anchor_tup:.0e}):")
        print(f"  Informative: {len(informative_tokens)}, Common: {len(common_tokens)}")

    if len(informative_tokens) < 1:
        print("  No informative tokens")
        return []

    # Cluster informative tokens
    clusters = cluster_tokens(informative_tokens, min_stem_len=min_stem_len, show_steps=False)
    for c in clusters:
        token_scores = [scored_lookup.get(t, 0.0) for t, s in c["tokens"]]
        c["combined_score"] = max(token_scores) if token_scores else 0.0
        c["max_splade"] = max(splade_lookup.get(t, 0.0) for t, s in c["tokens"])
        c["is_anchor"] = c["max_splade"] >= anchor_score

    # Extract syntactic links
    syntactic_links = []
    if query_text:
        syntactic_links = extract_syntactic_links(query_text, verbose=verbose)

    # Build word -> cluster mapping
    word_to_cluster = {}
    for i, c in enumerate(clusters):
        for token, score in c["tokens"]:
            word_to_cluster[token.lower()] = i
        word_to_cluster[c["stem"].lower()] = i

    # Find syntactic partners for each cluster
    cluster_partners = {}  # cluster_idx -> list of partner cluster_idx
    for link in syntactic_links:
        indices = set()
        for word in link["words"]:
            if word in word_to_cluster:
                indices.add(word_to_cluster[word])
        if len(indices) >= 2:
            for idx in indices:
                partners = indices - {idx}
                cluster_partners.setdefault(idx, set()).update(partners)

    # Separate anchors and informative
    anchors = [c for c in clusters if c["is_anchor"]]
    others = [c for c in clusters if not c["is_anchor"] and c["combined_score"] >= min_cluster_score]

    anchors.sort(key=lambda x: x["max_splade"], reverse=True)
    others.sort(key=lambda x: x["combined_score"], reverse=True)

    if verbose:
        print(f"\nAnchors ({len(anchors)}):")
        for c in anchors:
            tokens_str = " | ".join(t for t, s in c["tokens"])
            print(f"  [{c['combined_score']:6.2f}] {c['stem']:<15s} -> {tokens_str}")
        print(f"\nInformative ({len(others)}):")
        for c in others:
            tokens_str = " | ".join(t for t, s in c["tokens"])
            print(f"  [{c['combined_score']:6.2f}] {c['stem']:<15s} -> {tokens_str}")

    queries = []
    covered_anchors = set()  # anchors that already have enough docs

    # Phase 1: Process each anchor adaptively
    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase 1: Adaptive anchor processing")
        print(f"{'='*60}")

    for anchor in anchors:
        anchor_idx = clusters.index(anchor)
        best_token = anchor["tokens"][0][0]  # highest-scoring token
        best_ids = _encode(best_token, tokenizer)

        if not best_ids:
            continue

        if verbose:
            print(f"\n  Anchor: '{best_token}' (splade={anchor['max_splade']:.2f})")

        # Step 1: Check standalone count
        count, approx = _count(engine, best_ids)
        if verbose:
            print(f"    Standalone '{best_token}': {count:,d} hits")

        if count <= max_standalone and count > 0:
            # Low enough — use as simple find query
            queries.append({
                "type": "simple",
                "input_ids": best_ids,
                "description": f"{best_token}",
                "score": anchor["combined_score"],
                "estimated_count": count,
            })
            covered_anchors.add(anchor_idx)
            if verbose:
                print(f"    -> DIRECT: {count:,d} docs")
            continue

        # Step 2: Try with spaCy partner (e.g. "civilized" AND "community")
        if anchor_idx in cluster_partners:
            for partner_idx in cluster_partners[anchor_idx]:
                partner = clusters[partner_idx]
                partner_token = partner["tokens"][0][0]
                partner_ids = _encode(partner_token, tokenizer)
                if not partner_ids:
                    continue

                cnf = [[best_ids], [partner_ids]]
                cnt, approx = _count_cnf(engine, cnf, max_clause_freq)
                if verbose:
                    print(f"    With spaCy partner '{partner_token}': {cnt:,d} hits")

                if cnt <= max_standalone and cnt > 0:
                    queries.append({
                        "type": "cnf",
                        "cnf": cnf,
                        "description": f"({best_token}) AND ({partner_token})",
                        "score": anchor["combined_score"],
                        "estimated_count": cnt,
                    })
                    covered_anchors.add(anchor_idx)
                    if verbose:
                        print(f"    -> ANCHOR+CONTEXT: {cnt:,d} docs")
                    break

            if anchor_idx in covered_anchors:
                continue

        # Step 3: Try with cluster OR expansions + spaCy partner
        # Build OR clause from anchor cluster
        anchor_or = []
        anchor_names = []
        for token, score in anchor["tokens"]:
            ids = _encode(token, tokenizer)
            if ids:
                anchor_or.append(ids)
                anchor_names.append(token)

        # Also try n-1 prefix: "civilized" -> check "civil"
        prefix = best_token[:-2] if len(best_token) > 4 else best_token
        prefix_ids = _encode(prefix, tokenizer)
        if prefix_ids and prefix_ids not in anchor_or:
            prefix_count, _ = _count(engine, prefix_ids)
            if verbose:
                print(f"    Prefix '{prefix}': {prefix_count:,d} hits")
            # Only add prefix if it's not too broad
            if prefix_count < max_standalone * 5:
                anchor_or.append(prefix_ids)
                anchor_names.append(prefix)

        # Try anchor OR clause with spaCy partner OR clause
        if anchor_idx in cluster_partners:
            for partner_idx in cluster_partners[anchor_idx]:
                partner = clusters[partner_idx]
                partner_or = []
                partner_names = []
                for token, score in partner["tokens"]:
                    ids = _encode(token, tokenizer)
                    if ids:
                        partner_or.append(ids)
                        partner_names.append(token)

                if not partner_or:
                    continue

                cnf = [anchor_or, partner_or]
                cnt, approx = _count_cnf(engine, cnf, max_clause_freq)
                if verbose:
                    a_str = " OR ".join(anchor_names)
                    p_str = " OR ".join(partner_names)
                    print(f"    ({a_str}) AND ({p_str}): {cnt:,d} hits")

                if cnt <= max_refined and cnt > 0:
                    a_str = " OR ".join(anchor_names)
                    p_str = " OR ".join(partner_names)
                    queries.append({
                        "type": "cnf",
                        "cnf": cnf,
                        "description": f"({a_str}) AND ({p_str})",
                        "score": anchor["combined_score"],
                        "estimated_count": cnt,
                    })
                    covered_anchors.add(anchor_idx)
                    if verbose:
                        print(f"    -> EXPANDED ANCHOR+CONTEXT: {cnt:,d} docs")
                    break

        if anchor_idx in covered_anchors:
            continue

        # Step 4: Anchor is still too broad — AND with informative tokens
        if verbose:
            print(f"    Still too broad, adding informative AND clauses...")

        for other in others:
            other_or = []
            other_names = []
            for token, score in other["tokens"]:
                ids = _encode(token, tokenizer)
                if ids:
                    other_or.append(ids)
                    other_names.append(token)
            if not other_or:
                continue

            cnf = [anchor_or, other_or]
            cnt, approx = _count_cnf(engine, cnf, max_clause_freq)

            if cnt > 0 and cnt <= max_refined:
                a_str = " OR ".join(anchor_names)
                o_str = " OR ".join(other_names)
                queries.append({
                    "type": "cnf",
                    "cnf": cnf,
                    "description": f"({a_str}) AND ({o_str})",
                    "score": anchor["combined_score"] * other["combined_score"],
                    "estimated_count": cnt,
                })
                if verbose:
                    print(f"      + ({o_str}): {cnt:,d} hits")

        covered_anchors.add(anchor_idx)

    # Phase 2: Cross-anchor combinations for anchors that are too broad
    # and informative-only combinations
    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase 2: Cross-anchor and informative combinations")
        print(f"{'='*60}")

    # Anchor AND anchor (if both are broad)
    from itertools import combinations
    for i, j in combinations(range(len(anchors)), 2):
        if len(queries) >= max_queries:
            break
        a, b = anchors[i], anchors[j]
        a_or = []
        a_names = []
        for token, score in a["tokens"]:
            ids = _encode(token, tokenizer)
            if ids:
                a_or.append(ids)
                a_names.append(token)
        b_or = []
        b_names = []
        for token, score in b["tokens"]:
            ids = _encode(token, tokenizer)
            if ids:
                b_or.append(ids)
                b_names.append(token)

        if not a_or or not b_or:
            continue

        cnf = [a_or, b_or]
        cnt, approx = _count_cnf(engine, cnf, max_clause_freq)

        if cnt > 0:
            a_str = " OR ".join(a_names)
            b_str = " OR ".join(b_names)
            queries.append({
                "type": "cnf",
                "cnf": cnf,
                "description": f"({a_str}) AND ({b_str})",
                "score": a["combined_score"] * b["combined_score"],
                "estimated_count": cnt,
            })
            if verbose:
                print(f"  ({a_str}) AND ({b_str}): {cnt:,d} hits")

    # Sort by score, truncate
    queries.sort(key=lambda x: x["score"], reverse=True)
    queries = queries[:max_queries]

    # Summary
    if verbose:
        total_estimated = sum(q["estimated_count"] for q in queries)
        print(f"\n{'='*60}")
        print(f"Summary: {len(queries)} queries, ~{total_estimated:,d} estimated total docs")
        print(f"{'='*60}")
        for i, q in enumerate(queries):
            marker = "FIND" if q["type"] == "simple" else "CNF "
            print(f"  {i+1:3d}. [{marker}] [{q['score']:8.2f}] "
                  f"~{q['estimated_count']:>8,d}  {q['description']}")

    return queries


def run_adaptive(
    engine,
    queries: list[dict],
    max_clause_freq: int = 100000,
    min_retrieved_docs: int = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Execute adaptive queries against the engine.

    Handles both 'simple' (find) and 'cnf' (find_cnf) query types.

    Args:
        engine: Infini-gram engine.
        queries: From build_adaptive_queries().
        max_clause_freq: For CNF sampling.
        min_retrieved_docs: Stop after enough pointers.
        verbose: Print progress.

    Returns:
        List of executed query dicts with results.
    """
    total_ptrs = 0
    executed = []

    for q in tqdm(queries, desc="Executing queries", disable=not verbose):
        t0 = time.perf_counter()

        if q["type"] == "simple":
            result = _find_simple(engine, q["input_ids"])
            q["cnt"] = result.get("cnt", 0)
            q["approx"] = result.get("approx", False)
            # Simple find returns segment_by_shard
            q["segment_by_shard"] = result.get("segment_by_shard", [])
            q["ptrs_by_shard"] = []  # not applicable
            ptrs = sum(
                (seg[1] - seg[0]) for seg in q["segment_by_shard"]
                if isinstance(seg, (list, tuple)) and len(seg) >= 2
            )
        else:
            kwargs = {"cnf": q["cnf"]}
            if max_clause_freq:
                kwargs["max_clause_freq"] = max_clause_freq
            result = engine.find_cnf(**kwargs)
            q["cnt"] = result.get("cnt", 0)
            q["approx"] = result.get("approx", False)
            q["ptrs_by_shard"] = result.get("ptrs_by_shard", [])
            q["segment_by_shard"] = []
            ptrs = sum(len(p) for p in q["ptrs_by_shard"])

        q["time"] = time.perf_counter() - t0
        total_ptrs += ptrs
        executed.append(q)

        if min_retrieved_docs and total_ptrs >= min_retrieved_docs:
            if verbose:
                print(f"\n  Reached {total_ptrs} pointers after {len(executed)} queries, stopping.")
            break

    # Summary
    if verbose:
        executed_sorted = sorted(executed, key=lambda x: x.get("cnt", 0), reverse=True)
        print(f"\nResults (sorted by count):")
        print(f"{'#':>4s} {'Type':<5s} {'Count':>10s} {'Approx':>6s} {'Score':>8s} {'Query'}")
        print("-" * 80)
        for i, q in enumerate(executed_sorted):
            approx = "~" if q.get("approx") else ""
            marker = "FIND" if q["type"] == "simple" else "CNF"
            print(f"{i+1:4d} {marker:<5s} {q.get('cnt', 0):>10,d}"
                  f"{approx:>6s} {q['score']:>8.2f} {q['description']}")

    return executed