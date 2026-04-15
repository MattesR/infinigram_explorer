"""
Count-adaptive query construction.

Peeks at index counts to decide query granularity dynamically:
- Low-count anchors: direct find() queries
- Medium-count: AND with spaCy context or cluster expansions
- High-count: AND with informative tokens

Usage:
    from adaptive_queries import build_adaptive_queries, run_adaptive
"""

import time
from itertools import combinations
from tqdm import tqdm
from wordpiece_cluster import cluster_tokens
from phrase_extraction import extract_syntactic_links


def _encode(token: str, tokenizer) -> list[int]:
    """Encode a token/phrase to token IDs."""
    return tokenizer.encode(token.lstrip("#"), add_special_tokens=False)


def _count(engine, input_ids: list[int]) -> int:
    """Get count for a token sequence."""
    return engine.count(input_ids=input_ids).get("count", 0)


def _count_cnf(engine, cnf: list, max_clause_freq: int = None) -> int:
    """Get count for a CNF query."""
    kwargs = {"cnf": cnf}
    if max_clause_freq:
        kwargs["max_clause_freq"] = max_clause_freq
    return engine.find_cnf(**kwargs).get("cnt", 0)


def _make_or_clause(cluster, tokenizer):
    """Build an OR clause (list of token ID lists) from a cluster."""
    or_ids = []
    names = []
    for token, score in cluster["tokens"]:
        ids = _encode(token, tokenizer)
        if ids:
            or_ids.append(ids)
            names.append(token)
    return or_ids, names


def _query_key(q):
    """Generate a hashable key for deduplication."""
    if q["type"] == "simple":
        return ("simple", tuple(q["input_ids"]))
    else:
        # Sort clauses to catch (A AND B) == (B AND A)
        cnf_key = tuple(
            tuple(tuple(alt) for alt in sorted(clause))
            for clause in sorted(q["cnf"], key=lambda c: str(c))
        )
        return ("cnf", cnf_key)


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
    max_clause_freq: int = 8000000,
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
        max_refined: Max count for refined AND queries.
        min_cluster_score: Min combined score for informative clusters.
        max_queries: Max total queries.
        max_clause_freq: For engine.find_cnf sampling.
        verbose: Print decisions.

    Returns:
        List of query dicts with 'type', 'input_ids'/'cnf',
        'description', 'score', 'estimated_count'.
    """
    scored_lookup = {t[0]: t[3] for t in top_tokens}
    splade_lookup = {t[0]: t[1] for t in top_tokens}

    # Split by TUP
    informative_tokens = [(t[0], t[1]) for t in top_tokens if t[2] <= max_anchor_tup]
    if verbose:
        print(f"\n  Informative tokens: {len(informative_tokens)}")

    if len(informative_tokens) < 1:
        return []

    # Cluster
    clusters = cluster_tokens(informative_tokens, min_stem_len=min_stem_len, show_steps=False)
    for c in clusters:
        c["combined_score"] = max(scored_lookup.get(t, 0) for t, s in c["tokens"])
        c["max_splade"] = max(splade_lookup.get(t, 0) for t, s in c["tokens"])
        c["is_anchor"] = c["max_splade"] >= anchor_score

    # Syntactic links
    syntactic_links = []
    if query_text:
        syntactic_links = extract_syntactic_links(query_text, verbose=verbose)

    # Build word -> cluster mapping
    word_to_cluster = {}
    for i, c in enumerate(clusters):
        for token, score in c["tokens"]:
            word_to_cluster[token.lower()] = i
        word_to_cluster[c["stem"].lower()] = i

    # Find syntactic partners
    cluster_partners = {}
    for link in syntactic_links:
        indices = set()
        for word in link["words"]:
            if word in word_to_cluster:
                indices.add(word_to_cluster[word])
        if len(indices) >= 2:
            for idx in indices:
                cluster_partners.setdefault(idx, set()).update(indices - {idx})

    # Separate anchors and informative
    anchors = sorted(
        [c for c in clusters if c["is_anchor"]],
        key=lambda x: x["max_splade"], reverse=True,
    )
    others = sorted(
        [c for c in clusters if not c["is_anchor"] and c["combined_score"] >= min_cluster_score],
        key=lambda x: x["combined_score"], reverse=True,
    )

    if verbose:
        print(f"\nAnchors ({len(anchors)}):")
        for c in anchors:
            print(f"  [{c['combined_score']:6.2f}] {c['stem']:<15s} -> {' | '.join(t for t, s in c['tokens'])}")
        print(f"\nInformative ({len(others)}):")
        for c in others:
            print(f"  [{c['combined_score']:6.2f}] {c['stem']:<15s} -> {' | '.join(t for t, s in c['tokens'])}")

    queries = []
    seen_keys = set()  # for deduplication

    def _add_query(q):
        """Add query if not a duplicate."""
        key = _query_key(q)
        if key not in seen_keys:
            seen_keys.add(key)
            queries.append(q)
            return True
        return False

    # Track which anchors got covered by spaCy links
    # so we don't generate redundant cross-anchor queries
    linked_pairs = set()

    # Phase 1: Process each anchor
    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase 1: Adaptive anchor processing")
        print(f"{'='*60}")

    for anchor in anchors:
        anchor_idx = clusters.index(anchor)
        best_token = anchor["tokens"][0][0]
        best_ids = _encode(best_token, tokenizer)
        if not best_ids:
            continue

        anchor_or, anchor_names = _make_or_clause(anchor, tokenizer)
        if not anchor_or:
            continue

        if verbose:
            print(f"\n  Anchor: '{best_token}' (splade={anchor['max_splade']:.2f})")

        # Step 1: Standalone count
        count = _count(engine, best_ids)
        if verbose:
            print(f"    Standalone '{best_token}': {count:,d} hits")

        if 0 < count <= max_standalone:
            _add_query({
                "type": "simple",
                "input_ids": best_ids,
                "description": best_token,
                "score": anchor["combined_score"],
                "estimated_count": count,
            })
            if verbose:
                print(f"    -> DIRECT: {count:,d} docs")
            continue

        # Step 2: Try with spaCy partner — directly with OR expansions
        found_partner = False
        if anchor_idx in cluster_partners:
            for partner_idx in cluster_partners[anchor_idx]:
                partner = clusters[partner_idx]
                partner_or, partner_names = _make_or_clause(partner, tokenizer)
                if not partner_or:
                    continue

                # Try expanded version directly
                cnf = [anchor_or, partner_or]
                cnt = _count_cnf(engine, cnf, max_clause_freq)

                a_str = " OR ".join(anchor_names)
                p_str = " OR ".join(partner_names)
                if verbose:
                    print(f"    ({a_str}) AND ({p_str}): {cnt:,d} hits")

                if 0 < cnt <= max_refined:
                    _add_query({
                        "type": "cnf",
                        "cnf": cnf,
                        "description": f"({a_str}) AND ({p_str})",
                        "score": anchor["combined_score"] + partner.get("combined_score", 0),
                        "estimated_count": cnt,
                    })
                    linked_pairs.add((min(anchor_idx, partner_idx), max(anchor_idx, partner_idx)))
                    found_partner = True
                    if verbose:
                        print(f"    -> ANCHOR+CONTEXT: {cnt:,d} docs")
                    break

        if found_partner:
            continue

        # Step 3: AND with informative tokens
        if verbose:
            print(f"    Too broad, adding informative AND clauses...")

        for other in others:
            if len(queries) >= max_queries:
                break
            other_or, other_names = _make_or_clause(other, tokenizer)
            if not other_or:
                continue

            cnf = [anchor_or, other_or]
            cnt = _count_cnf(engine, cnf, max_clause_freq)

            if 0 < cnt <= max_refined:
                a_str = " OR ".join(anchor_names)
                o_str = " OR ".join(other_names)
                _add_query({
                    "type": "cnf",
                    "cnf": cnf,
                    "description": f"({a_str}) AND ({o_str})",
                    "score": anchor["combined_score"] * other["combined_score"],
                    "estimated_count": cnt,
                })
                if verbose:
                    print(f"      + ({o_str}): {cnt:,d} hits")

    # Phase 2: Cross-anchor combinations (skip already-linked pairs)
    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase 2: Cross-anchor combinations")
        print(f"{'='*60}")

    for i, j in combinations(range(len(anchors)), 2):
        if len(queries) >= max_queries:
            break

        ai = clusters.index(anchors[i])
        aj = clusters.index(anchors[j])
        pair = (min(ai, aj), max(ai, aj))
        if pair in linked_pairs:
            if verbose:
                print(f"  Skipping {anchors[i]['stem']} x {anchors[j]['stem']} (already linked)")
            continue

        a_or, a_names = _make_or_clause(anchors[i], tokenizer)
        b_or, b_names = _make_or_clause(anchors[j], tokenizer)
        if not a_or or not b_or:
            continue

        cnf = [a_or, b_or]
        cnt = _count_cnf(engine, cnf, max_clause_freq)

        if 0 < cnt <= max_refined:
            a_str = " OR ".join(a_names)
            b_str = " OR ".join(b_names)
            _add_query({
                "type": "cnf",
                "cnf": cnf,
                "description": f"({a_str}) AND ({b_str})",
                "score": anchors[i]["combined_score"] * anchors[j]["combined_score"],
                "estimated_count": cnt,
            })
            if verbose:
                print(f"  ({a_str}) AND ({b_str}): {cnt:,d} hits")
        elif verbose and cnt > max_refined:
            a_str = " OR ".join(a_names)
            b_str = " OR ".join(b_names)
            print(f"  ({a_str}) AND ({b_str}): {cnt:,d} hits — too broad, skipping")

    # Sort and truncate
    queries.sort(key=lambda x: x["score"], reverse=True)
    queries = queries[:max_queries]

    # Summary
    if verbose:
        total_est = sum(q["estimated_count"] for q in queries)
        print(f"\n{'='*60}")
        print(f"Summary: {len(queries)} queries, ~{total_est:,d} estimated total docs")
        print(f"{'='*60}")
        for i, q in enumerate(queries):
            marker = "FIND" if q["type"] == "simple" else "CNF "
            print(f"  {i+1:3d}. [{marker}] [{q['score']:8.2f}] "
                  f"~{q['estimated_count']:>8,d}  {q['description']}")

    return queries


def run_adaptive(
    engine,
    queries: list[dict],
    max_clause_freq: int = 8000000,
    min_retrieved_docs: int = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Execute adaptive queries. Handles both 'simple' and 'cnf' types.
    """
    total_ptrs = 0
    executed = []

    for q in tqdm(queries, desc="Executing queries", disable=not verbose):
        t0 = time.perf_counter()

        if q["type"] == "simple":
            result = engine.find(input_ids=q["input_ids"])
            q["cnt"] = result.get("cnt", 0)
            q["approx"] = result.get("approx", False)
            q["segment_by_shard"] = result.get("segment_by_shard", [])
            q["ptrs_by_shard"] = []
            ptrs = sum(
                (seg[1] - seg[0]) for seg in q["segment_by_shard"]
                if isinstance(seg, (list, tuple)) and len(seg) >= 2
            )
        else:
            kwargs = {"cnf": q["cnf"]}
            if max_clause_freq:
                kwargs["max_clause_freq"] = max_clause_freq
            if q.get("max_diff_tokens"):
                kwargs["max_diff_tokens"] = q["max_diff_tokens"]
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
                print(f"\n  Reached {total_ptrs} pointers after {len(executed)} queries.")
            break

    # Summary
    if verbose:
        executed_sorted = sorted(executed, key=lambda x: x.get("cnt", 0), reverse=True)
        print(f"\nResults (sorted by count):")
        print(f"{'#':>4s} {'Type':<5s} {'Count':>10s} {'Score':>8s} {'Query'}")
        print("-" * 80)
        for i, q in enumerate(executed_sorted):
            approx = "~" if q.get("approx") else ""
            marker = "FIND" if q["type"] == "simple" else "CNF"
            print(f"{i+1:4d} {marker:<5s} {q.get('cnt', 0):>10,d}"
                  f"{approx:>1s} {q.get('score', 0):>8.2f} {q['description']}")

    return executed