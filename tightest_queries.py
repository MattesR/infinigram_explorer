"""
For each relevant document, find all maximal CNF queries that match.
Maximal = uses the most keyword terms (tightest AND combination).

Rules:
- At most one keyword per KEY aspect
- Optionally one ASSOCIATED term
- Cross-aspect AND only

Usage:
    from tightest_queries import find_tightest_per_doc

    results = find_tightest_per_doc(
        found_path="./inspection/prog/2024-32912_found.jsonl",
        qid="2024-32912",
        expansions_path="./kiss.jsonl",
        tokenizer=tokenizer,
        engine=engine,
        prox=50,
    )
"""

import json
import itertools
from collections import defaultdict
from progressive_queries import build_pieces


def _find_subsequence(tokens, pattern):
    """Find all start positions where pattern (list of ints) appears in tokens."""
    positions = []
    plen = len(pattern)
    if plen == 0:
        return positions
    for i in range(len(tokens) - plen + 1):
        if tokens[i:i + plen] == pattern:
            positions.append(i)
    return positions


def cnf_match_positions(cnf, tokens):
    """
    For each clause in cnf, find all token positions where any alternative matches.
    Returns list of position lists (one per clause), or None if any clause has no match.

    Each clause is a list of alternatives (each alternative is a list of token IDs).
    """
    clause_positions = []
    for clause in cnf:
        positions = []
        for alternative in clause:
            for pos in _find_subsequence(tokens, list(alternative)):
                if pos not in positions:
                    positions.append(pos)
        if not positions:
            return None  # clause failed
        clause_positions.append(sorted(positions))
    return clause_positions


def collect_matches(pieces_list, tokens):
    """Check which pieces match in the document. Returns matching pieces."""
    matches = []
    for piece in pieces_list:
        if cnf_match_positions(piece["cnf"], tokens) is not None:
            matches.append(piece)
    return matches


def group_by_aspect(matches):
    """Group matched pieces by their source_aspect."""
    grouped = defaultdict(list)
    for m in matches:
        grouped[m.get("source_aspect", "unknown")].append(m)
    return grouped


def generate_key_combinations(grouped_keys):
    """
    Generate all combinations picking one keyword per aspect group.
    Also includes combinations where some aspects are omitted.
    Each combo is a list of pieces.
    """
    groups = list(grouped_keys.values())
    # Each group: pick one OR pick none
    choices_per_group = [group + [None] for group in groups]
    all_combos = []
    for combo in itertools.product(*choices_per_group):
        selected = [c for c in combo if c is not None]
        if selected:
            all_combos.append(selected)
    return all_combos


def attach_associated(key_combos, assoc_matches):
    """Attach each associated match (or None) to each key combination."""
    results = []
    for kc in key_combos:
        # Without associated is also valid
        results.append((kc, None))
        for a in assoc_matches:
            results.append((kc, a))
    return results


def combo_score(combo):
    """Score a combination: more terms = tighter = better."""
    keys, assoc = combo
    return len(keys) + (1 if assoc else 0)


def filter_maximal(combinations):
    """Keep only combinations with the maximum number of terms."""
    if not combinations:
        return []
    max_score = max(combo_score(c) for c in combinations)
    return [c for c in combinations if combo_score(c) == max_score]


def find_best_combinations(pieces, tokens):
    """
    Find all maximal CNF query combinations that match a document's tokens.

    Args:
        pieces: Output from build_pieces with key_pieces and associated.
        tokens: List of token IDs for the document.

    Returns (maximal, key_matches, assoc_matches, grouped) where:
    - maximal: list of (keys, assoc) tuples, each maximal
    - key_matches: all matching key pieces
    - assoc_matches: all matching associated pieces
    - grouped: key_matches grouped by aspect
    """
    # Flatten key_pieces into list with source_aspect
    key_pieces_flat = []
    for aspect_name, piece_list in pieces["key_pieces"].items():
        for p in piece_list:
            p_copy = dict(p)
            p_copy["source_aspect"] = aspect_name
            key_pieces_flat.append(p_copy)

    assoc_pieces = pieces["associated"]

    key_matches = collect_matches(key_pieces_flat, tokens)
    assoc_matches = collect_matches(assoc_pieces, tokens)

    grouped = group_by_aspect(key_matches)
    key_combos = generate_key_combinations(grouped)
    all_combos = attach_associated(key_combos, assoc_matches)
    maximal = filter_maximal(all_combos)

    return maximal, key_matches, assoc_matches, grouped


def combo_to_cnf(combo):
    """Convert a (keys, assoc) combination to a flat CNF clause list."""
    keys, assoc = combo
    cnf = []
    for piece in keys:
        cnf.extend(piece["cnf"])
    if assoc:
        cnf.extend(assoc["cnf"])
    return cnf


def combo_to_desc(combo):
    """Human-readable description of a combination."""
    keys, assoc = combo
    parts = [p["description"] for p in keys]
    if assoc:
        parts.append(p["description"] for p in [assoc])
        parts = [p["description"] for p in keys] + [assoc["description"]]
    return " AND ".join(f"({p})" for p in parts)


def compute_min_span(cnf, tokens):
    """
    Compute minimum token span to cover at least one match from each clause.
    Returns span (int) or None if not all clauses match.
    """
    clause_positions = cnf_match_positions(cnf, tokens)
    if clause_positions is None:
        return None
    if len(clause_positions) == 1:
        return 0

    # Sliding window over all positions
    events = []
    for clause_idx, positions in enumerate(clause_positions):
        for pos in positions:
            events.append((pos, clause_idx))
    events.sort()

    n_clauses = len(clause_positions)
    clause_count = defaultdict(int)
    unique = 0
    left = 0
    min_span = float('inf')

    for right in range(len(events)):
        r_pos, r_clause = events[right]
        if clause_count[r_clause] == 0:
            unique += 1
        clause_count[r_clause] += 1

        while unique == n_clauses:
            l_pos, l_clause = events[left]
            span = r_pos - l_pos
            min_span = min(min_span, span)
            clause_count[l_clause] -= 1
            if clause_count[l_clause] == 0:
                unique -= 1
            left += 1

    return min_span if min_span < float('inf') else None


def find_tightest_per_doc(
    found_path: str,
    qid: str,
    expansions_path: str,
    tokenizer,
    engine,
    prox: int = 50,
    max_clause_freq: int = 80000000,
    verbose: bool = True,
):
    """
    For each relevant document, find all maximal CNF queries that match.
    Then compute engine counts and find minimum-cost cover.
    """
    # Load docs
    docs = []
    with open(found_path) as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj.get("text"):
                docs.append(obj)

    # Build pieces
    pieces = build_pieces(qid, expansions_path, tokenizer, engine, verbose=False)

    aspect_names = list(pieces["key_pieces"].keys())

    if verbose:
        print(f"\nFinding tightest queries for {len(docs)} docs, {qid}")
        print(f"  Aspects: {aspect_names}")
        n_key = sum(len(v) for v in pieces["key_pieces"].values())
        print(f"  Key pieces: {n_key}")
        print(f"  Associated pieces: {len(pieces['associated'])}")
        print(f"  Proximity: {prox}")

    # Process each document
    doc_results = []
    all_query_coverage = defaultdict(set)  # query_desc -> set of doc_ids
    single_term_docs = []

    for doc_idx, doc in enumerate(docs):
        tokens = list(tokenizer.encode(doc["text"], add_special_tokens=False))

        # Find all maximal combinations
        maximal, key_matches, assoc_matches, grouped = find_best_combinations(
            pieces, tokens)

        # Build query descriptions and compute spans
        tightest = []
        for combo in maximal:
            desc = combo_to_desc(combo)
            cnf = combo_to_cnf(combo)
            span = compute_min_span(cnf, tokens)
            keys, assoc = combo
            tightest.append({
                "desc": desc,
                "cnf": cnf,
                "span": span,
                "n_key_aspects": len(set(p["source_aspect"] for p in keys)),
                "has_assoc": assoc is not None,
                "score": combo_score(combo),
            })
            all_query_coverage[desc].add(doc["doc_id"])

        n_aspects = len(grouped)
        if n_aspects <= 1 and not assoc_matches:
            single_term_docs.append(doc["doc_id"])

        doc_results.append({
            "doc_id": doc["doc_id"],
            "relevance": doc.get("relevance", 1),
            "n_aspects": n_aspects,
            "n_key_matches": len(key_matches),
            "n_assoc_matches": len(assoc_matches),
            "tightest": tightest,
            "key_matches": [m["description"] for m in key_matches],
            "assoc_matches": [m["description"] for m in assoc_matches],
        })

        if verbose and (doc_idx + 1) % 50 == 0:
            print(f"    Processed {doc_idx + 1}/{len(docs)} docs...")

    # ================================================================
    # Get engine counts for all unique queries
    # ================================================================
    if verbose:
        print(f"\n  Computing engine counts for {len(all_query_coverage)} unique queries...")

    query_engine_counts = {}
    query_cnfs = {}

    # Collect unique CNFs
    for r in doc_results:
        for t in r["tightest"]:
            if t["desc"] not in query_cnfs:
                query_cnfs[t["desc"]] = t["cnf"]

    if verbose:
        print(f"\n  Computing engine counts for {len(query_cnfs)} unique queries...")

    from tqdm import tqdm
    for desc, cnf in tqdm(query_cnfs.items(), desc="Engine counts", disable=not verbose):
        try:
            kwargs = {"max_clause_freq": max_clause_freq}
            if len(cnf) > 1:
                kwargs["max_diff_tokens"] = prox
            result = engine.count_cnf(cnf, **kwargs)
            query_engine_counts[desc] = result.get("count", 0)
        except Exception:
            query_engine_counts[desc] = -1

    # ================================================================
    # Print analysis
    # ================================================================
    if verbose:
        print(f"\n{'='*70}")
        print(f"TIGHTEST QUERY ANALYSIS ({len(doc_results)} docs, prox={prox})")
        print(f"{'='*70}")

        # Aspect coverage
        aspect_dist = defaultdict(int)
        for r in doc_results:
            aspect_dist[r["n_aspects"]] += 1
        print(f"\n  Aspect coverage:")
        for n in sorted(aspect_dist.keys()):
            print(f"    {n} aspects: {aspect_dist[n]} docs")

        print(f"\n  Docs with only single-term reach: {len(single_term_docs)}")

        # All unique tightest queries sorted by coverage × precision
        print(f"\n  All tightest queries (sorted by coverage):")
        print(f"  {'Docs':>5s} {'Engine':>8s} {'Prec':>7s}  Query")
        sorted_queries = sorted(all_query_coverage.items(), key=lambda x: -len(x[1]))
        for desc, doc_ids in sorted_queries[:30]:
            eng = query_engine_counts.get(desc, 0)
            prec = len(doc_ids) / eng if eng > 0 else 0
            print(f"  {len(doc_ids):>5d} {eng:>8,d} {prec:>7.4f}  {desc}")

        # ================================================================
        # Minimum cost set cover
        # ================================================================
        print(f"\n{'='*70}")
        print(f"MINIMUM COST SET COVER (prox={prox})")
        print(f"{'='*70}")

        uncovered = set(r["doc_id"] for r in doc_results)
        selected = []
        total_engine = 0

        while uncovered:
            best_desc = None
            best_eff = -1
            for desc, doc_ids in all_query_coverage.items():
                new_covered = len(doc_ids & uncovered)
                if new_covered == 0:
                    continue
                eng = query_engine_counts.get(desc, 1)
                if eng <= 0:
                    eng = 1
                eff = new_covered / eng
                if eff > best_eff:
                    best_eff = eff
                    best_desc = desc

            if best_desc is None:
                break

            doc_ids = all_query_coverage[best_desc]
            new_covered = doc_ids & uncovered
            eng = query_engine_counts.get(best_desc, 0)
            uncovered -= new_covered
            total_engine += eng
            selected.append({
                "desc": best_desc,
                "new_covered": len(new_covered),
                "total_covered": len(doc_ids),
                "engine_count": eng,
            })

        total_covered = len(doc_results) - len(uncovered)
        print(f"\n  {len(selected)} queries cover {total_covered}/{len(doc_results)} docs")
        print(f"  Total engine count: {total_engine:,d}")
        print(f"\n  {'#':>3s} {'New':>5s} {'Tot':>5s} {'Engine':>8s} {'Eff':>8s}  Query")
        for i, s in enumerate(selected):
            eff = s["new_covered"] / s["engine_count"] if s["engine_count"] > 0 else 0
            print(f"  {i+1:>3d} {s['new_covered']:>5d} {s['total_covered']:>5d} "
                  f"{s['engine_count']:>8,d} {eff:>8.4f}  {s['desc']}")
        if uncovered:
            print(f"\n  Uncovered: {len(uncovered)} docs")
            for did in list(uncovered)[:5]:
                r = next(r for r in doc_results if r["doc_id"] == did)
                print(f"    {did}: {r['n_aspects']} aspects, "
                      f"keys={r['key_matches']}, assoc={r['assoc_matches']}")

    return {
        "doc_results": doc_results,
        "all_query_coverage": dict(all_query_coverage),
        "query_engine_counts": query_engine_counts,
        "single_term_docs": single_term_docs,
        "aspect_names": aspect_names,
    }