"""
Oracle analysis of CNF query construction.

Given relevant documents (with text) and keyword pieces, analyze:
1. Which keyword pieces match each document (token-level)
2. Token distance between matches (for proximity tuning)
3. Coverage by query type (cross-aspect combinations)
4. Marginal value of each query step

Usage:
    from oracle_analysis import analyze_query_coverage

    results = analyze_query_coverage(
        found_path="./inspection/prog/2024-32912_found.jsonl",
        qid="2024-32912",
        expansions_path="./kiss.jsonl",
        tokenizer=tokenizer,
        engine=engine,
    )
"""

import json
import numpy as np
from itertools import combinations, product
from collections import defaultdict
from llm_keyword_filter import STOPWORDS


def _case_variants(word, tokenizer, engine):
    """Get all valid case variant encodings for a word."""
    candidates = {word, word.lower()}
    if len(word) > 1:
        candidates.add(word[0].upper() + word[1:].lower())
    valid = []
    seen = set()
    for w in candidates:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if not ids:
            continue
        ids_key = tuple(ids)
        if ids_key in seen:
            continue
        count = engine.count(input_ids=ids).get("count", 0)
        if count > 0:
            seen.add(ids_key)
            valid.append(ids)
    return valid


def _build_piece_token_patterns(keyword, tokenizer, engine):
    """
    Build token patterns for a keyword piece.
    Returns list of (word_patterns, keyword_str) where word_patterns is
    a list of sets of token tuples (one set per content word, alternatives OR'd).
    """
    words = keyword.strip().split()
    content = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]
    if not content:
        return None

    word_patterns = []
    for w in content:
        variants = _case_variants(w, tokenizer, engine)
        if not variants:
            return None
        # Each variant is a list of token IDs
        word_patterns.append([tuple(v) for v in variants])

    return word_patterns


def _find_token_positions(doc_tokens, pattern):
    """
    Find all positions where a token pattern (sequence of IDs) matches
    in a document's token array.

    Returns list of start positions.
    """
    doc_tokens = list(doc_tokens)
    pattern = list(pattern)
    positions = []
    for i in range(len(doc_tokens) - len(pattern) + 1):
        if doc_tokens[i:i+len(pattern)] == pattern:
            positions.append(i)
    return positions


def _find_word_positions(doc_tokens, word_variants):
    """
    Find all positions of any variant of a word in the document.
    Returns list of (start_pos, end_pos) tuples.
    """
    positions = []
    for variant in word_variants:
        for pos in _find_token_positions(doc_tokens, variant):
            positions.append((pos, pos + len(variant)))
    return sorted(positions)


def _compute_min_span(word_positions_list):
    """
    Given positions for multiple words, compute the minimum span
    (max_diff_tokens) that would cover at least one match of each word.

    word_positions_list: list of lists of (start, end) per word

    Returns min span (tokens between first and last match) or None if
    any word has no matches.
    """
    if not word_positions_list:
        return None

    # Check all words have matches
    for positions in word_positions_list:
        if not positions:
            return None

    if len(word_positions_list) == 1:
        return 0  # single word, no span needed

    # Try all combinations of one position per word, find minimum span
    # For efficiency, use a sliding window approach
    # But with small number of words, brute force is fine
    min_span = float('inf')

    # Get flat list of (position, word_index)
    events = []
    for word_idx, positions in enumerate(word_positions_list):
        for start, end in positions:
            events.append((start, end, word_idx))
    events.sort()

    n_words = len(word_positions_list)

    # Sliding window: find smallest window containing all words
    from collections import Counter
    word_count = Counter()
    unique_words = 0
    left = 0

    for right in range(len(events)):
        r_start, r_end, r_word = events[right]
        if word_count[r_word] == 0:
            unique_words += 1
        word_count[r_word] += 1

        while unique_words == n_words:
            l_start, l_end, l_word = events[left]
            span = r_end - l_start
            min_span = min(min_span, span)

            word_count[l_word] -= 1
            if word_count[l_word] == 0:
                unique_words -= 1
            left += 1

    return min_span if min_span < float('inf') else None


def analyze_query_coverage(
    found_path: str,
    qid: str,
    expansions_path: str,
    tokenizer,
    engine,
    verbose: bool = True,
):
    """
    Analyze keyword coverage and proximity in relevant documents.

    Returns dict with coverage stats, proximity distributions, and
    per-query-type analysis.
    """
    from llm_keyword_filter import load_all_expansions

    # Load found docs
    docs = []
    with open(found_path) as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj.get("text"):
                docs.append(obj)

    if verbose:
        print(f"Loaded {len(docs)} found documents for {qid}")

    # Load keywords
    data = load_all_expansions(expansions_path).get(qid, {})
    key_entities = data.get("KEY_ENTITIES", {})
    associated = data.get("ASSOCIATED_TERMS", data.get("ASSOCIATED", []))

    # Build token patterns for all keywords
    key_patterns = {}  # aspect_name -> [(word_patterns, keyword_str), ...]
    assoc_patterns = []  # [(word_patterns, keyword_str), ...]

    for aspect_name, terms in key_entities.items():
        if isinstance(terms, dict):
            flat = []
            for level in ["lexical", "conceptual", "referential"]:
                flat.extend(terms.get(level, []))
            terms = flat
        patterns = []
        for kw in [aspect_name] + (terms if isinstance(terms, list) else []):
            wp = _build_piece_token_patterns(kw, tokenizer, engine)
            if wp:
                patterns.append((wp, kw))
        key_patterns[aspect_name] = patterns

    for kw in associated:
        wp = _build_piece_token_patterns(kw, tokenizer, engine)
        if wp:
            assoc_patterns.append((wp, kw))

    if verbose:
        for name, patterns in key_patterns.items():
            print(f"  KEY '{name}': {len(patterns)} keyword patterns")
        print(f"  ASSOCIATED: {len(assoc_patterns)} keyword patterns")

    # Tokenize documents
    if verbose:
        print(f"\nTokenizing {len(docs)} documents...")

    doc_analyses = []
    for doc in docs:
        text = doc["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)

        # Check each keyword against this doc's tokens
        keyword_matches = {}  # keyword_str -> {positions, min_span}
        aspect_matches = {}   # aspect_name -> [keyword_str, ...]

        # KEY keywords
        for aspect_name, patterns in key_patterns.items():
            aspect_matches[aspect_name] = []
            for word_patterns, kw in patterns:
                # Find positions for each content word
                word_positions = []
                for word_variants in word_patterns:
                    positions = _find_word_positions(tokens, word_variants)
                    word_positions.append(positions)

                # Check if all words are present
                all_present = all(len(p) > 0 for p in word_positions)
                if all_present:
                    min_span = _compute_min_span(word_positions)
                    keyword_matches[kw] = {
                        "aspect": aspect_name,
                        "category": "key",
                        "min_span": min_span,
                        "n_words": len(word_patterns),
                    }
                    aspect_matches[aspect_name].append(kw)

        # ASSOCIATED keywords
        for word_patterns, kw in assoc_patterns:
            word_positions = []
            for word_variants in word_patterns:
                positions = _find_word_positions(tokens, word_variants)
                word_positions.append(positions)

            all_present = all(len(p) > 0 for p in word_positions)
            if all_present:
                min_span = _compute_min_span(word_positions)
                keyword_matches[kw] = {
                    "category": "assoc",
                    "min_span": min_span,
                    "n_words": len(word_patterns),
                }

        # Cross-aspect proximity: for each pair of aspects,
        # what's the minimum distance between any keyword from each?
        cross_aspect_prox = {}
        aspect_names = list(key_patterns.keys())
        for a, b in combinations(aspect_names, 2):
            if aspect_matches[a] and aspect_matches[b]:
                # Find min distance between any keyword from a and any from b
                min_dist = float('inf')
                for kw_a in aspect_matches[a]:
                    for kw_b in aspect_matches[b]:
                        # Get positions of all words in both keywords
                        all_positions_a = []
                        for word_patterns, kw_check in key_patterns[a]:
                            if kw_check == kw_a:
                                for wv in word_patterns:
                                    all_positions_a.extend(
                                        _find_word_positions(tokens, wv))
                        all_positions_b = []
                        for word_patterns, kw_check in key_patterns[b]:
                            if kw_check == kw_b:
                                for wv in word_patterns:
                                    all_positions_b.extend(
                                        _find_word_positions(tokens, wv))

                        if all_positions_a and all_positions_b:
                            for pa_start, pa_end in all_positions_a:
                                for pb_start, pb_end in all_positions_b:
                                    dist = abs(pa_start - pb_start)
                                    min_dist = min(min_dist, dist)

                if min_dist < float('inf'):
                    cross_aspect_prox[(a, b)] = min_dist

        doc_analyses.append({
            "doc_id": doc["doc_id"],
            "relevance": doc.get("relevance", 1),
            "n_tokens": len(tokens),
            "keyword_matches": keyword_matches,
            "aspect_matches": aspect_matches,
            "n_aspects_covered": sum(1 for v in aspect_matches.values() if v),
            "cross_aspect_prox": cross_aspect_prox,
        })

    # ================================================================
    # Aggregate statistics
    # ================================================================
    n_docs = len(doc_analyses)
    aspect_names = list(key_patterns.keys())
    n_aspects = len(aspect_names)

    if verbose:
        print(f"\n{'='*70}")
        print(f"COVERAGE ANALYSIS ({n_docs} relevant docs)")
        print(f"{'='*70}")

        # Aspect coverage distribution
        coverage_dist = defaultdict(int)
        for da in doc_analyses:
            coverage_dist[da["n_aspects_covered"]] += 1
        print(f"\n  Aspect coverage distribution:")
        for n in range(n_aspects + 1):
            count = coverage_dist[n]
            print(f"    {n}/{n_aspects} aspects: {count} docs ({count/n_docs*100:.1f}%)")

        # Per-aspect coverage
        print(f"\n  Per-aspect coverage:")
        for name in aspect_names:
            n_covered = sum(1 for da in doc_analyses if da["aspect_matches"][name])
            print(f"    {name}: {n_covered}/{n_docs} ({n_covered/n_docs*100:.1f}%)")

        # Per-keyword coverage
        print(f"\n  Top keywords by coverage:")
        kw_coverage = defaultdict(int)
        for da in doc_analyses:
            for kw in da["keyword_matches"]:
                kw_coverage[kw] += 1
        for kw, count in sorted(kw_coverage.items(), key=lambda x: -x[1])[:20]:
            info = None
            for da in doc_analyses:
                if kw in da["keyword_matches"]:
                    info = da["keyword_matches"][kw]
                    break
            cat = info.get("category", "?") if info else "?"
            print(f"    {count:>5d} ({count/n_docs*100:5.1f}%)  [{cat}] {kw}")

        # Proximity analysis
        print(f"\n  Cross-aspect proximity (min token distance):")
        for a, b in combinations(aspect_names, 2):
            distances = [da["cross_aspect_prox"].get((a, b), None)
                        for da in doc_analyses]
            distances = [d for d in distances if d is not None]
            if distances:
                print(f"    {a} <-> {b}:")
                print(f"      docs with both: {len(distances)}/{n_docs}")
                print(f"      min: {min(distances)}, median: {sorted(distances)[len(distances)//2]}, "
                      f"max: {max(distances)}")
                # What prox would catch X% of docs?
                for pct in [50, 75, 90, 95, 99]:
                    idx = min(int(len(distances) * pct / 100), len(distances) - 1)
                    print(f"      prox for {pct}%: {sorted(distances)[idx]}")

        # Within-keyword proximity (for multi-word keywords)
        print(f"\n  Within-keyword proximity (multi-word keywords):")
        all_spans = defaultdict(list)
        for da in doc_analyses:
            for kw, info in da["keyword_matches"].items():
                if info["n_words"] > 1 and info["min_span"] is not None:
                    all_spans[kw].append(info["min_span"])

        for kw, spans in sorted(all_spans.items(), key=lambda x: -len(x[1]))[:15]:
            if spans:
                print(f"    {kw}: median={sorted(spans)[len(spans)//2]}, "
                      f"max={max(spans)}, docs={len(spans)}")

        # Query type coverage simulation
        print(f"\n{'='*70}")
        print(f"QUERY TYPE COVERAGE SIMULATION")
        print(f"{'='*70}")

        # What would each query type cover?
        # Type 1: All aspects AND'd
        covered_all = sum(1 for da in doc_analyses
                         if da["n_aspects_covered"] == n_aspects)
        print(f"\n  All {n_aspects} aspects AND'd: {covered_all}/{n_docs} "
              f"({covered_all/n_docs*100:.1f}%)")

        # Type 2: Pairwise
        for a, b in combinations(aspect_names, 2):
            covered = sum(1 for da in doc_analyses
                         if da["aspect_matches"][a] and da["aspect_matches"][b])
            print(f"  ({a}) AND ({b}): {covered}/{n_docs} ({covered/n_docs*100:.1f}%)")

        # Type 3: Single aspect + any associated
        for name in aspect_names:
            covered_with_assoc = sum(
                1 for da in doc_analyses
                if da["aspect_matches"][name] and
                any(info.get("category") == "assoc" for info in da["keyword_matches"].values())
            )
            print(f"  ({name}) AND any_assoc: {covered_with_assoc}/{n_docs} "
                  f"({covered_with_assoc/n_docs*100:.1f}%)")

        # Unreachable docs
        no_keywords = sum(1 for da in doc_analyses if not da["keyword_matches"])
        print(f"\n  Docs with ZERO keyword matches: {no_keywords}/{n_docs}")

        # Docs reachable only by specific types
        only_single_aspect = sum(
            1 for da in doc_analyses
            if da["n_aspects_covered"] == 1 and da["keyword_matches"]
        )
        print(f"  Docs with only 1 aspect: {only_single_aspect}/{n_docs}")

    return {
        "doc_analyses": doc_analyses,
        "key_patterns": key_patterns,
        "assoc_patterns": assoc_patterns,
        "aspect_names": aspect_names,
    }


def find_optimal_queries(
    analysis_results: dict,
    engine,
    tokenizer,
    prox_values: list = None,
    max_clause_freq: int = 80000000,
    verbose: bool = True,
):
    """
    Find the tightest and fewest queries to cover all relevant documents.

    For each candidate query (cross-aspect AND, aspect+assoc AND):
    1. Check which relevant docs it covers at various proximity values
    2. Execute against engine to get total count (relevant + irrelevant)
    3. Compute precision = relevant_covered / total_count

    Then run greedy set cover to find minimum query set.

    Args:
        analysis_results: Output from analyze_query_coverage.
        prox_values: List of proximity values to test. Default [10, 20, 50, 100, 200].
    """
    if prox_values is None:
        prox_values = [10, 20, 50, 100, 200]

    doc_analyses = analysis_results["doc_analyses"]
    key_patterns = analysis_results["key_patterns"]
    assoc_patterns = analysis_results["assoc_patterns"]
    aspect_names = analysis_results["aspect_names"]

    n_docs = len(doc_analyses)
    all_doc_ids = {da["doc_id"] for da in doc_analyses}

    # ================================================================
    # Build candidate queries and check coverage per doc
    # ================================================================

    candidates = []  # list of {desc, cnf, covered_docs_by_prox, engine_counts}

    # Helper: build CNF from keyword patterns
    def _patterns_to_cnf(patterns_list):
        """Convert list of (word_patterns, keyword) to flat CNF clauses."""
        cnf = []
        for word_patterns, kw in patterns_list:
            for word_variants in word_patterns:
                cnf.append(word_variants)
        return cnf

    def _check_doc_coverage(doc_analysis, required_keywords, prox_val):
        """
        Check if a doc is covered by a query requiring specific keywords
        at a given proximity.

        required_keywords: list of keyword strings that must all match
        prox_val: max token span for cross-keyword proximity
        """
        matches = doc_analysis["keyword_matches"]

        # Check all keywords present
        for kw in required_keywords:
            if kw not in matches:
                return False

        # Check proximity: all matched keywords within prox_val tokens
        # Use cross_aspect_prox or within-keyword spans
        if len(required_keywords) <= 1:
            return True

        # Get all matched positions across required keywords
        # We need the raw positions — but we only stored min_span
        # For now, use a heuristic: if the keyword's min_span < prox_val, it's reachable
        # For cross-keyword, check cross_aspect_prox
        # This is approximate — full position check would need re-tokenization

        # Simple check: are all keywords from aspects that have cross-prox < prox_val?
        aspects_involved = set()
        for kw in required_keywords:
            info = matches[kw]
            if "aspect" in info:
                aspects_involved.add(info["aspect"])

        cross_prox = doc_analysis.get("cross_aspect_prox", {})
        for a, b in combinations(aspects_involved, 2):
            key = (a, b) if (a, b) in cross_prox else (b, a)
            dist = cross_prox.get(key, float('inf'))
            if dist > prox_val:
                return False

        return True

    if verbose:
        print(f"\n{'='*70}")
        print(f"OPTIMAL QUERY ANALYSIS")
        print(f"{'='*70}")

    # Candidate type 1: Pairwise aspect names
    for a, b in combinations(aspect_names, 2):
        kw_a = a  # aspect name as keyword
        kw_b = b
        desc = f"({a}) AND ({b})"

        covered_by_prox = {}
        for prox in prox_values:
            covered = set()
            for da in doc_analyses:
                if kw_a in da["keyword_matches"] and kw_b in da["keyword_matches"]:
                    # Check cross-aspect proximity
                    cross_prox = da.get("cross_aspect_prox", {})
                    key = (a, b) if (a, b) in cross_prox else (b, a)
                    dist = cross_prox.get(key, float('inf'))
                    if dist <= prox:
                        covered.add(da["doc_id"])
            covered_by_prox[prox] = covered

        # Get engine count at each prox
        # Build CNF from aspect name patterns
        cnf_clauses = []
        for name in [a, b]:
            for word_patterns, kw in key_patterns[name]:
                if kw.lower() == name.lower():
                    for word_variants in word_patterns:
                        cnf_clauses.append(word_variants)
                    break

        engine_counts = {}
        for prox in prox_values:
            try:
                result = engine.count_cnf(
                    cnf_clauses, max_clause_freq=max_clause_freq,
                    max_diff_tokens=prox)
                engine_counts[prox] = result.get("count", 0)
            except Exception:
                engine_counts[prox] = 0

        candidates.append({
            "desc": desc,
            "type": "pairwise",
            "cnf": cnf_clauses,
            "covered_by_prox": covered_by_prox,
            "engine_counts": engine_counts,
        })

    # Candidate type 2: All aspects AND'd
    if len(aspect_names) >= 2:
        desc = " AND ".join(f"({n})" for n in aspect_names)
        covered_by_prox = {}
        for prox in prox_values:
            covered = set()
            for da in doc_analyses:
                if da["n_aspects_covered"] == len(aspect_names):
                    # Check all cross-aspect proximities
                    all_close = True
                    cross_prox = da.get("cross_aspect_prox", {})
                    for a2, b2 in combinations(aspect_names, 2):
                        key = (a2, b2) if (a2, b2) in cross_prox else (b2, a2)
                        dist = cross_prox.get(key, float('inf'))
                        if dist > prox:
                            all_close = False
                            break
                    if all_close:
                        covered.add(da["doc_id"])
            covered_by_prox[prox] = covered

        cnf_clauses = []
        for name in aspect_names:
            for word_patterns, kw in key_patterns[name]:
                if kw.lower() == name.lower():
                    for word_variants in word_patterns:
                        cnf_clauses.append(word_variants)
                    break

        engine_counts = {}
        for prox in prox_values:
            try:
                result = engine.count_cnf(
                    cnf_clauses, max_clause_freq=max_clause_freq,
                    max_diff_tokens=prox)
                engine_counts[prox] = result.get("count", 0)
            except Exception:
                engine_counts[prox] = 0

        candidates.append({
            "desc": desc,
            "type": "all_aspects",
            "cnf": cnf_clauses,
            "covered_by_prox": covered_by_prox,
            "engine_counts": engine_counts,
        })

    # Candidate type 3: Aspect name AND associated term
    for name in aspect_names:
        for assoc_wp, assoc_kw in assoc_patterns:
            desc = f"({name}) AND ({assoc_kw})"
            covered_by_prox = {}
            for prox in prox_values:
                covered = set()
                for da in doc_analyses:
                    if name in [info.get("aspect") for info in da["keyword_matches"].values()
                                if info.get("aspect")]:
                        # Check if aspect name keyword matches
                        if name.lower() not in [kw.lower() for kw in da["keyword_matches"]]:
                            # Try any keyword from this aspect
                            if not da["aspect_matches"].get(name):
                                continue
                        if assoc_kw in da["keyword_matches"]:
                            covered.add(da["doc_id"])
                covered_by_prox[prox] = covered

            # Build CNF
            cnf_clauses = []
            for word_patterns, kw in key_patterns[name]:
                if kw.lower() == name.lower():
                    for word_variants in word_patterns:
                        cnf_clauses.append(word_variants)
                    break
            for word_variants in assoc_wp:
                cnf_clauses.append(word_variants)

            engine_counts = {}
            for prox in prox_values:
                try:
                    result = engine.count_cnf(
                        cnf_clauses, max_clause_freq=max_clause_freq,
                        max_diff_tokens=prox)
                    engine_counts[prox] = result.get("count", 0)
                except Exception:
                    engine_counts[prox] = 0

            candidates.append({
                "desc": desc,
                "type": "aspect_assoc",
                "cnf": cnf_clauses,
                "covered_by_prox": covered_by_prox,
                "engine_counts": engine_counts,
            })

    # Candidate type 4: Single keyword (for grabbed terms)
    for name in aspect_names:
        for word_patterns, kw in key_patterns[name]:
            covered = {da["doc_id"] for da in doc_analyses if kw in da["keyword_matches"]}
            if not covered:
                continue
            cnf_clauses = [wv for wv in word_patterns]
            engine_counts = {}
            for prox in prox_values:
                try:
                    kwargs = {"max_clause_freq": max_clause_freq}
                    if len(cnf_clauses) > 1:
                        kwargs["max_diff_tokens"] = prox
                    result = engine.count_cnf(cnf_clauses, **kwargs)
                    engine_counts[prox] = result.get("count", 0)
                except Exception:
                    engine_counts[prox] = 0
            candidates.append({
                "desc": f"{kw} (standalone)",
                "type": "standalone_key",
                "cnf": cnf_clauses,
                "covered_by_prox": {p: covered for p in prox_values},
                "engine_counts": engine_counts,
            })

    for assoc_wp, assoc_kw in assoc_patterns:
        covered = {da["doc_id"] for da in doc_analyses if assoc_kw in da["keyword_matches"]}
        if not covered:
            continue
        cnf_clauses = [wv for wv in assoc_wp]
        engine_counts = {}
        for prox in prox_values:
            try:
                kwargs = {"max_clause_freq": max_clause_freq}
                if len(cnf_clauses) > 1:
                    kwargs["max_diff_tokens"] = prox
                result = engine.count_cnf(cnf_clauses, **kwargs)
                engine_counts[prox] = result.get("count", 0)
            except Exception:
                engine_counts[prox] = 0
        candidates.append({
            "desc": f"{assoc_kw} (standalone)",
            "type": "standalone_assoc",
            "cnf": cnf_clauses,
            "covered_by_prox": {p: covered for p in prox_values},
            "engine_counts": engine_counts,
        })

    # ================================================================
    # Print query analysis
    # ================================================================
    if verbose:
        # Pick a representative prox for display
        display_prox = 50

        print(f"\n  Candidate queries at prox={display_prox}:")
        print(f"  {'Type':<15s} {'Covered':>8s} {'Engine':>8s} {'Prec':>7s}  Query")
        print(f"  {'-'*70}")

        sorted_cands = sorted(candidates,
                              key=lambda c: len(c["covered_by_prox"].get(display_prox, set())),
                              reverse=True)

        for c in sorted_cands:
            covered = len(c["covered_by_prox"].get(display_prox, set()))
            eng_count = c["engine_counts"].get(display_prox, 0)
            prec = covered / eng_count if eng_count > 0 else 0
            if covered > 0:
                print(f"  {c['type']:<15s} {covered:>7d} {eng_count:>8d} {prec:>7.4f}  {c['desc']}")

        # Proximity sweep for top queries
        print(f"\n  Proximity sweep for top queries:")
        for c in sorted_cands[:5]:
            print(f"\n    {c['desc']}:")
            print(f"    {'prox':>6s} {'covered':>8s} {'engine':>8s} {'prec':>7s} {'recall':>7s}")
            for prox in prox_values:
                covered = len(c["covered_by_prox"].get(prox, set()))
                eng = c["engine_counts"].get(prox, 0)
                prec = covered / eng if eng > 0 else 0
                rec = covered / n_docs if n_docs > 0 else 0
                print(f"    {prox:>6d} {covered:>8d} {eng:>8d} {prec:>7.4f} {rec:>7.4f}")

    # ================================================================
    # Greedy set cover — find minimum queries to cover all docs
    # ================================================================
    if verbose:
        print(f"\n{'='*70}")
        print(f"GREEDY SET COVER")
        print(f"{'='*70}")

    for target_prox in [50, 100, 200]:
        uncovered = set(all_doc_ids)
        selected = []
        total_engine_count = 0

        while uncovered:
            # Find candidate covering most uncovered docs
            best = None
            best_new = 0
            for c in candidates:
                covered = c["covered_by_prox"].get(target_prox, set())
                new_covered = len(covered & uncovered)
                if new_covered > best_new:
                    best_new = new_covered
                    best = c
            if best is None or best_new == 0:
                break

            covered = best["covered_by_prox"][target_prox]
            eng_count = best["engine_counts"].get(target_prox, 0)
            uncovered -= covered
            total_engine_count += eng_count
            selected.append({
                "desc": best["desc"],
                "type": best["type"],
                "new_covered": best_new,
                "total_covered": len(covered),
                "engine_count": eng_count,
            })

        total_covered = n_docs - len(uncovered)

        if verbose:
            print(f"\n  prox={target_prox}: {len(selected)} queries cover "
                  f"{total_covered}/{n_docs} docs, ~{total_engine_count:,d} total engine hits")
            print(f"  {'#':>3s} {'Type':<15s} {'New':>5s} {'Tot':>5s} {'Engine':>8s}  Query")
            for i, s in enumerate(selected):
                print(f"  {i+1:>3d} {s['type']:<15s} {s['new_covered']:>5d} "
                      f"{s['total_covered']:>5d} {s['engine_count']:>8d}  {s['desc']}")
            print(f"  Uncovered: {len(uncovered)} docs")

    return {
        "candidates": candidates,
        "n_docs": n_docs,
    }