"""
Progressive CNF query construction.

Step 1: build_pieces — create CNF pieces from keywords
Step 2: peek_and_grab — peek counts, grab cheap ones
Step 3: build_combination_queries — combine remaining into cross-aspect queries

Usage:
    from progressive_queries import build_pieces, peek_and_grab, build_combination_queries

    pieces = build_pieces(qid, path, tokenizer, engine)
    peek = peek_and_grab(pieces, engine, tokenizer)
    combo = build_combination_queries(peek, engine, tokenizer)
    all_queries = peek["grabbed"] + combo
"""

from itertools import combinations
from llm_keyword_filter import load_all_expansions, STOPWORDS


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


def _make_term_piece(keyword, tokenizer, engine):
    """
    Make a CNF piece from a single keyword phrase.
    Multi-word: one clause per word (for proximity AND).
    Single word: single clause with case variants.
    Stopwords removed.
    """
    words = keyword.strip().split()
    content = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]
    if not content:
        return None
    # Skip multi-word that reduced to single word
    if len(words) > 1 and len(content) == 1:
        return None

    clauses = []
    for w in content:
        variants = _case_variants(w, tokenizer, engine)
        if not variants:
            return None
        clauses.append(variants)

    return {
        "cnf": clauses,
        "description": keyword,
        "n_words": len(clauses),
    }


def build_pieces(qid, expansions_path, tokenizer, engine, verbose=True):
    """
    Build CNF pieces from keyword expansions.

    Handles flat KEY_ENTITIES format:
    {"KEY_ENTITIES": {"aspect": ["term1", "term2", ...]}, "ASSOCIATED_TERMS": [...]}

    Returns:
        key_pieces: {aspect_name: [piece, ...]}
        associated: [piece, ...]
    """
    data = load_all_expansions(expansions_path).get(qid, {})

    key_pieces = {}
    associated = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"Building pieces for {qid}")
        print(f"{'='*70}")

    key_entities = data.get("KEY_ENTITIES", {})
    for name, terms in key_entities.items():
        if isinstance(terms, dict):
            # Old nested format — flatten
            flat = []
            for level in ["lexical", "conceptual", "referential"]:
                flat.extend(terms.get(level, []))
            terms = flat

        # Include aspect name as a keyword
        all_terms = [name] + (terms if isinstance(terms, list) else [])
        aspect_pieces = []

        if verbose:
            print(f"  KEY '{name}':")

        for kw in all_terms:
            p = _make_term_piece(kw, tokenizer, engine)
            if p:
                p["source_aspect"] = name
                aspect_pieces.append(p)
                if verbose:
                    print(f"    {p['description']}")

        key_pieces[name] = aspect_pieces

    for kw in data.get("ASSOCIATED_TERMS", data.get("ASSOCIATED", [])):
        p = _make_term_piece(kw, tokenizer, engine)
        if p:
            associated.append(p)

    if verbose:
        n_key = sum(len(v) for v in key_pieces.values())
        print(f"  Associated: {len(associated)} terms")
        print(f"  Total key pieces: {n_key}")

    return {"key_pieces": key_pieces, "associated": associated}


def peek_and_grab(
    pieces,
    engine,
    tokenizer,
    max_standalone_key: int = 1000,
    max_standalone_assoc: int = 200,
    prox_peek: int = 10,
    max_clause_freq: int = 80000000,
    verbose: bool = True,
):
    """
    Peek all terms as CNF queries and grab cheap ones.

    For each keyword:
    1. Build CNF (case variants OR'd per word, proximity AND at prox_peek)
    2. Peek count
    3. If count < threshold: grab as a query
    4. If count >= threshold: store as piece for combination
    """
    key_pieces = pieces["key_pieces"]
    associated = pieces["associated"]

    grabbed = []
    remaining_key_pieces = {}
    remaining_assoc_pieces = []
    grabbed_aspects = set()
    seen_ids = set()

    def _build_and_peek(keyword):
        words = keyword.strip().split()
        content = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]
        if not content:
            return None, 0
        if len(words) > 1 and len(content) == 1:
            return None, 0

        cnf = []
        for w in content:
            variants = _case_variants(w, tokenizer, engine)
            if not variants:
                return None, 0
            cnf.append(variants)

        try:
            kwargs = {"max_clause_freq": max_clause_freq}
            if len(cnf) > 1:
                kwargs["max_diff_tokens"] = prox_peek
            result = engine.count_cnf(cnf, **kwargs)
            return cnf, result.get("count", 0)
        except Exception:
            return None, 0

    def _grab(cnf, count, keyword, level):
        cnf_key = tuple(tuple(tuple(a) for a in c) for c in cnf)
        if cnf_key in seen_ids:
            return
        seen_ids.add(cnf_key)
        prox = prox_peek if len(cnf) > 1 else None
        grabbed.append({
            "type": "cnf",
            "cnf": cnf,
            "max_diff_tokens": prox,
            "description": keyword,
            "estimated_count": count,
            "level": level,
        })

    if verbose:
        print(f"\n{'='*70}")
        print(f"Peek and grab (key<{max_standalone_key}, assoc<{max_standalone_assoc}, prox={prox_peek})")
        print(f"{'='*70}")

    # KEY terms
    for aspect_name, aspect_term_pieces in key_pieces.items():
        aspect_remaining = []
        all_grabbed = True

        if verbose:
            print(f"\n  KEY: {aspect_name}")

        for term_piece in aspect_term_pieces:
            kw = term_piece["description"]
            cnf, count = _build_and_peek(kw)

            if cnf is None or count == 0:
                if verbose:
                    print(f"    {0:>8,d}  {kw} (skip)")
                continue

            if count <= max_standalone_key:
                _grab(cnf, count, kw, "S0_key")
                if verbose:
                    print(f"    {count:>8,d}  {kw} -> GRAB")
            else:
                aspect_remaining.append({
                    "keyword": kw,
                    "cnf": cnf,
                    "count": count,
                })
                all_grabbed = False
                if verbose:
                    print(f"    {count:>8,d}  {kw} -> KEEP")

        aspect_remaining.sort(key=lambda p: p["count"])
        remaining_key_pieces[aspect_name] = aspect_remaining

        if all_grabbed and aspect_term_pieces:
            grabbed_aspects.add(aspect_name)
            if verbose:
                print(f"    -> aspect '{aspect_name}' fully grabbed")

    # Associated terms
    if verbose:
        print(f"\n  ASSOCIATED:")

    for piece in associated:
        kw = piece["description"]
        cnf, count = _build_and_peek(kw)

        if cnf is None or count == 0:
            if verbose:
                print(f"    {0:>8,d}  {kw} (skip)")
            continue

        if count <= max_standalone_assoc:
            _grab(cnf, count, kw, "S0_assoc")
            if verbose:
                print(f"    {count:>8,d}  {kw} -> GRAB")
        else:
            remaining_assoc_pieces.append({
                "keyword": kw,
                "cnf": cnf,
                "count": count,
            })
            if verbose:
                print(f"    {count:>8,d}  {kw} -> KEEP")

    remaining_assoc_pieces.sort(key=lambda p: p["count"])

    if verbose:
        total_grabbed = sum(q["estimated_count"] for q in grabbed)
        n_rem_key = sum(len(v) for v in remaining_key_pieces.values())
        print(f"\n{'='*70}")
        print(f"Peek summary:")
        print(f"  Grabbed: {len(grabbed)} queries, ~{total_grabbed:,d} docs")
        print(f"  Grabbed aspects: {grabbed_aspects or 'none'}")
        print(f"  Remaining key: {n_rem_key}")
        print(f"  Remaining associated: {len(remaining_assoc_pieces)}")
        print(f"{'='*70}")

    return {
        "grabbed": grabbed,
        "remaining_key_pieces": remaining_key_pieces,
        "remaining_assoc_pieces": remaining_assoc_pieces,
        "grabbed_aspects": grabbed_aspects,
        "seen_ids": seen_ids,
    }


def build_combination_queries(
    peek,
    engine,
    tokenizer,
    max_docs: int = 10000,
    max_total: int = 200,
    prox_cross: int = 50,
    prox_assoc: int = 80,
    max_clause_freq: int = 80000000,
    verbose: bool = True,
):
    """
    Build combination queries by concatenating CNF pieces across aspects.

    Step 1: All aspects — piece from each aspect AND'd together
    Step 2: Pairwise aspects — every piece_A × piece_B
    Step 3: Key pieces AND associated pieces

    Each query is piece_A["cnf"] + piece_B["cnf"] with max_diff_tokens.
    """
    remaining_key = peek["remaining_key_pieces"]
    remaining_assoc = peek["remaining_assoc_pieces"]
    seen = set(peek.get("seen_ids", set()))

    grabbed_total = sum(q["estimated_count"] for q in peek["grabbed"])
    budget = max_docs - grabbed_total

    aspect_names = [n for n, pieces in remaining_key.items() if pieces]
    queries = []

    def _qkey(cnf, prox):
        return (tuple(tuple(tuple(a) for a in c) for c in cnf), prox)

    def _add(cnf, prox, desc, level, count=None):
        nonlocal budget
        key = _qkey(cnf, prox)
        if key in seen or len(queries) >= max_total:
            return False
        if count is None:
            try:
                kwargs = {"max_clause_freq": max_clause_freq}
                if prox:
                    kwargs["max_diff_tokens"] = prox
                result = engine.count_cnf(cnf, **kwargs)
                count = result.get("count", 0)
            except Exception:
                count = 0
        if count == 0:
            return False
        if count > budget:
            if verbose:
                print(f"    {count:>8,d}  [{level}] {desc} (SKIP: exceeds budget {budget:,d})")
            return False
        seen.add(key)
        queries.append({
            "type": "cnf",
            "cnf": cnf,
            "max_diff_tokens": prox,
            "description": desc,
            "estimated_count": count,
            "level": level,
        })
        budget -= count
        if verbose:
            print(f"    {count:>8,d}  [{level}] {desc} (budget: {budget:,d})")
        return True

    if verbose:
        print(f"\n{'='*70}")
        print(f"Building combination queries (CNF piece concatenation)")
        print(f"  Budget: ~{budget:,d} docs")
        print(f"  Aspects: {aspect_names}")
        for name in aspect_names:
            n = len(remaining_key[name])
            print(f"    {name}: {n} pieces")
        print(f"  Associated: {len(remaining_assoc)} pieces")
        print(f"{'='*70}")

    if budget <= 0:
        if verbose:
            print(f"  Budget exhausted.")
        return queries

    # ================================================================
    # Step 1: All aspects — one piece per aspect, all combinations
    # ================================================================
    if len(aspect_names) >= 3 and verbose:
        print(f"\nStep 1: All aspects AND'd")

    if len(aspect_names) >= 3:
        # Build all combinations: one piece from each aspect
        from itertools import product
        aspect_piece_lists = [remaining_key[n] for n in aspect_names]
        for combo in product(*aspect_piece_lists):
            if budget <= 0 or len(queries) >= max_total:
                break
            cnf = []
            descs = []
            for piece in combo:
                cnf.extend(piece["cnf"])
                descs.append(piece["keyword"])
            desc = " AND ".join(f"({d})" for d in descs)
            _add(cnf, prox_cross, desc, "S1_all")

    # ================================================================
    # Step 2: Pairwise — every piece_A × piece_B across aspects
    # ================================================================
    if verbose:
        print(f"\nStep 2: Pairwise aspect ANDs")

    if len(aspect_names) >= 2:
        # Sort pairs so we try combinations with smallest pieces first
        pair_candidates = []
        for a_name, b_name in combinations(aspect_names, 2):
            for a_piece in remaining_key[a_name]:
                for b_piece in remaining_key[b_name]:
                    est = min(a_piece["count"], b_piece["count"])
                    pair_candidates.append((est, a_piece, b_piece, a_name, b_name))

        pair_candidates.sort(key=lambda x: x[0])

        for est, a_piece, b_piece, a_name, b_name in pair_candidates:
            if budget <= 0 or len(queries) >= max_total:
                break
            cnf = a_piece["cnf"] + b_piece["cnf"]
            desc = f"({a_piece['keyword']}) AND ({b_piece['keyword']})"
            _add(cnf, prox_cross, desc, "S2_pair")

    # ================================================================
    # Step 3: Key pieces AND associated pieces
    # ================================================================
    if verbose:
        print(f"\nStep 3: Key AND associated")

    # Sort: smallest key × smallest assoc first
    assoc_candidates = []
    for name in aspect_names:
        for key_piece in remaining_key[name]:
            for assoc_piece in remaining_assoc:
                est = min(key_piece["count"], assoc_piece["count"])
                assoc_candidates.append((est, key_piece, assoc_piece, name))

    assoc_candidates.sort(key=lambda x: x[0])

    for est, key_piece, assoc_piece, name in assoc_candidates:
        if budget <= 0 or len(queries) >= max_total:
            break
        cnf = key_piece["cnf"] + assoc_piece["cnf"]
        desc = f"({key_piece['keyword']}) AND ({assoc_piece['keyword']})"
        _add(cnf, prox_assoc, desc, "S3_assoc")

    # Summary
    if verbose:
        total_est = sum(q["estimated_count"] for q in queries)
        by_level = {}
        for q in queries:
            by_level.setdefault(q["level"], []).append(q)
        print(f"\n{'='*70}")
        print(f"Combination queries: {len(queries)}, ~{total_est:,d} estimated docs")
        for level, qs in sorted(by_level.items()):
            print(f"  {level}: {len(qs)} queries, ~{sum(q['estimated_count'] for q in qs):,d} docs")
        print(f"  Budget remaining: ~{budget:,d}")
        print(f"{'='*70}")

    return queries