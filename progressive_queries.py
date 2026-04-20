"""
Progressive CNF query construction.

Step 0: Peek all terms (KEY + referential + associated), grab cheap ones
Step 1+: Combine remaining terms intelligently

Usage:
    from progressive_queries import build_pieces, peek_and_grab, build_combination_queries

    pieces = build_pieces(qid, path, tokenizer, engine)
    peek = peek_and_grab(pieces, engine, tokenizer)
    queries = peek["grabbed"] + build_combination_queries(peek, engine, tokenizer)
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


def _make_base_piece(keywords, tokenizer, engine):
    """Pool keyword phrases into one OR clause with case variants per word."""
    all_words = set()
    for kw in keywords:
        for w in kw.strip().split():
            if w.lower() not in STOPWORDS and len(w) > 1:
                all_words.add(w.lower())
    if not all_words:
        return None

    or_ids = []
    seen = set()
    word_variants = {}
    for w in all_words:
        variants = _case_variants(w, tokenizer, engine)
        if variants:
            word_variants[w] = variants
            for ids in variants:
                k = tuple(ids)
                if k not in seen:
                    seen.add(k)
                    or_ids.append(ids)
    if not or_ids:
        return None

    return {
        "cnf_clause": or_ids,
        "words": sorted(word_variants.keys()),
        "description": " OR ".join(sorted(word_variants.keys())),
        "source_keywords": keywords,
    }


def _make_term_piece(keyword, tokenizer, engine):
    """Make a CNF piece from a single keyword phrase."""
    words = keyword.strip().split()
    content = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]
    if not content:
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


def _peek_keyword(keyword, tokenizer, engine):
    """
    Peek a single keyword: try all case variants of the full phrase.
    Returns list of (ids, count, variant_text) and total count.

    Skips multi-word keywords that reduce to a single word after stopword removal
    (e.g., "dealing with" -> "dealing" is too broad to be useful).
    """
    words = keyword.strip().split()
    content = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]
    if not content:
        return [], 0

    # Skip degenerate cases: multi-word input that became single word
    if len(words) > 1 and len(content) == 1:
        return [], 0

    phrase = " ".join(content)
    variants_to_try = {phrase, phrase.lower()}
    if len(phrase) > 1:
        variants_to_try.add(phrase[0].upper() + phrase[1:])
    variants_to_try.add(" ".join(w.capitalize() for w in content))

    results = []
    seen = set()
    total = 0
    for variant in variants_to_try:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if not ids:
            continue
        ids_key = tuple(ids)
        if ids_key in seen:
            continue
        seen.add(ids_key)
        count = engine.count(input_ids=ids).get("count", 0)
        if count > 0:
            results.append({"ids": ids, "count": count, "text": variant})
            total += count

    return results, total


def build_pieces(qid, expansions_path, tokenizer, engine, verbose=True):
    """
    Build CNF pieces from keyword expansions.
    Each keyword becomes its own CNF piece (word-level AND with case variants OR).

    Returns:
        key_pieces: {aspect_name: [piece, ...]}  — lexical expansion pieces per aspect
        referential: [piece, ...] — conceptual + referential pieces
        associated: [piece, ...] — associated term pieces
    """
    data = load_all_expansions(expansions_path).get(qid, {})

    key_pieces = {}
    referential = []
    associated = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"Building pieces for {qid}")
        print(f"{'='*70}")

    key_entities = data.get("KEY_ENTITIES", {})
    for name, aspect in key_entities.items():
        if isinstance(aspect, dict):
            lexical = aspect.get("lexical", [])
            lex_keywords = [name] + lexical

            aspect_pieces = []
            if verbose:
                print(f"  KEY '{name}':")

            for kw in lex_keywords:
                p = _make_term_piece(kw, tokenizer, engine)
                if p:
                    p["source_aspect"] = name
                    p["category"] = "lexical"
                    aspect_pieces.append(p)
                    if verbose:
                        print(f"    lexical: {p['description']}")

            key_pieces[name] = aspect_pieces

            for kw in aspect.get("referential", []):
                p = _make_term_piece(kw, tokenizer, engine)
                if p:
                    p["source_aspect"] = name
                    p["category"] = "referential"
                    referential.append(p)

            for kw in aspect.get("conceptual", []):
                p = _make_term_piece(kw, tokenizer, engine)
                if p:
                    p["source_aspect"] = name
                    p["category"] = "conceptual"
                    referential.append(p)

        elif isinstance(aspect, list):
            aspect_pieces = []
            for kw in [name] + aspect:
                p = _make_term_piece(kw, tokenizer, engine)
                if p:
                    p["source_aspect"] = name
                    p["category"] = "lexical"
                    aspect_pieces.append(p)
            key_pieces[name] = aspect_pieces
            if verbose:
                print(f"  KEY '{name}': {len(aspect_pieces)} terms")

    for kw in data.get("ASSOCIATED_TERMS", data.get("ASSOCIATED", [])):
        p = _make_term_piece(kw, tokenizer, engine)
        if p:
            associated.append(p)

    if verbose:
        n_key = sum(len(v) for v in key_pieces.values())
        print(f"  Referential/conceptual: {len(referential)} terms")
        print(f"  Associated: {len(associated)} terms")
        print(f"  Total key pieces: {n_key}")

    return {"key_pieces": key_pieces, "referential": referential, "associated": associated}


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

    Returns dict with:
        grabbed: list of query dicts ready to execute
        remaining_key_pieces: {aspect_name: [{keyword, cnf, count}, ...]}
        remaining_ref_pieces: [{keyword, cnf, count, source_aspect}, ...]
        remaining_assoc_pieces: [{keyword, cnf, count}, ...]
        grabbed_aspects: set of fully-grabbed aspect names
    """
    key_pieces = pieces["key_pieces"]
    referential = pieces["referential"]
    associated = pieces["associated"]

    grabbed = []
    remaining_key_pieces = {}
    remaining_ref_pieces = []
    remaining_assoc_pieces = []
    grabbed_aspects = set()
    seen_ids = set()

    def _build_and_peek(keyword):
        """Build CNF for keyword and peek count. Returns (cnf, count) or (None, 0)."""
        words = keyword.strip().split()
        content = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]
        if not content:
            return None, 0
        # Skip multi-word that reduced to single word
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
        """Add a query to grabbed list."""
        cnf_key = tuple(tuple(tuple(a) for a in c) for c in cnf)
        if cnf_key in seen_ids:
            return
        seen_ids.add(cnf_key)
        prox = prox_peek if len(cnf) > 1 else None
        grabbed.append({
            "type": "cnf",
            "cnf": cnf,
            "max_diff_tokens": prox,
            "description": f"{keyword}",
            "estimated_count": count,
            "level": level,
        })

    if verbose:
        print(f"\n{'='*70}")
        print(f"Peek and grab (key<{max_standalone_key}, assoc<{max_standalone_assoc}, prox={prox_peek})")
        print(f"{'='*70}")

    # ---- KEY terms ----
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

    # ---- Referential/conceptual terms ----
    if verbose:
        print(f"\n  REFERENTIAL/CONCEPTUAL:")

    for piece in referential:
        kw = piece["description"]
        cnf, count = _build_and_peek(kw)

        if cnf is None or count == 0:
            if verbose:
                print(f"    {0:>8,d}  {kw} (skip)")
            continue

        if count <= max_standalone_assoc:
            _grab(cnf, count, kw, "S0_ref")
            if verbose:
                print(f"    {count:>8,d}  {kw} -> GRAB")
        else:
            remaining_ref_pieces.append({
                "keyword": kw,
                "cnf": cnf,
                "count": count,
                "source_aspect": piece.get("source_aspect"),
                "category": piece.get("category", "referential"),
            })
            if verbose:
                print(f"    {count:>8,d}  {kw} -> KEEP")

    # ---- Associated terms ----
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

    # Sort remaining by count ascending
    remaining_ref_pieces.sort(key=lambda p: p["count"])
    remaining_assoc_pieces.sort(key=lambda p: p["count"])

    if verbose:
        total_grabbed = sum(q["estimated_count"] for q in grabbed)
        n_rem_key = sum(len(v) for v in remaining_key_pieces.values())
        print(f"\n{'='*70}")
        print(f"Peek summary:")
        print(f"  Grabbed: {len(grabbed)} queries, ~{total_grabbed:,d} docs")
        print(f"  Grabbed aspects: {grabbed_aspects or 'none'}")
        print(f"  Remaining key: {n_rem_key}")
        print(f"  Remaining ref/conceptual: {len(remaining_ref_pieces)}")
        print(f"  Remaining associated: {len(remaining_assoc_pieces)}")
        print(f"{'='*70}")

    return {
        "grabbed": grabbed,
        "remaining_key_pieces": remaining_key_pieces,
        "remaining_ref_pieces": remaining_ref_pieces,
        "remaining_assoc_pieces": remaining_assoc_pieces,
        "grabbed_aspects": grabbed_aspects,
        "seen_ids": seen_ids,
    }


def _and_pieces(*pieces):
    """AND multiple pieces together. Returns flat CNF clause list."""
    cnf = []
    for p in pieces:
        if "cnf_clause" in p:
            cnf.append(p["cnf_clause"])
        elif "cnf" in p:
            cnf.extend(p["cnf"])
    return cnf


def _build_phrase_or_clause(pieces_list, tokenizer):
    """
    Build one OR clause from a list of CNF pieces by encoding
    each keyword as a contiguous phrase (all case variants).

    Each piece's keyword becomes multiple alternatives (case variants)
    as contiguous token sequences.

    Returns (or_clause, descriptions) where or_clause is a list of
    token ID sequences to OR together.
    """
    or_ids = []
    seen = set()
    descriptions = []

    for piece in pieces_list:
        kw = piece["keyword"]
        descriptions.append(kw)

        # Remove stopwords
        words = kw.strip().split()
        content = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]
        if not content:
            continue

        phrase = " ".join(content)

        # Generate case variants of the full phrase
        variants = {phrase, phrase.lower()}
        if len(phrase) > 1:
            variants.add(phrase[0].upper() + phrase[1:])
        variants.add(" ".join(w.capitalize() for w in content))

        for variant in variants:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if not ids:
                continue
            ids_key = tuple(ids)
            if ids_key not in seen:
                seen.add(ids_key)
                or_ids.append(ids)

    return or_ids, descriptions


def build_combination_queries(
    peek,
    engine,
    tokenizer,
    max_docs: int = 30000,
    max_query_count: int = 5000,
    max_total: int = 40,
    prox_tight: int = 20,
    prox_medium: int = 50,
    prox_wide: int = 80,
    max_clause_freq: int = 80000000,
    verbose: bool = True,
):
    """
    Build combination queries from remaining terms.

    Uses phrase-OR clauses: each aspect's remaining keywords become one
    OR clause of contiguous phrase variants. Clauses are AND'd across aspects.

    Step 1: All aspects AND'd (one clause per aspect, OR over phrases)
    Step 2: Pairwise aspect ANDs
    Step 3: Each aspect AND'd with each associated piece
    Step 4: Each aspect AND'd with each ref/conceptual piece from other aspects
    """
    remaining_key = peek["remaining_key_pieces"]
    remaining_ref = peek["remaining_ref_pieces"]
    remaining_assoc = peek["remaining_assoc_pieces"]
    seen = set(peek.get("seen_ids", set()))

    grabbed_total = sum(q["estimated_count"] for q in peek["grabbed"])
    budget = max_docs - grabbed_total

    aspect_names = list(remaining_key.keys())
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
        if count > max_query_count:
            if verbose:
                print(f"    {count:>8,d}  [{level}] {desc} (SKIP: >{max_query_count:,d})")
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

    # Build phrase-OR clause per aspect
    aspect_clauses = {}  # aspect_name -> (or_ids, descriptions)
    for name, pieces_list in remaining_key.items():
        or_ids, descs = _build_phrase_or_clause(pieces_list, tokenizer)
        if or_ids:
            aspect_clauses[name] = (or_ids, descs)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Building combination queries")
        print(f"  Budget: ~{budget:,d} docs")
        print(f"  Aspects with phrase-OR clauses: {list(aspect_clauses.keys())}")
        for name, (or_ids, descs) in aspect_clauses.items():
            print(f"    {name}: {len(or_ids)} phrase variants from {len(descs)} keywords")
        print(f"  Remaining ref: {len(remaining_ref)}, assoc: {len(remaining_assoc)}")
        print(f"{'='*70}")

    if budget <= 0:
        if verbose:
            print(f"  Budget exhausted.")
        return queries

    active_names = list(aspect_clauses.keys())

    # ================================================================
    # Step 1: All aspects AND'd
    # ================================================================
    if verbose:
        print(f"\nStep 1: All aspects AND'd")

    if len(active_names) >= 2:
        cnf = [aspect_clauses[n][0] for n in active_names]
        desc = " AND ".join(f"[{n}]" for n in active_names)
        _add(cnf, prox_tight, desc, "S1_all")

    # ================================================================
    # Step 2: Pairwise aspect ANDs
    # ================================================================
    if verbose:
        print(f"\nStep 2: Pairwise aspects")

    for a, b in combinations(active_names, 2):
        if budget <= 0 or len(queries) >= max_total:
            break
        cnf = [aspect_clauses[a][0], aspect_clauses[b][0]]
        desc = f"[{a}] AND [{b}]"
        _add(cnf, prox_medium, desc, "S2_pair")

    # ================================================================
    # Step 3: Each aspect clause AND'd with associated pieces
    # ================================================================
    if verbose:
        print(f"\nStep 3: Aspect AND associated")

    for piece in remaining_assoc:
        if budget <= 0 or len(queries) >= max_total:
            break
        # Build phrase-OR for the associated term too
        assoc_or, _ = _build_phrase_or_clause([piece], tokenizer)
        if not assoc_or:
            continue
        for name in active_names:
            if budget <= 0 or len(queries) >= max_total:
                break
            cnf = [aspect_clauses[name][0], assoc_or]
            desc = f"[{name}] AND ({piece['keyword']})"
            _add(cnf, prox_wide, desc, "S3_assoc")

    # ================================================================
    # Step 4: Each aspect AND'd with ref/conceptual from other aspects
    # ================================================================
    if verbose:
        print(f"\nStep 4: Aspect AND ref/conceptual")

    for piece in remaining_ref:
        if budget <= 0 or len(queries) >= max_total:
            break
        source = piece.get("source_aspect")
        ref_or, _ = _build_phrase_or_clause([piece], tokenizer)
        if not ref_or:
            continue
        for name in active_names:
            if name == source:
                continue
            if budget <= 0 or len(queries) >= max_total:
                break
            cnf = [aspect_clauses[name][0], ref_or]
            desc = f"[{name}] AND ({piece['keyword']})"
            _add(cnf, prox_wide, desc, "S4_ref")

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