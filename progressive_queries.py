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
    """Build CNF pieces from keyword expansions."""
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
            # Only include aspect name if it's multi-word (single-word aspect names
            # like "coping" or "economy" are too broad on their own)
            lex_for_piece = list(lexical)
            if " " in name.strip():
                lex_for_piece = [name] + lex_for_piece

            piece = _make_base_piece(lex_for_piece, tokenizer, engine)
            if piece:
                key_pieces[name] = piece
                if verbose:
                    print(f"  KEY '{name}': ({piece['description']})")

            # Referential + conceptual both become individual term pieces
            # (conceptual terms often introduce new vocabulary, not just narrowing)
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
                    referential.append(p)  # treat same as referential for now

        elif isinstance(aspect, list):
            lex_for_piece = list(aspect)
            if " " in name.strip():
                lex_for_piece = [name] + lex_for_piece
            piece = _make_base_piece(lex_for_piece, tokenizer, engine)
            if piece:
                key_pieces[name] = piece
                if verbose:
                    print(f"  KEY '{name}': ({piece['description']})")

    for kw in data.get("ASSOCIATED_TERMS", data.get("ASSOCIATED", [])):
        p = _make_term_piece(kw, tokenizer, engine)
        if p:
            associated.append(p)

    if verbose:
        print(f"  Referential: {len(referential)} terms")
        print(f"  Associated: {len(associated)} terms")

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
    Peek all terms with two strategies and grab cheap ones.

    For each keyword:
    1. Peek exact-phrase count (sum across case variants)
    2. If total < threshold: grab exact variants, done.
    3. Else: peek CNF version at prox_peek (with case variants in OR)
       - If CNF count < threshold: grab CNF query only (superset), done.
       - Else: grab exact variants AND store CNF piece for later combination.

    Returns dict with:
        grabbed: list of query dicts (exact + some CNFs)
        remaining_key_pieces: {aspect_name: [(keyword, piece_with_cnf, count), ...]}
                              ordered by count ascending
        remaining_assoc_pieces: [(keyword, piece_with_cnf, count), ...]
                                ordered by count ascending
        key_pieces: original pooled OR pieces (for Last-resort combinations)
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

    def _grab_exact_variants(variants, category):
        """Add all exact-phrase variants to grabbed list."""
        for v in variants:
            ids_key = tuple(v["ids"])
            if ids_key not in seen_ids:
                seen_ids.add(ids_key)
                grabbed.append({
                    "type": "simple",
                    "input_ids": v["ids"],
                    "description": f"{v['text']} (exact)",
                    "estimated_count": v["count"],
                    "level": f"S0_{category}",
                })

    def _peek_cnf_version(keyword):
        """
        Build CNF for a keyword (OR case variants per word) and peek its count.
        Returns (cnf, count) or (None, 0) if can't build.

        For single words, returns a 1-clause CNF (OR over case variants).
        For multi-word, returns multi-clause CNF for proximity AND.
        """
        words = keyword.strip().split()
        content = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]
        if not content:
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

    def _process_keyword(kw, threshold, category):
        """
        Process a single keyword. Returns:
            ("grabbed_exact", None) — fully grabbed via exact-phrase variants
            ("grabbed_cnf", None) — fully grabbed via CNF query
            ("remaining", piece_dict) — CNF kept for combination
            ("skipped", None) — no hits at all
        """
        variants, exact_total = _peek_keyword(kw, tokenizer, engine)
        if exact_total == 0:
            return "skipped", None, 0

        if exact_total <= threshold:
            _grab_exact_variants(variants, category)
            return "grabbed_exact", None, exact_total

        # Exact total exceeds threshold — try CNF
        cnf, cnf_count = _peek_cnf_version(kw)

        if cnf is None or cnf_count == 0:
            return "skipped", None, exact_total

        if 0 < cnf_count <= threshold:
            # CNF fits — grab it (superset of exact)
            cnf_tuple = tuple(tuple(tuple(a) for a in c) for c in cnf)
            cnf_key = (cnf_tuple, prox_peek)
            if cnf_key not in seen_ids:
                seen_ids.add(cnf_key)
                grabbed.append({
                    "type": "cnf",
                    "cnf": cnf,
                    "max_diff_tokens": prox_peek,
                    "description": f"{kw} (CNF prox={prox_peek})",
                    "estimated_count": cnf_count,
                    "level": f"S0_{category}_cnf",
                })
            return "grabbed_cnf", None, cnf_count

        # Both exact and CNF too big — don't grab anything, keep CNF for combinations
        return "remaining", {
            "keyword": kw,
            "cnf": cnf,
            "count": cnf_count,
            "exact_count": exact_total,
        }, cnf_count

    if verbose:
        print(f"\n{'='*70}")
        print(f"Peek and grab (key<{max_standalone_key}, assoc<{max_standalone_assoc})")
        print(f"{'='*70}")

    # ---- KEY terms ----
    for aspect_name, piece in key_pieces.items():
        source_kws = piece.get("source_keywords", [])
        aspect_remaining = []
        any_grabbed = False
        any_remaining = False

        if verbose:
            print(f"\n  KEY: {aspect_name}")

        for kw in source_kws:
            status, remaining_piece, count = _process_keyword(kw, max_standalone_key, "key")
            if status == "grabbed_exact":
                any_grabbed = True
                if verbose:
                    print(f"    {count:>8,d}  {kw} -> GRAB exact")
            elif status == "grabbed_cnf":
                any_grabbed = True
                if verbose:
                    print(f"    {count:>8,d}  {kw} -> GRAB CNF (prox={prox_peek})")
            elif status == "remaining":
                aspect_remaining.append(remaining_piece)
                any_remaining = True
                if verbose:
                    print(f"    {remaining_piece['exact_count']:>8,d}/{count:,d}  {kw} -> CNF kept for combination")
            elif verbose:
                print(f"    {0:>8,d}  {kw} (not in corpus)")

        # Sort remaining by CNF count ascending
        aspect_remaining.sort(key=lambda p: p["count"])
        remaining_key_pieces[aspect_name] = aspect_remaining

        if not any_remaining and any_grabbed:
            grabbed_aspects.add(aspect_name)
            if verbose:
                print(f"    -> aspect '{aspect_name}' fully grabbed")

    # ---- Referential terms ----
    if verbose:
        print(f"\n  REFERENTIAL:")

    for piece in referential:
        kw = piece["description"]
        status, remaining_piece, count = _process_keyword(kw, max_standalone_assoc, "ref")
        if status == "grabbed_exact":
            if verbose:
                print(f"    {count:>8,d}  {kw} -> GRAB exact")
        elif status == "grabbed_cnf":
            if verbose:
                print(f"    {count:>8,d}  {kw} -> GRAB CNF")
        elif status == "remaining":
            remaining_piece["source_aspect"] = piece.get("source_aspect")
            remaining_piece["category"] = piece.get("category", "referential")
            remaining_ref_pieces.append(remaining_piece)
            if verbose:
                print(f"    {remaining_piece['exact_count']:>8,d}/{count:,d}  {kw} -> CNF kept for combination")
        elif verbose:
            print(f"    {0:>8,d}  {kw} (not in corpus)")

    # ---- Associated terms ----
    if verbose:
        print(f"\n  ASSOCIATED:")

    for piece in associated:
        kw = piece["description"]
        status, remaining_piece, count = _process_keyword(kw, max_standalone_assoc, "assoc")
        if status == "grabbed_exact":
            if verbose:
                print(f"    {count:>8,d}  {kw} -> GRAB exact")
        elif status == "grabbed_cnf":
            if verbose:
                print(f"    {count:>8,d}  {kw} -> GRAB CNF")
        elif status == "remaining":
            remaining_assoc_pieces.append(remaining_piece)
            if verbose:
                print(f"    {remaining_piece['exact_count']:>8,d}/{count:,d}  {kw} -> CNF kept for combination")
        elif verbose:
            print(f"    {0:>8,d}  {kw} (not in corpus)")

    # Sort remaining
    remaining_ref_pieces.sort(key=lambda p: p["count"])
    remaining_assoc_pieces.sort(key=lambda p: p["count"])

    if verbose:
        total_grabbed = sum(q["estimated_count"] for q in grabbed)
        print(f"\n{'='*70}")
        print(f"Peek summary:")
        print(f"  Grabbed: {len(grabbed)} queries, ~{total_grabbed:,d} docs")
        print(f"  Grabbed aspects: {grabbed_aspects or 'none'}")
        n_rem_key = sum(len(v) for v in remaining_key_pieces.values())
        print(f"  Remaining key CNFs: {n_rem_key}")
        print(f"  Remaining referential CNFs: {len(remaining_ref_pieces)}")
        print(f"  Remaining associated CNFs: {len(remaining_assoc_pieces)}")
        print(f"{'='*70}")

    return {
        "grabbed": grabbed,
        "remaining_key_pieces": remaining_key_pieces,
        "remaining_ref_pieces": remaining_ref_pieces,
        "remaining_assoc_pieces": remaining_assoc_pieces,
        "key_pieces": key_pieces,
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


def build_combination_queries(
    peek,
    engine,
    tokenizer,
    max_docs: int = 30000,
    max_query_count: int = 5000,
    max_total: int = 40,
    prox_tight: int = 20,
    prox_medium: int = 30,
    prox_wide: int = 80,
    max_clause_freq: int = 80000000,
    verbose: bool = True,
):
    """
    Build combination queries from remaining terms, filling a doc budget.

    Each remaining CNF piece is AND'd with the pooled OR clause of another aspect.
    All candidates are sorted by count ascending (most specific first) and
    added greedily until the budget is filled.

    Args:
        max_docs: Total doc budget (grabbed + combination).
        max_query_count: Skip individual queries above this count.
        max_total: Max number of combination queries.
    """
    key_pieces = peek["key_pieces"]
    grabbed_aspects = peek["grabbed_aspects"]
    remaining_key = peek["remaining_key_pieces"]
    remaining_ref = peek["remaining_ref_pieces"]
    remaining_assoc = peek["remaining_assoc_pieces"]
    seen = set(peek.get("seen_ids", set()))

    # Budget: subtract already-grabbed docs
    grabbed_total = sum(q["estimated_count"] for q in peek["grabbed"])
    budget = max_docs - grabbed_total

    aspect_names = list(key_pieces.keys())
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

    if verbose:
        print(f"\n{'='*70}")
        print(f"Building combination queries")
        print(f"  Grabbed so far: ~{grabbed_total:,d} docs")
        print(f"  Budget remaining: ~{budget:,d} docs")
        print(f"  Aspects: {aspect_names}")
        print(f"{'='*70}")

    if budget <= 0:
        if verbose:
            print(f"  Budget exhausted, skipping combinations.")
        return queries

    # ================================================================
    # Build all candidate queries, then sort by count and fill budget
    # ================================================================

    # A candidate is (estimated_count, cnf, prox, desc, level)
    candidates = []

    # 1. Remaining KEY lexical pieces AND other aspects' pooled OR
    for aspect_name, pieces_list in remaining_key.items():
        for piece in pieces_list:
            # AND with each OTHER aspect's pooled OR
            for other_name in aspect_names:
                if other_name == aspect_name:
                    continue
                cnf = [key_pieces[other_name]["cnf_clause"]] + piece["cnf"]
                desc = f"({key_pieces[other_name]['description']}) AND ({piece['keyword']})"
                candidates.append((piece["count"], cnf, prox_tight, desc, "C1_key"))

    # 2. Remaining referential/conceptual AND other aspects' pooled OR
    for piece in remaining_ref:
        source = piece.get("source_aspect")
        for other_name in aspect_names:
            if other_name == source:
                continue
            cnf = [key_pieces[other_name]["cnf_clause"]] + piece["cnf"]
            desc = f"({key_pieces[other_name]['description']}) AND ({piece['keyword']})"
            candidates.append((piece["count"], cnf, prox_medium, desc, "C2_ref"))

    # 3. Remaining associated AND each aspect's pooled OR
    for piece in remaining_assoc:
        for name in aspect_names:
            cnf = [key_pieces[name]["cnf_clause"]] + piece["cnf"]
            desc = f"({key_pieces[name]['description']}) AND ({piece['keyword']})"
            candidates.append((piece["count"], cnf, prox_wide, desc, "C3_assoc"))

    # Sort by count ascending (most specific first)
    candidates.sort(key=lambda x: x[0])

    if verbose:
        print(f"\n  {len(candidates)} candidate queries, filling budget...")
        print()

    # Greedily add candidates
    for est_count, cnf, prox, desc, level in candidates:
        if budget <= 0 or len(queries) >= max_total:
            break
        _add(cnf, prox, desc, level)

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