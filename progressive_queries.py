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
    Peek all terms and build CNF pieces for combination queries.
    No standalone grabs — everything goes through combinations.

    For each keyword:
    1. Build CNF version (case variants OR'd per word, proximity AND)
    2. Peek its count
    3. Store as a piece for later combination

    Returns dict with:
        grabbed: [] (empty — no standalone grabs)
        remaining_key_pieces: {aspect_name: [{keyword, cnf, count}, ...]}
        remaining_ref_pieces: [{keyword, cnf, count, source_aspect}, ...]
        remaining_assoc_pieces: [{keyword, cnf, count}, ...]
        key_pieces: original pooled OR pieces
        grabbed_aspects: set() (empty)
    """
    key_pieces = pieces["key_pieces"]
    referential = pieces["referential"]
    associated = pieces["associated"]

    remaining_key_pieces = {}
    remaining_ref_pieces = []
    remaining_assoc_pieces = []

    def _build_cnf(keyword):
        """Build CNF for a keyword with case variants per word."""
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

    if verbose:
        print(f"\n{'='*70}")
        print(f"Peek all terms (prox={prox_peek})")
        print(f"{'='*70}")

    # ---- KEY terms ----
    for aspect_name, aspect_term_pieces in key_pieces.items():
        aspect_pieces = []

        if verbose:
            print(f"\n  KEY: {aspect_name}")

        for term_piece in aspect_term_pieces:
            kw = term_piece["description"]
            cnf, count = _build_cnf(kw)
            if cnf is None or count == 0:
                if verbose:
                    print(f"    {0:>8,d}  {kw} (skip)")
                continue

            aspect_pieces.append({
                "keyword": kw,
                "cnf": cnf,
                "count": count,
            })
            if verbose:
                print(f"    {count:>8,d}  {kw}")

        aspect_pieces.sort(key=lambda p: p["count"])
        remaining_key_pieces[aspect_name] = aspect_pieces

    # ---- Referential/conceptual terms ----
    if verbose:
        print(f"\n  REFERENTIAL/CONCEPTUAL:")

    for piece in referential:
        kw = piece["description"]
        cnf, count = _build_cnf(kw)
        if cnf is None or count == 0:
            if verbose:
                print(f"    {0:>8,d}  {kw} (skip)")
            continue

        remaining_ref_pieces.append({
            "keyword": kw,
            "cnf": cnf,
            "count": count,
            "source_aspect": piece.get("source_aspect"),
            "category": piece.get("category", "referential"),
        })
        if verbose:
            print(f"    {count:>8,d}  {kw}")

    # ---- Associated terms ----
    if verbose:
        print(f"\n  ASSOCIATED:")

    for piece in associated:
        kw = piece["description"]
        cnf, count = _build_cnf(kw)
        if cnf is None or count == 0:
            if verbose:
                print(f"    {0:>8,d}  {kw} (skip)")
            continue

        remaining_assoc_pieces.append({
            "keyword": kw,
            "cnf": cnf,
            "count": count,
        })
        if verbose:
            print(f"    {count:>8,d}  {kw}")

    # Sort by count ascending
    remaining_ref_pieces.sort(key=lambda p: p["count"])
    remaining_assoc_pieces.sort(key=lambda p: p["count"])

    if verbose:
        n_key = sum(len(v) for v in remaining_key_pieces.values())
        print(f"\n{'='*70}")
        print(f"Peek summary:")
        print(f"  Key pieces: {n_key}")
        print(f"  Referential/conceptual pieces: {len(remaining_ref_pieces)}")
        print(f"  Associated pieces: {len(remaining_assoc_pieces)}")
        print(f"{'='*70}")

    return {
        "grabbed": [],
        "remaining_key_pieces": remaining_key_pieces,
        "remaining_ref_pieces": remaining_ref_pieces,
        "remaining_assoc_pieces": remaining_assoc_pieces,
        "key_pieces": key_pieces,
        "grabbed_aspects": set(),
        "seen_ids": set(),
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

    Each remaining piece is AND'd with pieces from other aspects.
    All candidates are sorted by estimated count ascending and
    added greedily until the budget is filled.

    Step 1: Cross-aspect key piece × key piece ANDs
    Step 2: Remaining key/ref/assoc pieces AND'd with other aspects' key pieces
    """
    remaining_key = peek["remaining_key_pieces"]  # {aspect: [{keyword, cnf, count}, ...]}
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

    if verbose:
        print(f"\n{'='*70}")
        print(f"Building combination queries")
        print(f"  Budget: ~{budget:,d} docs")
        print(f"  Aspects: {aspect_names}")
        n_key = sum(len(v) for v in remaining_key.values())
        print(f"  Key pieces: {n_key}, ref: {len(remaining_ref)}, assoc: {len(remaining_assoc)}")
        print(f"{'='*70}")

    if budget <= 0:
        if verbose:
            print(f"  Budget exhausted.")
        return queries

    # ================================================================
    # Build all candidate queries
    # ================================================================
    candidates = []

    # 1. Cross-aspect key × key: piece from aspect A AND piece from aspect B
    for a_name, a_pieces in remaining_key.items():
        for b_name, b_pieces in remaining_key.items():
            if a_name >= b_name:  # avoid duplicates + self
                continue
            for a_piece in a_pieces:
                for b_piece in b_pieces:
                    cnf = a_piece["cnf"] + b_piece["cnf"]
                    est = min(a_piece["count"], b_piece["count"])  # rough lower bound
                    desc = f"({a_piece['keyword']}) AND ({b_piece['keyword']})"
                    candidates.append((est, cnf, prox_tight, desc, "C1_key_x_key"))

    # 2. Ref/conceptual pieces AND key pieces from other aspects
    for piece in remaining_ref:
        source = piece.get("source_aspect")
        for other_name, other_pieces in remaining_key.items():
            if other_name == source:
                continue
            for other_piece in other_pieces:
                cnf = other_piece["cnf"] + piece["cnf"]
                est = min(other_piece["count"], piece["count"])
                desc = f"({other_piece['keyword']}) AND ({piece['keyword']})"
                candidates.append((est, cnf, prox_medium, desc, "C2_ref"))

    # 3. Associated pieces AND key pieces from each aspect
    for piece in remaining_assoc:
        for name, key_pieces_list in remaining_key.items():
            for key_piece in key_pieces_list:
                cnf = key_piece["cnf"] + piece["cnf"]
                est = min(key_piece["count"], piece["count"])
                desc = f"({key_piece['keyword']}) AND ({piece['keyword']})"
                candidates.append((est, cnf, prox_wide, desc, "C3_assoc"))

    # Sort by estimated count ascending (most specific first)
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