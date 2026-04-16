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
    """
    words = keyword.strip().split()
    content = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]
    if not content:
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
            piece = _make_base_piece([name] + lexical, tokenizer, engine)
            if piece:
                key_pieces[name] = piece
                if verbose:
                    print(f"  KEY '{name}': ({piece['description']})")
            for kw in aspect.get("referential", []):
                p = _make_term_piece(kw, tokenizer, engine)
                if p:
                    p["source_aspect"] = name
                    referential.append(p)
        elif isinstance(aspect, list):
            piece = _make_base_piece([name] + aspect, tokenizer, engine)
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
    verbose: bool = True,
):
    """
    Peek all terms, grab cheap ones. Returns structured overview.

    Returns dict with:
        grabbed: list of query dicts (ready to execute)
        remaining_key_terms: {aspect_name: [(keyword, total_count, variants), ...]}
        remaining_referential: [(piece, total_count), ...]
        remaining_associated: [(piece, total_count), ...]
        key_pieces: the original pooled OR pieces (for combination queries)
        grabbed_aspects: set of fully-grabbed aspect names
    """
    key_pieces = pieces["key_pieces"]
    referential = pieces["referential"]
    associated = pieces["associated"]

    grabbed = []
    remaining_key_terms = {}
    grabbed_aspects = set()
    seen_ids = set()

    if verbose:
        print(f"\n{'='*70}")
        print(f"Peeking all terms")
        print(f"{'='*70}")

    # ---- KEY terms ----
    for aspect_name, piece in key_pieces.items():
        source_kws = piece.get("source_keywords", [])
        aspect_remaining = []
        all_grabbed = True

        if verbose:
            print(f"\n  KEY: {aspect_name}")

        for kw in source_kws:
            variants, total = _peek_keyword(kw, tokenizer, engine)
            if total == 0:
                if verbose:
                    print(f"    {0:>8,d}  {kw} (not in corpus)")
                continue

            if total <= max_standalone_key:
                # Grab all variants
                for v in variants:
                    ids_key = tuple(v["ids"])
                    if ids_key not in seen_ids:
                        seen_ids.add(ids_key)
                        grabbed.append({
                            "type": "simple",
                            "input_ids": v["ids"],
                            "description": f"{v['text']} (exact)",
                            "estimated_count": v["count"],
                            "level": "S0_key",
                        })
                if verbose:
                    print(f"    {total:>8,d}  {kw} -> GRAB ({len(variants)} variants)")
            else:
                aspect_remaining.append((kw, total, variants))
                all_grabbed = False
                if verbose:
                    print(f"    {total:>8,d}  {kw} -> KEEP")

        remaining_key_terms[aspect_name] = aspect_remaining
        if all_grabbed and source_kws:
            grabbed_aspects.add(aspect_name)
            if verbose:
                print(f"    -> aspect '{aspect_name}' fully grabbed")

    # ---- Referential terms ----
    if verbose:
        print(f"\n  REFERENTIAL:")

    remaining_ref = []
    for piece in referential:
        variants, total = _peek_keyword(piece["description"], tokenizer, engine)
        if total == 0:
            if verbose:
                print(f"    {0:>8,d}  {piece['description']} (not in corpus)")
            continue

        if total <= max_standalone_assoc:
            for v in variants:
                ids_key = tuple(v["ids"])
                if ids_key not in seen_ids:
                    seen_ids.add(ids_key)
                    grabbed.append({
                        "type": "simple",
                        "input_ids": v["ids"],
                        "description": f"{v['text']} (ref)",
                        "estimated_count": v["count"],
                        "level": "S0_ref",
                    })
            if verbose:
                print(f"    {total:>8,d}  {piece['description']} -> GRAB")
        else:
            remaining_ref.append((piece, total))
            if verbose:
                print(f"    {total:>8,d}  {piece['description']} -> KEEP")

    # ---- Associated terms ----
    if verbose:
        print(f"\n  ASSOCIATED:")

    remaining_assoc = []
    for piece in associated:
        variants, total = _peek_keyword(piece["description"], tokenizer, engine)
        if total == 0:
            if verbose:
                print(f"    {0:>8,d}  {piece['description']} (not in corpus)")
            continue

        if total <= max_standalone_assoc:
            for v in variants:
                ids_key = tuple(v["ids"])
                if ids_key not in seen_ids:
                    seen_ids.add(ids_key)
                    grabbed.append({
                        "type": "simple",
                        "input_ids": v["ids"],
                        "description": f"{v['text']} (assoc)",
                        "estimated_count": v["count"],
                        "level": "S0_assoc",
                    })
            if verbose:
                print(f"    {total:>8,d}  {piece['description']} -> GRAB")
        else:
            remaining_assoc.append((piece, total))
            if verbose:
                print(f"    {total:>8,d}  {piece['description']} -> KEEP")

    # Sort remaining by count ascending (most specific first)
    remaining_ref.sort(key=lambda x: x[1])
    remaining_assoc.sort(key=lambda x: x[1])

    # Summary
    if verbose:
        total_grabbed = sum(q["estimated_count"] for q in grabbed)
        print(f"\n{'='*70}")
        print(f"Peek summary:")
        print(f"  Grabbed: {len(grabbed)} queries, ~{total_grabbed:,d} docs")
        print(f"  Grabbed aspects: {grabbed_aspects or 'none'}")
        print(f"  Remaining KEY terms: {sum(len(v) for v in remaining_key_terms.values())}")
        print(f"  Remaining referential: {len(remaining_ref)}")
        print(f"  Remaining associated: {len(remaining_assoc)}")
        print(f"{'='*70}")

    return {
        "grabbed": grabbed,
        "remaining_key_terms": remaining_key_terms,
        "remaining_referential": remaining_ref,
        "remaining_associated": remaining_assoc,
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
    max_query_count: int = 5000,
    max_total: int = 40,
    prox_tight: int = 20,
    prox_medium: int = 30,
    prox_wide: int = 80,
    max_clause_freq: int = 80000000,
    verbose: bool = True,
):
    """
    Build combination queries from remaining (non-grabbed) terms.

    Uses the peek results to know what's left and their counts.
    """
    key_pieces = peek["key_pieces"]
    grabbed_aspects = peek["grabbed_aspects"]
    remaining_key = peek["remaining_key_terms"]
    remaining_ref = peek["remaining_referential"]
    remaining_assoc = peek["remaining_associated"]
    seen = set(peek.get("seen_ids", set()))

    # Only use non-grabbed aspects for combinations
    remaining_pieces = {n: p for n, p in key_pieces.items() if n not in grabbed_aspects}
    remaining_names = list(remaining_pieces.keys())

    queries = []

    def _qkey(cnf, prox):
        return (tuple(tuple(tuple(a) for a in c) for c in cnf), prox)

    def _add(cnf, prox, desc, level, count=None):
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
            if verbose:
                print(f"           0  [{level}] {desc} (SKIP)")
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
        if verbose:
            print(f"    {count:>8,d}  [{level}] {desc}")
        return True

    if verbose:
        print(f"\n{'='*70}")
        print(f"Building combination queries")
        print(f"  Remaining aspects: {remaining_names}")
        print(f"  Remaining ref: {len(remaining_ref)}, assoc: {len(remaining_assoc)}")
        print(f"{'='*70}")

    # ================================================================
    # Cross-aspect ANDs
    # ================================================================
    if verbose:
        print(f"\nCross-aspect ANDs:")

    if len(remaining_names) >= 2:
        # All remaining AND'd
        cnf = [remaining_pieces[n]["cnf_clause"] for n in remaining_names]
        desc = " AND ".join(f"({remaining_pieces[n]['description']})" for n in remaining_names)
        _add(cnf, prox_tight, desc, "S1_all")

        # Pairwise
        for a, b in combinations(remaining_names, 2):
            if len(queries) >= max_total:
                break
            cnf = [remaining_pieces[a]["cnf_clause"], remaining_pieces[b]["cnf_clause"]]
            desc = f"({remaining_pieces[a]['description']}) AND ({remaining_pieces[b]['description']})"
            _add(cnf, prox_medium, desc, "S1_pair")

    # ================================================================
    # Referential AND other aspects
    # ================================================================
    if verbose:
        print(f"\nReferential AND other aspects:")

    for piece, ref_count in remaining_ref:
        if len(queries) >= max_total:
            break
        source = piece.get("source_aspect")
        for name in remaining_names:
            if name == source:
                continue
            if len(queries) >= max_total:
                break
            cnf = _and_pieces(remaining_pieces[name], piece)
            desc = f"({remaining_pieces[name]['description']}) AND ({piece['description']})"
            _add(cnf, prox_wide, desc, "S2_ref")

    # ================================================================
    # Associated AND aspects
    # ================================================================
    if verbose:
        print(f"\nAssociated AND aspects:")

    narrowest = min(remaining_names, key=lambda n: len(remaining_pieces[n]["cnf_clause"])) if remaining_names else None

    for piece, assoc_count in remaining_assoc:
        if len(queries) >= max_total:
            break
        if narrowest:
            cnf = _and_pieces(remaining_pieces[narrowest], piece)
            desc = f"({remaining_pieces[narrowest]['description']}) AND ({piece['description']})"
            _add(cnf, prox_wide, desc, "S3_assoc")

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
        print(f"{'='*70}")
