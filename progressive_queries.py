"""
Progressive CNF query construction.

Step 1: Build base OR pieces per aspect from lexical expansions
Step 2: Cross-aspect ANDs (core queries)
Step 3: Referential terms standalone (grab specific ones)
Step 4: Remaining referential AND narrowest base
Step 5: Associated terms AND narrowest base

Skips: conceptual terms (covered by base pieces), verbs (too broad)

Usage:
    from progressive_queries import build_pieces, build_queries

    pieces = build_pieces("2024-32912", "./kws.jsonl", tokenizer, engine)
    queries = build_queries(pieces, engine, tokenizer)
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
    """
    Pool keyword phrases into one OR clause.
    Extracts all unique content words, gets case variants for each.
    """
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
    }


def _make_term_piece(keyword, tokenizer, engine):
    """
    Make a CNF piece from a single keyword phrase.
    Multi-word: one clause per word (for proximity AND).
    Single word: single clause with case variants.
    """
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


def build_pieces(qid, expansions_path, tokenizer, engine, verbose=True):
    """
    Build CNF pieces from keyword expansions.

    Returns dict with key_pieces, referential_pieces, associated_pieces.
    """
    data = load_all_expansions(expansions_path).get(qid, {})

    key_pieces = {}       # aspect_name -> base OR piece
    referential = []      # flat list of term pieces
    associated = []       # flat list of term pieces

    if verbose:
        print(f"\n{'='*70}")
        print(f"Building pieces for {qid}")
        print(f"{'='*70}")

    # KEY_ENTITIES
    key_entities = data.get("KEY_ENTITIES", {})
    for name, aspect in key_entities.items():
        if isinstance(aspect, dict):
            lexical = aspect.get("lexical", [])
            # Build base from aspect name + lexical expansions
            piece = _make_base_piece([name] + lexical, tokenizer, engine)
            if piece:
                key_pieces[name] = piece
                if verbose:
                    print(f"  KEY '{name}': ({piece['description']})")

            # Collect referential terms
            for kw in aspect.get("referential", []):
                p = _make_term_piece(kw, tokenizer, engine)
                if p:
                    p["source_aspect"] = name
                    referential.append(p)

            # Skip conceptual — covered by base pieces

        elif isinstance(aspect, list):
            piece = _make_base_piece([name] + aspect, tokenizer, engine)
            if piece:
                key_pieces[name] = piece
                if verbose:
                    print(f"  KEY '{name}': ({piece['description']})")

    # ASSOCIATED_TERMS
    for kw in data.get("ASSOCIATED_TERMS", data.get("ASSOCIATED", [])):
        p = _make_term_piece(kw, tokenizer, engine)
        if p:
            associated.append(p)

    if verbose:
        print(f"  Referential: {len(referential)} terms")
        print(f"  Associated: {len(associated)} terms")

    return {
        "key_pieces": key_pieces,
        "referential": referential,
        "associated": associated,
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


def build_queries(
    pieces,
    engine,
    tokenizer,
    max_standalone: int = 500,
    max_query_count: int = 5000,
    max_total: int = 40,
    prox_tight: int = 20,
    prox_medium: int = 30,
    prox_wide: int = 80,
    max_clause_freq: int = 80000000,
    verbose: bool = True,
):
    """
    Build ~25-30 queries progressively.

    Step 2: Cross-aspect ANDs
    Step 3: Referential standalone
    Step 4: Remaining referential AND narrowest base
    Step 5: Associated AND narrowest base
    """
    key_pieces = pieces["key_pieces"]
    referential = pieces["referential"]
    associated = pieces["associated"]

    queries = []
    seen = set()

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

    aspect_names = list(key_pieces.keys())

    if verbose:
        print(f"\n{'='*70}")
        print(f"Building queries ({len(aspect_names)} aspects, "
              f"{len(referential)} ref, {len(associated)} assoc)")
        print(f"{'='*70}")

    # ================================================================
    # Step 2: Cross-aspect ANDs
    # ================================================================
    if verbose:
        print(f"\nStep 2: Cross-aspect ANDs")

    # All aspects AND'd
    if len(aspect_names) >= 2:
        cnf = [key_pieces[n]["cnf_clause"] for n in aspect_names]
        desc = " AND ".join(f"({key_pieces[n]['description']})" for n in aspect_names)
        _add(cnf, prox_tight, desc, "S2_all")

    # Pairwise
    for a, b in combinations(aspect_names, 2):
        if len(queries) >= max_total:
            break
        cnf = [key_pieces[a]["cnf_clause"], key_pieces[b]["cnf_clause"]]
        desc = f"({key_pieces[a]['description']}) AND ({key_pieces[b]['description']})"
        _add(cnf, prox_medium, desc, "S2_pair")

    # N-1 (drop one) — useful for 3+ aspects
    if len(aspect_names) >= 3:
        for drop in aspect_names:
            if len(queries) >= max_total:
                break
            remaining = [n for n in aspect_names if n != drop]
            cnf = [key_pieces[n]["cnf_clause"] for n in remaining]
            desc = " AND ".join(f"({key_pieces[n]['description']})" for n in remaining)
            desc += f" [no {drop}]"
            # Don't add if it's same as a pairwise (when 3 aspects, N-1 == pairwise)
            _add(cnf, prox_medium, desc, "S2_drop1")

    # ================================================================
    # Step 3: Referential standalone (grab specific ones)
    # ================================================================
    if verbose:
        print(f"\nStep 3: Referential standalone (< {max_standalone})")

    remaining_ref = []
    for piece in referential:
        count = 0
        try:
            kwargs = {"max_clause_freq": max_clause_freq}
            if piece["n_words"] > 1:
                kwargs["max_diff_tokens"] = prox_tight
            result = engine.count_cnf(piece["cnf"], **kwargs)
            count = result.get("count", 0)
        except Exception:
            pass

        if 0 < count <= max_standalone:
            _add(piece["cnf"], prox_tight if piece["n_words"] > 1 else None,
                 f"{piece['description']} (standalone)", "S3_ref", count=count)
        elif count > max_standalone:
            remaining_ref.append((piece, count))

    # ================================================================
    # Step 4: Remaining referential AND narrowest base
    # ================================================================
    if verbose:
        print(f"\nStep 4: Remaining referential AND base")

    # Find narrowest aspect (fewest OR variants = most specific)
    narrowest = min(aspect_names, key=lambda n: len(key_pieces[n]["cnf_clause"])) if aspect_names else None

    for piece, ref_count in remaining_ref:
        if len(queries) >= max_total:
            break
        # AND with a different aspect than the source
        source = piece.get("source_aspect")
        for name in aspect_names:
            if name == source:
                continue  # skip own aspect
            if len(queries) >= max_total:
                break
            cnf = _and_pieces(key_pieces[name], piece)
            desc = f"({key_pieces[name]['description']}) AND ({piece['description']})"
            _add(cnf, prox_wide, desc, "S4_ref")

    # ================================================================
    # Step 5: Associated AND narrowest base
    # ================================================================
    if verbose:
        print(f"\nStep 5: Associated AND base")

    # Sort associated by specificity (fewer words = more specific usually)
    for piece in associated:
        if len(queries) >= max_total:
            break

        # Try AND with narrowest aspect first
        if narrowest:
            cnf = _and_pieces(key_pieces[narrowest], piece)
            desc = f"({key_pieces[narrowest]['description']}) AND ({piece['description']})"
            _add(cnf, prox_wide, desc, "S5_assoc")

    # Summary
    if verbose:
        total_est = sum(q["estimated_count"] for q in queries)
        by_level = {}
        for q in queries:
            by_level.setdefault(q["level"], []).append(q)

        print(f"\n{'='*70}")
        print(f"Summary: {len(queries)} queries, ~{total_est:,d} estimated docs")
        for level, qs in sorted(by_level.items()):
            level_est = sum(q["estimated_count"] for q in qs)
            print(f"  {level}: {len(qs)} queries, ~{level_est:,d} docs")
        print(f"{'='*70}")

    return queries