"""
Progressive CNF query construction.

Step 1: Build CNF pieces from keywords
  - Lexical pieces per aspect (pooled synonyms)
  - Conceptual/referential pieces per aspect (pooled where possible)
  - Associated pieces

Step 2: Iteratively build queries, exhausting cheap grabs first
  - Standalone KEY pieces under threshold → grab all
  - Standalone associated pieces under threshold → grab all
  - Remaining associated AND'd with remaining KEY pieces
  - Cross-KEY AND combinations
  - Conceptual/referential AND'd with KEY pieces

Usage:
    from term_expansion import expand_and_peek
    from progressive_queries import build_pieces, build_queries

    terms = expand_and_peek(qid, path, tokenizer, engine)
    pieces = build_pieces(qid, path, terms)
    queries = build_queries(pieces, engine, tokenizer)
"""

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


def _make_piece(keywords, tokenizer, engine, max_diff_tokens=5):
    """
    Pool a list of keyword phrases into one CNF piece.

    Groups words by position-independent pooling:
    ["Vietnam War", "Vietnam conflict", "Vietnamese war"] ->
    {
      pool1: (Vietnam, vietnam, Vietnamese, vietnamese),
      pool2: (War, war, conflict, Conflict)
    }

    Strategy: collect all unique content words across all terms,
    then build one OR clause per unique root word (case-expanded).

    Returns dict with 'cnf', 'words', 'description', or None if invalid.
    """
    if not keywords:
        return None

    # Collect all unique content words across all keyword phrases
    all_words = set()
    for kw in keywords:
        words = kw.strip().split()
        for w in words:
            if w.lower() not in STOPWORDS and len(w) > 1:
                all_words.add(w.lower())

    if not all_words:
        return None

    # For each unique word, get case variants
    word_variants = {}  # word_lower -> list of valid ids
    for w in all_words:
        variants = _case_variants(w, tokenizer, engine)
        if variants:
            word_variants[w] = variants

    if not word_variants:
        return None

    # Build CNF: one clause with ALL word variants OR'd together
    # This is a single OR clause, not multi-clause AND
    all_ids = []
    seen = set()
    for w, variants in word_variants.items():
        for ids in variants:
            ids_key = tuple(ids)
            if ids_key not in seen:
                seen.add(ids_key)
                all_ids.append(ids)

    if not all_ids:
        return None

    return {
        "cnf_clause": all_ids,  # single OR clause
        "words": sorted(word_variants.keys()),
        "description": " OR ".join(sorted(word_variants.keys())),
        "n_words": len(word_variants),
        "source_keywords": keywords,
    }


def _make_term_piece(keyword, tokenizer, engine, max_diff_tokens=5):
    """
    Make a CNF piece from a single keyword phrase.
    Multi-word: proximity AND with case variants per word.
    Single word: single OR clause with case variants.

    Returns dict with 'cnf' (list of clauses), 'description', etc.
    """
    words = keyword.strip().split()
    content_words = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]

    if not content_words:
        return None

    clauses = []
    word_names = []

    for w in content_words:
        variants = _case_variants(w, tokenizer, engine)
        if not variants:
            return None  # word not in corpus
        clauses.append(variants)
        word_names.append(w)

    return {
        "cnf": clauses,  # list of clauses, each clause is list of OR'd ids
        "words": word_names,
        "description": keyword,
        "n_words": len(word_names),
        "max_diff_tokens": max_diff_tokens if len(clauses) > 1 else None,
    }


def build_pieces(
    qid: str,
    expansions_path: str,
    terms: dict,
    tokenizer,
    engine,
    max_diff_tokens: int = 5,
    verbose: bool = True,
):
    """
    Build CNF pieces from keyword expansions.

    Returns:
        {
            "key_pieces": {aspect_name: piece},  # lexical pools per aspect
            "conceptual_pieces": {aspect_name: [piece, ...]},
            "referential_pieces": {aspect_name: [piece, ...]},
            "associated_pieces": [piece, ...],
            "verb_pieces": [piece, ...],
        }
    """
    all_exp = load_all_expansions(expansions_path)
    data = all_exp.get(qid, {})

    key_pieces = {}
    conceptual_pieces = {}
    referential_pieces = {}
    associated_pieces = []
    verb_pieces = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"Building CNF pieces for {qid}")
        print(f"{'='*70}")

    # KEY_ENTITIES aspects
    key_entities = data.get("KEY_ENTITIES", {})
    for aspect_name, aspect_data in key_entities.items():
        if isinstance(aspect_data, dict):
            # Lexical -> pool into one piece
            lexical = aspect_data.get("lexical", [])
            if lexical:
                # Also add the aspect name itself as a keyword
                all_lexical = [aspect_name] + lexical
                piece = _make_piece(all_lexical, tokenizer, engine, max_diff_tokens)
                if piece:
                    key_pieces[aspect_name] = piece
                    if verbose:
                        print(f"\n  KEY '{aspect_name}' lexical piece: ({piece['description']})")

            # Conceptual -> individual pieces
            conceptual = aspect_data.get("conceptual", [])
            concept_list = []
            for kw in conceptual:
                p = _make_term_piece(kw, tokenizer, engine, max_diff_tokens)
                if p:
                    concept_list.append(p)
                    if verbose:
                        print(f"    conceptual: {p['description']}")
            conceptual_pieces[aspect_name] = concept_list

            # Referential -> individual pieces
            referential = aspect_data.get("referential", [])
            ref_list = []
            for kw in referential:
                p = _make_term_piece(kw, tokenizer, engine, max_diff_tokens)
                if p:
                    ref_list.append(p)
                    if verbose:
                        print(f"    referential: {p['description']}")
            referential_pieces[aspect_name] = ref_list

        elif isinstance(aspect_data, list):
            # Flat list — treat all as lexical
            piece = _make_piece([aspect_name] + aspect_data, tokenizer, engine, max_diff_tokens)
            if piece:
                key_pieces[aspect_name] = piece
                if verbose:
                    print(f"\n  KEY '{aspect_name}' piece: ({piece['description']})")

    # If no KEY_ENTITIES, try aspects from faceted keywords
    if not key_pieces:
        aspects = terms.get("aspect_terms", {})
        for name, exp_terms in aspects.items():
            kws = [t["original"] for t in exp_terms if t.get("valid", False)]
            piece = _make_piece([name] + kws, tokenizer, engine, max_diff_tokens)
            if piece:
                key_pieces[name] = piece
                if verbose:
                    print(f"\n  Aspect '{name}' piece: ({piece['description']})")

    # ASSOCIATED_TERMS
    associated = data.get("ASSOCIATED_TERMS", data.get("ASSOCIATED", []))
    if isinstance(associated, list):
        if verbose and associated:
            print(f"\n  ASSOCIATED:")
        for kw in associated:
            p = _make_term_piece(kw, tokenizer, engine, max_diff_tokens)
            if p:
                associated_pieces.append(p)
                if verbose:
                    print(f"    {p['description']}")

    # VERBS
    verb_data = data.get("VERBS", [])
    if isinstance(verb_data, dict):
        all_verbs = []
        for verb, expansions in verb_data.items():
            all_verbs.append(verb)
            all_verbs.extend(expansions)
        verb_data = all_verbs

    if verb_data:
        if verbose:
            print(f"\n  VERBS:")
        for v in verb_data:
            p = _make_term_piece(v, tokenizer, engine, max_diff_tokens)
            if p:
                verb_pieces.append(p)
                if verbose:
                    print(f"    {p['description']}")

    if verbose:
        print(f"\n  Total: {len(key_pieces)} key pieces, "
              f"{sum(len(v) for v in conceptual_pieces.values())} conceptual, "
              f"{sum(len(v) for v in referential_pieces.values())} referential, "
              f"{len(associated_pieces)} associated, {len(verb_pieces)} verbs")

    return {
        "key_pieces": key_pieces,
        "conceptual_pieces": conceptual_pieces,
        "referential_pieces": referential_pieces,
        "associated_pieces": associated_pieces,
        "verb_pieces": verb_pieces,
    }


def _count_piece(piece, engine, max_clause_freq=80000000, max_diff_tokens=None):
    """Count hits for a piece (single clause or multi-clause CNF)."""
    if "cnf_clause" in piece:
        # Single OR clause — count it
        try:
            # Wrap in CNF format
            result = engine.count_cnf([piece["cnf_clause"]],
                                       max_clause_freq=max_clause_freq)
            return result.get("count", 0)
        except Exception:
            return 0
    elif "cnf" in piece:
        # Multi-clause AND
        diff = max_diff_tokens or piece.get("max_diff_tokens") or 10
        try:
            kwargs = {"max_clause_freq": max_clause_freq}
            if piece.get("n_words", 1) > 1:
                kwargs["max_diff_tokens"] = diff
            result = engine.count_cnf(piece["cnf"], **kwargs)
            return result.get("count", 0)
        except Exception:
            return 0
    return 0


def _and_pieces(piece_a, piece_b, max_diff_tokens):
    """
    AND two pieces together. Returns a new CNF (list of clauses).

    piece_a can be a key_piece (has cnf_clause) or term_piece (has cnf).
    """
    clauses = []

    if "cnf_clause" in piece_a:
        clauses.append(piece_a["cnf_clause"])
    elif "cnf" in piece_a:
        clauses.extend(piece_a["cnf"])

    if "cnf_clause" in piece_b:
        clauses.append(piece_b["cnf_clause"])
    elif "cnf" in piece_b:
        clauses.extend(piece_b["cnf"])

    return clauses


def build_queries(
    pieces: dict,
    engine,
    tokenizer,
    max_key_standalone: int = 500,
    max_associated_standalone: int = 50,
    min_associated_count: int = 7,
    max_diff_tokens_tight: int = 10,
    max_diff_tokens_medium: int = 30,
    max_diff_tokens_wide: int = 80,
    max_query_count: int = 5000,
    max_total: int = 100,
    max_clause_freq: int = 80000000,
    verbose: bool = True,
):
    """
    Build queries iteratively, exhausting cheap grabs first.

    Order:
    1. KEY pieces standalone (< max_key_standalone)
    2. Associated pieces standalone (< max_associated_standalone)
    3. Top associated AND'd with remaining KEY pieces
    4. Cross-KEY AND combinations
    5. Conceptual/referential AND'd with KEY pieces
    6. Remaining associated AND'd with KEY pieces
    """
    key_pieces = pieces["key_pieces"]
    conceptual_pieces = pieces.get("conceptual_pieces", {})
    referential_pieces = pieces.get("referential_pieces", {})
    associated_pieces = pieces["associated_pieces"]
    verb_pieces = pieces.get("verb_pieces", [])

    queries = []
    seen = set()
    grabbed_keys = set()  # key aspect names already fully grabbed

    def _qkey(cnf, max_diff):
        return (tuple(tuple(tuple(a) for a in c) for c in cnf), max_diff)

    def _add(cnf, max_diff, description, level, count=None):
        key = _qkey(cnf, max_diff)
        if key in seen or len(queries) >= max_total:
            return False

        if count is None:
            try:
                kwargs = {"max_clause_freq": max_clause_freq}
                if max_diff:
                    kwargs["max_diff_tokens"] = max_diff
                result = engine.count_cnf(cnf, **kwargs)
                count = result.get("count", 0)
            except Exception:
                count = 0

        if count == 0:
            return False

        if count > max_query_count:
            if verbose:
                print(f"    {count:>8,d}  [{level}] {description} (SKIP: >{max_query_count:,d})")
            return False

        seen.add(key)
        queries.append({
            "type": "cnf",
            "cnf": cnf,
            "max_diff_tokens": max_diff,
            "description": description,
            "estimated_count": count,
            "level": level,
        })
        if verbose:
            print(f"    {count:>8,d}  [{level}] {description}")
        return True

    # ================================================================
    # Step 1: KEY pieces standalone
    # ================================================================
    if verbose:
        print(f"\n{'='*70}")
        print(f"Step 1: KEY pieces standalone (< {max_key_standalone})")
        print(f"{'='*70}")

    remaining_keys = {}
    for name, piece in key_pieces.items():
        count = _count_piece(piece, engine, max_clause_freq)
        if verbose:
            print(f"  {name}: {count:,d} hits")
        if count <= max_key_standalone and count > 0:
            _add([piece["cnf_clause"]], None, f"{name} (standalone)", "S1_key", count=count)
            grabbed_keys.add(name)
        else:
            remaining_keys[name] = (piece, count)

    # ================================================================
    # Step 2: Associated pieces standalone
    # ================================================================
    if verbose:
        print(f"\n{'='*70}")
        print(f"Step 2: Associated standalone (< {max_associated_standalone})")
        print(f"{'='*70}")

    remaining_associated = []
    for i, piece in enumerate(associated_pieces):
        count = _count_piece(piece, engine, max_clause_freq, max_diff_tokens=max_diff_tokens_tight)
        if count <= max_associated_standalone and count > 0:
            cnf = piece["cnf"] if "cnf" in piece else [piece["cnf_clause"]]
            diff = piece.get("max_diff_tokens", max_diff_tokens_tight)
            _add(cnf, diff, f"{piece['description']} (standalone)", "S2_assoc", count=count)
        else:
            remaining_associated.append((piece, count))

    # ================================================================
    # Step 3: Top associated AND'd with KEY pieces (tightest first)
    # ================================================================
    if verbose:
        print(f"\n{'='*70}")
        print(f"Step 3: Top {min_associated_count} associated AND KEY pieces")
        print(f"{'='*70}")

    # Sort remaining associated by count (most specific first)
    remaining_associated.sort(key=lambda x: x[1] if x[1] > 0 else float("inf"))
    top_associated = remaining_associated[:min_associated_count]
    still_remaining_associated = []

    key_names = list(remaining_keys.keys())

    for assoc_piece, assoc_count in top_associated:
        if len(queries) >= max_total:
            still_remaining_associated.append((assoc_piece, assoc_count))
            continue

        grabbed = False

        # 3a: Try associated AND ALL remaining key pieces
        if len(remaining_keys) >= 2:
            cnf = []
            for name in key_names:
                cnf.append(remaining_keys[name][0]["cnf_clause"])
            # Add associated piece
            if "cnf" in assoc_piece:
                cnf.extend(assoc_piece["cnf"])
            else:
                cnf.append(assoc_piece["cnf_clause"])

            all_desc = " AND ".join(f"({remaining_keys[n][0]['description']})" for n in key_names)
            desc = f"{all_desc} AND ({assoc_piece['description']})"

            try:
                kwargs = {"max_clause_freq": max_clause_freq, "max_diff_tokens": max_diff_tokens_tight}
                result = engine.count_cnf(cnf, **kwargs)
                cnt = result.get("count", 0)
            except Exception:
                cnt = 0

            if 0 < cnt <= max_query_count:
                _add(cnf, max_diff_tokens_tight, desc, "S3a_all_keys", count=cnt)
                grabbed = True

        # 3b: If too few or too many with all keys, try pairwise key AND associated
        if not grabbed:
            for key_name, (key_piece, key_count) in remaining_keys.items():
                if len(queries) >= max_total:
                    break
                cnf = _and_pieces(key_piece, assoc_piece, max_diff_tokens_medium)
                desc = f"({key_piece['description']}) AND ({assoc_piece['description']})"
                _add(cnf, max_diff_tokens_medium, desc, "S3b_one_key")

        if not grabbed:
            still_remaining_associated.append((assoc_piece, assoc_count))

    remaining_associated = still_remaining_associated + remaining_associated[min_associated_count:]

    # ================================================================
    # Step 4: Cross-KEY AND combinations
    # ================================================================
    if verbose:
        print(f"\n{'='*70}")
        print(f"Step 4: Cross-KEY AND combinations")
        print(f"{'='*70}")

    key_names = list(remaining_keys.keys())
    if len(key_names) >= 2:
        from itertools import combinations

        # All keys AND'd
        if len(key_names) >= 2:
            all_cnf = []
            for name in key_names:
                piece = remaining_keys[name][0]
                all_cnf.append(piece["cnf_clause"])
            desc = " AND ".join(f"({remaining_keys[n][0]['description']})" for n in key_names)
            _add(all_cnf, max_diff_tokens_tight, desc, "S4_all_keys")

        # Pairwise
        for a, b in combinations(key_names, 2):
            if len(queries) >= max_total:
                break
            cnf = _and_pieces(remaining_keys[a][0], remaining_keys[b][0], max_diff_tokens_medium)
            desc = f"({remaining_keys[a][0]['description']}) AND ({remaining_keys[b][0]['description']})"
            _add(cnf, max_diff_tokens_medium, desc, "S4_pair_keys")

    # ================================================================
    # Step 5: Conceptual/referential AND'd with KEY pieces
    # ================================================================
    if verbose:
        print(f"\n{'='*70}")
        print(f"Step 5: Conceptual/referential AND KEY pieces")
        print(f"{'='*70}")

    for key_name, (key_piece, key_count) in remaining_keys.items():
        # Conceptual
        for cp in conceptual_pieces.get(key_name, []):
            if len(queries) >= max_total:
                break
            cnf = _and_pieces(key_piece, cp, max_diff_tokens_wide)
            desc = f"({key_piece['description']}) AND ({cp['description']})"
            _add(cnf, max_diff_tokens_wide, desc, "S5_concept")

        # Referential
        for rp in referential_pieces.get(key_name, []):
            if len(queries) >= max_total:
                break
            cnf = _and_pieces(key_piece, rp, max_diff_tokens_wide)
            desc = f"({key_piece['description']}) AND ({rp['description']})"
            _add(cnf, max_diff_tokens_wide, desc, "S5_ref")

    # Also AND conceptual/referential with OTHER key pieces
    for key_name in key_names:
        key_piece = remaining_keys[key_name][0]
        for other_name in key_names:
            if other_name == key_name:
                continue
            for cp in conceptual_pieces.get(other_name, []):
                if len(queries) >= max_total:
                    break
                cnf = _and_pieces(key_piece, cp, max_diff_tokens_wide)
                desc = f"({key_piece['description']}) AND ({cp['description']})"
                _add(cnf, max_diff_tokens_wide, desc, "S5_cross")

            for rp in referential_pieces.get(other_name, []):
                if len(queries) >= max_total:
                    break
                cnf = _and_pieces(key_piece, rp, max_diff_tokens_wide)
                desc = f"({key_piece['description']}) AND ({rp['description']})"
                _add(cnf, max_diff_tokens_wide, desc, "S5_cross")

    # ================================================================
    # Step 6: Remaining associated AND'd with KEY pieces
    # ================================================================
    if verbose:
        print(f"\n{'='*70}")
        print(f"Step 6: Remaining associated AND KEY pieces")
        print(f"{'='*70}")

    for assoc_piece, assoc_count in remaining_associated:
        if len(queries) >= max_total:
            break
        for key_name, (key_piece, key_count) in remaining_keys.items():
            if len(queries) >= max_total:
                break
            cnf = _and_pieces(key_piece, assoc_piece, max_diff_tokens_wide)
            desc = f"({key_piece['description']}) AND ({assoc_piece['description']})"
            _add(cnf, max_diff_tokens_wide, desc, "S6_assoc")

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