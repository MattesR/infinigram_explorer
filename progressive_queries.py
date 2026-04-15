"""
Progressive CNF query construction from expanded terms.

Starts tight (AND all aspects), loosens progressively.

Level 1: AND all aspects at tight proximity
Level 2: AND any N-1 aspects (drop one at a time)
Level 3: Each ASSOCIATED term AND'd with narrowest aspect
Level 4: Very specific terms standalone

Usage:
    from term_expansion import expand_and_peek
    from progressive_queries import build_progressive_queries

    terms = expand_and_peek(qid, expansions_path, tokenizer, engine)
    queries = build_progressive_queries(terms, engine, tokenizer)
"""

from itertools import combinations


def _aspect_or_clause(aspect_terms):
    """
    Build one OR clause from all valid terms in an aspect.
    Flattens all CNF clauses into one big OR.

    For single-word terms: OR all case variants
    For multi-word terms: can't OR them (they're ANDs), skip for OR clause

    Returns (or_ids, term_names) or (None, None) if no valid OR terms.
    """
    or_ids = []
    names = []
    seen = set()

    for term in aspect_terms:
        if not term.get("valid", False):
            continue

        if term["n_words"] == 1:
            # Single word — add all case variants to OR
            for v in term["words"][0]["variants"]:
                ids_key = tuple(v["ids"])
                if ids_key not in seen:
                    seen.add(ids_key)
                    or_ids.append(v["ids"])
            names.append(term["original"])
        else:
            # Multi-word — add original phrase as contiguous token IDs
            # This is an approximation — the exact phrase may not exist
            # but individual words as OR entries are too broad
            # So we add each word's variants as separate entries
            # Actually: we can't put ANDs inside ORs
            # Skip multi-word terms for OR clauses
            pass

    if not or_ids:
        return None, None
    return or_ids, names


def _aspect_or_clause_with_multiword(aspect_terms):
    """
    Build OR clause including multi-word terms by adding all
    individual word variants. This is broader but includes
    multi-word keywords in the OR.

    For multi-word terms, we add each word separately — this means
    the OR clause matches if ANY word from ANY term appears.
    Use this for the "descriptive" aspects where broad matching is ok.
    """
    or_ids = []
    names = []
    seen = set()

    for term in aspect_terms:
        if not term.get("valid", False):
            continue

        for wd in term["words"]:
            for v in wd["variants"]:
                ids_key = tuple(v["ids"])
                if ids_key not in seen:
                    seen.add(ids_key)
                    or_ids.append(v["ids"])

        names.append(term["original"])

    if not or_ids:
        return None, None
    return or_ids, names


def _term_to_cnf_clause(term):
    """
    Convert a single expanded term to a CNF clause (list of OR alternatives).
    For single words: [[ids_variant1, ids_variant2, ...]]
    For multi-word: [[word1_v1, word1_v2], [word2_v1, word2_v2]] (multiple clauses)
    """
    if not term.get("valid", False):
        return None
    return term["cnf"]


def _describe_or_clause(names, max_show=4):
    shown = names[:max_show]
    desc = " OR ".join(shown)
    if len(names) > max_show:
        desc += f" +{len(names) - max_show}"
    return f"({desc})"


def build_progressive_queries(
    terms: dict,
    engine,
    tokenizer,
    max_diff_tokens_tight: int = 20,
    max_diff_tokens_medium: int = 50,
    max_diff_tokens_wide: int = 100,
    max_standalone: int = 500,
    max_per_level: int = 20,
    max_total: int = 80,
    max_clause_freq: int = 80000000,
    verbose: bool = True,
):
    """
    Build queries progressively from tight to loose.

    Args:
        terms: Output from expand_and_peek().
        engine: Infini-gram engine.
        tokenizer: Infini-gram tokenizer.
        max_diff_tokens_tight: Proximity for Level 1 (all aspects).
        max_diff_tokens_medium: Proximity for Level 2 (N-1 aspects).
        max_diff_tokens_wide: Proximity for Level 3 (aspect + associated).
        max_standalone: Max count for Level 4 standalone grabs.
        max_per_level: Max queries per level.
        max_total: Max total queries.
        max_clause_freq: For engine.find_cnf/count_cnf.
        verbose: Print query construction.

    Returns:
        List of query dicts ready for execution.
    """
    aspect_terms = terms.get("aspect_terms", {})
    associated_terms = terms.get("associated_terms", [])
    verb_terms = terms.get("verb_terms", [])

    queries = []
    seen_keys = set()

    def _query_key(cnf, max_diff):
        """Hashable key for dedup."""
        return (tuple(tuple(tuple(alt) for alt in clause) for clause in cnf), max_diff)

    def _add_query(cnf, max_diff, description, level, count=None):
        key = _query_key(cnf, max_diff)
        if key in seen_keys:
            return False
        if len(queries) >= max_total:
            return False

        # Peek count if not provided
        if count is None:
            try:
                result = engine.count_cnf(
                    cnf,
                    max_clause_freq=max_clause_freq,
                    max_diff_tokens=max_diff,
                )
                count = result.get("count", 0)
            except Exception:
                count = 0

        if count == 0:
            if verbose:
                print(f"      0 hits: {description}")
            return False

        seen_keys.add(key)
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

    aspect_names = list(aspect_terms.keys())
    n_aspects = len(aspect_names)

    # Build per-aspect OR clauses (single-word variants only for tight matching)
    aspect_or = {}
    aspect_or_names = {}
    for name, terms_list in aspect_terms.items():
        or_ids, or_names = _aspect_or_clause_with_multiword(terms_list)
        if or_ids:
            aspect_or[name] = or_ids
            aspect_or_names[name] = or_names

    if verbose:
        print(f"\n{'='*70}")
        print(f"Building progressive queries")
        print(f"  Aspects: {aspect_names}")
        print(f"  Aspects with OR clauses: {list(aspect_or.keys())}")
        print(f"  Associated terms: {len(associated_terms)}")
        print(f"  Verb terms: {len(verb_terms)}")
        print(f"{'='*70}")

    # ================================================================
    # Level 1: AND all aspects at tight proximity
    # ================================================================
    if verbose:
        print(f"\nLevel 1: AND all aspects (prox={max_diff_tokens_tight})")

    if len(aspect_or) >= 2:
        # All aspects AND'd
        cnf = [aspect_or[name] for name in aspect_names if name in aspect_or]
        desc_parts = [_describe_or_clause(aspect_or_names[name])
                      for name in aspect_names if name in aspect_or]
        desc = " AND ".join(desc_parts)
        _add_query(cnf, max_diff_tokens_tight, desc, "L1_all")

    # ================================================================
    # Level 2: Drop one aspect at a time (N-1 combinations)
    # ================================================================
    if verbose:
        print(f"\nLevel 2: N-1 aspect combinations (prox={max_diff_tokens_medium})")

    if len(aspect_or) >= 3:
        for drop_name in aspect_names:
            if len(queries) >= max_total:
                break
            remaining = [n for n in aspect_names if n != drop_name and n in aspect_or]
            if len(remaining) < 2:
                continue

            cnf = [aspect_or[n] for n in remaining]
            desc_parts = [_describe_or_clause(aspect_or_names[n]) for n in remaining]
            desc = " AND ".join(desc_parts) + f" [drop: {drop_name}]"
            _add_query(cnf, max_diff_tokens_medium, desc, "L2_drop1")

    elif len(aspect_or) == 2:
        # Only 2 aspects — each one individually is already Level 2
        for name in aspect_names:
            if name not in aspect_or:
                continue
            if len(queries) >= max_total:
                break
            # Single aspect alone — skip, too broad usually

    # ================================================================
    # Level 3: ASSOCIATED terms AND'd with aspects
    # ================================================================
    if verbose:
        print(f"\nLevel 3: ASSOCIATED AND aspects (prox={max_diff_tokens_wide})")

    # Find the narrowest aspect to anchor with
    narrowest = None
    narrowest_size = float("inf")
    for name, or_ids in aspect_or.items():
        if len(or_ids) < narrowest_size:
            narrowest = name
            narrowest_size = len(or_ids)

    if narrowest and associated_terms:
        anchor_or = aspect_or[narrowest]
        anchor_desc = _describe_or_clause(aspect_or_names[narrowest])
        n_added = 0

        for term in associated_terms:
            if not term.get("valid", False) or n_added >= max_per_level:
                continue
            if len(queries) >= max_total:
                break

            # Build: anchor_aspect AND associated_term
            term_clauses = _term_to_cnf_clause(term)
            if term_clauses is None:
                continue

            cnf = [anchor_or] + term_clauses
            desc = f"{anchor_desc} AND ({term['original']})"
            _add_query(cnf, max_diff_tokens_wide, desc, "L3_assoc")
            n_added += 1

    # Also AND associated terms with each aspect
    for aspect_name in aspect_names:
        if aspect_name not in aspect_or or aspect_name == narrowest:
            continue
        a_or = aspect_or[aspect_name]
        a_desc = _describe_or_clause(aspect_or_names[aspect_name])
        n_added = 0

        for term in associated_terms:
            if not term.get("valid", False) or n_added >= max_per_level // 2:
                continue
            if len(queries) >= max_total:
                break

            term_clauses = _term_to_cnf_clause(term)
            if term_clauses is None:
                continue

            cnf = [a_or] + term_clauses
            desc = f"{a_desc} AND ({term['original']})"
            _add_query(cnf, max_diff_tokens_wide, desc, "L3_assoc")
            n_added += 1

    # ================================================================
    # Level 4: Very specific terms standalone
    # ================================================================
    if verbose:
        print(f"\nLevel 4: Specific terms standalone (count < {max_standalone})")

    all_terms_flat = terms.get("all_terms", [])
    for term in sorted(all_terms_flat, key=lambda t: t.get("count", 0) or float("inf")):
        if not term.get("valid", False):
            continue
        count = term.get("count", 0)
        if count is None or count > max_standalone or count == 0:
            continue
        if len(queries) >= max_total:
            break

        term_clauses = _term_to_cnf_clause(term)
        if term_clauses is None:
            continue

        # For single-word terms, use simple find
        if term["n_words"] == 1:
            # Add as CNF with single clause
            _add_query(
                term_clauses, max_diff_tokens_tight,
                f"{term['original']} (standalone)",
                "L4_standalone", count=count,
            )
        else:
            _add_query(
                term_clauses, term.get("max_diff_tokens", max_diff_tokens_tight),
                f"{term['original']} (standalone)",
                "L4_standalone", count=count,
            )

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