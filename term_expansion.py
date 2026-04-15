"""
Term expansion and count peeking for CNF query construction.

Step 1: Expand every keyword into a proximity-AND query with case variants.
Step 2: Peek counts for every expanded term.

Usage:
    from term_expansion import expand_and_peek

    terms = expand_and_peek(
        qid="2024-32912",
        expansions_path="./keyword_expansions.jsonl",
        tokenizer=tokenizer,
        engine=engine,
    )
"""

import json
from llm_keyword_filter import load_faceted_keywords, STOPWORDS


def _case_variants(word):
    """Generate case variants for a word: original, lower, Title."""
    variants = set()
    variants.add(word)
    variants.add(word.lower())
    if len(word) > 1:
        variants.add(word[0].upper() + word[1:].lower())  # Title case
    return list(variants)


def _encode_variants(word, tokenizer, engine):
    """
    Encode all case variants of a word, keep those with >0 hits.
    Returns list of (ids, count) pairs, or empty if none work.
    """
    variants = _case_variants(word)
    valid = []
    seen_ids = set()

    for v in variants:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if not ids:
            continue
        ids_key = tuple(ids)
        if ids_key in seen_ids:
            continue
        seen_ids.add(ids_key)
        count = engine.count(input_ids=ids).get("count", 0)
        if count > 0:
            valid.append({"text": v, "ids": ids, "count": count})

    return valid


def expand_term(keyword, tokenizer, engine, max_diff_tokens=5):
    """
    Expand a single keyword into a proximity-AND query with case variants.

    "Vietnam War" -> {
        "original": "Vietnam War",
        "words": [
            {"word": "Vietnam", "variants": [{"text": "Vietnam", "ids": [...], "count": 50000}, ...]},
            {"word": "War", "variants": [{"text": "War", ...}, {"text": "war", ...}]},
        ],
        "cnf": [[[ids_Vietnam], [ids_vietnam]], [[ids_War], [ids_war]]],
        "max_diff_tokens": 5,
        "count": 1234,  # proximity-AND count
        "n_words": 2,
        "valid": True,
    }

    "guns and butter" -> removes "and", keeps "guns" and "butter"

    Single word "economy" -> {
        "words": [{"word": "economy", "variants": [...]}],
        "cnf": [[[ids_economy], [ids_Economy]]],
        "count": 50000,
        "n_words": 1,
    }
    """
    if not keyword or not keyword.strip():
        return None

    # Split into words, remove stopwords
    raw_words = keyword.strip().split()
    words = [w for w in raw_words if w.lower() not in STOPWORDS and len(w) > 1]

    if not words:
        return None

    # Expand each word with case variants
    word_data = []
    for w in words:
        variants = _encode_variants(w, tokenizer, engine)
        if not variants:
            # This word has no hits in any case — term is invalid
            return {
                "original": keyword,
                "words": [],
                "cnf": [],
                "n_words": len(words),
                "valid": False,
                "invalid_word": w,
            }
        word_data.append({
            "word": w,
            "variants": variants,
        })

    # Build CNF: one clause per word, OR across case variants
    cnf = []
    for wd in word_data:
        clause = [v["ids"] for v in wd["variants"]]
        cnf.append(clause)

    return {
        "original": keyword,
        "words": word_data,
        "cnf": cnf,
        "n_words": len(word_data),
        "max_diff_tokens": max_diff_tokens if len(word_data) > 1 else None,
        "valid": True,
    }


def expand_and_peek(
    qid: str,
    expansions_path: str,
    tokenizer,
    engine,
    max_diff_tokens: int = 5,
    verbose: bool = True,
):
    """
    Expand all keywords for a query with case variants and proximity AND.

    Handles multiple keyword formats:
    - Old: KEY:/SUP: prefixed facets
    - New: KEY_ENTITIES with nested lexical/conceptual/referential + ASSOCIATED_TERMS + VERBS

    Returns dict with:
        - aspect_terms: dict of aspect_name -> list of expanded terms
        - associated_terms: list of expanded terms from ASSOCIATED
        - verb_terms: list of expanded verb terms
        - all_terms: flat list of all expanded terms
    """
    facets = load_faceted_keywords(qid, expansions_path)
    aspects = facets.get("aspects", facets.get("core_facets", {}))
    associated = facets.get("associated", [])
    verbs = facets.get("verbs", [])

    if verbose:
        print(f"\nExpanding keywords for {qid}")
        print(f"  Aspects: {len(aspects)}")
        print(f"  Associated: {len(associated)} terms")
        print(f"  Verbs: {len(verbs)}")
        print(f"  max_diff_tokens: {max_diff_tokens}")

    aspect_terms = {}
    associated_expanded = []
    verb_expanded = []
    all_terms = []

    # Expand aspects
    for aspect_name, keywords in aspects.items():
        if verbose:
            print(f"\n  ASPECT: {aspect_name}")
        expanded = []
        for kw in keywords:
            term = expand_term(kw, tokenizer, engine, max_diff_tokens)
            if term is None:
                continue
            expanded.append(term)
            all_terms.append(term)
            if verbose:
                _print_term(term)
        aspect_terms[aspect_name] = expanded

    # Expand associated terms
    if associated:
        if verbose:
            print(f"\n  ASSOCIATED:")
        for kw in associated:
            term = expand_term(kw, tokenizer, engine, max_diff_tokens)
            if term is None:
                continue
            associated_expanded.append(term)
            all_terms.append(term)
            if verbose:
                _print_term(term)

    # Expand verbs
    if verbs:
        if verbose:
            print(f"\n  VERBS:")
        for v in verbs:
            term = expand_term(v, tokenizer, engine, max_diff_tokens)
            if term is None:
                continue
            verb_expanded.append(term)
            all_terms.append(term)
            if verbose:
                _print_term(term)

    # Summary
    if verbose:
        n_valid = sum(1 for t in all_terms if t["valid"])
        n_invalid = sum(1 for t in all_terms if not t["valid"])
        print(f"\n{'='*60}")
        print(f"Summary: {len(all_terms)} terms ({n_valid} valid, {n_invalid} invalid)")
        for name, terms in aspect_terms.items():
            n_v = sum(1 for t in terms if t["valid"])
            print(f"  {name}: {len(terms)} terms ({n_v} valid)")
        print(f"  ASSOCIATED: {len(associated_expanded)} terms")
        print(f"  VERBS: {len(verb_expanded)} terms")
        print(f"{'='*60}")

    return {
        "aspect_terms": aspect_terms,
        "associated_terms": associated_expanded,
        "verb_terms": verb_expanded,
        "all_terms": all_terms,
        # Backward compat
        "key_terms": all_terms,
        "sup_terms": associated_expanded,
        "key_facets": aspect_terms,
        "sup_facets": {"associated": associated_expanded} if associated_expanded else {},
    }


def _print_term(term):
    """Pretty print a single expanded term."""
    if not term["valid"]:
        print(f"    INVALID  {term['original']} ('{term.get('invalid_word', '?')}' not in corpus)")
        return

    if term["n_words"] == 1:
        variants = term["words"][0]["variants"]
        var_str = " OR ".join(f"{v['text']}({v['count']:,d})" for v in variants)
        print(f"    {term['original']} -> ({var_str})")
    else:
        parts = []
        for wd in term["words"]:
            if len(wd["variants"]) == 1:
                parts.append(wd["variants"][0]["text"])
            else:
                var_str = "|".join(v["text"] for v in wd["variants"])
                parts.append(f"({var_str})")
        and_str = " AND ".join(parts)
        prox = f" [prox={term['max_diff_tokens']}]" if term.get("max_diff_tokens") else ""
        print(f"    {term['original']} -> {and_str}{prox}")