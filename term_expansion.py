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
                "count": 0,
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

    # Peek count
    if len(cnf) == 1:
        # Single word — count is sum of all variant counts
        total_count = sum(v["count"] for v in word_data[0]["variants"])
        count = total_count
    else:
        # Multi-word — use proximity AND
        try:
            result = engine.count_cnf(
                cnf,
                max_clause_freq=80000000,
                max_diff_tokens=max_diff_tokens,
            )
            count = result.get("count", 0)
        except Exception as e:
            count = 0

    return {
        "original": keyword,
        "words": word_data,
        "cnf": cnf,
        "count": count,
        "n_words": len(word_data),
        "max_diff_tokens": max_diff_tokens if len(word_data) > 1 else None,
        "valid": count > 0,
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
    Expand all keywords for a query and peek their counts.

    Returns dict with:
        - key_terms: list of expanded terms from KEY groups
        - sup_terms: list of expanded terms from SUP groups
        - verb_terms: list of expanded verb terms
        - all_terms: flat list of all expanded terms
        - facets: original facet structure with expanded terms
    """
    facets = load_faceted_keywords(qid, expansions_path)
    core_facets = facets["core_facets"]
    aux_facets = facets["aux_facets"]
    verbs = facets.get("verbs", [])

    if verbose:
        print(f"\nExpanding keywords for {qid}")
        print(f"  KEY facets: {len(core_facets)}")
        print(f"  SUP facets: {len(aux_facets)}")
        print(f"  Verbs: {len(verbs)}")
        print(f"  max_diff_tokens: {max_diff_tokens}")

    key_terms = []
    sup_terms = []
    verb_terms = []

    # Key facets with structure preserved
    key_facets_expanded = {}
    for facet_name, keywords in core_facets.items():
        if verbose:
            print(f"\n  KEY: {facet_name}")
        expanded = []
        for kw in keywords:
            term = expand_term(kw, tokenizer, engine, max_diff_tokens)
            if term is None:
                continue
            expanded.append(term)
            key_terms.append(term)
            if verbose:
                _print_term(term)
        key_facets_expanded[facet_name] = expanded

    # Sup facets
    sup_facets_expanded = {}
    for facet_name, keywords in aux_facets.items():
        if verbose:
            print(f"\n  SUP: {facet_name}")
        expanded = []
        for kw in keywords:
            term = expand_term(kw, tokenizer, engine, max_diff_tokens)
            if term is None:
                continue
            expanded.append(term)
            sup_terms.append(term)
            if verbose:
                _print_term(term)
        sup_facets_expanded[facet_name] = expanded

    # Verbs
    if verbs:
        if verbose:
            print(f"\n  VERBS:")
        for v in verbs:
            term = expand_term(v, tokenizer, engine, max_diff_tokens)
            if term is None:
                continue
            verb_terms.append(term)
            if verbose:
                _print_term(term)

    all_terms = key_terms + sup_terms + verb_terms

    # Summary
    if verbose:
        n_valid = sum(1 for t in all_terms if t["valid"])
        n_invalid = sum(1 for t in all_terms if not t["valid"])
        print(f"\n{'='*60}")
        print(f"Summary: {len(all_terms)} terms ({n_valid} valid, {n_invalid} invalid)")
        print(f"  KEY: {len(key_terms)} terms")
        print(f"  SUP: {len(sup_terms)} terms")
        print(f"  VERBS: {len(verb_terms)} terms")

        # Show count distribution
        valid_counts = sorted([t["count"] for t in all_terms if t["valid"]])
        if valid_counts:
            print(f"  Count range: {valid_counts[0]:,d} - {valid_counts[-1]:,d}")
            print(f"  < 100: {sum(1 for c in valid_counts if c < 100)}")
            print(f"  100-1k: {sum(1 for c in valid_counts if 100 <= c < 1000)}")
            print(f"  1k-5k: {sum(1 for c in valid_counts if 1000 <= c < 5000)}")
            print(f"  5k-10k: {sum(1 for c in valid_counts if 5000 <= c < 10000)}")
            print(f"  10k+: {sum(1 for c in valid_counts if c >= 10000)}")
        print(f"{'='*60}")

    return {
        "key_terms": key_terms,
        "sup_terms": sup_terms,
        "verb_terms": verb_terms,
        "all_terms": all_terms,
        "key_facets": key_facets_expanded,
        "sup_facets": sup_facets_expanded,
    }


def _print_term(term):
    """Pretty print a single expanded term."""
    if not term["valid"]:
        print(f"    {0:>8,d}  {term['original']} (INVALID: '{term.get('invalid_word', '?')}' not in corpus)")
        return

    # Build description
    if term["n_words"] == 1:
        variants = term["words"][0]["variants"]
        var_str = " OR ".join(f"{v['text']}({v['count']:,d})" for v in variants)
        print(f"    {term['count']:>8,d}  {term['original']} -> ({var_str})")
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
        print(f"    {term['count']:>8,d}  {term['original']} -> {and_str}{prox}")