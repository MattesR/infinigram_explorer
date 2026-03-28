"""
Filter and validate LLM-generated keyword expansions for suffix array search.

Takes raw keywords from an LLM (e.g. Claude batch API), extracts searchable
noun phrases and content words via spaCy, validates against the index via
engine.count(), and returns a ranked list of validated search terms.

Usage:
    from keyword_filter import filter_and_validate_keywords, load_and_filter

    # From raw keyword list
    validated = filter_and_validate_keywords(
        keywords=["coping with vicarious trauma", "compassion fatigue", ...],
        tokenizer=tokenizer,
        engine=engine,
    )

    # From cached expansions file
    validated = load_and_filter(
        qid="2024-145979",
        expansions_path="keyword_expansions.json",
        tokenizer=tokenizer,
        engine=engine,
    )
"""

import json
import spacy

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_noun_phrases(keywords: list[str]) -> list[str]:
    """
    Extract searchable noun phrases and content words from LLM keywords.

    For each keyword string:
    - Extract noun chunks (multi-word noun phrases)
    - Strip determiners, pronouns, prepositions from chunks
    - If no noun chunks, keep individual nouns/adjectives/proper nouns
    - Deduplicate and sort by length (longer = more specific = better)

    Args:
        keywords: Raw keyword strings from LLM.

    Returns:
        Deduplicated list of noun phrases and content words,
        sorted by length descending (most specific first).
    """
    nlp = _get_nlp()
    filtered = set()

    for kw in keywords:
        if not kw or not kw.strip():
            continue

        doc = nlp(kw.strip())
        found_chunks = False

        # Extract noun chunks
        for chunk in doc.noun_chunks:
            tokens = [t for t in chunk if t.pos_ not in ("DET", "PRON", "ADP", "PUNCT")]
            if tokens:
                phrase = " ".join(t.text.lower() for t in tokens)
                if len(phrase) > 2:
                    filtered.add(phrase)
                    found_chunks = True

        # If no noun chunks, keep content words
        if not found_chunks:
            for token in doc:
                if token.pos_ in ("NOUN", "PROPN", "ADJ") and len(token.text) > 2:
                    filtered.add(token.text.lower())

    # Sort by length descending (more specific phrases first)
    return sorted(filtered, key=lambda x: len(x), reverse=True)


def validate_against_index(
    phrases: list[str],
    tokenizer,
    engine,
    max_count: int = None,
    verbose: bool = False,
) -> list[dict]:
    """
    Check each phrase against the index and return those with >0 hits.

    Args:
        phrases: List of keyword/phrase strings.
        tokenizer: Infini-gram tokenizer.
        engine: Infini-gram engine.
        max_count: If set, skip phrases with count above this (too broad).
        verbose: Print counts.

    Returns:
        List of dicts with 'phrase', 'input_ids', 'count', sorted by count ascending
        (most specific first).
    """
    validated = []

    for phrase in phrases:
        ids = tokenizer.encode(phrase, add_special_tokens=False)
        if not ids:
            continue

        count = engine.count(input_ids=ids).get("count", 0)

        if count == 0:
            if verbose:
                print(f"    {0:>10,d}  {phrase} (SKIP: not in corpus)")
            continue

        if max_count and count > max_count:
            if verbose:
                print(f"    {count:>10,d}  {phrase} (SKIP: too broad)")
            continue

        validated.append({
            "phrase": phrase,
            "input_ids": ids,
            "count": count,
        })

        if verbose:
            print(f"    {count:>10,d}  {phrase}")

    # Sort by count ascending (most specific first)
    validated.sort(key=lambda x: x["count"])
    return validated


def filter_and_validate_keywords(
    keywords: list[str],
    tokenizer,
    engine,
    max_count: int = 500000,
    verbose: bool = True,
) -> list[dict]:
    """
    Full pipeline: raw LLM keywords -> spaCy filtering -> index validation.

    Args:
        keywords: Raw keyword strings from LLM.
        tokenizer: Infini-gram tokenizer.
        engine: Infini-gram engine.
        max_count: Skip phrases above this count.
        verbose: Print intermediate results.

    Returns:
        List of validated keyword dicts with 'phrase', 'input_ids', 'count'.
    """
    if verbose:
        print(f"  Raw keywords: {len(keywords)}")

    # Step 1: Extract noun phrases
    phrases = extract_noun_phrases(keywords)
    if verbose:
        print(f"  After NP extraction: {len(phrases)}")

    # Step 2: Validate against index
    if verbose:
        print(f"  Validating against index (max_count={max_count:,d}):")
    validated = validate_against_index(
        phrases, tokenizer, engine,
        max_count=max_count, verbose=verbose,
    )
    if verbose:
        print(f"  Validated: {len(validated)} terms")

    return validated


def load_and_filter(
    qid: str,
    expansions_path: str,
    tokenizer,
    engine,
    max_count: int = 500000,
    use_core_only: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """
    Load keyword expansions for a query and filter+validate them.

    Args:
        qid: Query ID.
        expansions_path: Path to keyword_expansions.json.
        tokenizer: Infini-gram tokenizer.
        engine: Infini-gram engine.
        max_count: Skip phrases above this count.
        use_core_only: If True, only use core keywords (skip expansion).
        verbose: Print details.

    Returns:
        List of validated keyword dicts.
    """
    with open(expansions_path) as f:
        expansions = json.load(f)

    data = expansions.get(qid, {})
    if not data:
        if verbose:
            print(f"  No expansions found for {qid}")
        return []

    keywords = list(data.get("core", []))
    if not use_core_only:
        keywords.extend(data.get("expansion", []))

    if verbose:
        print(f"  Query {qid}: {len(data.get('core', []))} core + "
              f"{len(data.get('expansion', []))} expansion keywords")

    return filter_and_validate_keywords(
        keywords, tokenizer, engine,
        max_count=max_count, verbose=verbose,
    )


def load_all_expansions(expansions_path: str) -> dict:
    """Load the full expansions dict."""
    with open(expansions_path) as f:
        return json.load(f)