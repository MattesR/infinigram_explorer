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
    phrases,
    tokenizer,
    engine,
    max_count: int = None,
    max_standalone: int = 5000,
    verbose: bool = False,
) -> list[dict]:
    """
    Validate phrases against the index with a cascade of fallbacks:

    1. Try the full phrase as-is (preserving case)
    2. Try the full phrase lowercased
    3. If both cases work, create OR (case_variants)
    4. If phrase has 0 hits, split on stopwords into chunks
    5. For each chunk: try as-is, try lowercase, try both
    6. If any chunk has 0 hits, skip entire phrase
    7. If all chunks valid, create AND query (cnf)

    Single-word chunks from stopword splits get standalone=False.

    Returns:
        List of validated term dicts, sorted by count ascending.
    """
    validated = []
    seen = set()

    def _has_capital(s):
        return any(c.isupper() for c in s)

    def _try_encode(text):
        """Encode and count a phrase. Returns (ids, count) or None."""
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return None
        count = engine.count(input_ids=ids).get("count", 0)
        return (ids, count) if count > 0 else None

    def _try_with_case(phrase):
        """Try original case and lowercase. Returns best result + metadata."""
        r_orig = _try_encode(phrase)
        r_lower = _try_encode(phrase.lower()) if _has_capital(phrase) else None

        if r_orig and r_lower and r_orig[0] != r_lower[0]:
            # Both cases work with different tokenizations — OR them
            total = r_orig[1] + r_lower[1]
            return {
                "ids": r_orig[0],
                "ids_alt": r_lower[0],
                "count": total,
                "case_variants": True,
                "display": f"{phrase} (OR {phrase.lower()})",
            }
        elif r_orig:
            return {"ids": r_orig[0], "count": r_orig[1], "display": phrase}
        elif r_lower:
            return {"ids": r_lower[0], "count": r_lower[1], "display": phrase.lower()}
        return None

    for item in phrases:
        if isinstance(item, dict):
            phrase = item["phrase"]
            standalone = item.get("standalone", True)
        else:
            phrase = item
            standalone = True

        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)

        # Step 1: Try full phrase (with case variants)
        result = _try_with_case(phrase)

        if result:
            count = result["count"]
            if max_count and count > max_count:
                if verbose:
                    print(f"    {count:>10,d}  {result['display']} (SKIP: too broad)")
                continue

            entry = {
                "phrase": phrase,
                "input_ids": result["ids"],
                "count": count,
                "standalone": standalone,
            }
            if result.get("case_variants"):
                entry["input_ids_alt"] = result["ids_alt"]
                entry["case_variants"] = True
            if verbose:
                marker = "" if standalone else " [AND-only]"
                print(f"    {count:>10,d}  {result['display']}{marker}")
            validated.append(entry)
            continue

        # Step 2: Full phrase has 0 hits — split on stopwords
        words = phrase.split()
        if len(words) < 2:
            if verbose:
                print(f"    {0:>10,d}  {phrase} (SKIP: not in corpus)")
            continue

        # Split into chunks on stopwords
        chunks = []
        current_chunk = []
        for w in words:
            if w.lower() in STOPWORDS:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
            else:
                current_chunk.append(w)
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        if not chunks:
            if verbose:
                print(f"    {0:>10,d}  {phrase} (SKIP: all stopwords)")
            continue

        # If only one chunk and it's the same as original (no stopwords were found),
        # the phrase simply doesn't exist in corpus
        if len(chunks) == 1 and chunks[0].lower() == phrase.lower():
            # Try AND-ing individual words as last resort
            chunks = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]
            if len(chunks) < 2:
                if verbose:
                    print(f"    {0:>10,d}  {phrase} (SKIP: not in corpus)")
                continue

        # Step 3: Validate each chunk with case variants
        chunk_results = []
        all_valid = True
        for chunk in chunks:
            cr = _try_with_case(chunk)
            if cr is None:
                all_valid = False
                if verbose:
                    print(f"    {0:>10,d}  {phrase} (SKIP: chunk '{chunk}' not in corpus)")
                break
            chunk_results.append(cr)

        if not all_valid or len(chunk_results) < 2:
            continue

        # Step 4: Build CNF from chunks
        cnf_clauses = []
        chunk_names = []
        for cr in chunk_results:
            if cr.get("case_variants"):
                cnf_clauses.append([cr["ids"], cr["ids_alt"]])
            else:
                cnf_clauses.append([cr["ids"]])
            chunk_names.append(cr["display"])

        try:
            result = engine.find_cnf(cnf=cnf_clauses)
            and_count = result.get("cnt", 0)
        except Exception:
            and_count = 0

        if and_count > 0 and and_count <= max_standalone:
            and_desc = " AND ".join(chunk_names)
            phrase_ids = tokenizer.encode(phrase, add_special_tokens=False)

            # Determine standalone: single-word chunks from splits are AND-only
            chunk_standalone = standalone
            if len(chunks) > 1:
                single_word_chunks = [c for c in chunks if " " not in c]
                if len(single_word_chunks) == len(chunks):
                    # All chunks are single words — mark AND-only
                    chunk_standalone = False

            validated.append({
                "phrase": phrase,
                "input_ids": phrase_ids,
                "count": and_count,
                "standalone": chunk_standalone,
                "cnf": cnf_clauses,
                "description": and_desc,
            })
            if verbose:
                marker = "" if chunk_standalone else " [AND-only]"
                print(f"    {and_count:>10,d}  {phrase} -> AND({and_desc}){marker}")
        elif verbose:
            and_desc = " AND ".join(chunk_names)
            if and_count > max_standalone:
                print(f"    {and_count:>10,d}  {phrase} -> AND({and_desc}) too broad")
            else:
                print(f"    {0:>10,d}  {phrase} (SKIP: AND has 0 hits)")

    validated.sort(key=lambda x: x["count"])
    return validated


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall", "it",
    "its", "this", "that", "these", "those", "what", "which", "who",
    "whom", "how", "when", "where", "why", "not", "no", "nor", "if",
    "then", "than", "so", "up", "out", "about", "into", "through",
    "during", "before", "after", "above", "below", "between", "under",
    "again", "further", "once", "here", "there", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "only",
    "own", "same", "too", "very", "just", "also",
}


def stopword_filter(keywords: list[str]) -> list[dict]:
    """
    Prepare keyword phrases for index validation.
    Preserves original case and keeps phrases intact.

    Returns each keyword as-is. The validate_against_index function
    handles trying full phrase, case variants, stopword splitting,
    and AND fallback.

    Deduplicates case-insensitively.

    Args:
        keywords: Raw keyword strings from LLM.

    Returns:
        Deduplicated list of dicts with 'phrase' and 'standalone' flag.
    """
    seen = set()
    results = []

    for kw in keywords:
        if not kw or not kw.strip():
            continue
        phrase = kw.strip()
        key = phrase.lower()
        if key in seen or len(phrase) <= 2:
            continue
        seen.add(key)
        results.append({"phrase": phrase, "standalone": True})

    # Sort by length descending (more specific first)
    results.sort(key=lambda x: len(x["phrase"]), reverse=True)
    return results


def filter_and_validate_keywords(
    keywords: list[str],
    tokenizer,
    engine,
    max_count: int = 500000,
    filter_mode: str = "stopword",
    verbose: bool = True,
) -> list[dict]:
    """
    Full pipeline: raw LLM keywords -> filtering -> index validation.

    Args:
        keywords: Raw keyword strings from LLM.
        tokenizer: Infini-gram tokenizer.
        engine: Infini-gram engine.
        max_count: Skip phrases above this count.
        filter_mode: "stopword" (keep all content words) or "noun_phrase" (spaCy NP extraction).
        verbose: Print intermediate results.

    Returns:
        List of validated keyword dicts with 'phrase', 'input_ids', 'count'.
    """
    if verbose:
        print(f"  Raw keywords: {len(keywords)}")

    # Step 1: Filter
    if filter_mode == "noun_phrase":
        phrases = extract_noun_phrases(keywords)
        if verbose:
            print(f"  After NP extraction: {len(phrases)}")
    else:
        phrases = stopword_filter(keywords)
        if verbose:
            print(f"  After stopword filter: {len(phrases)}")

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
    filter_mode: str = "stopword",
    verbose: bool = True,
) -> list[dict]:
    """
    Load keyword expansions for a query and filter+validate them.

    Supports two formats:
    - Flat: {"core": [...], "expansion": [...]}
    - Faceted: {"CORE: concept": [...], "AUX: aspect": [...], ...}

    Args:
        qid: Query ID.
        expansions_path: Path to expansions file (JSON or JSONL).
        tokenizer: Infini-gram tokenizer.
        engine: Infini-gram engine.
        max_count: Skip phrases above this count.
        use_core_only: If True, only use CORE facets (skip AUX).
        verbose: Print details.

    Returns:
        List of validated keyword dicts.
    """
    expansions = load_all_expansions(expansions_path)

    data = expansions.get(qid, {})
    if not data:
        if verbose:
            print(f"  No expansions found for {qid}")
        return []

    # Detect format and extract keywords
    keywords = []
    n_core = 0
    n_aux = 0

    if "core" in data and isinstance(data["core"], list):
        # Flat format: {"core": [...], "expansion": [...]}
        keywords.extend(data["core"])
        n_core = len(data["core"])
        if not use_core_only:
            keywords.extend(data.get("expansion", []))
            n_aux = len(data.get("expansion", []))
    else:
        # Faceted format: {"CORE/KEY: x": [...], "AUX/SUP: y": [...], "VERBS": [...]}
        for key, values in data.items():
            if not isinstance(values, list):
                continue
            upper = key.upper()
            is_core = upper.startswith("CORE") or upper.startswith("KEY")
            is_verb = upper == "VERBS"
            if is_core:
                keywords.extend(values)
                n_core += len(values)
            elif not use_core_only:
                keywords.extend(values)
                n_aux += len(values)

    if verbose:
        print(f"  Query {qid}: {n_core} core + {n_aux} aux keywords")

    return filter_and_validate_keywords(
        keywords, tokenizer, engine,
        max_count=max_count, filter_mode=filter_mode, verbose=verbose,
    )


def load_faceted_keywords(qid: str, expansions_path: str) -> dict:
    """
    Load faceted keywords preserving the facet structure.

    Supports prefixes: CORE/KEY (-> core_facets), AUX/SUP (-> aux_facets), VERBS.

    Returns:
        Dict with 'core_facets', 'aux_facets', and 'verbs' (list).
    """
    expansions = load_all_expansions(expansions_path)
    data = expansions.get(qid, {})

    core_facets = {}
    aux_facets = {}
    verbs = []

    for key, values in data.items():
        if not isinstance(values, list):
            continue
        upper = key.upper()
        # Strip prefix for clean facet name
        if upper.startswith("CORE:") or upper.startswith("KEY:"):
            clean = key.split(":", 1)[1].strip()
            core_facets[clean] = values
        elif upper.startswith("AUX:") or upper.startswith("SUP:"):
            clean = key.split(":", 1)[1].strip()
            aux_facets[clean] = values
        elif upper == "VERBS":
            verbs = values
        else:
            # No recognized prefix — treat as aux
            aux_facets[key] = values

    return {"core_facets": core_facets, "aux_facets": aux_facets, "verbs": verbs}


def load_all_expansions(expansions_path: str) -> dict:
    """
    Load keyword expansions from JSON or JSONL file.

    JSON: {"qid": {...}, ...}
    JSONL: one {"qid": "...", ...} per line

    Returns:
        Dict mapping qid -> keyword data dict.
    """
    if expansions_path.endswith(".jsonl"):
        expansions = {}
        with open(expansions_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                qid = entry.pop("qid")
                expansions[qid] = entry
        return expansions
    else:
        with open(expansions_path) as f:
            return json.load(f)