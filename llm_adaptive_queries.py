"""
LLM-keyword-driven adaptive query construction.

Uses faceted keyword expansions from an LLM (via batch_keywords.py)
combined with index count validation to build efficient queries.

Core facets are the main concepts — their terms get grabbed directly
if low-count, or AND'd across facets if high-count.

Aux facets narrow down high-count core terms.

Usage:
    from llm_adaptive_queries import build_llm_adaptive_queries

    queries, validated = build_llm_adaptive_queries(
        qid="2024-145979",
        expansions_path="keyword_expansions.jsonl",
        tokenizer=tokenizer,
        engine=engine,
    )
"""

import time
from tqdm import tqdm
from llm_keyword_filter import (
    load_faceted_keywords,
    extract_noun_phrases,
    validate_against_index,
)


def _encode(token: str, tokenizer) -> list[int]:
    return tokenizer.encode(token.lstrip("#"), add_special_tokens=False)


def _query_key(q):
    """Hashable key for deduplication."""
    if q["type"] == "simple":
        return ("simple", tuple(q["input_ids"]))
    else:
        cnf_key = tuple(
            tuple(tuple(alt) for alt in sorted(clause))
            for clause in sorted(q["cnf"], key=lambda c: str(c))
        )
        return ("cnf", cnf_key)


def build_llm_adaptive_queries(
    qid: str,
    expansions_path: str,
    tokenizer,
    engine,
    max_standalone: int = 10000,
    max_refined: int = 50000,
    max_count: int = 500000,
    max_queries: int = 50,
    max_clause_freq: int = 100000,
    use_core_only: bool = False,
    filter_mode: str = "stopword",
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Build adaptive queries from LLM faceted keyword expansions.

    Args:
        qid: Query ID.
        expansions_path: Path to keyword expansions JSONL.
        tokenizer: Infini-gram tokenizer.
        engine: Infini-gram engine.
        max_standalone: Grab directly if count below this.
        max_refined: Max count for AND-refined queries.
        max_count: Skip terms above this entirely.
        max_queries: Max total queries.
        max_clause_freq: For CNF sampling.
        use_core_only: Only use core facets.
        filter_mode: "stopword" or "noun_phrase".
        verbose: Print decisions.

    Returns:
        Tuple of (queries, all_validated_terms).
    """
    # Load faceted keywords
    facets = load_faceted_keywords(qid, expansions_path)
    core_facets = facets["core_facets"]
    aux_facets = facets["aux_facets"]

    if verbose:
        print(f"\nQuery {qid}")
        print(f"  Core facets: {len(core_facets)}")
        for name, terms in core_facets.items():
            print(f"    {name}: {len(terms)} terms")
        print(f"  Aux facets: {len(aux_facets)}")
        for name, terms in aux_facets.items():
            print(f"    {name}: {len(terms)} terms")

    # Validate all terms against index
    if verbose:
        print(f"\nValidating core facet terms...")

    core_validated = {}  # facet_name -> list of validated term dicts
    for name, terms in core_facets.items():
        if filter_mode == "noun_phrase":
            phrases = extract_noun_phrases(terms)
        else:
            from llm_keyword_filter import stopword_filter
            phrases = stopword_filter(terms)
        validated = validate_against_index(
            phrases, tokenizer, engine,
            max_count=max_count, verbose=False,
        )
        core_validated[name] = validated
        if verbose:
            n_valid = len(validated)
            counts = [v["count"] for v in validated]
            count_range = f"{min(counts):,d}-{max(counts):,d}" if counts else "none"
            print(f"    {name}: {len(terms)} raw -> {n_valid} valid ({count_range})")

    aux_validated = {}
    if not use_core_only:
        if verbose:
            print(f"\nValidating aux facet terms...")
        for name, terms in aux_facets.items():
            if filter_mode == "noun_phrase":
                phrases = extract_noun_phrases(terms)
            else:
                from llm_keyword_filter import stopword_filter
                phrases = stopword_filter(terms)
            validated = validate_against_index(
                phrases, tokenizer, engine,
                max_count=max_count, verbose=False,
            )
            aux_validated[name] = validated
            if verbose:
                n_valid = len(validated)
                counts = [v["count"] for v in validated]
                count_range = f"{min(counts):,d}-{max(counts):,d}" if counts else "none"
                print(f"    {name}: {len(terms)} raw -> {n_valid} valid ({count_range})")

    # Collect all validated terms for crude scoring later
    all_validated = []
    for terms in list(core_validated.values()) + list(aux_validated.values()):
        all_validated.extend(terms)

    # Build queries
    queries = []
    seen_keys = set()

    def _add(q):
        key = _query_key(q)
        if key not in seen_keys:
            seen_keys.add(key)
            queries.append(q)
            return True
        return False

    if verbose:
        print(f"\n{'='*60}")
        print(f"Building queries")
        print(f"{'='*60}")

    # Phase 1: Direct grabs from core facets
    # Low-count terms get grabbed directly
    if verbose:
        print(f"\nPhase 1: Direct grabs (count < {max_standalone:,d})")

    for name, validated in core_validated.items():
        for term in validated:
            if term["count"] <= max_standalone:
                _add({
                    "type": "simple",
                    "input_ids": term["input_ids"],
                    "description": term["phrase"],
                    "score": max_standalone - term["count"],  # prefer specific
                    "estimated_count": term["count"],
                    "facet": name,
                })
                if verbose:
                    print(f"    DIRECT: {term['phrase']} ({term['count']:,d})")

    # Phase 2: Cross-facet AND for high-count core terms
    # If a core facet has only high-count terms, AND with terms from other core facets
    if verbose:
        print(f"\nPhase 2: Cross-facet AND for high-count core terms")

    core_names = list(core_validated.keys())
    for i, name_a in enumerate(core_names):
        high_a = [t for t in core_validated[name_a] if t["count"] > max_standalone]
        if not high_a:
            continue

        for j, name_b in enumerate(core_names):
            if i == j:
                continue

            terms_b = core_validated[name_b]
            if not terms_b:
                continue

            # Try each high-count term from A with each term from B
            for ta in high_a:
                for tb in terms_b:
                    if len(queries) >= max_queries:
                        break

                    cnf = [[ta["input_ids"]], [tb["input_ids"]]]

                    # Estimate count (cheaper than actually counting)
                    # Use min of the two as upper bound
                    est = min(ta["count"], tb["count"])

                    if est > max_refined:
                        continue

                    # Actually count
                    kwargs = {"cnf": cnf}
                    if max_clause_freq:
                        kwargs["max_clause_freq"] = max_clause_freq
                    result = engine.find_cnf(**kwargs)
                    cnt = result.get("cnt", 0)

                    if 0 < cnt <= max_refined:
                        desc = f"({ta['phrase']}) AND ({tb['phrase']})"
                        _add({
                            "type": "cnf",
                            "cnf": cnf,
                            "description": desc,
                            "score": (max_standalone - min(ta["count"], max_standalone)) +
                                     (max_standalone - min(tb["count"], max_standalone)),
                            "estimated_count": cnt,
                            "facets": [name_a, name_b],
                        })
                        if verbose:
                            print(f"    CNF: {desc} ({cnt:,d})")

    # Phase 3: Use OR clauses within facets for broader coverage
    # Build one OR clause per core facet (low+medium count terms), AND across facets
    if verbose:
        print(f"\nPhase 3: Facet OR clauses AND'd across facets")

    core_or_clauses = {}  # facet_name -> (or_ids, names)
    for name, validated in core_validated.items():
        or_ids = []
        or_names = []
        for term in validated:
            if term["count"] <= max_refined:
                or_ids.append(term["input_ids"])
                or_names.append(term["phrase"])
        if or_ids:
            core_or_clauses[name] = (or_ids, or_names)

    if len(core_or_clauses) >= 2:
        from itertools import combinations
        for (name_a, (or_a, names_a)), (name_b, (or_b, names_b)) in combinations(core_or_clauses.items(), 2):
            if len(queries) >= max_queries:
                break

            cnf = [or_a, or_b]
            kwargs = {"cnf": cnf}
            if max_clause_freq:
                kwargs["max_clause_freq"] = max_clause_freq
            result = engine.find_cnf(**kwargs)
            cnt = result.get("cnt", 0)

            if cnt > 0:
                a_str = " OR ".join(names_a[:5])
                if len(names_a) > 5:
                    a_str += f" +{len(names_a)-5}"
                b_str = " OR ".join(names_b[:5])
                if len(names_b) > 5:
                    b_str += f" +{len(names_b)-5}"
                desc = f"({a_str}) AND ({b_str})"

                _add({
                    "type": "cnf",
                    "cnf": cnf,
                    "description": desc,
                    "score": sum(max_standalone - min(t["count"], max_standalone)
                                for t in core_validated[name_a]) +
                            sum(max_standalone - min(t["count"], max_standalone)
                                for t in core_validated[name_b]),
                    "estimated_count": cnt,
                    "facets": [name_a, name_b],
                })
                if verbose:
                    print(f"    FACET OR: {desc} ({cnt:,d})")

    # Phase 4: Aux facets AND'd with high-count core terms
    if not use_core_only and aux_validated:
        if verbose:
            print(f"\nPhase 4: Aux facets narrowing high-count core terms")

        for core_name, core_terms in core_validated.items():
            high_core = [t for t in core_terms if t["count"] > max_standalone]
            if not high_core:
                continue

            for aux_name, aux_terms in aux_validated.items():
                if len(queries) >= max_queries:
                    break

                for ct in high_core[:3]:  # top 3 high-count core terms
                    for at in aux_terms[:5]:  # top 5 aux terms
                        if len(queries) >= max_queries:
                            break

                        cnf = [[ct["input_ids"]], [at["input_ids"]]]
                        est = min(ct["count"], at["count"])
                        if est > max_refined:
                            continue

                        kwargs = {"cnf": cnf}
                        if max_clause_freq:
                            kwargs["max_clause_freq"] = max_clause_freq
                        result = engine.find_cnf(**kwargs)
                        cnt = result.get("cnt", 0)

                        if 0 < cnt <= max_refined:
                            desc = f"({ct['phrase']}) AND ({at['phrase']})"
                            _add({
                                "type": "cnf",
                                "cnf": cnf,
                                "description": desc,
                                "score": max_standalone - min(at["count"], max_standalone),
                                "estimated_count": cnt,
                                "facets": [core_name, aux_name],
                            })
                            if verbose:
                                print(f"    AUX: {desc} ({cnt:,d})")

    # Sort and truncate
    queries.sort(key=lambda x: x["score"], reverse=True)
    queries = queries[:max_queries]

    # Summary
    if verbose:
        n_simple = sum(1 for q in queries if q["type"] == "simple")
        n_cnf = sum(1 for q in queries if q["type"] == "cnf")
        total_est = sum(q["estimated_count"] for q in queries)
        print(f"\n{'='*60}")
        print(f"Summary: {len(queries)} queries ({n_simple} FIND, {n_cnf} CNF)")
        print(f"  ~{total_est:,d} estimated total docs")
        print(f"{'='*60}")
        for i, q in enumerate(queries):
            marker = "FIND" if q["type"] == "simple" else "CNF "
            print(f"  {i+1:3d}. [{marker}] [{q['score']:8.0f}] "
                  f"~{q['estimated_count']:>8,d}  {q['description']}")

    return queries, all_validated


def run_llm_adaptive(
    engine,
    queries: list[dict],
    max_clause_freq: int = 100000,
    min_retrieved_docs: int = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Execute queries. Handles both 'simple' and 'cnf' types.
    Same as run_adaptive but imported here for convenience.
    """
    from adaptive_queries import run_adaptive
    return run_adaptive(
        engine, queries,
        max_clause_freq=max_clause_freq,
        min_retrieved_docs=min_retrieved_docs,
        verbose=verbose,
    )