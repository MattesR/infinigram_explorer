"""
LLM-keyword-driven adaptive query construction.

Clean 3-phase approach:
  Phase 1: Grab all validated terms (KEY + SUP) below max_standalone directly.
  Phase 2: For remaining high-count terms, AND with KEY OR-clause.
  Phase 3: OR across facets AND'd across facets for broad coverage.

No splitting phrases into words. No redundant AND queries for already-grabbed terms.

Usage:
    from llm_adaptive_queries import build_llm_adaptive_queries, run_llm_adaptive
"""

import time
from tqdm import tqdm
from llm_keyword_filter import (
    load_faceted_keywords,
    extract_noun_phrases,
    stopword_filter,
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
    max_standalone: int = 5000,
    max_standalone_sup: int = 1000,
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

    Phase 1: Grab KEY terms below max_standalone and SUP terms below
             max_standalone_sup directly.
    Phase 2: AND remaining high-count terms with KEY OR-clause
             (threshold: max_standalone).
    Phase 3: OR within facets, AND across facets.

    Returns:
        Tuple of (queries, all_validated_terms).
    """
    # Load faceted keywords
    facets = load_faceted_keywords(qid, expansions_path)
    core_facets = facets["core_facets"]
    aux_facets = facets["aux_facets"]
    verbs = facets.get("verbs", [])

    if verbose:
        print(f"\nQuery {qid}")
        print(f"  Core facets: {len(core_facets)}")
        for name, terms in core_facets.items():
            print(f"    {name}: {len(terms)} terms")
        print(f"  Aux facets: {len(aux_facets)}")
        for name, terms in aux_facets.items():
            print(f"    {name}: {len(terms)} terms")
        if verbs:
            print(f"  Verbs: {verbs}")

    # Choose filter
    def _filter(terms):
        if filter_mode == "noun_phrase":
            return extract_noun_phrases(terms)
        else:
            return stopword_filter(terms)

    # Validate all facets
    if verbose:
        print(f"\nValidating terms (filter_mode={filter_mode})...")

    core_validated = {}
    for name, terms in core_facets.items():
        phrases = _filter(terms)
        validated = validate_against_index(
            phrases, tokenizer, engine,
            max_count=max_count, verbose=False,
        )
        core_validated[name] = validated
        if verbose:
            n = len(validated)
            counts = [v["count"] for v in validated]
            cr = f"{min(counts):,d}-{max(counts):,d}" if counts else "none"
            print(f"    KEY {name}: {len(terms)} raw -> {n} valid ({cr})")

    aux_validated = {}
    if not use_core_only:
        for name, terms in aux_facets.items():
            phrases = _filter(terms)
            validated = validate_against_index(
                phrases, tokenizer, engine,
                max_count=max_count, verbose=False,
            )
            aux_validated[name] = validated
            if verbose:
                n = len(validated)
                counts = [v["count"] for v in validated]
                cr = f"{min(counts):,d}-{max(counts):,d}" if counts else "none"
                print(f"    SUP {name}: {len(terms)} raw -> {n} valid ({cr})")

    # Collect all validated for scoring
    all_validated = []
    for terms in list(core_validated.values()) + list(aux_validated.values()):
        all_validated.extend(terms)

    # Build queries
    queries = []
    seen_keys = set()
    grabbed_terms = set()  # phrases already grabbed directly

    def _add(q):
        key = _query_key(q)
        if key not in seen_keys:
            seen_keys.add(key)
            queries.append(q)
            return True
        return False

    # ================================================================
    # Phase 1: Direct grabs for all low-count terms (KEY + SUP)
    # ================================================================
    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase 1: Direct grabs (KEY < {max_standalone:,d}, SUP < {max_standalone_sup:,d})")
        print(f"{'='*60}")

    # Grab KEY terms (only standalone ones)
    for facet_name, validated in core_validated.items():
        for term in validated:
            if not term.get("standalone", True):
                continue  # AND-only terms skip direct grab
            if term["count"] <= max_standalone and term["count"] > 0:
                if "cnf" in term:
                    # AND fallback — use CNF query
                    desc = term.get("description", term["phrase"])
                    _add({
                        "type": "cnf",
                        "cnf": term["cnf"],
                        "description": desc,
                        "score": max_standalone - term["count"],
                        "estimated_count": term["count"],
                        "facet": facet_name,
                    })
                    if verbose:
                        print(f"    KEY AND: {desc} ({term['count']:,d})")
                else:
                    _add({
                        "type": "simple",
                        "input_ids": term["input_ids"],
                        "description": term["phrase"],
                        "score": max_standalone - term["count"],
                        "estimated_count": term["count"],
                        "facet": facet_name,
                    })
                    if verbose:
                        print(f"    KEY DIRECT: {term['phrase']} ({term['count']:,d})")
                grabbed_terms.add(term["phrase"])

    # Grab SUP terms (only standalone ones)
    for facet_name, validated in aux_validated.items():
        for term in validated:
            if not term.get("standalone", True):
                continue  # AND-only terms skip direct grab
            if term["count"] <= max_standalone_sup and term["count"] > 0:
                if "cnf" in term:
                    desc = term.get("description", term["phrase"])
                    _add({
                        "type": "cnf",
                        "cnf": term["cnf"],
                        "description": desc,
                        "score": max_standalone_sup - term["count"],
                        "estimated_count": term["count"],
                        "facet": facet_name,
                    })
                    if verbose:
                        print(f"    SUP AND: {desc} ({term['count']:,d})")
                else:
                    _add({
                        "type": "simple",
                        "input_ids": term["input_ids"],
                        "description": term["phrase"],
                        "score": max_standalone_sup - term["count"],
                        "estimated_count": term["count"],
                        "facet": facet_name,
                    })
                    if verbose:
                        print(f"    SUP DIRECT: {term['phrase']} ({term['count']:,d})")
                grabbed_terms.add(term["phrase"])

    # ================================================================
    # Phase 2: AND high-count terms with KEY OR-clause
    # ================================================================
    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase 2: AND high-count terms with KEY OR-clause")
        print(f"{'='*60}")

    # Build the KEY OR-clause from all core validated terms
    key_or_ids = []
    key_or_names = []
    for facet_name, validated in core_validated.items():
        for term in validated:
            ids = term["input_ids"]
            if ids and ids not in key_or_ids:
                key_or_ids.append(ids)
                key_or_names.append(term["phrase"])

    if key_or_ids and verbose:
        print(f"  KEY OR-clause: ({' OR '.join(key_or_names[:5])}"
              f"{'...' if len(key_or_names) > 5 else ''})")

    # Find high-count terms not yet grabbed
    high_count_terms = []
    for facet_name, validated in core_validated.items():
        for term in validated:
            if term["phrase"] not in grabbed_terms and term["count"] > max_standalone:
                high_count_terms.append(term)
    for facet_name, validated in aux_validated.items():
        for term in validated:
            if term["phrase"] not in grabbed_terms and term["count"] > max_standalone_sup:
                high_count_terms.append(term)

    for term in high_count_terms:
        if len(queries) >= max_queries:
            break

        if not key_or_ids:
            continue

        cnf = [key_or_ids, [term["input_ids"]]]

        # Count
        kwargs = {"cnf": cnf}
        if max_clause_freq:
            kwargs["max_clause_freq"] = max_clause_freq
        result = engine.find_cnf(**kwargs)
        cnt = result.get("cnt", 0)

        if cnt > 0 and cnt <= max_refined:
            key_str = " OR ".join(key_or_names[:5])
            if len(key_or_names) > 5:
                key_str += f" +{len(key_or_names)-5}"
            desc = f"({key_str}) AND ({term['phrase']})"
            _add({
                "type": "cnf",
                "cnf": cnf,
                "description": desc,
                "score": max_standalone,
                "estimated_count": cnt,
            })
            if verbose:
                print(f"    AND: ({term['phrase']}) -> {cnt:,d} hits")
        elif verbose and cnt > max_refined:
            print(f"    SKIP: ({term['phrase']}) -> {cnt:,d} hits (too broad)")

    # ================================================================
    # Phase 3: OR within facets, AND across facets
    # ================================================================
    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase 3: Facet OR-clauses AND'd across facets")
        print(f"{'='*60}")

    # Build OR clause per facet (using all validated terms, not just ungrabbed)
    facet_or_clauses = {}
    for facet_name, validated in list(core_validated.items()) + list(aux_validated.items()):
        or_ids = []
        or_names = []
        for term in validated:
            if term["count"] <= max_refined:
                or_ids.append(term["input_ids"])
                or_names.append(term["phrase"])
        if or_ids:
            facet_or_clauses[facet_name] = (or_ids, or_names)

    # AND core facets with each other
    from itertools import combinations
    core_facet_names = list(core_validated.keys())

    for name_a, name_b in combinations(core_facet_names, 2):
        if len(queries) >= max_queries:
            break
        if name_a not in facet_or_clauses or name_b not in facet_or_clauses:
            continue

        or_a, names_a = facet_or_clauses[name_a]
        or_b, names_b = facet_or_clauses[name_b]

        cnf = [or_a, or_b]
        kwargs = {"cnf": cnf}
        if max_clause_freq:
            kwargs["max_clause_freq"] = max_clause_freq
        result = engine.find_cnf(**kwargs)
        cnt = result.get("cnt", 0)

        if cnt > 0:
            a_str = " OR ".join(names_a[:4])
            if len(names_a) > 4:
                a_str += f" +{len(names_a)-4}"
            b_str = " OR ".join(names_b[:4])
            if len(names_b) > 4:
                b_str += f" +{len(names_b)-4}"
            desc = f"({a_str}) AND ({b_str})"
            _add({
                "type": "cnf",
                "cnf": cnf,
                "description": desc,
                "score": max_standalone * 2,
                "estimated_count": cnt,
            })
            if verbose:
                print(f"    FACET AND: {desc} -> {cnt:,d} hits")

    # AND core with aux facets
    for core_name in core_facet_names:
        if core_name not in facet_or_clauses:
            continue
        or_core, names_core = facet_or_clauses[core_name]

        for aux_name, validated in aux_validated.items():
            if len(queries) >= max_queries:
                break
            if aux_name not in facet_or_clauses:
                continue

            or_aux, names_aux = facet_or_clauses[aux_name]
            cnf = [or_core, or_aux]
            kwargs = {"cnf": cnf}
            if max_clause_freq:
                kwargs["max_clause_freq"] = max_clause_freq
            result = engine.find_cnf(**kwargs)
            cnt = result.get("cnt", 0)

            if 0 < cnt <= max_refined:
                c_str = " OR ".join(names_core[:4])
                if len(names_core) > 4:
                    c_str += f" +{len(names_core)-4}"
                a_str = " OR ".join(names_aux[:4])
                if len(names_aux) > 4:
                    a_str += f" +{len(names_aux)-4}"
                desc = f"({c_str}) AND ({a_str})"
                _add({
                    "type": "cnf",
                    "cnf": cnf,
                    "description": desc,
                    "score": max_standalone,
                    "estimated_count": cnt,
                })
                if verbose:
                    print(f"    CORE x AUX: {desc} -> {cnt:,d} hits")

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
        print(f"  Grabbed directly: {len(grabbed_terms)} terms")
        print(f"{'='*60}")
        for i, q in enumerate(queries):
            marker = "FIND" if q["type"] == "simple" else "CNF "
            print(f"  {i+1:3d}. [{marker}] [{q['score']:>8.0f}] "
                  f"~{q['estimated_count']:>8,d}  {q['description']}")

    return queries, all_validated


def run_llm_adaptive(
    engine,
    queries: list[dict],
    max_clause_freq: int = 100000,
    min_retrieved_docs: int = None,
    verbose: bool = True,
) -> list[dict]:
    """Execute queries. Handles both 'simple' and 'cnf' types."""
    from adaptive_queries import run_adaptive
    return run_adaptive(
        engine, queries,
        max_clause_freq=max_clause_freq,
        min_retrieved_docs=min_retrieved_docs,
        verbose=verbose,
    )