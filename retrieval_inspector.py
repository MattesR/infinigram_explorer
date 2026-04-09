"""
Qualitative analysis of retrieval results.

Inspect what documents were found, what was missed, and what's irrelevant.

Usage:
    from retrieval_inspector import inspect_query

    inspect_query(
        qid="2024-32912",
        query_text="how bad did the vietnam war devastate the economy in 1968",
        pipeline=pipeline,
        engine=engine,
        tokenizer=tokenizer,
        qrels_path="./qrels.rag24.test-umbrela-all.txt",
        expansions_path="./keyword_expansions.jsonl",
    )
"""

import json
from trec_output import load_qrels
from resolve_documents import resolve_all_queries


def inspect_query(
    qid: str,
    query_text: str,
    pipeline,
    engine,
    tokenizer,
    qrels_path: str,
    expansions_path: str = None,
    index_dir: str = "../msmarco_segmented_index/",
    mode: str = "llm_adaptive",
    max_standalone: int = 10000,
    max_standalone_sup: int = 1000,
    max_doc_len: int = 200,
    show_text_len: int = 300,
    show_found: int = 10,
    show_missed: int = 10,
    show_irrelevant: int = 10,
    save_irrelevant: int = 100,
    filter_mode: str = "stopword",
    output_dir: str = "./inspection",
    corpus_dir: str = None,
):
    """
    Inspect retrieval results for a single query.

    Shows:
    - Found relevant docs (with text snippets)
    - Missed relevant docs (from qrels but not retrieved)
    - Sample irrelevant docs (retrieved but not in qrels)
    """
    qrels = load_qrels(qrels_path)
    relevant = {did: rel for did, rel in qrels.get(qid, {}).items() if rel > 0}

    print(f"{'='*80}")
    print(f"Query: [{qid}] {query_text}")
    print(f"Relevant docs in qrels: {len(relevant)}")
    print(f"{'='*80}")

    # Run retrieval
    if mode == "llm_adaptive" and expansions_path:
        executed, scored, validated = pipeline.build_and_run_llm_adaptive(
            query_text,
            qid=qid,
            engine=engine,
            expansions_path=expansions_path,
            max_standalone=max_standalone,
            max_standalone_sup=max_standalone_sup,
            filter_mode=filter_mode,
            verbose=True,
        )
    else:
        executed, scored = pipeline.build_and_run_adaptive(
            query_text,
            engine=engine,
            max_standalone=max_standalone,
            verbose=True,
        )

    if not executed:
        print("No queries executed!")
        return

    # Resolve all docs (no filtering)
    docs = resolve_all_queries(
        queries=executed,
        index_dir=index_dir,
        tokenizer=tokenizer,
        max_doc_len=max_doc_len,
    )

    retrieved_ids = {d["doc_id"]: d for d in docs if d["doc_id"]}
    found = {did: retrieved_ids[did] for did in relevant if did in retrieved_ids}
    missed = {did: rel for did, rel in relevant.items() if did not in retrieved_ids}
    irrelevant = {did: d for did, d in retrieved_ids.items() if did not in relevant}

    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"  Retrieved:   {len(retrieved_ids)}")
    print(f"  Relevant:    {len(relevant)}")
    print(f"  Found:       {len(found)} ({len(found)/len(relevant)*100:.1f}%)")
    print(f"  Missed:      {len(missed)}")
    print(f"  Irrelevant:  {len(irrelevant)}")

    # Show found relevant docs
    print(f"\n{'='*80}")
    print(f"FOUND RELEVANT DOCS ({len(found)} total, showing {min(show_found, len(found))})")
    print(f"{'='*80}")
    for i, (did, doc) in enumerate(sorted(found.items(), key=lambda x: relevant[x[0]], reverse=True)):
        if i >= show_found:
            break
        rel = relevant[did]
        text = doc.get("text", "")[:show_text_len] if doc.get("text") else "[no text]"
        queries_hit = doc.get("from_queries", [])
        print(f"\n  [{i+1}] rel={rel} | {did}")
        print(f"      Found by queries: {queries_hit}")
        print(f"      {text}...")

    # Show missed relevant docs
    print(f"\n{'='*80}")
    print(f"MISSED RELEVANT DOCS ({len(missed)} total, showing {min(show_missed, len(missed))})")
    print(f"{'='*80}")
    missed_sorted = sorted(missed.items(), key=lambda x: x[1], reverse=True)

    # Look up missed doc texts from corpus if available
    missed_texts = {}
    if corpus_dir and missed:
        print(f"  Looking up missed docs from corpus...")
        missed_texts = _lookup_docs_from_corpus(set(missed.keys()), corpus_dir)
        print(f"  Found text for {len(missed_texts)}/{len(missed)} missed docs")

    for i, (did, rel) in enumerate(missed_sorted):
        if i >= show_missed:
            break
        print(f"\n  [{i+1}] rel={rel} | {did}")
        if did in missed_texts:
            doc_obj = missed_texts[did]
            text = doc_obj.get("segment", doc_obj.get("body", doc_obj.get("text", "")))
            print(f"      {text[:show_text_len]}...")
        else:
            print(f"      [text not available]")

    # Crude score all docs for ranking irrelevant samples
    if mode == "llm_adaptive" and validated:
        import math
        scoring_words = []
        for term in validated:
            ids = set(term["input_ids"])
            weight = 1.0 / math.log(term["count"] + 2)
            scoring_words.append((ids, weight))
        for doc in docs:
            doc_set = set(doc["tokens"].tolist())
            doc["crude_score"] = sum(w for ids, w in scoring_words if ids.issubset(doc_set))
    else:
        for doc in docs:
            doc["crude_score"] = 0.0

    # Re-sort irrelevant by crude score descending (highest-scored irrelevant first)
    irrelevant_ranked = sorted(
        [(did, d) for did, d in irrelevant.items()],
        key=lambda x: x[1].get("crude_score", 0),
        reverse=True,
    )

    # Show sample irrelevant docs (top by crude score)
    print(f"\n{'='*80}")
    print(f"TOP IRRELEVANT DOCS by crude score ({len(irrelevant)} total, showing {min(show_irrelevant, len(irrelevant))})")
    print(f"{'='*80}")
    for i, (did, doc) in enumerate(irrelevant_ranked[:show_irrelevant]):
        text = doc.get("text", "")[:show_text_len] if doc.get("text") else "[no text]"
        queries_hit = doc.get("from_queries", [])
        score = doc.get("crude_score", 0)
        print(f"\n  [{i+1}] score={score:.3f} | {did}")
        print(f"      Found by queries: {queries_hit}")
        print(f"      {text}...")

    # Relevance grade distribution of found docs
    print(f"\n{'='*80}")
    print(f"RELEVANCE DISTRIBUTION")
    print(f"{'='*80}")
    from collections import Counter
    found_grades = Counter(relevant[did] for did in found)
    missed_grades = Counter(rel for rel in missed.values())
    print(f"  Found:  {dict(sorted(found_grades.items(), reverse=True))}")
    print(f"  Missed: {dict(sorted(missed_grades.items(), reverse=True))}")

    # Save to files
    import os
    os.makedirs(output_dir, exist_ok=True)

    def _doc_to_record(did, doc, rel=None):
        record = {
            "doc_id": did,
            "text": doc.get("text", "")[:2000] if doc.get("text") else "",
            "from_queries": doc.get("from_queries", []),
            "crude_score": doc.get("crude_score", 0),
        }
        if rel is not None:
            record["relevance"] = rel
        return record

    # Save found relevant
    found_path = os.path.join(output_dir, f"{qid}_found.jsonl")
    with open(found_path, "w") as f:
        for did in sorted(found, key=lambda d: relevant[d], reverse=True):
            record = _doc_to_record(did, found[did], rel=relevant[did])
            f.write(json.dumps(record) + "\n")
    print(f"\nSaved {len(found)} found docs to {found_path}")

    # Save missed relevant (with corpus text if available)
    missed_path = os.path.join(output_dir, f"{qid}_missed.jsonl")
    # Look up all missed docs from corpus if not already done
    if corpus_dir and missed and not missed_texts:
        missed_texts = _lookup_docs_from_corpus(set(missed.keys()), corpus_dir)
    with open(missed_path, "w") as f:
        for did, rel in sorted(missed.items(), key=lambda x: x[1], reverse=True):
            record = {"doc_id": did, "relevance": rel}
            if did in missed_texts:
                doc_obj = missed_texts[did]
                text = doc_obj.get("segment", doc_obj.get("body", doc_obj.get("text", "")))
                record["text"] = text[:2000]
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(missed)} missed docs to {missed_path}")

    # Save top irrelevant by crude score
    irr_path = os.path.join(output_dir, f"{qid}_irrelevant_top{save_irrelevant}.jsonl")
    with open(irr_path, "w") as f:
        for did, doc in irrelevant_ranked[:save_irrelevant]:
            record = _doc_to_record(did, doc)
            f.write(json.dumps(record) + "\n")
    print(f"Saved {min(save_irrelevant, len(irrelevant))} top irrelevant docs to {irr_path}")

    # Save query info
    info_path = os.path.join(output_dir, f"{qid}_info.json")
    with open(info_path, "w") as f:
        json.dump({
            "qid": qid,
            "query": query_text,
            "n_relevant": len(relevant),
            "n_retrieved": len(retrieved_ids),
            "n_found": len(found),
            "n_missed": len(missed),
            "n_irrelevant": len(irrelevant),
            "recall": len(found) / len(relevant) if relevant else 0,
            "relevance_found": dict(sorted(found_grades.items(), reverse=True)),
            "relevance_missed": dict(sorted(missed_grades.items(), reverse=True)),
            "queries_executed": [
                {"description": q.get("description", ""), "count": q.get("cnt", 0)}
                for q in executed
            ],
        }, f, indent=2)
    print(f"Saved query info to {info_path}")

    return {
        "found": found,
        "missed": missed,
        "irrelevant": irrelevant,
        "docs": docs,
        "executed": executed,
    }


def _lookup_docs_from_corpus(doc_ids, corpus_dir):
    """
    Look up documents from the gzipped MSMARCO corpus files.
    Groups by file number and scans each file once.

    Args:
        doc_ids: Set or list of doc ID strings.
        corpus_dir: Path to directory with msmarco_v2.1_doc_segmented_XX.json.gz files.

    Returns:
        Dict mapping doc_id -> {"text": ..., "title": ..., ...}
    """
    import gzip

    if not doc_ids:
        return {}

    # Group by file number
    by_file = {}
    for did in doc_ids:
        try:
            parts = did.split("_")
            fnum = int(parts[4])
            by_file.setdefault(fnum, set()).add(did)
        except (IndexError, ValueError):
            continue

    results = {}
    for fnum, ids in sorted(by_file.items()):
        path = f"{corpus_dir}/msmarco_v2.1_doc_segmented_{fnum:02d}.json.gz"
        try:
            with gzip.open(path, "rt") as f:
                for line in f:
                    obj = json.loads(line)
                    did = obj.get("docid", obj.get("doc_id", ""))
                    if did in ids:
                        results[did] = obj
                        ids.discard(did)
                        if not ids:
                            break
        except FileNotFoundError:
            print(f"      [corpus file not found: {path}]")

    return results


def compare_found_missed_keywords(
    qid: str,
    found: dict,
    missed: dict,
    expansions_path: str = None,
):
    """
    Analyze which LLM keywords appear in found vs missed docs.
    Helps diagnose why documents are being missed.
    """
    if not expansions_path:
        print("Need expansions_path to load keywords")
        return

    from llm_keyword_filter import load_faceted_keywords
    facets = load_faceted_keywords(qid, expansions_path)

    all_kws = []
    for name, terms in facets["core_facets"].items():
        all_kws.extend(terms)
    for name, terms in facets["aux_facets"].items():
        all_kws.extend(terms)

    print(f"\nKeyword presence in found vs missed docs:")
    print(f"{'Keyword':<40s} {'In Found':>10s} {'In Missed':>10s}")
    print("-" * 65)

    for kw in all_kws[:30]:
        kw_lower = kw.lower()
        in_found = sum(1 for d in found.values()
                      if d.get("text") and kw_lower in d["text"].lower())
        # Can't check missed docs easily since we don't have their text
        print(f"  {kw:<38s} {in_found:>10d}       ?")