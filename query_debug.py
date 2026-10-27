"""
Quick doc lookup for a single CNF query.

Usage:
    from query_debug import get_docs_for_query

    docs = get_docs_for_query(
        cnf=[[tokenizer.encode("Great Society")]],
        engine=engine,
        tokenizer=tokenizer,
        max_diff_tokens=10,
    )
    print(f"{len(docs)} docs found")
    for d in docs[:5]:
        print(d["doc_id"])
"""

from resolve_documents import resolve_all_queries


def get_docs_for_query(
    cnf,
    engine,
    tokenizer,
    max_diff_tokens=None,
    max_clause_freq=80000000,
    index_dir="../msmarco_segmented_index/",
    max_doc_len=200,
):
    """
    Execute a single CNF query and return resolved documents.

    Args:
        cnf: CNF query (list of clauses)
        engine: infini-gram engine
        tokenizer: tokenizer
        max_diff_tokens: proximity constraint
        max_clause_freq: max clause frequency
        index_dir: path to index
        max_doc_len: max tokens per doc

    Returns:
        List of doc dicts with doc_id, text, etc.
    """
    kwargs = {"cnf": cnf, "max_clause_freq": max_clause_freq}
    if max_diff_tokens:
        kwargs["max_diff_tokens"] = max_diff_tokens

    result = engine.find_cnf(**kwargs)
    cnt = result.get("cnt", 0)
    print(f"Found {cnt} segments")

    if cnt == 0:
        return []

    # Wrap as a query dict for resolve_all_queries
    query = {
        "type": "cnf",
        "cnf": cnf,
        "cnt": cnt,
        "ptrs_by_shard": result.get("ptrs_by_shard", []),
        "segment_by_shard": result.get("segment_by_shard", []),
        "description": "debug query",
    }

    docs = resolve_all_queries(
        queries=[query],
        index_dir=index_dir,
        tokenizer=tokenizer,
        max_doc_len=max_doc_len,
    )

    doc_ids = {d["doc_id"] for d in docs if d["doc_id"]}
    print(f"Resolved to {len(doc_ids)} unique documents")

    return docs