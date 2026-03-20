"""
Similarity search combining infini-gram index retrieval with SPLADE re-ranking.

Usage:
    from similarity_search import similarity_search

    results = similarity_search(
        reference="What is the definition of community?",
        engine=engine,
        tokenizer=tokenizer,       # infini-gram tokenizer (e.g. Llama)
        model=splade_model,         # SparseEncoder model
        queries=[
            input_ids_community,                          # simple keyword query
            [[input_ids_community], [input_ids_definition]],  # CNF query (AND)
        ],
        index_dir='../msmarco_segmented_index/',
    )
"""

import torch
import numpy as np
from tqdm import tqdm
from fast_extensions import get_extensions, get_extensions_cnf


def _is_cnf_query(query):
    """
    Distinguish between a simple keyword query (list of ints) and a CNF query
    (list of lists of lists of ints).

    Simple keyword: [tok1, tok2, ...]           — list of ints
    CNF:            [[[tok1, tok2], [tok3]], ...] — list of list of list of ints
    """
    if not query:
        return False
    first = query[0]
    if not isinstance(first, list):
        return False
    # CNF needs triple nesting: first element is a clause (list),
    # and the clause's first element is an alternative (also a list)
    if not first:
        return False
    return isinstance(first[0], list)


def similarity_search(
    reference: str,
    engine,
    tokenizer,
    model,
    queries: list,
    index_dir: str = "../msmarco_segmented_index/",
    ext_len: int = 50,
    read_len: int = 100,
    batch_size: int = 10_000,
    max_candidates_per_query: int = 1000,
    token_width: int = 2,
    sa_entry_bytes: int = 5,
    pad_token_id: int = 0,
    max_clause_freq: int | None = None,
) -> dict:
    """
    For each query, retrieve candidate passages from the infini-gram index,
    encode them with SPLADE, and return the top candidates ranked by
    similarity to the reference.

    Args:
        reference: The reference text to rank candidates against.
        engine: Infini-gram engine instance.
        tokenizer: Tokenizer used by the infini-gram index (for decoding).
        model: SparseEncoder (SPLADE) model for encoding and similarity.
        queries: List of queries. Each query is either:
            - A list of token IDs (simple keyword query, uses engine.find)
            - A CNF query (list of lists of lists, uses engine.find_cnf)
        index_dir: Path to the infini-gram index directory.
        ext_len: Number of extension tokens to read for simple queries.
        read_len: Number of tokens to read for CNF queries.
        batch_size: Batch size for SPLADE encoding.
        max_candidates_per_query: Number of top candidates to return per query.
        token_width: Bytes per token in the index.
        sa_entry_bytes: Bytes per suffix array entry.
        pad_token_id: Pad token ID for truncated reads.
        max_clause_freq: If set, passed to engine.find_cnf to avoid sampling.

    Returns:
        Dict mapping query index to a list of dicts, each with:
            - 'text': the decoded candidate passage
            - 'score': similarity score to the reference
    """
    # Step 1: Encode the reference with SPLADE
    print("Step 1/4: Encoding reference with SPLADE...")
    ref_embedding = model.encode_query([reference])

    results = {}

    for q_idx, query in enumerate(queries):
        print(f"\n{'='*60}")
        print(f"Query {q_idx + 1}/{len(queries)}")
        print(f"{'='*60}")

        is_cnf = _is_cnf_query(query)

        # Step 2: Retrieve candidates from infini-gram index
        if is_cnf:
            print("Step 2/4: Running CNF query on infini-gram index...")
            find_kwargs = {"cnf": query}
            if max_clause_freq is not None:
                find_kwargs["max_clause_freq"] = max_clause_freq
            find_result = engine.find_cnf(**find_kwargs)
            if "error" in find_result:
                print(f"  ERROR from engine: {find_result['error']}")
                results[q_idx] = []
                continue
            print(f"  Found {find_result['cnt']} matches (approx={find_result['approx']})")
            total_ptrs = sum(len(p) for p in find_result['ptrs_by_shard'])
            print(f"  Total pointers across shards: {total_ptrs}")

            print("Step 2b/4: Reading token sequences from disk...")
            token_arrays, truncated = get_extensions_cnf(
                index_dir=index_dir,
                ptrs_by_shard=find_result['ptrs_by_shard'],
                read_len=read_len,
                token_width=token_width,
                pad_token_id=pad_token_id,
            )
        else:
            print("Step 2/4: Running keyword query on infini-gram index...")
            find_result = engine.find(input_ids=query)
            total_hits = find_result['cnt']
            print(f"  Found {total_hits} occurrences")

            print("Step 2b/4: Reading token extensions from disk...")
            token_arrays, truncated = get_extensions(
                index_dir=index_dir,
                segment_by_shard=find_result['segment_by_shard'],
                query_len=len(query),
                ext_len=ext_len,
                token_width=token_width,
                sa_entry_bytes=sa_entry_bytes,
                pad_token_id=pad_token_id,
            )

        print(f"  Retrieved {len(token_arrays)} sequences ({truncated} truncated)")

        if len(token_arrays) == 0:
            print("  No candidates found, skipping.")
            results[q_idx] = []
            continue

        # Deduplicate before decoding
        print("Step 2c/4: Deduplicating sequences...")
        unique, inverse, counts = np.unique(
            token_arrays, axis=0, return_inverse=True, return_counts=True
        )
        print(f"  {len(token_arrays)} total -> {len(unique)} unique sequences")

        # Step 3: Decode token IDs to strings
        print("Step 3/4: Decoding token sequences to text...")
        chunk_size = 100_000
        unique_strings = []
        for i in tqdm(range(0, len(unique), chunk_size), desc="Decoding"):
            batch = unique[i:i + chunk_size].tolist()
            unique_strings.extend(tokenizer.batch_decode(batch, skip_special_tokens=True))

        # Filter out empty strings
        valid_mask = [bool(s.strip()) for s in unique_strings]
        valid_strings = [s for s, v in zip(unique_strings, valid_mask) if v]
        print(f"  {len(valid_strings)} non-empty unique passages")

        if not valid_strings:
            print("  No valid candidates after decoding, skipping.")
            results[q_idx] = []
            continue

        # Step 4: Encode with SPLADE and rank
        print(f"Step 4/4: Encoding {len(valid_strings)} candidates with SPLADE and ranking...")
        doc_embeddings = model.encode_document(
            valid_strings,
            show_progress_bar=True,
        )

        scores = model.similarity(ref_embedding, doc_embeddings)[0]
        top_k = min(max_candidates_per_query, len(valid_strings))
        top_indices = scores.argsort(descending=True)[:top_k].cpu().tolist()

        results[q_idx] = [
            {
                "text": valid_strings[i],
                "score": scores[i].item(),
            }
            for i in top_indices
        ]
        print(f"  Top score: {results[q_idx][0]['score']:.4f}, "
              f"Bottom score: {results[q_idx][-1]['score']:.4f}")

    print(f"\n{'='*60}")
    print("Done!")
    return results