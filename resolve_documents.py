"""
Resolve CNF query pointers to unique documents, fetch metadata and
token sequences in one efficient pass.

Usage:
    from resolve_documents import resolve_all_queries

    docs = resolve_all_queries(
        queries=queries,  # from run_queries(), each has 'ptrs_by_shard'
        index_dir='../msmarco_segmented_index/',
        max_doc_len=200,
    )
    # docs is a list of dicts with 'shard', 'doc_index', 'metadata', 'tokens', 'text'
"""

import json
import mmap
import numpy as np
from pathlib import Path
from tqdm import tqdm


def resolve_all_queries(
    queries: list[dict],
    index_dir: str,
    tokenizer=None,
    max_doc_len: int = 200,
    token_width: int = 2,
    top_splade_filter: int = None,
    top_tokens: list[tuple] = None,
) -> list[dict]:
    """
    Take all queries with ptrs_by_shard, resolve to unique documents,
    and fetch metadata + token sequences.

    Args:
        queries: List of query dicts from run_queries(), each with 'ptrs_by_shard'.
        index_dir: Path to the infini-gram index directory.
        tokenizer: If provided, decode tokens to text (infini-gram tokenizer).
        max_doc_len: Maximum number of tokens to read per document.
        token_width: Bytes per token (2 for 16-bit).
        top_splade_filter: If set, crude-score all documents using the important
            tokens and only keep this many top-scoring documents.
        top_tokens: List of (token, splade_score, tup, combined_score) tuples.
            Required if top_splade_filter is set.

    Returns:
        List of document dicts, each with:
            - 'shard': shard index
            - 'doc_index': document index within shard
            - 'metadata': parsed metadata dict (or raw string if not JSON)
            - 'doc_id': extracted doc ID string (if available)
            - 'tokens': np.ndarray of token IDs (up to max_doc_len)
            - 'text': decoded text (if tokenizer provided)
            - 'from_queries': list of query indices that found this document
            - 'crude_score': (if top_splade_filter set) bag-of-words relevance score
    """
    index_dir = Path(index_dir)
    dtype = np.dtype(f"<u{token_width}")

    # Step 1: Collect all pointers across all queries, grouped by shard
    print("Step 1: Collecting pointers from all queries...")
    # shard -> list of (ptr, query_idx)
    shard_ptrs = {}

    # Handle CNF queries (ptrs_by_shard)
    for q_idx, q in enumerate(queries):
        ptrs_by_shard = q.get("ptrs_by_shard", [])
        for s, ptrs in enumerate(ptrs_by_shard):
            if s not in shard_ptrs:
                shard_ptrs[s] = []
            for ptr in ptrs:
                shard_ptrs[s].append((int(ptr), q_idx))

    # Handle simple find() queries (segment_by_shard)
    # Convert rank ranges to byte pointers via the suffix array
    sa_entry_bytes = 5
    for q_idx, q in enumerate(queries):
        segments = q.get("segment_by_shard", [])
        if not segments:
            continue
        for s, seg in enumerate(segments):
            if not isinstance(seg, (list, tuple)) or len(seg) < 2:
                continue
            start, end = int(seg[0]), int(seg[1])
            n = end - start
            if n <= 0:
                continue

            # Read SA entries to get byte pointers
            sa_path = index_dir / f"table.{s}"
            with open(sa_path, "rb") as sa_f:
                import mmap as mmap_module
                sa_mm = mmap_module.mmap(sa_f.fileno(), 0, access=mmap_module.ACCESS_READ)
                try:
                    sa_offset = start * sa_entry_bytes
                    sa_bytes = sa_mm[sa_offset: sa_offset + n * sa_entry_bytes]
                    sa_raw = np.frombuffer(sa_bytes, dtype=np.uint8).reshape(n, sa_entry_bytes)
                    sa_padded = np.zeros((n, 8), dtype=np.uint8)
                    sa_padded[:, :sa_entry_bytes] = sa_raw
                    byte_ptrs = np.frombuffer(sa_padded.tobytes(), dtype="<u8").reshape(n)
                finally:
                    sa_mm.close()

            if s not in shard_ptrs:
                shard_ptrs[s] = []
            for ptr in byte_ptrs:
                shard_ptrs[s].append((int(ptr), q_idx))

    total_ptrs = sum(len(v) for v in shard_ptrs.values())
    print(f"  Total pointers across all queries: {total_ptrs}")

    # Step 2: Resolve pointers to document indices per shard
    print("Step 2: Resolving pointers to document indices...")
    # Collect unique (shard, doc_index) -> set of query indices
    doc_to_queries = {}

    for s, ptr_list in shard_ptrs.items():
        off_path = index_dir / f"offset.{s}"
        doc_offsets = np.memmap(off_path, dtype="<u8", mode="r")

        ptrs = np.array([p for p, _ in ptr_list], dtype=np.uint64)
        query_indices = [qi for _, qi in ptr_list]

        doc_indices = np.searchsorted(doc_offsets, ptrs, side="right") - 1
        doc_indices = np.maximum(doc_indices, 0)

        for i, doc_idx in enumerate(doc_indices):
            key = (s, int(doc_idx))
            if key not in doc_to_queries:
                doc_to_queries[key] = set()
            doc_to_queries[key].add(query_indices[i])

    print(f"  {total_ptrs} pointers -> {len(doc_to_queries)} unique documents")

    # Step 3: Fetch metadata and tokens for each unique document
    print(f"Step 3: Fetching metadata and tokens for {len(doc_to_queries)} documents...")

    # Group by shard for efficient I/O
    by_shard = {}
    for (s, doc_idx), q_indices in doc_to_queries.items():
        if s not in by_shard:
            by_shard[s] = []
        by_shard[s].append((doc_idx, q_indices))

    documents = []

    for s, doc_list in by_shard.items():
        tok_path = index_dir / f"tokenized.{s}"
        off_path = index_dir / f"offset.{s}"
        metaoff_path = index_dir / f"metaoff.{s}"
        metadata_path = index_dir / f"metadata.{s}"

        tok_size = tok_path.stat().st_size
        meta_size = metadata_path.stat().st_size

        doc_offsets = np.memmap(off_path, dtype="<u8", mode="r")
        tok_tokens = np.memmap(tok_path, dtype=dtype, mode="r")
        metaoff = np.memmap(metaoff_path, dtype="<u8", mode="r")

        with open(metadata_path, "rb") as meta_f:
            for doc_idx, q_indices in tqdm(doc_list, desc=f"Shard {s}", leave=False):
                # Read tokens
                doc_start = int(doc_offsets[doc_idx])
                if doc_idx + 1 < len(doc_offsets):
                    doc_end = int(doc_offsets[doc_idx + 1])
                else:
                    doc_end = tok_size

                doc_start_tok = doc_start // token_width
                doc_end_tok = doc_end // token_width
                n_tokens = min(doc_end_tok - doc_start_tok, max_doc_len)

                tokens = tok_tokens[doc_start_tok : doc_start_tok + n_tokens].copy()

                # Read metadata
                meta_start = int(metaoff[doc_idx])
                if doc_idx + 1 < len(metaoff):
                    meta_end = int(metaoff[doc_idx + 1])
                else:
                    meta_end = meta_size

                meta_f.seek(meta_start)
                raw_meta = meta_f.read(meta_end - meta_start).decode("utf-8", errors="replace").strip()

                try:
                    parsed_meta = json.loads(raw_meta)
                except json.JSONDecodeError:
                    parsed_meta = raw_meta

                # Extract doc_id
                doc_id = None
                if isinstance(parsed_meta, dict):
                    doc_id = parsed_meta.get("metadata", {}).get("docid", None)
                    if doc_id is None:
                        doc_id = parsed_meta.get("docid", None)

                # Decode text
                text = None
                if tokenizer is not None:
                    text = tokenizer.decode(tokens.tolist(), skip_special_tokens=True)

                documents.append({
                    "shard": s,
                    "doc_index": doc_idx,
                    "doc_id": doc_id,
                    "metadata": parsed_meta,
                    "tokens": tokens,
                    "text": text,
                    "from_queries": sorted(q_indices),
                })

    print(f"\nDone! {len(documents)} unique documents resolved.")
    n_multi = sum(1 for d in documents if len(d["from_queries"]) > 1)
    print(f"  {n_multi} documents found by multiple queries")

    # Crude SPLADE-based filtering
    if top_splade_filter is not None and top_tokens is not None:
        print(f"\nStep 4: Crude scoring and filtering to top {top_splade_filter}...")

        # Precompute important token IDs (SPLADE/BERT -> Llama)
        important_words = []
        for token, splade_score, tup, combined in top_tokens:
            clean = token.lstrip("#")
            ids = tokenizer.encode(clean, add_special_tokens=False)
            if ids:
                important_words.append((set(ids), combined))

        # Score each document
        for doc in tqdm(documents, desc="Scoring"):
            doc_set = set(doc["tokens"].tolist())
            doc["crude_score"] = sum(
                combined
                for ids, combined in important_words
                if ids.issubset(doc_set)
            )

        # Sort and filter
        documents.sort(key=lambda x: x["crude_score"], reverse=True)

        if len(documents) > 0:
            print(f"  Score range: {documents[0]['crude_score']:.2f} (best) "
                  f"-> {documents[-1]['crude_score']:.2f} (worst)")

        documents = documents[:top_splade_filter]
        print(f"  Kept top {len(documents)} documents")

    return documents