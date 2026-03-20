"""
Fast extraction of token extensions from an infini-gram index by reading
the suffix array (table.{s}) and tokenized corpus (tokenized.{s}) directly.

The suffix array stores byte offsets into the tokenized corpus. Each suffix
array entry is `sa_entry_bytes` wide (default 5), and each token in the
corpus is `token_width` bytes wide (default 2, i.e. 16-bit token IDs).
All values are little-endian.

Usage:
    from fast_extensions import get_extensions

    # Get the 3-token extensions of 'community' across all shards
    result = engine.find(input_ids=input_ids)
    extensions = get_extensions(
        index_dir='../msmarco_segmented_index/',
        segment_by_shard=result['segment_by_shard'],
        query_len=len(input_ids),
        ext_len=3,
    )
    # extensions is a numpy array of shape (total_occurrences, query_len + ext_len)
    # each row is the query token IDs followed by ext_len extension token IDs
"""

import mmap
import numpy as np
from collections import Counter
from tqdm import tqdm
from pathlib import Path


def get_extensions(
    index_dir: str,
    segment_by_shard: list,
    query_len: int,
    ext_len: int = 3,
    token_width: int = 2,
    sa_entry_bytes: int = 5,
    pad_token_id: int = 0,
) -> tuple[np.ndarray, int]:
    """
    Extract token extensions directly from the on-disk suffix array and
    tokenized corpus, bypassing the Python engine API entirely.

    Uses the document offset file (offset.{s}) to avoid reading past
    document boundaries.

    Args:
        index_dir: Path to the infini-gram index directory.
        segment_by_shard: List of (start, end) rank tuples from engine.find().
        query_len: Number of tokens in the query (len(input_ids)).
        ext_len: How many tokens after the query to extract.
        token_width: Bytes per token in the tokenized file (2 for 16-bit).
        sa_entry_bytes: Bytes per suffix array entry (5 for infini-gram).
        pad_token_id: Token ID used to pad rows truncated by document boundaries.

    Returns:
        Tuple of:
        - np.ndarray of shape (N, query_len + ext_len). Each row contains the
          query tokens followed by the extension tokens. Rows truncated by
          document boundaries are right-padded with pad_token_id.
        - int: number of rows that were truncated.
    """
    index_dir = Path(index_dir)
    all_extensions = []
    total_truncated = 0
    total_len = query_len + ext_len

    for s, (start, end) in enumerate(segment_by_shard):
        n = end - start
        if n == 0:
            continue

        sa_path = index_dir / f"table.{s}"
        tok_path = index_dir / f"tokenized.{s}"
        off_path = index_dir / f"offset.{s}"
        tok_size = tok_path.stat().st_size

        # Load document offsets: each is an 8-byte little-endian pointer
        # into tokenized.{s} marking where each document starts.
        doc_offsets = np.memmap(off_path, dtype="<u8", mode="r")

        with open(sa_path, "rb") as sa_f, open(tok_path, "rb") as tok_f:
            sa_mm = mmap.mmap(sa_f.fileno(), 0, access=mmap.ACCESS_READ)
            tok_mm = mmap.mmap(tok_f.fileno(), 0, access=mmap.ACCESS_READ)

            try:
                # Read the contiguous SA segment in one go
                sa_offset = start * sa_entry_bytes
                sa_bytes = sa_mm[sa_offset : sa_offset + n * sa_entry_bytes]

                # Parse SA entries (variable-width little-endian integers)
                # Pad each entry to 8 bytes so we can read as uint64
                sa_raw = np.frombuffer(sa_bytes, dtype=np.uint8).reshape(n, sa_entry_bytes)
                sa_padded = np.zeros((n, 8), dtype=np.uint8)
                sa_padded[:, :sa_entry_bytes] = sa_raw
                byte_offsets = np.frombuffer(sa_padded.tobytes(), dtype="<u8").reshape(n)

                # For each byte offset, read query + ext_len tokens
                max_read_bytes = total_len * token_width
                dtype = np.dtype(f"<u{token_width}")

                extensions = np.full((n, total_len), pad_token_id, dtype=dtype)
                for i in tqdm(range(n), desc=f"Shard {s}", leave=False):
                    occurrence_pos = int(byte_offsets[i])

                    # Find the end of the enclosing document
                    doc_idx = np.searchsorted(doc_offsets, occurrence_pos, side="right") - 1
                    if doc_idx + 1 < len(doc_offsets):
                        doc_end = int(doc_offsets[doc_idx + 1])
                    else:
                        doc_end = tok_size

                    # Clamp read to document boundary and file boundary
                    available_bytes = min(doc_end, tok_size) - occurrence_pos
                    read_bytes = min(max_read_bytes, max(0, available_bytes))
                    n_tokens = read_bytes // token_width

                    if n_tokens > 0:
                        raw = tok_mm[occurrence_pos : occurrence_pos + n_tokens * token_width]
                        extensions[i, :n_tokens] = np.frombuffer(raw, dtype=dtype)

                    if n_tokens < total_len:
                        total_truncated += 1

                all_extensions.append(extensions)
            finally:
                sa_mm.close()
                tok_mm.close()

    return np.concatenate(all_extensions, axis=0), total_truncated


def get_extensions_cnf(
    index_dir: str,
    ptrs_by_shard: list,
    read_len: int = 100,
    token_width: int = 2,
    pad_token_id: int = 0,
) -> tuple[np.ndarray, int]:
    """
    Extract token sequences from CNF query pointers by reading the tokenized
    corpus directly. The pointers are already byte offsets into tokenized.{s},
    so no suffix array lookup is needed.

    Args:
        index_dir: Path to the infini-gram index directory.
        ptrs_by_shard: List of pointer lists from engine.find_cnf()['ptrs_by_shard'].
        read_len: Number of tokens to read starting at each pointer.
        token_width: Bytes per token in the tokenized file (2 for 16-bit).
        pad_token_id: Token ID used to pad rows truncated by document boundaries.

    Returns:
        Tuple of:
        - np.ndarray of shape (N, read_len). Each row is read_len tokens
          starting from the pointer position, clamped to document boundaries.
        - int: number of rows that were truncated.
    """
    index_dir = Path(index_dir)
    all_sequences = []
    total_truncated = 0

    for s, ptrs in enumerate(ptrs_by_shard):
        n = len(ptrs)
        if n == 0:
            continue

        tok_path = index_dir / f"tokenized.{s}"
        off_path = index_dir / f"offset.{s}"
        tok_size = tok_path.stat().st_size

        doc_offsets = np.memmap(off_path, dtype="<u8", mode="r")

        with open(tok_path, "rb") as tok_f:
            tok_mm = mmap.mmap(tok_f.fileno(), 0, access=mmap.ACCESS_READ)

            try:
                max_read_bytes = read_len * token_width
                dtype = np.dtype(f"<u{token_width}")

                sequences = np.full((n, read_len), pad_token_id, dtype=dtype)
                for i, ptr in enumerate(tqdm(ptrs, desc=f"Shard {s}", leave=False)):
                    ptr = int(ptr)

                    # Find the end of the enclosing document
                    doc_idx = np.searchsorted(doc_offsets, ptr, side="right") - 1
                    if doc_idx + 1 < len(doc_offsets):
                        doc_end = int(doc_offsets[doc_idx + 1])
                    else:
                        doc_end = tok_size

                    # Also find the start of the document so we read from doc start
                    doc_start = int(doc_offsets[doc_idx]) if doc_idx >= 0 else 0

                    # Read from doc_start, clamped to document boundary
                    available_bytes = min(doc_end, tok_size) - doc_start
                    read_bytes = min(max_read_bytes, max(0, available_bytes))
                    n_tokens = read_bytes // token_width

                    if n_tokens > 0:
                        raw = tok_mm[doc_start : doc_start + n_tokens * token_width]
                        sequences[i, :n_tokens] = np.frombuffer(raw, dtype=dtype)

                    if n_tokens < read_len:
                        total_truncated += 1

                all_sequences.append(sequences)
            finally:
                tok_mm.close()

    return np.concatenate(all_sequences, axis=0), total_truncated


def count_extensions(
    index_dir: str,
    segment_by_shard: list,
    query_len: int,
    ext_len: int = 3,
    token_width: int = 2,
    sa_entry_bytes: int = 5,
    pad_token_id: int = 0,
    top_k: int = 50,
) -> tuple[Counter, int]:
    """
    Like get_extensions, but returns a Counter of the most common
    extension tuples instead of the full array.

    Args:
        ... (same as get_extensions)
        top_k: If set, return only the top_k most common extensions.

    Returns:
        Tuple of:
        - Counter mapping (tok1, tok2, ...) tuples to their counts.
        - int: number of rows that were truncated by document boundaries.
    """
    extensions, truncated = get_extensions(
        index_dir=index_dir,
        segment_by_shard=segment_by_shard,
        query_len=query_len,
        ext_len=ext_len,
        token_width=token_width,
        sa_entry_bytes=sa_entry_bytes,
        pad_token_id=pad_token_id,
    )
    # Convert rows to tuples for counting
    ext_tuples = [tuple(row) for row in extensions]
    counts = Counter(ext_tuples)
    if top_k:
        return Counter(dict(counts.most_common(top_k))), truncated
    return counts, truncated