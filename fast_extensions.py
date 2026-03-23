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


def _parse_sa_entries(sa_mm, start: int, n: int, sa_entry_bytes: int) -> np.ndarray:
    """Read n suffix array entries and return as uint64 byte offsets."""
    sa_offset = start * sa_entry_bytes
    sa_bytes = sa_mm[sa_offset : sa_offset + n * sa_entry_bytes]
    sa_raw = np.frombuffer(sa_bytes, dtype=np.uint8).reshape(n, sa_entry_bytes)
    sa_padded = np.zeros((n, 8), dtype=np.uint8)
    sa_padded[:, :sa_entry_bytes] = sa_raw
    return np.frombuffer(sa_padded.tobytes(), dtype="<u8").reshape(n)


def _compute_read_limits(
    byte_offsets: np.ndarray,
    doc_offsets: np.ndarray,
    tok_size: int,
    total_len: int,
    token_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized: for each byte offset, compute how many tokens can be read
    without crossing a document boundary.
    Returns:
        - n_tokens: array of shape (n,) with clamped token counts.
        - doc_indices: array of shape (n,) with the document index for each entry.
    """
    doc_indices = np.searchsorted(doc_offsets, byte_offsets, side="right") - 1
    next_doc_indices = doc_indices + 1
    within_bounds = next_doc_indices < len(doc_offsets)
    doc_ends = np.where(
        within_bounds,
        doc_offsets[np.minimum(next_doc_indices, len(doc_offsets) - 1)],
        tok_size,
    )
    doc_ends = np.minimum(doc_ends, tok_size)
    available_bytes = doc_ends.astype(np.int64) - byte_offsets.astype(np.int64)
    available_bytes = np.maximum(available_bytes, 0)
    max_read_bytes = total_len * token_width
    read_bytes = np.minimum(available_bytes, max_read_bytes)
    n_tokens = (read_bytes // token_width).astype(np.int64)
    return n_tokens, doc_indices


def get_metadata(index_dir: str, shard: int, doc_idx: int) -> str:
    """
    Read the metadata string for a single document by index, using
    metaoff.{s} and metadata.{s}.
    """
    index_dir = Path(index_dir)
    metaoff_path = index_dir / f"metaoff.{shard}"
    metadata_path = index_dir / f"metadata.{shard}"

    metaoff = np.memmap(metaoff_path, dtype="<u8", mode="r")

    start = int(metaoff[doc_idx])
    if doc_idx + 1 < len(metaoff):
        end = int(metaoff[doc_idx + 1])
    else:
        end = metadata_path.stat().st_size

    with open(metadata_path, "rb") as f:
        f.seek(start)
        return f.read(end - start).decode("utf-8", errors="replace").strip()


def get_metadata_batch(index_dir: str, shard: int, doc_indices: np.ndarray) -> list[str]:
    """
    Read metadata strings for multiple documents at once.
    Deduplicates internally to avoid redundant reads.
    """
    index_dir = Path(index_dir)
    metaoff_path = index_dir / f"metaoff.{shard}"
    metadata_path = index_dir / f"metadata.{shard}"
    meta_size = metadata_path.stat().st_size

    metaoff = np.memmap(metaoff_path, dtype="<u8", mode="r")

    # Deduplicate doc indices for efficient reading
    unique_docs, inverse = np.unique(doc_indices, return_inverse=True)

    unique_metadata = []
    with open(metadata_path, "rb") as f:
        for doc_idx in unique_docs:
            start = int(metaoff[doc_idx])
            if doc_idx + 1 < len(metaoff):
                end = int(metaoff[doc_idx + 1])
            else:
                end = meta_size
            f.seek(start)
            meta = f.read(end - start).decode("utf-8", errors="replace").strip()
            unique_metadata.append(meta)

    # Map back to original order
    return [unique_metadata[i] for i in inverse]


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

    Uses vectorized numpy operations for speed. The tokenized corpus is
    memory-mapped as a numpy array, and full-length entries are gathered
    with fancy indexing (no Python for-loop).

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
        - np.ndarray of shape (N, 2), dtype int64. Each row is (shard, doc_index)
          identifying which document the sequence came from.
        - int: number of rows that were truncated.
    """
    index_dir = Path(index_dir)
    all_extensions = []
    all_doc_info = []
    total_truncated = 0
    total_len = query_len + ext_len
    dtype = np.dtype(f"<u{token_width}")

    for s, (start, end) in enumerate(segment_by_shard):
        n = end - start
        if n == 0:
            continue

        sa_path = index_dir / f"table.{s}"
        tok_path = index_dir / f"tokenized.{s}"
        off_path = index_dir / f"offset.{s}"
        tok_size = tok_path.stat().st_size

        doc_offsets = np.memmap(off_path, dtype="<u8", mode="r")
        tok_tokens = np.memmap(tok_path, dtype=dtype, mode="r")

        with open(sa_path, "rb") as sa_f:
            sa_mm = mmap.mmap(sa_f.fileno(), 0, access=mmap.ACCESS_READ)

            try:
                print(f"  Shard {s}: reading {n:,} SA entries...")
                byte_offsets = _parse_sa_entries(sa_mm, start, n, sa_entry_bytes)
                token_offsets = (byte_offsets // token_width).astype(np.int64)

                print(f"  Shard {s}: computing document boundaries...")
                n_tokens_per_entry, doc_indices = _compute_read_limits(
                    byte_offsets, doc_offsets, tok_size, total_len, token_width
                )
                total_truncated += int(np.sum(n_tokens_per_entry < total_len))

                # Store (shard, doc_index) for each entry
                shard_doc_info = np.column_stack([
                    np.full(n, s, dtype=np.int64),
                    doc_indices.astype(np.int64),
                ])
                all_doc_info.append(shard_doc_info)

                full_mask = n_tokens_per_entry >= total_len
                n_full = int(np.sum(full_mask))
                n_partial = n - n_full

                print(f"  Shard {s}: {n_full:,} full, {n_partial:,} truncated")
                print(f"  Shard {s}: gathering {n:,} sequences (vectorized)...")

                extensions = np.full((n, total_len), pad_token_id, dtype=dtype)

                if n_full > 0:
                    full_offsets = token_offsets[full_mask]
                    col_offsets = np.arange(total_len, dtype=np.int64)
                    gather_indices = full_offsets[:, None] + col_offsets[None, :]
                    gather_indices = np.clip(gather_indices, 0, len(tok_tokens) - 1)
                    extensions[full_mask] = tok_tokens[gather_indices]

                if n_partial > 0:
                    partial_indices = np.where(~full_mask)[0]
                    for idx in tqdm(partial_indices, desc=f"Shard {s} partial", leave=False):
                        t_off = int(token_offsets[idx])
                        n_tok = int(n_tokens_per_entry[idx])
                        if n_tok > 0:
                            extensions[idx, :n_tok] = tok_tokens[t_off : t_off + n_tok]

                all_extensions.append(extensions)
            finally:
                sa_mm.close()

    return (
        np.concatenate(all_extensions, axis=0),
        np.concatenate(all_doc_info, axis=0),
        total_truncated,
    )


def get_extensions_cnf(
    index_dir: str,
    ptrs_by_shard: list,
    read_len: int = 100,
    token_width: int = 2,
    pad_token_id: int = 0,
) -> tuple[np.ndarray, int]:
    """
    Extract token sequences from CNF query pointers by reading the tokenized
    corpus directly. Reads from document start. Vectorized.

    Args:
        index_dir: Path to the infini-gram index directory.
        ptrs_by_shard: List of pointer lists from engine.find_cnf()['ptrs_by_shard'].
        read_len: Number of tokens to read from document start.
        token_width: Bytes per token in the tokenized file (2 for 16-bit).
        pad_token_id: Token ID used to pad rows truncated by document boundaries.

    Returns:
        Tuple of:
        - np.ndarray of shape (N, read_len). Each row is read_len tokens
          from the document start, clamped to document boundaries.
        - int: number of rows that were truncated.
    """
    index_dir = Path(index_dir)
    all_sequences = []
    total_truncated = 0
    dtype = np.dtype(f"<u{token_width}")

    for s, ptrs in enumerate(ptrs_by_shard):
        n = len(ptrs)
        if n == 0:
            continue

        tok_path = index_dir / f"tokenized.{s}"
        off_path = index_dir / f"offset.{s}"
        tok_size = tok_path.stat().st_size

        doc_offsets = np.memmap(off_path, dtype="<u8", mode="r")
        tok_tokens = np.memmap(tok_path, dtype=dtype, mode="r")

        ptr_array = np.array(ptrs, dtype=np.uint64)

        doc_indices = np.searchsorted(doc_offsets, ptr_array, side="right") - 1
        doc_indices = np.maximum(doc_indices, 0)
        doc_starts = doc_offsets[doc_indices]
        doc_start_tokens = (doc_starts // token_width).astype(np.int64)

        next_doc_indices = doc_indices + 1
        within_bounds = next_doc_indices < len(doc_offsets)
        doc_ends = np.where(
            within_bounds,
            doc_offsets[np.minimum(next_doc_indices, len(doc_offsets) - 1)],
            tok_size,
        )
        doc_ends = np.minimum(doc_ends, tok_size)

        available_tokens = (doc_ends.astype(np.int64) - doc_starts.astype(np.int64)) // token_width
        available_tokens = np.maximum(available_tokens, 0)
        n_tokens_per_entry = np.minimum(available_tokens, read_len)
        total_truncated += int(np.sum(n_tokens_per_entry < read_len))

        full_mask = n_tokens_per_entry >= read_len
        sequences = np.full((n, read_len), pad_token_id, dtype=dtype)

        n_full = int(np.sum(full_mask))
        if n_full > 0:
            full_offsets = doc_start_tokens[full_mask]
            col_offsets = np.arange(read_len, dtype=np.int64)
            gather_indices = full_offsets[:, None] + col_offsets[None, :]
            gather_indices = np.clip(gather_indices, 0, len(tok_tokens) - 1)
            sequences[full_mask] = tok_tokens[gather_indices]

        partial_indices = np.where(~full_mask)[0]
        for idx in partial_indices:
            t_off = int(doc_start_tokens[idx])
            n_tok = int(n_tokens_per_entry[idx])
            if n_tok > 0:
                sequences[idx, :n_tok] = tok_tokens[t_off : t_off + n_tok]

        all_sequences.append(sequences)

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
    """
    extensions, doc_info, truncated = get_extensions(
        index_dir=index_dir,
        segment_by_shard=segment_by_shard,
        query_len=query_len,
        ext_len=ext_len,
        token_width=token_width,
        sa_entry_bytes=sa_entry_bytes,
        pad_token_id=pad_token_id,
    )
    ext_tuples = [tuple(row) for row in extensions]
    counts = Counter(ext_tuples)
    if top_k:
        return Counter(dict(counts.most_common(top_k))), truncated
    return counts, truncated