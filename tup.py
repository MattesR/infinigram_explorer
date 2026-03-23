"""
Compute token unigram probabilities (TUP) for all tokens in the vocabulary
using the infini-gram engine.

Usage:
    from token_unigram_prob import compute_tup, load_tup
    
    tup = compute_tup(engine, tokenizer, index_dir='../msmarco_segmented_index/')
    tup.to_parquet('tup.parquet')

    # Later:
    tup = load_tup('tup.parquet')
    print(tup.loc[tup['token'] == 'community'])
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def compute_tup(
    engine,
    tokenizer,
    index_dir: str = "../msmarco_segmented_index/",
    token_width: int = 2,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Compute token unigram probability for every token in the vocabulary.

    Args:
        engine: Infini-gram engine instance.
        tokenizer: Tokenizer (must have vocab_size or get_vocab()).
        index_dir: Path to index dir (to compute total token count from file sizes).
        token_width: Bytes per token (2 for 16-bit).
        save_path: If set, save the result to this path (parquet or csv).

    Returns:
        DataFrame with columns: token_id, token, count, probability
    """
    # Get total token count from tokenized file sizes
    index_dir = Path(index_dir)
    total_tokens = sum(
        f.stat().st_size // token_width
        for f in index_dir.glob("tokenized.*")
    )
    print(f"Total tokens in corpus: {total_tokens:,}")

    # Get vocab size
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size:,}")

    # Reverse mapping: id -> token string
    id_to_token = {v: k for k, v in vocab.items()}

    # Query each token
    print(f"Querying unigram counts for {vocab_size:,} tokens...")
    # Build numpy arrays: index = token_id, value = count / probability
    counts = np.zeros(vocab_size, dtype=np.int64)
    for token_id in tqdm(range(vocab_size), desc="Counting"):
        try:
            result = engine.find(input_ids=[token_id])
            counts[token_id] = result["cnt"]
        except Exception:
            counts[token_id] = 0

    probs = counts.astype(np.float64) / total_tokens

    # Save fast arrays
    np.save(str(Path(save_path).with_suffix('.counts.npy')) if save_path else 'tup.counts.npy', counts)
    np.save(str(Path(save_path).with_suffix('.probs.npy')) if save_path else 'tup.probs.npy', probs)
    print(f"Saved numpy arrays (.counts.npy, .probs.npy)")

    # Also build DataFrame for inspection
    id_to_token = {v: k for k, v in vocab.items()}
    df = pd.DataFrame({
        "token_id": np.arange(vocab_size),
        "token": [id_to_token.get(i, f"[UNK_{i}]") for i in range(vocab_size)],
        "count": counts,
        "probability": probs,
    })
    df = df.sort_values("count", ascending=False).reset_index(drop=True)

    print(f"\nTop 20 most frequent tokens:")
    print(df.head(20).to_string(index=False))

    print(f"\nBottom 20 (non-zero) tokens:")
    nonzero = df[df["count"] > 0]
    print(nonzero.tail(20).to_string(index=False))

    print(f"\nTokens with zero count: {(df['count'] == 0).sum():,}")

    if save_path:
        if save_path.endswith(".parquet"):
            df.to_parquet(save_path, index=False)
        else:
            df.to_csv(save_path, index=False)
        print(f"Saved DataFrame to {save_path}")

    return counts, probs, df


def load_tup(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load previously computed TUP arrays.
    
    Args:
        path: Base path (e.g. 'tup' or 'tup.parquet') — will look for
              .counts.npy and .probs.npy alongside it.
    
    Returns:
        (counts, probs) numpy arrays indexed by token_id.
    """
    base = Path(path).with_suffix('')
    counts = np.load(str(base) + '.counts.npy')
    probs = np.load(str(base) + '.probs.npy')
    return counts, probs


def load_tup_df(path: str) -> pd.DataFrame:
    """Load the DataFrame version for inspection."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def lookup_tup(tup_df: pd.DataFrame, tokens: list[str]) -> pd.DataFrame:
    """Look up TUP for a list of token strings."""
    return tup_df[tup_df["token"].isin(tokens)].sort_values("probability", ascending=True)


def filter_by_tup(
    splade_tokens: list[tuple[str, float]],
    probs: np.ndarray,
    tokenizer,
    max_probability: float = 0.001,
) -> list[tuple[str, float]]:
    """
    Filter SPLADE tokens, removing those with unigram probability above threshold.

    Uses the fast numpy array for lookup. Since SPLADE tokens come from BERT's
    vocabulary, we need to convert them to the infini-gram tokenizer's token IDs
    to look up their corpus probability.

    Args:
        splade_tokens: List of (token_string, score) from SPLADE.
        probs: Probability array from compute_tup (indexed by token_id).
        tokenizer: The infini-gram tokenizer (for encoding token strings to IDs).
        max_probability: Remove tokens with probability above this.

    Returns:
        Filtered list of (token_string, score).
    """
    kept = []
    removed = []
    for token, score in splade_tokens:
        clean = token.lstrip("#")
        ids = tokenizer.encode(clean, add_special_tokens=False)
        if not ids:
            kept.append((token, score))
            continue

        # Use first token's probability as proxy
        token_id = ids[0]
        prob = probs[token_id] if token_id < len(probs) else 0.0

        if prob <= max_probability:
            kept.append((token, score))
        else:
            removed.append((token, score, prob))

    if removed:
        print(f"Filtered out {len(removed)} high-frequency tokens (p > {max_probability}):")
        for token, score, prob in removed:
            print(f"  {token:>15s}  splade={score:.3f}  tup={prob:.6f}")

    print(f"Kept {len(kept)} tokens")
    return kept


def word_tup(word, tokenizer, probs):
    """
    Compute the token unigram probability for a word,
    as the product of its constituent token probabilities.
    """
    ids = tokenizer.encode(word, add_special_tokens=False)
    return np.prod([probs[tid] for tid in ids])
