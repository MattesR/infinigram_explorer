"""
Cluster SPLADE tokens by shared WordPiece stems.

WordPiece tokenizers (like BERT's) produce tokens like:
  'community', 'communities', '##ity'
  'civilized', 'civilization', 'civil'
  'science', 'scientific', 'scientist'

We can group these by longest shared prefix to form OR-clauses
in CNF queries, without any embedding step.

Usage:
    from wordpiece_cluster import cluster_tokens, show_clusters
    clusters = cluster_tokens(splade_tokens, min_stem_len=5)
    show_clusters(clusters)
"""

from collections import defaultdict


def _normalize_token(token: str) -> str:
    """Strip WordPiece prefix markers."""
    return token.lstrip("#")


def _get_stems(word: str, min_stem_len: int = 4) -> list[str]:
    """
    Generate candidate stems from a word by taking prefixes
    of decreasing length. E.g. 'community' -> ['community', 'communit', 'communi', 'commun', 'commu', 'comm']
    """
    word = word.lower()
    return [word[:i] for i in range(len(word), min_stem_len - 1, -1)]


def cluster_tokens(
    splade_tokens: list[tuple[str, float]],
    min_stem_len: int = 5,
    show_steps: bool = True,
) -> list[dict]:
    """
    Cluster SPLADE tokens by shared stems using a greedy approach:
    1. For each token, generate all possible stems (prefixes)
    2. For each stem, collect all tokens that share it
    3. Greedily merge: pick the longest stem that groups 2+ tokens,
       assign those tokens, remove them, repeat
    4. Remaining singletons form their own clusters

    Args:
        splade_tokens: List of (token_string, score) tuples.
        min_stem_len: Minimum prefix length to consider as a shared stem.
        show_steps: Print detailed steps.

    Returns:
        List of cluster dicts, each with:
            - 'stem': the shared stem (or full token for singletons)
            - 'tokens': list of (token_string, score) in this cluster
            - 'max_score': highest SPLADE score in cluster
    """
    # Normalize tokens
    token_data = []
    for token, score in splade_tokens:
        normalized = _normalize_token(token)
        token_data.append({
            "original": token,
            "normalized": normalized,
            "score": score,
        })

    if show_steps:
        print(f"Clustering {len(token_data)} tokens with min_stem_len={min_stem_len}")
        print()

    # Build stem -> tokens mapping
    stem_to_tokens = defaultdict(set)
    for i, td in enumerate(token_data):
        for stem in _get_stems(td["normalized"], min_stem_len):
            stem_to_tokens[stem].add(i)

    # Sort stems by: number of tokens covered (desc), then stem length (desc)
    # This prioritizes stems that group the most tokens with the longest match
    stem_candidates = [
        (stem, indices)
        for stem, indices in stem_to_tokens.items()
        if len(indices) >= 2
    ]
    stem_candidates.sort(key=lambda x: (len(x[1]), len(x[0])), reverse=True)

    if show_steps:
        print("Stem candidates (grouping 2+ tokens):")
        for stem, indices in stem_candidates[:20]:
            tokens_str = ", ".join(token_data[i]["original"] for i in indices)
            print(f"  '{stem}' ({len(stem)} chars) -> [{tokens_str}]")
        if len(stem_candidates) > 20:
            print(f"  ... and {len(stem_candidates) - 20} more")
        print()

    # Greedy assignment
    assigned = set()
    clusters = []

    for stem, indices in stem_candidates:
        # Only consider unassigned tokens
        available = indices - assigned
        if len(available) < 2:
            continue

        cluster_tokens_list = [
            (token_data[i]["original"], token_data[i]["score"])
            for i in sorted(available)
        ]
        cluster_tokens_list.sort(key=lambda x: x[1], reverse=True)

        clusters.append({
            "stem": stem,
            "tokens": cluster_tokens_list,
            "max_score": max(s for _, s in cluster_tokens_list),
        })
        assigned.update(available)

        if show_steps:
            tokens_str = ", ".join(f"{t}({s:.2f})" for t, s in cluster_tokens_list)
            print(f"  Cluster: '{stem}' -> [{tokens_str}]")

    # Add singletons
    if show_steps and assigned != set(range(len(token_data))):
        print()
        print("Singletons (no shared stem found):")

    for i, td in enumerate(token_data):
        if i not in assigned:
            clusters.append({
                "stem": td["normalized"],
                "tokens": [(td["original"], td["score"])],
                "max_score": td["score"],
            })
            if show_steps:
                print(f"  '{td['original']}' ({td['score']:.2f})")

    # Sort clusters by max score
    clusters.sort(key=lambda x: x["max_score"], reverse=True)

    if show_steps:
        print()
        print(f"Result: {len(clusters)} clusters "
              f"({sum(1 for c in clusters if len(c['tokens']) > 1)} multi-token, "
              f"{sum(1 for c in clusters if len(c['tokens']) == 1)} singletons)")

    return clusters


def show_clusters(clusters: list[dict]):
    """Pretty-print clusters."""
    print(f"\n{'='*60}")
    print(f"{'Stem':<20s} {'Tokens':<50s} {'Max Score':>10s}")
    print(f"{'='*60}")

    for c in clusters:
        tokens_str = " | ".join(f"{t}({s:.2f})" for t, s in c["tokens"])
        marker = "  " if len(c["tokens"]) == 1 else "* "
        print(f"{marker}{c['stem']:<18s} {tokens_str:<50s} {c['max_score']:>10.3f}")


def clusters_to_or_clauses(clusters: list[dict], tokenizer) -> list[dict]:
    """
    Convert clusters to OR-clause token IDs ready for CNF queries.

    Returns:
        List of dicts with:
            - 'stem': cluster stem
            - 'clause': list of token ID lists (OR alternatives)
            - 'max_score': highest SPLADE score
            - 'tokens': original token strings
    """
    or_clauses = []

    for c in clusters:
        alternatives = []
        for token, score in c["tokens"]:
            clean = _normalize_token(token)
            ids = tokenizer.encode(clean, add_special_tokens=False)
            if ids:
                alternatives.append(ids)

        if alternatives:
            or_clauses.append({
                "stem": c["stem"],
                "clause": alternatives,
                "max_score": c["max_score"],
                "tokens": [t for t, s in c["tokens"]],
            })

    return or_clauses