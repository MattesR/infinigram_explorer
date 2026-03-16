from __future__ import annotations


import time
from typing import Any, Dict, Optional, Tuple
from collections import deque
import numpy as np
from tqdm.auto import tqdm
import copy





def add_cos_sim_to_out_data_batched(
    out_data: Dict[Tuple[int, ...], Dict[str, Any]],
    tokenizer,
    st_model,                      # preloaded SentenceTransformer
    reference_text: str,
    *,
    prefix_text: str = "",
    normalize_embeddings: bool = True,
    batch_size: int = 128,
    in_place: bool = True,
    model_prefix = None,
) -> Dict[Tuple[int, ...], Dict[str, Any]]:
    """
    Post-hoc: add cos_sim + delta_cos_sim to out_data, but encode in batches.

    Adds per node:
      - "cos_sim": cosine(node_text, reference)
      - "delta_cos_sim": cos_sim(node) - cos_sim(parent) (None for root)

    Assumptions:
      - out_data keys are token-id tuples
      - parent key is key[:-1]
      - root is the shortest key (or any key with parent missing)
    """

    target = out_data if in_place else copy.deepcopy(out_data)
    cos_sim_prefix = f'{model_prefix}_cos_sim' if model_prefix else 'cos_sim'
    # reference embedding once
    ref = st_model.encode(reference_text, normalize_embeddings=normalize_embeddings, show_progress_bar=False)
    ref = np.asarray(ref, dtype=np.float32)

    # sort keys so parents come before children (by length)
    keys = sorted(target.keys(), key=len)

    # group keys by length (depth-like)
    by_len: Dict[int, list[Tuple[int, ...]]] = {}
    for k in keys:
        by_len.setdefault(len(k), []).append(k)

    # helper: compute cosine scores from embeddings
    def cos_from_embs(embs: np.ndarray) -> np.ndarray:
        if normalize_embeddings:
            # normalized vectors -> cosine = dot with normalized ref (already normalized by st_model)
            return embs @ ref
        ref_norm = np.linalg.norm(ref)
        emb_norms = np.linalg.norm(embs, axis=1)
        denom = emb_norms * ref_norm
        # avoid divide-by-zero
        denom = np.where(denom == 0, 1e-12, denom)
        return (embs @ ref) / denom

    pbar = tqdm(sorted(by_len.items()), desc="cos_sim by length", unit="len")
    for _, layer_keys in pbar:
        # encode this layer in chunks
        for i in tqdm(range(0, len(layer_keys), batch_size), desc="encode batch", unit="batch", leave=False):
            chunk_keys = layer_keys[i : i + batch_size]

            texts = [prefix_text + tokenizer.decode(list(k)) for k in chunk_keys]
            embs = st_model.encode(
                texts,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=False,
            )
            embs = np.asarray(embs, dtype=np.float32)
            cos_sims = cos_from_embs(embs)

            # store cos_sim + delta
            for k, cs in zip(chunk_keys, cos_sims):
                node = target[k]
                node[cos_sim_prefix] = float(cs)

                parent = k[:-1] if len(k) > 1 else None
                if parent is not None and parent in target and target[parent].get(cos_sim_prefix) is not None:
                    node[f"delta_{cos_sim_prefix}"] = float(cs - target[parent][cos_sim_prefix])
                else:
                    node[f"delta_{cos_sim_prefix}"] = None

    return target



def convert_outdata_ids_to_tokens(out_data, tokenizer):
    """
    Convert token-id tuples in out_data to readable tokens/strings.

    Adds:
        - token_str_seq        : decoded full sequence
        - added_token_str      : decoded last token
        - token_piece_seq      : list of decoded token pieces

    Returns:
        new_out_data (dict) with identical structure + readable fields
    """

    new_out_data = {}

    for token_ids, node in out_data.items():

        token_ids_list = list(token_ids)

        token_piece_seq = [
            tokenizer.decode([tid])
            for tid in token_ids_list
        ]

        token_str_seq = tokenizer.decode(token_ids_list)

        added_token_str = (
            tokenizer.decode([token_ids_list[-1]])
            if len(token_ids_list) > 0
            else None
        )

        new_node = dict(node)
        new_node["token_ids"] = token_ids_list
        new_node["token_piece_seq"] = token_piece_seq
        new_node["token_str_seq"] = token_str_seq
        new_node["added_token_str"] = added_token_str

        new_out_data[token_ids] = new_node

    return new_out_data



def collect_infgram_ntd_tree(
    engine,
    tokenizer,
    query: str,
    *,
    max_depth: int,
    use_infgram: bool = True,
    max_support: Optional[int] = None,
    ban_token_ids: Optional[set[int]] = None,
    store_full_out: bool = True,
) -> Dict[Tuple[int, ...], Dict[str, Any]]:
    """
    Build an NTD expansion tree up to max_depth (inclusive), starting from `query`.

    For each node (a prompt token-id sequence), run ntd/infgram_ntd and create children
    for ALL continuation tokens in out["result_by_token_id"] (minus banned tokens).

    Returns:
        out_data: dict keyed by token-id tuples (the full prompt_ids at that node).
                  Each value is a dict with timing + metadata + optional full out.

    Keying scheme (beam lineage):
        - The dict key IS the full token id tuple for that node.
        - That uniquely determines its parent (key[:-1]) and the extension token (key[-1]).

    WARNING:
        This can explode combinatorially. You requested "all extensions", so no pruning is done.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")

    fn = engine.infgram_ntd if use_infgram else engine.ntd
    kwargs = {} if max_support is None else {"max_support": max_support}
    banned = ban_token_ids or set()

    root_ids: Tuple[int, ...] = tuple(tokenizer.encode(query, add_special_tokens=False))

    out_data: Dict[Tuple[int, ...], Dict[str, Any]] = {}

    # BFS queue: (prompt_ids_tuple, depth)
    q = deque([(root_ids, 0)])

    pbar = tqdm(desc="NTD nodes processed", unit="node")
    try:
        while q:
            prompt_ids, depth = q.popleft()

            # Time the call
            t0 = time.perf_counter_ns()
            out = fn(prompt_ids=list(prompt_ids), **kwargs)
            t1 = time.perf_counter_ns()
            dur_ns = t1 - t0

            token_dict = out.get("result_by_token_id", {}) or {}
            # filter tokens (keep full dict for accuracy, but build children list from filtered keys)
            child_token_ids = [int(tok) for tok in token_dict.keys() if int(tok) not in banned]

            node_rec: Dict[str, Any] = {
                "depth": depth,
                "duration_ns": int(dur_ns),
                "duration_ms": float(dur_ns) / 1e6,
                "vocab_size": int(len(token_dict)),
                "vocab_size_filtered": int(len(child_token_ids)),
                "parent": prompt_ids[:-1] if len(prompt_ids) > 0 else None,
                "added_token_id": prompt_ids[-1] if len(prompt_ids) > 0 else None,
                "children": [],  # list of child keys (token-id tuples)
            }
            if store_full_out:
                node_rec["out"] = out

            out_data[prompt_ids] = node_rec

            # Enqueue children if we can go deeper
            if depth < max_depth:
                # Create child nodes for ALL continuation tokens
                for tok in child_token_ids:
                    child = prompt_ids + (tok,)
                    node_rec["children"].append(child)
                    q.append((child, depth + 1))

            pbar.update(1)
            pbar.set_postfix(ms=f"{dur_ns/1e6:.2f}", V=len(token_dict))
    finally:
        pbar.close()

    return out_data
