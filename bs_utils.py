from __future__ import annotations


import time
from typing import Any, Dict, Optional, Tuple
from collections import deque
import numpy as np
from tqdm.auto import tqdm
import copy


def add_text_and_children_str(
    out_data: Dict[Tuple[int, ...], Dict[str, Any]],
    tokenizer,
    in_place: bool = True,
) -> Dict[Tuple[int, ...], Dict[str, Any]]:
    target = out_data if in_place else copy.deepcopy(out_data)

    for k, v in tqdm(target.items(), desc="Adding text and children_str", unit="node"):
        # Decoded text for this node
        v['text'] = tokenizer.decode(list(k))

        # Children as a dict of {token_id: decoded_string}
        v['children_str'] = {
            child_id: tokenizer.decode([child_id])
            for child_id in tqdm(v.get('children', []), desc=f"children of {k}", leave=False, unit="child")
        }

    return target


import pandas as pd

def get_leaf_df_from_out_data(out_data):
    leaves = {k: v for k, v in out_data.items() if not v.get('children')}
    rows = []
    for leaf_key in leaves:
        # reconstruct path: (), (t1,), (t1,t2,), ...
        path_keys = [leaf_key[:i] for i in range(len(leaf_key) + 1)]
        for node_key in path_keys:
            node = out_data.get(node_key)
            if node is None:
                continue
            rows.append({
                'leaf_id':       leaf_key,
                'node_key':      node_key,
                'depth':         node['depth'],
                'gemma_cos_sim':       node.get('gemma_cos_sim'),
                'gemma_delta_cos_sim': node.get('delta_gemma_cos_sim'),
                'qwen_small_cos_sim':       node.get('qwen_small_cos_sim'),
                'qwen_small_delta_cos_sim': node.get('delta_qwen_small_cos_sim'),
                'text':          node.get('text', ''),
            })

    df = pd.DataFrame(rows)
    return df


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def plot_cos_sim_analysis(df, outpath, prefix):
    """
    Plot cos_sim and delta_cos_sim analysis for a given model prefix.

    Args:
        df:      DataFrame with columns {prefix}_cos_sim, {prefix}_delta_cos_sim, depth, leaf_id
        outpath: filepath for the output PDF (e.g. 'analysis_gemma.pdf')
        prefix:  column prefix, e.g. 'gemma' or 'qwen_small'
    """
    cos_col   = f"{prefix}_cos_sim"
    delta_col = f"{prefix}_delta_cos_sim"

    with PdfPages(outpath) as pdf:

        # 1. Mean cos_sim by depth
        fig, ax = plt.subplots()
        df.groupby('depth')[cos_col].mean().plot(ax=ax)
        ax.set_title(f'[{prefix}] Mean cos_sim by depth')
        ax.set_xlabel('Depth')
        ax.set_ylabel('cos_sim')
        pdf.savefig(fig)
        plt.close(fig)

        # 2. Mean delta_cos_sim by depth
        fig, ax = plt.subplots()
        df.groupby('depth')[delta_col].mean().plot(ax=ax)
        ax.set_title(f'[{prefix}] Mean delta_cos_sim by depth')
        ax.set_xlabel('Depth')
        ax.set_ylabel('delta_cos_sim')
        pdf.savefig(fig)
        plt.close(fig)

        # 3. Distribution of cos_sim per depth (boxplot)
        fig, ax = plt.subplots()
        df.boxplot(column=cos_col, by='depth', ax=ax)
        ax.set_title(f'[{prefix}] cos_sim distribution by depth')
        fig.suptitle('')
        ax.set_xlabel('Depth')
        ax.set_ylabel('cos_sim')
        pdf.savefig(fig)
        plt.close(fig)

        # 4. Distribution of delta_cos_sim per depth (boxplot)
        fig, ax = plt.subplots()
        df.boxplot(column=delta_col, by='depth', ax=ax)
        ax.set_title(f'[{prefix}] delta_cos_sim distribution by depth')
        fig.suptitle('')
        ax.set_xlabel('Depth')
        ax.set_ylabel('delta_cos_sim')
        pdf.savefig(fig)
        plt.close(fig)

        # 5. Overall distribution of cos_sim (histogram + KDE)
        fig, ax = plt.subplots()
        df[cos_col].dropna().plot.hist(bins=50, density=True, alpha=0.6, ax=ax)
        df[cos_col].dropna().plot.kde(ax=ax)
        ax.set_title(f'[{prefix}] Overall cos_sim distribution')
        ax.set_xlabel('cos_sim')
        pdf.savefig(fig)
        plt.close(fig)

        # 6. Overall distribution of delta_cos_sim (histogram + KDE)
        fig, ax = plt.subplots()
        df[delta_col].dropna().plot.hist(bins=50, density=True, alpha=0.6, ax=ax)
        df[delta_col].dropna().plot.kde(ax=ax)
        ax.set_title(f'[{prefix}] Overall delta_cos_sim distribution')
        ax.set_xlabel('delta_cos_sim')
        pdf.savefig(fig)
        plt.close(fig)

        pdf.infodict().update({'Title': f'Cosine Similarity Analysis — {prefix}'})

    print(f"Saved {outpath}")


def plot_top_leaves_trajectories(df, outpath, prefix, top_ns=(100, 1000, 10000)):
    cos_col = f"{prefix}_cos_sim"

    leaf_final = (
        df[df['leaf_id'].notna() & (df['depth'] > 0)]
        .loc[lambda x: x.groupby('leaf_id')['depth'].transform('max') == x['depth']]
        .set_index('leaf_id')[cos_col]
        .sort_values(ascending=False)
    )

    with PdfPages(outpath) as pdf:
        for n in top_ns:
            if n > len(leaf_final):
                print(f"Skipping top-{n}: only {len(leaf_final)} leaves available")
                continue

            top_leaves = leaf_final.iloc[:n].index
            subset = df[df['leaf_id'].isin(top_leaves)]
            all_traj = df.groupby('depth')[cos_col].mean()
            traj = subset.groupby('depth')[cos_col].agg(['mean', 'std', 'median'])

            # 1. Mean/median trajectory with std band
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(traj.index, traj['mean'],   label='mean',   color='steelblue')
            ax.plot(traj.index, traj['median'], label='median', color='orange', linestyle='--')
            ax.fill_between(
                traj.index,
                traj['mean'] - traj['std'],
                traj['mean'] + traj['std'],
                alpha=0.2, color='steelblue', label='±1 std'
            )
            ax.set_title(f'[{prefix}] cos_sim trajectory — top {n} leaves')
            ax.set_xlabel('Depth')
            ax.set_ylabel('cos_sim')
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)

            # 2. Boxplot per depth
            fig, ax = plt.subplots(figsize=(8, 5))
            subset.boxplot(column=cos_col, by='depth', ax=ax)
            ax.set_title(f'[{prefix}] cos_sim spread by depth — top {n} leaves')
            fig.suptitle('')
            ax.set_xlabel('Depth')
            ax.set_ylabel('cos_sim')
            pdf.savefig(fig)
            plt.close(fig)

            # 3. Top-n vs all leaves
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(all_traj.index, all_traj.values, label='all leaves',      color='grey',      linestyle='--')
            ax.plot(traj.index,     traj['mean'],     label=f'top {n} leaves', color='steelblue')
            ax.set_title(f'[{prefix}] Top {n} vs all leaves — mean cos_sim by depth')
            ax.set_xlabel('Depth')
            ax.set_ylabel('cos_sim')
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)

            # 4. Quantile bands
            quantiles = (
                subset.groupby('depth')[cos_col]
                .quantile([0.1, 0.25, 0.5, 0.75, 0.9])
                .unstack(level=-1)
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(quantiles.index, quantiles[0.5], label='median (p50)', color='steelblue', linewidth=2)
            ax.fill_between(quantiles.index, quantiles[0.25], quantiles[0.75],
                            alpha=0.3, color='steelblue', label='p25–p75')
            ax.fill_between(quantiles.index, quantiles[0.1],  quantiles[0.9],
                            alpha=0.15, color='steelblue', label='p10–p90')
            ax.set_title(f'[{prefix}] Quantile bands — top {n} leaves')
            ax.set_xlabel('Depth')
            ax.set_ylabel('cos_sim')
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)

        pdf.infodict().update({'Title': f'Top Leaves Trajectory Analysis — {prefix}'})

    print(f"Saved {outpath}")


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
    sparse_model = False
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
    if not sparse_model:
        ref = st_model.encode(reference_text, normalize_embeddings=normalize_embeddings, show_progress_bar=False)
        ref = np.asarray(ref, dtype=np.float32)
    else:
        ref = st_model.encode(reference_text, show_progress_bar=False)

    # sort keys so parents come before children (by length)
    keys = sorted(target.keys(), key=len)

    # group keys by length (depth-like)
    by_len: Dict[int, list[Tuple[int, ...]]] = {}
    for k in keys:
        by_len.setdefault(len(k), []).append(k)

    # helper: compute cosine scores from embeddings
    def cos_from_embs(embs: np.ndarray) -> np.ndarray:
        if sparse_model:
            return embs @ ref
        elif normalize_embeddings:
            # normalized vectors -> cosine = dot with normalized ref (already normalized by st_model)
            return embs @ ref
        else:
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


def get_tokens_from_splade(sparse_embedding, tokenizer):
    """Extract tokens and weights from a SPLADE sparse embedding.
    
    Args:
        sparse_embedding: PyTorch sparse tensor (e.g. shape [1, vocab_size])
        tokenizer: the matching tokenizer (e.g. from naver/splade-cocondenser-ensembledistil)
    
    Returns:
        List of (token, weight) tuples sorted by weight descending.
    """
    sparse = sparse_embedding.coalesce()
    indices = sparse.indices()
    values = sparse.values()

    # indices shape is [ndim, nnz] — grab the vocab dimension (last row)
    token_ids = indices[-1].tolist()
    weights = values.tolist()

    token_weight_pairs = [
        (tokenizer.convert_ids_to_tokens(tid), w)
        for tid, w in zip(token_ids, weights)
    ]

    token_weight_pairs.sort(key=lambda x: -x[1])
    return token_weight_pairs


def get_text_by_rank(engine, tokenizer, shard,rank,max_display_len=100):
    return tokenizer.decode(engine.get_doc_by_rank(s=shard, rank=rank, max_disp_len=max_display_len)['token_ids'])