"""
Recalculate metrics from saved pipeline results.

Usage:
    from recalc_metrics import recalc_from_pickle, recalc_all_runs

    # Single run
    df = recalc_from_pickle("./runs/exp01/results.pkl", "./qrels.txt")

    # All runs in a directory
    df = recalc_all_runs("./runs/", "./qrels.txt")
"""

import os
import pickle
import pandas as pd
from glob import glob
from pathlib import Path
from trec_output import load_qrels
from ir_metrics import compute_metrics


def recalc_from_pickle(
    results_path: str,
    qrels_path: str,
    stages: list = None,
    top_k_list: list = None,
    output_csv: str = None,
    verbose: bool = True,
):
    """
    Recalculate metrics from a saved results.pkl file.

    Args:
        results_path: Path to results.pkl from pipeline.
        qrels_path: Path to qrels file.
        stages: Which stages to eval. Default: all available.
        top_k_list: Cutoffs. Default: [10, 100, 1000].
        output_csv: If set, save CSV to this path.

    Returns DataFrame with per-query, per-stage metrics.
    """
    if top_k_list is None:
        top_k_list = [10, 100, 1000]

    if stages is None:
        stages = ["pool", "biencoder", "crossencoder"]

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    qrels = load_qrels(qrels_path)
    run_name = Path(results_path).parent.name

    rows = []
    n_skipped = 0

    for r in results:
        qid = r["qid"]
        relevant = {did: rel for did, rel in qrels.get(qid, {}).items() if rel > 0}

        # Common pool stats
        pool_ids = [d["doc_id"] for d in r.get("pool_docs", [])] if r.get("pool_docs") else []
        pool_found = len(set(pool_ids) & set(relevant.keys()))
        pool_recall = pool_found / len(relevant) if relevant else 0
        pool_size = len(pool_ids)

        # Pool
        if "pool" in stages and pool_ids:
            metrics = compute_metrics(pool_ids, qrels, qid, top_k_list=top_k_list)

            row = {
                "run": run_name,
                "stage": "pool",
                "qid": qid,
                "n_relevant": len(relevant),
                "pool_found": pool_found,
                "pool_recall": pool_recall,
                "pool_size": pool_size,
                "mrr": metrics["mrr"],
                "map": metrics.get("map", 0),
            }
            for k in top_k_list:
                row[f"recall_{k}"] = metrics[k]["recall"]
                row[f"ndcg_{k}"] = metrics[k]["ndcg"]
                row[f"precision_{k}"] = metrics[k]["precision"]
                row[f"found_{k}"] = metrics[k]["found"]
            rows.append(row)

        # Bi-encoder
        if "biencoder" in stages and r.get("biencoder_ranked"):
            bi_ids = [d["doc_id"] for d in r["biencoder_ranked"]]
            bi_scores = [d.get("score", len(bi_ids) - i) for i, d in enumerate(r["biencoder_ranked"])]

            metrics = compute_metrics(bi_ids, qrels, qid, top_k_list=top_k_list,
                                       ranked_scores=bi_scores)

            row = {
                "run": run_name,
                "stage": "biencoder",
                "qid": qid,
                "n_relevant": len(relevant),
                "pool_found": pool_found,
                "pool_recall": pool_recall,
                "pool_size": pool_size,
                "n_input": r.get("pool_size", pool_size),
                "n_output": len(bi_ids),
                "mrr": metrics["mrr"],
                "map": metrics.get("map", 0),
            }
            for k in top_k_list:
                row[f"recall_{k}"] = metrics[k]["recall"]
                row[f"ndcg_{k}"] = metrics[k]["ndcg"]
                row[f"precision_{k}"] = metrics[k]["precision"]
                row[f"found_{k}"] = metrics[k]["found"]
            rows.append(row)

        # Cross-encoder
        if "crossencoder" in stages and r.get("crossencoder_ranked"):
            ce_ids = [d["doc_id"] for d in r["crossencoder_ranked"]]
            ce_scores = [d.get("score", len(ce_ids) - i) for i, d in enumerate(r["crossencoder_ranked"])]

            metrics = compute_metrics(ce_ids, qrels, qid, top_k_list=top_k_list,
                                       ranked_scores=ce_scores)

            row = {
                "run": run_name,
                "stage": "crossencoder",
                "qid": qid,
                "n_relevant": len(relevant),
                "pool_found": pool_found,
                "pool_recall": pool_recall,
                "pool_size": pool_size,
                "n_input": len(r.get("biencoder_ranked", [])),
                "n_output": len(ce_ids),
                "mrr": metrics["mrr"],
                "map": metrics.get("map", 0),
            }
            for k in top_k_list:
                row[f"recall_{k}"] = metrics[k]["recall"]
                row[f"ndcg_{k}"] = metrics[k]["ndcg"]
                row[f"precision_{k}"] = metrics[k]["precision"]
                row[f"found_{k}"] = metrics[k]["found"]
            rows.append(row)

        if not pool_ids and not r.get("biencoder_ranked") and not r.get("crossencoder_ranked"):
            n_skipped += 1
            continue

    df = pd.DataFrame(rows)

    if output_csv is None:
        output_csv = results_path.replace("results.pkl", "metrics.csv")

    df.to_csv(output_csv, index=False)

    if verbose:
        print(f"Recalculated metrics for {len(results)} queries from {results_path}")
        if n_skipped:
            print(f"  Skipped {n_skipped} queries with no results")
        print(f"Saved to {output_csv}")
        _print_summary(df, top_k_list)

    return df


def recalc_all_runs(
    base_dir: str,
    stages: list = None,
    top_k_list: list = None,
    qrels_filename: str = "qrels.txt",
    results_filename: str = "results.pkl",
    verbose: bool = True,
):
    """
    Recalculate metrics for all results.pkl files in a directory tree.
    Expects qrels to be in the same folder as results.pkl.

    Directory structure:
        base_dir/
            dataset1/
                qrels.txt
                results.pkl
            dataset2/
                qrels.txt
                results.pkl

    Returns combined DataFrame across all runs.
    """
    if top_k_list is None:
        top_k_list = [10, 100, 1000]

    pkl_files = sorted(glob(os.path.join(base_dir, f"**/{results_filename}"), recursive=True))

    if verbose:
        print(f"Found {len(pkl_files)} result files in {base_dir}")

    all_dfs = []
    for pkl_path in pkl_files:
        run_dir = Path(pkl_path).parent
        run_name = run_dir.name
        qrels_path = run_dir / qrels_filename

        if not qrels_path.exists():
            if verbose:
                print(f"  {run_name}: no {qrels_filename}, skipping")
            continue

        if verbose:
            print(f"\n  Processing: {run_name}")
        try:
            df = recalc_from_pickle(
                str(pkl_path), str(qrels_path),
                stages=stages, top_k_list=top_k_list,
                verbose=False,
            )
            all_dfs.append(df)
        except Exception as e:
            print(f"    ERROR: {e}")

    if not all_dfs:
        print("No results found")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Save combined CSV
    combined_path = os.path.join(base_dir, "all_metrics.csv")
    combined.to_csv(combined_path, index=False)

    if verbose:
        print(f"\n{'='*80}")
        print(f"COMBINED RESULTS ({len(pkl_files)} runs)")
        print(f"{'='*80}")
        _print_comparison(combined, top_k_list)
        print(f"\nSaved to {combined_path}")

    return combined


def _print_summary(df, top_k_list):
    """Print summary for a single run."""
    for stage in df["stage"].unique():
        sdf = df[df["stage"] == stage]
        n = len(sdf)
        print(f"\n  {stage.upper()} ({n} queries):")
        print(f"    MRR: {sdf['mrr'].mean():.4f}")
        print(f"    MAP: {sdf['map'].mean():.4f}")
        if "pool_recall" in sdf.columns:
            print(f"    Pool recall: {sdf['pool_recall'].mean():.4f}")
            print(f"    Avg pool size: {sdf['pool_size'].mean():.0f}")
        header = "    "
        for k in top_k_list:
            header += f"{'nDCG@'+str(k):>10s} {'R@'+str(k):>8s}"
        print(header)
        line = "    "
        for k in top_k_list:
            col_n = f"ndcg_{k}"
            col_r = f"recall_{k}"
            if col_n in sdf.columns:
                line += f"{sdf[col_n].mean():>10.4f} {sdf[col_r].mean():>8.4f}"
        print(line)


def _print_comparison(df, top_k_list):
    """Print comparison table across runs and stages."""
    primary_k = top_k_list[0] if top_k_list else 10

    print(f"\n{'Run':<25s} {'Stage':<15s} {'#Q':>5s} {'MRR':>7s} {'MAP':>7s}", end="")
    for k in top_k_list:
        print(f" {'nDCG@'+str(k):>9s}", end="")
    for k in top_k_list:
        print(f" {'R@'+str(k):>7s}", end="")
    print()
    print("-" * (60 + 10 * len(top_k_list) + 8 * len(top_k_list)))

    for run in df["run"].unique():
        for stage in df[df["run"] == run]["stage"].unique():
            sdf = df[(df["run"] == run) & (df["stage"] == stage)]
            n = len(sdf)
            print(f"{run:<25s} {stage:<15s} {n:>5d} "
                  f"{sdf['mrr'].mean():>7.4f} {sdf['map'].mean():>7.4f}", end="")
            for k in top_k_list:
                print(f" {sdf[f'ndcg_{k}'].mean():>9.4f}", end="")
            for k in top_k_list:
                print(f" {sdf[f'recall_{k}'].mean():>7.4f}", end="")
            print()