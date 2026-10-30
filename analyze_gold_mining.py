"""
analyze_gold_mining.py -- diagnostics over the outputs of gold_mining.py.

  1. Cost breakdown of the oracle traces: how much recall each budget band buys,
     how many documents 1k of count buys there, and what kind of steps are taken.
  2. Worst queries: the queries where the mined held-out oracle does worst, with
     their size, grade mix, BM25 recall and the terms that were selected.

Usage:
    python analyze_gold_mining.py --out_dir ./gold_mining --bm25_pool_pkl ./bm_25_baseline_results_large_pool.pkl
Writes <out_dir>/diagnostics.md and <out_dir>/worst_queries.csv and prints both.
"""

import argparse
import json
import os
import pickle

import pandas as pd

BANDS = [0, 1_000, 5_000, 10_000, 25_000, 40_000, 100_000]


def to_md(df, floatfmt=".3f"):
    """Markdown table from a DataFrame (no dependency on the optional `tabulate` package)."""
    def cell(v):
        if isinstance(v, float):
            return format(v, floatfmt)
        return str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)


def load_runs(out_dir, pool, data_dir=None):
    """Per-query summary rows and all oracle steps of one pool."""
    queries, steps = [], []
    for qid in sorted(os.listdir(out_dir)):
        qdir = os.path.join(out_dir, qid)
        if not os.path.exists(os.path.join(qdir, "summary.json")):
            continue
        with open(os.path.join(qdir, "summary.json")) as f:
            s = json.load(f)
        docs_path = os.path.join(data_dir or out_dir, qid, "relevant_docs.jsonl")
        grades = [json.loads(l)["grade"] for l in open(docs_path)]
        trace = pd.read_csv(os.path.join(qdir, f"{pool}_oracle.csv"))
        sel = pd.read_csv(os.path.join(qdir, f"{pool}_selected.csv"))
        queries.append({
            "qid": qid, "query": s["query"], "n_relevant": s["n_relevant"],
            "share_grade2plus": sum(g >= 2 for g in grades) / len(grades),
            **{f"{p}@{int(B) // 1000}k": r for p, v in s["recall_at"].items() for B, r in v.items()},
            "final_cost": int(trace["cum_cost"].max()) if len(trace) else 0,
            "final_recall": float(trace["recall"].max()) if len(trace) else 0.0,
            "n_steps": len(trace),
            "top_terms": "; ".join(sel.sort_values("H", ascending=False)["query"].head(5)),
        })
        if len(trace):
            trace["qid"] = qid
            trace["n_relevant"] = s["n_relevant"]
            steps.append(trace)
    return pd.DataFrame(queries), pd.concat(steps, ignore_index=True)


def add_bm25(queries, pkl_path):
    if not pkl_path or not os.path.exists(pkl_path):
        queries["bm25"] = float("nan")
        return queries
    with open(pkl_path, "rb") as f:
        bm25 = {r["qid"]: r["recall"] for r in pickle.load(f)}
    queries["bm25"] = queries["qid"].map(bm25)
    return queries


def cost_breakdown(steps, n_queries):
    """Recall bought per budget band (band = where the step's cumulative cost ends)."""
    steps = steps.copy()
    steps["band"] = pd.cut(steps["cum_cost"], BANDS, right=True,
                           labels=[f"{a // 1000}k-{b // 1000}k" for a, b in zip(BANDS, BANDS[1:])])
    steps["recall_gain"] = steps["new_docs"] / steps["n_relevant"]
    rows = []
    for band, g in steps.groupby("band", observed=False):
        if not len(g):
            continue
        rows.append({
            "band": band,
            "steps / query": len(g) / n_queries,
            "recall gained (mean / query)": g["recall_gain"].sum() / n_queries,
            "rel. docs per 1k count": 1000 * g["new_docs"].sum() / g["count"].sum(),
            "median step count": int(g["count"].median()),
            "median new docs / step": g["new_docs"].median(),
            "share pairs": (g["kind"] == "pair").mean(),
            "share triples": (g["kind"] == "triple").mean(),
            "share steps with <= 2 new docs": (g["new_docs"] <= 2).mean(),
        })
    return pd.DataFrame(rows)


def stop_breakdown(queries):
    """Where the oracle stopped: far below the 100k budget = no affordable candidate added anything."""
    bins = [-1, 10_000, 25_000, 50_000, 90_000, 100_000]
    labels = ["< 10k", "10k-25k", "25k-50k", "50k-90k", "90k-100k"]
    q = queries.copy()
    q["stopped at"] = pd.cut(q["final_cost"], bins, labels=labels)
    return q.groupby("stopped at", observed=False).agg(
        queries=("qid", "count"),
        mean_final_recall=("final_recall", "mean"),
        mean_n_relevant=("n_relevant", "mean"),
    ).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./gold_mining", help="results of the run to analyse")
    ap.add_argument("--data_dir", default="./gold_mining", help="where relevant_docs.jsonl lives")
    ap.add_argument("--pool", default="mined_heldout")
    ap.add_argument("--bm25_pool_pkl", default="./bm_25_baseline_results_large_pool.pkl")
    ap.add_argument("--n_worst", type=int, default=30)
    args = ap.parse_args()

    queries, steps = load_runs(args.out_dir, args.pool, args.data_dir)
    queries = add_bm25(queries, args.bm25_pool_pkl)
    n = len(queries)
    p = args.pool
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 70)

    out = [f"# Diagnostics for {p} ({n} queries)", ""]

    out += ["## 1. Cost breakdown of the oracle traces", "",
            to_md(cost_breakdown(steps, n), floatfmt=".3f"), "",
            "Where the oracle stopped (summed count at its last step; budget is 100k):", "",
            to_md(stop_breakdown(queries), floatfmt=".3f"), ""]

    corr = queries[[f"{p}@25k", "n_relevant", "share_grade2plus", "bm25"]].corr(method="spearman")[f"{p}@25k"]
    bm = queries.dropna(subset=["bm25"])
    out += ["## 2. Which queries fail", "",
            f"Spearman correlation of {p}@25k with: number of relevant segments {corr['n_relevant']:.2f}, "
            f"share of grade >= 2 {corr['share_grade2plus']:.2f}, BM25 recall {corr['bm25']:.2f}.", "",
            f"Mean BM25 recall (25.7k unique docs): {bm['bm25'].mean():.3f}. "
            f"{p}@25k >= BM25 on {(bm[f'{p}@25k'] >= bm['bm25']).sum()} of {len(bm)} queries, "
            f"{p}@40k >= BM25 on {(bm[f'{p}@40k'] >= bm['bm25']).sum()}.", ""]

    queries["size_quartile"] = pd.qcut(queries["n_relevant"], 4, labels=["Q1 (fewest)", "Q2", "Q3", "Q4 (most)"])
    by_size = queries.groupby("size_quartile", observed=False).agg(
        relevant=("n_relevant", "median"), mined_25k=(f"{p}@25k", "mean"),
        insample_25k=("mined_insample@25k", "mean"), opus_25k=("opus_heldout@25k", "mean"),
        bm25=("bm25", "mean")).reset_index()
    out += ["Mean recall at 25k by number of relevant segments:", "",
            to_md(by_size, floatfmt=".3f"), ""]

    worst = queries.sort_values(f"{p}@25k").head(args.n_worst)
    cols = ["qid", "query", "n_relevant", "share_grade2plus", "bm25", f"{p}@25k", "mined_insample@25k",
            "opus_heldout@25k", f"{p}@100k", "final_cost", "n_steps", "top_terms"]
    out += [f"## Worst {args.n_worst} queries by {p}@25k", "",
            to_md(worst[cols], floatfmt=".2f"), ""]

    text = "\n".join(out)
    print(text)
    with open(os.path.join(args.out_dir, "diagnostics.md"), "w") as f:
        f.write(text)
    worst[cols].to_csv(os.path.join(args.out_dir, "worst_queries.csv"), index=False)
    queries.to_csv(os.path.join(args.out_dir, f"diagnostics_per_query_{p}.csv"), index=False)


if __name__ == "__main__":
    main()
