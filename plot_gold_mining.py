"""
plot_gold_mining.py -- where does the oracle's recall curve flatten out?

Two panels over the cumulative index count, from the oracle traces of one run:
  top:    mean recall (with interquartile band) of the chosen pool, linear count axis
  bottom: marginal precision of the oracle's steps = new relevant docs per 1,000 counts,
          per step (dots) and aggregated per cost bin (line), with the threshold tau.
The x where the binned marginal precision drops below tau is marked in both panels.

Usage:
    python plot_gold_mining.py --out_dir ./gold_mining --pool mined_heldout --tau 5
Writes <out_dir>/figures/tail_<pool>.png and .pdf and prints the numbers behind the figure.
"""

import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, BLUE_DARK, ORANGE = "#2a78d6", "#1c5aa6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_traces(out_dir, pool):
    traces = []
    for qid in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, qid, f"{pool}_oracle.csv")
        if os.path.exists(path):
            t = pd.read_csv(path)
            t["qid"] = qid
            traces.append(t)
    return traces


def recall_curves(traces, grid):
    """Recall of every query at every budget on the grid (step function of the trace)."""
    rows = []
    for t in traces:
        if len(t) == 0:
            rows.append(np.zeros(len(grid)))
            continue
        idx = np.searchsorted(t["cum_cost"].to_numpy(), grid, side="right") - 1
        rec = t["recall"].to_numpy()
        rows.append(np.where(idx >= 0, rec[np.clip(idx, 0, None)], 0.0))
    return np.vstack(rows)


def binned_precision(steps, edges):
    """Aggregated marginal precision per cost bin: 1000 * sum(new docs) / sum(count)."""
    b = np.digitize(steps["cum_cost"], edges) - 1
    out = []
    for i in range(len(edges) - 1):
        g = steps[b == i]
        if len(g) >= 20:
            out.append((np.sqrt(edges[i] * edges[i + 1]), 1000 * g["new_docs"].sum() / g["count"].sum(), len(g)))
    return pd.DataFrame(out, columns=["x", "precision", "n_steps"])


def crossing(binned, tau):
    """First x (log-interpolated) where the binned precision falls below tau."""
    for (x0, p0), (x1, p1) in zip(binned[["x", "precision"]].values, binned[["x", "precision"]].values[1:]):
        if p0 >= tau > p1:
            f = (np.log(p0) - np.log(tau)) / (np.log(p0) - np.log(p1))
            return float(np.exp(np.log(x0) + f * (np.log(x1) - np.log(x0))))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./gold_mining")
    ap.add_argument("--pool", default="mined_heldout")
    ap.add_argument("--tau", type=float, default=5.0, help="threshold in relevant docs per 1,000 count")
    ap.add_argument("--bm25_pool_pkl", default="./bm_25_baseline_results_large_pool.pkl")
    args = ap.parse_args()

    traces = load_traces(args.out_dir, args.pool)
    steps = pd.concat([t for t in traces if len(t)], ignore_index=True)
    steps = steps[steps["count"] > 0].copy()
    steps["precision"] = 1000 * steps["new_docs"] / steps["count"]

    grid = np.logspace(0, 5, 200)
    curves = recall_curves(traces, grid)
    mean, q25, q75 = curves.mean(0), np.percentile(curves, 25, 0), np.percentile(curves, 75, 0)
    edges = np.logspace(0, 5, 21)
    binned = binned_precision(steps, edges)
    x_star = crossing(binned, args.tau)

    bm25 = None
    if args.bm25_pool_pkl and os.path.exists(args.bm25_pool_pkl):
        qids = {t["qid"].iloc[0] if len(t) else None for t in traces}
        with open(args.bm25_pool_pkl, "rb") as f:
            rows = [r for r in pickle.load(f) if r["qid"] in qids or not qids]
        bm25 = (np.mean([r["n_retrieved"] for r in rows]), np.mean([r["recall"] for r in rows]))

    # ---- figure ----
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
                         "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 7.0),
                                   gridspec_kw={"height_ratios": [1, 1], "hspace": 0.38})
    lin = np.linspace(0, 1e5, 1001)
    curves_lin = recall_curves(traces, lin)
    mean_lin = curves_lin.mean(0)
    q25_lin, q75_lin = np.percentile(curves_lin, 25, 0), np.percentile(curves_lin, 75, 0)

    ax1.fill_between(lin, q25_lin, q75_lin, color=BLUE, alpha=0.13, linewidth=0, label="interquartile range")
    ax1.plot(lin, mean_lin, color=BLUE, linewidth=2, label="mean recall (oracle, mined held-out)")
    ax1.set_xlim(0, 1e5)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax1.set_xlabel("Cumulative index count of the oracle set (summed, linear scale)")
    if bm25:
        ax1.plot([bm25[0]], [bm25[1]], marker="o", markersize=8, color=ORANGE, markeredgecolor="white",
                 markeredgewidth=1.5, linestyle="none", label=f"BM25: {bm25[1]:.2f} at {bm25[0]/1000:.1f}k unique docs")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Recall (grade ≥ 1)")
    ax1.grid(axis="y", color=GRID, linewidth=0.8)
    ax1.legend(loc="lower right", frameon=False, fontsize=9)

    dots = ax2.scatter(steps["cum_cost"], steps["precision"], s=5, color=INK_2, alpha=0.10, linewidths=0,
                       label="single oracle step")
    ax2.plot(binned["x"], binned["precision"], color=BLUE_DARK, linewidth=2, marker="o", markersize=4,
             label="aggregated per cost bin")
    ax2.axhline(args.tau, color=ORANGE, linewidth=1.25, linestyle="--")
    ax2.text(1.3, args.tau * 1.15, f"τ = {args.tau:g} per 1,000 ({args.tau / 10:g}% marginal precision)",
             color=INK, fontsize=9)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(0.1, 1000)
    ax2.set_xlim(1, 1e5)
    ax2.set_ylabel("New relevant docs per 1,000 count")
    ax2.set_xlabel("Cumulative index count of the oracle set (summed, log scale)")
    ax2.grid(axis="y", color=GRID, linewidth=0.8)
    leg = ax2.legend(loc="lower left", frameon=False, fontsize=9)
    leg.legend_handles[0].set_alpha(0.6)

    if x_star:
        r_star = float(np.interp(np.log(x_star), np.log(grid), mean))
        for ax in (ax1, ax2):
            ax.axvline(x_star, color=INK_2, linewidth=1, linestyle=":")
        ax1.annotate(f"{x_star/1000:.1f}k count: mean recall {r_star:.2f}", xy=(x_star, r_star),
                     xytext=(x_star + 9000, r_star - 0.2), fontsize=9, color=INK,
                     arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8})
        ax2.text(x_star * 1.15, 300, f"{x_star/1000:.1f}k", fontsize=9, color=INK_2)

    if x_star:
        share_cost = 1 - steps.loc[steps["cum_cost"] > x_star, "count"].sum() / steps["count"].sum()
        share_rec = float(np.interp(np.log(x_star), np.log(grid), mean)) / float(mean[-1])
        title = (f"The first {share_cost:.0%} of the count buy {share_rec:.0%} of the recall; "
                 f"after that, steps fall below τ = {args.tau:g} relevant docs per 1,000")
    else:
        title = "Oracle recall and marginal precision"
    fig.suptitle(title, x=0.08, ha="left", fontsize=11, color=INK)
    fig.text(0.08, 0.93, f"{len(traces)} TREC RAG 24 topics, pool {args.pool}, run 1 (pairs only). "
             "Top: linear count axis. Bottom: log-log.", fontsize=9, color=INK_2)

    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    base = os.path.join(fig_dir, f"tail_{args.pool}")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")

    # ---- numbers behind the figure ----
    print(binned.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    if x_star:
        r_star = float(np.interp(np.log(x_star), np.log(grid), mean))
        r_end = float(mean[-1])
        after = steps[steps["cum_cost"] > x_star]
        print(f"\nbinned marginal precision falls below tau={args.tau:g} at {x_star:,.0f} summed count")
        print(f"mean recall there {r_star:.3f}; at 100k {r_end:.3f} "
              f"-> {r_star / r_end:.0%} of the final recall is reached before tau")
        print(f"steps after tau: {len(after) / len(steps):.0%} of all steps, "
              f"{after['count'].sum() / steps['count'].sum():.0%} of all count, "
              f"{after['new_docs'].sum() / steps['new_docs'].sum():.0%} of all covered relevant docs")
    print(f"\nsaved {base}.png and .pdf")


if __name__ == "__main__":
    main()
