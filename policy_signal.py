"""
policy_signal.py -- can the oracle's decisions be predicted from index counts?

Before training a model to "use counts", check in the mined data that counts carry the signal:
  1. Role of each selected term in the oracle set (used alone / inside pairs / inside triples / unused),
     by its index count. If roles change sharply with the count, a model that sees counts can learn them.
  2. Pair counts versus the product of the two piece counts. If they track each other, the model can
     anticipate how specific a pair is from the piece counts it sees, before any pair is counted.
  3. AUC of the count, and of the held-out coverage H (which the model never sees), for each decision.

Usage: python policy_signal.py --out_dir ./gold_mining --pool mined_heldout
Writes <out_dir>/figures/policy_signal_<pool>.png/.pdf and prints the numbers.
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b4b2ac"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PAIR_RE = re.compile(r"^\((.*)\) AND \((.*)\)$")
TRIPLE_RE = re.compile(r"^\((.*)\) AND \((.*)\) AND \((.*)\)$")


def auc(scores, labels):
    """Probability that a random positive scores higher than a random negative (ties count half)."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):          # average ranks of ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def load(out_dir, pool):
    terms, pairs, triples = [], [], []
    for qid in sorted(os.listdir(out_dir)):
        qdir = os.path.join(out_dir, qid)
        paths = [os.path.join(qdir, f"{pool}_{x}.csv") for x in ("selected", "oracle", "pairs")]
        if not all(os.path.exists(p) for p in paths):
            continue
        sel, trace, prs = (pd.read_csv(p) for p in paths)
        if sel.empty:
            continue
        count_of = dict(zip(sel["query"], sel["count"]))
        kinds = trace["kind"] if len(trace) else pd.Series(dtype=str)
        alone = set(trace.loc[kinds == "single", "query"]) if len(trace) else set()
        chosen_pairs = set(trace.loc[kinds == "pair", "query"]) if len(trace) else set()
        chosen_triples = set(trace.loc[kinds == "triple", "query"]) if len(trace) else set()
        in_pair, in_triple = set(), set()
        for q in chosen_pairs:
            m = PAIR_RE.match(q)
            if m:
                in_pair |= {m.group(1), m.group(2)}
        for q in chosen_triples:
            m = TRIPLE_RE.match(q)
            if m:
                in_triple |= {m.group(1), m.group(2), m.group(3)}
        for _, r in sel.iterrows():
            t = r["query"]
            role = ("used alone" if t in alone else "inside pairs" if t in in_pair
                    else "inside triples only" if t in in_triple else "unused")
            terms.append({"qid": qid, "term": t, "count": r["count"], "H": r["H"], "role": role})
        tpath = os.path.join(qdir, f"{pool}_triples.csv")
        if os.path.exists(tpath):
            for _, r in pd.read_csv(tpath).iterrows():
                m = TRIPLE_RE.match(r["query"])
                if m and all(count_of.get(g, 0) > 0 for g in m.groups()):
                    # sum of logs, not the product: three counts in the millions overflow int64
                    log_prod = sum(np.log10(float(count_of[g])) for g in m.groups())
                    triples.append({"qid": qid, "count": r["count"], "pair_count": r["pair_count"],
                                    "x": log_prod, "prod": 10 ** log_prod,
                                    "chosen": r["query"] in chosen_triples})
        for _, r in prs.iterrows():
            m = PAIR_RE.match(r["query"])
            if m and m.group(1) in count_of and m.group(2) in count_of:
                pairs.append({"qid": qid, "pair_count": r["count"], "n_a": count_of[m.group(1)],
                              "n_b": count_of[m.group(2)], "H": r["H"], "chosen": r["query"] in chosen_pairs})
    return pd.DataFrame(terms), pd.DataFrame(pairs), pd.DataFrame(triples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./gold_mining")
    ap.add_argument("--pool", default="mined_heldout")
    args = ap.parse_args()
    terms, pairs, triples = load(args.out_dir, args.pool)
    terms = terms[terms["count"] > 0].copy()
    pairs = pairs[(pairs["pair_count"] > 0) & (pairs["n_a"] > 0) & (pairs["n_b"] > 0)].copy()

    # ---------- 1. roles by count bucket ----------
    edges = [1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e9]
    labels = ["<100", "100–1k", "1k–10k", "10k–100k", "100k–1M", "1M–10M", ">10M"]
    terms["bucket"] = pd.cut(terms["count"], edges, labels=labels, right=False)
    roles = ["used alone", "inside pairs", "inside triples only", "unused"]
    share = (terms.groupby("bucket", observed=False)["role"].value_counts(normalize=True)
             .unstack(fill_value=0).reindex(columns=roles, fill_value=0))
    n_per = terms.groupby("bucket", observed=False).size()

    # ---------- 2. pair counts vs product of piece counts ----------
    pairs["x"] = np.log10(pairs["n_a"].astype(float)) + np.log10(pairs["n_b"].astype(float))
    pairs["y"] = np.log10(pairs["pair_count"])
    slope, intercept = np.polyfit(pairs["x"], pairs["y"], 1)
    resid = pairs["y"] - (slope * pairs["x"] + intercept)
    within10 = (resid.abs() <= 1).mean()
    rho = pairs[["x", "y"]].corr(method="spearman").iloc[0, 1]
    pairs["x_min"] = np.log10(np.minimum(pairs["n_a"], pairs["n_b"]))
    rho_min = pairs[["x_min", "y"]].corr(method="spearman").iloc[0, 1]

    has_triples = len(triples) > 0 and (triples["count"] > 0).any()
    if has_triples:
        triples = triples[triples["count"] > 0].copy()
        triples["y"] = np.log10(triples["count"].astype(float))
        triples = triples[np.isfinite(triples["x"]) & np.isfinite(triples["y"])]
        t_slope, t_icpt = np.polyfit(triples["x"], triples["y"], 1)
        t_rho = triples[["x", "y"]].corr(method="spearman").iloc[0, 1]
        t_within10 = ((triples["y"] - (t_slope * triples["x"] + t_icpt)).abs() <= 1).mean()
        t_reduction = (triples["pair_count"] / triples["count"]).median()

    # ---------- 3. AUCs: what counts explain vs. what only gold coverage explains ----------
    used = terms["role"] != "unused"
    a_alone_count = auc(-np.log10(terms.loc[used, "count"]), terms.loc[used, "role"] == "used alone")
    a_alone_H = auc(terms.loc[used, "H"], terms.loc[used, "role"] == "used alone")
    a_used_count = auc(-np.abs(np.log10(terms["count"]) - np.log10(terms.loc[used, "count"]).median()), used)
    a_used_H = auc(terms["H"], used)
    a_pair_count = auc(-pairs["y"], pairs["chosen"])
    a_pair_H = auc(pairs["H"], pairs["chosen"])

    # ---------- figure ----------
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
                         "xtick.color": INK_2, "ytick.color": INK_2,
                         "axes.spines.top": False, "axes.spines.right": False})
    ncols = 3 if has_triples else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.6), gridspec_kw={"wspace": 0.3})
    ax1, ax2 = axes[0], axes[1]
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for role, color in zip(roles, [BLUE, ORANGE, "#1baf7a", GRAY]):
        vals = share[role].to_numpy()
        ax1.bar(x, vals, bottom=bottom, color=color, width=0.72, edgecolor="white", linewidth=2, label=role)
        bottom += vals
    for i, n in enumerate(n_per):
        ax1.text(i, 1.02, f"n={n:,}", ha="center", fontsize=8, color=INK_2)
    ax1.set_xticks(x, labels, rotation=30, ha="right")
    ax1.set_ylim(0, 1.1)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax1.set_xlabel("Index count of the selected term")
    ax1.set_ylabel("Share of selected terms")
    ax1.set_title("Role in the oracle set changes with the count", loc="left", fontsize=10, color=INK)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False, fontsize=9)

    ax2.scatter(10 ** pairs["x"], pairs["pair_count"], s=4, color=INK_2, alpha=0.08, linewidths=0,
                label="counted pair")
    ch = pairs[pairs["chosen"]]
    ax2.scatter(10 ** ch["x"], ch["pair_count"], s=6, color=BLUE, alpha=0.35, linewidths=0,
                label="pair chosen by the oracle")
    xs = np.linspace(pairs["x"].min(), pairs["x"].max(), 50)
    ax2.plot(10 ** xs, 10 ** (slope * xs + intercept), color=ORANGE, linewidth=2, label="log-linear fit")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Product of the two piece counts")
    ax2.set_ylabel("Pair count (AND within window)")
    ax2.set_title(f"Pair counts follow the piece counts (Spearman {rho:.2f})", loc="left", fontsize=10, color=INK)
    leg = ax2.legend(loc="upper left", frameon=False, fontsize=9)
    for h in leg.legend_handles[:2]:
        h.set_alpha(0.8)
    ax2.grid(color=GRID, linewidth=0.6)

    if has_triples:
        ax3 = axes[2]
        ax3.scatter(triples.loc[~triples["chosen"], "prod"], triples.loc[~triples["chosen"], "count"], s=6,
                    color=INK_2, alpha=0.25, linewidths=0, label="counted triple")
        ax3.scatter(triples.loc[triples["chosen"], "prod"], triples.loc[triples["chosen"], "count"], s=10,
                    color=BLUE, alpha=0.6, linewidths=0, label="triple chosen by the oracle")
        xs3 = np.linspace(triples["x"].min(), triples["x"].max(), 50)
        ax3.plot(10 ** xs3, 10 ** (t_slope * xs3 + t_icpt), color=ORANGE, linewidth=2, label="log-linear fit")
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.set_xlabel("Product of the three piece counts")
        ax3.set_ylabel("Triple count (AND within window)")
        ax3.set_title(f"Triple counts follow the piece counts (Spearman {t_rho:.2f})", loc="left",
                      fontsize=10, color=INK)
        ax3.legend(loc="upper left", frameon=False, fontsize=9)
        ax3.grid(color=GRID, linewidth=0.6)

    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    base = os.path.join(fig_dir, f"policy_signal_{args.pool}")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")

    # ---------- numbers ----------
    print(f"{len(terms):,} selected terms in {terms['qid'].nunique()} queries; {len(pairs):,} counted pairs\n")
    print("Role share by count bucket:")
    print((share * 100).round(1).assign(n=n_per).to_string())
    print(f"\nPair count vs piece counts: Spearman {rho:.2f} (with min of the two counts: {rho_min:.2f}); "
          f"slope {slope:.2f}; {within10:.0%} of pairs within a factor 10 of the fit")
    if has_triples:
        print(f"Triple count vs piece counts: Spearman {t_rho:.2f}; slope {t_slope:.2f}; "
              f"{t_within10:.0%} within a factor 10 of the fit; a triple has median {t_reduction:.1f}x fewer "
              f"matches than its pair; {triples['chosen'].mean():.0%} of counted triples were chosen")
    print("\nAUC (0.5 = no signal) for each decision: count (seen by the model) vs held-out coverage H (not seen)")
    print(f"  term used alone (among used terms):  count {a_alone_count:.2f}   H {a_alone_H:.2f}")
    print(f"  term used at all:                    count {a_used_count:.2f}   H {a_used_H:.2f}")
    print(f"  pair chosen (among counted pairs):   pair count {a_pair_count:.2f}   H {a_pair_H:.2f}")
    print("\nRole share over all selected terms:", (terms["role"].value_counts(normalize=True) * 100).round(1).to_dict())
    print(f"\nsaved {base}.png and .pdf")


if __name__ == "__main__":
    main()
