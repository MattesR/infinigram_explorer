"""
target_checks.py -- are the mined training targets free of search artifacts?

The policy the generator should learn has two parts:
  importance:  a term that is central to the question has to stay in the query set;
  specificity: if its count is too high, it gets partners (a pair, or a triple if the pair is still too broad).
In the oracle's output, important broad terms show up as HUBS: never alone, but inside several chosen pairs/triples.

This script checks two things over several runs of gold_mining.py:
  1. Hub structure: for the important terms of each question (top third by held-out coverage H),
     by count bucket: how many are dropped (unused), and in how many chosen queries they appear.
  2. Stability against the reference run (the first run given, normally the most exhaustive search):
     do the pair labels (kept / refined / unused) and the term roles change when the search limits change?
     If they do not, the targets are free of the search-limit artifact.

Usage:
    python target_checks.py mining_runs/conv_all_rall mining_runs/conv_tpp5_r1 mining_runs/conv_tpp20_r1 ...
Writes <parent of first run>/target_checks.md and figures/target_checks.png/.pdf, and prints the tables.
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAIR_RE = re.compile(r"^\((.*)\) AND \((.*)\)$")
TRIPLE_RE = re.compile(r"^\((.*)\) AND \((.*)\) AND \((.*)\)$")
EDGES = [1, 1e3, 1e4, 1e5, 1e6, 1e9]
BUCKETS = ["<1k", "1k–10k", "10k–100k", "100k–1M", ">1M"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6b62d6", "#d14081"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def parts(kind, query):
    m = (TRIPLE_RE if kind == "triple" else PAIR_RE).match(query) if kind != "single" else None
    return list(m.groups()) if m else [query]


def load_run(run_dir, pool):
    terms, pairs = [], []
    for qid in sorted(os.listdir(run_dir)):
        qdir = os.path.join(run_dir, qid)
        sp, op, pp = (os.path.join(qdir, f"{pool}_{x}.csv") for x in ("selected", "oracle", "pairs"))
        if not (os.path.exists(sp) and os.path.exists(op)):
            continue
        sel, trace = pd.read_csv(sp), pd.read_csv(op)
        if sel.empty:
            continue
        degree, role = {}, {}
        for _, t in trace.iterrows():
            ps = parts(t["kind"], t["query"])
            for x in ps:
                if t["kind"] == "single":
                    role[x] = "used alone"
                else:
                    degree[x] = degree.get(x, 0) + 1
                    if role.get(x) != "used alone":
                        role[x] = "inside pairs" if t["kind"] == "pair" or role.get(x) == "inside pairs" \
                            else "inside triples only"
        cut = sel["H"].quantile(2 / 3)
        for _, r in sel.iterrows():
            terms.append({"qid": qid, "term": r["query"], "count": r["count"], "H": r["H"],
                          "important": r["H"] >= cut and r["H"] > 0,
                          "role": role.get(r["query"], "unused"), "degree": degree.get(r["query"], 0)})
        if os.path.exists(pp):
            p = pd.read_csv(pp)
            if "label" in p.columns:
                p["qid"] = qid
                pairs.append(p[["qid", "query", "count", "label"]])
    terms = pd.DataFrame(terms)
    terms["bucket"] = pd.cut(terms["count"], EDGES, labels=BUCKETS, right=False)
    return terms, (pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame())


def md(df, floatfmt=".2f"):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join(format(v, floatfmt) if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run folders; the first is the reference")
    ap.add_argument("--pool", default="mined_heldout")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    names = [os.path.basename(os.path.normpath(r)) for r in args.runs]
    data = {n: load_run(r, args.pool) for n, r in zip(names, args.runs)}
    ref = names[0]
    lines = [f"# Target checks (pool {args.pool}, reference run: {ref})", ""]

    # ---- 1. hub structure of important terms ----
    dropped, hub = {}, {}
    for n, (terms, _) in data.items():
        imp = terms[terms["important"]]
        dropped[n] = imp.groupby("bucket", observed=False)["role"].apply(lambda r: (r == "unused").mean())
        used = imp[imp["role"] != "unused"]
        hub[n] = used.groupby("bucket", observed=False)["degree"].mean()
    t_drop = pd.DataFrame(dropped).reset_index().rename(columns={"bucket": "count of the term"})
    t_hub = pd.DataFrame(hub).reset_index().rename(columns={"bucket": "count of the term"})
    n_imp = data[ref][0].query("important").groupby("bucket", observed=False).size()
    lines += ["## 1. Hub structure of important terms (top third by held-out coverage per question)", "",
              f"Important terms per count bucket in {ref}: " +
              ", ".join(f"{b} {int(v)}" for b, v in n_imp.items()), "",
              "Share of important terms that are DROPPED (unused in the oracle set):", "",
              md(t_drop), "",
              "Mean number of chosen pairs/triples an important (used) term appears in (hub degree):", "",
              md(t_hub), ""]

    # ---- 2. stability against the reference ----
    ref_terms, ref_pairs = data[ref]
    rows_t, rows_p = [], []
    conf_text = []
    for n in names[1:]:
        terms, prs = data[n]
        m = ref_terms.merge(terms, on=["qid", "term"], suffixes=("_ref", ""))
        rows_t.append({"run": n, "terms compared": len(m),
                       "same role": (m["role_ref"] == m["role"]).mean(),
                       "important: same role": (m.loc[m["important_ref"], "role_ref"]
                                                == m.loc[m["important_ref"], "role"]).mean()})
        if len(ref_pairs) and len(prs):
            mp = ref_pairs.merge(prs, on=["qid", "query"], suffixes=("_ref", ""))
            rows_p.append({"run": n, "pairs compared": len(mp), "same label": (mp["label_ref"] == mp["label"]).mean(),
                           "ref refined, here kept": ((mp["label_ref"] == "refined") & (mp["label"] == "kept")).mean(),
                           "ref refined, here unused": ((mp["label_ref"] == "refined") & (mp["label"] == "unused")).mean()})
            conf = pd.crosstab(mp["label_ref"], mp["label"], normalize="index").round(3)
            conf_text += [f"Pair labels, rows = {ref}, columns = {n} (share of row):", "",
                          md(conf.reset_index().rename(columns={"label_ref": "reference"}), ".3f"), ""]
    lines += ["## 2. Stability of the targets against the reference", "",
              "Term roles (used alone / inside pairs / inside triples only / unused):", "",
              md(pd.DataFrame(rows_t), ".3f") if rows_t else "(only one run given)", ""]
    if rows_p:
        lines += ["Pair labels (kept / refined / unused):", "", md(pd.DataFrame(rows_p), ".3f"), ""] + conf_text
    else:
        lines += ["Pair labels: not available (runs made before the label columns existed).", ""]

    # ---- figure ----
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.28})
    x = np.arange(len(BUCKETS))
    for i, n in enumerate(names):
        c = COLORS[i % len(COLORS)]
        a1.plot(x, dropped[n].reindex(BUCKETS).values, marker="o", color=c, linewidth=2, label=n)
        a2.plot(x, hub[n].reindex(BUCKETS).values, marker="o", color=c, linewidth=2, label=n)
    a1.set_ylabel("Share of important terms dropped")
    a1.set_ylim(0, 1)
    a1.set_title("Important terms that drop out of the query set", loc="left", fontsize=10, color=INK)
    a2.set_ylabel("Chosen pairs/triples per important term")
    a2.set_title("Hub degree of important terms", loc="left", fontsize=10, color=INK)
    for a in (a1, a2):
        a.set_xticks(x, BUCKETS)
        a.set_xlabel("Index count of the term")
        a.grid(axis="y", color=GRID, linewidth=0.8)
    a2.legend(frameon=False, fontsize=9, loc="upper left")
    base_dir = os.path.dirname(os.path.normpath(args.runs[0]))
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.out))[0] if args.out else "target_checks"
    fig.savefig(os.path.join(fig_dir, f"{stem}.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, f"{stem}.pdf"), bbox_inches="tight")

    text = "\n".join(lines)
    print(text)
    out = args.out or os.path.join(base_dir, "target_checks.md")
    with open(out, "w") as f:
        f.write(text)
    print(f"written to {out} and {fig_dir}/{stem}.png")


if __name__ == "__main__":
    main()
