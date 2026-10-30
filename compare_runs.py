"""
compare_runs.py -- one overview comparing several gold_mining.py runs.

Reads <run_dir>/<qid>/summary.json of every run and prints three tables (one row per run):
  1. recall per budget (summed index count), overall and for query groups of the subset file
  2. recall after the first N queries, and how many queries the oracle set needs
  3. index cost while mining: count calls per kind (requested / sent) and seconds per call

Usage:
    python compare_runs.py mining_runs/base mining_runs/tau_5 --subset gold_mining/subset_60.txt --groups 30
Writes <first run's parent>/comparison.md and prints it.
"""

import argparse
import json
import os
from collections import defaultdict


def load_run(run_dir):
    out = {}
    for qid in sorted(os.listdir(run_dir)):
        path = os.path.join(run_dir, qid, "summary.json")
        if os.path.exists(path):
            with open(path) as f:
                out[qid] = json.load(f)
    return out


def label(B):
    return f"{B // 1000}k" if B >= 1000 else str(B)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def md_table(rows, cols, fmt):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(c, r.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def fmt(col, v):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    if col == "queries":
        return str(int(v))
    if col.startswith(("R@", "R ")):
        return f"{v:.3f}"
    if col.startswith("s/"):
        return f"{v:.3f}" if v == v else "-"
    return f"{v:,.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run folders (output folders of gold_mining.py)")
    ap.add_argument("--pool", default="mined_heldout")
    ap.add_argument("--subset", help="qid file; with --groups, its first N ids form group 1, the rest group 2")
    ap.add_argument("--groups", type=int, default=0)
    ap.add_argument("--out", default=None, help="where to write comparison.md")
    args = ap.parse_args()
    p = args.pool

    runs = {os.path.basename(os.path.normpath(r)): load_run(r) for r in args.runs}
    first = next(iter(next(iter(runs.values())).values()))
    budgets = [int(b) for b in first["config"]["budgets"]]
    qbudgets = sorted(int(n) for n in first.get("recall_after", {}).get(p, {}))

    groups = {}
    if args.subset and args.groups:
        ids = [l.strip() for l in open(args.subset) if l.strip()]
        groups = {f"first {args.groups}": set(ids[:args.groups]), "rest": set(ids[args.groups:])}

    t1, t2, t3 = [], [], []
    for name, sums in runs.items():
        S = list(sums.values())
        r1 = {"run": name, "queries": len(S)}
        for B in budgets:
            r1[f"R@{label(B)}"] = mean(s["recall_at"][p][str(B)] for s in S)
        ref = 25_000 if 25_000 in budgets else budgets[len(budgets) // 2]
        for g, ids in groups.items():
            r1[f"R@{label(ref)} {g}"] = mean(s["recall_at"][p][str(ref)] for q, s in sums.items() if q in ids)
        t1.append(r1)

        r2 = {"run": name}
        for N in qbudgets:
            r2[f"R@{N}q"] = mean(s["recall_after"][p][str(N)] for s in S if "recall_after" in s)
        for B in [b for b in (25_000, 100_000) if b in budgets]:
            r2[f"queries@{B // 1000}k"] = mean(s["queries_at"][p][str(B)]["total"] for s in S if "queries_at" in s)
            r2[f"triples@{B // 1000}k"] = mean(s["queries_at"][p][str(B)]["triple"] for s in S if "queries_at" in s)
        r2["pairs refined"] = mean(s["pools"][p].get("n_refined", 0) for s in S)
        t2.append(r2)

        tot = defaultdict(float)
        for s in S:
            for c in s.get("count_calls", []):
                if c["label"] == p:
                    tot[(c["kind"], c["field"])] += c["value"]
        r3 = {"run": name}
        for kind in ("pair", "triple"):
            r3[f"{kind}s requested"] = tot[(kind, "requested")] / len(S)
            r3[f"{kind}s sent"] = tot[(kind, "fresh")] / len(S)
            fresh = tot[(kind, "fresh")]
            r3[f"s/{kind}"] = tot[(kind, "seconds")] / fresh if fresh else float("nan")
        t3.append(r3)

    text = "\n".join([
        f"# Run comparison (pool {p})", "",
        "## Recall per budget (summed index count)", "",
        md_table(t1, list(t1[0]), fmt), "",
        "## Recall per number of queries", "",
        "R@Nq = recall after the first N queries of the oracle set; queries@B = queries the set needs "
        "within budget B (of which triples).", "",
        md_table(t2, list(t2[0]), fmt), "",
        "## Index cost while mining (mean per query; s/kind = engine seconds per call sent)", "",
        "Calls sent depend on run order: later runs reuse counts cached by earlier ones. "
        "Compare 'requested'.", "",
        md_table(t3, list(t3[0]), fmt), "",
    ])
    print(text)
    out = args.out or os.path.join(os.path.dirname(os.path.normpath(args.runs[0])), "comparison.md")
    with open(out, "w") as f:
        f.write(text)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
