"""Fit the count rule of the counts oracle on the decisions of the protocol oracle.

The protocol oracle decides with gold information: a single, pair or triple is accepted if it adds
>= tau new relevant docs per 1,000 count. A model only sees counts. This script asks: which count
threshold per kind best reproduces those decisions ("accept if count <= threshold")? It reports
how well count alone separates them (AUC, balanced accuracy) and writes the thresholds as
"single,pair,triple" for `gold_mining.py --oracle counts --count_rule ...`.

Decisions used (pool mined_heldout):
  single: alone (accept) vs combined / dropped (reject); subsumed terms are left out
  pair:   accepted vs rejected / refined; pairs never considered are left out
  triple: accepted vs rejected; subsumed triples are left out

Usage: python fit_count_rule.py mining_runs/protocol [--out mining_runs/count_rule.txt]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

KINDS = {
    "single": ("selected", {"alone"}, {"combined", "dropped"}),
    "pair": ("pairs", {"accepted"}, {"rejected", "refined"}),
    "triple": ("triples", {"accepted"}, {"rejected"}),
}


def load(run, part, pool):
    frames = [pd.read_csv(f).assign(qid=f.split(os.sep)[-2])
              for f in glob.glob(os.path.join(run, "*", f"{pool}_{part}.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def auc(pos, neg):
    """P(a random accepted query has a lower count than a random rejected one); ties count half."""
    pos, neg = np.sort(pos), np.sort(neg)
    lower = np.searchsorted(neg, pos, side="right")   # rejected with count <= this accepted
    ties = lower - np.searchsorted(neg, pos, side="left")
    return float(((len(neg) - lower) + 0.5 * ties).sum() / (len(pos) * len(neg)))


def best_threshold(pos, neg):
    """Count threshold with the highest balanced accuracy for 'accept if count <= threshold'."""
    best = (0.0, None)
    for thr in np.unique(np.concatenate([pos, neg])):
        bacc = 0.5 * ((pos <= thr).mean() + (neg > thr).mean())
        if bacc > best[0]:
            best = (bacc, thr)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="output folder of a protocol-oracle run")
    ap.add_argument("--pool", default="mined_heldout")
    ap.add_argument("--out", default=None, help="file to write 'single,pair,triple' thresholds to")
    args = ap.parse_args()

    thresholds, lines = [], ["| kind | accepted | rejected | AUC of count | threshold | balanced accuracy |",
                             "|---|---|---|---|---|---|"]
    for kind, (part, yes, no) in KINDS.items():
        df = load(args.run, part, args.pool)
        if df.empty or "decision" not in df:
            raise SystemExit(f"no decisions for {kind} in {args.run}: was it run with --oracle protocol?")
        pos = df.loc[df.decision.isin(yes), "count"].to_numpy(float)
        neg = df.loc[df.decision.isin(no), "count"].to_numpy(float)
        if len(pos) == 0 or len(neg) == 0:     # one class only: accept none / all of this kind
            thr = 0.0 if len(pos) == 0 else float(pos.max())
            thresholds.append(thr)
            lines.append(f"| {kind} | {len(pos)} | {len(neg)} | n/a | {thr:,.0f} | n/a (one class only) |")
            continue
        bacc, thr = best_threshold(pos, neg)
        thresholds.append(thr)
        lines.append(f"| {kind} | {len(pos)} | {len(neg)} | {auc(pos, neg):.3f} | {thr:,.0f} | {bacc:.3f} |")

    print(f"# Count rule fitted on {args.run} (pool {args.pool})\n")
    print("Accept if count <= threshold. AUC = chance that an accepted query has a lower count "
          "than a rejected one.\n")
    print("\n".join(lines))
    rule = ",".join(f"{t:.0f}" for t in thresholds)
    print(f"\n--count_rule {rule}")
    if args.out:
        with open(args.out, "w") as f:
            f.write(rule + "\n")


if __name__ == "__main__":
    main()
