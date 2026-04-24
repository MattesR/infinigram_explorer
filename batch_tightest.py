"""
Batch tightest query analysis across all found.jsonl files.

Usage:
    from batch_tightest import batch_tightest_analysis

    df = batch_tightest_analysis(
        found_dir="./inspection/prog/",
        expansions_path="./kiss.jsonl",
        tokenizer=tokenizer,
        engine=engine,
        prox=50,
    )
    print(df)
"""

import os
import time
import json
import pandas as pd
from glob import glob
from tqdm import tqdm
from tightest_queries import find_tightest_per_doc


def batch_tightest_analysis(
    found_dir: str,
    expansions_path: str,
    tokenizer,
    engine,
    prox: int = 50,
    max_clause_freq: int = 80000000,
    pattern: str = "*_found.jsonl",
    verbose: bool = True,
):
    """
    Run tightest query analysis for all found.jsonl files in a directory.

    Returns a DataFrame with per-query statistics.
    """
    files = sorted(glob(os.path.join(found_dir, pattern)))
    if verbose:
        print(f"Found {len(files)} files matching {pattern} in {found_dir}")

    rows = []

    for fpath in tqdm(files, desc="Analyzing", disable=not verbose):
        fname = os.path.basename(fpath)
        qid = fname.replace("_found.jsonl", "")

        # Count docs
        with open(fpath) as f:
            lines = [l.strip() for l in f if l.strip()]
        n_docs = len(lines)
        n_with_text = sum(1 for l in lines if json.loads(l).get("text"))

        if n_with_text == 0:
            if verbose:
                print(f"  {qid}: no docs with text, skipping")
            continue

        t0 = time.perf_counter()
        try:
            results = find_tightest_per_doc(
                found_path=fpath,
                qid=qid,
                expansions_path=expansions_path,
                tokenizer=tokenizer,
                engine=engine,
                prox=prox,
                max_clause_freq=max_clause_freq,
                verbose=False,
            )
        except Exception as e:
            if verbose:
                print(f"  {qid}: ERROR {e}")
            continue
        elapsed = time.perf_counter() - t0

        doc_results = results["doc_results"]
        query_coverage = results["all_query_coverage"]
        engine_counts = results["query_engine_counts"]
        single_term = results["single_term_docs"]

        # Set cover stats
        uncovered = set(r["doc_id"] for r in doc_results)
        selected = []
        total_engine = 0

        while uncovered:
            best_desc = None
            best_eff = -1
            for desc, doc_ids in query_coverage.items():
                new_covered = len(doc_ids & uncovered)
                if new_covered == 0:
                    continue
                eng = engine_counts.get(desc, 1)
                if eng <= 0:
                    eng = 1
                eff = new_covered / eng
                if eff > best_eff:
                    best_eff = eff
                    best_desc = desc

            if best_desc is None:
                break

            doc_ids = query_coverage[best_desc]
            new_covered = doc_ids & uncovered
            eng = engine_counts.get(best_desc, 0)
            uncovered -= new_covered
            total_engine += eng
            selected.append({
                "desc": best_desc,
                "new_covered": len(new_covered),
                "engine_count": eng,
            })

        total_covered = len(doc_results) - len(uncovered)

        # Aspect coverage distribution
        aspect_dist = {}
        for r in doc_results:
            n = r["n_aspects"]
            aspect_dist[n] = aspect_dist.get(n, 0) + 1

        # Best query engine counts
        best_counts = [
            min(t["engine_count"] for t in r["tightest"])
            for r in doc_results
            if r.get("tightest")
            and any(t.get("engine_count", -1) > 0 for t in r["tightest"])
        ]
        best_counts_valid = [c for c in best_counts if c > 0]

        # Average precision of selected queries
        avg_precision = 0
        if selected:
            precisions = [
                s["new_covered"] / s["engine_count"]
                for s in selected if s["engine_count"] > 0
            ]
            avg_precision = sum(precisions) / len(precisions) if precisions else 0

        row = {
            "qid": qid,
            "n_docs": n_with_text,
            "n_aspects": len(results["aspect_names"]),
            "n_unique_queries": len(query_coverage),
            "n_set_cover_queries": len(selected),
            "set_cover_docs": total_covered,
            "set_cover_engine_count": total_engine,
            "set_cover_uncovered": len(uncovered),
            "set_cover_avg_precision": round(avg_precision, 4),
            "n_single_term_docs": len(single_term),
            "median_best_count": sorted(best_counts_valid)[len(best_counts_valid) // 2] if best_counts_valid else 0,
            "max_best_count": max(best_counts_valid) if best_counts_valid else 0,
            "sum_best_counts": sum(best_counts_valid),
            "aspects_0": aspect_dist.get(0, 0),
            "aspects_1": aspect_dist.get(1, 0),
            "aspects_2": aspect_dist.get(2, 0),
            "aspects_3": aspect_dist.get(3, 0),
            "time_seconds": round(elapsed, 1),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if verbose and not df.empty:
        print(f"\n{'='*90}")
        print(f"BATCH TIGHTEST QUERY ANALYSIS ({len(df)} topics, prox={prox})")
        print(f"{'='*90}")

        print(f"\n{'QID':<15s} {'Docs':>5s} {'Uniq':>5s} {'Cover':>5s} "
              f"{'Eng':>8s} {'Uncov':>5s} {'1-term':>6s} {'P_avg':>7s} {'Time':>6s}")
        print("-" * 75)
        for _, r in df.iterrows():
            print(f"{r['qid']:<15s} {r['n_docs']:>5d} {r['n_unique_queries']:>5d} "
                  f"{r['n_set_cover_queries']:>5d} {r['set_cover_engine_count']:>8,d} "
                  f"{r['set_cover_uncovered']:>5d} {r['n_single_term_docs']:>6d} "
                  f"{r['set_cover_avg_precision']:>7.4f} {r['time_seconds']:>5.1f}s")

        print(f"\n  Averages:")
        for col in ["n_docs", "n_unique_queries", "n_set_cover_queries",
                     "set_cover_engine_count", "set_cover_uncovered",
                     "n_single_term_docs", "set_cover_avg_precision", "time_seconds"]:
            print(f"    {col}: {df[col].mean():.1f}")

        print(f"\n  Total engine count across all topics: {df['set_cover_engine_count'].sum():,d}")
        print(f"  Total docs covered: {df['set_cover_docs'].sum()} / {df['n_docs'].sum()}")
        print(f"  Total uncovered: {df['set_cover_uncovered'].sum()}")

    return df