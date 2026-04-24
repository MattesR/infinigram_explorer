"""
Batch tightest query analysis with proximity recommendation.

Runs tightest analysis once at wide proximity, then derives
optimal prox from span distributions.

Usage:
    from batch_tightest import batch_tightest_analysis

    df = batch_tightest_analysis(
        found_dir="./inspection/prog/",
        expansions_path="./kiss.jsonl",
        tokenizer=tokenizer,
        engine=engine,
    )
"""

import os
import time
import json
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from tightest_queries import find_tightest_per_doc


def batch_tightest_analysis(
    found_dir: str,
    expansions_path: str,
    tokenizer,
    engine,
    analysis_prox: int = 2000,
    max_clause_freq: int = 80000000,
    pattern: str = "*_found.jsonl",
    verbose: bool = True,
):
    """
    Run tightest query analysis for all found.jsonl files.

    Runs once at wide proximity, then derives proximity recommendations
    from actual span distributions in the documents.

    Returns (df_summary, df_prox_curve) where:
    - df_summary: per-qid statistics
    - df_prox_curve: recall/engine_count at different prox thresholds
    """
    files = sorted(glob(os.path.join(found_dir, pattern)))
    if verbose:
        print(f"Found {len(files)} files matching {pattern} in {found_dir}")
        print(f"Analysis proximity: {analysis_prox}")

    summary_rows = []
    all_spans = []  # (qid, span, doc_id, query_desc) for global analysis

    for fpath in tqdm(files, desc="Analyzing", disable=not verbose):
        fname = os.path.basename(fpath)
        qid = fname.replace("_found.jsonl", "")

        with open(fpath) as f:
            lines = [l.strip() for l in f if l.strip()]
        n_with_text = sum(1 for l in lines if json.loads(l).get("text"))
        if n_with_text == 0:
            continue

        t0 = time.perf_counter()
        try:
            results = find_tightest_per_doc(
                found_path=fpath,
                qid=qid,
                expansions_path=expansions_path,
                tokenizer=tokenizer,
                engine=engine,
                prox=analysis_prox,
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

        # Collect spans for each doc's best (tightest) query
        doc_best_spans = []
        for r in doc_results:
            if not r["tightest"]:
                doc_best_spans.append(None)
                continue
            # Best = lowest engine count query that has a valid span
            valid = [t for t in r["tightest"] if t.get("span") is not None]
            if valid:
                best = min(valid, key=lambda t: t.get("engine_count", float('inf'))
                           if t.get("engine_count", 0) > 0 else float('inf'))
                doc_best_spans.append(best["span"])
                all_spans.append({
                    "qid": qid,
                    "doc_id": r["doc_id"],
                    "span": best["span"],
                    "query": best["desc"],
                    "engine_count": best.get("engine_count", 0),
                    "n_aspects": r["n_aspects"],
                })
            else:
                doc_best_spans.append(None)

        valid_spans = [s for s in doc_best_spans if s is not None]

        summary_rows.append({
            "qid": qid,
            "n_docs": n_with_text,
            "n_aspects": len(results["aspect_names"]),
            "n_with_span": len(valid_spans),
            "n_no_span": n_with_text - len(valid_spans),
            "n_single_term": len(results["single_term_docs"]),
            "n_unique_queries": len(query_coverage),
            "span_min": min(valid_spans) if valid_spans else None,
            "span_median": int(np.median(valid_spans)) if valid_spans else None,
            "span_p75": int(np.percentile(valid_spans, 75)) if valid_spans else None,
            "span_p90": int(np.percentile(valid_spans, 90)) if valid_spans else None,
            "span_p95": int(np.percentile(valid_spans, 95)) if valid_spans else None,
            "span_max": max(valid_spans) if valid_spans else None,
            "time_seconds": round(elapsed, 1),
        })

    df_summary = pd.DataFrame(summary_rows)

    # Build proximity curve from all spans
    if all_spans:
        span_values = [s["span"] for s in all_spans]
        prox_thresholds = sorted(set(
            [5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000]
            + [int(np.percentile(span_values, p)) for p in [50, 75, 90, 95, 99]]
        ))

        prox_rows = []
        total_docs = len(all_spans)
        for prox in prox_thresholds:
            covered = sum(1 for s in all_spans if s["span"] <= prox)
            prox_rows.append({
                "prox": prox,
                "docs_covered": covered,
                "docs_total": total_docs,
                "recall": round(covered / total_docs, 4) if total_docs else 0,
            })
        df_prox = pd.DataFrame(prox_rows)
    else:
        df_prox = pd.DataFrame()

    if verbose and not df_summary.empty:
        print(f"\n{'='*90}")
        print(f"TIGHTEST QUERY ANALYSIS ({len(df_summary)} topics)")
        print(f"{'='*90}")

        print(f"\n  Per-query span statistics:")
        print(f"  {'QID':<15s} {'Docs':>5s} {'Span':>5s} {'Med':>5s} {'P75':>5s} "
              f"{'P90':>5s} {'P95':>5s} {'Max':>5s} {'NoSpan':>6s} {'Time':>6s}")
        print(f"  {'-'*75}")
        for _, r in df_summary.iterrows():
            print(f"  {r['qid']:<15s} {r['n_docs']:>5d} "
                  f"{r['n_with_span']:>5d} "
                  f"{r['span_median'] if r['span_median'] is not None else 'N/A':>5} "
                  f"{r['span_p75'] if r['span_p75'] is not None else 'N/A':>5} "
                  f"{r['span_p90'] if r['span_p90'] is not None else 'N/A':>5} "
                  f"{r['span_p95'] if r['span_p95'] is not None else 'N/A':>5} "
                  f"{r['span_max'] if r['span_max'] is not None else 'N/A':>5} "
                  f"{r['n_no_span']:>6d} {r['time_seconds']:>5.1f}s")

        # Global span percentiles
        if all_spans:
            sv = [s["span"] for s in all_spans]
            print(f"\n  Global span distribution ({len(sv)} doc-query pairs):")
            for p in [50, 75, 90, 95, 99]:
                print(f"    P{p}: {int(np.percentile(sv, p))} tokens")

        # Proximity recommendation
        if not df_prox.empty:
            print(f"\n  Proximity → Recall curve:")
            print(f"  {'Prox':>6s} {'Covered':>8s} {'Recall':>8s}")
            print(f"  {'-'*25}")
            for _, r in df_prox.iterrows():
                marker = ""
                if r["recall"] >= 0.90 and (df_prox[df_prox["prox"] < r["prox"]]["recall"].max() < 0.90
                                             if len(df_prox[df_prox["prox"] < r["prox"]]) > 0 else True):
                    marker = "  <-- 90% recall"
                elif r["recall"] >= 0.95 and (df_prox[df_prox["prox"] < r["prox"]]["recall"].max() < 0.95
                                               if len(df_prox[df_prox["prox"] < r["prox"]]) > 0 else True):
                    marker = "  <-- 95% recall"
                print(f"  {r['prox']:>6d} {r['docs_covered']:>8d} {r['recall']:>8.4f}{marker}")

            # Recommendation
            for target in [0.90, 0.95]:
                candidates = df_prox[df_prox["recall"] >= target]
                if not candidates.empty:
                    rec = candidates.iloc[0]
                    print(f"\n  Recommendation for {target:.0%} recall: prox={rec['prox']}")

    return df_summary, df_prox