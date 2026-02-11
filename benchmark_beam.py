#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import os
import statistics
import time
import tracemalloc
from typing import Optional

import psutil

# Your file containing expand_wk/_top_w_next_tokens/etc.
# If the filename is beam_search.py, keep this as:
import beam_search


def parse_int_list(s: str):
    # "5,10,20" -> [5,10,20]
    return [int(x) for x in s.split(",") if x.strip()]


def benchmark_expand_wk_grid(
    engine,
    tokenizer,
    query: str,
    *,
    ws,
    ks,
    repeats: int,
    use_infgram: bool,
    max_support: Optional[int],
    verbose: bool,
):
    """
    Benchmarks beam_search.expand_wk over a grid of (w,k).

    Note: expand_wk does NOT have B or d parameters. If you want B/d benchmarking,
    you need a beam search function that includes pruning and total steps.
    """
    proc = psutil.Process(os.getpid())
    rows = []

    combos = list(itertools.product(ws, ks))
    print(f"grid size: {len(combos)} combinations")
    print(f"ws={ws} ks={ks} repeats={repeats}\n")

    for idx, (w, k) in enumerate(combos, start=1):
        print(f"== [{idx}/{len(combos)}] w={w} k={k} ==")

        times = []
        peak_py_bytes = 0
        rss_before_max = 0
        rss_after_max = 0
        lens = []

        for r in range(1, repeats + 1):
            if verbose:
                print(f"  -- run {r}/{repeats} --")

            tracemalloc.start()
            rss_before = proc.memory_info().rss
            t0 = time.perf_counter()

            seqs = beam_search.expand_wk(
                engine,
                tokenizer,
                query,
                w=w,
                k=k,
                use_infgram=use_infgram,
                max_support=max_support,
            )

            t1 = time.perf_counter()
            rss_after = proc.memory_info().rss
            _, peak_py = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            elapsed = t1 - t0
            n = len(seqs)

            times.append(elapsed)
            lens.append(n)
            peak_py_bytes = max(peak_py_bytes, peak_py)
            rss_before_max = max(rss_before_max, rss_before)
            rss_after_max = max(rss_after_max, rss_after)

            print(
                f"  run={r} time={elapsed:.4f}s n_seqs={n} "
                f"rss_before={rss_before/1e6:.1f}MB rss_after={rss_after/1e6:.1f}MB "
                f"py_peak={peak_py/1e6:.1f}MB"
            )

        row = {
            "w": w,
            "k": k,
            "repeats": repeats,
            "time_mean_s": statistics.mean(times),
            "time_min_s": min(times),
            "time_max_s": max(times),
            "n_seqs_mean": statistics.mean(lens),
            "n_seqs_min": min(lens),
            "n_seqs_max": max(lens),
            "rss_before_mb_max": rss_before_max / 1e6,
            "rss_after_mb_max": rss_after_max / 1e6,
            "python_peak_mb": peak_py_bytes / 1e6,
        }
        rows.append(row)

        print(
            f"  => mean_time={row['time_mean_s']:.4f}s "
            f"mean_n_seqs={row['n_seqs_mean']:.1f} "
            f"py_peak={row['python_peak_mb']:.1f}MB "
            f"rss_after_max={row['rss_after_mb_max']:.1f}MB\n"
        )

    return rows


def main():
    ap = argparse.ArgumentParser(description="Grid benchmark for beam_search.expand_wk (w,k).")
    ap.add_argument("--index", required=True, default="/home/mruc/first_index/")
    ap.add_argument("--model", required=True, default = "meta-llama/Llama-2-7b-hf")
    ap.add_argument("--query", default="", help="Prompt")
    ap.add_argument("--ws", default="5,8,10,12", help="Comma list, e.g. 5,10,20")
    ap.add_argument("--ks", default="5,8,10,12", help="Comma list, e.g. 1,3,5")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--use-infgram", action="store_true")
    ap.add_argument("--max-support", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--csv", default="expand_wk_benchmark.csv")
    args = ap.parse_args()

    ws = parse_int_list(args.ws)
    ks = parse_int_list(args.ks)

    from infini_gram.engine import InfiniGramEngine
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, add_bos_token=False, add_eos_token=False)
    engine = InfiniGramEngine(index_dir=args.index, eos_token_id=tokenizer.eos_token_id)

    rows = benchmark_expand_wk_grid(
        engine,
        tokenizer,
        args.query,
        ws=ws,
        ks=ks,
        repeats=args.repeats,
        use_infgram=args.use_infgram,
        max_support=args.max_support,
        verbose=args.verbose,
    )

    if args.csv and rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
