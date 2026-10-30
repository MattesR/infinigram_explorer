#!/usr/bin/env bash
# Triples experiment (tau rule) on the 60-query subset: baseline (pairs only) and tau = 5 / 2.5 / 10.
# Runs one after the other (all runs share the count cache), then diagnostics, figures and a comparison.
#
# Usage:  bash run_triples_experiment.sh [index_dir]
# Needs:  gold_mining.py, analyze_gold_mining.py, plot_gold_mining.py, compare_runs.py,
#         gold_mining/subset_60.txt, gold_mining/<qid>/relevant_docs.jsonl, bm_25_baseline_results_large_pool.pkl
set -euo pipefail

IDX="${1:-/home/mruc/msmarco_segmented_index/}"
SUB="gold_mining/subset_60.txt"
DATA="gold_mining"
OUT="mining_runs"
TAUS=(5 2.5 10)

for f in gold_mining.py analyze_gold_mining.py plot_gold_mining.py compare_runs.py "$SUB" bm_25_baseline_results_large_pool.pkl; do
  [[ -e "$f" ]] || { echo "missing: $f"; exit 1; }
done
mkdir -p "$OUT"
echo "index: $IDX   subset: $SUB ($(wc -l < "$SUB") queries)   started: $(date)"

echo "=== base (pairs only) ==="
python -u gold_mining.py --qid_file "$SUB" --index_dir "$IDX" --data_dir "$DATA" \
  --out_dir "$OUT/base" 2>&1 | tee "$OUT/base.log"

for TAU in "${TAUS[@]}"; do
  echo "=== triples, tau = $TAU ==="
  python -u gold_mining.py --qid_file "$SUB" --index_dir "$IDX" --data_dir "$DATA" \
    --triples --tau "$TAU" --out_dir "$OUT/tau_$TAU" 2>&1 | tee "$OUT/tau_$TAU.log"
done

RUNS=("$OUT/base")
for TAU in "${TAUS[@]}"; do RUNS+=("$OUT/tau_$TAU"); done

for RUN in "${RUNS[@]}"; do
  echo "=== diagnostics and figure: $RUN ==="
  python analyze_gold_mining.py --out_dir "$RUN" --data_dir "$DATA" > "$RUN/diagnostics.log" 2>&1
  python plot_gold_mining.py --out_dir "$RUN" --tau 5 > "$RUN/figure.log" 2>&1
done

echo
echo "=== comparison ==="
python compare_runs.py "${RUNS[@]}" --subset "$SUB" --groups 30 --out "$OUT/comparison.md"
echo "finished: $(date)"
