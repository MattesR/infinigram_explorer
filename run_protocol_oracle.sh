#!/usr/bin/env bash
# Protocol oracle: does an oracle that works like the generator (propose -> piece counts -> pull or
# combine -> pair counts -> refine) lose recall against greedy max coverage, and how much of its
# decisions can a model reproduce from counts alone?
#
#   protocol             gold decisions (tau = 5), terms decided for "alone" in order of precision
#   protocol_importance  the same, terms decided in order of importance (held-out coverage)
#   protocol_counts      decisions from count thresholds only, fitted on `protocol` (fit_count_rule.py)
#   conv_all_r1          greedy reference from the convergence experiment (same tau and search limits)
#
# Usage:  bash run_protocol_oracle.sh [index_dir]
set -euo pipefail

IDX="${1:-/home/mruc/msmarco_segmented_index/}"
SUB="gold_mining/subset_60.txt"
DATA="gold_mining"
OUT="mining_runs"
COMMON=(--qid_file "$SUB" --index_dir "$IDX" --data_dir "$DATA" --tau 5 --triples_per_pair 0 --max_refinements 1)

for f in gold_mining.py fit_count_rule.py target_checks.py compare_runs.py "$SUB"; do
  [[ -e "$f" ]] || { echo "missing: $f"; exit 1; }
done
mkdir -p "$OUT"
echo "index: $IDX   subset: $SUB   started: $(date)"

if [[ ! -d "$OUT/conv_all_r1" ]]; then
  echo "=== greedy reference conv_all_r1 ==="
  python -u gold_mining.py "${COMMON[@]}" --triples --out_dir "$OUT/conv_all_r1" 2>&1 | tee "$OUT/conv_all_r1.log"
fi

echo "=== protocol (precision order) ==="
python -u gold_mining.py "${COMMON[@]}" --oracle protocol --protocol_order precision \
  --out_dir "$OUT/protocol" 2>&1 | tee "$OUT/protocol.log"

echo "=== protocol (importance order) ==="
python -u gold_mining.py "${COMMON[@]}" --oracle protocol --protocol_order importance \
  --out_dir "$OUT/protocol_importance" 2>&1 | tee "$OUT/protocol_importance.log"

echo "=== fit count rule on protocol ==="
python fit_count_rule.py "$OUT/protocol" --out "$OUT/count_rule.txt" | tee "$OUT/count_rule.md"
RULE="$(cat "$OUT/count_rule.txt")"

echo "=== protocol_counts (count rule $RULE) ==="
python -u gold_mining.py "${COMMON[@]}" --oracle counts --count_rule "$RULE" \
  --out_dir "$OUT/protocol_counts" 2>&1 | tee "$OUT/protocol_counts.log"

echo
echo "=== run comparison ==="
python compare_runs.py "$OUT/conv_all_r1" "$OUT/protocol" "$OUT/protocol_importance" "$OUT/protocol_counts" \
  --subset "$SUB" --groups 30 --out "$OUT/comparison_protocol.md" || echo "comparison FAILED"

echo
echo "=== target checks (reference: protocol) ==="
python target_checks.py "$OUT/protocol" "$OUT/protocol_importance" "$OUT/protocol_counts" "$OUT/conv_all_r1" \
  --out "$OUT/target_checks_protocol.md" || echo "target checks FAILED"
echo "finished: $(date)"
