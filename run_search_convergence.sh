#!/usr/bin/env bash
# Search-convergence experiment: do the search limits of the tau rule change the mined targets?
# All runs: triples with tau = 5 on the 60-query subset; only the two SEARCH LIMITS vary:
#   triples_per_pair (third terms counted per refinement; 0 = all) and max_refinements (0 = unlimited).
# Reference = the most exhaustive run (all third terms, unlimited refinement).
# If the cheaper runs give the same recall and the same pair labels / term roles, the limits are harmless.
#
# Usage:  bash run_search_convergence.sh [index_dir]
set -euo pipefail

IDX="${1:-/home/mruc/msmarco_segmented_index/}"
SUB="gold_mining/subset_60.txt"
DATA="gold_mining"
OUT="mining_runs"
TAU=5

for f in gold_mining.py target_checks.py compare_runs.py policy_signal.py "$SUB"; do
  [[ -e "$f" ]] || { echo "missing: $f"; exit 1; }
done
mkdir -p "$OUT"
echo "index: $IDX   subset: $SUB   tau: $TAU   started: $(date)"

# name                tpp  max_refinements     (cheapest first; later runs reuse cached counts)
RUNS=(
  "conv_tpp5_r1        5    1"
  "conv_tpp20_r1      20    1"
  "conv_all_r1         0    1"
  "conv_all_rall       0    0"
)
for spec in "${RUNS[@]}"; do
  read -r NAME TPP MAXR <<< "$spec"
  echo "=== $NAME (triples_per_pair=$TPP, max_refinements=$MAXR) ==="
  python -u gold_mining.py --qid_file "$SUB" --index_dir "$IDX" --data_dir "$DATA" \
    --triples --tau "$TAU" --triples_per_pair "$TPP" --max_refinements "$MAXR" \
    --out_dir "$OUT/$NAME" 2>&1 | tee "$OUT/$NAME.log"
done

echo
echo "=== target checks (reference: conv_all_rall) ==="
python target_checks.py "$OUT/conv_all_rall" "$OUT/conv_all_r1" "$OUT/conv_tpp20_r1" "$OUT/conv_tpp5_r1" \
  --out "$OUT/target_checks.md" || echo "target checks FAILED"

echo
echo "=== run comparison ==="
python compare_runs.py "$OUT/conv_tpp5_r1" "$OUT/conv_tpp20_r1" "$OUT/conv_all_r1" "$OUT/conv_all_rall" \
  --subset "$SUB" --groups 30 --out "$OUT/comparison_convergence.md" || echo "comparison FAILED"
echo "finished: $(date)"
