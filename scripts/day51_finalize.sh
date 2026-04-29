#!/usr/bin/env bash
# Day-51 finalize: runs aggregators + appends final numbers to docs/daily_log/day51.md.
# Triggered manually after the three background runs complete.
set -euo pipefail
cd "$(dirname "$0")/.."

source .venv/bin/activate

mkdir -p results/day51 results/day51_seed0_n200

echo "=== running day51_recompute_apples ==="
python scripts/day51_recompute_apples.py 2>&1 | tee /tmp/day51_apples.log

echo
echo "=== running day51_held_out_200_summary ==="
python scripts/day51_held_out_200_summary.py 2>&1 | tee /tmp/day51_ho200.log

echo
echo "=== summary files written ==="
ls -lh results/day51_seed0_n200/apples_to_apples.json
ls -lh results/day51/stablehlo_held_out_200_summary.json
ls -lh results/day51/functional_n30.json
ls -lh results/day51/verify_concordance_n50.json

echo
echo "Append the final numbers to docs/daily_log/day51.md by hand or with a sed splice."
