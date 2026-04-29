"""Day-51 / Item 1 aggregator: recompute apples_to_apples.json.

Inputs:
  - results/day51_seed0_n200/multiseed_seed0.jsonl  (8 cells × seed=0 at uniform n)
  - results/day39pm/multiseed_fulln.jsonl           (5 cells × seeds 1,2 at uniform n)
  - results/day50/baseline_linalg_multiseed.jsonl   (3 baseline-linalg cells × seeds 1,2)

Output (results/day51_seed0_n200/apples_to_apples.json):
  one block per (model, dialect, constraint_config) cell containing
  {"seeds": {0: ..., 1: ..., 2: ...}, "mean": ..., "half_range": ...,
   "ci_low": ..., "ci_high": ...}.

Bootstrap CIs are computed across the per-prompt verify_valid arrays for
each (cell, seed), then aggregated by mean ± half_range across seeds. The
"ci_low/ci_high" fields are the within-seed bootstrap (eval/stats.py
defaults: 10000 resamples, seed=0). Intent: show that the within-cell
sampling noise (CI) is smaller than the cross-seed half-range — if it
isn't, we say so.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from eval.stats import bootstrap_ci

# 9 cell × 3 seed rows we expect after the join (8 multi-seed cells + 1
# Item-3 fp16 cell on linalg).
CELL_KEYS = [
    ("smollm2-c1c2c3",        "arith+func"),
    ("smollm2-c1c2c3",        "linalg"),
    ("granite-c1c3",          "arith+func"),
    ("granite-c1c3",          "linalg"),
    ("codellama-c1c3",        "arith+func"),
    ("codellama-c1c3",        "linalg"),
    ("starcoder2-c1c3",       "arith+func"),
    ("starcoder2-c1c3",       "linalg"),
    # Item-3: Granite-Code-8B fp16 (MLX) on linalg, 3 seeds.
    ("granite-8b-fp16-c1c3",  "linalg"),
]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    rows = []
    for p in [
        REPO / "results/day51_seed0_n200/multiseed_seed0.jsonl",
        REPO / "results/day39pm/multiseed_fulln.jsonl",
        REPO / "results/day50/baseline_linalg_multiseed.jsonl",
        REPO / "results/day51/granite_8b_fp16_linalg.jsonl",
    ]:
        more = _load_jsonl(p)
        rows.extend(more)
        print(f"  loaded {len(more):4d} from {p.name}", file=sys.stderr)

    # Group by (model, dialect, seed) → list of {0,1} verify_valid.
    by_cell_seed: dict[tuple, list[int]] = defaultdict(list)
    for r in rows:
        model = r.get("model", "")
        dialect = r.get("dialect", "")
        seed = r.get("seed", 0)
        if (model, dialect) not in CELL_KEYS:
            continue
        by_cell_seed[(model, dialect, seed)].append(1 if r.get("verify_valid") else 0)

    # Aggregate per cell.
    cells_out: dict[str, dict] = {}
    for (model, dialect) in CELL_KEYS:
        cell_id = f"{model}::{dialect}"
        seeds = {}
        for s in (0, 1, 2):
            arr = by_cell_seed.get((model, dialect, s), [])
            if arr:
                ci = bootstrap_ci(arr, n_resamples=10_000, seed=0).as_dict()
                seeds[s] = {
                    "n": len(arr),
                    "verify_rate": sum(arr) / len(arr),
                    "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                }
            else:
                seeds[s] = None
        present = [seeds[s]["verify_rate"] for s in (0, 1, 2) if seeds[s]]
        if present:
            mean_v = sum(present) / len(present)
            half_range = (max(present) - min(present)) / 2.0
            ci_lows  = [seeds[s]["ci_low"]  for s in (0, 1, 2) if seeds[s]]
            ci_highs = [seeds[s]["ci_high"] for s in (0, 1, 2) if seeds[s]]
            ci_low_min = min(ci_lows)
            ci_high_max = max(ci_highs)
        else:
            mean_v = float("nan")
            half_range = float("nan")
            ci_low_min = ci_high_max = float("nan")
        cells_out[cell_id] = {
            "model": model, "dialect": dialect,
            "seeds": seeds,
            "mean": mean_v,
            "half_range": half_range,
            "ci_low": ci_low_min,
            "ci_high": ci_high_max,
            "n_per_seed": [seeds[s]["n"] if seeds[s] else 0 for s in (0, 1, 2)],
        }

    out_path = REPO / "results/day51_seed0_n200/apples_to_apples.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"cells": cells_out}, indent=2))
    print("\n=== APPLES-TO-APPLES (uniform n=200/125 across all 3 seeds) ===",
          file=sys.stderr)
    print(f"{'cell':40s}  {'mean':>6s}  {'±halfrange':>10s}  {'ci_low':>6s}  {'ci_high':>6s}  n_per_seed",
          file=sys.stderr)
    for cell_id, info in cells_out.items():
        ns = info["n_per_seed"]
        print(f"  {cell_id:40s}  {info['mean']:6.3f}  {info['half_range']:10.3f}  "
              f"{info['ci_low']:6.3f}  {info['ci_high']:6.3f}  {ns}",
              file=sys.stderr)
    print(f"\n[day51] → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
