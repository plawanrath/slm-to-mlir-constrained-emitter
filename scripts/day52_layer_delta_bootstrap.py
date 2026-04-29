"""Day-52: paired-bootstrap CIs for the three Table-2 layer deltas.

Today the §Ablations paragraph cites three CIs whose source file is unclear;
the point estimates match Table 2's seed-0 cells exactly, but the CI bounds
predate Day 51 and have no auditable JSON file. This script recomputes them
from the same frozen per-prompt jsonls that back Table 2 / §5.2 in the paper,
under the same paired-bootstrap protocol used by the rest of the paper.

The three deltas (all SmolLM2-1.7B at seed 0):

  Δ1: arith+func, n=200, (C1+C2+C3) − (C1+C2)        — expected +13.0 pp
  Δ2: linalg,     n=125, (C1+C2+C3) − (C1+C2)        — expected  +0.8 pp
  Δ3: linalg,     n=125, (C1+C2)    − (C1)           — expected  +4.0 pp

Protocol (matches eval/stats.py paired_bootstrap_diff):
  - 10,000 paired-bootstrap resamples at the prompt level (resample
    prompt_id with replacement; each resampled prompt_id contributes its
    paired (higher_cell, lower_cell) outcomes to both arms simultaneously)
  - 95% percentile-method CI on the per-resample mean paired difference
  - One-sided p-value: Pr[paired-mean ≤ 0] for positive deltas
  - numpy.random.default_rng(0) — deterministic, matches the paper's
    other paired bootstraps

Inputs (immutable frozen anchors — read-only):
  - results/frozen/week2_gate_day5/c3_smoke.jsonl
      SmolLM2 × {none, C1, C1+C2, C1+C2+C3} × arith+func, n=200, seed=0.
      4 constraint cells × 200 prompts = 800 rows.
  - results/frozen/day10_final_matrix/linalg_smoke.jsonl
      SmolLM2 × {none, C1, C1+C2, C1+C2+C3} × linalg, n=125, seed=0.
      4 constraint cells × 125 prompts = 500 rows.

Output:
  - results/day52/layer_delta_bootstrap.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)

ARITH_JSONL = REPO / "results/frozen/week2_gate_day5/c3_smoke.jsonl"
LINALG_JSONL = REPO / "results/frozen/day10_final_matrix/linalg_smoke.jsonl"
OUT_PATH = REPO / "results/day52/layer_delta_bootstrap.json"

CONSTRAINTS = ("none", "c1", "c1_c2", "c1_c2_c3")

# (delta_id, dialect, higher_cell, lower_cell, expected_pp, n_expected)
DELTAS = [
    ("arith+func_c1c2c3_minus_c1c2", "arith+func", "c1_c2_c3", "c1_c2", 13.0, 200),
    ("linalg_c1c2c3_minus_c1c2",     "linalg",     "c1_c2_c3", "c1_c2",  0.8, 125),
    ("linalg_c1c2_minus_c1",         "linalg",     "c1_c2",    "c1",     4.0, 125),
]


def _load_ladder(path: Path) -> dict[str, dict[int, int]]:
    """Return {constraint -> {prompt_id -> 0/1 verify_valid}}."""
    cells: dict[str, dict[int, int]] = {c: {} for c in CONSTRAINTS}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        c = r.get("constraint")
        pid = int(r["prompt_id"])
        if c not in cells:
            continue  # ignore any unexpected cells
        if pid in cells[c]:
            raise RuntimeError(f"duplicate (constraint={c}, prompt_id={pid}) in {path}")
        cells[c][pid] = 1 if r.get("verify_valid") else 0
    return cells


def _paired_bootstrap(
    higher: list[int],
    lower: list[int],
    n_resamples: int = 10_000,
    rng_seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Resample prompt indices with replacement; for each resample, compute
    the mean paired difference (higher - lower); return percentile CI and
    one-sided p-value Pr[paired-mean ≤ 0].
    """
    a = np.asarray(higher, dtype=float)
    b = np.asarray(lower,  dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    n = a.size
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    # Per-resample mean of (a - b) using the SAME prompt indices for both.
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    point = float(a.mean() - b.mean())
    low, high = np.quantile(diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    p_le_zero = float((diffs <= 0).mean())
    return {
        "point": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_one_sided_gt_zero": p_le_zero,  # Pr[bootstrap mean ≤ 0]
        "n_paired": int(n),
        "n_resamples": n_resamples,
    }


def main() -> None:
    arith = _load_ladder(ARITH_JSONL)
    linalg = _load_ladder(LINALG_JSONL)
    by_dialect = {"arith+func": arith, "linalg": linalg}

    # Pairing-validity check: the four constraint cells share the same
    # prompt_id list per dialect. (Already verified inline; assert.)
    for dialect, cells in by_dialect.items():
        ref = set(cells["none"].keys())
        for c in CONSTRAINTS:
            if set(cells[c].keys()) != ref:
                missing = ref.symmetric_difference(set(cells[c].keys()))
                print(f"ERR: {dialect}/{c} prompt_id mismatch vs none "
                      f"(diff={len(missing)})", file=sys.stderr)
                sys.exit(1)

    out: dict = {"deltas": {}}
    print()
    print(f"  {'delta_id':40s}  {'point':>7s}  {'CI_low':>7s}  {'CI_high':>7s}  {'p(≤0)':>7s}  {'n':>4s}")
    print(f"  {'-'*40}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*4}")
    for did, dialect, hi, lo, expected_pp, n_exp in DELTAS:
        cells = by_dialect[dialect]
        pids = sorted(cells["none"].keys())
        h = [cells[hi][p] for p in pids]
        l = [cells[lo][p] for p in pids]
        # Halt if the point estimate disagrees with the expected pinned value.
        observed_pp = (sum(h) / len(h) - sum(l) / len(l)) * 100.0
        if abs(observed_pp - expected_pp) > 0.05:
            print(
                f"ERR: point-estimate mismatch on {did}: "
                f"expected {expected_pp:+.1f} pp, got {observed_pp:+.2f} pp "
                f"(higher={hi} rate={sum(h)/len(h):.4f}, lower={lo} rate={sum(l)/len(l):.4f})",
                file=sys.stderr,
            )
            sys.exit(2)
        if len(h) != n_exp:
            print(f"ERR: n_paired mismatch on {did}: expected {n_exp}, got {len(h)}",
                  file=sys.stderr)
            sys.exit(3)
        res = _paired_bootstrap(h, l, n_resamples=10_000, rng_seed=0)
        out["deltas"][did] = res
        print(f"  {did:40s}  "
              f"{res['point']*100:+7.2f}  "
              f"{res['ci_low']*100:+7.2f}  "
              f"{res['ci_high']*100:+7.2f}  "
              f"{res['p_one_sided_gt_zero']:7.4f}  "
              f"{res['n_paired']:4d}")
    print()

    out["protocol"] = {
        "n_resamples": 10_000,
        "ci": "percentile_95",
        "rng_seed": 0,
        "rng_impl": "numpy.random.default_rng",
        "p_value_definition": "one_sided_pr_paired_mean_le_zero",
        "pairing": "by_prompt_id",
        "paired_bootstrap_method": "resample_prompt_indices_with_replacement",
    }
    out["source_files"] = {
        "arith+func": [
            "results/frozen/week2_gate_day5/c3_smoke.jsonl  (constraint=none,  n=200)",
            "results/frozen/week2_gate_day5/c3_smoke.jsonl  (constraint=c1,    n=200)",
            "results/frozen/week2_gate_day5/c3_smoke.jsonl  (constraint=c1_c2, n=200)",
            "results/frozen/week2_gate_day5/c3_smoke.jsonl  (constraint=c1_c2_c3, n=200)",
        ],
        "linalg": [
            "results/frozen/day10_final_matrix/linalg_smoke.jsonl  (constraint=none, n=125)",
            "results/frozen/day10_final_matrix/linalg_smoke.jsonl  (constraint=c1, n=125)",
            "results/frozen/day10_final_matrix/linalg_smoke.jsonl  (constraint=c1_c2, n=125)",
            "results/frozen/day10_final_matrix/linalg_smoke.jsonl  (constraint=c1_c2_c3, n=125)",
        ],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"  → {OUT_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
