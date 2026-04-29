"""Day-52 aggregator: append the granite-8b-fp16-c1c3::arith+func cell to
results/day51_seed0_n200/apples_to_apples.json.

Loads the existing apples_to_apples.json (preserves all 9 cells unchanged),
adds the new arith+func cell (3 seeds × n=200 from
results/day52/granite_8b_fp16_arith.jsonl), and computes paired-bootstrap
deltas vs SmolLM2 + C1+C2+C3 and CodeLlama-34B + C1+C3 on the same
arith+func prompt pool (paired by (seed, prompt_id)).

Inputs:
  - results/day51_seed0_n200/apples_to_apples.json    (existing 9 cells)
  - results/day52/granite_8b_fp16_arith.jsonl         (new cell, 600 rows)
  - results/day51_seed0_n200/multiseed_seed0.jsonl    (seed 0 references)
  - results/day39pm/multiseed_fulln.jsonl             (seed 1,2 references)

Output:
  - results/day51_seed0_n200/apples_to_apples.json    (in-place; +1 cell)

Regression invariant: every existing cell's mean/half_range/CI bytes are
identical after this script runs (we re-serialize the dict but only write
a new key — nothing else is recomputed).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from eval.stats import bootstrap_ci, paired_bootstrap_diff

NEW_MODEL = "granite-8b-fp16-c1c3"
NEW_DIALECT = "arith+func"
CELL_ID = f"{NEW_MODEL}::{NEW_DIALECT}"
SEEDS = (0, 1, 2)
N_PER_SEED = 200

REF_CELLS = [
    ("smollm2-c1c2c3", "smollm2"),
    ("codellama-c1c3", "codellama_34b"),
]

APPLES_PATH = REPO / "results/day51_seed0_n200/apples_to_apples.json"
NEW_JSONL = REPO / "results/day52/granite_8b_fp16_arith.jsonl"
REF_JSONLS = [
    REPO / "results/day51_seed0_n200/multiseed_seed0.jsonl",
    REPO / "results/day39pm/multiseed_fulln.jsonl",
]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _by_seed_pid(rows: list[dict], model: str, dialect: str) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for r in rows:
        if r.get("model") != model or r.get("dialect") != dialect:
            continue
        seed = int(r.get("seed", 0))
        pid = int(r["prompt_id"])
        out[(seed, pid)] = 1 if r.get("verify_valid") else 0
    return out


def main() -> None:
    if not APPLES_PATH.exists():
        print(f"ERR: missing {APPLES_PATH}", file=sys.stderr)
        sys.exit(1)
    existing = json.loads(APPLES_PATH.read_text())
    pre_keys = sorted(existing["cells"].keys())
    pre_snapshot = {k: json.dumps(existing["cells"][k], sort_keys=True)
                    for k in pre_keys if k != CELL_ID}

    new_rows = _load_jsonl(NEW_JSONL)
    print(f"  loaded {len(new_rows):4d} from {NEW_JSONL.name}", file=sys.stderr)
    new_pairs = _by_seed_pid(new_rows, NEW_MODEL, NEW_DIALECT)

    # Per-seed bootstrap CIs + cross-seed mean/half-range.
    seeds: dict[int, dict | None] = {}
    for s in SEEDS:
        arr = [v for (sd, _pid), v in sorted(new_pairs.items()) if sd == s]
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
    present = [seeds[s]["verify_rate"] for s in SEEDS if seeds[s]]
    if not present:
        print(f"ERR: no rows in {NEW_JSONL}", file=sys.stderr)
        sys.exit(1)
    mean_v = sum(present) / len(present)
    half_range = (max(present) - min(present)) / 2.0
    ci_lows = [seeds[s]["ci_low"] for s in SEEDS if seeds[s]]
    ci_highs = [seeds[s]["ci_high"] for s in SEEDS if seeds[s]]
    n_per_seed = [seeds[s]["n"] if seeds[s] else 0 for s in SEEDS]

    # Paired bootstrap vs reference cells, pooling all 3 seeds.
    ref_rows: list[dict] = []
    for p in REF_JSONLS:
        ref_rows.extend(_load_jsonl(p))
    paired_vs: dict[str, dict] = {}
    a_vec_cache = {(s, pid): v for (s, pid), v in new_pairs.items()}
    for ref_model, label in REF_CELLS:
        ref_pairs = _by_seed_pid(ref_rows, ref_model, NEW_DIALECT)
        keys = sorted(set(a_vec_cache.keys()) & set(ref_pairs.keys()))
        a_vec = [a_vec_cache[k] for k in keys]
        b_vec = [ref_pairs[k] for k in keys]
        if not a_vec:
            print(f"  WARN: no paired keys vs {ref_model}", file=sys.stderr)
            continue
        diff = paired_bootstrap_diff(a_vec, b_vec, n_resamples=10_000, seed=0)
        paired_vs[label] = {
            "ref_model": ref_model,
            "n_paired": len(a_vec),
            "rate_a": float(sum(a_vec) / len(a_vec)),
            "rate_b": float(sum(b_vec) / len(b_vec)),
            "delta": diff.point,
            "ci_low": diff.ci_low,
            "ci_high": diff.ci_high,
            "p_value_a_gt_b": diff.p_value,
        }
        print(f"  paired vs {ref_model:20s}: Δ={diff.point:+.3f} "
              f"[{diff.ci_low:+.3f},{diff.ci_high:+.3f}] "
              f"p(a>b)={diff.p_value:.3f}  n_paired={len(a_vec)}",
              file=sys.stderr)

    new_cell = {
        "model": NEW_MODEL, "dialect": NEW_DIALECT,
        "seeds": seeds,
        "mean": mean_v,
        "half_range": half_range,
        "ci_low": min(ci_lows),
        "ci_high": max(ci_highs),
        "n_per_seed": n_per_seed,
        "paired_vs": paired_vs,
    }
    existing["cells"][CELL_ID] = new_cell

    # Regression: every previously-existing cell must be byte-identical.
    for k, snap in pre_snapshot.items():
        cur = json.dumps(existing["cells"][k], sort_keys=True)
        if cur != snap:
            print(f"ERR: cell {k} changed unexpectedly", file=sys.stderr)
            sys.exit(2)

    APPLES_PATH.write_text(json.dumps(existing, indent=2))
    print(f"\n[day52] cell {CELL_ID}: mean={mean_v:.3f} ±{half_range:.3f}  "
          f"CI=[{min(ci_lows):.3f},{max(ci_highs):.3f}]  "
          f"n_per_seed={n_per_seed}",
          file=sys.stderr)
    print(f"[day52] → {APPLES_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
