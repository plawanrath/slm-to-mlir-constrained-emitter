"""Hidden-Cost-of-Structure replication + reversal (design §5.3 item 2).

Replicates the RANLP 2025 "Hidden Cost of Structure" finding that plain
CFG-grammar-constrained decoding can hurt small models, then shows that
adding C2 (ODS-derived type+arity) reverses the effect.

Produces results/hcs_ablation.json with:
  {
    "model": "...",
    "dialect": "...",
    "pass_rates": {"base": ..., "c1_only": ..., "c1_c2": ...},
    "reversal_detected": bool,
    "c1_vs_base_paired_diff": {...},
    "c1c2_vs_c1_paired_diff": {...},
  }

Usage:
    python -m eval.ablations.hcs_replication \
        --matrix results/matrix.jsonl \
        --model phi-3.5-mini \
        --dialect arith+func \
        --out results/hcs_ablation.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.stats import paired_bootstrap_diff, pass_rate


def _filter(rows: list[dict], **eq) -> list[dict]:
    return [r for r in rows if all(r.get(k) == v for k, v in eq.items())]


def _align_by_prompt(a: list[dict], b: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Align two row lists by prompt_id, return paired binary arrays."""
    a_map = {(r["prompt_id"], r["seed"]): int(r["passed"]) for r in a}
    b_map = {(r["prompt_id"], r["seed"]): int(r["passed"]) for r in b}
    keys = sorted(set(a_map) & set(b_map))
    aa = np.array([a_map[k] for k in keys], dtype=int)
    bb = np.array([b_map[k] for k in keys], dtype=int)
    return aa, bb


def run(rows: list[dict], model: str, dialect: str) -> dict:
    base = _filter(rows, model=model, dialect=dialect, constraint="none")
    c1 = _filter(rows, model=model, dialect=dialect, constraint="c1")
    c1c2 = _filter(rows, model=model, dialect=dialect, constraint="c1_c2")

    base_rate = pass_rate([int(r["passed"]) for r in base])
    c1_rate = pass_rate([int(r["passed"]) for r in c1])
    c1c2_rate = pass_rate([int(r["passed"]) for r in c1c2])

    c1_vs_base = paired_bootstrap_diff(*_align_by_prompt(c1, base))
    c1c2_vs_c1 = paired_bootstrap_diff(*_align_by_prompt(c1c2, c1))

    # "Reversal" = plain C1 hurt the model, and C1+C2 restored or surpassed base.
    reversal = (c1_vs_base.point < 0) and (c1c2_rate >= base_rate)

    return {
        "model": model,
        "dialect": dialect,
        "pass_rates": {
            "base": base_rate,
            "c1_only": c1_rate,
            "c1_c2": c1c2_rate,
        },
        "reversal_detected": reversal,
        "c1_vs_base_paired_diff": c1_vs_base.as_dict(),
        "c1c2_vs_c1_paired_diff": c1c2_vs_c1.as_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dialect", required=True)
    ap.add_argument("--out", type=Path, default=Path("results/hcs_ablation.json"))
    args = ap.parse_args()
    rows = [json.loads(l) for l in args.matrix.read_text().splitlines() if l.strip()]
    result = run(rows, args.model, args.dialect)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"[hcs] wrote → {args.out}")


if __name__ == "__main__":
    main()
