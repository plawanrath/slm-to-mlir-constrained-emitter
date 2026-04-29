"""Day-5 §4.0 diagnostic: how well does C3 scope validation predict
mlir-opt verify pass/fail on the Day-4 SmolLM2 + C1+C2 cell?

This is a pure post-mortem on existing generations — no model calls. It tells
us:
  - scope-pass ∩ verify-pass: upper-bound on what C3 rejection sampling
    lets through.
  - scope-pass ∩ verify-fail: false positives (C3 accepted, mlir-opt still
    rejected — error class outside C3's reach).
  - scope-fail ∩ verify-pass: false negatives (C3 would reject something
    mlir-opt accepts — grammar / validator bug).
  - scope-fail ∩ verify-fail: the sweet spot — ejections that mlir-opt would
    have caught.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
os.chdir(REPO)
sys.path.insert(0, REPO)

from decoder.c3_scope import summarize_violations, validate


def main() -> None:
    path = Path("results/day4/slm_smoke.jsonl")
    rows = [json.loads(l) for l in path.read_text().splitlines()]

    cells: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        cells.setdefault((r["model"], r["constraint"]), []).append(r)

    for (model, constraint), cell in sorted(cells.items()):
        total = len(cell)
        # 2x2 confusion matrix
        sp_vp = sp_vf = sf_vp = sf_vf = 0
        kind_counts: dict[str, int] = {}
        for r in cell:
            rep = validate(r["generated"])
            scope_pass = rep.passed
            verify_pass = bool(r.get("verify_valid", False))
            if scope_pass and verify_pass:
                sp_vp += 1
            elif scope_pass and not verify_pass:
                sp_vf += 1
            elif (not scope_pass) and verify_pass:
                sf_vp += 1
            else:
                sf_vf += 1
            if not scope_pass:
                for k, v in summarize_violations(rep.violations).items():
                    kind_counts[k] = kind_counts.get(k, 0) + v
        pct = lambda n: f"{n}/{total} ({100*n/total:.1f}%)"
        print(f"\n{model} / {constraint}  (n={total})")
        print(f"  scope_pass  verify_pass : {pct(sp_vp)}")
        print(f"  scope_pass  verify_FAIL : {pct(sp_vf)}  ← C3 misses (not SSA/type)")
        print(f"  scope_FAIL  verify_pass : {pct(sf_vp)}  ← C3 false-rejects")
        print(f"  scope_FAIL  verify_FAIL : {pct(sf_vf)}  ← C3 would correctly reject")
        if kind_counts:
            print(f"  violation kinds: {kind_counts}")


if __name__ == "__main__":
    main()
