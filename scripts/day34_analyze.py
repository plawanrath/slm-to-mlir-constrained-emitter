"""Day-34 analyze: multi-seed comparison table.

Loads seed-0 data from Day-5/9/18/21b + seed-1,2 from Day-34. Reports
each cell's 3-seed verify rate (mean ± range-half-width) on the shared
first-100-prompt subset for apples-to-apples comparison.

Output: results/day34/multiseed_summary.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = "/Users/plawanrath/Documents/GitHub-public/slm-to-mlir-constrained-emitter"
os.chdir(REPO); sys.path.insert(0, REPO)


def _verify_first_n(path: Path, filt, n=100) -> float | None:
    if not path.exists(): return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if json.loads(l).get("verify_valid") is not None]
    rows = [r for r in rows if filt(r)]
    rows.sort(key=lambda r: r.get("prompt_id", 0))
    if not rows: return None
    sub = rows[:n]
    return sum(int(bool(r["verify_valid"])) for r in sub) / len(sub)


def main() -> None:
    # Seed-0 first-100 from existing frozen runs.
    seed0 = {
        "smollm2-arith": _verify_first_n(
            Path("results/day5/c3_smoke.jsonl"),
            lambda r: r.get("constraint") == "c1_c2_c3"),
        "smollm2-linalg": _verify_first_n(
            Path("results/day9/linalg_smoke.jsonl"),
            lambda r: r.get("constraint") == "c1_c2_c3"),
        "granite-c1c3-arith": _verify_first_n(
            Path("results/day18/30b_c1c3_rejection.jsonl"),
            lambda r: r.get("model") == "granite-c1c3" and r.get("dialect", "arith+func") == "arith+func"),
        "codellama-c1c3-arith": _verify_first_n(
            Path("results/day18/30b_c1c3_rejection.jsonl"),
            lambda r: r.get("model") == "codellama-c1c3" and r.get("dialect", "arith+func") == "arith+func"),
        "starcoder2-c1c3-arith": _verify_first_n(
            Path("results/day21/starcoder2_c1c3.jsonl"),
            lambda r: r.get("model") == "starcoder2-c1c3" and r.get("dialect") == "arith+func"),
    }

    # Seed-1, seed-2 from Day 34.
    day34 = [json.loads(l) for l in Path("results/day34/multiseed.jsonl").read_text().splitlines()]

    def day34_cell(model: str, dialect: str, seed: int) -> float | None:
        cell = [r for r in day34
                if r["model"] == model and r["dialect"] == dialect and r["seed"] == seed]
        if not cell: return None
        return sum(int(bool(r["verify_valid"])) for r in cell) / len(cell)

    seed1 = {
        "smollm2-arith":         day34_cell("smollm2-c1c2c3",  "arith+func", 1),
        "smollm2-linalg":        day34_cell("smollm2-c1c2c3",  "linalg",     1),
        "granite-c1c3-arith":    day34_cell("granite-c1c3",    "arith+func", 1),
        "codellama-c1c3-arith":  day34_cell("codellama-c1c3",  "arith+func", 1),
        "starcoder2-c1c3-arith": day34_cell("starcoder2-c1c3", "arith+func", 1),
    }
    seed2 = {
        "smollm2-arith":         day34_cell("smollm2-c1c2c3",  "arith+func", 2),
        "smollm2-linalg":        day34_cell("smollm2-c1c2c3",  "linalg",     2),
        "granite-c1c3-arith":    day34_cell("granite-c1c3",    "arith+func", 2),
        "codellama-c1c3-arith":  day34_cell("codellama-c1c3",  "arith+func", 2),
        "starcoder2-c1c3-arith": day34_cell("starcoder2-c1c3", "arith+func", 2),
    }

    print(f"{'cell':<28s}  {'s0':>7s}  {'s1':>7s}  {'s2':>7s}  {'mean':>7s}  {'range/2':>7s}")
    summary: dict = {"cells": {}}
    for key in seed0:
        s0, s1, s2 = seed0[key], seed1[key], seed2[key]
        vals = [v for v in (s0, s1, s2) if v is not None]
        if not vals: continue
        mean = sum(vals) / len(vals)
        half_range = (max(vals) - min(vals)) / 2
        print(f"{key:<28s}  "
              f"{100*s0:>6.1f}%" if s0 is not None else f"{key:<28s}       --"
              ,
              end="  ",
        )
        print(f"{100*s1:>6.1f}%" if s1 is not None else "     --", end="  ")
        print(f"{100*s2:>6.1f}%" if s2 is not None else "     --", end="  ")
        print(f"{100*mean:>6.1f}%  ±{100*half_range:>5.1f}pp")
        summary["cells"][key] = {
            "s0": s0, "s1": s1, "s2": s2, "mean": mean, "half_range": half_range,
        }

    out = Path("results/day34/multiseed_summary.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary → {out}")


if __name__ == "__main__":
    main()
