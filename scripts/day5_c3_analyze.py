"""Day-5 §4.3 results analysis: reads results/day5/c3_smoke.jsonl and reports:

- per-cell pass rate + 95% bootstrap CI (parse + verify)
- paired bootstrap Δ for {c1_c2_c3 vs c1_c2} and {c1_c2_c3 vs c1}
- mean attempts per prompt in the C3 cell
- revised Week-2 gate decision (c1+c2+c3 ≥ c1+c2 by ≥10pp, CI lower bound > 0)

Writes results/day5/c3_summary.json for downstream figure generation.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = "/Users/plawanrath/Documents/GitHub-public/slm-to-mlir-constrained-emitter"
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np

from eval.stats import bootstrap_ci, paired_bootstrap_diff


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines()]


def _per_cell(rows: list[dict]) -> dict[str, list[dict]]:
    cells: dict[str, list[dict]] = {}
    for r in rows:
        cells.setdefault(r["constraint"], []).append(r)
    for c in cells.values():
        c.sort(key=lambda r: r["prompt_id"])
    return cells


def _aligned_pair(a: list[dict], b: list[dict], field: str) -> tuple[list[int], list[int]]:
    """Keep only prompt_ids present in both cells, aligned by prompt_id."""
    a_by = {r["prompt_id"]: r for r in a}
    b_by = {r["prompt_id"]: r for r in b}
    ids = sorted(set(a_by) & set(b_by))
    av = [int(bool(a_by[i][field])) for i in ids]
    bv = [int(bool(b_by[i][field])) for i in ids]
    return av, bv


def main() -> None:
    in_path = Path("results/day5/c3_smoke.jsonl")
    rows = _load(in_path)
    cells = _per_cell(rows)

    summary: dict = {
        "input": str(in_path),
        "n_rows": len(rows),
        "cells": {},
        "paired_diffs": {},
        "gate": {},
    }

    print(f"{'cell':>10s}  {'n':>4s}  {'parse%':>12s}  {'verify%':>12s}  {'mean_att':>9s}  {'mean_dt_s':>10s}")
    for name, cell in cells.items():
        n = len(cell)
        parse = [int(bool(r["parse_valid"])) for r in cell]
        verify = [int(bool(r["verify_valid"])) for r in cell]
        att = float(np.mean([r.get("attempts", 1) for r in cell]))
        dt = float(np.mean([r.get("dt", 0.0) for r in cell]))
        p_ci = bootstrap_ci(parse)
        v_ci = bootstrap_ci(verify)
        summary["cells"][name] = {
            "n": n, "mean_attempts": att, "mean_dt_s": dt,
            "parse": p_ci.as_dict(), "verify": v_ci.as_dict(),
        }
        print(
            f"{name:>10s}  {n:>4d}  "
            f"{100*p_ci.point:>5.1f} [{100*p_ci.ci_low:>4.1f},{100*p_ci.ci_high:>4.1f}]  "
            f"{100*v_ci.point:>5.1f} [{100*v_ci.ci_low:>4.1f},{100*v_ci.ci_high:>4.1f}]  "
            f"{att:>9.2f}  {dt:>10.2f}"
        )

    def _paired(name_a: str, name_b: str) -> None:
        if name_a not in cells or name_b not in cells:
            return
        av, bv = _aligned_pair(cells[name_a], cells[name_b], "verify_valid")
        diff = paired_bootstrap_diff(av, bv)
        summary["paired_diffs"][f"{name_a}_vs_{name_b}"] = diff.as_dict()
        sign = "+" if diff.point >= 0 else ""
        print(
            f"  Δverify  {name_a} − {name_b}: "
            f"{sign}{100*diff.point:>5.1f}pp "
            f"[{100*diff.ci_low:>5.1f}, {100*diff.ci_high:>5.1f}]  "
            f"p_one-sided={diff.p_value:.4f}  n_pair={diff.n}"
        )

    print("\nPaired differences (verify, aligned by prompt_id):")
    _paired("c1_c2_c3", "c1_c2")
    _paired("c1_c2_c3", "c1")
    _paired("c1_c2", "c1")
    _paired("c1", "none")

    # Revised Week-2 gate: c1+c2+c3 beats c1+c2 by ≥10pp, CI lower bound > 0.
    gate_key = "c1_c2_c3_vs_c1_c2"
    if gate_key in summary["paired_diffs"]:
        d = summary["paired_diffs"][gate_key]
        delta_pp = 100 * d["point"]
        low_pp = 100 * d["ci_low"]
        met = (delta_pp >= 10.0) and (low_pp > 0.0)
        summary["gate"]["week_2_c3_vs_c1c2"] = {
            "delta_pp": delta_pp, "ci_low_pp": low_pp,
            "target_delta_pp": 10.0, "met": met,
        }
        print(
            f"\nRevised Week-2 gate (c1+c2+c3 − c1+c2 ≥ 10pp, CI_low > 0):\n"
            f"  Δ = {delta_pp:.1f}pp, CI_low = {low_pp:.1f}pp  →  "
            f"{'MET' if met else 'NOT MET'}"
        )

    out = Path("results/day5/c3_summary.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written → {out}")


if __name__ == "__main__":
    main()
