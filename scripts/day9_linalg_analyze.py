"""Day-9 analysis: bootstrap CIs + paired deltas on results/day9/linalg_smoke.jsonl."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = "/Users/plawanrath/Documents/GitHub-public/slm-to-mlir-constrained-emitter"
os.chdir(REPO); sys.path.insert(0, REPO)

import numpy as np

from eval.stats import bootstrap_ci, paired_bootstrap_diff


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines()]


def _per_cell(rows):
    cells = {}
    for r in rows:
        cells.setdefault(r["constraint"], []).append(r)
    for c in cells.values():
        c.sort(key=lambda r: r["prompt_id"])
    return cells


def _pair(a, b, field):
    a_by = {r["prompt_id"]: r for r in a}
    b_by = {r["prompt_id"]: r for r in b}
    ids = sorted(set(a_by) & set(b_by))
    return ([int(bool(a_by[i][field])) for i in ids],
            [int(bool(b_by[i][field])) for i in ids])


def main():
    p = Path("results/day9/linalg_smoke.jsonl")
    rows = _load(p)
    cells = _per_cell(rows)

    summary = {"input": str(p), "n_rows": len(rows), "cells": {}, "paired_diffs": {}, "gate": {}}
    print(f"{'cell':>10s}  {'n':>4s}  {'parse%':>18s}  {'verify%':>18s}  {'att':>4s}  {'dt_s':>5s}")
    for name, cell in cells.items():
        n = len(cell)
        parse = [int(bool(r["parse_valid"])) for r in cell]
        verify = [int(bool(r["verify_valid"])) for r in cell]
        att = float(np.mean([r.get("attempts", 1) for r in cell]))
        dt = float(np.mean([r.get("dt", 0.0) for r in cell]))
        p_ci = bootstrap_ci(parse); v_ci = bootstrap_ci(verify)
        summary["cells"][name] = {"n": n, "mean_attempts": att, "mean_dt_s": dt,
                                  "parse": p_ci.as_dict(), "verify": v_ci.as_dict()}
        print(
            f"{name:>10s}  {n:>4d}  "
            f"{100*p_ci.point:>5.1f} [{100*p_ci.ci_low:>4.1f},{100*p_ci.ci_high:>4.1f}]  "
            f"{100*v_ci.point:>5.1f} [{100*v_ci.ci_low:>4.1f},{100*v_ci.ci_high:>4.1f}]  "
            f"{att:>4.2f}  {dt:>5.2f}"
        )

    def _paired(a, b):
        if a not in cells or b not in cells: return
        av, bv = _pair(cells[a], cells[b], "verify_valid")
        d = paired_bootstrap_diff(av, bv)
        summary["paired_diffs"][f"{a}_vs_{b}"] = d.as_dict()
        sign = "+" if d.point >= 0 else ""
        print(f"  Δverify  {a} − {b}: {sign}{100*d.point:>5.1f}pp "
              f"[{100*d.ci_low:>5.1f},{100*d.ci_high:>5.1f}]  p={d.p_value:.4f}  n={d.n}")

    print("\nPaired diffs (verify):")
    _paired("c1_c2_c3", "c1_c2")
    _paired("c1_c2_c3", "c1")
    _paired("c1_c2", "c1")
    _paired("c1", "none")

    gate_key = "c1_c2_c3_vs_c1_c2"
    if gate_key in summary["paired_diffs"]:
        d = summary["paired_diffs"][gate_key]
        dpp = 100 * d["point"]; lpp = 100 * d["ci_low"]
        summary["gate"]["week2_linalg_c3_vs_c1c2"] = {
            "delta_pp": dpp, "ci_low_pp": lpp, "target_delta_pp": 10.0,
            "met": (dpp >= 10.0) and (lpp > 0.0),
        }
        print(f"\nWeek-2 gate on linalg (C3 vs C1+C2 ≥10pp, CI_low>0): "
              f"Δ={dpp:.1f}pp, CI_low={lpp:.1f}pp → "
              f"{'MET' if summary['gate']['week2_linalg_c3_vs_c1c2']['met'] else 'NOT MET'}")

    out = Path("results/day9/linalg_summary.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary → {out}")


if __name__ == "__main__":
    main()
