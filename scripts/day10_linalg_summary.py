"""Day-10 final summary: merge SmolLM2 linalg + 30B linalg + error categories,
produce the cross-dialect §5.2 matrix + Week-3 gate decision.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
os.chdir(REPO); sys.path.insert(0, REPO)

import numpy as np
from eval.stats import bootstrap_ci, paired_bootstrap_diff


def _load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines()]


def _cell_rate(rows, field="verify_valid"):
    vals = [int(bool(r[field])) for r in rows]
    ci = bootstrap_ci(vals)
    return ci


def _per_cell(rows, key_field="constraint"):
    cells = {}
    for r in rows:
        cells.setdefault(r[key_field], []).append(r)
    for c in cells.values():
        c.sort(key=lambda r: r["prompt_id"])
    return cells


def _per_cell_by_model(rows):
    cells = {}
    for r in rows:
        cells.setdefault(r["model"], []).append(r)
    for c in cells.values():
        c.sort(key=lambda r: r["prompt_id"])
    return cells


def main():
    out = {
        "arith_func": {},
        "linalg": {},
        "week3_gate": {},
    }

    # ----- arith+func from Day 5 frozen -----
    day5 = _load("results/frozen/week2_gate_day5/c3_smoke.jsonl")
    d5 = _per_cell(day5)
    for name in ("none", "c1", "c1_c2", "c1_c2_c3"):
        ci = _cell_rate(d5[name])
        out["arith_func"][f"smollm2-{name}"] = ci.as_dict()
    # arith+func 30B from Day 4
    d4 = _load("results/day4/baselines_30b.jsonl")
    b4 = _per_cell_by_model(d4)  # model field already uniquely names the cell
    for name, cell in b4.items():
        out["arith_func"][name] = _cell_rate(cell).as_dict()

    # ----- linalg from Day 9 SmolLM2 -----
    day9 = _load("results/day9/linalg_smoke.jsonl")
    d9 = _per_cell(day9)
    for name in ("none", "c1", "c1_c2", "c1_c2_c3"):
        ci = _cell_rate(d9[name])
        out["linalg"][f"smollm2-{name}"] = ci.as_dict()
    # linalg 30B from Day 10
    try:
        day10 = _load("results/day10/linalg_baselines.jsonl")
        b10 = _per_cell_by_model(day10)
        for name, cell in b10.items():
            out["linalg"][name] = _cell_rate(cell).as_dict()
    except FileNotFoundError:
        pass

    # ----- Week-3 gate: SLM + C1+C2 within 3pp of 30B + C1 on ≥1 dialect -----
    for dialect in ("arith_func", "linalg"):
        cells = out[dialect]
        slm = cells.get("smollm2-c1_c2_c3") or cells.get("smollm2-c1_c2")
        if slm is None:
            continue
        best_delta = None
        best_b = None
        for b_name in ("codellama-c1", "granite-c1"):
            b = cells.get(b_name)
            if b is None:
                continue
            delta = 100 * (slm["point"] - b["point"])
            if best_delta is None or delta > best_delta:
                best_delta = delta
                best_b = b_name
        out["week3_gate"][dialect] = {
            "slm_system": "smollm2-c1_c2_c3" if "smollm2-c1_c2_c3" in cells else "smollm2-c1_c2",
            "best_baseline": best_b,
            "slm_verify_pct": 100 * slm["point"],
            "best_baseline_verify_pct": 100 * (cells[best_b]["point"] if best_b else 0),
            "delta_pp": best_delta,
            "within_3pp": (best_delta is not None and best_delta >= -3.0),
            "beats":        (best_delta is not None and best_delta > 0),
        }

    # ----- print cross-dialect table -----
    print("\n=== §5.2 Matrix (SmolLM2-1.7B + 30B baselines, verify%) ===")
    SYS_ORDER = [
        ("smollm2-none",    "SmolLM2-1.7B, free"),
        ("smollm2-c1",      "SmolLM2-1.7B + C1"),
        ("smollm2-c1_c2",   "SmolLM2-1.7B + C1+C2"),
        ("smollm2-c1_c2_c3","SmolLM2-1.7B + C1+C2+C3 (ours)"),
        ("granite-free",    "Granite-Code-34B, free"),
        ("granite-c1",      "Granite-Code-34B + C1"),
        ("codellama-c1",    "CodeLlama-34B + C1"),
    ]
    print(f"{'System':<40s}  {'arith+func (n=200)':>24s}  {'linalg (n=125)':>24s}")
    for key, label in SYS_ORDER:
        a = out["arith_func"].get(key); l = out["linalg"].get(key)
        def _f(c):
            if c is None:
                return "—".rjust(24)
            return f"{100*c['point']:>5.1f} [{100*c['ci_low']:>4.1f}, {100*c['ci_high']:>4.1f}]"
        print(f"{label:<40s}  {_f(a):>24s}  {_f(l):>24s}")

    print("\n=== Week-3 Gate (SLM + C1+C2 within 3pp of 30B + C1 on ≥1 dialect) ===")
    for dialect, g in out["week3_gate"].items():
        print(f"  {dialect}:")
        print(f"    {g['slm_system']} = {g['slm_verify_pct']:.1f}%")
        if g.get("best_baseline") is None:
            print(f"    no baseline data available")
            continue
        print(f"    {g['best_baseline']} = {g['best_baseline_verify_pct']:.1f}%")
        print(f"    Δ = {g['delta_pp']:+.1f}pp  within-3pp={g['within_3pp']}  beats={g['beats']}")

    Path("results/day10").mkdir(exist_ok=True)
    summary_path = Path("results/day10/linalg_matrix_summary.json")
    summary_path.write_text(json.dumps(out, indent=2))
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
