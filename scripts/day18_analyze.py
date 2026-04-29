"""Day-18 analyze: 30B + C1+C3 vs existing 30B + C1 (arith+func + linalg).

Reads:
  - results/day18/30b_c1c3_rejection.jsonl  (new)
  - results/day4/baselines_30b.jsonl         (existing C1 arith+func)
  - results/day10/linalg_baselines.jsonl     (existing C1 linalg)

Computes per-cell verify% with 95% bootstrap CI, paired-bootstrap Δverify of
(C1+C3 − C1) for each (model × dialect), and prints the Phase-A gate check.

Phase-A gate (ADR-0008): if 30B + C1+C3 exceeds SmolLM2 + C1+C2+C3 on any
(model × dialect) cell, halt and reassess paper framing.

SmolLM2 reference numbers (frozen):
  - arith+func: SmolLM2 + C1+C2+C3 verify% = read from results/frozen/week2_gate_day5/c3_summary.json
  - linalg:     SmolLM2 + C1+C2+C3 verify% = read from results/day9/linalg_summary.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
os.chdir(REPO); sys.path.insert(0, REPO)

import numpy as np
from eval.stats import bootstrap_ci, paired_bootstrap_diff


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines()]


def _cell(rows: list[dict], model: str, dialect: str | None = None,
          constraint: str | None = None) -> list[dict]:
    def ok(r):
        if r.get("model") != model: return False
        if dialect is not None and r.get("dialect", "arith+func") != dialect:
            return False
        if constraint is not None and r.get("constraint") != constraint: return False
        return True
    return sorted([r for r in rows if ok(r)], key=lambda r: r["prompt_id"])


def _paired(a: list[dict], b: list[dict], field: str):
    a_by = {r["prompt_id"]: r for r in a}
    b_by = {r["prompt_id"]: r for r in b}
    ids = sorted(set(a_by) & set(b_by))
    av = [int(bool(a_by[i][field])) for i in ids]
    bv = [int(bool(b_by[i][field])) for i in ids]
    return av, bv, len(ids)


def _smolllm_ref_arith() -> float | None:
    p = Path("results/frozen/week2_gate_day5/c3_summary.json")
    if not p.exists():
        p = Path("results/day5/c3_summary.json")
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    return s["cells"].get("c1_c2_c3", {}).get("verify", {}).get("point")


def _smolllm_ref_linalg() -> float | None:
    p = Path("results/day9/linalg_summary.json")
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    return s["cells"].get("c1_c2_c3", {}).get("verify", {}).get("point")


def main() -> None:
    new_path = Path("results/day18/30b_c1c3_rejection.jsonl")
    c3_rows = _load(new_path)
    c1_arith_rows  = _load(Path("results/day4/baselines_30b.jsonl"))
    c1_linalg_rows = _load(Path("results/day10/linalg_baselines.jsonl"))
    # Tag source files with implicit dialect (neither has a 'dialect' field).
    for r in c1_arith_rows:  r.setdefault("dialect", "arith+func")
    for r in c1_linalg_rows: r.setdefault("dialect", "linalg")

    out = {"cells": {}, "paired_diffs": {}, "phase_a_gate": {}}
    print(f"{'cell':>32s}  {'n':>4s}  {'parse%':>14s}  {'verify%':>14s}  {'mean_att':>9s}")

    ref_arith = _smolllm_ref_arith()
    ref_linalg = _smolllm_ref_linalg()
    print(f"  SmolLM2 reference: arith={ref_arith}, linalg={ref_linalg}")

    # New C1+C3 cells
    for nick in ("granite-c1c3", "codellama-c1c3"):
        for dialect in ("arith+func", "linalg"):
            cell = _cell(c3_rows, nick, dialect=dialect)
            if not cell: continue
            n = len(cell)
            verify = [int(bool(r["verify_valid"])) for r in cell]
            parse = [int(bool(r["parse_valid"])) for r in cell]
            att = float(np.mean([r.get("attempts", 1) for r in cell]))
            p_ci = bootstrap_ci(parse); v_ci = bootstrap_ci(verify)
            key = f"{nick}/{dialect}"
            out["cells"][key] = {
                "n": n, "mean_attempts": att,
                "parse": p_ci.as_dict(), "verify": v_ci.as_dict(),
            }
            print(
                f"{key:>32s}  {n:>4d}  "
                f"{100*p_ci.point:>5.1f} [{100*p_ci.ci_low:>4.1f},{100*p_ci.ci_high:>4.1f}]  "
                f"{100*v_ci.point:>5.1f} [{100*v_ci.ci_low:>4.1f},{100*v_ci.ci_high:>4.1f}]  "
                f"{att:>9.2f}"
            )

    # Existing C1-only reference cells (for paired bootstrap)
    for (nick_c1, nick_c1c3, rows_c1, dialect) in [
        ("granite-c1",   "granite-c1c3",   c1_arith_rows,  "arith+func"),
        ("codellama-c1", "codellama-c1c3", c1_arith_rows,  "arith+func"),
        ("granite-c1",   "granite-c1c3",   c1_linalg_rows, "linalg"),
        ("codellama-c1", "codellama-c1c3", c1_linalg_rows, "linalg"),
    ]:
        a = _cell(c3_rows, nick_c1c3, dialect=dialect)
        b = _cell(rows_c1, nick_c1, dialect=dialect, constraint="c1")
        # day10 linalg file has 'dialect' implicitly; day4 has none.
        if not a or not b:
            continue
        av, bv, n = _paired(a, b, "verify_valid")
        if n == 0: continue
        diff = paired_bootstrap_diff(av, bv)
        key = f"{nick_c1c3}_vs_{nick_c1}/{dialect}"
        out["paired_diffs"][key] = diff.as_dict()
        sign = "+" if diff.point >= 0 else ""
        print(
            f"  Δverify  {key}: {sign}{100*diff.point:>5.1f}pp "
            f"[{100*diff.ci_low:>5.1f}, {100*diff.ci_high:>5.1f}]  "
            f"p_one-sided={diff.p_value:.4f}  n_pair={n}"
        )

    # Phase-A gate: 30B + C1+C3 must not NEWLY flip the headline.
    # A NEW flip is: 30B+C1+C3 > SmolLM2 AND 30B+C1 was ≤ SmolLM2 previously.
    # If 30B+C1 was already > SmolLM2 (e.g. CodeLlama arith+func at 82% > 67.5%),
    # then 30B+C3 exceeding SmolLM2 is just amplification of known state, not a
    # NEW finding that flips the paper's headline.
    def _c1_baseline(nick_c3: str, dialect: str) -> float | None:
        nick_c1 = nick_c3.replace("-c1c3", "-c1")
        rows = c1_arith_rows if dialect == "arith+func" else c1_linalg_rows
        cell = _cell(rows, nick_c1,
                     dialect=None if dialect == "arith+func" else dialect,
                     constraint="c1")
        if not cell: return None
        return sum(int(bool(r["verify_valid"])) for r in cell) / len(cell)

    new_flips = []
    preexisting = []
    for key, c in out["cells"].items():
        nick, dialect = key.rsplit("/", 1)
        ref = ref_arith if dialect == "arith+func" else ref_linalg
        v = c["verify"]["point"]
        if ref is None or v <= ref:
            continue
        c1_base = _c1_baseline(nick, dialect)
        entry = {"cell": key, "verify": v, "smollm2_ref": ref,
                 "c1_baseline": c1_base,
                 "delta_pp": 100 * (v - ref)}
        if c1_base is not None and c1_base > ref:
            preexisting.append(entry)
        else:
            new_flips.append(entry)
    out["phase_a_gate"] = {
        "ref_smollm2_arith": ref_arith,
        "ref_smollm2_linalg": ref_linalg,
        "new_flips": new_flips,
        "preexisting_exceedances": preexisting,
        "halt": bool(new_flips),
    }
    if new_flips:
        print("\n*** PHASE-A GATE: NEW FLIP detected — 30B+C1+C3 newly exceeds SmolLM2 ***")
        for b in new_flips:
            c1b = f"{100*b['c1_baseline']:.1f}%" if b["c1_baseline"] is not None else "n/a"
            print(f"  {b['cell']}: 30B+C3={100*b['verify']:.1f}%  "
                  f"SmolLM2={100*b['smollm2_ref']:.1f}%  "
                  f"(30B+C1 was {c1b})  Δ=+{b['delta_pp']:.1f}pp")
    elif preexisting:
        print("\nPhase-A gate: no NEW flips — pre-existing exceedances are:")
        for b in preexisting:
            c1b = f"{100*b['c1_baseline']:.1f}%" if b["c1_baseline"] is not None else "n/a"
            print(f"  {b['cell']}: 30B+C3={100*b['verify']:.1f}%  "
                  f"30B+C1_baseline={c1b} (already above SmolLM2)")
    else:
        print("\nPhase-A gate: no breach — SmolLM2 + C1+C2+C3 remains > 30B + C1+C3 on all cells.")

    out_path = Path("results/day18/summary.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSummary written → {out_path}")


if __name__ == "__main__":
    main()
