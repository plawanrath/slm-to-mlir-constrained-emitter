"""Day-6 §5 summary: compose HCS + error-category tables into a single
structured JSON plus a plain-text report for the daily log.

Writes:
  results/day6/day6_summary.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = "/Users/plawanrath/Documents/GitHub-public/slm-to-mlir-constrained-emitter"
os.chdir(REPO); sys.path.insert(0, REPO)


def main() -> None:
    out = {
        "hcs": {},
        "error_categories": {},
        "narrative": {},
    }
    for p in sorted(Path("results/day6").glob("hcs_*.json")):
        out["hcs"][p.stem] = json.loads(p.read_text())

    cats = json.loads(Path("results/day6/error_categories.json").read_text())
    # Group by (model, constraint) keeping only arith+func dialect.
    grouped: dict[str, dict[str, dict[str, int]]] = {}
    for key, counts in cats.items():
        m, c, d = key.split("|")
        if d != "arith+func":
            continue
        grouped.setdefault(m, {})[c] = counts
    out["error_categories"] = grouped

    # Compact narrative
    sm = grouped.get("smollm2-1.7b", {})
    phi = grouped.get("phi-3.5-mini", {})

    def _get(cell: dict, k: str) -> int:
        return int(cell.get(k, 0))

    # C3's hero claim: type_ssa collapse
    if "c1_c2" in sm and "c1_c2_c3" in sm:
        sm_type_ssa_before = _get(sm["c1_c2"], "type_ssa")
        sm_type_ssa_after = _get(sm["c1_c2_c3"], "type_ssa")
        out["narrative"]["c3_type_ssa_collapse"] = {
            "before_c1_c2": sm_type_ssa_before,
            "after_c1_c2_c3": sm_type_ssa_after,
            "absolute_reduction": sm_type_ssa_before - sm_type_ssa_after,
            "pct_reduction": (
                100.0 * (sm_type_ssa_before - sm_type_ssa_after) / max(sm_type_ssa_before, 1)
            ),
        }

    # C2's within-op type claim
    if "c1" in sm and "c1_c2" in sm:
        sm_type_before = _get(sm["c1"], "type")
        sm_type_after = _get(sm["c1_c2"], "type")
        out["narrative"]["c2_type_collapse"] = {
            "before_c1": sm_type_before,
            "after_c1_c2": sm_type_after,
            "absolute_reduction": sm_type_before - sm_type_after,
        }

    Path("results/day6/day6_summary.json").write_text(json.dumps(out, indent=2))
    print(f"[day6] summary → results/day6/day6_summary.json")

    # Print narrative
    n = out["narrative"]
    if "c3_type_ssa_collapse" in n:
        d = n["c3_type_ssa_collapse"]
        print(
            f"  C3 type_ssa collapse (SmolLM2): {d['before_c1_c2']} → "
            f"{d['after_c1_c2_c3']} ({d['absolute_reduction']:+d}, "
            f"{d['pct_reduction']:.1f}% reduction)"
        )
    if "c2_type_collapse" in n:
        d = n["c2_type_collapse"]
        print(
            f"  C2 type collapse (SmolLM2):     {d['before_c1']} → "
            f"{d['after_c1_c2']} ({d['absolute_reduction']:+d})"
        )


if __name__ == "__main__":
    main()
