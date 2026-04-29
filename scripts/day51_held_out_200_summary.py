"""Day-51 / Item 2 aggregator: Held-Out-200 baseline summary + CI vs SmolLM2.

Reads:
  - results/day51/stablehlo_held_out_200_baselines.jsonl  (3 baselines, seed=1)
  - results/day_f8/heldout_200_matrix.jsonl               (SmolLM2 c1_c3, no seed)

Computes per-baseline verify-valid + paired-bootstrap CI vs SmolLM2 c1_c3.

Output: results/day51/stablehlo_held_out_200_summary.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from eval.stats import bootstrap_ci, paired_bootstrap_diff


def main():
    base_path = REPO / "results/day51/stablehlo_held_out_200_baselines.jsonl"
    smol_path = REPO / "results/day_f8/heldout_200_matrix.jsonl"

    baselines = [json.loads(l) for l in base_path.read_text().splitlines() if l.strip()]
    smol_all = [json.loads(l) for l in smol_path.read_text().splitlines() if l.strip()]

    # SmolLM2 c1_c3 cell, paired by prompt_id.
    smol_c1c3 = [r for r in smol_all if r.get("constraint") == "c1_c3"]
    smol_by_pid = {r["prompt_id"]: int(bool(r.get("verify_valid"))) for r in smol_c1c3}

    summary = {"smollm2-c1c3": {}, "baselines": {}}

    smol_arr = [smol_by_pid[i] for i in sorted(smol_by_pid.keys())]
    s_ci = bootstrap_ci(smol_arr).as_dict()
    summary["smollm2-c1c3"] = {
        "n": len(smol_arr),
        "verify_rate": s_ci["point"],
        "ci_low": s_ci["ci_low"],
        "ci_high": s_ci["ci_high"],
    }

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in baselines:
        by_model[r["model"]].append(r)

    for model, rows in by_model.items():
        rows = sorted(rows, key=lambda r: r["prompt_id"])
        # Align with SmolLM2 prompt_ids that are present in both.
        common = sorted(set(r["prompt_id"] for r in rows) & set(smol_by_pid))
        a_arr = []
        b_arr = []
        for pid in common:
            row = next(r for r in rows if r["prompt_id"] == pid)
            b_arr.append(int(bool(row.get("verify_valid"))))
            a_arr.append(smol_by_pid[pid])
        b_ci = bootstrap_ci(b_arr).as_dict()
        diff = paired_bootstrap_diff(a_arr, b_arr).as_dict()
        summary["baselines"][model] = {
            "n": len(b_arr),
            "verify_rate": b_ci["point"],
            "ci_low": b_ci["ci_low"],
            "ci_high": b_ci["ci_high"],
            "paired_vs_smollm2": {
                "n_paired": len(common),
                "delta_smollm2_minus_baseline": diff["point"],
                "delta_ci_low": diff["ci_low"],
                "delta_ci_high": diff["ci_high"],
                "p_value_smollm2_greater": diff["p_value"],
            },
        }

    out_path = REPO / "results/day51/stablehlo_held_out_200_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print("\n=== Held-Out-200 baselines vs SmolLM2 ===", file=sys.stderr)
    s = summary["smollm2-c1c3"]
    print(f"  smollm2-c1c3: {s['verify_rate']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}] (n={s['n']})",
          file=sys.stderr)
    for model, info in summary["baselines"].items():
        d = info["paired_vs_smollm2"]
        print(f"  {model:25s}: {info['verify_rate']:.3f} [{info['ci_low']:.3f}, {info['ci_high']:.3f}]"
              f" | Δ(smol−this)={d['delta_smollm2_minus_baseline']:+.3f} "
              f"[{d['delta_ci_low']:+.3f},{d['delta_ci_high']:+.3f}] p={d['p_value_smollm2_greater']:.4f}",
              file=sys.stderr)
    print(f"\n[day51-ho200] → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
