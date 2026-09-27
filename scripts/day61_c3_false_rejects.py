"""Day-61: C3 scope-validator confusion against mlir-opt on the frozen released pool.

Context (camera-ready, NeurIPS 2026 E&D). The submission's soundness appendix
reports that c3_scope has zero false rejects against mlir-opt on SmolLM2's
output distribution (Day-4/5 confusion matrix, pre-freeze prompts) but not on
CodeLlama's, quantified as a -22.2pp C1 -> C1+C3 drop. That drop compared a
pre-freeze draft-pool C1-only cell (82.0) with a frozen-pool C1+C3 cell (59.8).
On the frozen pool the CodeLlama C1-only cell is 60.5
(scripts/day61_frozen_c1only_baselines.py), so the drop no longer measures the
false-reject rate. This script measures that rate directly, with no inference
and no new verification: for every parse-valid output in a frozen-pool cell it
applies c3_scope and cross-tabulates against the recorded verify outcome.

  false reject  = verify-valid (mlir-opt accepts) but c3_scope rejects
  false accept  = c3_scope accepts but mlir-opt rejects (C3 is not a verifier;
                  reported for completeness)

Cells: SmolLM2 C1 and C1+C2 ladder rungs (Day-53 arith, Day-59 linalg; the
generations C3 filters in the full stack) and every baseline C1-only cell
(Day-61 arith CodeLlama/Granite, Day-19 StarCoder2, Day-10 linalg
CodeLlama/Granite).

Output: results/day61/c3_false_rejects.json (confusion per cell, plus every
false-reject program with its violation kinds, for root-cause inspection), and
results/day61/paper_assets/fig1_scope_validator_confusion.pdf (the paper's
Figure 1, now on the frozen SmolLM2 C1 rung, arith+func n=200).
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from decoder.c3_scope import accept_or_reject, summarize_violations  # noqa: E402

CELLS = [
    ("smollm2 C1", "arith+func", "results/day53/e10_ladder_frozen.jsonl", lambda r: r["constraint"] == "c1"),
    ("smollm2 C1+C2", "arith+func", "results/day53/e10_ladder_frozen.jsonl", lambda r: r["constraint"] == "c1_c2"),
    ("smollm2 C1", "linalg", "results/day59/e10b_ladder_linalg_frozen.jsonl", lambda r: r["constraint"] == "c1"),
    ("smollm2 C1+C2", "linalg", "results/day59/e10b_ladder_linalg_frozen.jsonl", lambda r: r["constraint"] == "c1_c2"),
    ("codellama C1-only", "arith+func", "results/day61/frozen_c1only_arith.jsonl", lambda r: r["model"] == "codellama-c1"),
    ("granite C1-only", "arith+func", "results/day61/frozen_c1only_arith.jsonl", lambda r: r["model"] == "granite-c1"),
    ("starcoder2 C1-only", "arith+func", "results/day19/starcoder2_baselines.jsonl",
     lambda r: r["model"] == "starcoder2-c1" and r["dialect"] == "arith+func"),
    ("codellama C1-only", "linalg", "results/day10/linalg_baselines.jsonl", lambda r: r["model"] == "codellama-c1"),
    ("granite C1-only", "linalg", "results/day10/linalg_baselines.jsonl", lambda r: r["model"] == "granite-c1"),
    ("starcoder2 C1-only", "linalg", "results/day19/starcoder2_baselines.jsonl",
     lambda r: r["model"] == "starcoder2-c1" and r["dialect"] == "linalg"),
]


def main() -> int:
    out = {"cells": {}, "false_reject_programs": []}
    print(f"{'cell':22s} {'dialect':10s} {'n':>4s} {'parse':>5s} {'verify':>6s} {'C3 rej':>6s} "
          f"{'false rej':>9s} {'FR rate':>8s} {'false acc':>9s}")
    for name, dialect, path, keep in CELLS:
        if not Path(path).exists():
            print(f"{name:22s} {dialect:10s} (missing {path})")
            continue
        rows = [r for r in map(json.loads, open(path)) if keep(r)]
        conf = Counter()
        for r in rows:
            if not r.get("parse_valid"):
                continue
            ok, rep = accept_or_reject(r["generated"])
            vv = bool(r["verify_valid"])
            conf[("scope_pass" if ok else "scope_fail", "verify_pass" if vv else "verify_fail")] += 1
            if vv and not ok:
                out["false_reject_programs"].append({
                    "cell": name, "dialect": dialect, "prompt_id": r["prompt_id"],
                    "violations": summarize_violations(rep.violations), "generated": r["generated"]})
        verify_pass = conf[("scope_pass", "verify_pass")] + conf[("scope_fail", "verify_pass")]
        fr = conf[("scope_fail", "verify_pass")]
        cell = {"n": len(rows), "parse_valid": sum(conf.values()), "verify_valid": verify_pass,
                "scope_reject": conf[("scope_fail", "verify_pass")] + conf[("scope_fail", "verify_fail")],
                "false_reject": fr, "false_reject_rate_among_verify_valid": round(fr / verify_pass, 3) if verify_pass else None,
                "false_accept": conf[("scope_pass", "verify_fail")],
                "confusion": {f"{a}|{b}": v for (a, b), v in conf.items()}}
        out["cells"][f"{name}|{dialect}"] = cell
        print(f"{name:22s} {dialect:10s} {cell['n']:4d} {cell['parse_valid']:5d} {verify_pass:6d} "
              f"{cell['scope_reject']:6d} {fr:9d} {100 * (cell['false_reject_rate_among_verify_valid'] or 0):7.1f}% "
              f"{cell['false_accept']:9d}")
    kinds = Counter()
    for p in out["false_reject_programs"]:
        kinds.update(p["violations"])
    out["false_reject_violation_kinds"] = dict(kinds)
    print(f"violation kinds among false rejects: {dict(kinds)}")
    Path("results/day61").mkdir(parents=True, exist_ok=True)
    Path("results/day61/c3_false_rejects.json").write_text(json.dumps(out, indent=2))
    confusion_figure(out["cells"]["smollm2 C1|arith+func"],
                     Path("results/day61/paper_assets/fig1_scope_validator_confusion.pdf"))
    return 0


def confusion_figure(cell: dict, path: Path) -> None:
    """2x2 C3 scope decision vs mlir-opt verdict; the false-reject cell is scope FAIL / verify PASS."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})
    c = cell["confusion"]
    cm = np.array([[c.get("scope_pass|verify_pass", 0), c.get("scope_pass|verify_fail", 0)],
                   [c.get("scope_fail|verify_pass", 0), c.get("scope_fail|verify_fail", 0)]])
    total = int(cm.sum())
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            ax.text(j, i, f"{val}\n({100 * val / total:.1f}%)", ha="center", va="center",
                    color="white" if val > cm.max() * 0.5 else "black", fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["verify PASS", "verify FAIL"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["scope PASS", "scope FAIL"])
    ax.set_xlabel("mlir-opt --verify-diagnostics")
    ax.set_ylabel("C3 scope validator")
    fr, vv = cell["false_reject"], cell["verify_valid"]
    ax.set_title(f"C3 scope validator vs mlir-opt (SmolLM2 + C1, arith+func, n={total}, frozen pool)\n"
                 f"false rejects (scope FAIL, verify PASS): {fr} of {vv} verify-valid")
    ax.add_patch(Rectangle((-0.5, 0.5), 1, 1, fill=False, edgecolor="crimson", lw=2))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
