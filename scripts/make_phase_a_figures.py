"""Day-21: Phase-A integration figures.

Produces:
  - fig9_apples_to_apples.pdf — bar chart comparing SmolLM2+C1+C2+C3
    against the four 30B+C1+C3 cells + StarCoder2+C1 cells on both
    dialects.
  - table_apples_to_apples.tex — matching LaTeX table (already in main.tex
    inline; this regenerates an external form for reference).
  - Fig1 re-render with StarCoder2 + Day-18 cells added.

Data sources:
  - results/frozen/week2_gate_day5/c3_summary.json  (SmolLM2 C1+C2+C3 arith)
  - results/day9/linalg_summary.json                 (SmolLM2 C1+C2+C3 linalg)
  - results/day18/summary.json                       (30B C1+C3 matrix)
  - results/day19/starcoder2_baselines.jsonl         (StarCoder2 free + C1)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = "/Users/plawanrath/Documents/GitHub-public/slm-to-mlir-constrained-emitter"
os.chdir(REPO); sys.path.insert(0, REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGS = Path("docs/paper/figures")
FIGS.mkdir(parents=True, exist_ok=True)


def _smollm2_arith() -> float:
    p = Path("results/frozen/week2_gate_day5/c3_summary.json")
    if not p.exists():
        p = Path("results/day5/c3_summary.json")
    return json.loads(p.read_text())["cells"]["c1_c2_c3"]["verify"]["point"]


def _smollm2_linalg() -> float:
    return json.loads(Path("results/day9/linalg_summary.json").read_text())[
        "cells"]["c1_c2_c3"]["verify"]["point"]


def _day18() -> dict[str, float]:
    s = json.loads(Path("results/day18/summary.json").read_text())
    out = {}
    for key, c in s["cells"].items():
        out[key] = c["verify"]["point"]
    return out


def _day19_cell(rows: list[dict], model_nick: str, dialect: str) -> float:
    cell = [r for r in rows
            if r["model"] == model_nick and r.get("dialect") == dialect]
    if not cell:
        return 0.0
    return sum(int(bool(r["verify_valid"])) for r in cell) / len(cell)


def _day19_all() -> dict[str, float]:
    rows = [json.loads(l) for l in Path("results/day19/starcoder2_baselines.jsonl").read_text().splitlines()]
    return {
        "starcoder2-c1/arith+func":  _day19_cell(rows, "starcoder2-c1",   "arith+func"),
        "starcoder2-c1/linalg":      _day19_cell(rows, "starcoder2-c1",   "linalg"),
        "starcoder2-free/arith+func": _day19_cell(rows, "starcoder2-free", "arith+func"),
        "starcoder2-free/linalg":    _day19_cell(rows, "starcoder2-free", "linalg"),
    }


def _day21b_all() -> dict[str, float]:
    p = Path("results/day21/starcoder2_c1c3.jsonl")
    if not p.exists():
        return {}
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    return {
        "starcoder2-c1c3/arith+func": _day19_cell(rows, "starcoder2-c1c3", "arith+func"),
        "starcoder2-c1c3/linalg":     _day19_cell(rows, "starcoder2-c1c3", "linalg"),
    }


def apples_to_apples_fig() -> None:
    """Grouped bar: 4 systems × 2 dialects. SLM+C3, 30B+C3 (2 variants), SC2+C3."""
    smol_a = _smollm2_arith()
    smol_l = _smollm2_linalg()
    d18 = _day18()
    d21 = _day21b_all()

    systems = [
        ("SmolLM2-1.7B\n+ C1+C2+C3", smol_a, smol_l, "#d73027", True),
        ("CodeLlama-34B\n+ C1+C3",   d18["codellama-c1c3/arith+func"],
                                     d18["codellama-c1c3/linalg"],    "#4575b4", False),
        ("Granite-34B\n+ C1+C3",     d18["granite-c1c3/arith+func"],
                                     d18["granite-c1c3/linalg"],      "#74add1", False),
        ("StarCoder2-15B\n+ C1+C3",  d21["starcoder2-c1c3/arith+func"],
                                     d21["starcoder2-c1c3/linalg"],   "#fdae61", False),
    ]

    labels = [s[0] for s in systems]
    arith_vals = [100*s[1] for s in systems]
    linalg_vals = [100*s[2] for s in systems]
    colors = [s[3] for s in systems]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(labels))
    w = 0.38
    b1 = ax.bar(x - w/2, arith_vals,  w, label="arith+func",
                color=colors, edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + w/2, linalg_vals, w, label="linalg",
                color=colors, edgecolor="black", linewidth=0.8, alpha=0.6)

    # Highlight our system with red border
    for i, (bars, val) in enumerate(zip([b1, b2], [arith_vals, linalg_vals])):
        bars[0].set_edgecolor("#d73027")
        bars[0].set_linewidth(2.2)

    for rect, v in zip(list(b1) + list(b2), arith_vals + linalg_vals):
        ax.text(rect.get_x() + rect.get_width()/2, v + 1,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("verify-valid (%)", fontsize=11)
    ax.set_ylim(0, 85)
    ax.set_title(
        "Apples-to-apples: SmolLM2-1.7B + C1+C2+C3 vs 30B + C1+C3 vs StarCoder2 + C1",
        fontsize=11,
    )
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    # Custom legend with dialect hatches rather than duplicate colors
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="grey", edgecolor="black", label="arith+func"),
        Patch(facecolor="grey", edgecolor="black", alpha=0.6, label="linalg"),
    ]
    ax.legend(handles=legend_items, loc="upper right", frameon=True)

    plt.tight_layout()
    out = FIGS / "fig9_apples_to_apples.pdf"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close()


def table_apples_to_apples() -> None:
    smol_a = _smollm2_arith()
    smol_l = _smollm2_linalg()
    d18 = _day18()
    d21 = _day21b_all()
    tex = r"""\begin{tabular}{lrr}
\toprule
System & arith+func & linalg \\
\midrule
""" + \
          f"Granite-Code-34B + C1+C3         & {100*d18['granite-c1c3/arith+func']:.1f}\\% & {100*d18['granite-c1c3/linalg']:.1f}\\% \\\\\n" + \
          f"CodeLlama-34B + C1+C3             & {100*d18['codellama-c1c3/arith+func']:.1f}\\% & {100*d18['codellama-c1c3/linalg']:.1f}\\% \\\\\n" + \
          f"StarCoder2-15B:instruct + C1+C3  & {100*d21['starcoder2-c1c3/arith+func']:.1f}\\% & {100*d21['starcoder2-c1c3/linalg']:.1f}\\% \\\\\n" + \
          f"\\textbf{{SmolLM2-1.7B + C1+C2+C3}} & \\textbf{{{100*smol_a:.1f}\\%}} & \\textbf{{{100*smol_l:.1f}\\%}} \\\\\n" + \
          r"""\bottomrule
\end{tabular}
"""
    out = FIGS / "table_apples_to_apples.tex"
    out.write_text(tex)
    print(f"wrote {out}")


if __name__ == "__main__":
    apples_to_apples_fig()
    table_apples_to_apples()
