"""Phase F F7: single-figure headline (4 systems × 2 stable dialects).

Drops arith+func (non-win cell per F2) and renders a clean bar chart
of verify-valid rates across the four systems on linalg and
StableHLO-Spec-30 (post-Phase-F fix).

Output: docs/paper/figures/fig1_headline.pdf
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
os.chdir(REPO); sys.path.insert(0, REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGS = Path("docs/paper/figures")

# Post-Day-50 numbers. All four systems now have 3-seed data on linalg.
# Linalg: SmolLM2 {s0: 0.73, s1: 0.80, s2: 0.80} → mean 0.777
# Linalg: CodeLlama {s0: 0.592, s1: 0.608, s2: 0.560} → mean 0.587
# Linalg: Granite {s0: 0.344, s1: 0.408, s2: 0.320} → mean 0.357
# Linalg: StarCoder2 {s0: 0.536, s1: 0.552, s2: 0.560} → mean 0.549
# StableHLO: seed-0 only (Spec-30 single-run, post-methodology-fix)
DATA = {
    "SmolLM2-1.7B\n+C1+C2+C3": {
        "linalg": {"mean": 0.777, "seeds": [0.73, 0.80, 0.80]},
        "stablehlo": {"mean": 0.633, "seeds": [0.633]},
    },
    "CodeLlama-34B\n+C1+C3": {
        "linalg": {"mean": 0.587, "seeds": [0.592, 0.608, 0.560]},
        "stablehlo": {"mean": 0.367, "seeds": [0.367]},
    },
    "Granite-34B\n+C1+C3": {
        "linalg": {"mean": 0.357, "seeds": [0.344, 0.408, 0.320]},
        "stablehlo": {"mean": 0.333, "seeds": [0.333]},
    },
    "StarCoder2-15B\n+C1+C3": {
        "linalg": {"mean": 0.549, "seeds": [0.536, 0.552, 0.560]},
        "stablehlo": {"mean": 0.600, "seeds": [0.600]},
    },
}


def main():
    systems = list(DATA.keys())
    dialects = ["linalg", "stablehlo"]
    dialect_labels = {"linalg": "linalg ($n{=}125$)", "stablehlo": "StableHLO ($n{=}30$)"}

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(systems))
    w = 0.38
    colors = {"linalg": "#2E7D32", "stablehlo": "#1565C0"}

    for di, dialect in enumerate(dialects):
        means, errs_lo, errs_hi = [], [], []
        for sys_ in systems:
            d = DATA[sys_][dialect]
            m = d["mean"]; seeds = d["seeds"]
            means.append(m)
            errs_lo.append(m - min(seeds))
            errs_hi.append(max(seeds) - m)
        bars = ax.bar(
            x + (di - 0.5) * w, means, w,
            yerr=[errs_lo, errs_hi], capsize=4,
            label=dialect_labels[dialect],
            color=colors[dialect],
            edgecolor=("crimson" if False else "black"),
            linewidth=0.8,
        )
        # highlight ours (leftmost system)
        bars[0].set_edgecolor("crimson")
        bars[0].set_linewidth(2.0)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=9)
    ax.set_ylabel("verify-valid rate (structural)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_title("Headline: verify-valid rate, stable dialects only\n"
                  "(arith+func is a non-win cell, reported in Limitations)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGS / "fig1_headline.pdf"
    plt.savefig(out)
    plt.close()
    print(f"[f7] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
