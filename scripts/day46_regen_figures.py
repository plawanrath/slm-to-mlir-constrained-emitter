"""Day-46: regenerate figures with Day-39-PM full-n multi-seed data.

Inputs:
  - `results/day39pm/multiseed_fulln.jsonl` — new full-n (n=200/125),
    seeds 1 and 2.
  - Existing seed-0 point estimates from `results/frozen/...`.

Outputs (regenerated):
  - `docs/paper/figures/fig9_apples_to_apples.pdf` — with multi-seed
    whiskers showing the range across seeds 0/1/2.
  - `docs/paper/figures/table_apples_to_apples.tex` — with both
    seed-0 and multi-seed-mean columns.

This is invoked AFTER Day-39 PM completes. Before that it will run on
whatever partial data is present (no-op if Day-39 PM hasn't yet written
any rows).
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = "/Users/plawanrath/Documents/GitHub-public/slm-to-mlir-constrained-emitter"
os.chdir(REPO); sys.path.insert(0, REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGS = Path("docs/paper/figures")
MULTISEED = Path("results/day39pm/multiseed_fulln.jsonl")
DAY50 = Path("results/day50/baseline_linalg_multiseed.jsonl")
FALLBACK = Path("results/day34/multiseed.jsonl")


def _load_multiseed(path: Path) -> dict:
    """Group rows by (model, dialect, seed) and compute verify rate per cell."""
    cells: dict[tuple[str, str, int], list[bool]] = defaultdict(list)
    if not path.exists():
        return {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            k = (r["model"], r["dialect"], r["seed"])
            cells[k].append(bool(r.get("verify_valid", False)))
    out: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for (model, dialect, seed), flags in cells.items():
        if flags:
            out[(model, dialect)][seed] = sum(flags) / len(flags)
    return out


def _regen_apples_to_apples() -> None:
    # Day-39-PM is the primary source for SmolLM2-all-cells and baseline-arith+func.
    # Day-50 fills baseline-linalg seeds 1, 2. Merge both.
    source = MULTISEED if MULTISEED.exists() and MULTISEED.stat().st_size > 0 else FALLBACK
    print(f"[day46] primary multi-seed source: {source}", file=sys.stderr)
    cells = _load_multiseed(source)
    if DAY50.exists() and DAY50.stat().st_size > 0:
        print(f"[day46] merging Day-50 baseline-linalg from: {DAY50}", file=sys.stderr)
        d50 = _load_multiseed(DAY50)
        for key, seeds in d50.items():
            cells[key].update(seeds)
    # Add seed-0 frozen values at full n (Day-5/Day-9 for SmolLM2,
    # Day-18 for CodeLlama/Granite, Day-21 for StarCoder2) so every
    # cell has a 3-seed range at the same n as seeds 1 and 2.
    # Previous values used Day-34 first-100 subset; correcting to full-n.
    seed0 = {
        ("smollm2-c1c2c3",  "arith+func"): 0.675,  # Day-5, n=200
        ("smollm2-c1c2c3",  "linalg"):     0.728,  # Day-9, n=125
        ("codellama-c1c3",  "arith+func"): 0.605,  # Day-18, n=200
        ("codellama-c1c3",  "linalg"):     0.592,  # Day-18, n=125
        ("granite-c1c3",    "arith+func"): 0.49,   # Day-18, n=200
        ("granite-c1c3",    "linalg"):     0.344,  # Day-18, n=125
        ("starcoder2-c1c3", "arith+func"): 0.68,   # Day-21, n=200
        ("starcoder2-c1c3", "linalg"):     0.536,  # Day-21, n=125
    }
    for key, v in seed0.items():
        if key in cells and 0 not in cells[key]:
            cells[key][0] = v
    if not cells:
        print("[day46] no multi-seed data found — skipping regen", file=sys.stderr)
        return

    # Group by model for the figure
    # X-axis: cells (SmolLM2, CodeLlama, Granite, StarCoder2) × (arith, linalg)
    model_order = ["smollm2-c1c2c3", "codellama-c1c3", "granite-c1c3", "starcoder2-c1c3"]
    labels = {
        "smollm2-c1c2c3": "SmolLM2\n1.7B+C1+C2+C3",
        "codellama-c1c3": "CodeLlama-34B\n+C1+C3",
        "granite-c1c3":    "Granite-34B\n+C1+C3",
        "starcoder2-c1c3": "StarCoder2-15B\n+C1+C3",
    }
    dialects = ["arith+func", "linalg"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(model_order))
    bar_w = 0.35
    for di, dialect in enumerate(dialects):
        means = []
        lows = []
        highs = []
        for m in model_order:
            seeds = cells.get((m, dialect), {})
            vals = list(seeds.values())
            if not vals:
                means.append(0.0); lows.append(0.0); highs.append(0.0)
            else:
                means.append(float(np.mean(vals)))
                lows.append(float(np.min(vals)))
                highs.append(float(np.max(vals)))
        err_lo = [m - lo for m, lo in zip(means, lows)]
        err_hi = [hi - m for m, hi in zip(means, highs)]
        ax.bar(x + (di - 0.5) * bar_w, means, bar_w,
               yerr=[err_lo, err_hi], capsize=4,
               label=dialect,
               color=("steelblue" if dialect == "arith+func" else "seagreen"),
               edgecolor="crimson" if False else "black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in model_order], fontsize=9)
    ax.set_ylabel("mlir-opt verify-valid rate")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.legend(title="Dialect", loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Apples-to-apples: three-seed mean with across-seed range\n"
                  "(SmolLM2 + C1+C2+C3 vs 15B/34B + C1+C3; "
                  "n=200 arith+func, n=125 linalg)")
    plt.tight_layout()
    out = FIGS / "fig9_apples_to_apples.pdf"
    plt.savefig(out)
    plt.close()
    print(f"[day46] wrote {out}", file=sys.stderr)

    # Also write a fresh table_apples_to_apples.tex
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"System & arith+func (mean, range) & linalg (mean, range) \\",
        r"\midrule",
    ]
    for m in model_order:
        name = labels[m].replace("\n", " ")
        parts = []
        for dialect in dialects:
            seeds = cells.get((m, dialect), {})
            vals = list(seeds.values())
            if not vals:
                parts.append("—")
            else:
                mean = np.mean(vals)
                hr = (max(vals) - min(vals)) / 2
                parts.append(f"{mean*100:.1f}\\% ($\\pm${hr*100:.1f}pp)")
        lines.append(f"{name} & {parts[0]} & {parts[1]} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    tbl = FIGS / "table_apples_to_apples.tex"
    tbl.write_text("\n".join(lines))
    print(f"[day46] wrote {tbl}", file=sys.stderr)


if __name__ == "__main__":
    _regen_apples_to_apples()
