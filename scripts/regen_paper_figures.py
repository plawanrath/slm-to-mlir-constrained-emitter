"""Regenerate paper figures from current uniform-n / three-seed sources.

Outputs (in docs/paper/figures/):
  - fig1_headline.pdf            — headline win matrix on stable dialects
  - fig4_efficiency_frontier.pdf — wall-clock vs verify%
  - fig9_apples_to_apples.pdf    — apples 5 systems x 2 dialects
  - fig7_per_op_linalg.pdf       — per-op linalg verify rate (seed-0 uniform-n)

Inputs:
  - results/day51_seed0_n200/apples_to_apples.json
  - results/day51_seed0_n200/multiseed_seed0.jsonl
  - results/day51/stablehlo_held_out_200_summary.json
  - eval/benchmarks/linalg_spec_30/examples/*.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
os.chdir(REPO); sys.path.insert(0, REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("docs/paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

C_SLM = "#1f77b4"
C_GRN = "#a28bd4"
C_CLL = "#d62728"
C_STC = "#2ca02c"
C_FP16 = "#ff7f0e"
C_POS = "#4C78A8"
C_TYPE_SSA = "#E45756"


def _load_apples():
    return json.loads(Path("results/day51_seed0_n200/apples_to_apples.json").read_text())


def _stablehlo_summary():
    return json.loads(Path("results/day51/stablehlo_held_out_200_summary.json").read_text())


# Spec-30 single-run StableHLO numbers post-methodology-fix
# These come from the Phase-F headline regen and are corpus-conditional
# (Spec-30 only; Held-Out-200 saturates baselines per
# results/day51/stablehlo_held_out_200_summary.json).
STABLEHLO_SPEC30 = {
    "smollm2-c1c2c3":  0.633,
    "codellama-c1c3":  0.367,
    "granite-c1c3":    0.333,
    "starcoder2-c1c3": 0.600,
}


def fig_headline():
    """Headline figure: linalg + StableHLO across 4 systems.

    StableHLO panel is corpus-conditional (Spec-30 only); Held-Out-200
    saturates baselines and is reported in a separate table.
    """
    apples = _load_apples()["cells"]

    systems = [
        ("smollm2-c1c2c3",  "SmolLM2-1.7B\n+C1+C2+C3"),
        ("codellama-c1c3",  "CodeLlama-34B\n+C1+C3"),
        ("granite-c1c3",    "Granite-34B\n+C1+C3"),
        ("starcoder2-c1c3", "StarCoder2-15B\n+C1+C3"),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(systems))
    w = 0.38
    colors = {"linalg": "#2E7D32", "stablehlo": "#1565C0"}

    # Linalg bars (3-seed mean ± across-seed range from apples_to_apples.json)
    means_l, lo_l, hi_l = [], [], []
    for key, _ in systems:
        cell = apples[f"{key}::linalg"]
        seeds = [cell["seeds"][s]["verify_rate"] for s in ("0", "1", "2") if cell["seeds"].get(s)]
        m = float(np.mean(seeds))
        means_l.append(m)
        lo_l.append(max(0.0, m - min(seeds)))
        hi_l.append(max(0.0, max(seeds) - m))

    # StableHLO Spec-30 single-run (corpus-conditional)
    means_s = [STABLEHLO_SPEC30[k] for k, _ in systems]

    bars_l = ax.bar(x - w/2, means_l, w, yerr=[lo_l, hi_l], capsize=4,
                    label="linalg ($n{=}125$, three-seed mean)",
                    color=colors["linalg"], edgecolor="black", linewidth=0.8)
    bars_s = ax.bar(x + w/2, means_s, w,
                    label="StableHLO-Spec-30 ($n{=}30$, seed-0)",
                    color=colors["stablehlo"], edgecolor="black", linewidth=0.8)
    bars_l[0].set_edgecolor("crimson"); bars_l[0].set_linewidth(2.0)
    bars_s[0].set_edgecolor("crimson"); bars_s[0].set_linewidth(2.0)

    for xi, v in zip(x - w/2, means_l):
        ax.text(xi, v + 0.015, f"{v*100:.1f}%", ha="center", fontsize=8)
    for xi, v in zip(x + w/2, means_s):
        ax.text(xi, v + 0.015, f"{v*100:.1f}%", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in systems], fontsize=9)
    ax.set_ylabel("verify-valid rate (structural)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_title("Headline: verify-valid rate, stable dialects only\n"
                  "(StableHLO panel is corpus-conditional: Spec-30; "
                  "Held-Out-200 saturates baselines)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = OUT / "fig1_headline.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[regen] wrote {out}", file=sys.stderr)


def fig_efficiency():
    """Wall-clock vs verify% efficiency frontier."""
    apples = _load_apples()["cells"]

    # Wall-clock seconds per generation (median, prior runs)
    walls = {
        "smollm2-c1c2c3":  1.86,
        "smollm2-arith":   1.65,
        "codellama-c1c3":  16.0,
        "granite-c1c3":    40.0,
        "starcoder2-c1c3": 12.0,
    }

    # 3-seed mean verify rates from apples_to_apples.json
    def mean(key, dialect):
        cell = apples[f"{key}::{dialect}"]
        return float(np.mean([cell["seeds"][s]["verify_rate"]
                              for s in ("0", "1", "2") if cell["seeds"].get(s)]))

    points = [
        ("SmolLM2 +C1+C2+C3 (linalg)",     1.86, mean("smollm2-c1c2c3",  "linalg"),     C_SLM, "s"),
        ("SmolLM2 +C1+C2+C3 (arith+func)", 1.65, mean("smollm2-c1c2c3",  "arith+func"), C_SLM, "o"),
        ("CodeLlama-34B +C1+C3 (linalg)",  16.0, mean("codellama-c1c3",  "linalg"),     C_CLL, "s"),
        ("CodeLlama-34B +C1+C3 (arith+func)", 16.0, mean("codellama-c1c3", "arith+func"), C_CLL, "o"),
        ("Granite-34B +C1+C3 (linalg)",    40.0, mean("granite-c1c3",    "linalg"),     C_GRN, "s"),
        ("Granite-34B +C1+C3 (arith+func)", 40.0, mean("granite-c1c3",    "arith+func"), C_GRN, "o"),
        ("StarCoder2-15B +C1+C3 (linalg)", 12.0, mean("starcoder2-c1c3", "linalg"),     C_STC, "s"),
        ("StarCoder2-15B +C1+C3 (arith+func)", 12.0, mean("starcoder2-c1c3", "arith+func"), C_STC, "o"),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for lbl, t, v, color, m in points:
        ax.scatter(t, v * 100, s=140, color=color, marker=m, edgecolor="black",
                   linewidth=0.6, zorder=3)
        # Annotation offsets to avoid overlap
        dx = 1.05; dy = 1.5
        if "linalg" in lbl: dy = 2.0
        if "arith+func" in lbl: dy = -3.5
        if "Granite" in lbl and "linalg" in lbl: dy = 1.2
        ax.annotate(lbl, xy=(t, v * 100), xytext=(t * dx, v * 100 + dy),
                    fontsize=7.2, color=color)

    # Pareto-best line for SmolLM2 linalg
    slm_lin = mean("smollm2-c1c2c3", "linalg") * 100
    ax.axhline(y=slm_lin, linestyle="--", color="crimson", alpha=0.5, lw=0.9)
    ax.text(2.1, slm_lin + 1.5, f"SmolLM2 +C1+C2+C3 linalg ({slm_lin:.1f}%)",
            fontsize=7.5, color="crimson")

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="arith+func",
               markerfacecolor="gray", markersize=10, markeredgecolor="black"),
        Line2D([0], [0], marker="s", color="w", label="linalg",
               markerfacecolor="gray", markersize=10, markeredgecolor="black"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.95, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("wall-clock per generation (s, log scale)")
    ax.set_ylabel("verify-valid % (three-seed mean)")
    ax.set_xlim(1.2, 70)
    ax.set_ylim(20, 95)
    ax.grid(alpha=0.25, which="both")
    ax.set_title("Efficiency frontier — three-seed mean verify rate at uniform $n$\n"
                  "(top-left is best: low latency, high verify rate)")
    fig.tight_layout()
    out = OUT / "fig4_efficiency_frontier.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[regen] wrote {out}", file=sys.stderr)


def fig_apples():
    """5 systems x 2 dialects, three-seed mean ± half-range."""
    apples = _load_apples()["cells"]
    systems = [
        ("smollm2-c1c2c3",       "SmolLM2-1.7B\n+C1+C2+C3"),
        ("granite-8b-fp16-c1c3", "Granite-Code-8B\nfp16 +C1+C3"),
        ("starcoder2-c1c3",      "StarCoder2-15B\n+C1+C3"),
        ("codellama-c1c3",       "CodeLlama-34B\n+C1+C3"),
        ("granite-c1c3",         "Granite-34B\n+C1+C3"),
    ]
    dialects = ["arith+func", "linalg"]
    colors = {"arith+func": "steelblue", "linalg": "seagreen"}
    n_per = {"arith+func": 200, "linalg": 125}

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(systems))
    w = 0.38

    for di, dialect in enumerate(dialects):
        means, errs_lo, errs_hi = [], [], []
        for key, _ in systems:
            cell = apples[f"{key}::{dialect}"]
            seeds = [cell["seeds"][s]["verify_rate"]
                     for s in ("0", "1", "2") if cell["seeds"].get(s)]
            m = float(np.mean(seeds)); hr = (max(seeds) - min(seeds)) / 2.0
            means.append(m); errs_lo.append(hr); errs_hi.append(hr)
        bars = ax.bar(x + (di - 0.5) * w, means, w,
                      yerr=[errs_lo, errs_hi], capsize=4,
                      label=f"{dialect} ($n{{=}}{n_per[dialect]}$ per seed)",
                      color=colors[dialect],
                      edgecolor="black", linewidth=0.9)
        bars[0].set_edgecolor("crimson"); bars[0].set_linewidth(2.0)
        for xi, v, hr in zip(x + (di - 0.5) * w, means, errs_lo):
            ax.text(xi, v + hr + 0.015, f"{v*100:.1f}%", ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in systems], fontsize=9)
    ax.set_ylabel("mlir-opt verify-valid rate")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.legend(title="Dialect", loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Apples-to-apples: three-seed mean at uniform $n$, "
                  "error bars are across-seed half-range\n"
                  "(includes Granite-Code-8B-fp16 precision control)")
    fig.tight_layout()
    out = OUT / "fig9_apples_to_apples.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[regen] wrote {out}", file=sys.stderr)


def fig_per_op_linalg():
    """Per-op verify% for SmolLM2 + C1+C2+C3 on linalg, seed-0 uniform-n."""
    rows = [json.loads(l)
            for l in Path("results/day51_seed0_n200/multiseed_seed0.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("model") == "smollm2-c1c2c3"
            and r.get("dialect") == "linalg" and int(r.get("seed", 0)) == 0]

    seed_ops: dict[int, str] = {}
    for i, p in enumerate(sorted(Path("eval/benchmarks/linalg_spec_30/examples").glob("*.json"))):
        d = json.loads(p.read_text())
        m = re.search(r"linalg\.(\w+)", d.get("mlir", ""))
        if m:
            seed_ops[i] = m.group(1)

    op_stats: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        pid = int(r["prompt_id"])
        op = seed_ops.get(pid)
        if op is None:
            m = re.search(r"linalg\.(\w+)", r.get("generated", ""))
            if m:
                op = m.group(1)
        if op is None:
            continue
        if op not in {"matmul", "matvec", "fill", "copy", "transpose", "broadcast",
                      "add", "sub", "mul", "div", "exp", "abs"}:
            continue
        op_stats[op].append(int(bool(r.get("verify_valid"))))

    ordered = sorted(op_stats.items(),
                     key=lambda kv: (-np.mean(kv[1]) if kv[1] else 0, kv[0]))
    ops = [op for op, _ in ordered]
    rates = [100 * np.mean(v) for _, v in ordered]
    ns = [len(v) for _, v in ordered]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.bar(ops, rates, color=C_TYPE_SSA, edgecolor="black", linewidth=0.5)
    for b, v, n in zip(bars, rates, ns):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%\nn={n}",
                ha="center", fontsize=7)
    ax.set_ylabel("verify-valid %")
    ax.set_ylim(0, 110)
    ax.set_title("Per-op verify rate — SmolLM2 + C1+C2+C3 on linalg "
                  "(seed-0, uniform $n{=}125$)")
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    out = OUT / "fig7_per_op_linalg.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[regen] wrote {out}", file=sys.stderr)


def main():
    fig_headline()
    fig_efficiency()
    fig_apples()
    fig_per_op_linalg()


if __name__ == "__main__":
    main()
