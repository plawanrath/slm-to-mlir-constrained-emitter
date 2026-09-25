"""Final publication-quality figure generator for the paper.

Deterministic: re-running on the same inputs produces byte-identical PDFs.
Headless: uses matplotlib Agg backend; no display needed.

Outputs (see docs/paper/figures/FIGURES.md for per-figure descriptions):
  fig1_main_matrix.pdf          — hero: verify% per (system, dialect)
  fig2_constraint_progression.pdf — verify% vs constraint level
  fig3_error_collapse.pdf       — error-bucket stacked bars
  fig4_efficiency_frontier.pdf  — wall-clock vs verify%
  fig5_scope_validator_confusion.pdf — 2x2 confusion on Day-4 postmortem
  fig6_hcs_null.pdf             — HCS non-reproduction
  fig7_per_op_linalg.pdf        — per-op verify% under C1+C2+C3
  fig8_paired_bootstrap.pdf     — paired bootstrap deltas + CIs

LaTeX tables (unchanged; see scripts/day11_make_paper_figures.py).
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
from matplotlib.patches import Rectangle

from eval.stats import bootstrap_ci, paired_bootstrap_diff

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
    "pdf.fonttype": 42,  # TrueType (embeddable, editable text)
})

# Consistent palette
C_SLM = "#1f77b4"   # blue — SmolLM2
C_PHI = "#8ecae6"   # light blue — Phi (secondary SLM)
C_GRN = "#a28bd4"   # purple — Granite
C_CLL = "#d62728"   # red — CodeLlama
C_POS = "#4C78A8"; C_TYPE = "#E45756"; C_TYPE_SSA = "#F58518"
C_SYN = "#72B7B2"; C_OTHER = "#B279A2"


# =============== data loaders ===============

def _load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines()]


def _load_summary():
    return json.loads(Path("results/frozen/day10_final_matrix/linalg_matrix_summary.json").read_text())


def _err_cats():
    a = json.loads(Path("results/day6/error_categories.json").read_text())
    l = json.loads(Path("results/day10/linalg_error_categories.json").read_text())
    return a, l


# =============== FIG 1: main matrix ===============

def fig1_main_matrix():
    summary = _load_summary()
    systems = [
        ("smollm2-none",     "SmolLM2\nfree"),
        ("smollm2-c1",       "SmolLM2\n+C1"),
        ("smollm2-c1_c2",    "SmolLM2\n+C1+C2"),
        ("smollm2-c1_c2_c3", "SmolLM2\n+C1+C2+C3\n(ours)"),
        ("granite-free",     "Granite-34B\nfree"),
        ("granite-c1",       "Granite-34B\n+C1"),
        ("codellama-c1",     "CodeLlama-34B\n+C1"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    idx = np.arange(len(systems))
    w = 0.4
    arith, arith_err = [], []
    linalg, linalg_err = [], []
    for key, _ in systems:
        a = summary["arith_func"].get(key, {})
        l = summary["linalg"].get(key, {})
        if a:
            arith.append(100 * a["point"])
            arith_err.append([
                100 * (a["point"] - a["ci_low"]),
                100 * (a["ci_high"] - a["point"]),
            ])
        else:
            arith.append(0); arith_err.append([0, 0])
        if l:
            linalg.append(100 * l["point"])
            linalg_err.append([
                100 * (l["point"] - l["ci_low"]),
                100 * (l["ci_high"] - l["point"]),
            ])
        else:
            linalg.append(0); linalg_err.append([0, 0])
    arith_err = np.array(arith_err).T; linalg_err = np.array(linalg_err).T
    b1 = ax.bar(idx - w/2, arith,  w, yerr=arith_err,  label="arith+func (n=200)",
                color=C_POS, edgecolor="black", linewidth=0.5, capsize=3)
    b2 = ax.bar(idx + w/2, linalg, w, yerr=linalg_err, label="linalg (n=125)",
                color=C_TYPE_SSA, edgecolor="black", linewidth=0.5, capsize=3)
    # Highlight ours with a thicker red border + annotation
    ours = 3
    b1[ours].set_edgecolor("crimson"); b1[ours].set_linewidth(2.0)
    b2[ours].set_edgecolor("crimson"); b2[ours].set_linewidth(2.0)
    ax.annotate("ours", xy=(ours, max(arith[ours], linalg[ours]) + 3),
                ha="center", fontsize=9, fontweight="bold", color="crimson")
    for x, v in zip(idx, arith):
        ax.text(x - w/2, v + 1.5, f"{v:.1f}", ha="center", fontsize=7, color="black")
    for x, v in zip(idx, linalg):
        ax.text(x + w/2, v + 1.5, f"{v:.1f}", ha="center", fontsize=7, color="black")
    ax.set_xticks(idx)
    ax.set_xticklabels([lbl for _, lbl in systems], fontsize=8)
    ax.set_ylabel("verify-valid %")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Main results: verify-valid across dialects\n"
                 "SmolLM2-1.7B + C1+C2+C3 beats both 30B baselines on linalg")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_main_matrix.pdf")
    plt.close(fig)


# =============== FIG 2: constraint-layer progression ===============

def fig2_constraint_progression():
    summary = _load_summary()
    constraints = ["smollm2-none", "smollm2-c1", "smollm2-c1_c2", "smollm2-c1_c2_c3"]
    labels = ["free", "+C1", "+C1+C2", "+C1+C2+C3"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    for dialect, color, marker, n in [
        ("arith_func", C_POS,      "o", "n=200"),
        ("linalg",     C_TYPE_SSA, "s", "n=125"),
    ]:
        ys, lows, highs = [], [], []
        for c in constraints:
            d = summary[dialect][c]
            ys.append(100 * d["point"])
            lows.append(100 * (d["point"] - d["ci_low"]))
            highs.append(100 * (d["ci_high"] - d["point"]))
        ax.errorbar(range(4), ys, yerr=[lows, highs], marker=marker,
                    markersize=8, linewidth=2, color=color, capsize=4,
                    label=f"{dialect.replace('_', '+')} ({n})")

    # Annotate deltas
    ax.annotate("+13.0pp\np<0.0001", xy=(3, 67.5), xytext=(2.3, 82),
                fontsize=8, color=C_POS,
                arrowprops=dict(arrowstyle="->", color=C_POS, lw=1))
    ax.annotate("+4.0pp\np=0.006", xy=(2, 72.0), xytext=(0.9, 90),
                fontsize=8, color=C_TYPE_SSA,
                arrowprops=dict(arrowstyle="->", color=C_TYPE_SSA, lw=1))

    ax.set_xticks(range(4)); ax.set_xticklabels(labels)
    ax.set_ylabel("verify-valid %")
    ax.set_ylim(20, 100)
    ax.set_title("Verify-valid vs. constraint level (SmolLM2-1.7B)\n"
                 "Different layers dominate on different dialects")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_constraint_progression.pdf")
    plt.close(fig)


# =============== FIG 3: error-bucket collapse ===============

def fig3_error_collapse():
    a_cats, l_cats = _err_cats()
    cats_order = ["passed", "type", "type_ssa", "syntax", "other"]
    colors = {"passed": C_POS, "type": C_TYPE, "type_ssa": C_TYPE_SSA,
              "syntax": C_SYN, "other": C_OTHER}
    constraints = ["none", "c1", "c1_c2", "c1_c2_c3"]
    labels = ["free", "C1", "C1+C2", "C1+C2+C3"]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=False)
    for ax, (dialect, data, ntot) in zip(axes, [
        ("arith+func (n=200)", a_cats, 200),
        ("linalg (n=125)",     l_cats, 125),
    ]):
        by_c = {c: {} for c in constraints}
        for key, counts in data.items():
            m, c, _ = key.split("|")
            if m != "smollm2-1.7b" or c not in by_c: continue
            by_c[c] = counts
        bottoms = np.zeros(len(constraints))
        for cat in cats_order:
            vals = np.array([by_c[c].get(cat, 0) for c in constraints], dtype=float)
            ax.bar(labels, vals, bottom=bottoms, label=cat, color=colors[cat],
                   edgecolor="black", linewidth=0.3)
            bottoms += vals
        ax.set_title(f"SmolLM2-1.7B — {dialect}")
        ax.set_ylabel("# of samples")
        ax.set_ylim(0, ntot * 1.05)
        ax.grid(axis="y", alpha=0.25)
    # Annotations on the arith+func panel — C3 collapses type_ssa
    ax_a = axes[0]
    ax_a.annotate("type_ssa\n56 → 17\n(−70%)", xy=(3, 150), xytext=(2.5, 180),
                  fontsize=8, color=C_TYPE_SSA, ha="center",
                  arrowprops=dict(arrowstyle="->", color=C_TYPE_SSA, lw=1))
    axes[1].annotate("syntax\n22 → 0", xy=(2, 98), xytext=(2.0, 118),
                     fontsize=8, color=C_SYN, ha="center",
                     arrowprops=dict(arrowstyle="->", color=C_SYN, lw=1))
    axes[0].legend(loc="upper right", framealpha=0.95, fontsize=7)
    fig.suptitle("Error-category progression — each constraint layer owns its bucket",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_error_collapse.pdf")
    plt.close(fig)


# =============== FIG 4: efficiency frontier ===============

def fig4_efficiency_frontier():
    # Three-seed mean verify rates from apples-to-apples cells; wall-clock
    # per generation from per-prompt JSONL dt fields, aggregated by dialect.
    # 5 systems x 2 dialects = 10 points. Color = system, marker = dialect.
    systems = [
        # (display name,                                 color,     ours)
        ("SmolLM2 1.7B + C1+C2+C3 (ours)",               C_SLM,     True),
        ("Granite-Code-34B + C1+C3",                     C_GRN,     False),
        ("CodeLlama-34B + C1+C3",                        C_CLL,     False),
        ("StarCoder2-15B + C1+C3",                       "#2ca02c", False),
        ("Granite-Code-8B-fp16 + C1+C3 (prec. control)", "#e377c2", False),
    ]
    # (system_idx, marker, wall_s, verify%)
    data = [
        (0, "o", 1.65,  53.2),  # SmolLM2 arith+func
        (0, "s", 1.86,  80.0),  # SmolLM2 linalg
        (1, "o", 40.0,  51.5),  # Granite-34B arith+func
        (1, "s", 40.0,  35.7),  # Granite-34B linalg
        (2, "o", 16.0,  59.8),  # CodeLlama-34B arith+func
        (2, "s", 16.0,  58.7),  # CodeLlama-34B linalg
        (3, "o", 11.6,  66.8),  # StarCoder2-15B arith+func
        (3, "s", 15.8,  54.9),  # StarCoder2-15B linalg
        (4, "o", 11.9,  68.2),  # Granite-8B-fp16 arith+func
        (4, "s", 45.6,  50.1),  # Granite-8B-fp16 linalg
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for idx, m, t, v in data:
        _, color, ours = systems[idx]
        lw = 1.4 if ours else 0.6
        sz = 200 if ours else 140
        ax.scatter(t, v, s=sz, color=color, marker=m, edgecolor="black",
                   linewidth=lw, zorder=3)

    # Annotate the SmolLM2 "ours" points with their headline numbers
    ax.annotate("80.0% / 1.86 s\n(ours, linalg)", xy=(1.86, 80.0),
                xytext=(2.7, 82.5), fontsize=9.5, color=C_SLM, weight="bold")
    ax.annotate("53.2% / 1.65 s\n(ours, arith+func)", xy=(1.65, 53.2),
                xytext=(2.7, 47), fontsize=9.5, color=C_SLM, weight="bold")

    # Reference line at the headline 80.0% verify rate
    ax.axhline(y=80.0, linestyle=":", color=C_SLM, alpha=0.45, lw=1.0)

    from matplotlib.lines import Line2D
    # System legend (upper-right; mostly-empty quadrant)
    sys_handles = [
        Line2D([0], [0], marker="o", color="w", label=name,
               markerfacecolor=color, markersize=11,
               markeredgecolor="black", markeredgewidth=1.4 if ours else 0.6)
        for name, color, ours in systems
    ]
    sys_legend = ax.legend(handles=sys_handles, loc="upper right",
                           framealpha=0.95, fontsize=8.5,
                           title="System", title_fontsize=9.5)
    ax.add_artist(sys_legend)

    # Dialect legend (lower-right) — circle vs square
    marker_handles = [
        Line2D([0], [0], marker="o", color="w", label="arith+func",
               markerfacecolor="gray", markersize=11, markeredgecolor="black"),
        Line2D([0], [0], marker="s", color="w", label="linalg",
               markerfacecolor="gray", markersize=11, markeredgecolor="black"),
    ]
    ax.legend(handles=marker_handles, loc="lower right",
              framealpha=0.95, fontsize=9,
              title="Dialect", title_fontsize=10)

    ax.set_xscale("log")
    ax.set_xlabel("Wall-clock per generation (s, log scale)", fontsize=11)
    ax.set_ylabel("Verify-valid rate (%)", fontsize=11)
    ax.set_xlim(1.2, 60)
    ax.set_ylim(25, 95)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_efficiency_frontier.pdf")
    plt.close(fig)


# =============== FIG 5: scope validator confusion matrix ===============

def fig5_scope_validator_confusion():
    # Re-compute confusion from Day-4 SmolLM2 + C1 data + c3_scope.validate
    from decoder.c3_scope import validate
    rows = [json.loads(l) for l in Path("results/day4/slm_smoke.jsonl").read_text().splitlines()
            if '"smollm2-1.7b"' in l and '"c1"' in l]
    sp_vp = sp_vf = sf_vp = sf_vf = 0
    for r in rows:
        scope = validate(r["generated"]).passed
        verify = bool(r["verify_valid"])
        if scope and verify: sp_vp += 1
        elif scope and not verify: sp_vf += 1
        elif not scope and verify: sf_vp += 1
        else: sf_vf += 1
    total = sp_vp + sp_vf + sf_vp + sf_vf
    cm = np.array([[sp_vp, sp_vf], [sf_vp, sf_vf]])
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() * 0.5 else "black"
            ax.text(j, i, f"{val}\n({100*val/total:.1f}%)", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["verify PASS", "verify FAIL"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["scope PASS", "scope FAIL"])
    ax.set_xlabel("mlir-opt --verify")
    ax.set_ylabel("C3 scope validator")
    ax.set_title("Scope validator vs mlir-opt (SmolLM2 +C1, n=200, Day-4)\n"
                 "Zero false rejects (upper-right quadrant could harm rejection-sampling)")
    # Highlight the zero-false-reject cell
    ax.add_patch(Rectangle((0.5, 0.5), 1, 1, fill=False, edgecolor="crimson", lw=2))
    fig.tight_layout()
    fig.savefig(OUT / "fig5_scope_validator_confusion.pdf")
    plt.close(fig)


# =============== FIG 6: HCS non-reproduction ===============

def fig6_hcs_null():
    hcs_sm  = json.loads(Path("results/day6/hcs_smollm2-1_7b.json").read_text())
    hcs_phi = json.loads(Path("results/day6/hcs_phi-3_5-mini.json").read_text())

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    labels = ["free", "+C1", "+C1+C2"]
    x = np.arange(len(labels))
    w = 0.35
    sm_vals = [100 * hcs_sm["pass_rates"][k] for k in ["base", "c1_only", "c1_c2"]]
    phi_vals = [100 * hcs_phi["pass_rates"][k] for k in ["base", "c1_only", "c1_c2"]]
    ax.bar(x - w/2, sm_vals,  w, color=C_SLM, edgecolor="black", linewidth=0.5, label="SmolLM2-1.7B")
    ax.bar(x + w/2, phi_vals, w, color=C_PHI, edgecolor="black", linewidth=0.5, label="Phi-3.5-mini")
    for xi, v in zip(x - w/2, sm_vals):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=8)
    for xi, v in zip(x + w/2, phi_vals):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("verify-valid %")
    ax.set_ylim(0, 80)
    ax.set_title("Hidden Cost of Structure replication — null result\n"
                 "C1 is monotonically beneficial on both SLMs under few-shot priming")
    ax.legend(framealpha=0.95, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    # Annotate paired Δ
    ax.annotate(f"Δ SmolLM2 C1−free = +20.0pp, p<0.0001",
                xy=(1, 55), xytext=(0.3, 70), fontsize=8, color=C_SLM,
                arrowprops=dict(arrowstyle="->", color=C_SLM, lw=0.8))
    fig.tight_layout()
    fig.savefig(OUT / "fig6_hcs_null.pdf")
    plt.close(fig)


# =============== FIG 7: per-op linalg accuracy ===============

def fig7_per_op_linalg():
    # For each linalg op, compute verify% under SmolLM2 +C1+C2+C3.
    # A prompt is associated with ops referenced in its gold MLIR if it's a seed;
    # otherwise fall back to ops appearing in the generated output.
    rows = [json.loads(l) for l in Path("results/day9/linalg_smoke.jsonl").read_text().splitlines()
            if json.loads(l)["constraint"] == "c1_c2_c3"]
    # Build prompt_id → op from Linalg-Spec-30 seeds (first 30 prompts)
    seed_ops: dict[int, str] = {}
    for i, p in enumerate(sorted(Path("eval/benchmarks/linalg_spec_30/examples").glob("*.json"))):
        d = json.loads(p.read_text())
        m = re.search(r"linalg\.(\w+)", d["mlir"])
        if m: seed_ops[i] = m.group(1)
    # For other prompts, parse op from generated text
    op_stats: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        pid = r["prompt_id"]
        op = seed_ops.get(pid)
        if op is None:
            m = re.search(r"linalg\.(\w+)", r["generated"])
            if m: op = m.group(1)
        if op is None: continue
        # Only count ops in our 12-op scope
        if op not in {"matmul","matvec","fill","copy","transpose","broadcast",
                      "add","sub","mul","div","exp","abs"}:
            continue
        op_stats[op].append(int(bool(r["verify_valid"])))
    ordered = sorted(op_stats.items(), key=lambda kv: (-np.mean(kv[1]) if kv[1] else 0, kv[0]))
    ops = [op for op, _ in ordered]
    rates = [100 * np.mean(v) for _, v in ordered]
    ns    = [len(v) for _, v in ordered]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.bar(ops, rates, color=C_TYPE_SSA, edgecolor="black", linewidth=0.5)
    for b, v, n in zip(bars, rates, ns):
        ax.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.0f}%\nn={n}",
                ha="center", fontsize=7)
    ax.set_ylabel("verify-valid %")
    ax.set_ylim(0, 110)
    ax.set_title("Per-op verify rate — SmolLM2 + C1+C2+C3 on linalg")
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(OUT / "fig7_per_op_linalg.pdf")
    plt.close(fig)


# =============== FIG 8: paired bootstrap deltas ===============

def fig8_paired_bootstrap():
    # For each (comparison, dialect), show the paired bootstrap CI as a horizontal
    # interval plot; annotate the gate (vertical line at 0 and at 10pp).
    summary_aR = json.loads(Path("results/frozen/week2_gate_day5/c3_summary.json").read_text())
    summary_la = json.loads(Path("results/day9/linalg_summary.json").read_text())

    comparisons = [
        ("C1+C2+C3 − C1+C2", "arith+func", summary_aR["paired_diffs"]["c1_c2_c3_vs_c1_c2"]),
        ("C1+C2+C3 − C1+C2", "linalg",     summary_la["paired_diffs"]["c1_c2_c3_vs_c1_c2"]),
        ("C1+C2 − C1",       "arith+func", summary_aR["paired_diffs"]["c1_c2_vs_c1"]),
        ("C1+C2 − C1",       "linalg",     summary_la["paired_diffs"]["c1_c2_vs_c1"]),
        ("C1 − free",        "arith+func", summary_aR["paired_diffs"]["c1_vs_none"]),
        ("C1 − free",        "linalg",     summary_la["paired_diffs"]["c1_vs_none"]),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    y = np.arange(len(comparisons))
    for i, (cmp, dialect, d) in enumerate(comparisons):
        lo = 100 * d["ci_low"]; hi = 100 * d["ci_high"]; pt = 100 * d["point"]
        color = C_POS if dialect == "arith+func" else C_TYPE_SSA
        ax.plot([lo, hi], [i, i], color=color, lw=3, solid_capstyle="round")
        ax.scatter([pt], [i], s=80, color=color, edgecolor="black", zorder=3)
        ax.text(hi + 1, i, f"{pt:+.1f}pp", fontsize=8, va="center", color=color)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(10, color="gray", linewidth=0.8, linestyle="--")
    ax.text(10.3, len(comparisons) - 0.6, "Week-2 gate\n(≥10pp)", fontsize=8, color="gray")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{cmp}  ({d})" for cmp, d, _ in comparisons])
    ax.invert_yaxis()
    ax.set_xlabel("Δ verify-valid (pp)")
    ax.set_xlim(-5, 35)
    ax.set_title("Paired bootstrap Δverify (10k resamples, 95% CI)\n"
                 "C1+C2+C3 − C1+C2 = +13.0pp on arith+func (MET); +0.8pp on linalg (NULL, near zero)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_paired_bootstrap.pdf")
    plt.close(fig)


# =============== driver ===============

def main():
    fig1_main_matrix();             print(f"  {OUT}/fig1_main_matrix.pdf")
    fig2_constraint_progression();  print(f"  {OUT}/fig2_constraint_progression.pdf")
    fig3_error_collapse();          print(f"  {OUT}/fig3_error_collapse.pdf")
    fig4_efficiency_frontier();     print(f"  {OUT}/fig4_efficiency_frontier.pdf")
    fig5_scope_validator_confusion(); print(f"  {OUT}/fig5_scope_validator_confusion.pdf")
    fig6_hcs_null();                print(f"  {OUT}/fig6_hcs_null.pdf")
    fig7_per_op_linalg();           print(f"  {OUT}/fig7_per_op_linalg.pdf")
    fig8_paired_bootstrap();        print(f"  {OUT}/fig8_paired_bootstrap.pdf")


if __name__ == "__main__":
    main()
