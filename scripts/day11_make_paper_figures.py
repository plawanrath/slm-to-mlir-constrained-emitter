"""Day-11: deterministic figure + table generator reading frozen Day-10 matrix.

Outputs:
  docs/paper/figures/fig_main_matrix.pdf       — §5.2 cross-dialect verify matrix (bar chart)
  docs/paper/figures/fig_error_categories.pdf  — stacked bar, SmolLM2 per dialect
  docs/paper/figures/table_main.tex            — LaTeX Table 1
  docs/paper/figures/table_error_cat.tex       — LaTeX error-category table

Headless matplotlib so it runs in CI / the reproducibility Docker.
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

OUT = Path("docs/paper/figures")
OUT.mkdir(parents=True, exist_ok=True)


# ---------- main matrix bar chart ----------

def fig_main_matrix():
    summary = json.loads(Path("results/frozen/day10_final_matrix/linalg_matrix_summary.json").read_text())
    systems = [
        ("smollm2-none",     "SmolLM2, free"),
        ("smollm2-c1",       "SmolLM2 +C1"),
        ("smollm2-c1_c2",    "SmolLM2 +C1+C2"),
        ("smollm2-c1_c2_c3", "SmolLM2 +C1+C2+C3"),
        ("granite-free",     "Granite-34B, free"),
        ("granite-c1",       "Granite-34B +C1"),
        ("codellama-c1",     "CodeLlama-34B +C1"),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    idx = np.arange(len(systems))
    w = 0.38
    arith = []; linalg = []
    for key, _ in systems:
        a = summary["arith_func"].get(key, {})
        l = summary["linalg"].get(key, {})
        arith.append(100 * a.get("point", 0) if a else 0)
        linalg.append(100 * l.get("point", 0) if l else 0)
    b1 = ax.bar(idx - w/2, arith, w, label="arith+func (n=200)", color="#4C78A8")
    b2 = ax.bar(idx + w/2, linalg, w, label="linalg (n=125)",    color="#F58518")
    # Highlight our system
    for b, a_v in zip(b1, arith):
        b.set_edgecolor("black"); b.set_linewidth(0.5)
    for b, l_v in zip(b2, linalg):
        b.set_edgecolor("black"); b.set_linewidth(0.5)
    # Red outline on our ours row
    ours_idx = 3
    b1[ours_idx].set_edgecolor("crimson"); b1[ours_idx].set_linewidth(2.0)
    b2[ours_idx].set_edgecolor("crimson"); b2[ours_idx].set_linewidth(2.0)
    ax.set_xticks(idx)
    ax.set_xticklabels([lbl for _, lbl in systems], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("verify-valid %")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("§5.2 Verify-valid across dialects (SmolLM2-1.7B vs 30B baselines)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_main_matrix.pdf", dpi=200)
    plt.close(fig)


# ---------- error categories stacked bar ----------

def fig_error_categories():
    # Use SmolLM2 linalg + arith+func side-by-side.
    cats_order = ["passed", "type", "type_ssa", "syntax", "other"]
    colors = {"passed": "#4C78A8", "type": "#E45756", "type_ssa": "#F58518",
              "syntax": "#72B7B2", "other": "#B279A2"}
    constraints = ["none", "c1", "c1_c2", "c1_c2_c3"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (dialect, path) in zip(axes, [
        ("arith+func", "results/day6/error_categories.json"),
        ("linalg",     "results/day10/linalg_error_categories.json"),
    ]):
        d = json.loads(Path(path).read_text())
        # Filter to smollm2-1.7b
        by_c = {c: {} for c in constraints}
        for key, counts in d.items():
            m, c, _ = key.split("|")
            if m != "smollm2-1.7b" or c not in by_c: continue
            by_c[c] = counts
        bottoms = np.zeros(len(constraints))
        for cat in cats_order:
            vals = np.array([by_c[c].get(cat, 0) for c in constraints], dtype=float)
            ax.bar(constraints, vals, bottom=bottoms, label=cat, color=colors[cat])
            bottoms += vals
        ax.set_title(f"SmolLM2-1.7B — {dialect}")
        ax.set_ylabel("# of n=200 (arith) / n=125 (linalg)")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper right", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_error_categories.pdf", dpi=200)
    plt.close(fig)


# ---------- LaTeX tables ----------

def tex_table_main():
    summary = json.loads(Path("results/frozen/day10_final_matrix/linalg_matrix_summary.json").read_text())
    rows = [
        ("SmolLM2-1.7B, free",              "smollm2-none"),
        ("SmolLM2-1.7B + C1",               "smollm2-c1"),
        ("SmolLM2-1.7B + C1+C2",            "smollm2-c1_c2"),
        ("\\textbf{SmolLM2-1.7B + C1+C2+C3 (ours)}", "smollm2-c1_c2_c3"),
        ("Granite-Code-34B, free",          "granite-free"),
        ("Granite-Code-34B + C1",           "granite-c1"),
        ("CodeLlama-34B + C1",              "codellama-c1"),
    ]
    def _fmt(c):
        if c is None:
            return "---"
        return f"{100*c['point']:.1f} [{100*c['ci_low']:.1f}, {100*c['ci_high']:.1f}]"
    tex = []
    tex.append(r"\begin{tabular}{l r r}")
    tex.append(r"\toprule")
    tex.append(r"System & arith+func (n=200) & linalg (n=125) \\")
    tex.append(r"\midrule")
    for label, key in rows:
        a = summary["arith_func"].get(key); l = summary["linalg"].get(key)
        tex.append(f"{label} & {_fmt(a)} & {_fmt(l)} \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    (OUT / "table_main.tex").write_text("\n".join(tex))


def tex_table_error_cat():
    d_arith  = json.loads(Path("results/day6/error_categories.json").read_text())
    d_linalg = json.loads(Path("results/day10/linalg_error_categories.json").read_text())
    cats = ("passed", "type", "type_ssa", "syntax", "other")
    def _extract(d, constraint):
        for key, counts in d.items():
            m, c, _ = key.split("|")
            if m == "smollm2-1.7b" and c == constraint:
                return counts
        return {}
    tex = []
    tex.append(r"\begin{tabular}{l l r r r r r}")
    tex.append(r"\toprule")
    tex.append(r"Dialect & Constraint & passed & type & type_ssa & syntax & other \\")
    tex.append(r"\midrule")
    for dialect, data in [("arith+func", d_arith), ("linalg", d_linalg)]:
        for c in ("none", "c1", "c1_c2", "c1_c2_c3"):
            counts = _extract(data, c)
            if not counts: continue
            tex.append(
                f"{dialect} & {c} & "
                f"{counts.get('passed',0)} & {counts.get('type',0)} & "
                f"{counts.get('type_ssa',0)} & {counts.get('syntax',0)} & "
                f"{counts.get('other',0)} \\\\"
            )
        tex.append(r"\midrule")
    tex.pop()  # remove trailing midrule
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    (OUT / "table_error_cat.tex").write_text("\n".join(tex))


def main():
    fig_main_matrix()
    print(f"  wrote {OUT}/fig_main_matrix.pdf")
    fig_error_categories()
    print(f"  wrote {OUT}/fig_error_categories.pdf")
    tex_table_main()
    print(f"  wrote {OUT}/table_main.tex")
    tex_table_error_cat()
    print(f"  wrote {OUT}/table_error_cat.tex")


if __name__ == "__main__":
    main()
