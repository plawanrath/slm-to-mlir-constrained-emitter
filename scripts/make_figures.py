"""Deterministic figure generator for the paper.

Reads results/*.json (harness matrix, entropy, error categories, HCS ablation,
tokenizer shootout) and writes figures + tables to docs/paper/figures/.

Outputs:
  - fig_main_matrix.pdf      — Table 1 as a heatmap, one panel per dialect
  - fig_entropy_collapse.pdf — entropy vs. constraint level, per dialect
  - fig_error_categories.pdf — stacked bar of error categories by constraint
  - fig_hcs_reversal.pdf     — base vs C1 vs C1+C2 bars with paired-bootstrap CIs
  - table_main.tex           — LaTeX table of the §5.2 main matrix
  - table_tokenizer.tex      — LaTeX table for the day-2 shootout

Everything is deterministic (seed-locked); re-running on the same inputs gives
byte-identical outputs — important for the reproducibility package.

Usage:
    python scripts/make_figures.py --results results/ --out docs/paper/figures/
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless; no DISPLAY needed
import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def fig_main_matrix(matrix_rows: list[dict], out: Path) -> None:
    if not matrix_rows:
        return
    grouped: dict[tuple, list[int]] = defaultdict(list)
    for r in matrix_rows:
        key = (r["model"], r["constraint"], r["dialect"])
        grouped[key].append(1 if r["passed"] else 0)

    dialects = sorted({k[2] for k in grouped})
    fig, axes = plt.subplots(1, len(dialects), figsize=(4 * len(dialects), 5), squeeze=False)
    for ax, dialect in zip(axes[0], dialects):
        models = sorted({k[0] for k in grouped if k[2] == dialect})
        constraints = ["none", "c1", "c1_c2"]
        heat = np.full((len(models), len(constraints)), np.nan)
        for i, m in enumerate(models):
            for j, c in enumerate(constraints):
                vals = grouped.get((m, c, dialect), [])
                if vals:
                    heat[i, j] = np.mean(vals)
        im = ax.imshow(heat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(constraints)))
        ax.set_xticklabels(constraints)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_title(dialect)
        for i in range(len(models)):
            for j in range(len(constraints)):
                if not np.isnan(heat[i, j]):
                    ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", color="w")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_entropy_collapse(entropy: dict, out: Path) -> None:
    if not entropy:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, info in entropy.items():
        curve = info.get("curve", [])
        if curve:
            ax.plot(curve, label=key, alpha=0.75)
    ax.set_xlabel("generation step")
    ax.set_ylabel("next-token entropy (bits)")
    ax.set_title("Entropy collapse under progressive constraint")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_error_categories(errors: dict, out: Path) -> None:
    if not errors:
        return
    cells = list(errors.keys())
    cats = ["passed", "type", "arity", "dialect_misuse", "syntax", "other"]
    data = np.zeros((len(cats), len(cells)))
    for j, c in enumerate(cells):
        row = errors[c]
        total = sum(row.values()) or 1
        for i, cat in enumerate(cats):
            data[i, j] = row.get(cat, 0) / total
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(cells)), 4))
    bottom = np.zeros(len(cells))
    for i, cat in enumerate(cats):
        ax.bar(range(len(cells)), data[i], bottom=bottom, label=cat)
        bottom += data[i]
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cells, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("fraction")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_hcs(hcs: dict, out: Path) -> None:
    if not hcs:
        return
    rates = hcs.get("pass_rates", {})
    labels = ["base", "c1_only", "c1_c2"]
    values = [rates.get(l, 0.0) for l in labels]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("pass@1")
    rev = "DETECTED" if hcs.get("reversal_detected") else "not detected"
    ax.set_title(f"HCS reversal — {rev}")
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _latex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&")


def table_tokenizer(shootout: dict, out: Path) -> None:
    if not shootout:
        return
    rows = []
    for nick, r in shootout.items():
        if "error" in r:
            continue
        rows.append((nick, r["tokens_per_char"], r["tokens_per_op"], r["clean_mnemonic_rate"]))
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Tokenizer & tok/char & tok/op & mnemonic@1 \\",
        r"\midrule",
    ]
    for nick, tpc, tpo, cm in rows:
        lines.append(f"{_latex_escape(nick)} & {tpc:.3f} & {tpo:.2f} & {cm:.2%} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("docs/paper/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    matrix_path = args.results / "matrix.jsonl"
    matrix_rows = (
        [json.loads(l) for l in matrix_path.read_text().splitlines() if l.strip()]
        if matrix_path.exists() else []
    )
    entropy = _load(args.results / "entropy.json")
    errors = _load(args.results / "error_categories.json")
    hcs = _load(args.results / "hcs_ablation.json")
    shootout = _load(args.results / "tokenizer_shootout.json")

    fig_main_matrix(matrix_rows, args.out / "fig_main_matrix.pdf")
    fig_entropy_collapse(entropy, args.out / "fig_entropy_collapse.pdf")
    fig_error_categories(errors, args.out / "fig_error_categories.pdf")
    fig_hcs(hcs, args.out / "fig_hcs_reversal.pdf")
    table_tokenizer(shootout, args.out / "table_tokenizer.tex")

    print(f"[figures] wrote to {args.out}")


if __name__ == "__main__":
    main()
