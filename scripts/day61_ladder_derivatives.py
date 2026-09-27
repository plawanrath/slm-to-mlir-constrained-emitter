"""Day-61: ladder-derived table and figure recomputed from the frozen-pool ladder runs.

Context (camera-ready, NeurIPS 2026 E&D). The submitted Table 6 (error
categories per constraint level) and Figure 5 (verify-valid vs constraint
level) were computed from the pre-freeze draft-pool ladder runs (Day-5 arith,
Day-10 linalg). The response period regenerated both ladders on the frozen
released pool (Day-53 arith, Day-59 linalg). This script recomputes the two
derivatives from those frozen runs. No inference and no new verification:
verifier stderr comes from the sqlite verify cache written by the pinned
container during the Day-53/59 runs, exactly as scripts/day6_project_to_matrix.py
did for the submitted table (parse-valid rows take their cached stderr;
parse-invalid rows have empty stderr and fall into "other").

Checks: per rung, the "passed" count equals the ladder's verify-valid count
(arith 40/89/91/104 of 200; linalg 65/80/92/97 of 125) and categories sum to n.
Figure 5 annotations use the paired rung deltas as posted in the author
response (arith C3 +6.5pp, p=0.002; linalg C2 +9.6pp, p<0.0001).

Outputs (paper assets are staged here and copied into the paper later):
  results/day61/ladder_error_categories.json
  results/day61/paper_assets/table_error_cat.tex
  results/day61/paper_assets/fig5_constraint_progression.pdf
  results/day61/paper_assets/fig6_error_collapse.pdf   (annotations computed from the counts)
  results/day61/paper_assets/fig7_paired_bootstrap.pdf (paired rung deltas, eval.stats.paired_bootstrap_diff)
  results/day61/ladder_paired_deltas.json

Paired rung deltas use eval.stats.paired_bootstrap_diff (10,000 resamples, seed 0, paired by
prompt_id) and are checked against the values posted in the author response.
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

from eval.error_categories import categorize  # noqa: E402
from scripts.env.verify_cache import VerifyCache  # noqa: E402

LADDERS = {
    "arith+func": ("results/day53/e10_ladder_frozen.jsonl", "results/day53/e10_summary.json", 200),
    "linalg": ("results/day59/e10b_ladder_linalg_frozen.jsonl", "results/day59/e10b_summary.json", 125),
}
RUNGS = ["none", "c1", "c1_c2", "c1_c2_c3"]
CATS = ("passed", "type", "type_ssa", "syntax", "other")
OUT = REPO / "results/day61"
ASSETS = OUT / "paper_assets"


def error_categories() -> dict:
    cache = VerifyCache()
    out: dict = {}
    ok = True
    for dialect, (path, summ_path, n) in LADDERS.items():
        summary = json.loads(Path(summ_path).read_text())
        rows = [json.loads(l) for l in open(path) if l.strip()]
        for rung in RUNGS:
            sub = [r for r in rows if r["constraint"] == rung]
            counts: Counter = Counter()
            misses = 0
            for r in sub:
                if r["verify_valid"]:
                    counts["passed"] += 1
                    continue
                stderr = ""
                if r["parse_valid"]:
                    hit = cache.get(r["generated"], ("--verify-diagnostics",))
                    if hit is None:
                        misses += 1
                    else:
                        stderr = hit["stderr"] or ""
                counts[categorize(stderr)] += 1
            want = summary[rung]["verify_valid"]
            good = len(sub) == n and counts["passed"] == want and sum(counts.values()) == n and misses == 0
            ok &= good
            print(f"{'ok ' if good else 'DIFF'} {dialect:10s} {rung:9s} n={len(sub)} passed={counts['passed']} "
                  f"(ladder {want}) cache_misses={misses} {dict(counts)}")
            out[f"smollm2-1.7b|{rung}|{dialect}"] = dict(counts)
    if not ok:
        raise SystemExit("error-category recomputation does not match the frozen ladders")
    return out


def tex_table(cats: dict) -> str:
    esc = lambda s: s.replace("_", r"\_")
    lines = [r"\begin{tabular}{l l r r r r r}", r"\toprule",
             r"Dialect & Constraint & passed & type & type\_ssa & syntax & other \\", r"\midrule"]
    for i, dialect in enumerate(LADDERS):
        if i:
            lines.append(r"\midrule")
        for rung in RUNGS:
            c = cats[f"smollm2-1.7b|{rung}|{dialect}"]
            lines.append(f"{dialect} & {esc(rung)} & " + " & ".join(str(c.get(k, 0)) for k in CATS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def figure(path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})
    colors = {"arith+func": ("#4C78A8", "o"), "linalg": ("#F58518", "s")}
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    pts = {}
    for dialect, (_, summ_path, n) in LADDERS.items():
        s = json.loads(Path(summ_path).read_text())
        ys = [100 * s[r]["verify_rate"] for r in RUNGS]
        lo = [100 * (s[r]["verify_rate"] - s[r]["verify_ci95"][0]) for r in RUNGS]
        hi = [100 * (s[r]["verify_ci95"][1] - s[r]["verify_rate"]) for r in RUNGS]
        col, mk = colors[dialect]
        ax.errorbar(range(4), ys, yerr=[lo, hi], marker=mk, markersize=8, linewidth=2,
                    color=col, capsize=4, label=f"{dialect} (n={n})")
        pts[dialect] = ys
    ax.annotate("C3: +6.5pp\np=0.002", xy=(3, pts["arith+func"][3]), xytext=(2.35, 30),
                fontsize=8, color=colors["arith+func"][0],
                arrowprops=dict(arrowstyle="->", color=colors["arith+func"][0], lw=1))
    ax.annotate("C2: +9.6pp\np<0.0001", xy=(2, pts["linalg"][2]), xytext=(1.0, 90),
                fontsize=8, color=colors["linalg"][0],
                arrowprops=dict(arrowstyle="->", color=colors["linalg"][0], lw=1))
    ax.set_xticks(range(4))
    ax.set_xticklabels(["free", "+C1", "+C1+C2", "+C1+C2+C3"])
    ax.set_ylabel("verify-valid %")
    ax.set_ylim(10, 100)
    ax.set_title("Verify-valid vs. constraint level (SmolLM2-1.7B, frozen released pool)\n"
                 "Different layers dominate on different dialects")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_errors(cats: dict, path: Path) -> dict:
    """Error-bucket stacked bars (make_paper_figures_final.fig3_error_collapse), frozen pool."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.fontsize": 8,
                         "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})
    colors = {"passed": "#4C78A8", "type": "#E45756", "type_ssa": "#F58518",
              "syntax": "#72B7B2", "other": "#B279A2"}
    labels = ["free", "C1", "C1+C2", "C1+C2+C3"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for ax, (dialect, (_, _, n)) in zip(axes, LADDERS.items()):
        bottoms = np.zeros(4)
        for cat in CATS:
            vals = np.array([cats[f"smollm2-1.7b|{r}|{dialect}"].get(cat, 0) for r in RUNGS], dtype=float)
            ax.bar(labels, vals, bottom=bottoms, label=cat, color=colors[cat], edgecolor="black", linewidth=0.3)
            bottoms += vals
        ax.set_title(f"SmolLM2-1.7B, {dialect} (n={n})")
        ax.set_ylabel("# of samples")
        ax.set_ylim(0, n * 1.32)
        ax.grid(axis="y", alpha=0.25)
    a = [cats[f"smollm2-1.7b|{r}|arith+func"].get("type_ssa", 0) for r in RUNGS]
    l = [cats[f"smollm2-1.7b|{r}|linalg"].get("syntax", 0) for r in RUNGS]
    notes = {"arith_type_ssa_c1c2_to_c1c2c3": [a[2], a[3], round(100 * (a[3] - a[2]) / a[2])],
             "linalg_syntax_c1_to_c1c2": [l[1], l[2]]}
    axes[0].annotate(f"type_ssa {a[2]} \u2192 {a[3]} ({notes['arith_type_ssa_c1c2_to_c1c2c3'][2]:+d}%)",
                     xy=(3, 203), xytext=(2.1, 245), fontsize=8, color=colors["type_ssa"], ha="center",
                     arrowprops=dict(arrowstyle="->", color=colors["type_ssa"], lw=1))
    axes[1].annotate(f"syntax {l[1]} \u2192 {l[2]}", xy=(2, 127), xytext=(1.5, 153), fontsize=8,
                     color=colors["syntax"], ha="center",
                     arrowprops=dict(arrowstyle="->", color=colors["syntax"], lw=1))
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=5, framealpha=0.95, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Error-category progression on the frozen released pool", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return notes


POSTED_DELTAS = {  # (dialect, a, b): (point, ci_low, ci_high) in pp, as posted in the author response
    ("arith+func", "c1", "none"): (24.5, 17.5, 31.5),
    ("arith+func", "c1_c2", "c1"): (1.0, 0.0, 2.5),
    ("arith+func", "c1_c2_c3", "c1_c2"): (6.5, 2.0, 11.0),
    ("linalg", "c1_c2", "c1"): (9.6, 4.8, 15.2),
}


def paired_deltas() -> dict:
    from eval.stats import paired_bootstrap_diff
    out, ok = {}, True
    for dialect, (path, _, n) in LADDERS.items():
        rows = [json.loads(l) for l in open(path) if l.strip()]
        v = {c: [int(r["verify_valid"]) for r in sorted((r for r in rows if r["constraint"] == c),
                                                          key=lambda r: int(r["prompt_id"]))] for c in RUNGS}
        for a, b in (("c1", "none"), ("c1_c2", "c1"), ("c1_c2_c3", "c1_c2"), ("c1_c2_c3", "none")):
            r = paired_bootstrap_diff(v[a], v[b])
            d = {"point": round(100 * r.point, 1), "ci": [round(100 * r.ci_low, 1), round(100 * r.ci_high, 1)],
                 "p_one_sided": round(r.p_value, 4), "n": r.n}
            out[f"{dialect}|{a}-{b}"] = d
            want = POSTED_DELTAS.get((dialect, a, b))
            if want:
                good = (d["point"], *d["ci"]) == want
                ok &= good
                print(f"{'ok ' if good else 'DIFF'} {dialect} {a}-{b}: {d['point']:+.1f} {d['ci']} (posted {want})")
    if not ok:
        raise SystemExit("paired rung deltas do not reproduce the posted values")
    return out


def figure_paired(deltas: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})
    rows = [("C1+C2+C3 \u2212 C1+C2", "arith+func", "c1_c2_c3-c1_c2"), ("C1+C2+C3 \u2212 C1+C2", "linalg", "c1_c2_c3-c1_c2"),
            ("C1+C2 \u2212 C1", "arith+func", "c1_c2-c1"), ("C1+C2 \u2212 C1", "linalg", "c1_c2-c1"),
            ("C1 \u2212 free", "arith+func", "c1-none"), ("C1 \u2212 free", "linalg", "c1-none")]
    colors = {"arith+func": "#4C78A8", "linalg": "#F58518"}
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for i, (label, dialect, key) in enumerate(rows):
        d = deltas[f"{dialect}|{key}"]
        c = colors[dialect]
        ax.plot(d["ci"], [i, i], color=c, lw=3, solid_capstyle="round")
        ax.scatter([d["point"]], [i], s=80, color=c, edgecolor="black", zorder=3)
        ax.text(d["ci"][1] + 1, i, f"{d['point']:+.1f}pp", fontsize=8, va="center", color=c)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([f"{lab}  ({d})" for lab, d, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("\u0394 verify-valid (pp)")
    ax.set_xlim(-5, 40)
    ax.set_title("Paired bootstrap \u0394 verify-valid per constraint layer (frozen released pool)\n"
                 "10,000 resamples, 95% CI, paired by prompt")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    cats = error_categories()
    (OUT / "ladder_error_categories.json").write_text(json.dumps(cats, indent=2))
    (ASSETS / "table_error_cat.tex").write_text(tex_table(cats))
    figure(ASSETS / "fig5_constraint_progression.pdf")
    notes = figure_errors(cats, ASSETS / "fig6_error_collapse.pdf")
    (OUT / "ladder_error_categories.json").write_text(json.dumps({"counts": cats, "fig6_annotations": notes}, indent=2))
    print(f"fig6 annotations: {notes}")
    deltas = paired_deltas()
    (OUT / "ladder_paired_deltas.json").write_text(json.dumps(deltas, indent=2))
    figure_paired(deltas, ASSETS / "fig7_paired_bootstrap.pdf")
    print("wrote fig7_paired_bootstrap.pdf")
    print(f"wrote {ASSETS.relative_to(REPO)}/table_error_cat.tex and fig5_constraint_progression.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
