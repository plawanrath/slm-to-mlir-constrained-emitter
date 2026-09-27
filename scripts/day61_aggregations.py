"""Day-61: provenance labels and both aggregations for every pooled table (frozen released pool).

Context (camera-ready, NeurIPS 2026 E&D). The author responses commit to
reporting every table on the single frozen released pool, in both aggregations
(all prompts and unique prompts), with per-prompt provenance labels. This
script produces:

  results/day61/provenance.jsonl
      one row per (dialect, prompt_id) of the n=200 arith+func and n=125 linalg
      pools: source (spec / pad), spec example id or L3 corpus row, and the
      unique-string group (duplicates share a group id).
  results/day61/aggregations.json
      for every pooled cell used by a paper table: n, all-prompt rate with a
      95% percentile bootstrap CI (10,000 resamples, seed 0, prompt unit; the
      same procedure as scripts/day53_e10_ladder_frozen.py), and the
      unique-prompt rate (one value per unique prompt string = mean over its
      duplicates; three-seed cells are aggregated per seed, then averaged,
      as in the unique-prompt robustness check).
  results/day61/paper_assets/table_main.tex         frozen-pool Table 2
  results/day61/paper_assets/table_best_config.tex  best-configuration table
  results/day61/paper_assets/table_unique_prompt.tex appendix twin table

Checks (exit 1 on any disagreement): ladder rungs match the Day-53/59
summaries; three-seed means match the published multi-seed values; the
unique-prompt means reproduce the values posted in the author response;
StableHLO pools contain no duplicate prompts, so both aggregations
coincide there.

The frozen-pool C1-only / free arith+func baseline cells come from
scripts/day61_frozen_c1only_baselines.py; pass --allow-missing to run before
that finishes (tables are then not written).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from scripts.day18_30b_c3_rejection import _prompts_arith, _prompts_linalg  # noqa: E402

OUT = REPO / "results/day61"
ASSETS = OUT / "paper_assets"
DIALECT_N = {"arith+func": 200, "linalg": 125}
MULTISEED = ["results/day51_seed0_n200/multiseed_seed0.jsonl", "results/day39pm/multiseed_fulln.jsonl",
             "results/day50/baseline_linalg_multiseed.jsonl", "results/day51/granite_8b_fp16_linalg.jsonl",
             "results/day52/granite_8b_fp16_arith.jsonl"]


def load(p) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def boot_ci(flags: list[bool], n_boot: int = 10_000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    n = len(flags)
    rates = sorted(sum(flags[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return [round(100 * rates[int(0.025 * n_boot)], 1), round(100 * rates[int(0.975 * n_boot)], 1)]


def unique_rate(rows: list[dict]) -> float:
    g = defaultdict(list)
    for r in rows:
        g[r["nl"]].append(bool(r["verify_valid"]))
    return 100 * sum(sum(v) / len(v) for v in g.values()) / len(g)


def provenance() -> dict:
    out = {}
    lines = []
    for dialect, prompts, n_spec, ex_dir in (
            ("arith+func", _prompts_arith(200), 150, "eval/benchmarks/mlir_spec_150/examples"),
            ("linalg", _prompts_linalg(125), 30, "eval/benchmarks/linalg_spec_30/examples")):
        spec_ids = [json.loads(p.read_text())["id"] for p in sorted(Path(ex_dir).glob("*.json"))]
        first: dict[str, int] = {}
        for i, nl in enumerate(prompts):
            group = first.setdefault(nl, i)
            lines.append({"dialect": dialect, "prompt_id": i,
                          "source": "spec" if i < n_spec else "pad_l3_mined",
                          "spec_id": spec_ids[i] if i < n_spec else None,
                          "unique_group": group, "is_first_occurrence": group == i})
        out[dialect] = {"n": len(prompts), "unique": len(first), "spec": n_spec, "pad": len(prompts) - n_spec}
    (OUT / "provenance.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    return out


def single_cell(rows: list[dict], dialect: str) -> dict:
    assert len(rows) == DIALECT_N[dialect], (dialect, len(rows))
    flags = [bool(r["verify_valid"]) for r in sorted(rows, key=lambda r: int(r["prompt_id"]))]
    return {"n": len(rows), "all": round(100 * sum(flags) / len(flags), 1), "ci95": boot_ci(flags),
            "unique": round(unique_rate(rows), 1)}


def three_seed_cell(rows: list[dict], model: str, dialect: str) -> dict:
    per_all, per_uniq = [], []
    for s in (0, 1, 2):
        sub = [r for r in rows if r["model"] == model and r["dialect"] == dialect and int(r["seed"]) == s]
        assert len(sub) == DIALECT_N[dialect], (model, dialect, s, len(sub))
        per_all.append(100 * sum(bool(r["verify_valid"]) for r in sub) / len(sub))
        per_uniq.append(unique_rate(sub))
    return {"n_per_seed": DIALECT_N[dialect], "per_seed": [round(x, 1) for x in per_all],
            "all": round(sum(per_all) / 3, 1), "half_range": round((max(per_all) - min(per_all)) / 2, 1),
            "unique": round(sum(per_uniq) / 3, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(key, got, want):
        checks.append((key, got, want, got == want))

    res: dict = {"provenance": provenance(), "ladder": {}, "three_seed": {}, "single_seed_baselines": {}}

    # ladders (seed 0, frozen pool)
    for dialect, path, summ in (("arith+func", "results/day53/e10_ladder_frozen.jsonl", "results/day53/e10_summary.json"),
                                ("linalg", "results/day59/e10b_ladder_linalg_frozen.jsonl", "results/day59/e10b_summary.json")):
        rows, s = load(path), json.loads(Path(summ).read_text())
        for rung in ("none", "c1", "c1_c2", "c1_c2_c3"):
            cell = single_cell([r for r in rows if r["constraint"] == rung], dialect)
            res["ladder"][f"{dialect}.{rung}"] = cell
            check(f"ladder.{dialect}.{rung}", cell["all"], round(100 * s[rung]["verify_rate"], 1))
            check(f"ladder_ci.{dialect}.{rung}", cell["ci95"], [round(100 * x, 1) for x in s[rung]["verify_ci95"]])

    # three-seed cells
    ms = sum((load(p) for p in MULTISEED), [])
    posted_all = {("smollm2-c1c2c3", "linalg"): 80.0, ("smollm2-c1c2c3", "arith+func"): 53.2,
                  ("codellama-c1c3", "linalg"): 58.7, ("starcoder2-c1c3", "linalg"): 54.9,
                  ("granite-c1c3", "linalg"): 35.7, ("codellama-c1c3", "arith+func"): 59.8,
                  ("starcoder2-c1c3", "arith+func"): 66.8, ("granite-c1c3", "arith+func"): 51.5,
                  ("granite-8b-fp16-c1c3", "arith+func"): 68.2, ("granite-8b-fp16-c1c3", "linalg"): 50.1}
    posted_uniq = {("smollm2-c1c2c3", "linalg"): 75.7, ("codellama-c1c3", "linalg"): 63.6,
                   ("starcoder2-c1c3", "linalg"): 40.4, ("granite-c1c3", "linalg"): 38.6,
                   ("smollm2-c1c2c3", "arith+func"): 52.2, ("codellama-c1c3", "arith+func"): 59.4,
                   ("starcoder2-c1c3", "arith+func"): 67.2, ("granite-c1c3", "arith+func"): 53.0}
    for (m, d), want in posted_all.items():
        cell = three_seed_cell(ms, m, d)
        res["three_seed"][f"{m}.{d}"] = cell
        check(f"three_seed.{m}.{d}", cell["all"], want)
        if (m, d) in posted_uniq:
            check(f"unique.{m}.{d}", cell["unique"], posted_uniq[(m, d)])

    # single-seed baseline cells without C3 (frozen pool)
    src = {("codellama-c1", "linalg"): "results/day10/linalg_baselines.jsonl",
           ("granite-c1", "linalg"): "results/day10/linalg_baselines.jsonl",
           ("granite-free", "linalg"): "results/day10/linalg_baselines.jsonl",
           ("starcoder2-c1", "linalg"): "results/day19/starcoder2_baselines.jsonl",
           ("starcoder2-free", "linalg"): "results/day19/starcoder2_baselines.jsonl",
           ("starcoder2-c1", "arith+func"): "results/day19/starcoder2_baselines.jsonl",
           ("starcoder2-free", "arith+func"): "results/day19/starcoder2_baselines.jsonl",
           ("codellama-c1", "arith+func"): "results/day61/frozen_c1only_arith.jsonl",
           ("granite-c1", "arith+func"): "results/day61/frozen_c1only_arith.jsonl",
           ("codellama-free", "arith+func"): "results/day61/frozen_c1only_arith.jsonl",
           ("granite-free", "arith+func"): "results/day61/frozen_c1only_arith.jsonl"}
    pools = {"arith+func": _prompts_arith(200), "linalg": _prompts_linalg(125)}
    missing = []
    for (m, d), path in src.items():
        rows = [r for r in load(path) if r["model"] == m and r.get("dialect") in (None, d)] if Path(path).exists() else []
        if len(rows) != DIALECT_N[d]:
            missing.append(f"{m}.{d} ({len(rows)}/{DIALECT_N[d]})")
            continue
        assert [r["nl"] for r in sorted(rows, key=lambda r: int(r["prompt_id"]))] == pools[d], f"{m}.{d} not on the frozen pool"
        res["single_seed_baselines"][f"{m}.{d}"] = {**single_cell(rows, d), "source": path}
    for m, d, want in (("codellama-c1", "linalg", 56.8), ("granite-c1", "linalg", 33.6), ("starcoder2-c1", "linalg", 47.2),
                       ("starcoder2-c1", "arith+func", 67.0), ("granite-free", "linalg", 27.2)):
        check(f"single.{m}.{d}", res["single_seed_baselines"][f"{m}.{d}"]["all"], want)

    # StableHLO pools: no duplicates, so both aggregations coincide
    for name, path in (("StableHLO-Spec-30", "results/day33/stablehlo_baselines.jsonl"),
                       ("StableHLO-Held-Out-200", "results/day51/stablehlo_held_out_200_baselines.jsonl")):
        rows = load(path)
        m0 = rows[0]["model"]
        nls = [r["nl"] for r in rows if r["model"] == m0]
        check(f"stablehlo_no_duplicates.{name}", len(set(nls)) == len(nls), True)
    res["stablehlo_aggregations_coincide"] = True

    bad = [c for c in checks if not c[3]]
    for key, got, want, ok in checks:
        if not ok:
            print(f"DIFF {key}: computed={got} expected={want}")
    print(f"{len(checks) - len(bad)}/{len(checks)} checks agree")
    res["checks"] = [{"key": k, "computed": g, "expected": w, "agree": ok} for k, g, w, ok in checks]
    res["missing_cells"] = missing
    (OUT / "aggregations.json").write_text(json.dumps(res, indent=2))
    if missing:
        print("missing cells: " + ", ".join(missing))
        if not args.allow_missing:
            return 1
        return 1 if bad else 0

    write_tables(res)
    return 1 if bad else 0


def fmt(c: dict) -> str:
    return f"{c['all']:.1f} [{c['ci95'][0]:.1f}, {c['ci95'][1]:.1f}]"


def write_tables(res: dict) -> None:
    L, S = res["ladder"], res["single_seed_baselines"]
    rows = [("SmolLM2-1.7B, free", L["arith+func.none"], L["linalg.none"]),
            ("SmolLM2-1.7B + C1", L["arith+func.c1"], L["linalg.c1"]),
            ("SmolLM2-1.7B + C1+C2", L["arith+func.c1_c2"], L["linalg.c1_c2"]),
            (r"\textbf{SmolLM2-1.7B + C1+C2+C3 (ours)}", L["arith+func.c1_c2_c3"], L["linalg.c1_c2_c3"]),
            ("Granite-Code-34B, free", S["granite-free.arith+func"], S["granite-free.linalg"]),
            ("Granite-Code-34B + C1", S["granite-c1.arith+func"], S["granite-c1.linalg"]),
            ("CodeLlama-34B + C1", S["codellama-c1.arith+func"], S["codellama-c1.linalg"])]
    t = [r"\begin{tabular}{l r r}", r"\toprule", r"System & arith+func (n=200) & linalg (n=125) \\", r"\midrule"]
    t += [f"{name} & {fmt(a)} & {fmt(b)} \\\\" for name, a, b in rows]
    t += [r"\bottomrule", r"\end{tabular}"]
    (ASSETS / "table_main.tex").write_text("\n".join(t) + "\n")

    T = res["three_seed"]
    b = [r"\begin{tabular}{l r r r r}", r"\toprule",
         r"System & arith C1-only & arith C1+C3 & linalg C1-only & linalg C1+C3 \\", r"\midrule"]
    for name, key in (("CodeLlama-34B", "codellama"), ("StarCoder2-15B", "starcoder2"), ("Granite-Code-34B", "granite")):
        b.append(f"{name} & {S[f'{key}-c1.arith+func']['all']:.1f} & {T[f'{key}-c1c3.arith+func']['all']:.1f} & "
                 f"{S[f'{key}-c1.linalg']['all']:.1f} & {T[f'{key}-c1c3.linalg']['all']:.1f} \\\\")
    b += [r"\bottomrule", r"\end{tabular}"]
    (ASSETS / "table_best_config.tex").write_text("\n".join(b) + "\n")

    u = [r"\begin{tabular}{l r r r r}", r"\toprule",
         r"Cell & arith+func all & arith+func unique & linalg all & linalg unique \\", r"\midrule"]
    for rung, name in (("none", "SmolLM2 free"), ("c1", "SmolLM2 + C1"), ("c1_c2", "SmolLM2 + C1+C2"),
                       ("c1_c2_c3", "SmolLM2 + C1+C2+C3 (seed-0 ladder)")):
        a, l = L[f"arith+func.{rung}"], L[f"linalg.{rung}"]
        u.append(f"{name} & {a['all']:.1f} & {a['unique']:.1f} & {l['all']:.1f} & {l['unique']:.1f} \\\\")
    u.append(r"\midrule")
    for m, name in (("smollm2-c1c2c3", "SmolLM2 + C1+C2+C3 (3 runs)"), ("codellama-c1c3", "CodeLlama-34B + C1+C3"),
                    ("starcoder2-c1c3", "StarCoder2-15B + C1+C3"), ("granite-c1c3", "Granite-Code-34B + C1+C3"),
                    ("granite-8b-fp16-c1c3", "Granite-Code-8B-fp16 + C1+C3")):
        a, l = T[f"{m}.arith+func"], T[f"{m}.linalg"]
        u.append(f"{name} & {a['all']:.1f} & {a['unique']:.1f} & {l['all']:.1f} & {l['unique']:.1f} \\\\")
    u.append(r"\midrule")
    for m, name in (("codellama-c1", "CodeLlama-34B + C1"), ("starcoder2-c1", "StarCoder2-15B + C1"),
                    ("granite-c1", "Granite-Code-34B + C1"), ("granite-free", "Granite-Code-34B free")):
        a, l = S[f"{m}.arith+func"], S[f"{m}.linalg"]
        u.append(f"{name} & {a['all']:.1f} & {a['unique']:.1f} & {l['all']:.1f} & {l['unique']:.1f} \\\\")
    u += [r"\bottomrule", r"\end{tabular}"]
    (ASSETS / "table_unique_prompt.tex").write_text("\n".join(u) + "\n")
    print(f"wrote {ASSETS.relative_to(REPO)}/table_main.tex, table_best_config.tex, table_unique_prompt.tex")


if __name__ == "__main__":
    sys.exit(main())
