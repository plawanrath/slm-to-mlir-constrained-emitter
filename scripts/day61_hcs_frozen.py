"""Day-61: Hidden Cost of Structure (HCS) replication on the frozen released pool.

Context (camera-ready, NeurIPS 2026 E&D). The submitted HCS replication (Figure 8,
Ablations) came from the Day-4/5 smoke runs, whose arith+func prompt set was a
pre-freeze draft (15 spec prompts + 185 mined prompts). The camera-ready reports
every comparison on the frozen released pool, so the replication is regenerated
there:

  * SmolLM2-1.7B: the frozen Day-53 ladder already contains the HCS cells (free,
    C1, C1+C2) under the Day-4 protocol (same few-shot block, grammars, chat
    template, max_tokens=600; verified identical in code). No new inference.
  * Phi-3.5-mini-instruct (revision 2fe192450127e6a83f7441aef6e3ca586c338b77, the
    only revision since 2025-12-10): the Day-4 Phi protocol verbatim
    (scripts/day4_slm_smoke._build_prompt(nl, "phi"), same grammars and
    generators), free / C1 / C1+C2 on the frozen n=200 arith+func pool.

Gates before any Phi output is used:
  1. MLX reproduction: regenerate the first N frozen SmolLM2 C1 outputs
     (results/day53/e10_ladder_frozen.jsonl) and require byte-identical text.
  2. Verifier: re-verify cached July generations through the live mlir-opt
     wrapper (container from SLM_MLIR_CONTAINER) and require identical output.
After the run, a Day-4 identity check compares the new Phi outputs with the
original Day-4 Phi outputs on the prompts both pools share (identical outputs
and verify outcomes confirm the same model, protocol and runtime).

Statistics: eval.ablations.hcs_replication.run (paired bootstrap, 10,000
resamples), the same code that produced the submitted HCS numbers.

Output: results/day61/hcs_phi_frozen.jsonl, results/day61/hcs_frozen_summary.json,
        results/day61/paper_assets/fig8_hcs_null.pdf
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
os.environ["PATH"] = f"{REPO}/scripts/env/bin:" + os.environ["PATH"]

PHI_ID = "microsoft/Phi-3.5-mini-instruct"
PHI_REV = "2fe192450127e6a83f7441aef6e3ca586c338b77"
SMOL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
OUT = REPO / "results/day61"
RUNGS = ("none", "c1", "c1_c2")


def _generators(model_id: str, revision: str | None = None):
    import outlines
    from mlx_lm import load as mlx_load
    from huggingface_hub import snapshot_download
    path = snapshot_download(model_id, revision=revision) if revision else model_id
    model_raw, tokenizer = mlx_load(path)
    mlx_model = outlines.from_mlxlm(model_raw, tokenizer)
    from scripts.day4_slm_smoke import GRAMMARS
    gens = {c: outlines.Generator(mlx_model, outlines.cfg(GRAMMARS[c])) for c in ("c1", "c1_c2")}
    return model_raw, tokenizer, gens


def _generate(model_raw, tokenizer, gens, prompt: str, rung: str) -> str:
    from mlx_lm import generate as mlx_generate_free
    if rung == "none":
        return mlx_generate_free(model_raw, tokenizer, prompt=prompt, max_tokens=600)
    return gens[rung](prompt, max_tokens=600)


def mlx_repro_check(n: int) -> dict:
    from scripts.day53_e10_ladder_frozen import _build_prompt
    rows = sorted((json.loads(l) for l in open("results/day53/e10_ladder_frozen.jsonl")
                   if '"constraint": "c1"' in l), key=lambda r: int(r["prompt_id"]))
    model_raw, tokenizer, gens = _generators(SMOL_ID)
    same = sum(_generate(model_raw, tokenizer, gens, _build_prompt(r["nl"]), "c1") == r["generated"]
               for r in rows[:n])
    print(f"[day61-hcs] MLX repro check: {same}/{n} byte-identical to frozen SmolLM2 C1 outputs", file=sys.stderr)
    del model_raw, gens
    return {"n": n, "byte_identical": same}


def run_phi(out_path: Path) -> None:
    from grammar.parser import is_parse_valid
    from scripts.day4_slm_smoke import _build_prompt
    from scripts.day18_30b_c3_rejection import _prompts_arith
    from scripts.env.verify_cache import verify
    prompts = _prompts_arith(200)
    done = set()
    if out_path.exists():
        done = {(r["constraint"], r["prompt_id"]) for r in map(json.loads, out_path.read_text().splitlines())}
    model_raw, tokenizer, gens = _generators(PHI_ID, PHI_REV)
    t0_all = time.perf_counter()
    with out_path.open("a") as f:
        for rung in RUNGS:
            for i, nl in enumerate(prompts):
                if (rung, i) in done:
                    continue
                t0 = time.perf_counter()
                out = _generate(model_raw, tokenizer, gens, _build_prompt(nl, "phi"), rung)
                pv = is_parse_valid(out)
                vv = bool(pv) and verify(out)["returncode"] == 0
                f.write(json.dumps({"model": "phi-3.5-mini", "revision": PHI_REV, "constraint": rung,
                                    "dialect": "arith+func", "seed": 0, "pool": "frozen_released",
                                    "prompt_id": i, "nl": nl, "generated": out, "parse_valid": pv,
                                    "verify_valid": vv, "dt": time.perf_counter() - t0}) + "\n")
                f.flush()
                if (i + 1) % 25 == 0:
                    print(f"  [phi/{rung}] {i+1}/200 elapsed={(time.perf_counter()-t0_all)/60:.1f}min", file=sys.stderr)


def day4_identity_check() -> dict:
    old = [json.loads(l) for l in open("results/day4/slm_smoke.jsonl") if '"phi-3.5-mini"' in l]
    new = [json.loads(l) for l in open(OUT / "hcs_phi_frozen.jsonl")]
    res = {}
    for rung in RUNGS:
        o, n = {}, {}
        for r in old:
            if r["constraint"] == rung:
                o.setdefault(r["nl"], r)
        for r in new:
            if r["constraint"] == rung:
                n.setdefault(r["nl"], r)
        shared = sorted(set(o) & set(n))
        res[rung] = {"shared": len(shared),
                     "byte_identical": sum(o[k]["generated"] == n[k]["generated"] for k in shared),
                     "same_verify_outcome": sum(bool(o[k]["verify_valid"]) == bool(n[k]["verify_valid"]) for k in shared)}
    print(f"[day61-hcs] Day-4 identity check (Phi, shared prompts): {res}", file=sys.stderr)
    return res


def hcs_stats() -> dict:
    from eval.ablations.hcs_replication import run as hcs_run
    rows = []
    for r in map(json.loads, open("results/day53/e10_ladder_frozen.jsonl")):
        if r["constraint"] in RUNGS:
            rows.append({"model": "smollm2-1.7b", "dialect": "arith+func", "constraint": r["constraint"],
                         "prompt_id": r["prompt_id"], "seed": 0, "passed": bool(r["verify_valid"])})
    for r in map(json.loads, open(OUT / "hcs_phi_frozen.jsonl")):
        rows.append({"model": "phi-3.5-mini", "dialect": "arith+func", "constraint": r["constraint"],
                     "prompt_id": r["prompt_id"], "seed": 0, "passed": bool(r["verify_valid"])})
    return {m: hcs_run(rows, m, "arith+func") for m in ("smollm2-1.7b", "phi-3.5-mini")}


def figure(stats: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})
    labels = ["free", "+C1", "+C1+C2"]
    x, w = np.arange(3), 0.35
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for off, m, name, col in ((-w / 2, "smollm2-1.7b", "SmolLM2-1.7B", "#1f77b4"),
                              (w / 2, "phi-3.5-mini", "Phi-3.5-mini", "#8ecae6")):
        vals = [100 * stats[m]["pass_rates"][k] for k in ("base", "c1_only", "c1_c2")]
        ax.bar(x + off, vals, w, color=col, edgecolor="black", linewidth=0.5, label=name)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("verify-valid %")
    ax.set_ylim(0, 80)
    rev = any(stats[m]["reversal_detected"] for m in stats)
    ax.set_title("Hidden Cost of Structure replication on the frozen released pool\n"
                 + ("a reversal is observed" if rev else "C1 does not hurt either SLM under few-shot priming"))
    ax.legend(framealpha=0.95, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro-n", type=int, default=20)
    args = ap.parse_args()
    summ_path = OUT / "hcs_frozen_summary.json"
    summ = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    if "mlx_repro_check" not in summ:
        summ["mlx_repro_check"] = mlx_repro_check(args.repro_n)
        summ_path.write_text(json.dumps(summ, indent=2))
    if summ["mlx_repro_check"]["byte_identical"] < summ["mlx_repro_check"]["n"]:
        print("[day61-hcs] STOP: MLX stack does not reproduce the frozen outputs", file=sys.stderr)
        return 2
    from scripts.day61_frozen_c1only_baselines import verifier_check
    summ["verifier_check"] = verifier_check()
    summ_path.write_text(json.dumps(summ, indent=2))
    if summ["verifier_check"]["identical"] < summ["verifier_check"]["n"]:
        print("[day61-hcs] STOP: verifier differs from the pinned July verifier", file=sys.stderr)
        return 3
    run_phi(OUT / "hcs_phi_frozen.jsonl")
    summ["day4_identity_check"] = day4_identity_check()
    stats = hcs_stats()
    summ["hcs"] = stats
    summ_path.write_text(json.dumps(summ, indent=2))
    figure(stats, OUT / "paper_assets/fig8_hcs_null.pdf")
    for m, s in stats.items():
        d = s["c1_vs_base_paired_diff"]
        print(f"[day61-hcs] {m}: {', '.join(f'{k} {100*v:.1f}' for k, v in s['pass_rates'].items())}; "
              f"C1 - free {100*d['point']:+.1f}pp CI [{100*d['ci_low']:+.1f}, {100*d['ci_high']:+.1f}] p={d['p_value']:.4f}; "
              f"reversal={s['reversal_detected']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
