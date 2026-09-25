"""Day-59 (E10b): SmolLM2 constraint ladder on the FROZEN linalg pool.

Context: Day-53/E10 regenerated the arith+func ladder on the frozen released
pool (none 20.0 / C1 44.5 / C1+C2 45.5 / C1+C2+C3 52.0; full stack reproduced
the multi-seed seed-0 cell exactly). The submitted linalg ladder
(49.6 -> 68.0 -> 72.0 -> 72.8, Table 2 / tab:main) is from the same pre-freeze
draft run and disagrees with the multi-seed table's frozen-pool seed-0 cell
(80.0% = 100/125, results/day51_seed0_n200/multiseed_seed0.jsonl). E10b closes
that gap: the full ladder — SmolLM2 x {none, C1, C1+C2, C1+C2+C3} x n=125
linalg — on the frozen released pool.

Reproduction check: the c1_c2_c3 rung
must be compared against 100/125 = 80.0%. The generation logic for that rung
mirrors scripts/day34_multiseed.py::_c1c2c3_mlx EXACTLY (the code path that
produced the 80.0 cell via day51's monkey-patched rerun): attempt 1 greedy on
the C1+C2 grammar, retries at temp=0.8, retry while NOT (parse_valid AND
scope-ok), 5 attempts, max_tokens=600. (Day-53's arith rung retried on the
scope gate only; with a greedy first attempt the two are identical whenever
attempt 1 parses, and Day-53 reproduced 52.0 exactly. E10b uses the day34
condition verbatim so the reproduction check is apples-to-apples.)

Same model (HuggingFaceTB/SmolLM2-1.7B-Instruct, fp16 via mlx-lm), same
grammars (grammar/mlir_gen_c1.lark, grammar/mlir_gen_c1c2.lark), same linalg
few-shot block and chat template as day34/day51 (imported, not copied), same
parse gate and cached verifier. Pool source is
scripts.day18_30b_c3_rejection._prompts_linalg(125) — the canonical frozen
pool used by every day34/day50/day51 linalg cell.

Output: results/day59/e10b_ladder_linalg_frozen.jsonl (one row per (constraint, prompt))
        results/day59/e10b_summary.json               (per-rung rates + CIs + repro check)

Writes NEW files only; no released artifact is modified.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
os.chdir(REPO)
sys.path.insert(0, REPO)
os.environ["PATH"] = f"{REPO}/scripts/env/bin:" + os.environ["PATH"]

import outlines
from mlx_lm import generate as mlx_generate_free
from mlx_lm import load as mlx_load
from mlx_lm.sample_utils import make_sampler

from decoder.c3_scope import accept_or_reject
from grammar.parser import is_parse_valid
from scripts.env.verify_cache import verify

from scripts.day18_30b_c3_rejection import _prompts_linalg
from scripts.day34_multiseed import FEW_SHOT_LINALG_MLX
from scripts.day53_e10_ladder_frozen import _bootstrap_ci

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
N_PROMPTS = 125
C3_MAX_RETRIES = 5
REPRO_TARGET = 100  # multi-seed seed-0 linalg cell: 100/125 = 80.0%

GRAMMARS = {
    "c1":    Path("grammar/mlir_gen_c1.lark").read_text(),
    "c1_c2": Path("grammar/mlir_gen_c1c2.lark").read_text(),
}


def _build_prompt(nl: str) -> str:
    # Byte-identical to day34_multiseed._mlx_prompt(nl, "linalg").
    return (
        "<|im_start|>system\nOutput only valid MLIR. Reuse parameter names exactly.<|im_end|>\n"
        f"<|im_start|>user\n{FEW_SHOT_LINALG_MLX}\n\nNow this task:\nTask: {nl}\nMLIR:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _score(text: str) -> tuple[bool, bool]:
    pv = is_parse_valid(text)
    vv = bool(pv) and verify(text)["returncode"] == 0
    return pv, vv


def run(out_path: Path, summary_path: Path) -> None:
    prompts = _prompts_linalg(N_PROMPTS)
    assert len(prompts) == N_PROMPTS, f"pool size {len(prompts)} != {N_PROMPTS}"
    print(f"[day59] {len(prompts)} prompts (frozen linalg pool)", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[day59] loading {MODEL_ID}", file=sys.stderr)
    model_raw, tokenizer = mlx_load(MODEL_ID)
    mlx_model = outlines.from_mlxlm(model_raw, tokenizer)
    gen_c1    = outlines.Generator(mlx_model, outlines.cfg(GRAMMARS["c1"]))
    gen_c1_c2 = outlines.Generator(mlx_model, outlines.cfg(GRAMMARS["c1_c2"]))

    summary: dict[str, dict] = {}
    t_run = time.perf_counter()
    with out_path.open("w") as fout:
        for constraint in ("none", "c1", "c1_c2", "c1_c2_c3"):
            parse_ok = verify_ok = 0
            attempts_total = 0
            verify_flags: list[bool] = []
            cell_durs: list[float] = []
            for i, nl in enumerate(prompts):
                prompt = _build_prompt(nl)
                t0 = time.perf_counter()
                attempts = 1
                gate_passed_per_try: list[bool] = []
                try:
                    if constraint == "none":
                        out = mlx_generate_free(model_raw, tokenizer, prompt=prompt, max_tokens=600)
                    elif constraint == "c1":
                        out = gen_c1(prompt, max_tokens=600)
                    elif constraint == "c1_c2":
                        out = gen_c1_c2(prompt, max_tokens=600)
                    else:
                        # c1_c2_c3: day34_multiseed._c1c2c3_mlx logic verbatim.
                        # Attempt 1 greedy, retries temp=0.8; retry while NOT
                        # (parse_valid AND scope-ok), including on generation
                        # exceptions (day34 `continue`s to the next attempt);
                        # keep the last output on exhaustion.
                        out, ok = "", False
                        for attempts in range(1, C3_MAX_RETRIES + 1):
                            try:
                                if attempts == 1:
                                    out = gen_c1_c2(prompt, max_tokens=600)
                                else:
                                    out = gen_c1_c2(
                                        prompt,
                                        max_tokens=600,
                                        sampler=make_sampler(temp=0.8),
                                    )
                            except Exception as e:
                                print(f"  [c1_c2_c3] {i} attempt {attempts}: mlx err: {e}",
                                      file=sys.stderr)
                                gate_passed_per_try.append(False)
                                continue
                            ok = is_parse_valid(out) and accept_or_reject(out)[0]
                            gate_passed_per_try.append(ok)
                            if ok:
                                break
                except Exception as e:
                    print(f"  [{constraint}] {i}: gen err: {e}", file=sys.stderr)
                    verify_flags.append(False)
                    continue
                dt = time.perf_counter() - t0
                cell_durs.append(dt)
                attempts_total += attempts
                pv, vv = _score(out)
                if pv: parse_ok += 1
                if vv: verify_ok += 1
                verify_flags.append(vv)
                fout.write(json.dumps({
                    "model": "smollm2-1.7b",
                    "constraint": constraint,
                    "pool": "frozen_released_n125_linalg",
                    "prompt_id": i, "nl": nl, "generated": out,
                    "parse_valid": pv, "verify_valid": vv,
                    "attempts": attempts,
                    "gate_passed_per_try": gate_passed_per_try,
                    "dt": dt,
                }) + "\n")
                fout.flush()
                if (i + 1) % 25 == 0:
                    print(
                        f"  [{constraint}] {i+1}/{len(prompts)} "
                        f"parse={parse_ok}({parse_ok/(i+1):.0%}) "
                        f"verify={verify_ok}({verify_ok/(i+1):.0%}) "
                        f"attempts_avg={attempts_total/(i+1):.2f} "
                        f"elapsed={(time.perf_counter()-t_run)/60:.1f}min",
                        file=sys.stderr,
                    )
            lo, hi = _bootstrap_ci(verify_flags)
            summary[constraint] = {
                "n": len(prompts),
                "parse_valid": parse_ok,
                "verify_valid": verify_ok,
                "verify_rate": verify_ok / len(prompts),
                "verify_ci95": [lo, hi],
                "mean_attempts": attempts_total / max(len(prompts), 1),
                "mean_gen_s": sum(cell_durs) / max(len(cell_durs), 1),
            }
            if constraint == "c1_c2_c3":
                summary["repro_check"] = {
                    "target": f"{REPRO_TARGET}/{N_PROMPTS} = {REPRO_TARGET/N_PROMPTS:.1%} (multi-seed seed-0 cell)",
                    "observed": f"{verify_ok}/{N_PROMPTS} = {verify_ok/N_PROMPTS:.1%}",
                    "exact_match": verify_ok == REPRO_TARGET,
                }
            summary_path.write_text(json.dumps(summary, indent=2))
            print(
                f"[day59] {constraint} done: "
                f"parse={parse_ok}/{len(prompts)}={parse_ok/len(prompts):.1%} "
                f"verify={verify_ok}/{len(prompts)}={verify_ok/len(prompts):.1%} "
                f"CI95=[{lo:.3f},{hi:.3f}] "
                f"mean_attempts={summary[constraint]['mean_attempts']:.2f}",
                file=sys.stderr,
            )
    total = time.perf_counter() - t_run
    rc = summary.get("repro_check", {})
    print(f"\n[day59] REPRO CHECK: target={rc.get('target')} observed={rc.get('observed')} "
          f"exact={rc.get('exact_match')}", file=sys.stderr)
    print(f"[day59] ALL DONE in {total/60:.1f} min -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    run(
        Path("results/day59/e10b_ladder_linalg_frozen.jsonl"),
        Path("results/day59/e10b_summary.json"),
    )
