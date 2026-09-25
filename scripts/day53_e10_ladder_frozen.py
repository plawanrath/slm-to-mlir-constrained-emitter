"""Day-53 (E10): SmolLM2 constraint ladder on the FROZEN released pool.

Context: the submitted ladder (Day-5, results/frozen/week2_gate_day5/c3_smoke.jsonl,
67.5% at C1+C2+C3) predates the MLIR-Spec-150 benchmark freeze; the multi-seed
tables use the released pool (52.0% at seed 0). To report every number on one
locked pool, this script regenerates the
full ladder — SmolLM2 × {none, C1, C1+C2, C1+C2+C3} × n=200 arith+func — on the
frozen released pool, under a protocol byte-identical to Day-5:

  - same model (HuggingFaceTB/SmolLM2-1.7B-Instruct, fp16 via mlx-lm),
  - same grammars (grammar/mlir_gen_c1.lark, grammar/mlir_gen_c1c2.lark),
  - same few-shot prompt + chat template, max_tokens=600,
  - same C3 rejection schedule (attempt 1 greedy, retries temp=0.8, 5 tries),
  - same parse gate (grammar.parser.is_parse_valid) and cached verifier.

The only difference vs Day-5 is the prompt pool revision. Pool source is
scripts.day18_30b_c3_rejection._prompts_arith(200) — the canonical frozen pool
used by every Day-34/51/52 cell (verified byte-identical to Day-5's inline
builder on the current benchmark, and row-identical to
results/day51_seed0_n200/multiseed_seed0.jsonl, 200/200).

Output: results/day53/e10_ladder_frozen.jsonl  (one row per (constraint, prompt))
        results/day53/e10_summary.json         (per-rung rates + bootstrap CIs)

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

from scripts.day18_30b_c3_rejection import _prompts_arith

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

GRAMMARS = {
    "c1":    Path("grammar/mlir_gen_c1.lark").read_text(),
    "c1_c2": Path("grammar/mlir_gen_c1c2.lark").read_text(),
}

# Day-5's few-shot block, verbatim (spaces-around-punctuation format matching
# the LARK generation grammar's explicit WS terminals).
FEW_SHOT = """Example 1:
Task: Write a function that adds two i32 values.
MLIR:
module {
  func.func @f(%a : i32 , %b : i32) -> i32 {
    %0 = arith.addi %a , %b : i32
    return %0 : i32
  }
}

Example 2:
Task: Write a function returning the i32 constant 42.
MLIR:
module {
  func.func @f() -> i32 {
    %0 = arith.constant 42 : i32
    return %0 : i32
  }
}

Example 3:
Task: Load element at index i from a memref.
MLIR:
module {
  func.func @f(%m : memref<?xf32> , %i : index) -> f32 {
    %0 = memref.load %m[%i] : memref<?xf32>
    return %0 : f32
  }
}
"""

N_PROMPTS = 200
C3_MAX_RETRIES = 5


def _build_prompt(nl: str) -> str:
    return (
        "<|im_start|>system\nOutput only valid MLIR. Reuse parameter names exactly.<|im_end|>\n"
        f"<|im_start|>user\n{FEW_SHOT}\n\nNow this task:\nTask: {nl}\nMLIR:<|im_end|>\n<|im_start|>assistant\n"
    )


def _score(text: str) -> tuple[bool, bool]:
    pv = is_parse_valid(text)
    vv = bool(pv) and verify(text)["returncode"] == 0
    return pv, vv


def _bootstrap_ci(flags: list[bool], n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    import random as _random
    rng = _random.Random(seed)
    n = len(flags)
    rates = []
    for _ in range(n_boot):
        s = sum(flags[rng.randrange(n)] for _ in range(n))
        rates.append(s / n)
    rates.sort()
    return rates[int(0.025 * n_boot)], rates[int(0.975 * n_boot)]


def run(out_path: Path, summary_path: Path) -> None:
    prompts = _prompts_arith(N_PROMPTS)
    print(f"[day53] {len(prompts)} prompts (frozen released pool)", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[day53] loading {MODEL_ID}", file=sys.stderr)
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
                scope_passed_per_try: list[bool] = []
                try:
                    if constraint == "none":
                        out = mlx_generate_free(model_raw, tokenizer, prompt=prompt, max_tokens=600)
                    elif constraint == "c1":
                        out = gen_c1(prompt, max_tokens=600)
                    elif constraint == "c1_c2":
                        out = gen_c1_c2(prompt, max_tokens=600)
                    else:
                        # c1_c2_c3: rejection sample against c1_c2 generator.
                        # First try greedy (temp=0), identical to C1+C2 by
                        # construction. Retries use temp=0.8 for diversity.
                        out = gen_c1_c2(prompt, max_tokens=600)
                        ok, _ = accept_or_reject(out)
                        scope_passed_per_try.append(ok)
                        while (not ok) and attempts < C3_MAX_RETRIES:
                            attempts += 1
                            out = gen_c1_c2(
                                prompt,
                                max_tokens=600,
                                sampler=make_sampler(temp=0.8),
                            )
                            ok, _ = accept_or_reject(out)
                            scope_passed_per_try.append(ok)
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
                    "pool": "frozen_released_n200",
                    "prompt_id": i, "nl": nl, "generated": out,
                    "parse_valid": pv, "verify_valid": vv,
                    "attempts": attempts,
                    "scope_passed_per_try": scope_passed_per_try,
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
            summary_path.write_text(json.dumps(summary, indent=2))
            print(
                f"[day53] {constraint} done: "
                f"parse={parse_ok}/{len(prompts)}={parse_ok/len(prompts):.1%} "
                f"verify={verify_ok}/{len(prompts)}={verify_ok/len(prompts):.1%} "
                f"CI95=[{lo:.3f},{hi:.3f}] "
                f"mean_attempts={summary[constraint]['mean_attempts']:.2f}",
                file=sys.stderr,
            )
    total = time.perf_counter() - t_run
    print(f"\n[day53] ALL DONE in {total/60:.1f} min → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    run(
        Path("results/day53/e10_ladder_frozen.jsonl"),
        Path("results/day53/e10_summary.json"),
    )
