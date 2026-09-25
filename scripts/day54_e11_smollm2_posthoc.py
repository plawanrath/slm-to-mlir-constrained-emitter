"""Day-54 (E11): SmolLM2 under the baseline post-hoc protocol + token cap.

Context: the SLM runs token-level C1+C2 masked decoding while the 15B/34B
baselines run free decode with post-hoc rejection, and max_tokens differs
(600 vs 256). llama.cpp
cannot run our masks, so the converse cell is the informative one: run SmolLM2
(the one model supporting both) under the baselines' protocol and quantify the
masked-decoding advantage directly.

Three cells, all on the frozen released pool (n=200 arith+func, identical to
Day-53; pool source scripts.day18_30b_c3_rejection._prompts_arith(200)):

  1. posthoc-c1c3-600: free decode, max_tokens=600, attempt 1 greedy, retries
     temp=0.8, gates = C1 parse filter + C3 scope, 5 tries. Identical to the
     Day-53 c1_c2_c3 rung EXCEPT generation is unmasked. Isolates token-level
     masking as the single variable (vs Day-53's 52.0%).
  2. posthoc-c1c3-256: same gates, but the literal baseline budget and
     schedule: max_tokens=256, temp=0.2 attempt 1, temp=0.3 retries
     (scripts/day18_30b_c3_rejection._c1c3_rejection_sample). SmolLM2 under
     exactly the baseline protocol.
  3. free-256: single greedy free decode at max_tokens=256, no retries. The
     token-limit mirror of the Day-53 'none' rung (600): isolates the cap on
     the SLM side.

Scoring matches Day-53: raw output (no extraction, matching the
CodeLlama/Granite handling) -> grammar.parser.is_parse_valid -> cached
mlir-opt verify.

Output: results/day54/e11_smollm2_posthoc.jsonl (one row per (cell, prompt))
        results/day54/e11_summary.json         (per-cell rates + CIs)

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

from mlx_lm import generate as mlx_generate_free
from mlx_lm import load as mlx_load
from mlx_lm.sample_utils import make_sampler

from decoder.c3_scope import accept_or_reject
from grammar.parser import is_parse_valid
from scripts.env.verify_cache import verify

from scripts.day18_30b_c3_rejection import _prompts_arith

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Day-5/Day-53 few-shot block, verbatim.
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
N_RETRIES = 5

# (cell_id, max_tokens, retries?, temp_first, temp_retry)
CELLS = [
    ("posthoc-c1c3-600", 600, True,  0.0, 0.8),
    ("posthoc-c1c3-256", 256, True,  0.2, 0.3),
    ("free-256",         256, False, 0.0, None),
]


def _build_prompt(nl: str) -> str:
    return (
        "<|im_start|>system\nOutput only valid MLIR. Reuse parameter names exactly.<|im_end|>\n"
        f"<|im_start|>user\n{FEW_SHOT}\n\nNow this task:\nTask: {nl}\nMLIR:<|im_end|>\n<|im_start|>assistant\n"
    )


def _free_gen(model_raw, tokenizer, prompt: str, max_tokens: int, temp: float) -> str:
    if temp == 0.0:
        return mlx_generate_free(model_raw, tokenizer, prompt=prompt, max_tokens=max_tokens)
    return mlx_generate_free(
        model_raw, tokenizer, prompt=prompt, max_tokens=max_tokens,
        sampler=make_sampler(temp=temp),
    )


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
    print(f"[day54] {len(prompts)} prompts (frozen released pool)", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[day54] loading {MODEL_ID}", file=sys.stderr)
    model_raw, tokenizer = mlx_load(MODEL_ID)

    summary: dict[str, dict] = {}
    t_run = time.perf_counter()
    with out_path.open("w") as fout:
        for cell_id, max_tokens, retries, temp_first, temp_retry in CELLS:
            parse_ok = verify_ok = 0
            attempts_total = 0
            verify_flags: list[bool] = []
            cell_durs: list[float] = []
            for i, nl in enumerate(prompts):
                prompt = _build_prompt(nl)
                t0 = time.perf_counter()
                attempts = 1
                gates_passed_per_try: list[bool] = []
                try:
                    out = _free_gen(model_raw, tokenizer, prompt, max_tokens, temp_first)
                    if retries:
                        ok = is_parse_valid(out) and accept_or_reject(out)[0]
                        gates_passed_per_try.append(ok)
                        while (not ok) and attempts < N_RETRIES:
                            attempts += 1
                            out = _free_gen(model_raw, tokenizer, prompt, max_tokens, temp_retry)
                            ok = is_parse_valid(out) and accept_or_reject(out)[0]
                            gates_passed_per_try.append(ok)
                except Exception as e:
                    print(f"  [{cell_id}] {i}: gen err: {e}", file=sys.stderr)
                    verify_flags.append(False)
                    continue
                dt = time.perf_counter() - t0
                cell_durs.append(dt)
                attempts_total += attempts
                pv = is_parse_valid(out)
                vv = bool(pv) and verify(out)["returncode"] == 0
                if pv: parse_ok += 1
                if vv: verify_ok += 1
                verify_flags.append(vv)
                fout.write(json.dumps({
                    "model": "smollm2-1.7b",
                    "cell": cell_id,
                    "protocol": "free+posthoc-c1c3" if retries else "free",
                    "max_tokens": max_tokens,
                    "pool": "frozen_released_n200",
                    "prompt_id": i, "nl": nl, "generated": out,
                    "parse_valid": pv, "verify_valid": vv,
                    "attempts": attempts,
                    "gates_passed_per_try": gates_passed_per_try,
                    "dt": dt,
                }) + "\n")
                fout.flush()
                if (i + 1) % 25 == 0:
                    print(
                        f"  [{cell_id}] {i+1}/{len(prompts)} "
                        f"parse={parse_ok}({parse_ok/(i+1):.0%}) "
                        f"verify={verify_ok}({verify_ok/(i+1):.0%}) "
                        f"attempts_avg={attempts_total/(i+1):.2f} "
                        f"elapsed={(time.perf_counter()-t_run)/60:.1f}min",
                        file=sys.stderr,
                    )
            lo, hi = _bootstrap_ci(verify_flags)
            summary[cell_id] = {
                "n": len(prompts),
                "max_tokens": max_tokens,
                "parse_valid": parse_ok,
                "verify_valid": verify_ok,
                "verify_rate": verify_ok / len(prompts),
                "verify_ci95": [lo, hi],
                "mean_attempts": attempts_total / max(len(prompts), 1),
                "mean_gen_s": sum(cell_durs) / max(len(cell_durs), 1),
            }
            summary_path.write_text(json.dumps(summary, indent=2))
            print(
                f"[day54] {cell_id} done: "
                f"parse={parse_ok}/{len(prompts)}={parse_ok/len(prompts):.1%} "
                f"verify={verify_ok}/{len(prompts)}={verify_ok/len(prompts):.1%} "
                f"CI95=[{lo:.3f},{hi:.3f}] "
                f"mean_attempts={summary[cell_id]['mean_attempts']:.2f}",
                file=sys.stderr,
            )
    total = time.perf_counter() - t_run
    print(f"\n[day54] ALL DONE in {total/60:.1f} min → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    run(
        Path("results/day54/e11_smollm2_posthoc.jsonl"),
        Path("results/day54/e11_summary.json"),
    )
