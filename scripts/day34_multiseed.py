"""Day-34: Multi-seed re-runs on 5 critical cells for tighter CIs (ADR-0008).

Runs seeds 1, 2 (seed 0 already done in earlier days) × n=100 on:

  1. SmolLM2 + C1+C2+C3 × arith+func  (main SLM cell, arith+func)
  2. SmolLM2 + C1+C2+C3 × linalg      (main SLM cell, linalg)
  3. Granite-34B + C1+C3 × arith+func (+22pp C3 gain — biggest positive)
  4. CodeLlama-34B + C1+C3 × arith+func (-21.5pp C3 drop — most surprising)
  5. StarCoder2-15B:instruct + C1+C3 × arith+func (closest competitor)

Output: results/day34/multiseed.jsonl

Rationale: the paper's primary tables are seed-0 only. Two additional
seeds give each cell 3 independent samples; we then report mean ± half-
range (or across-seed bootstrap CI) as the "multi-seed" tightening row.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
os.chdir(REPO); sys.path.insert(0, REPO)
os.environ["PATH"] = f"{REPO}/scripts/env/bin:" + os.environ["PATH"]

from decoder.c3_scope import accept_or_reject
from eval.baselines.run_ollama import ollama_generate_raw
from grammar.parser import is_parse_valid
from scripts.env.verify_cache import verify

from scripts.day18_30b_c3_rejection import (
    FEW_SHOT_ARITH, FEW_SHOT_LINALG, TARGET_LINALG_OPS,
    _prompts_arith as _prompts_arith_day18,
    _prompts_linalg as _prompts_linalg_day18,
    _build_prompt_arith, _build_prompt_linalg,
)
from scripts.day19_starcoder2_baseline import _extract_mlir

# Day-5's few-shot template uses the spaces-around-punctuation format that
# matches our LARK generation grammar (explicit WS terminals). Using Day-18's
# no-space format confuses the CFG-constrained model (observed 67.5%→31%
# verify drop). MLX+CFG cells use this template; Ollama cells use Day-18's.
FEW_SHOT_ARITH_MLX = """Example 1:
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

FEW_SHOT_LINALG_MLX = """Example 1:
Task: Perform matmul of two 2-D f32 memrefs into an output memref.
MLIR:
module {
  func.func @mm(%A : memref<?x?xf32> , %B : memref<?x?xf32> , %C : memref<?x?xf32>) {
    linalg.matmul ins(%A , %B : memref<?x?xf32>, memref<?x?xf32>) outs(%C : memref<?x?xf32>)
    return
  }
}

Example 2:
Task: Fill a 1-D f32 memref with a given f32 value.
MLIR:
module {
  func.func @fill(%v : f32 , %m : memref<?xf32>) {
    linalg.fill ins(%v : f32) outs(%m : memref<?xf32>)
    return
  }
}

Example 3:
Task: Apply elementwise exp to a 1-D f32 memref.
MLIR:
module {
  func.func @ex(%x : memref<?xf32> , %y : memref<?xf32>) {
    linalg.exp ins(%x : memref<?xf32>) outs(%y : memref<?xf32>)
    return
  }
}
"""

N_PER_CELL = 100
SEEDS = [1, 2]   # seed 0 already done; add 1 and 2 for multi-seed
N_RETRIES = 5


# 5 critical cells (cell_id, model_tag, dialect, protocol)
#   protocol: "ollama-c1c3" (rejection), "mlx-c1c2c3" (inline for SmolLM2)
CELLS = [
    ("smollm2-c1c2c3",    "HuggingFaceTB/SmolLM2-1.7B-Instruct",   "arith+func", "mlx-c1c2c3"),
    ("smollm2-c1c2c3",    "HuggingFaceTB/SmolLM2-1.7B-Instruct",   "linalg",     "mlx-c1c2c3"),
    ("granite-c1c3",      "granite-code:34b-instruct-q4_K_M",       "arith+func", "ollama-c1c3"),
    ("codellama-c1c3",    "codellama:34b-instruct-q4_K_M",          "arith+func", "ollama-c1c3"),
    ("starcoder2-c1c3",   "starcoder2:instruct",                     "arith+func", "ollama-c1c3"),
]


def _c1c3_ollama(prompt: str, model: str, seed: int, extract: bool) -> tuple[str, int]:
    last = ""
    for attempt in range(1, N_RETRIES + 1):
        s = seed + (attempt - 1) * 1000
        raw, _ = ollama_generate_raw(
            prompt=prompt, model=model, max_tokens=256,
            temperature=0.3 if attempt > 1 else 0.2, seed=s,
        )
        last = raw
        text = _extract_mlir(raw) if extract else raw
        if not is_parse_valid(text): continue
        ok, _rep = accept_or_reject(text)
        if ok: return raw, attempt
    return last, N_RETRIES


# MLX + Outlines path — loaded lazily on first use (slow).
_MLX_CACHE: dict[str, object] = {}


def _mlx_generators(model_id: str) -> object:
    if model_id in _MLX_CACHE:
        return _MLX_CACHE[model_id]
    import outlines
    from mlx_lm import load as mlx_load
    model_raw, tokenizer = mlx_load(model_id)
    mlx_model = outlines.from_mlxlm(model_raw, tokenizer)
    c1c2 = outlines.Generator(
        mlx_model,
        outlines.cfg(Path("grammar/mlir_gen_c1c2.lark").read_text()),
    )
    _MLX_CACHE[model_id] = (mlx_model, c1c2)
    return _MLX_CACHE[model_id]


def _mlx_prompt(nl: str, dialect: str) -> str:
    few_shot = FEW_SHOT_LINALG_MLX if dialect == "linalg" else FEW_SHOT_ARITH_MLX
    return (
        "<|im_start|>system\nOutput only valid MLIR. Reuse parameter names exactly.<|im_end|>\n"
        f"<|im_start|>user\n{few_shot}\n\nNow this task:\nTask: {nl}\nMLIR:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _c1c2c3_mlx(prompt_nl: str, model: str, seed: int, dialect: str) -> tuple[str, int]:
    """SmolLM2 + C1+C2+C3 via Outlines: 5-retry rejection with temp=0.0 first,
    temp=0.8 retries for diversity. Seed parameter is reserved for future
    Outlines extension; current impl uses mlx_lm's default sampling."""
    from mlx_lm.sample_utils import make_sampler
    _, gen_c1c2 = _mlx_generators(model)
    prompt = _mlx_prompt(prompt_nl, dialect)
    last = ""
    for attempt in range(1, N_RETRIES + 1):
        try:
            if attempt == 1:
                last = gen_c1c2(prompt, max_tokens=600)
            else:
                last = gen_c1c2(
                    prompt, max_tokens=600,
                    sampler=make_sampler(temp=0.8),
                )
        except Exception as e:
            print(f"    mlx err: {e}", file=sys.stderr)
            continue
        if not is_parse_valid(last): continue
        ok, _rep = accept_or_reject(last)
        if ok: return last, attempt
    return last, N_RETRIES


def run(out_path: Path) -> None:
    arith_prompts  = _prompts_arith_day18(N_PER_CELL)
    linalg_prompts = _prompts_linalg_day18(N_PER_CELL)
    print(f"[day34] arith={len(arith_prompts)}  linalg={len(linalg_prompts)}  seeds={SEEDS}",
          file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0_all = time.perf_counter()
    with out_path.open("w") as f:
        for nick, model, dialect, protocol in CELLS:
            prompts = arith_prompts if dialect == "arith+func" else linalg_prompts
            build = _build_prompt_arith if dialect == "arith+func" else _build_prompt_linalg
            is_starcoder2 = "starcoder2" in nick
            for seed in SEEDS:
                print(f"\n[day34] {nick} × {dialect} × seed={seed}", file=sys.stderr)
                parse_ok = verify_ok = 0
                attempts_total = 0
                for i, nl in enumerate(prompts):
                    prompt = build(nl)
                    t0 = time.perf_counter()
                    try:
                        if protocol == "mlx-c1c2c3":
                            raw, attempts = _c1c2c3_mlx(nl, model, seed, dialect)
                            out = raw
                        else:
                            raw, attempts = _c1c3_ollama(prompt, model, seed, extract=is_starcoder2)
                            out = _extract_mlir(raw) if is_starcoder2 else raw
                    except Exception as e:
                        print(f"  [{nick}/s{seed}] {i}: err: {e}", file=sys.stderr); continue
                    dt = time.perf_counter() - t0
                    attempts_total += attempts
                    pv = is_parse_valid(out); vv = False
                    if pv:
                        parse_ok += 1
                        vv = verify(out)["returncode"] == 0
                        if vv: verify_ok += 1
                    f.write(json.dumps({
                        "model": nick, "backend": protocol, "constraint": "c1_c3",
                        "dialect": dialect, "seed": seed,
                        "prompt_id": i, "nl": nl,
                        "generated": out, "parse_valid": pv, "verify_valid": vv,
                        "attempts": attempts, "dt": dt,
                    }) + "\n")
                    f.flush()
                    if (i + 1) % 20 == 0:
                        print(
                            f"  [{nick}/s{seed}/{dialect}] {i+1}/{len(prompts)} "
                            f"parse={parse_ok}({parse_ok/(i+1):.0%}) "
                            f"verify={verify_ok}({verify_ok/(i+1):.0%}) "
                            f"att_avg={attempts_total/(i+1):.2f} "
                            f"elapsed={(time.perf_counter()-t0_all)/60:.1f}min",
                            file=sys.stderr,
                        )
                print(
                    f"[day34] {nick}/s{seed}/{dialect} done: "
                    f"parse={parse_ok}/{len(prompts)}={parse_ok/len(prompts):.1%} "
                    f"verify={verify_ok}/{len(prompts)}={verify_ok/len(prompts):.1%}",
                    file=sys.stderr,
                )
    total = time.perf_counter() - t0_all
    print(f"\n[day34] ALL DONE in {total/60:.1f} min → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    run(Path("results/day34/multiseed.jsonl"))
