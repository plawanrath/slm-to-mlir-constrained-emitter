"""Day-4 §3.2 SLM smoke matrix:  {Phi, SmolLM2} × {none, C1, C1+C2} × arith+func.

Uses few-shot prompts (3 in-context examples) for consistency across conditions.
Writes results/day4/slm_smoke.jsonl with one row per (model, constraint, prompt).
"""
from __future__ import annotations

import json
import os
import random
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

from grammar.parser import is_parse_valid
from scripts.env.verify_cache import verify


MODELS = [
    ("phi-3.5-mini",  "microsoft/Phi-3.5-mini-instruct"),
    # SmolLM2 runs after Phi to keep a single model loaded in memory.
    ("smollm2-1.7b",  "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
]

GRAMMARS = {
    "c1":    Path("grammar/mlir_gen_c1.lark").read_text(),
    "c1_c2": Path("grammar/mlir_gen_c1c2.lark").read_text(),
}

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

N_PROMPTS = 200  # smoke: 200 prompts per cell


def _build_prompt(nl: str, model_family: str) -> str:
    if model_family == "phi":
        return (
            "<|system|>\nOutput only valid MLIR. Reuse parameter names exactly.\n<|end|>\n"
            f"<|user|>\n{FEW_SHOT}\n\nNow this task:\nTask: {nl}\nMLIR:\n<|end|>\n<|assistant|>\n"
        )
    if model_family == "smollm":
        return (
            "<|im_start|>system\nOutput only valid MLIR. Reuse parameter names exactly.<|im_end|>\n"
            f"<|im_start|>user\n{FEW_SHOT}\n\nNow this task:\nTask: {nl}\nMLIR:<|im_end|>\n<|im_start|>assistant\n"
        )
    # fallback
    return f"{FEW_SHOT}\n\nTask: {nl}\nMLIR:\n"


def _load_prompts() -> list[str]:
    """200 prompts: 15 from MLIR-Spec seeds + 185 from L3 weak_nl (shuffled)."""
    prompts: list[str] = []
    for p in sorted(Path("eval/benchmarks/mlir_spec_150/examples").glob("*.json")):
        prompts.append(json.loads(p.read_text())["nl"])
    lines = Path("data/processed/l3_tests.jsonl").read_text().splitlines()
    rng = random.Random(0); rng.shuffle(lines)
    for l in lines:
        if len(prompts) >= N_PROMPTS: break
        r = json.loads(l); nl = r.get("weak_nl", "").strip()
        if nl: prompts.append(nl)
    return prompts


def run(out_path: Path) -> None:
    prompts = _load_prompts()
    print(f"[smoke] {len(prompts)} prompts", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_run = time.perf_counter()
    with out_path.open("w") as fout:
        for nick, model_id in MODELS:
            family = "phi" if "Phi" in model_id else "smollm"
            print(f"\n[smoke] loading {model_id}", file=sys.stderr)
            model_raw, tokenizer = mlx_load(model_id)
            mlx_model = outlines.from_mlxlm(model_raw, tokenizer)

            # Precompute constrained generators per grammar
            generators = {
                "c1":    outlines.Generator(mlx_model, outlines.cfg(GRAMMARS["c1"])),
                "c1_c2": outlines.Generator(mlx_model, outlines.cfg(GRAMMARS["c1_c2"])),
            }
            for constraint in ("none", "c1", "c1_c2"):
                parse_ok = verify_ok = 0
                cell_durs = []
                for i, nl in enumerate(prompts):
                    prompt = _build_prompt(nl, family)
                    t0 = time.perf_counter()
                    try:
                        if constraint == "none":
                            out = mlx_generate_free(
                                model_raw, tokenizer, prompt=prompt, max_tokens=600,
                            )
                        else:
                            out = generators[constraint](prompt, max_tokens=600)
                    except Exception as e:
                        print(f"  [{nick}/{constraint}] {i}: gen err: {e}", file=sys.stderr)
                        continue
                    dt = time.perf_counter() - t0
                    cell_durs.append(dt)
                    pv = is_parse_valid(out)
                    vv = False
                    if pv:
                        parse_ok += 1
                        vv = verify(out)["returncode"] == 0
                        if vv: verify_ok += 1
                    fout.write(json.dumps({
                        "model": nick, "constraint": constraint, "prompt_id": i,
                        "nl": nl, "generated": out,
                        "parse_valid": pv, "verify_valid": vv, "dt": dt,
                    }) + "\n")
                    fout.flush()
                    if (i + 1) % 25 == 0:
                        print(
                            f"  [{nick}/{constraint}] {i+1}/{len(prompts)} "
                            f"parse={parse_ok}({parse_ok/(i+1):.0%}) "
                            f"verify={verify_ok}({verify_ok/(i+1):.0%})",
                            file=sys.stderr,
                        )
                print(
                    f"[smoke] {nick} / {constraint} done: "
                    f"parse={parse_ok}/{len(prompts)}={parse_ok/len(prompts):.1%} "
                    f"verify={verify_ok}/{len(prompts)}={verify_ok/len(prompts):.1%}",
                    file=sys.stderr,
                )
    total = time.perf_counter() - t_run
    print(f"\n[smoke] ALL DONE in {total/60:.1f} min → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    out = Path("results/day4/slm_smoke.jsonl")
    run(out)
