"""Day-9 §7.4: SmolLM2 × {none, C1, C1+C2, C1+C2+C3} × linalg ×
(Linalg-Spec-30 + L3 linalg-subset held-out).

Same rejection-sampling pattern as scripts/day5_c3_smoke.py. Linalg few-shot
examples substitute the arith/memref ones. C3 validator now includes the
12 linalg op handlers (ADR-0007).
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path

REPO = "/Users/plawanrath/Documents/GitHub-public/slm-to-mlir-constrained-emitter"
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


MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

GRAMMARS = {
    "c1":    Path("grammar/mlir_gen_c1.lark").read_text(),
    "c1_c2": Path("grammar/mlir_gen_c1c2.lark").read_text(),
}

# Linalg-specific few-shot. All three examples use memref semantics.
FEW_SHOT = """Example 1:
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

TARGET_OPS = {"matmul", "matvec", "fill", "copy", "transpose", "broadcast",
              "add", "sub", "mul", "div", "exp", "abs"}
N_PROMPTS = 200   # Linalg-Spec-30 + 170 L3 linalg-subset held-out
C3_MAX_RETRIES = 5


def _build_prompt(nl: str) -> str:
    return (
        "<|im_start|>system\nOutput only valid MLIR using the linalg dialect with memref semantics. Reuse parameter names exactly.<|im_end|>\n"
        f"<|im_start|>user\n{FEW_SHOT}\n\nNow this task:\nTask: {nl}\nMLIR:<|im_end|>\n<|im_start|>assistant\n"
    )


def _load_prompts() -> list[str]:
    prompts: list[str] = []
    # Linalg-Spec-30 seeds first.
    for p in sorted(Path("eval/benchmarks/linalg_spec_30/examples").glob("*.json")):
        prompts.append(json.loads(p.read_text())["nl"])
    # Fill with L3 linalg subset whose ops are all in TARGET_OPS + memref form.
    lines = Path("data/processed/l3_tests.jsonl").read_text().splitlines()
    rng = random.Random(0); rng.shuffle(lines)
    for l in lines:
        if len(prompts) >= N_PROMPTS: break
        r = json.loads(l)
        if "linalg." not in r.get("mlir", ""): continue
        ops = set(m.group(1) for m in re.finditer(r"linalg\.(\w+)", r["mlir"]))
        if not ops or not (ops <= TARGET_OPS): continue
        if "tensor<" in r["mlir"]: continue
        nl = r.get("weak_nl", "").strip()
        if not nl: continue
        prompts.append(nl)
    return prompts


def _score(text: str) -> tuple[bool, bool]:
    pv = is_parse_valid(text)
    vv = bool(pv) and verify(text)["returncode"] == 0
    return pv, vv


def run(out_path: Path) -> None:
    prompts = _load_prompts()
    print(f"[day9] {len(prompts)} prompts", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[day9] loading {MODEL_ID}", file=sys.stderr)
    model_raw, tokenizer = mlx_load(MODEL_ID)
    mlx_model = outlines.from_mlxlm(model_raw, tokenizer)
    gen_c1    = outlines.Generator(mlx_model, outlines.cfg(GRAMMARS["c1"]))
    gen_c1_c2 = outlines.Generator(mlx_model, outlines.cfg(GRAMMARS["c1_c2"]))

    t_run = time.perf_counter()
    with out_path.open("w") as fout:
        for constraint in ("none", "c1", "c1_c2", "c1_c2_c3"):
            parse_ok = verify_ok = 0
            attempts_total = 0
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
                        out = gen_c1_c2(prompt, max_tokens=600)
                        ok, _ = accept_or_reject(out)
                        scope_passed_per_try.append(ok)
                        while (not ok) and attempts < C3_MAX_RETRIES:
                            attempts += 1
                            out = gen_c1_c2(prompt, max_tokens=600,
                                            sampler=make_sampler(temp=0.8))
                            ok, _ = accept_or_reject(out)
                            scope_passed_per_try.append(ok)
                except Exception as e:
                    print(f"  [{constraint}] {i}: gen err: {e}", file=sys.stderr)
                    continue
                dt = time.perf_counter() - t0
                cell_durs.append(dt)
                attempts_total += attempts
                pv, vv = _score(out)
                if pv: parse_ok += 1
                if vv: verify_ok += 1
                fout.write(json.dumps({
                    "model": "smollm2-1.7b",
                    "constraint": constraint,
                    "prompt_id": i, "nl": nl, "generated": out,
                    "parse_valid": pv, "verify_valid": vv,
                    "attempts": attempts,
                    "scope_passed_per_try": scope_passed_per_try,
                    "dt": dt,
                }) + "\n")
                fout.flush()
                if (i + 1) % 25 == 0:
                    mean_att = attempts_total / (i + 1)
                    print(
                        f"  [{constraint}] {i+1}/{len(prompts)} "
                        f"parse={parse_ok}({parse_ok/(i+1):.0%}) "
                        f"verify={verify_ok}({verify_ok/(i+1):.0%}) "
                        f"attempts_avg={mean_att:.2f}",
                        file=sys.stderr,
                    )
            mean_att = attempts_total / max(len(prompts), 1)
            print(
                f"[day9] {constraint} done: "
                f"parse={parse_ok}/{len(prompts)}={parse_ok/len(prompts):.1%} "
                f"verify={verify_ok}/{len(prompts)}={verify_ok/len(prompts):.1%} "
                f"mean_attempts={mean_att:.2f} "
                f"mean_gen_s={(sum(cell_durs)/max(len(cell_durs),1)):.2f}",
                file=sys.stderr,
            )
    total = time.perf_counter() - t_run
    print(f"\n[day9] ALL DONE in {total/60:.1f} min → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    out = Path("results/day9/linalg_smoke.jsonl")
    run(out)
