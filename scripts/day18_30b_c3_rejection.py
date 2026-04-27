"""Day-18 Phase-A §1: 30B + (C1 ∩ C3) rejection sampling on both dialects.

For each prompt, generate with Ollama; accept iff (parses under LARK grammar)
AND (C3 scope validator passes). On reject, resample with a new seed up to 5
total attempts — same budget as the SLM's C1+C2+C3 rejection loop. This is
the ``constraint-asymmetry-objection-killer'' experiment per ADR-0008
Phase A.

Cells run:
  - Granite-Code-34B + C1+C3 × arith+func (n=200) and linalg (n=125)
  - CodeLlama-34B + C1+C3 × arith+func (n=200) and linalg (n=125)

Re-uses Day-4 arith+func prompt set and Day-10 linalg prompt set for paired
bootstrap against the existing C1-only 30B cells.
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
os.chdir(REPO); sys.path.insert(0, REPO)
os.environ["PATH"] = f"{REPO}/scripts/env/bin:" + os.environ["PATH"]

from decoder.c3_scope import accept_or_reject
from eval.baselines.run_ollama import ollama_generate_raw
from grammar.parser import is_parse_valid
from scripts.env.verify_cache import verify

N_RETRIES = 5
SEED = 0

CELLS = [
    ("granite-c1c3",   "granite-code:34b-instruct-q4_K_M",   "arith+func"),
    ("codellama-c1c3", "codellama:34b-instruct-q4_K_M",      "arith+func"),
    ("granite-c1c3",   "granite-code:34b-instruct-q4_K_M",   "linalg"),
    ("codellama-c1c3", "codellama:34b-instruct-q4_K_M",      "linalg"),
]

FEW_SHOT_ARITH = """Example 1:
Task: Write a function that adds two i32 values.
MLIR:
module {
  func.func @f(%a: i32, %b: i32) -> i32 {
    %0 = arith.addi %a, %b : i32
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
  func.func @f(%m: memref<?xf32>, %i: index) -> f32 {
    %0 = memref.load %m[%i] : memref<?xf32>
    return %0 : f32
  }
}
"""

FEW_SHOT_LINALG = """Example 1:
Task: Perform matmul of two 2-D f32 memrefs into an output memref.
MLIR:
module {
  func.func @mm(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
    linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>) outs(%C : memref<?x?xf32>)
    return
  }
}

Example 2:
Task: Fill a 1-D f32 memref with a given f32 value.
MLIR:
module {
  func.func @fill(%v: f32, %m: memref<?xf32>) {
    linalg.fill ins(%v : f32) outs(%m : memref<?xf32>)
    return
  }
}

Example 3:
Task: Apply elementwise exp to a 1-D f32 memref.
MLIR:
module {
  func.func @ex(%x: memref<?xf32>, %y: memref<?xf32>) {
    linalg.exp ins(%x : memref<?xf32>) outs(%y : memref<?xf32>)
    return
  }
}
"""

TARGET_LINALG_OPS = {"matmul","matvec","fill","copy","transpose","broadcast",
                     "add","sub","mul","div","exp","abs"}


def _prompts_arith(n=200) -> list[str]:
    prompts: list[str] = []
    for p in sorted(Path("eval/benchmarks/mlir_spec_150/examples").glob("*.json")):
        prompts.append(json.loads(p.read_text())["nl"])
    lines = Path("data/processed/l3_tests.jsonl").read_text().splitlines()
    rng = random.Random(SEED); rng.shuffle(lines)
    for l in lines:
        if len(prompts) >= n: break
        r = json.loads(l); nl = r.get("weak_nl", "").strip()
        if nl: prompts.append(nl)
    return prompts[:n]


def _prompts_linalg(n=125) -> list[str]:
    prompts: list[str] = []
    for p in sorted(Path("eval/benchmarks/linalg_spec_30/examples").glob("*.json")):
        prompts.append(json.loads(p.read_text())["nl"])
    lines = Path("data/processed/l3_tests.jsonl").read_text().splitlines()
    rng = random.Random(SEED); rng.shuffle(lines)
    for l in lines:
        if len(prompts) >= n: break
        r = json.loads(l)
        if "linalg." not in r.get("mlir", ""): continue
        ops = set(m.group(1) for m in re.finditer(r"linalg\.(\w+)", r["mlir"]))
        if not ops or not (ops <= TARGET_LINALG_OPS): continue
        if "tensor<" in r["mlir"]: continue
        nl = r.get("weak_nl", "").strip()
        if nl: prompts.append(nl)
    return prompts[:n]


def _build_prompt_arith(nl: str) -> str:
    return (
        "You are an MLIR emitter. Output ONLY valid MLIR code using arith, func, "
        "and memref dialects. Reuse parameter names exactly. No prose, no markdown.\n\n"
        f"{FEW_SHOT_ARITH}\n\nNow this task:\nTask: {nl}\nMLIR:\n"
    )


def _build_prompt_linalg(nl: str) -> str:
    return (
        "You are an MLIR emitter. Output ONLY valid MLIR code using the linalg, "
        "memref, and func dialects, with memref semantics (no tensors). Reuse "
        "parameter names exactly. No prose, no markdown.\n\n"
        f"{FEW_SHOT_LINALG}\n\nNow this task:\nTask: {nl}\nMLIR:\n"
    )


def _c1c3_rejection_sample(prompt: str, model: str) -> tuple[str, int]:
    """Generate; accept iff parse AND scope pass. 5-retry budget."""
    last = ""
    for attempt in range(1, N_RETRIES + 1):
        seed = SEED + (attempt - 1) * 1000
        text, _meta = ollama_generate_raw(
            prompt=prompt, model=model, max_tokens=256,
            temperature=0.3 if attempt > 1 else 0.2, seed=seed,
        )
        last = text
        if not is_parse_valid(text):
            continue
        ok, _rep = accept_or_reject(text)
        if ok:
            return text, attempt
    return last, N_RETRIES


def run(out_path: Path) -> None:
    arith_prompts  = _prompts_arith(200)
    linalg_prompts = _prompts_linalg(125)
    print(f"[30b-c3] arith prompts={len(arith_prompts)}  linalg prompts={len(linalg_prompts)}",
          file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0_all = time.perf_counter()
    with out_path.open("w") as f:
        for nick, model, dialect in CELLS:
            prompts = arith_prompts if dialect == "arith+func" else linalg_prompts
            build = _build_prompt_arith if dialect == "arith+func" else _build_prompt_linalg
            print(f"\n[30b-c3] {nick} × {dialect} × n={len(prompts)}", file=sys.stderr)
            parse_ok = verify_ok = 0
            attempts_total = 0
            for i, nl in enumerate(prompts):
                prompt = build(nl)
                t0 = time.perf_counter()
                try:
                    out, attempts = _c1c3_rejection_sample(prompt, model)
                except Exception as e:
                    print(f"  [{nick}/{dialect}] {i}: err: {e}", file=sys.stderr); continue
                dt = time.perf_counter() - t0
                attempts_total += attempts
                pv = is_parse_valid(out); vv = False
                if pv:
                    parse_ok += 1
                    vv = verify(out)["returncode"] == 0
                    if vv: verify_ok += 1
                f.write(json.dumps({
                    "model": nick, "backend": "ollama", "constraint": "c1_c3",
                    "dialect": dialect, "prompt_id": i, "nl": nl,
                    "generated": out, "parse_valid": pv, "verify_valid": vv,
                    "attempts": attempts, "dt": dt,
                }) + "\n")
                f.flush()
                if (i + 1) % 10 == 0:
                    print(
                        f"  [{nick}/{dialect}] {i+1}/{len(prompts)} "
                        f"parse={parse_ok}({parse_ok/(i+1):.0%}) "
                        f"verify={verify_ok}({verify_ok/(i+1):.0%}) "
                        f"att_avg={attempts_total/(i+1):.2f} "
                        f"elapsed={(time.perf_counter()-t0_all)/60:.1f}min",
                        file=sys.stderr,
                    )
            print(
                f"[30b-c3] {nick}/{dialect} done: "
                f"parse={parse_ok}/{len(prompts)}={parse_ok/len(prompts):.1%} "
                f"verify={verify_ok}/{len(prompts)}={verify_ok/len(prompts):.1%} "
                f"mean_att={attempts_total/len(prompts):.2f}",
                file=sys.stderr,
            )
    total = time.perf_counter() - t0_all
    print(f"\n[30b-c3] ALL DONE in {total/60:.1f} min → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    run(Path("results/day18/30b_c1c3_rejection.jsonl"))
