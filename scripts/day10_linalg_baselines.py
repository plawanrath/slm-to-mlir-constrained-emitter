"""Day-10 §7.5: 30B baselines on linalg.
CodeLlama-34B + C1, Granite-Code-34B + {free, C1}, linalg prompts.

Same rejection-sampling protocol as Day-4 (5 retries against LARK parse grammar,
which now recognizes linalg named ops after Day-8 extension).
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

from eval.baselines.run_ollama import ollama_generate_raw
from grammar.parser import is_parse_valid
from scripts.env.verify_cache import verify

CELLS = [
    ("granite-free", "granite-code:34b-instruct-q4_K_M", "none"),
    ("granite-c1",   "granite-code:34b-instruct-q4_K_M", "c1"),
    ("codellama-c1", "codellama:34b-instruct-q4_K_M",    "c1"),
]

TARGET_OPS = {"matmul", "matvec", "fill", "copy", "transpose", "broadcast",
              "add", "sub", "mul", "div", "exp", "abs"}
N_PER_CELL = 125  # match Day-9 SmolLM2 matrix size (same prompt set)
SEED = 0

FEW_SHOT = """Example 1:
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


def _prompts() -> list[str]:
    prompts: list[str] = []
    for p in sorted(Path("eval/benchmarks/linalg_spec_30/examples").glob("*.json")):
        prompts.append(json.loads(p.read_text())["nl"])
    lines = Path("data/processed/l3_tests.jsonl").read_text().splitlines()
    rng = random.Random(SEED); rng.shuffle(lines)
    for l in lines:
        if len(prompts) >= N_PER_CELL: break
        r = json.loads(l)
        if "linalg." not in r.get("mlir", ""): continue
        ops = set(m.group(1) for m in re.finditer(r"linalg\.(\w+)", r["mlir"]))
        if not ops or not (ops <= TARGET_OPS): continue
        if "tensor<" in r["mlir"]: continue
        nl = r.get("weak_nl", "").strip()
        if not nl: continue
        prompts.append(nl)
    return prompts


def _build_prompt(nl: str) -> str:
    return (
        "You are an MLIR emitter. Output ONLY valid MLIR code using the linalg, "
        "memref, and func dialects, with memref semantics (no tensors). Reuse "
        "parameter names exactly. No prose, no markdown.\n\n"
        f"{FEW_SHOT}\n\nNow this task:\nTask: {nl}\nMLIR:\n"
    )


def _c1_rejection_sample(prompt: str, model: str, n_retries: int = 5) -> str:
    last = ""
    for attempt in range(n_retries):
        seed = SEED + attempt * 1000
        text, _meta = ollama_generate_raw(
            prompt=prompt, model=model, max_tokens=256,
            temperature=0.3 if attempt > 0 else 0.2, seed=seed,
        )
        last = text
        if is_parse_valid(text):
            return text
    return last


def run(out_path: Path) -> None:
    prompts = _prompts()
    print(f"[30b-linalg] {len(prompts)} prompts", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0_all = time.perf_counter()
    with out_path.open("w") as f:
        for nick, model, constraint in CELLS:
            print(f"\n[30b-linalg] cell: {nick} ({model}, {constraint})", file=sys.stderr)
            parse_ok = verify_ok = 0
            durs = []
            for i, nl in enumerate(prompts):
                prompt = _build_prompt(nl)
                t0 = time.perf_counter()
                try:
                    if constraint == "none":
                        out, meta = ollama_generate_raw(
                            prompt=prompt, model=model, max_tokens=256,
                            temperature=0.2, seed=SEED,
                        )
                    else:
                        out = _c1_rejection_sample(prompt, model)
                except Exception as e:
                    print(f"  [{nick}] {i}: err: {e}", file=sys.stderr); continue
                dt = time.perf_counter() - t0
                durs.append(dt)
                pv = is_parse_valid(out); vv = False
                if pv:
                    parse_ok += 1
                    vv = verify(out)["returncode"] == 0
                    if vv: verify_ok += 1
                f.write(json.dumps({
                    "model": nick, "backend": "ollama", "constraint": constraint,
                    "prompt_id": i, "nl": nl, "generated": out,
                    "parse_valid": pv, "verify_valid": vv, "dt": dt,
                }) + "\n")
                f.flush()
                if (i + 1) % 10 == 0:
                    print(
                        f"  [{nick}] {i+1}/{len(prompts)} "
                        f"parse={parse_ok}({parse_ok/(i+1):.0%}) "
                        f"verify={verify_ok}({verify_ok/(i+1):.0%}) "
                        f"elapsed={(time.perf_counter()-t0_all)/60:.1f}min",
                        file=sys.stderr,
                    )
            print(
                f"[30b-linalg] {nick} done: parse={parse_ok}/{len(prompts)}={parse_ok/len(prompts):.1%} "
                f"verify={verify_ok}/{len(prompts)}={verify_ok/len(prompts):.1%}",
                file=sys.stderr,
            )
    total = time.perf_counter() - t0_all
    print(f"\n[30b-linalg] ALL DONE in {total/60:.1f} min → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    run(Path("results/day10/linalg_baselines.jsonl"))
