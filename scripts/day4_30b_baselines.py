"""Day-4 §3.3 30B baselines: CodeLlama-34B × C1, Granite-Code-34B × {free, C1}.

C1 on Ollama is implemented as post-hoc rejection sampling against the LARK
parse grammar (not the gen grammar): Ollama/llama.cpp doesn't accept LARK
directly, so we generate freely with up to 5 retries and accept the first that
parses under `grammar/mlir.lark` (the permissive 95.2%-coverage grammar).
"""
from __future__ import annotations

import json
import os
import random
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
    # (nickname, ollama_tag, constraint)
    ("granite-free", "granite-code:34b-instruct-q4_K_M", "none"),
    ("granite-c1",   "granite-code:34b-instruct-q4_K_M", "c1"),
    ("codellama-c1", "codellama:34b-instruct-q4_K_M",    "c1"),
]

N_PER_CELL = 200  # smoke-matrix size
SEED = 0  # single seed for smoke

FEW_SHOT = """Example 1:
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


def _prompts() -> list[str]:
    prompts: list[str] = []
    for p in sorted(Path("eval/benchmarks/mlir_spec_150/examples").glob("*.json")):
        prompts.append(json.loads(p.read_text())["nl"])
    lines = Path("data/processed/l3_tests.jsonl").read_text().splitlines()
    rng = random.Random(SEED); rng.shuffle(lines)
    for l in lines:
        if len(prompts) >= N_PER_CELL: break
        r = json.loads(l); nl = r.get("weak_nl", "").strip()
        if nl: prompts.append(nl)
    return prompts


def _build_prompt(nl: str) -> str:
    return (
        "You are an MLIR emitter. Output ONLY valid MLIR code using arith, func, "
        "and memref dialects. Reuse parameter names exactly. No prose, no markdown.\n\n"
        f"{FEW_SHOT}\n\nNow this task:\nTask: {nl}\nMLIR:\n"
    )


def _c1_rejection_sample(prompt: str, model: str, n_retries: int = 5) -> str:
    """Generate freely; retry up to n_retries with different seeds until
    the output parses under the LARK coverage grammar."""
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
    print(f"[30b] {len(prompts)} prompts", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0_all = time.perf_counter()
    with out_path.open("w") as f:
        for nick, model, constraint in CELLS:
            print(f"\n[30b] cell: {nick} ({model}, {constraint})", file=sys.stderr)
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
                    else:  # c1 via rejection sampling
                        out = _c1_rejection_sample(prompt, model)
                        meta = {"num_tokens": -1}  # not tracked
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
                f"[30b] {nick} done: parse={parse_ok}/{len(prompts)}={parse_ok/len(prompts):.1%} "
                f"verify={verify_ok}/{len(prompts)}={verify_ok/len(prompts):.1%}",
                file=sys.stderr,
            )
    total = time.perf_counter() - t0_all
    print(f"\n[30b] ALL DONE in {total/60:.1f} min → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    run(Path("results/day4/baselines_30b.jsonl"))
