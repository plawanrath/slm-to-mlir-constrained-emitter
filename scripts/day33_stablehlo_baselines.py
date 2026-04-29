"""Day-33: 30B + StarCoder2 baselines on StableHLO-Spec-30.

Mirrors Day 10 linalg baselines but targets StableHLO. Uses iree-compile
for verify-valid (same as Day 32).

Cells: {granite-34B, codellama-34B, starcoder2-15B:instruct} × {free, C1}
× StableHLO-Spec-30 (n=30). C1 uses our LARK parse grammar (mlir.lark)
as the parse gate — it doesn't yet model stablehlo specifically, so we
use `verify_stablehlo`'s compile-to-input stage as the validity gate
for the "parse" column too.

Output: results/day33/stablehlo_baselines.jsonl
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
os.chdir(REPO); sys.path.insert(0, REPO)
os.environ["PATH"] = f"{REPO}/scripts/env/bin:" + os.environ["PATH"]

from eval.baselines.run_ollama import ollama_generate_raw
from scripts.env.verify_stablehlo import verify_stablehlo
from scripts.day19_starcoder2_baseline import _extract_mlir

CELLS = [
    ("granite-free",      "granite-code:34b-instruct-q4_K_M",   "none"),
    ("granite-c1",        "granite-code:34b-instruct-q4_K_M",   "c1"),
    ("codellama-free",    "codellama:34b-instruct-q4_K_M",      "none"),
    ("codellama-c1",      "codellama:34b-instruct-q4_K_M",      "c1"),
    ("starcoder2-free",   "starcoder2:instruct",                 "none"),
    ("starcoder2-c1",     "starcoder2:instruct",                 "c1"),
]

N_RETRIES = 5
SEED = 0

FEW_SHOT = """Example 1:
Task: Add two 1-D f32 tensors of 16 elements.
MLIR:
module {
  func.func @a(%a: tensor<16xf32>, %b: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.add %a, %b : tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}

Example 2:
Task: Compute elementwise absolute value of a 1-D f32 tensor.
MLIR:
module {
  func.func @ab(%a: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.abs %a : tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}

Example 3:
Task: Transpose a 4x8 f32 tensor to 8x4.
MLIR:
module {
  func.func @t(%a: tensor<4x8xf32>) -> tensor<8x4xf32> {
    %0 = stablehlo.transpose %a, dims = [1, 0] : (tensor<4x8xf32>) -> tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
  }
}
"""


def _build_prompt(nl: str) -> str:
    return (
        "You are an MLIR emitter. Output ONLY valid MLIR StableHLO code. "
        "Reuse parameter names exactly. No prose, no markdown.\n\n"
        f"{FEW_SHOT}\n\nNow this task:\nTask: {nl}\nMLIR:\n"
    )


def _is_parseable(text: str) -> bool:
    """Use iree-compile as the StableHLO parse gate."""
    return verify_stablehlo(text)["returncode"] == 0


def _c1_rejection_sample(prompt: str, model: str, is_starcoder2: bool = False) -> str:
    """Generate freely up to N_RETRIES; accept the first that parses + verifies
    under iree-compile (stablehlo gate). Like Day-4/10 but with StableHLO."""
    last = ""
    for attempt in range(N_RETRIES):
        s = SEED + attempt * 1000
        raw, _ = ollama_generate_raw(
            prompt=prompt, model=model, max_tokens=256,
            temperature=0.3 if attempt > 0 else 0.2, seed=s,
        )
        last = raw
        text = _extract_mlir(raw) if is_starcoder2 else raw
        if _is_parseable(text):
            return raw
    return last


def _load_prompts() -> list[tuple[int, str]]:
    return [(i, json.loads(p.read_text())["nl"])
            for i, p in enumerate(sorted(
                Path("eval/benchmarks/stablehlo_spec_30/examples").glob("*.json")
            ))]


def run(out_path: Path) -> None:
    prompts = _load_prompts()
    print(f"[day33] {len(prompts)} StableHLO prompts", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0_all = time.perf_counter()
    with out_path.open("w") as f:
        for nick, model, constraint in CELLS:
            is_sc2 = "starcoder2" in nick
            print(f"\n[day33] {nick} × {constraint}", file=sys.stderr)
            verify_ok = 0
            for i, nl in prompts:
                prompt = _build_prompt(nl)
                t0 = time.perf_counter()
                try:
                    if constraint == "none":
                        raw, _ = ollama_generate_raw(
                            prompt=prompt, model=model, max_tokens=256,
                            temperature=0.2, seed=SEED,
                        )
                    else:
                        raw = _c1_rejection_sample(prompt, model, is_sc2)
                    out = _extract_mlir(raw) if is_sc2 else raw
                except Exception as e:
                    print(f"  [{nick}] {i}: err: {e}", file=sys.stderr); continue
                dt = time.perf_counter() - t0
                r = verify_stablehlo(out)
                vv = (r["returncode"] == 0)
                if vv: verify_ok += 1
                f.write(json.dumps({
                    "model": nick, "backend": "ollama", "constraint": constraint,
                    "dialect": "stablehlo+func", "prompt_id": i, "nl": nl,
                    "generated": out, "verify_valid": vv, "dt": dt,
                }) + "\n")
                f.flush()
                if (i + 1) % 5 == 0:
                    print(
                        f"  [{nick}/{constraint}] {i+1}/{len(prompts)} "
                        f"verify={verify_ok}({verify_ok/(i+1):.0%}) "
                        f"elapsed={(time.perf_counter()-t0_all)/60:.1f}min",
                        file=sys.stderr,
                    )
            print(
                f"[day33] {nick}/{constraint} done: "
                f"verify={verify_ok}/{len(prompts)}={verify_ok/len(prompts):.1%}",
                file=sys.stderr,
            )
    total = time.perf_counter() - t0_all
    print(f"\n[day33] ALL DONE in {total/60:.1f} min → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    run(Path("results/day33/stablehlo_baselines.jsonl"))
