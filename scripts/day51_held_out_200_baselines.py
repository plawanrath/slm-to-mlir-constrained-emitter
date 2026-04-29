"""Day-51 / Item 2: 30B + 15B baselines on stablehlo_held_out_200.

Closes the §6 gap that the StableHLO Held-Out-200 result reports SmolLM2
(61.5%) without any baseline numbers.

Cells (3 = single-seed):
  CodeLlama-34B    + C1+C3 (rejection)  × stablehlo_held_out_200 (n=200)
  Granite-Code-34B + C1+C3 (rejection)  × stablehlo_held_out_200 (n=200)
  StarCoder2-15B   + C1+C3 (rejection)  × stablehlo_held_out_200 (n=200)

Protocol matches Day-26 / Day-50:
  - 5-retry rejection (temp=0.0 attempt 1, temp=0.8 retries — see
    scripts/day26_inline_vs_rejection.py:70)
  - seed=1 (fresh from the seeds 0/1/2 used elsewhere)
  - parse gate via iree-compile (StableHLO doesn't currently exercise C2;
    the C3 scope validator abstains on stablehlo ops, which is safe per
    decoder/c3_scope.py docstring — so "C1+C3" reduces to C1 in practice
    for this dialect)
  - verify via scripts/env/verify_stablehlo.py (with empty-stdin guard intact)

Output: results/day51/stablehlo_held_out_200_baselines.jsonl
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

from eval.baselines.run_ollama import ollama_generate_raw
from scripts.env.verify_stablehlo import verify_stablehlo
from scripts.day19_starcoder2_baseline import _extract_mlir
from decoder.c3_scope import accept_or_reject
from grammar.parser import is_parse_valid

CELLS = [
    ("codellama-c1c3",  "codellama:34b-instruct-q4_K_M"),
    ("granite-c1c3",    "granite-code:34b-instruct-q4_K_M"),
    ("starcoder2-c1c3", "starcoder2:instruct"),
]
SEED = 1
N_RETRIES = 5

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


def _load_held_out_200() -> list[tuple[int, str]]:
    return [
        (i, json.loads(p.read_text())["nl"])
        for i, p in enumerate(sorted(
            Path("eval/benchmarks/stablehlo_held_out_200/examples").glob("*.json")
        ))
    ]


def _c1c3_rejection(prompt: str, model: str, is_starcoder2: bool) -> tuple[str, int]:
    """5-retry rejection: parse-valid (LARK) + scope-valid (C3, abstain on
    stablehlo). On accept, return (raw, attempts). On exhaustion, return
    last raw."""
    last = ""
    for attempt in range(1, N_RETRIES + 1):
        s = SEED + (attempt - 1) * 1000
        raw, _ = ollama_generate_raw(
            prompt=prompt, model=model, max_tokens=256,
            temperature=0.0 if attempt == 1 else 0.8, seed=s,
        )
        last = raw
        text = _extract_mlir(raw) if is_starcoder2 else raw
        # Parse gate: iree-compile (StableHLO-aware) instead of LARK
        # (mlir.lark doesn't model stablehlo). C3 abstains on stablehlo
        # ops so contributes nothing — the gate is effectively C1 here.
        if verify_stablehlo(text)["returncode"] != 0:
            continue
        # C3 scope validator (abstains safely on stablehlo ops):
        ok, _rep = accept_or_reject(text)
        if ok:
            return raw, attempt
    return last, N_RETRIES


def run(out_path: Path) -> None:
    prompts = _load_held_out_200()
    print(f"[day51-ho200] {len(prompts)} prompts × {len(CELLS)} cells × seed={SEED}",
          file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0_all = time.perf_counter()
    with out_path.open("w") as f:
        for nick, model in CELLS:
            is_sc2 = "starcoder2" in nick
            print(f"\n[day51-ho200] {nick}", file=sys.stderr)
            verify_ok = 0
            attempts_total = 0
            for i, nl in prompts:
                prompt = _build_prompt(nl)
                t0 = time.perf_counter()
                try:
                    raw, attempts = _c1c3_rejection(prompt, model, is_sc2)
                    out = _extract_mlir(raw) if is_sc2 else raw
                except Exception as e:
                    print(f"  [{nick}] {i}: err: {e}", file=sys.stderr); continue
                dt = time.perf_counter() - t0
                attempts_total += attempts
                vv = (verify_stablehlo(out)["returncode"] == 0)
                if vv: verify_ok += 1
                f.write(json.dumps({
                    "model": nick, "backend": "ollama", "constraint": "c1_c3",
                    "dialect": "stablehlo+func", "seed": SEED,
                    "prompt_id": i, "nl": nl,
                    "generated": out, "verify_valid": vv,
                    "attempts": attempts, "dt": dt,
                }) + "\n")
                f.flush()
                if (i + 1) % 10 == 0:
                    print(
                        f"  [{nick}] {i+1}/{len(prompts)} "
                        f"verify={verify_ok}({verify_ok/(i+1):.0%}) "
                        f"att_avg={attempts_total/(i+1):.2f} "
                        f"elapsed={(time.perf_counter()-t0_all)/60:.1f}min",
                        file=sys.stderr,
                    )
            print(
                f"[day51-ho200] {nick} done: "
                f"verify={verify_ok}/{len(prompts)}={verify_ok/len(prompts):.1%} "
                f"mean_att={attempts_total/len(prompts):.2f}",
                file=sys.stderr,
            )
    total = time.perf_counter() - t0_all
    print(f"\n[day51-ho200] ALL DONE in {total/60:.1f} min → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    run(Path("results/day51/stablehlo_held_out_200_baselines.jsonl"))
