"""Day-32: SmolLM2 × {free, C1, C1+C2, C1+C2+C3} × StableHLO matrix.

Uses `iree-compile --iree-input-type=stablehlo --compile-to=input` as the
verify-valid gate (Day 31 unblocked this — the vendored mlir-opt lacks
StableHLO dialect).

For C1+C2 constraint, we use the new `grammar/mlir_gen_stablehlo.lark`
grammar via Outlines CFG guide.

C3 scope validation uses the StableHLO-aware
`decoder/c3_scope.py` (extended Day 29-30).

Output: results/day32/stablehlo_smoke.jsonl (one row per constraint ×
prompt × seed-0).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
os.chdir(REPO); sys.path.insert(0, REPO)
os.environ["PATH"] = f"{REPO}/scripts/env/bin:" + os.environ["PATH"]

import outlines
from mlx_lm import generate as mlx_generate_free
from mlx_lm import load as mlx_load
from mlx_lm.sample_utils import make_sampler

from decoder.c3_scope import accept_or_reject
from grammar.parser import is_parse_valid
from scripts.env.verify_stablehlo import verify_stablehlo

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
N_PROMPTS = 30    # StableHLO-Spec-30
C3_MAX_RETRIES = 5

GRAMMAR_SHLO = Path("grammar/mlir_gen_stablehlo.lark").read_text()

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
        "<|im_start|>system\nOutput only valid MLIR StableHLO. Reuse parameter names exactly.<|im_end|>\n"
        f"<|im_start|>user\n{FEW_SHOT}\n\nNow this task:\nTask: {nl}\nMLIR:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _load_prompts() -> list[tuple[int, str, str]]:
    out = []
    for i, p in enumerate(sorted(Path("eval/benchmarks/stablehlo_spec_30/examples").glob("*.json"))):
        r = json.loads(p.read_text())
        out.append((i, r["nl"], r["mlir"]))
    return out


def _score(text: str) -> tuple[bool, bool]:
    """For StableHLO, iree-compile is both the parse and verify gate
    (our mlir.lark doesn't model StableHLO)."""
    r = verify_stablehlo(text)
    ok = (r["returncode"] == 0)
    return ok, ok


def run(out_path: Path) -> None:
    prompts = _load_prompts()
    print(f"[day32] {len(prompts)} StableHLO prompts", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[day32] loading {MODEL_ID}", file=sys.stderr)
    model_raw, tokenizer = mlx_load(MODEL_ID)
    mlx_model = outlines.from_mlxlm(model_raw, tokenizer)
    gen_c1 = outlines.Generator(mlx_model, outlines.cfg(GRAMMAR_SHLO))

    t_run = time.perf_counter()
    with out_path.open("w") as fout:
        for constraint in ("none", "c1", "c1_c3"):
            parse_ok = verify_ok = 0
            attempts_total = 0
            for i, nl, _ in prompts:
                prompt = _build_prompt(nl)
                t0 = time.perf_counter()
                attempts = 1
                try:
                    if constraint == "none":
                        out = mlx_generate_free(model_raw, tokenizer, prompt=prompt, max_tokens=600)
                    elif constraint == "c1":
                        out = gen_c1(prompt, max_tokens=600)
                    else:
                        out = gen_c1(prompt, max_tokens=600)
                        ok, _ = accept_or_reject(out)
                        while (not ok) and attempts < C3_MAX_RETRIES:
                            attempts += 1
                            out = gen_c1(
                                prompt, max_tokens=600,
                                sampler=make_sampler(temp=0.8),
                            )
                            ok, _ = accept_or_reject(out)
                except Exception as e:
                    print(f"  [{constraint}] {i}: err: {e}", file=sys.stderr)
                    continue
                dt = time.perf_counter() - t0
                attempts_total += attempts
                pv, vv = _score(out)
                if pv: parse_ok += 1
                if vv: verify_ok += 1
                fout.write(json.dumps({
                    "model": "smollm2-1.7b",
                    "constraint": constraint,
                    "dialect": "stablehlo+func",
                    "prompt_id": i, "nl": nl, "generated": out,
                    "parse_valid": pv, "verify_valid": vv,
                    "attempts": attempts, "dt": dt,
                }) + "\n")
                fout.flush()
                if (i + 1) % 10 == 0:
                    print(
                        f"  [{constraint}] {i+1}/{len(prompts)} "
                        f"parse={parse_ok}({parse_ok/(i+1):.0%}) "
                        f"verify={verify_ok}({verify_ok/(i+1):.0%}) "
                        f"attempts_avg={attempts_total/(i+1):.2f}",
                        file=sys.stderr,
                    )
            print(
                f"[day32] {constraint} done: "
                f"parse={parse_ok}/{len(prompts)}={parse_ok/len(prompts):.1%} "
                f"verify={verify_ok}/{len(prompts)}={verify_ok/len(prompts):.1%}",
                file=sys.stderr,
            )
    total = time.perf_counter() - t_run
    print(f"\n[day32] ALL DONE in {total/60:.1f} min → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    run(Path("results/day32/stablehlo_smoke.jsonl"))
