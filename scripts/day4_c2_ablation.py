"""B2 ablation — test whether C2 lifts verify-valid when the model is NOT
primed with few-shot examples.

Design doc §5.3 item 4 asks for an error-category analysis showing C2
specifically resolves type/arity errors. With few-shot prompts the model
already produces type-correct code (SmolLM2 hit 58% verify at both C1 and
C1+C2 in the smoke matrix). This ablation strips the examples to isolate
C2's contribution.
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

import outlines
from mlx_lm import load as mlx_load

from grammar.parser import is_parse_valid
from scripts.env.verify_cache import verify

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
N_PROMPTS = 200
GRAMMARS = {
    "c1":    Path("grammar/mlir_gen_c1.lark").read_text(),
    "c1_c2": Path("grammar/mlir_gen_c1c2.lark").read_text(),
}


def _zero_shot_prompt(nl: str) -> str:
    # Deliberately spartan — no examples, just the task.
    return (
        "<|im_start|>system\n"
        "Output only valid MLIR using arith, func, and memref dialects. "
        "Use parameter names consistently in the body.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\nTask: {nl}\nMLIR:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _prompts() -> list[str]:
    """Same prompt set as day4_slm_smoke for apples-to-apples comparison."""
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
    prompts = _prompts()
    print(f"[c2-ablation] {len(prompts)} prompts", file=sys.stderr)
    print(f"[c2-ablation] loading {MODEL_ID}", file=sys.stderr)
    model_raw, tokenizer = mlx_load(MODEL_ID)
    model = outlines.from_mlxlm(model_raw, tokenizer)
    generators = {
        "c1":    outlines.Generator(model, outlines.cfg(GRAMMARS["c1"])),
        "c1_c2": outlines.Generator(model, outlines.cfg(GRAMMARS["c1_c2"])),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0_all = time.perf_counter()
    with out_path.open("w") as f:
        for constraint in ("c1", "c1_c2"):
            parse_ok = verify_ok = 0
            durs = []
            for i, nl in enumerate(prompts):
                prompt = _zero_shot_prompt(nl)
                t0 = time.perf_counter()
                try:
                    out = generators[constraint](prompt, max_tokens=600)
                except Exception as e:
                    print(f"  [{constraint}] {i}: err: {e}", file=sys.stderr); continue
                dt = time.perf_counter() - t0
                durs.append(dt)
                pv = is_parse_valid(out); vv = False
                err = ""
                if pv:
                    parse_ok += 1
                    v = verify(out)
                    vv = v["returncode"] == 0
                    if vv: verify_ok += 1
                    else: err = v["stderr"][:500]
                f.write(json.dumps({
                    "model": "smollm2-1.7b", "constraint": constraint,
                    "prompt_id": i, "nl": nl, "generated": out,
                    "parse_valid": pv, "verify_valid": vv,
                    "verify_stderr": err, "dt": dt,
                    "prompting": "zero_shot",
                }) + "\n")
                f.flush()
                if (i+1) % 25 == 0:
                    print(
                        f"  [{constraint}] {i+1}/{len(prompts)} "
                        f"parse={parse_ok}({parse_ok/(i+1):.0%}) "
                        f"verify={verify_ok}({verify_ok/(i+1):.0%})",
                        file=sys.stderr,
                    )
            print(
                f"[c2-ablation] {constraint} done: "
                f"parse={parse_ok}/{len(prompts)}={parse_ok/len(prompts):.1%} "
                f"verify={verify_ok}/{len(prompts)}={verify_ok/len(prompts):.1%}",
                file=sys.stderr,
            )
    total = time.perf_counter() - t0_all
    print(f"\n[c2-ablation] DONE in {total/60:.1f} min → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    run(Path("results/day4/c2_ablation_zeroshot.jsonl"))
