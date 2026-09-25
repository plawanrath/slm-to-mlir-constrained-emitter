"""Day-52: Granite-Code-8B fp16 (MLX-LM) on arith+func — fp16 control extension.

Day-51 / Item 3 closed the quantization-asymmetry concern on linalg only.
The fp16 control should run on arith+func too so the precision-asymmetry
argument is symmetric across dialects. This script adds the arith+func cell at n=200 × seeds 0/1/2 under
the same C1+C3 protocol as the apples_to_apples 34B/15B baselines.

Cell name: granite-8b-fp16-c1c3::arith+func
Output: results/day52/granite_8b_fp16_arith.jsonl

Protocol parity with day51_seed0_n200.py (arith+func cells):
  - Same prompt set: arith from day18._prompts_arith(200) (SEED=0,
    deterministic; preserves prompt_id pairing with the SmolLM2 /
    CodeLlama-34B / Granite-34B / StarCoder2 cells)
  - Same prompt builder: Granite-Code-Instruct chat format with the
    Day-18 arith FEW_SHOT block (matches the linalg fp16 cell's
    chat-template style)
  - Same parse gate (LARK is_parse_valid) + C3 acceptor
    (decoder.c3_scope.accept_or_reject)
  - Same 5-retry budget; temp=0.0 attempt 1, temp=0.8 retries
  - Same verifier (cached mlir-opt --verify)
  - max_tokens=600 (MLX-LM default for SmolLM2 + Granite-8B-fp16 cells)

Conversion is reused — model already on disk at
models/mlx-granite-8b-code-instruct-fp16/, pinned at
ibm-granite/granite-8b-code-instruct@b996987fb31d62d4cb4a19adad1f06ee8a3d6109
in scripts/env/models.lock.txt. Do NOT re-convert.
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

from decoder.c3_scope import accept_or_reject
from grammar.parser import is_parse_valid
from scripts.env.verify_cache import verify
from scripts.day18_30b_c3_rejection import FEW_SHOT_ARITH, _prompts_arith

LOCAL_MLX_PATH = Path("models/mlx-granite-8b-code-instruct-fp16")
HF_REPO = "ibm-granite/granite-8b-code-instruct"
HF_REVISION = "b996987fb31d62d4cb4a19adad1f06ee8a3d6109"

SEEDS = [0, 1, 2]
N_RETRIES = 5
N_PROMPTS = 200
MAX_TOKENS = 600


def _build_chat_prompt(nl: str) -> str:
    """Granite-Code-Instruct chat format. Mirrors the linalg fp16 cell's
    structure; system message tuned for arith+func+memref."""
    return (
        "System:\nYou are an MLIR emitter. Output ONLY valid MLIR code using "
        "arith, func, and memref dialects. Reuse parameter names exactly. No "
        "prose, no markdown.\n\n"
        "Question:\n"
        f"{FEW_SHOT_ARITH}\n\nNow this task:\nTask: {nl}\nMLIR:\n\nAnswer:\n"
    )


_MLX_CACHE: dict[str, object] = {}


def _load() -> tuple:
    if "model" not in _MLX_CACHE:
        if not LOCAL_MLX_PATH.exists():
            raise RuntimeError(
                f"MLX conversion missing at {LOCAL_MLX_PATH}. Run:\n"
                f"  python -m mlx_lm convert --hf-path {HF_REPO} "
                f"--hf-revision {HF_REVISION} "
                f"--mlx-path {LOCAL_MLX_PATH} --dtype float16"
            )
        from mlx_lm import load as mlx_load
        from mlx_lm import generate as mlx_generate_fn
        from mlx_lm.sample_utils import make_sampler
        model, tokenizer = mlx_load(str(LOCAL_MLX_PATH))
        _MLX_CACHE["model"] = model
        _MLX_CACHE["tokenizer"] = tokenizer
        _MLX_CACHE["mlx_generate_fn"] = mlx_generate_fn
        _MLX_CACHE["make_sampler"] = make_sampler
    return (_MLX_CACHE["model"], _MLX_CACHE["tokenizer"],
            _MLX_CACHE["mlx_generate_fn"], _MLX_CACHE["make_sampler"])


def _generate(prompt: str, temperature: float, seed: int) -> str:
    import mlx.core as mx
    model, tokenizer, mlx_generate_fn, make_sampler = _load()
    mx.random.seed(seed)
    sampler = make_sampler(temp=temperature)
    return mlx_generate_fn(
        model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS,
        sampler=sampler, verbose=False,
    )


def _c1c3_rejection(prompt: str, seed: int) -> tuple[str, int]:
    last = ""
    for attempt in range(1, N_RETRIES + 1):
        s = seed + (attempt - 1) * 1000
        try:
            text = _generate(
                prompt, temperature=0.0 if attempt == 1 else 0.8, seed=s,
            )
        except Exception as e:
            print(f"    mlx err: {e}", file=sys.stderr)
            continue
        last = text
        if not is_parse_valid(text):
            continue
        ok, _rep = accept_or_reject(text)
        if ok:
            return text, attempt
    return last, N_RETRIES


def run(out_path: Path) -> None:
    prompts = _prompts_arith(N_PROMPTS)
    print(f"[day52-granite8b-fp16] arith+func n={len(prompts)} × seeds={SEEDS}",
          file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0_all = time.perf_counter()
    with out_path.open("w") as f:
        for seed in SEEDS:
            print(f"\n[day52-granite8b-fp16] seed={seed}", file=sys.stderr)
            parse_ok = verify_ok = 0
            attempts_total = 0
            for i, nl in enumerate(prompts):
                prompt = _build_chat_prompt(nl)
                t0 = time.perf_counter()
                try:
                    raw, attempts = _c1c3_rejection(prompt, seed)
                except Exception as e:
                    print(f"  [s{seed}] {i}: err: {e}", file=sys.stderr)
                    continue
                dt = time.perf_counter() - t0
                attempts_total += attempts
                pv = is_parse_valid(raw)
                vv = False
                if pv:
                    parse_ok += 1
                    vv = verify(raw)["returncode"] == 0
                    if vv:
                        verify_ok += 1
                f.write(json.dumps({
                    "model": "granite-8b-fp16-c1c3", "backend": "mlx-c1c3",
                    "constraint": "c1_c3", "dialect": "arith+func", "seed": seed,
                    "prompt_id": i, "nl": nl,
                    "generated": raw, "parse_valid": pv, "verify_valid": vv,
                    "attempts": attempts, "dt": dt,
                }) + "\n")
                f.flush()
                if (i + 1) % 20 == 0:
                    print(
                        f"  [s{seed}] {i+1}/{len(prompts)} "
                        f"parse={parse_ok}({parse_ok/(i+1):.0%}) "
                        f"verify={verify_ok}({verify_ok/(i+1):.0%}) "
                        f"att_avg={attempts_total/(i+1):.2f} "
                        f"elapsed={(time.perf_counter()-t0_all)/60:.1f}min",
                        file=sys.stderr,
                    )
            print(
                f"[day52-granite8b-fp16] s{seed} done: "
                f"parse={parse_ok}/{len(prompts)}={parse_ok/len(prompts):.1%} "
                f"verify={verify_ok}/{len(prompts)}={verify_ok/len(prompts):.1%}",
                file=sys.stderr,
            )
    total = time.perf_counter() - t0_all
    print(f"\n[day52-granite8b-fp16] ALL DONE in {total/60:.1f} min → {out_path}",
          file=sys.stderr)


if __name__ == "__main__":
    run(Path("results/day52/granite_8b_fp16_arith.jsonl"))
