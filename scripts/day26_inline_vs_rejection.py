"""Day-26 Phase-B gate: in-line C3 decoder vs rejection-sampled C3 decoder.

Paired matrix: same SmolLM2 model, same prompt set (MLIR-Spec-150 arith+func
first 100 for speed), two decoder paths:

  A. C1+C2+C3 via Outlines + post-hoc rejection sampling (existing path)
  B. C1+C2+C3 via in-line coupled decoder (Days 22-25)

For each prompt, we emit one generation per path (seed 0) and measure:

  - parse_valid (under LARK parse grammar)
  - verify_valid (under mlir-opt --verify)
  - wall-clock dt
  - mean attempts (N/A for in-line, 1 by construction)

Then paired-bootstrap Δverify = (in-line − rejection) aligned by prompt_id.
◆ **Gate**: CI-lower-bound for Δverify ≥ 0. If in-line significantly
under-performs, halt and debug in-line implementation.

NOTE: this script REQUIRES the Day-25 `mlx_generate` entry point to be
implemented. Currently `generate.py` ships only `generate_text_mock`.
Running this script before `mlx_generate` is wired will raise
`NotImplementedError`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = "/Users/plawanrath/Documents/GitHub-public/slm-to-mlir-constrained-emitter"
os.chdir(REPO); sys.path.insert(0, REPO)

import numpy as np

from decoder.generate import GenerateRequest, GenerateResult, ConstraintLevel
from decoder.c1_cfg import mlx_generate
from decoder.c3_scope import accept_or_reject
from grammar.parser import is_parse_valid
from scripts.env.verify_cache import verify

MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
N_PROMPTS = 100
SEED = 0


def _prompts() -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for i, p in enumerate(sorted(
        Path("eval/benchmarks/mlir_spec_150/examples").glob("*.json")
    )):
        if i >= N_PROMPTS: break
        r = json.loads(p.read_text())
        out.append((i, r["nl"]))
    return out


def _gen_rejection(prompt: str) -> tuple[str, int, float]:
    """Existing C1+C2+C3 path via post-hoc rejection sampling, 5-retry."""
    from decoder.generate import GenerateRequest, ConstraintLevel
    t0 = time.perf_counter()
    attempts = 0
    last_text = ""
    for attempt in range(5):
        attempts += 1
        req = GenerateRequest(
            prompt=prompt, model=MODEL, constraint=ConstraintLevel.C1_C2,
            temperature=0.0 if attempt == 0 else 0.8, seed=SEED + attempt * 1000,
            max_tokens=256, dialect="arith+func",
        )
        res: GenerateResult = mlx_generate(req)
        last_text = res.text
        if not is_parse_valid(res.text): continue
        ok, _rep = accept_or_reject(res.text)
        if ok:
            return res.text, attempts, time.perf_counter() - t0
    return last_text, attempts, time.perf_counter() - t0


def _gen_inline(prompt: str) -> tuple[str, int, float]:
    """In-line coupled decoder path (Days 22-25)."""
    try:
        from decoder.inline_c3 import mlx_generate_coupled  # TODO: implement
    except ImportError:
        raise NotImplementedError(
            "mlx_generate_coupled is not yet implemented — wire the MLX sampling "
            "loop into decoder/inline_c3/generate.py before running Day 26."
        )
    t0 = time.perf_counter()
    res = mlx_generate_coupled(prompt=prompt, model=MODEL, seed=SEED, max_tokens=256)
    return res.text, 1, time.perf_counter() - t0


def run(out_path: Path) -> None:
    prompts = _prompts()
    print(f"[day26] n_prompts={len(prompts)}", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i, nl in prompts:
            # A: rejection-sampled path
            text_a, att_a, dt_a = _gen_rejection(nl)
            pv_a = is_parse_valid(text_a)
            vv_a = (pv_a and verify(text_a)["returncode"] == 0)
            f.write(json.dumps({
                "decoder": "rejection", "prompt_id": i, "nl": nl,
                "generated": text_a, "parse_valid": pv_a, "verify_valid": vv_a,
                "attempts": att_a, "dt": dt_a,
            }) + "\n")
            f.flush()
            # B: in-line path
            try:
                text_b, att_b, dt_b = _gen_inline(nl)
            except NotImplementedError as e:
                print(f"[day26] halting: {e}", file=sys.stderr)
                return
            pv_b = is_parse_valid(text_b)
            vv_b = (pv_b and verify(text_b)["returncode"] == 0)
            f.write(json.dumps({
                "decoder": "inline", "prompt_id": i, "nl": nl,
                "generated": text_b, "parse_valid": pv_b, "verify_valid": vv_b,
                "attempts": att_b, "dt": dt_b,
            }) + "\n")
            f.flush()


if __name__ == "__main__":
    run(Path("results/day26/c3_inline_vs_rejection.jsonl"))
