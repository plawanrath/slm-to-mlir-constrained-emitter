"""Day-39 AM smoke test: does the post-fix MLX in-line decoder produce
parse-valid MLIR on at least one prompt?

Runs `mlx_generate_coupled` against SmolLM2-1.7B-Instruct on a single
canonical arith+func prompt and checks the output with the LARK parser.

DoD (from docs/daily_log/day39_plan.md):
  MLX decoder produces parse-valid output on >=1 test prompt, e.g.:
    module { func.func @f(%a : i32) -> i32 {
      %0 = arith.addi %a , %a : i32
      return %0 : i32
    }}

Prints the generated text + parse/verify status + closed-terminals count.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = "/Users/plawanrath/Documents/GitHub-public/slm-to-mlir-constrained-emitter"
os.chdir(REPO); sys.path.insert(0, REPO)

from decoder.inline_c3 import mlx_generate_coupled
from grammar.parser import is_parse_valid
from scripts.env.verify_cache import verify

MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
GRAMMAR = "grammar/mlir_gen_c1c2.lark"

PROMPT = (
    "Emit the following MLIR function verbatim inside a module:\n\n"
    "module { func.func @f(%a : i32) -> i32 { "
    "%0 = arith.addi %a , %a : i32 "
    "return %0 : i32 }}\n\n"
    "Output only the MLIR, no prose.\n"
)


def main() -> int:
    t0 = time.perf_counter()
    print(f"[day39-smoke] model={MODEL} grammar={GRAMMAR}")
    res = mlx_generate_coupled(
        prompt=PROMPT,
        model_id=MODEL,
        grammar_path=GRAMMAR,
        max_tokens=256,
        verbose=True,
    )
    dt = time.perf_counter() - t0
    pv = is_parse_valid(res.text)
    vv = pv and verify(res.text)["returncode"] == 0
    print(f"\n--- generated ({len(res.tokens)} tokens, {dt:.1f}s) ---")
    print(res.text)
    print("--- end ---")
    print(f"closed_terminals = {res.closed_terminals}")
    print(f"accepted        = {res.accepted}")
    print(f"parse_valid     = {pv}")
    print(f"verify_valid    = {vv}")

    out = Path("results/day39/smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": MODEL, "prompt": PROMPT, "text": res.text,
        "tokens": len(res.tokens), "dt_s": dt,
        "closed_terminals": res.closed_terminals,
        "accepted": res.accepted,
        "parse_valid": pv, "verify_valid": vv,
    }, indent=2))
    print(f"wrote {out}")

    return 0 if pv else 1


if __name__ == "__main__":
    sys.exit(main())
