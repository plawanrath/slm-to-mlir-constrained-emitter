"""Day-21b: StarCoder2-15B:instruct + C1+C3 rejection sampling, both dialects.

Closes the apples-to-apples loop for the modern-baseline system. Same
protocol as Day 18 (5-retry, accept iff parse-valid AND scope-valid).
Uses markdown-fence extraction like Day 19.

Output: results/day21/starcoder2_c1c3.jsonl
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
os.chdir(REPO); sys.path.insert(0, REPO)
os.environ["PATH"] = f"{REPO}/scripts/env/bin:" + os.environ["PATH"]

from decoder.c3_scope import accept_or_reject
from eval.baselines.run_ollama import ollama_generate_raw
from grammar.parser import is_parse_valid
from scripts.env.verify_cache import verify

from scripts.day19_starcoder2_baseline import (
    FEW_SHOT_ARITH, FEW_SHOT_LINALG,
    _extract_mlir, _prompts_arith, _prompts_linalg, _build_prompt,
)

MODEL = "starcoder2:instruct"
SEED = 0
N_RETRIES = 5

CELLS = [
    ("starcoder2-c1c3", "arith+func"),
    ("starcoder2-c1c3", "linalg"),
]


def _c1c3_rejection_sample(prompt: str) -> tuple[str, int]:
    last = ""
    for attempt in range(1, N_RETRIES + 1):
        seed = SEED + (attempt - 1) * 1000
        raw, _ = ollama_generate_raw(
            prompt=prompt, model=MODEL, max_tokens=256,
            temperature=0.3 if attempt > 1 else 0.2, seed=seed,
        )
        last = raw
        extracted = _extract_mlir(raw)
        if not is_parse_valid(extracted): continue
        ok, _rep = accept_or_reject(extracted)
        if ok: return raw, attempt
    return last, N_RETRIES


def run(out_path: Path) -> None:
    arith  = _prompts_arith()
    linalg = _prompts_linalg()
    print(f"[sc2-c3] arith={len(arith)} linalg={len(linalg)}", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0_all = time.perf_counter()
    with out_path.open("w") as f:
        for nick, dialect in CELLS:
            prompts = arith if dialect == "arith+func" else linalg
            print(f"\n[sc2-c3] {nick} × {dialect} × n={len(prompts)}", file=sys.stderr)
            parse_ok = verify_ok = 0
            attempts_total = 0
            for i, nl in enumerate(prompts):
                prompt = _build_prompt(nl, dialect)
                t0 = time.perf_counter()
                try:
                    raw, attempts = _c1c3_rejection_sample(prompt)
                    out = _extract_mlir(raw)
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
                f"[sc2-c3] {nick}/{dialect} done: "
                f"parse={parse_ok}/{len(prompts)}={parse_ok/len(prompts):.1%} "
                f"verify={verify_ok}/{len(prompts)}={verify_ok/len(prompts):.1%} "
                f"mean_att={attempts_total/len(prompts):.2f}",
                file=sys.stderr,
            )
    total = time.perf_counter() - t0_all
    print(f"\n[sc2-c3] ALL DONE in {total/60:.1f} min → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    run(Path("results/day21/starcoder2_c1c3.jsonl"))
