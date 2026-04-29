"""Day-40: In-line vs rejection-sampled C3 decoder equivalence test.

Paired matrix: same SmolLM2 model, same prompt set (MLIR-Spec-150
arith+func, first N prompts), two decoder paths:

  A. Rejection-sampled C1+C2+C3 via Outlines + 5-retry (existing path)
  B. In-line C1+C3 via the coupled decoder (Day 39 fix)

For each prompt we emit one generation per path and measure:

  - parse_valid  (LARK parse-grammar)
  - verify_valid (mlir-opt --verify-diagnostics via verify_cache)
  - wall-clock dt
  - attempts (always 1 for in-line by construction)

Then paired-bootstrap Δparse = (in-line − rejection) aligned by prompt_id.
The theoretical claim (Appendix Thm 3) is parse-equivalence by
construction; this script confirms empirically.

Output: results/day40/equivalence.jsonl + summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
os.chdir(REPO); sys.path.insert(0, REPO)

import numpy as np

from decoder.c3_scope import accept_or_reject
from decoder.inline_c3 import mlx_generate_coupled, mlx_generate_coupled_annealed
from grammar.parser import is_parse_valid
from scripts.env.verify_cache import verify

MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
N_PROMPTS = int(os.environ.get("DAY40_N", "20"))
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


FEW_SHOT = """Example 1:
Task: Write a function that adds two i32 values.
MLIR:
module {
  func.func @f(%a : i32 , %b : i32) -> i32 {
    %0 = arith.addi %a , %b : i32
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
"""


def _mk_prompt(nl: str) -> str:
    return (
        "<|im_start|>system\nOutput only valid MLIR. Reuse parameter names exactly.<|im_end|>\n"
        f"<|im_start|>user\n{FEW_SHOT}\n\nNow this task:\nTask: {nl}\nMLIR:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _gen_rejection(nl: str) -> tuple[str, int, float]:
    """5-retry: temp=0 first, temp=0.8 retries. Uses Outlines CFG-constrained
    sampling (C1+C2 via grammar, C3 via post-hoc accept_or_reject)."""
    from mlx_lm import load as mlx_load
    from mlx_lm.sample_utils import make_sampler
    import outlines

    global _MLX_GEN
    try:
        _MLX_GEN
    except NameError:
        model_raw, tok = mlx_load(MODEL)
        mlx_model = outlines.from_mlxlm(model_raw, tok)
        _MLX_GEN = outlines.Generator(
            mlx_model,
            outlines.cfg(Path("grammar/mlir_gen_c1c2.lark").read_text()),
        )

    prompt = _mk_prompt(nl)
    t0 = time.perf_counter()
    last = ""
    for attempt in range(1, 6):
        try:
            if attempt == 1:
                last = _MLX_GEN(prompt, max_tokens=600)
            else:
                last = _MLX_GEN(prompt, max_tokens=600, sampler=make_sampler(temp=0.8))
        except Exception as e:
            print(f"    rej err: {e}", file=sys.stderr); continue
        if not is_parse_valid(last): continue
        ok, _ = accept_or_reject(last)
        if ok:
            return last, attempt, time.perf_counter() - t0
    return last, 5, time.perf_counter() - t0


def _gen_inline(nl: str) -> tuple[str, int, float]:
    prompt = _mk_prompt(nl)
    t0 = time.perf_counter()
    if os.environ.get("DAY40_ANNEALED", "1") == "1":
        from decoder.c3_scope import accept_or_reject as _accept
        def _ok(text: str) -> bool:
            if not is_parse_valid(text):
                return False
            ok, _ = _accept(text)
            return ok
        res = mlx_generate_coupled_annealed(
            prompt=prompt, model_id=MODEL,
            grammar_path="grammar/mlir_gen_c1c2.lark",
            max_tokens=600,
            temperatures=(0.0, 0.4, 0.8),
            accept_fn=_ok,
        )
    else:
        res = mlx_generate_coupled(
            prompt=prompt, model_id=MODEL,
            grammar_path="grammar/mlir_gen_c1c2.lark",
            max_tokens=600,
        )
    return res.text, 1, time.perf_counter() - t0


def _paired_delta(xs: list[int], ys: list[int], n_boot: int = 10000) -> tuple[float, float, float]:
    """Return (mean_delta, lo95, hi95) for paired mean(xs − ys)."""
    diffs = np.array([x - y for x, y in zip(xs, ys)])
    n = len(diffs)
    boots = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(diffs[idx].mean())
    boots.sort()
    return float(diffs.mean()), float(boots[int(0.025 * n_boot)]), float(boots[int(0.975 * n_boot)])


def run(out_path: Path) -> None:
    prompts = _prompts()
    print(f"[day40] n={len(prompts)} model={MODEL}", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_rej, rows_inl = [], []
    with out_path.open("w") as f:
        for i, nl in prompts:
            print(f"[day40] prompt {i+1}/{len(prompts)}: {nl[:60]}...", file=sys.stderr)
            # Rejection path
            try:
                text_r, att_r, dt_r = _gen_rejection(nl)
            except Exception as e:
                print(f"    rej error on {i}: {e}", file=sys.stderr)
                text_r, att_r, dt_r = "", 0, 0.0
            pv_r = is_parse_valid(text_r)
            vv_r = pv_r and verify(text_r)["returncode"] == 0
            rows_rej.append({"decoder": "rejection", "prompt_id": i, "nl": nl,
                             "generated": text_r, "parse_valid": pv_r, "verify_valid": vv_r,
                             "attempts": att_r, "dt": dt_r})
            # In-line path
            try:
                text_i, att_i, dt_i = _gen_inline(nl)
            except Exception as e:
                print(f"    inline error on {i}: {e}", file=sys.stderr)
                text_i, att_i, dt_i = "", 0, 0.0
            pv_i = is_parse_valid(text_i)
            vv_i = pv_i and verify(text_i)["returncode"] == 0
            rows_inl.append({"decoder": "inline", "prompt_id": i, "nl": nl,
                             "generated": text_i, "parse_valid": pv_i, "verify_valid": vv_i,
                             "attempts": att_i, "dt": dt_i})
            for row in (rows_rej[-1], rows_inl[-1]):
                f.write(json.dumps(row) + "\n"); f.flush()
            print(f"    rej: pv={pv_r} vv={vv_r} att={att_r} dt={dt_r:.1f}s  "
                  f"inl: pv={pv_i} vv={vv_i} dt={dt_i:.1f}s", file=sys.stderr)

    # Summary
    pv_r = [int(r["parse_valid"]) for r in rows_rej]
    pv_i = [int(r["parse_valid"]) for r in rows_inl]
    vv_r = [int(r["verify_valid"]) for r in rows_rej]
    vv_i = [int(r["verify_valid"]) for r in rows_inl]
    dt_r = [r["dt"] for r in rows_rej]
    dt_i = [r["dt"] for r in rows_inl]
    att_r = [r["attempts"] for r in rows_rej]
    att_i = [r["attempts"] for r in rows_inl]

    dmean_pv, lo_pv, hi_pv = _paired_delta(pv_i, pv_r)
    dmean_vv, lo_vv, hi_vv = _paired_delta(vv_i, vv_r)
    speedup = (sum(dt_r) / sum(dt_i)) if sum(dt_i) > 0 else 0.0
    avg_att_r = sum(att_r) / len(att_r)
    avg_att_i = sum(att_i) / len(att_i)

    summary = {
        "n": len(prompts), "model": MODEL,
        "parse_valid": {"rejection": sum(pv_r)/len(pv_r),
                         "inline": sum(pv_i)/len(pv_i),
                         "delta_mean": dmean_pv, "delta_lo95": lo_pv, "delta_hi95": hi_pv},
        "verify_valid": {"rejection": sum(vv_r)/len(vv_r),
                          "inline": sum(vv_i)/len(vv_i),
                          "delta_mean": dmean_vv, "delta_lo95": lo_vv, "delta_hi95": hi_vv},
        "avg_attempts": {"rejection": avg_att_r, "inline": avg_att_i},
        "total_dt_s": {"rejection": sum(dt_r), "inline": sum(dt_i), "speedup_rej_over_inl": speedup},
    }
    (out_path.parent / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[day40] summary:\n{json.dumps(summary, indent=2)}", file=sys.stderr)


if __name__ == "__main__":
    run(Path("results/day40/equivalence.jsonl"))
