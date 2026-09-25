"""Day-55 (E9): Ollama baselines rerun at max_tokens=600.

Context: the headline tables have a token-budget asymmetry: SmolLM2
generates at max_tokens=600
(MLX) while the CodeLlama-34B / Granite-34B / StarCoder2-15B baselines were
run at max_tokens=256 (Ollama). Day-53/E8 bounded the possible harm from
released data (all truncated outputs already verify-fail); this run settles
the question directly by rerunning all three baselines at the SLM's budget.

Protocol: identical to the headline baseline cells (day34/day39pm/day50/day51:
day18 prompt builders and few-shot blocks, C1 parse gate + C3 scope acceptor,
5-retry rejection with temp=0.2 attempt 1 / 0.3 retries, seeded Ollama
generation, StarCoder2 outputs pass through _extract_mlir) with EXACTLY ONE
change: max_tokens 256 -> 600.

Cells: 3 models x {linalg n=125, arith+func n=200} x seeds {0, 1, 2}.
Ordering: linalg for all models first (the headline dialect), then arith, so
partial results are maximally useful if the run is interrupted; within a
dialect, cells are grouped by model to minimize Ollama model reloads.

Output: results/day55/e9_baselines_600.jsonl (one row per (cell, seed, prompt))
        results/day55/e9_summary.json        (per cell x seed rates, updated live)

Writes NEW files only; no released artifact is modified.
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
from eval.baselines.run_ollama import ollama_generate_raw
from grammar.parser import is_parse_valid
from scripts.env.verify_cache import verify

from scripts.day18_30b_c3_rejection import (
    _prompts_arith, _prompts_linalg,
    _build_prompt_arith, _build_prompt_linalg,
)
from scripts.day19_starcoder2_baseline import _extract_mlir

MAX_TOKENS = 600          # the single change vs the headline runs (was 256)
N_RETRIES = 5
SEEDS = [0, 1, 2]

MODELS = [
    ("codellama-c1c3",  "codellama:34b-instruct-q4_K_M"),
    ("granite-c1c3",    "granite-code:34b-instruct-q4_K_M"),
    ("starcoder2-c1c3", "starcoder2:instruct"),
]

# linalg first: the headline dialect, so partials are useful.
DIALECTS = [
    ("linalg",     125, _prompts_linalg, _build_prompt_linalg),
    ("arith+func", 200, _prompts_arith,  _build_prompt_arith),
]


def _c1c3_ollama(prompt: str, model: str, seed: int, extract: bool) -> tuple[str, int]:
    """Day-34 rejection loop, verbatim except MAX_TOKENS."""
    last = ""
    for attempt in range(1, N_RETRIES + 1):
        s = seed + (attempt - 1) * 1000
        raw, _ = ollama_generate_raw(
            prompt=prompt, model=model, max_tokens=MAX_TOKENS,
            temperature=0.3 if attempt > 1 else 0.2, seed=s,
        )
        last = raw
        text = _extract_mlir(raw) if extract else raw
        if not is_parse_valid(text): continue
        ok, _rep = accept_or_reject(text)
        if ok: return raw, attempt
    return last, N_RETRIES


def run(out_path: Path, summary_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    t_run = time.perf_counter()
    mode = "a" if out_path.exists() else "w"
    done_keys = set()
    if mode == "a":
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done_keys.add((r["model"], r["dialect"], r["seed"], r["prompt_id"]))
        print(f"[day55] resuming: {len(done_keys)} rows already present", file=sys.stderr)

    with out_path.open(mode) as fout:
        for dialect, n, prompt_fn, build in DIALECTS:
            prompts = prompt_fn(n)
            for nick, model in MODELS:
                is_sc2 = "starcoder2" in nick
                for seed in SEEDS:
                    cell_key = f"{nick}::{dialect}::seed{seed}"
                    parse_ok = verify_ok = 0
                    attempts_total = 0
                    n_new = 0
                    print(f"\n[day55] {cell_key} (n={len(prompts)}, max_tokens={MAX_TOKENS})",
                          file=sys.stderr)
                    for i, nl in enumerate(prompts):
                        if (nick, dialect, seed, i) in done_keys:
                            continue
                        prompt = build(nl)
                        t0 = time.perf_counter()
                        try:
                            raw, attempts = _c1c3_ollama(prompt, model, seed, extract=is_sc2)
                            out = _extract_mlir(raw) if is_sc2 else raw
                        except Exception as e:
                            print(f"  [{cell_key}] {i}: err: {e}", file=sys.stderr)
                            continue
                        dt = time.perf_counter() - t0
                        attempts_total += attempts
                        n_new += 1
                        pv = is_parse_valid(out); vv = False
                        if pv:
                            parse_ok += 1
                            vv = verify(out)["returncode"] == 0
                            if vv: verify_ok += 1
                        fout.write(json.dumps({
                            "model": nick, "backend": "ollama-c1c3",
                            "constraint": "c1_c3", "max_tokens": MAX_TOKENS,
                            "dialect": dialect, "seed": seed,
                            "pool": "frozen_released",
                            "prompt_id": i, "nl": nl,
                            "generated": out, "parse_valid": pv, "verify_valid": vv,
                            "attempts": attempts, "dt": dt,
                        }) + "\n")
                        fout.flush()
                        if n_new % 20 == 0:
                            print(
                                f"  [{cell_key}] {i+1}/{len(prompts)} "
                                f"parse={parse_ok}({parse_ok/max(n_new,1):.0%}) "
                                f"verify={verify_ok}({verify_ok/max(n_new,1):.0%}) "
                                f"att_avg={attempts_total/max(n_new,1):.2f} "
                                f"elapsed={(time.perf_counter()-t_run)/60:.1f}min",
                                file=sys.stderr,
                            )
                    if n_new:
                        summary[cell_key] = {
                            "n_new_rows": n_new,
                            "parse_valid": parse_ok,
                            "verify_valid": verify_ok,
                            "verify_rate_new_rows": verify_ok / n_new,
                            "mean_attempts": attempts_total / n_new,
                        }
                        summary_path.write_text(json.dumps(summary, indent=2))
                    print(
                        f"[day55] {cell_key} done: new={n_new} "
                        f"parse={parse_ok} verify={verify_ok}"
                        + (f" ({verify_ok/n_new:.1%})" if n_new else " (all cached)"),
                        file=sys.stderr,
                    )
    total = time.perf_counter() - t_run
    print(f"\n[day55] ALL DONE in {total/60:.1f} min → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    run(
        Path("results/day55/e9_baselines_600.jsonl"),
        Path("results/day55/e9_summary.json"),
    )
