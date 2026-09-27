"""Day-61: frozen-pool arith+func baselines without the C3 gate (C1-only and free).

Context (camera-ready, NeurIPS 2026 E&D). The best-configuration baseline table
(C1-only vs C1+C3) and the raw free-decoding rates quoted for the 34B baselines
on arith+func were taken from the Day-4 run (results/day4/baselines_30b.jsonl),
which used a pre-freeze draft of MLIR-Spec-150: only 58 of its 134 unique
prompts are in the released pool. Every other cell of that table (StarCoder2
arith, all linalg cells) is already on the frozen released pool. This script
regenerates the missing arith+func cells on the frozen pool.

Protocol: identical to the headline baseline cells (day18 prompt builders and
few-shot blocks, seeded Ollama generation at max_tokens=256, temperature 0.2 on
attempt 1 and 0.3 on retries, seed + (attempt-1)*1000, 5 attempts), with ONE
change per cell type:
  * "-c1":   the acceptance gate is the C1 parse check only (the C3 scope
             acceptor is removed), i.e. the Day-4/Day-19 C1-only protocol;
  * "-free": a single attempt, no gate (Day-4/Day-19 free protocol).
Seed 0 only, matching every other C1-only / free cell in the table.

Before the new cells, --repro-check regenerates the first N published
codellama C1+C3 seed-0 outputs (results/day51_seed0_n200/multiseed_seed0.jsonl)
and requires byte-identical output, so the rerun environment is shown to
reproduce the published baseline runs.

Runtime pin: the published baseline cells were generated under an Ollama build
that Day-55 (Ollama 0.32.1) reproduced byte-identically on 90-99% of outputs;
Ollama 0.34.0 reproduces only 4/20 of the repro-check prompts. Run this
against Ollama 0.32.1 (e.g. the official CLI binary, `OLLAMA_HOST=127.0.0.1:11435
ollama serve`, with OLLAMA_HOST pointing the client at it). The server version is
recorded in the summary and the repro check gates the run.

Verifier pin: verification uses mlir-opt from LLVM llvmorg-19.1.7 through
scripts/env/bin/mlir-opt (container chosen by SLM_MLIR_CONTAINER). A second
gate re-verifies cached July generations (results/verify_cache.sqlite, written
by the pinned container during the published runs) and requires identical
return codes and byte-identical stderr before any new output is verified.

Output: results/day61/frozen_c1only_arith.jsonl, results/day61/frozen_c1only_summary.json
Writes NEW files only; no released artifact is modified.
"""
from __future__ import annotations

import argparse
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

from scripts.day18_30b_c3_rejection import _prompts_arith, _build_prompt_arith

MAX_TOKENS = 256
N_RETRIES = 5
SEED = 0
N_ARITH = 200

CELLS = [
    ("codellama-c1",   "codellama:34b-instruct-q4_K_M",    "c1"),
    ("granite-c1",     "granite-code:34b-instruct-q4_K_M", "c1"),
    ("codellama-free", "codellama:34b-instruct-q4_K_M",    "none"),
    ("granite-free",   "granite-code:34b-instruct-q4_K_M", "none"),
]


def _generate(prompt: str, model: str, constraint: str, seed: int) -> tuple[str, int]:
    """Day-34 rejection loop with the gate set by `constraint` (c1 / c1_c3 / none)."""
    if constraint == "none":
        raw, _ = ollama_generate_raw(prompt=prompt, model=model, max_tokens=MAX_TOKENS,
                                     temperature=0.2, seed=seed)
        return raw, 1
    last = ""
    for attempt in range(1, N_RETRIES + 1):
        s = seed + (attempt - 1) * 1000
        raw, _ = ollama_generate_raw(prompt=prompt, model=model, max_tokens=MAX_TOKENS,
                                     temperature=0.3 if attempt > 1 else 0.2, seed=s)
        last = raw
        if not is_parse_valid(raw):
            continue
        if constraint == "c1":
            return raw, attempt
        ok, _rep = accept_or_reject(raw)
        if ok:
            return raw, attempt
    return last, N_RETRIES


def repro_check(n: int) -> dict:
    published = [json.loads(l) for l in open("results/day51_seed0_n200/multiseed_seed0.jsonl")]
    published = sorted((r for r in published
                        if r["model"] == "codellama-c1c3" and r["dialect"] == "arith+func"),
                       key=lambda r: int(r["prompt_id"]))
    prompts = _prompts_arith(N_ARITH)
    same = 0
    diffs = []
    for r in published[:n]:
        i = int(r["prompt_id"])
        assert prompts[i] == r["nl"], f"pool misalignment at prompt {i}"
        raw, _ = _generate(_build_prompt_arith(prompts[i]), "codellama:34b-instruct-q4_K_M", "c1_c3", 0)
        if raw == r["generated"]:
            same += 1
        else:
            diffs.append(i)
    res = {"n": n, "byte_identical": same, "differing_prompt_ids": diffs}
    print(f"[day61] repro check: {same}/{n} byte-identical to published codellama C1+C3 seed 0",
          file=sys.stderr)
    return res


def verifier_check(n: int = 100) -> dict:
    """Re-verify n cached July generations through the live wrapper; require identical output."""
    import random
    import subprocess
    from scripts.env.verify_cache import MLIR_OPT, VerifyCache
    cache = VerifyCache()
    texts = []
    for p in ("results/day53/e10_ladder_frozen.jsonl", "results/day51_seed0_n200/multiseed_seed0.jsonl"):
        texts += [json.loads(l)["generated"] for l in open(p) if l.strip()]
    texts = [t for t in dict.fromkeys(texts) if cache.get(t, ("--verify-diagnostics",))]
    random.Random(0).shuffle(texts)
    same = 0
    for t in texts[:n]:
        want = cache.get(t, ("--verify-diagnostics",))
        got = subprocess.run([str(MLIR_OPT), "--verify-diagnostics"], input=t,
                             capture_output=True, text=True, timeout=60)
        same += got.returncode == want["returncode"] and got.stderr == want["stderr"]
    res = {"container": os.environ.get("SLM_MLIR_CONTAINER", "slm-mlir-llvm"), "n": min(n, len(texts)),
           "identical": same}
    print(f"[day61] verifier check: {same}/{res['n']} identical to cached July verifier output", file=sys.stderr)
    return res


def run(out_path: Path, summary_path: Path, repro_n: int) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    import requests
    from eval.baselines.run_ollama import OLLAMA_HOST
    summary["ollama_server_version"] = requests.get(f"{OLLAMA_HOST}/api/version", timeout=10).json()["version"]
    if repro_n and "repro_check" not in summary:
        summary["repro_check"] = repro_check(repro_n)
        summary_path.write_text(json.dumps(summary, indent=2))
        if summary["repro_check"]["byte_identical"] < repro_n:
            print("[day61] STOP: environment does not reproduce the published outputs", file=sys.stderr)
            return 2

    summary["verifier_check"] = verifier_check()
    summary_path.write_text(json.dumps(summary, indent=2))
    if summary["verifier_check"]["identical"] < summary["verifier_check"]["n"]:
        print("[day61] STOP: verifier differs from the pinned July verifier", file=sys.stderr)
        return 3

    prompts = _prompts_arith(N_ARITH)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["model"], r["prompt_id"]))
    t_run = time.perf_counter()
    with out_path.open("a") as fout:
        for nick, model, constraint in CELLS:
            print(f"\n[day61] {nick} arith+func n={len(prompts)} seed={SEED}", file=sys.stderr)
            for i, nl in enumerate(prompts):
                if (nick, i) in done:
                    continue
                t0 = time.perf_counter()
                raw, attempts = _generate(_build_prompt_arith(nl), model, constraint, SEED)
                pv = is_parse_valid(raw)
                vv = pv and verify(raw)["returncode"] == 0
                fout.write(json.dumps({
                    "model": nick, "backend": "ollama", "constraint": constraint,
                    "max_tokens": MAX_TOKENS, "dialect": "arith+func", "seed": SEED,
                    "pool": "frozen_released", "prompt_id": i, "nl": nl,
                    "generated": raw, "parse_valid": pv, "verify_valid": bool(vv),
                    "attempts": attempts, "dt": time.perf_counter() - t0,
                }) + "\n")
                fout.flush()
                if (i + 1) % 20 == 0:
                    print(f"  [{nick}] {i+1}/{len(prompts)} elapsed={(time.perf_counter()-t_run)/60:.1f}min",
                          file=sys.stderr)

    rows = [json.loads(l) for l in out_path.read_text().splitlines()]
    for nick, _, constraint in CELLS:
        sub = [r for r in rows if r["model"] == nick]
        if sub:
            summary[nick] = {
                "n": len(sub), "constraint": constraint,
                "parse_valid": sum(r["parse_valid"] for r in sub),
                "verify_valid": sum(r["verify_valid"] for r in sub),
                "verify_rate": round(100 * sum(r["verify_valid"] for r in sub) / len(sub), 1),
                "mean_attempts": round(sum(r["attempts"] for r in sub) / len(sub), 2),
            }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[day61] ALL DONE in {(time.perf_counter()-t_run)/60:.1f} min", file=sys.stderr)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro-n", type=int, default=20)
    a = ap.parse_args()
    sys.exit(run(Path("results/day61/frozen_c1only_arith.jsonl"),
                 Path("results/day61/frozen_c1only_summary.json"), a.repro_n))
