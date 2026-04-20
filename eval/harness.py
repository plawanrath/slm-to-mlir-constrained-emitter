"""Main eval harness.

Runs a grid over (model × constraint × dialect × seed), evaluates each
generation against mlir-opt --verify, writes one JSONL row per cell-sample.

Usage:
    python -m eval.harness \
        --eval-set data/processed/l3_tests_held_out.jsonl \
        --models phi-3.5-mini smollm2-1.7b \
        --constraints none c1 c1_c2 \
        --dialects arith+func linalg \
        --seeds 0 1 2 \
        --n-per-cell 1000 \
        --out results/matrix.jsonl

The harness is deliberately linear + subprocess-free for the main matrix.
Parallelization is by seed (trivially independent) and by running the 30B
Ollama baselines as a separate harness invocation.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

from decoder.generate import ConstraintLevel, GenerateRequest, generate
from scripts.env.verify_cache import VerifyCache, verify


def _load_eval_set(path: Path, n: int, seed: int) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:n]


def _build_prompt(record: dict, dialect: str) -> str:
    """Construct the NL prompt from an eval record.

    L1 records have `docstring`; L3 records have `weak_nl`; MLIR-Spec-150 has `nl`.
    """
    nl = record.get("nl") or record.get("docstring") or record.get("weak_nl") or ""
    system = (
        "You are an MLIR code emitter. Given a natural-language description, "
        f"emit a valid MLIR module using only the {dialect} dialect(s). "
        "Output only MLIR, no prose."
    )
    return f"{system}\n\nDescription:\n{nl}\n\nMLIR:\n"


def _run_cell(
    model: str,
    backend: str,
    constraint: ConstraintLevel,
    dialect: str,
    seed: int,
    eval_records: list[dict],
    cache: VerifyCache,
) -> list[dict]:
    rows = []
    for idx, rec in enumerate(eval_records):
        prompt = _build_prompt(rec, dialect)
        req = GenerateRequest(
            prompt=prompt, model=model, constraint=constraint,
            dialect=dialect, backend=backend, seed=seed,
        )
        try:
            result = generate(req)
        except Exception as e:  # noqa: BLE001 — log + continue
            rows.append({
                "model": model, "backend": backend, "constraint": constraint.value,
                "dialect": dialect, "seed": seed, "sample_idx": idx,
                "prompt_id": rec.get("id", idx), "error": str(e),
                "passed": False, "num_tokens": 0, "verify_stderr": "",
            })
            continue
        v = verify(result.text, cache=cache)
        rows.append({
            "model": model, "backend": backend, "constraint": constraint.value,
            "dialect": dialect, "seed": seed, "sample_idx": idx,
            "prompt_id": rec.get("id", idx),
            "generated": result.text,
            "num_tokens": result.num_tokens,
            "passed": v["returncode"] == 0,
            "verify_stderr": v["stderr"][:4000],
            "verify_cached": v.get("cached", False),
        })
    return rows


def _backend_for_model(model_name: str) -> str:
    if "phi" in model_name.lower() or "smollm" in model_name.lower() or "gemma" in model_name.lower():
        return "mlx"
    if "codellama" in model_name.lower() or "granite" in model_name.lower():
        return "ollama"
    return "mock"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True,
                    help="Model identifiers (MLX paths or Ollama names).")
    ap.add_argument("--constraints", nargs="+",
                    default=["none", "c1", "c1_c2"],
                    choices=[c.value for c in ConstraintLevel])
    ap.add_argument("--dialects", nargs="+", default=["arith+func", "linalg"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-per-cell", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("results/matrix.jsonl"))
    ap.add_argument("--backend-override", default=None,
                    help='"mlx" | "ollama" | "mock" — overrides auto-detection.')
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache = VerifyCache()

    with args.out.open("w") as f:
        for model, constraint_s, dialect, seed in itertools.product(
            args.models, args.constraints, args.dialects, args.seeds
        ):
            constraint = ConstraintLevel(constraint_s)
            backend = args.backend_override or _backend_for_model(model)
            # design §5.2: Ollama baselines are {free, C1} only.
            if backend == "ollama" and constraint == ConstraintLevel.C1_C2:
                print(f"[harness] skip {model}×{constraint_s} (C2 not on Ollama)", file=sys.stderr)
                continue
            eval_records = _load_eval_set(args.eval_set, args.n_per_cell, seed)
            print(f"[harness] cell: model={model} cons={constraint_s} dia={dialect} seed={seed} n={len(eval_records)}", file=sys.stderr)
            for row in _run_cell(
                model, backend, constraint, dialect, seed, eval_records, cache
            ):
                f.write(json.dumps(row) + "\n")
            f.flush()

    cache.close()
    print(f"[harness] done → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
