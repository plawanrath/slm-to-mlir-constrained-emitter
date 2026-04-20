"""Day-2 tokenizer shootout.

Compare Phi-3.5-mini, SmolLM2-1.7B, Gemma-2-2b tokenizers on a 10k-line MLIR
corpus (sampled from data/processed/l3_tests.jsonl if available, else from
llvm-project/mlir/test/**/*.mlir directly).

Metrics:
  - mean tokens / character
  - mean tokens / op-boundary (counted as occurrences of "^\\s*%[a-z0-9_]+ =" or
    op mnemonics with "." prefix)
  - fraction of MLIR op mnemonics that survive as a single token (the "clean
    mnemonic" rate — higher is better for constrained-decoding speed)

Writes results/tokenizer_shootout.json. ADR-0002 removed Llama-3.2-3B from the
candidate set; candidates are 3-wide. Primary/secondary SLM is locked by the
researcher after reading this output.

Usage:
    python scripts/tokenizer_shootout.py \
        --llvm-src /path/to/llvm-project \
        --out results/tokenizer_shootout.json \
        --sample-lines 10000
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path


CANDIDATES = {
    "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "gemma-2-2b": "google/gemma-2-2b-it",
}

MLIR_MNEMONICS = [
    "arith.addi", "arith.subi", "arith.muli", "arith.constant", "arith.cmpi",
    "arith.select", "arith.divsi", "arith.xori", "arith.andi", "arith.ori",
    "arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.cmpf",
    "func.func", "func.return", "func.call",
    "memref.alloc", "memref.dealloc", "memref.load", "memref.store",
    "memref.dim", "memref.cast",
    "linalg.matmul", "linalg.generic", "linalg.fill", "linalg.copy",
]


def _sample_mlir_lines(llvm_src: Path, n: int, seed: int) -> list[str]:
    lines: list[str] = []
    for path in (llvm_src / "mlir" / "test").rglob("*.mlir"):
        try:
            lines.extend(path.read_text().splitlines())
        except UnicodeDecodeError:
            continue
    rng = random.Random(seed)
    rng.shuffle(lines)
    return lines[:n]


def _tokens_per_char(tok, text: str) -> float:
    if not text:
        return 0.0
    ids = tok.encode(text, add_special_tokens=False)
    return len(ids) / max(len(text), 1)


def _clean_mnemonic_rate(tok) -> float:
    single = 0
    for m in MLIR_MNEMONICS:
        ids = tok.encode(m, add_special_tokens=False)
        if len(ids) == 1:
            single += 1
    return single / len(MLIR_MNEMONICS)


def _tokens_per_op(tok, text: str) -> float:
    ids = tok.encode(text, add_special_tokens=False)
    ops = len(re.findall(r"\b(?:arith|func|memref|linalg)\.[a-z_]+", text))
    if ops == 0:
        return float("nan")
    return len(ids) / ops


def measure(model_id: str, samples: list[str]) -> dict:
    from transformers import AutoTokenizer  # type: ignore
    print(f"[shootout] loading {model_id}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    full = "\n".join(samples)
    per_char = _tokens_per_char(tok, full)
    per_op = _tokens_per_op(tok, full)
    clean_mn = _clean_mnemonic_rate(tok)
    vocab_size = tok.vocab_size
    return {
        "model_id": model_id,
        "vocab_size": vocab_size,
        "tokens_per_char": per_char,
        "tokens_per_op": per_op,
        "clean_mnemonic_rate": clean_mn,
        "sample_chars": len(full),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llvm-src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/tokenizer_shootout.json"))
    ap.add_argument("--sample-lines", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    samples = _sample_mlir_lines(args.llvm_src, args.sample_lines, args.seed)
    print(f"[shootout] sampled {len(samples)} lines", file=sys.stderr)

    results = {}
    for nick, model_id in CANDIDATES.items():
        try:
            results[nick] = measure(model_id, samples)
        except Exception as e:  # noqa: BLE001
            results[nick] = {"error": str(e)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    # Pretty-print ranking.
    print("\n==== Tokenizer Shootout ====", file=sys.stderr)
    print(f"{'model':<20} {'tok/char':>10} {'tok/op':>10} {'mnemonic@1':>12}", file=sys.stderr)
    for nick, r in results.items():
        if "error" in r:
            print(f"{nick:<20} ERROR: {r['error']}", file=sys.stderr)
            continue
        print(f"{nick:<20} {r['tokens_per_char']:>10.3f} {r['tokens_per_op']:>10.2f} {r['clean_mnemonic_rate']:>12.1%}", file=sys.stderr)


if __name__ == "__main__":
    main()
