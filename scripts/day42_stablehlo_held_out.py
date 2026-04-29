"""Day-42: Re-run StableHLO matrix on the Day-41 held-out corpus.

Same cells and protocol as Day 32 (SmolLM2 × {free, C1, C1+C3}), but
the benchmark is `eval/benchmarks/stablehlo_held_out_50/examples/`
(programmatically generated, no grammar-design influence).

If the held-out numbers hold within a few pp of stablehlo_spec_30,
the StableHLO lead is corpus-agnostic → robust claim.

Output: results/day42/stablehlo_held_out.jsonl
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
os.chdir(REPO); sys.path.insert(0, REPO)
os.environ["PATH"] = f"{REPO}/scripts/env/bin:" + os.environ["PATH"]

import scripts.day32_stablehlo_smoke as d32


def _held_out_prompts() -> list[tuple[int, str, str]]:
    out = []
    for i, p in enumerate(sorted(
        Path("eval/benchmarks/stablehlo_held_out_50/examples").glob("*.json")
    )):
        r = json.loads(p.read_text())
        out.append((i, r["nl"], r["mlir"]))
    return out


d32._load_prompts = _held_out_prompts

if __name__ == "__main__":
    d32.run(Path("results/day42/stablehlo_held_out.jsonl"))
