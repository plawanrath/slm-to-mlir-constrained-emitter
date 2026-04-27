"""Phase F F8: run SmolLM2 × {free, C1, C1+C3} on the expanded held-
out-200 corpus (n=200). Tighter CI than the n=50 F6 run."""
from __future__ import annotations
import json, os, sys
from pathlib import Path

REPO = "/Users/plawanrath/Documents/GitHub-public/slm-to-mlir-constrained-emitter"
os.chdir(REPO); sys.path.insert(0, REPO)

import scripts.day32_stablehlo_smoke as d32


def _heldout_200_prompts():
    out = []
    for i, p in enumerate(sorted(Path("eval/benchmarks/stablehlo_held_out_200/examples").glob("*.json"))):
        r = json.loads(p.read_text())
        out.append((i, r["nl"], r["mlir"]))
    return out


d32._load_prompts = _heldout_200_prompts

if __name__ == "__main__":
    d32.run(Path("results/day_f8/heldout_200_matrix.jsonl"))
