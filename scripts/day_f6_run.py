"""Phase F F6: run SmolLM2 × {free, C1} on the out-of-grammar corpus
and report the graceful-degradation story.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

REPO = "/Users/plawanrath/Documents/GitHub-public/slm-to-mlir-constrained-emitter"
os.chdir(REPO); sys.path.insert(0, REPO)

import scripts.day32_stablehlo_smoke as d32


def _out_of_grammar_prompts():
    out = []
    for i, p in enumerate(sorted(Path("eval/benchmarks/stablehlo_outofgrammar_25/examples").glob("*.json"))):
        r = json.loads(p.read_text())
        out.append((i, r["nl"], r["mlir"]))
    return out


d32._load_prompts = _out_of_grammar_prompts

if __name__ == "__main__":
    d32.run(Path("results/day_f6/outofgrammar_matrix.jsonl"))
