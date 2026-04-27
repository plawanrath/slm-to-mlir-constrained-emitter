"""Day-39 PM: full-n multi-seed re-runs.

Same 5 critical cells as Day 34, but n=200 (arith+func) / n=125 (linalg)
instead of the Day-34 n=100 first-100 subset. This resolves the Phase E
critique "subset selection artifact concern" on the multi-seed CIs.

Output: results/day39pm/multiseed_fulln.jsonl
"""
from __future__ import annotations
import os, sys
from pathlib import Path

REPO = "/Users/plawanrath/Documents/GitHub-public/slm-to-mlir-constrained-emitter"
os.chdir(REPO); sys.path.insert(0, REPO)
os.environ["PATH"] = f"{REPO}/scripts/env/bin:" + os.environ["PATH"]

# Override N_PER_CELL by monkey-patching before importing run().
import scripts.day34_multiseed as m34
# Arith full n=200, linalg full n=125 — match Phase-A submission artifact.
# The module uses a single N_PER_CELL for both; since linalg has only 125
# prompts total, using 200 won't overflow (the function caps at available).
m34.N_PER_CELL = 200

if __name__ == "__main__":
    m34.run(Path("results/day39pm/multiseed_fulln.jsonl"))
