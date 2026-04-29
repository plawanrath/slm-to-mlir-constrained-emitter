"""Day-51 / Item 1: Re-run seed 0 at uniform n on all 8 (cell × dialect) combos.

Fixes the mixed-n bug in `results/day34/multiseed_summary.json` where seed-0
numbers were captured at n=100 while seeds 1,2 (in `results/day39pm/`) were
captured at n=200 (arith+func) / n=125 (linalg). Averaging across uneven-n
seeds is a methodology error.

This script reruns seed 0 only, at the uniform n used for seeds 1,2:
  n=200 for arith+func, n=125 for linalg.

Cells (8 = 4 systems × 2 dialects):
  SmolLM2-1.7B     + C1+C2+C3  (mlx)            × {arith+func, linalg}
  CodeLlama-34B    + C1+C3     (ollama Q4_K_M)  × {arith+func, linalg}
  Granite-Code-34B + C1+C3     (ollama Q4_K_M)  × {arith+func, linalg}
  StarCoder2-15B   + C1+C3     (ollama)         × {arith+func, linalg}

Output: results/day51_seed0_n200/multiseed_seed0.jsonl
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
os.chdir(REPO)
sys.path.insert(0, REPO)
os.environ["PATH"] = f"{REPO}/scripts/env/bin:" + os.environ["PATH"]

import scripts.day34_multiseed as m34

# Per-dialect n. day34.run uses one N_PER_CELL for both dialects (it caps
# linalg at the available 125 prompts), so 200 here is correct for both.
m34.N_PER_CELL = 200

# Seed 0 only.
m34.SEEDS = [0]

# All 8 cells (5 from day39pm + 3 from day50, full set at seed 0).
m34.CELLS = [
    ("smollm2-c1c2c3",  "HuggingFaceTB/SmolLM2-1.7B-Instruct", "arith+func", "mlx-c1c2c3"),
    ("smollm2-c1c2c3",  "HuggingFaceTB/SmolLM2-1.7B-Instruct", "linalg",     "mlx-c1c2c3"),
    ("granite-c1c3",    "granite-code:34b-instruct-q4_K_M",     "arith+func", "ollama-c1c3"),
    ("granite-c1c3",    "granite-code:34b-instruct-q4_K_M",     "linalg",     "ollama-c1c3"),
    ("codellama-c1c3",  "codellama:34b-instruct-q4_K_M",        "arith+func", "ollama-c1c3"),
    ("codellama-c1c3",  "codellama:34b-instruct-q4_K_M",        "linalg",     "ollama-c1c3"),
    ("starcoder2-c1c3", "starcoder2:instruct",                   "arith+func", "ollama-c1c3"),
    ("starcoder2-c1c3", "starcoder2:instruct",                   "linalg",     "ollama-c1c3"),
]

if __name__ == "__main__":
    out = Path("results/day51_seed0_n200/multiseed_seed0.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    m34.run(out)
