"""Day-50: Baseline linalg multi-seed fill.

Day-39-PM ran SmolLM2 linalg at seeds 1,2 but skipped the 15B/34B
baselines. This script adds seeds 1,2 for the three missing baseline
linalg cells (CodeLlama-34B, Granite-Code-34B, StarCoder2-15B) under
the same C1+C3 rejection-sampling protocol.

Output: results/day50/baseline_linalg_multiseed.jsonl
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

# n=200 caps at available; linalg has 125 prompts, so this matches Day-39-PM.
m34.N_PER_CELL = 200

# Only the 3 missing baseline x linalg cells.
m34.CELLS = [
    ("granite-c1c3",    "granite-code:34b-instruct-q4_K_M",  "linalg", "ollama-c1c3"),
    ("codellama-c1c3",  "codellama:34b-instruct-q4_K_M",     "linalg", "ollama-c1c3"),
    ("starcoder2-c1c3", "starcoder2:instruct",                "linalg", "ollama-c1c3"),
]

# m34.SEEDS is already [1, 2]; we inherit.

if __name__ == "__main__":
    out = Path("results/day50/baseline_linalg_multiseed.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    m34.run(out)
