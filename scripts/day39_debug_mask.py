"""Debug: what's in the mask at step 1 (after MODULE closed)?"""
from __future__ import annotations
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[1])
os.chdir(REPO); sys.path.insert(0, REPO)

from decoder.inline_c3 import CFGAutomaton
from decoder.inline_c3.coupled_state import CoupledState
from decoder.inline_c3.generate import _terminal_mask

from mlx_lm import load as mlx_load

MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

model, tok = mlx_load(MODEL)

GRAMMAR = "grammar/mlir_gen_c1c2.lark"
automaton = CFGAutomaton(GRAMMAR)
state = CoupledState.fresh(automaton)

print("State 0 legal:", sorted(state.cfg.legal_next_terminals()))
state = state.advance("MODULE", "module")
print("State after MODULE legal:", sorted(state.cfg.legal_next_terminals()))

specs = state.cfg.legal_terminal_specs()
for spec in specs:
    print(f"  terminal: name={spec.name} literal={spec.is_literal} pattern={spec.pattern!r}")

mask = set()
for spec in specs:
    m = _terminal_mask(tok, spec, "")
    print(f"  mask[{spec.name}] size = {len(m)}")
    sample = sorted(m)[:8]
    print(f"    sample ids: {sample}")
    for tid in sample:
        print(f"      id={tid} text={tok.decode([tid])!r}")
    mask |= m
print(f"TOTAL mask size: {len(mask)} / vocab={tok.vocab_size}")
print("Checking presence of 'module' token:")
for text in ["module", " module", "module ", "modul"]:
    ids = tok.encode(text)
    print(f"  encode({text!r}) = {ids}")
    for tid in ids:
        print(f"    id {tid} ({tok.decode([tid])!r}) in mask? {tid in mask}")
