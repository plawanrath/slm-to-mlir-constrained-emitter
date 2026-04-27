"""Live step-by-step trace of the MLX in-line decoder."""
from __future__ import annotations
import os, sys, math
REPO = "/Users/plawanrath/Documents/GitHub-public/slm-to-mlir-constrained-emitter"
os.chdir(REPO); sys.path.insert(0, REPO)

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.generate import generate_step

from decoder.inline_c3 import CFGAutomaton
from decoder.inline_c3.coupled_state import CoupledState
from decoder.inline_c3.generate import _terminal_mask, _consume_one_terminal
from decoder.inline_c3.mlx_generate import _compute_mask

MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
GRAMMAR = "grammar/mlir_gen_c1c2.lark"
MAX_TOKENS = 40

PROMPT = (
    "Emit the following MLIR function verbatim inside a module:\n\n"
    "module { func.func @f(%a : i32) -> i32 { "
    "%0 = arith.addi %a , %a : i32 "
    "return %0 : i32 }}\n\n"
    "Output only the MLIR, no prose.\n"
)


def main() -> int:
    model, tok = mlx_load(MODEL)
    automaton = CFGAutomaton(GRAMMAR)
    prompt_ids = mx.array(tok.encode(PROMPT))
    prompt_len = int(prompt_ids.shape[0])

    state_ref = {
        "state": CoupledState.fresh(automaton),
        "partial": "",
        "consumed": 1,
        "closed": 0,
    }

    def _sync_with_token(tid: int) -> None:
        piece = tok.decode([tid])
        state_ref["partial"] += piece
        while state_ref["partial"]:
            progressed = False
            for spec in state_ref["state"].cfg.legal_terminal_specs():
                result = _consume_one_terminal(spec, state_ref["partial"])
                if result is None:
                    continue
                matched, leftover = result
                try:
                    state_ref["state"] = state_ref["state"].advance(spec.name, matched)
                except Exception as e:
                    continue
                state_ref["partial"] = leftover
                state_ref["closed"] += 1
                progressed = True
                break
            if not progressed:
                break

    step_counter = [0]

    def _processor(tokens, logits):
        step_counter[0] += 1
        step = step_counter[0]
        nt = int(tokens.shape[0])
        consumed = state_ref["consumed"]
        if nt > consumed:
            for idx in range(consumed, nt):
                _sync_with_token(int(tokens[idx]))
            state_ref["consumed"] = nt

        s = state_ref["state"]
        p = state_ref["partial"]
        legal_specs = s.cfg.legal_terminal_specs()
        mask = _compute_mask(tok, s, p)

        legal_names = sorted([sp.name for sp in legal_specs])
        print(f"[proc step={step}] nt={nt} consumed={consumed} "
              f"closed={state_ref['closed']} "
              f"partial={p!r} legal={legal_names} mask_size={len(mask)}")
        if mask:
            sample = sorted(mask)[:5]
            for tid in sample:
                print(f"    mask id {tid} = {tok.decode([tid])!r}")

        if not mask:
            print("  (empty mask — returning raw logits)")
            return logits

        neg_inf = mx.array(-math.inf, dtype=logits.dtype)
        out = mx.full(logits.shape, neg_inf, dtype=logits.dtype)
        allowed_ids = mx.array(sorted(mask))
        if logits.ndim == 1:
            out[allowed_ids] = logits[allowed_ids]
        else:
            out[..., allowed_ids] = logits[..., allowed_ids]
        return out

    generated = []
    for i, (token, _lp) in enumerate(generate_step(
        prompt=prompt_ids,
        model=model,
        max_tokens=MAX_TOKENS,
        logits_processors=[_processor],
    )):
        tid = int(token)
        if tid == tok.eos_token_id:
            break
        generated.append(tid)
        print(f"  YIELD: tid={tid} text={tok.decode([tid])!r}")
        if state_ref["state"].is_accept() and not state_ref["partial"]:
            print("  [accepted — break]")
            break

    print(f"\nfinal text: {tok.decode(generated)!r}")
    print(f"closed_terminals = {state_ref['closed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
