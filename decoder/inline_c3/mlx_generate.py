"""Real MLX in-line C3 generator — production path for Days 26/39.

Wraps mlx_lm.generate.generate_step with a logits_processor that:

  1. Syncs the external CoupledState with any tokens generated since the last
     processor call (close-terminals on-the-fly, preserving leftover text).
  2. Consults the fresh CoupledState for legal next terminals.
  3. For each legal terminal, computes the token-id set that could be
     the next token given the accumulated partial text.
  4. Unions these sets; masks all other vocab ids to -inf before sampling.

Stops when the CoupledState reports `is_accept()` AND `partial` is empty.

Day-39 fix (vs Day 26): syncing moved INTO the processor (before mask
computation) and the close-loop now uses `_consume_one_terminal`, which
preserves leftover text when a single BPE token decodes to text spanning
multiple grammar terminals (e.g. " {" carrying WS + LBRACE). The old
close-loop zeroed `partial` after any close, silently dropping the
leftover char — that was the real "terminal-closure race" observed on
Day 26 (the `modulemodule...` output).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache

from .cfg_automaton import CFGAutomaton
from .coupled_state import CoupledState
from .generate import _terminal_mask, _consume_one_terminal


def _is_complete_match(spec, partial: str) -> bool:
    """True iff `partial` is exactly a complete match of `spec`'s terminal."""
    if not partial:
        return False
    if spec.is_literal:
        return partial == spec.literal
    try:
        m = re.fullmatch(spec.pattern, partial)
        return m is not None
    except re.error:
        return False


def _compute_mask(tokenizer, state, partial: str) -> set[int]:
    """Build the token-mask for the current step.

    Primary component: tokens that extend `partial` to a valid prefix of
    some legal terminal. Secondary component (LOOKAHEAD): if `partial`
    already completes some legal terminal, include tokens that would
    START the terminals legal in the state AFTER that close. Without the
    lookahead component, the model gets trapped once an in-flight
    regex-terminal (e.g. WS) has consumed enough characters to match
    but no "exit" token is permitted — observed on Day 39 as runaway
    whitespace after the first MODULE.
    """
    legal_specs = state.cfg.legal_terminal_specs()
    mask: set[int] = set()
    for spec in legal_specs:
        mask |= _terminal_mask(tokenizer, spec, partial)
    if partial:
        for spec in legal_specs:
            if not _is_complete_match(spec, partial):
                continue
            try:
                next_state = state.advance(spec.name, partial)
            except Exception:
                continue
            for next_spec in next_state.cfg.legal_terminal_specs():
                mask |= _terminal_mask(tokenizer, next_spec, "")
    return mask


@dataclass
class MLXGenerateResult:
    text: str
    tokens: list[int]
    accepted: bool
    closed_terminals: int
    backend: str = "mlx"


def _sync_with_token(state_ref: dict, tokenizer, tid: int) -> None:
    """Incorporate one newly-sampled token into `state_ref`.

    Appends decoded text to `partial`, then greedily consumes complete
    terminals from the start of `partial`, advancing CoupledState on each
    closure. Leftover chars remain in `partial` for the next token.
    """
    piece = tokenizer.decode([tid])
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
            except Exception:
                continue
            state_ref["partial"] = leftover
            state_ref["closed"] += 1
            progressed = True
            break
        if not progressed:
            break


def mlx_generate_coupled(
    prompt: str,
    model_id: str,
    *,
    grammar_path: str | None = None,
    max_tokens: int = 600,
    verbose: bool = False,
    temperature: float = 0.0,
    seed: int | None = None,
) -> MLXGenerateResult:
    """Single-pass in-line C3 generation on MLX.

    Args:
      prompt: text to prepend (the model completes from here).
      model_id: a mlx-lm-compatible model id (e.g. "HuggingFaceTB/SmolLM2-1.7B-Instruct").
      grammar_path: LARK grammar file. Defaults to grammar/mlir_gen_c1c2.lark.
      max_tokens: generation cap.
      temperature: 0 = greedy (argmax). >0 enables temperature sampling.
      seed: optional RNG seed for reproducibility under temperature>0.

    For temperature-annealed multi-attempt generation, use
    :func:`mlx_generate_coupled_annealed`.
    """
    try:
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler
    except ImportError as e:
        raise RuntimeError("mlx-lm unavailable; run on Apple Silicon") from e

    gram = grammar_path or "grammar/mlir_gen_c1c2.lark"
    automaton = CFGAutomaton(gram)

    model, tokenizer = _load_mlx_cached(model_id)
    prompt_ids = mx.array(tokenizer.encode(prompt))
    prompt_len = int(prompt_ids.shape[0])

    if seed is not None:
        mx.random.seed(seed)

    state_ref: dict = {
        "state": CoupledState.fresh(automaton),
        "partial": "",
        "consumed": 1,
        "closed": 0,
    }

    def _processor(tokens, logits):
        nt = int(tokens.shape[0]) if hasattr(tokens, "shape") else len(tokens)
        consumed = state_ref["consumed"]
        if nt > consumed:
            for idx in range(consumed, nt):
                _sync_with_token(state_ref, tokenizer, int(tokens[idx]))
            state_ref["consumed"] = nt

        s = state_ref["state"]
        p = state_ref["partial"]
        mask = _compute_mask(tokenizer, s, p)
        if not mask:
            return logits
        neg_inf = mx.array(-math.inf, dtype=logits.dtype)
        out = mx.full(logits.shape, neg_inf, dtype=logits.dtype)
        allowed_ids = mx.array(sorted(mask))
        if logits.ndim == 1:
            out[allowed_ids] = logits[allowed_ids]
        else:
            out[..., allowed_ids] = logits[..., allowed_ids]
        return out

    sampler = make_sampler(temp=temperature) if temperature > 0 else None

    tokens: list[int] = []
    step_kwargs = dict(prompt=prompt_ids, model=model,
                       max_tokens=max_tokens, logits_processors=[_processor])
    if sampler is not None:
        step_kwargs["sampler"] = sampler
    for i, (token, _lp) in enumerate(generate_step(**step_kwargs)):
        tid = int(token)
        if tid == tokenizer.eos_token_id:
            break
        tokens.append(tid)
        # NOTE: do NOT call _sync_with_token here. mlx_lm.generate_step
        # computes next_y (which invokes the processor for token N+1)
        # BEFORE yielding token N, so the processor has already synced
        # token N by the time we receive it. Calling _sync here would
        # double-advance the CoupledState. The processor-only sync is
        # correct and idempotent via state_ref["consumed"].
        if state_ref["state"].is_accept() and not state_ref["partial"]:
            break
        if verbose and (i + 1) % 20 == 0:
            print(f"  step {i+1}: closed={state_ref['closed']} partial={state_ref['partial']!r}")

    # Final sync: the processor's consumed counter tracks the number of
    # tokens it has seen via mlx_lm's processor-arg (prompt_tail + all
    # generated). The driver's `tokens` list counts only generated
    # tokens. After the generator exits, the last yielded token may not
    # have triggered another processor call, so sync any tail.
    # consumed - 1 = generated-tokens-synced (subtract the 1-token prompt tail).
    synced_generated = state_ref["consumed"] - 1
    if len(tokens) > synced_generated:
        for tid_i in tokens[synced_generated:]:
            _sync_with_token(state_ref, tokenizer, int(tid_i))
        state_ref["consumed"] = 1 + len(tokens)

    text = tokenizer.decode(tokens)
    accepted = state_ref["state"].is_accept() and not state_ref["partial"]
    return MLXGenerateResult(
        text=text, tokens=tokens, accepted=accepted,
        closed_terminals=state_ref["closed"],
    )


@lru_cache(maxsize=2)
def _load_mlx_cached(model_id: str):
    from mlx_lm import load as mlx_load
    return mlx_load(model_id)


def mlx_generate_coupled_annealed(
    prompt: str,
    model_id: str,
    *,
    grammar_path: str | None = None,
    max_tokens: int = 600,
    temperatures: tuple[float, ...] = (0.0, 0.4, 0.8),
    seed: int = 0,
    accept_fn=None,
) -> MLXGenerateResult:
    """Temperature-annealed in-line C3 generation.

    Runs :func:`mlx_generate_coupled` up to ``len(temperatures)`` times,
    each with a different temperature (default: 0.0, 0.4, 0.8). Returns
    the first result where the coupled state reached an accept state
    with empty partial, OR where ``accept_fn(result.text)`` is true.

    This is the inline-C3 analog of 5-retry rejection sampling: it
    recovers from greedy get-stuck cases by exploring with higher
    temperature on subsequent attempts, while the theorem-3 soundness
    guarantee still holds (each attempt produces a string in
    $L(\\mathcal{G})$ if it reaches accept).

    Args:
      accept_fn: optional function (str) -> bool. If provided, the
        first result with accept_fn(text)==True is returned. Otherwise
        the first result with `result.accepted` is returned.
    """
    last: MLXGenerateResult | None = None
    for attempt, temp in enumerate(temperatures):
        res = mlx_generate_coupled(
            prompt=prompt, model_id=model_id,
            grammar_path=grammar_path, max_tokens=max_tokens,
            temperature=temp, seed=seed + attempt * 1000,
        )
        last = res
        ok = res.accepted
        if accept_fn is not None and res.text:
            ok = ok and accept_fn(res.text)
        if ok:
            return res
    assert last is not None
    return last
