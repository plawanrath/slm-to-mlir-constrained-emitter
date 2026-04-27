"""In-line C3 sampling loop (Day 25 of ADR-0008).

High-level flow at each decode step:
  1. Query CoupledState for legal_next_terminals.
  2. For each legal terminal, compute the token-id set that could be its next
     BPE token (either literal-match, regex-prefix, or in-scope-name-trie).
  3. Union → `token_mask: set[int]`.
  4. Apply mask to the model's logit vector (MLX) or to a numpy vector (for
     text-level simulation), sample an id.
  5. Append the id to accumulated output; try to close zero or more terminals
     by re-lexing the accumulated-since-last-terminal substring; advance
     CoupledState on each closed terminal.
  6. Repeat until `CoupledState.is_accept()` AND the accumulated substring is
     empty (no in-flight terminal).

The `generate_text_mock(prompt_text, oracle_token_stream)` entry point lets us
unit-test the sampling loop without a real model: callers provide the exact
token-id sequence an "oracle" sampler would emit, and we verify that the loop
correctly advances CoupledState and closes terminals. This is how Day 26's
equivalence test will validate the implementation before we plug in the real
MLX forward pass.

The `mlx_generate(prompt, model_id, max_tokens)` entry point is the production
path — loads an mlx-lm model + tokenizer and runs the full forward+sample loop.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from .cfg_automaton import CFGAutomaton, TerminalSpec
from .coupled_state import CoupledState, SSARole, OpPhase
from .bpe_masks import InScopeNameTrie, regex_prefix_mask


@dataclass
class InlineC3Result:
    text: str
    num_tokens: int
    closed_terminals: int
    accepted: bool
    backend: str


def _terminal_mask(
    tokenizer: Any, spec: TerminalSpec, partial: str,
) -> set[int]:
    """Token ids that could be the NEXT BPE token of a terminal whose partial
    text consumed so far is `partial`."""
    if spec.is_literal:
        assert spec.literal is not None
        full = spec.literal
        if not full.startswith(partial):
            return set()
        remaining = full[len(partial):]
        # Any token whose text is a prefix of `remaining` (non-empty) is legal.
        out: set[int] = set()
        for i in range(tokenizer.vocab_size):
            try:
                text = tokenizer.decode([i])
            except Exception:
                continue
            if text and remaining.startswith(text):
                out.add(i)
        return out
    return regex_prefix_mask(tokenizer, spec.pattern, partial)


def _try_close_terminal(
    spec: TerminalSpec, partial: str,
) -> Optional[str]:
    """If `partial` is a complete match for the terminal, return the matched
    text; else None.

    For literal terminals: partial == literal.
    For regex terminals: partial matches fullmatch AND cannot be extended
    meaningfully (approximation: return as soon as fullmatch).
    """
    if spec.is_literal:
        if partial == spec.literal:
            return partial
        return None
    try:
        if re.fullmatch(spec.pattern, partial):
            return partial
    except re.error:
        pass
    return None


def _consume_one_terminal(
    spec: TerminalSpec, partial: str,
) -> Optional[tuple[str, str]]:
    """Try to consume one complete terminal from the start of `partial`.

    Returns (matched_text, leftover) or None.

    Handles the multi-terminal BPE case: a single BPE token may decode to
    text that spans two or more grammar terminals (e.g. "module " carries
    MODULE + WS; " {" carries WS + LBRACE). The old `_try_close_terminal`
    only closed when *all* of `partial` matched the terminal, which caused
    the close-loop to silently drop leftover characters when it set
    `partial = ""` after closing. This function instead returns the
    leftover so the caller can keep consuming.

    For literals: closes if `partial` starts with the literal.
    For regex: closes only when the match is *maximal within `partial`* —
    i.e., extending by one more character from `partial` would not extend
    the match. If the match consumes all of `partial`, we defer (more
    tokens may still extend it).
    """
    if not partial:
        return None
    if spec.is_literal:
        lit = spec.literal
        assert lit is not None
        if partial.startswith(lit):
            return lit, partial[len(lit):]
        return None
    try:
        m = re.match(spec.pattern, partial)
        if not m or m.end() == 0:
            return None
        end = m.end()
        if end == len(partial):
            return None
        m2 = re.match(spec.pattern, partial[: end + 1])
        if m2 and m2.end() > end:
            return None
        return m.group(0), partial[end:]
    except re.error:
        return None


def generate_text_mock(
    prompt: str,
    automaton: CFGAutomaton,
    tokenizer: Any,
    oracle: Iterator[int],
    max_steps: int = 512,
) -> InlineC3Result:
    """Mock sampler. Drives the sampling loop using a caller-provided token
    stream (the `oracle`). Raises if the oracle emits a token outside the
    legal mask at any step.
    """
    state = CoupledState.fresh(automaton)
    generated_ids: list[int] = []
    partial = ""       # text accumulated since last closed terminal
    closed = 0

    for _ in range(max_steps):
        if state.is_accept() and not partial:
            return InlineC3Result(
                text=tokenizer.decode(generated_ids),
                num_tokens=len(generated_ids),
                closed_terminals=closed,
                accepted=True, backend="mock",
            )
        # Build union mask across legal terminals
        legal = state.cfg.legal_terminal_specs()
        mask: set[int] = set()
        for spec in legal:
            mask |= _terminal_mask(tokenizer, spec, partial)
        if not mask:
            break
        try:
            tid = next(oracle)
        except StopIteration:
            break
        if tid not in mask:
            raise RuntimeError(
                f"Oracle token {tid} (={tokenizer.decode([tid])!r}) not in legal mask"
            )
        generated_ids.append(tid)
        partial += tokenizer.decode([tid])

        # Attempt to close one or more terminals greedily
        while True:
            closed_any = False
            for spec in state.cfg.legal_terminal_specs():
                m = _try_close_terminal(spec, partial)
                if m is not None:
                    state = state.advance(spec.name, m)
                    partial = ""
                    closed += 1
                    closed_any = True
                    break
            if not closed_any:
                break

    return InlineC3Result(
        text=tokenizer.decode(generated_ids),
        num_tokens=len(generated_ids),
        closed_terminals=closed,
        accepted=state.is_accept() and not partial,
        backend="mock",
    )
