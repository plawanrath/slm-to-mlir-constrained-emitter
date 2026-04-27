"""Tests for the in-line C3 sampling loop (Day 25 + Day 39 multi-terminal fix)."""
from __future__ import annotations

import pytest

from decoder.inline_c3 import CFGAutomaton, generate_text_mock
from decoder.inline_c3.cfg_automaton import TerminalSpec
from decoder.inline_c3.generate import _consume_one_terminal
from decoder.inline_c3.tests.test_bpe_masks import _StubTokenizer


GRAMMAR = "grammar/mlir_gen_c1c2.lark"


def _encode_text(tok: _StubTokenizer, text: str) -> list[int]:
    return tok.encode(text)


def test_generate_accepts_empty_oracle() -> None:
    tok = _StubTokenizer()
    automaton = CFGAutomaton(GRAMMAR)
    result = generate_text_mock("", automaton, tok, oracle=iter([]), max_steps=5)
    assert not result.accepted
    assert result.num_tokens == 0


def test_generate_walks_module_prefix() -> None:
    """Oracle emits chars of 'module {' — loop should close MODULE then LBRACE."""
    tok = _StubTokenizer()
    automaton = CFGAutomaton(GRAMMAR)
    text = "module {"
    ids = _encode_text(tok, text)
    result = generate_text_mock("", automaton, tok, oracle=iter(ids), max_steps=100)
    # All 8 chars consumed; MODULE, WS, LBRACE closed = 3 terminals
    assert result.num_tokens == 8
    assert result.closed_terminals == 3
    # Not yet accepted (partial parse)
    assert not result.accepted


def test_generate_rejects_oracle_violating_mask() -> None:
    """Oracle emits an illegal token at step 0 — should raise."""
    tok = _StubTokenizer()
    automaton = CFGAutomaton(GRAMMAR)
    # First legal token must be an 'm' (start of 'module')
    bad_id = tok._map["a"]
    with pytest.raises(RuntimeError, match="not in legal mask"):
        generate_text_mock("", automaton, tok, oracle=iter([bad_id]), max_steps=5)


def test_terminal_closing_advances_phase() -> None:
    """After closing MODULE, WS, LBRACE, WS, 'func.func' we should be in a
    state expecting WS before a SYM."""
    tok = _StubTokenizer()
    automaton = CFGAutomaton(GRAMMAR)
    text = "module { func.func "
    ids = _encode_text(tok, text)
    result = generate_text_mock("", automaton, tok, oracle=iter(ids), max_steps=200)
    assert result.num_tokens == len(text)
    # module, WS, {, WS, func.func, WS = 6 terminals
    assert result.closed_terminals >= 6


# --- Day 39: _consume_one_terminal with leftover-preservation ---


def _lit(name: str, s: str) -> TerminalSpec:
    return TerminalSpec(name=name, pattern=s, is_literal=True, literal=s)


def _rx(name: str, pat: str) -> TerminalSpec:
    return TerminalSpec(name=name, pattern=pat, is_literal=False, literal=None)


def test_consume_literal_exact() -> None:
    spec = _lit("MODULE", "module")
    assert _consume_one_terminal(spec, "module") == ("module", "")


def test_consume_literal_with_leftover() -> None:
    """Multi-terminal BPE token: 'module ' carries MODULE + leftover ' '."""
    spec = _lit("MODULE", "module")
    assert _consume_one_terminal(spec, "module ") == ("module", " ")


def test_consume_literal_partial_is_deferred() -> None:
    """Partial literal — wait for more tokens."""
    spec = _lit("MODULE", "module")
    assert _consume_one_terminal(spec, "mod") is None


def test_consume_literal_non_prefix_rejected() -> None:
    spec = _lit("MODULE", "module")
    assert _consume_one_terminal(spec, " module") is None


def test_consume_regex_defers_when_match_reaches_end() -> None:
    """partial = ' ' matches \\s+ but next token might extend — defer."""
    spec = _rx("WS", r"\s+")
    assert _consume_one_terminal(spec, " ") is None


def test_consume_regex_commits_when_non_extending_char_follows() -> None:
    """partial = ' {' — \\s+ matches ' ', and '{' can't extend WS. Commit."""
    spec = _rx("WS", r"\s+")
    assert _consume_one_terminal(spec, " {") == (" ", "{")


def test_consume_regex_greedy_within_partial() -> None:
    """partial = '   x' — WS matches all 3 spaces, 'x' is leftover."""
    spec = _rx("WS", r"\s+")
    assert _consume_one_terminal(spec, "   x") == ("   ", "x")


def test_consume_regex_non_match_returns_none() -> None:
    spec = _rx("WS", r"\s+")
    assert _consume_one_terminal(spec, "module") is None


def test_consume_on_empty_partial() -> None:
    assert _consume_one_terminal(_lit("X", "x"), "") is None
    assert _consume_one_terminal(_rx("Y", r"\w+"), "") is None


class _MultiCharTokenizer:
    """Stub where single tokens can span multiple grammar terminals.

    Models a realistic BPE-style tokenizer: tokens like 'module ' and ' {'
    that carry two terminals each. This is the exact shape that triggered
    the Day-26 'modulemodule' bug on SmolLM2's real tokenizer.
    """

    def __init__(self, pieces: list[str]) -> None:
        self._pieces = pieces
        self._map = {p: i for i, p in enumerate(pieces)}

    @property
    def vocab_size(self) -> int:
        return len(self._pieces)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        return "".join(self._pieces[i] for i in ids)


def test_day39_sync_handles_multi_terminal_token() -> None:
    """Day-39 regression: a single 'module ' token must advance past MODULE
    AND keep the trailing space as leftover, not drop it.

    Simulates the Day-26 bug's root cause: the old close-loop zeroed
    `partial` after closing MODULE, losing the ' ' — then the next mask
    computation was effectively unconstrained and the model re-emitted
    'module'.
    """
    from decoder.inline_c3.mlx_generate import _sync_with_token
    from decoder.inline_c3.coupled_state import CoupledState

    tok = _MultiCharTokenizer(["module ", " ", "{", "module"])
    automaton = CFGAutomaton(GRAMMAR)
    state_ref = {
        "state": CoupledState.fresh(automaton),
        "partial": "",
        "consumed": 0,
        "closed": 0,
    }

    _sync_with_token(state_ref, tok, 0)

    # MODULE closes from "module " prefix; " " stays as pending partial
    # because re.match(\s+, " ") reaches end-of-partial and defers (might
    # be extended by a future token). The key Day-39 property: the " " is
    # PRESERVED, not silently dropped as the Day-26 close-loop would have.
    assert state_ref["closed"] == 1
    assert state_ref["partial"] == " "

    _sync_with_token(state_ref, tok, 2)
    # Now partial = " {" — WS commits (leftover "{"), then LBRACE commits.
    assert state_ref["closed"] == 3
    assert state_ref["partial"] == ""
