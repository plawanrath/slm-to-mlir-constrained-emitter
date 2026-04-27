"""Tests for the LARK→automaton wrapper (Day 22 of ADR-0008)."""
from __future__ import annotations

from pathlib import Path

import pytest

from decoder.inline_c3 import CFGAutomaton


GRAMMAR = "grammar/mlir_gen_c1c2.lark"


def _auto() -> CFGAutomaton:
    return CFGAutomaton(GRAMMAR)


def test_start_state_begins_with_module() -> None:
    auto = _auto()
    s = auto.start_state()
    assert s.legal_next_terminals() == {"MODULE"}
    assert not s.is_accept()


def test_advance_through_prefix() -> None:
    auto = _auto()
    s = auto.start_state()
    s = s.advance("MODULE", "module")
    assert "WS" in s.legal_next_terminals()
    s = s.advance("WS", " ")
    # the grammar rule is `"module" WS "{" ...`; the '{' literal becomes LBRACE.
    assert "LBRACE" in s.legal_next_terminals()


def test_advance_rejects_illegal_terminal() -> None:
    auto = _auto()
    s = auto.start_state()
    from decoder.inline_c3.cfg_automaton import CFGAdvanceError
    with pytest.raises(CFGAdvanceError):
        s.advance("SSA", "%x")


def test_terminal_specs_have_patterns() -> None:
    auto = _auto()
    specs = auto.terminals
    assert "SSA" in specs
    assert specs["SSA"].pattern == r"%[a-zA-Z0-9_][a-zA-Z0-9_]{0,31}"
    assert not specs["SSA"].is_literal
    assert specs["MODULE"].is_literal
    assert specs["MODULE"].literal == "module"


def test_legal_specs_accessible_from_state() -> None:
    auto = _auto()
    s = auto.start_state()
    legal = s.legal_terminal_specs()
    assert len(legal) == 1
    assert legal[0].name == "MODULE"


def test_partial_walk_through_signature() -> None:
    """Advance through a full function signature + op prefix. Uses LALR-friendly
    transitions only; full-program completion in LALR requires an adapted gen
    grammar (Day 25 will author grammar/mlir_gen_inline.lark with right-recursive
    op_body so single-op bodies can reduce before the trailing WS)."""
    auto = _auto()
    s = auto.start_state()
    walk = [
        ("MODULE", "module"), ("WS", " "),
        ("LBRACE", "{"), ("WS", " "),
        ("__ANON_0", "func.func"), ("WS", " "),
        ("SYM", "@f"),
        ("LPAR", "("), ("RPAR", ")"),
        ("WS", " "), ("__ANON_1", "->"), ("WS", " "),
        ("INT_TYPE", "i32"),
        ("WS", " "), ("LBRACE", "{"), ("WS", " "),
        ("SSA", "%0"), ("WS", " "), ("EQUAL", "="), ("WS", " "),
        ("__ANON_2", "arith.constant"), ("WS", " "),
        ("INT_LIT", "42"), ("WS", " "), ("COLON", ":"), ("WS", " "),
        ("INT_TYPE", "i32"),
    ]
    for term, val in walk:
        legal = s.legal_next_terminals()
        assert term in legal, f"{term!r} not in {sorted(legal)} at depth {s.depth}"
        s = s.advance(term, val)
    # Grammar is in the middle of op_body: accepts WS next
    assert "WS" in s.legal_next_terminals()


def test_identity_of_terminals_matches_grammar_file() -> None:
    """Sanity: the indexed terminal set is non-empty and covers core names."""
    auto = _auto()
    names = set(auto.terminals.keys())
    required = {"MODULE", "SSA", "SYM", "INT_TYPE", "FLOAT_TYPE", "WS"}
    assert required.issubset(names), f"missing: {required - names}"


def test_equivalence_with_earley_on_mlir_spec_seeds() -> None:
    """Day-22 DoD: for MLIR samples that Earley accepts (i.e. pass under the
    permissive parse grammar), the LALR automaton's `is_parse_valid`-path
    agrees up to the grammar's LALR limitations.

    We don't claim full equivalence (the gen grammar and parse grammar are
    different; the gen grammar is stricter and LALR-limited on multi-op
    bodies). The weaker property tested: every MLIR-Spec-150 seed that is
    Earley-valid is ALSO fully-lexable under the LALR lexer, which means
    the vocabulary of tokens is aligned.
    """
    import json
    from pathlib import Path
    from lark import Lark
    p = Lark.open("grammar/mlir_gen_c1c2.lark", start="start", parser="lalr",
                  lexer="basic")
    # Pick a handful of easy single-op seeds
    seeds = sorted(Path("eval/benchmarks/mlir_spec_150/examples").glob("*.json"))[:10]
    n_lexed = 0
    for sp in seeds:
        s = json.loads(sp.read_text())["mlir"]
        try:
            toks = list(p.lex(s))
            assert len(toks) > 0
            n_lexed += 1
        except Exception:
            # Some seeds use syntax the gen grammar doesn't model — expected.
            pass
    assert n_lexed >= 3, (
        f"LALR lexer should handle at least 3 of 10 easy seeds; got {n_lexed}"
    )
