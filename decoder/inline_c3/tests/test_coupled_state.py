"""Tests for CoupledState (Day 23)."""
from __future__ import annotations

import pytest

from decoder.inline_c3 import (
    CFGAutomaton, CoupledState, SSARole, OpPhase, ScopeViolation,
)


GRAMMAR = "grammar/mlir_gen_c1c2.lark"


def _fresh() -> CoupledState:
    return CoupledState.fresh(CFGAutomaton(GRAMMAR))


def test_params_define_ssa_names() -> None:
    cs = _fresh()
    walk = [
        ("MODULE", "module"), ("WS", " "),
        ("LBRACE", "{"), ("WS", " "),
        ("__ANON_0", "func.func"), ("WS", " "),
        ("SYM", "@f"),
        ("LPAR", "("),
        ("SSA", "%a"), ("WS", " "), ("COLON", ":"), ("WS", " "),
        ("INT_TYPE", "i32"),
        ("WS", " "), ("COMMA", ","), ("WS", " "),
        ("SSA", "%b"), ("WS", " "), ("COLON", ":"), ("WS", " "),
        ("INT_TYPE", "i32"),
        ("RPAR", ")"),
    ]
    for term, val in walk:
        cs = cs.advance(term, val)
    assert cs.symbols.has("%a")
    assert cs.symbols.has("%b")
    assert cs.phase == OpPhase.FUNC_BODY


def test_use_of_undefined_raises() -> None:
    cs = _fresh()
    walk = [
        ("MODULE", "module"), ("WS", " "),
        ("LBRACE", "{"), ("WS", " "),
        ("__ANON_0", "func.func"), ("WS", " "),
        ("SYM", "@f"),
        ("LPAR", "("), ("RPAR", ")"),
        ("WS", " "), ("__ANON_1", "->"), ("WS", " "),
        ("INT_TYPE", "i32"),
        ("WS", " "), ("LBRACE", "{"), ("WS", " "),
        ("RETURN", "return"), ("WS", " "),
    ]
    for term, val in walk:
        cs = cs.advance(term, val)
    assert cs.phase == OpPhase.RETURN_VAL
    # Attempt to use an undefined SSA name
    with pytest.raises(ScopeViolation):
        cs.advance("SSA", "%undefined")


def test_lhs_defined_on_eq() -> None:
    """A `%x = arith.constant 42 : i32` should define %x when '=' is seen."""
    cs = _fresh()
    walk = [
        ("MODULE", "module"), ("WS", " "),
        ("LBRACE", "{"), ("WS", " "),
        ("__ANON_0", "func.func"), ("WS", " "),
        ("SYM", "@f"),
        ("LPAR", "("), ("RPAR", ")"),
        ("WS", " "), ("__ANON_1", "->"), ("WS", " "),
        ("INT_TYPE", "i32"),
        ("WS", " "), ("LBRACE", "{"), ("WS", " "),
        ("SSA", "%c"), ("WS", " "), ("EQUAL", "="),
    ]
    for term, val in walk:
        cs = cs.advance(term, val)
    assert cs.symbols.has("%c")
    assert cs.phase == OpPhase.AFTER_EQ


def test_symbol_table_immutability() -> None:
    cs = _fresh()
    _ = cs.symbols  # snapshot
    cs2 = cs.advance("MODULE", "module")
    assert cs.phase == OpPhase.OUTER
    assert cs.symbols.entries == ()
    assert cs2.symbols.entries == ()   # no defs yet


def test_duplicate_define_raises() -> None:
    cs = _fresh()
    walk = [
        ("MODULE", "module"), ("WS", " "),
        ("LBRACE", "{"), ("WS", " "),
        ("__ANON_0", "func.func"), ("WS", " "),
        ("SYM", "@f"),
        ("LPAR", "("),
        ("SSA", "%a"), ("WS", " "), ("COLON", ":"), ("WS", " "),
        ("INT_TYPE", "i32"),
        ("WS", " "), ("COMMA", ","), ("WS", " "),
    ]
    for term, val in walk:
        cs = cs.advance(term, val)
    with pytest.raises(ScopeViolation):
        cs.advance("SSA", "%a")   # same name


def test_consecutive_ops_detect_new_op_start(tmp_path) -> None:
    """After a full op (%c = arith.constant 42 : i32) then WS, the next SSA
    should be classified DEF again because it's the LHS of the next op.

    NOTE: The current coupled_state heuristic relies on the LALR parser's
    automatic transition back to OP_START context, but does not yet auto-reset
    phase. This test documents the current behavior and will be tightened in
    Day 25 with the LALR-friendly gen grammar.
    """
    cs = _fresh()
    walk = [
        ("MODULE", "module"), ("WS", " "),
        ("LBRACE", "{"), ("WS", " "),
        ("__ANON_0", "func.func"), ("WS", " "),
        ("SYM", "@f"),
        ("LPAR", "("), ("RPAR", ")"),
        ("WS", " "), ("__ANON_1", "->"), ("WS", " "),
        ("INT_TYPE", "i32"),
        ("WS", " "), ("LBRACE", "{"), ("WS", " "),
        ("SSA", "%c"), ("WS", " "), ("EQUAL", "="), ("WS", " "),
        ("__ANON_2", "arith.constant"), ("WS", " "),
        ("INT_LIT", "42"), ("WS", " "), ("COLON", ":"), ("WS", " "),
        ("INT_TYPE", "i32"),
    ]
    for term, val in walk:
        cs = cs.advance(term, val)
    # %c is in scope
    assert cs.symbols.has("%c")
    # phase after op completion is AFTER_EQ (stale) — Day 25 will tighten this
    assert cs.phase == OpPhase.AFTER_EQ


def test_candidate_uses_returns_in_scope_names() -> None:
    cs = _fresh()
    walk = [
        ("MODULE", "module"), ("WS", " "),
        ("LBRACE", "{"), ("WS", " "),
        ("__ANON_0", "func.func"), ("WS", " "),
        ("SYM", "@f"),
        ("LPAR", "("),
        ("SSA", "%a"), ("WS", " "), ("COLON", ":"), ("WS", " "),
        ("INT_TYPE", "i32"),
        ("WS", " "), ("COMMA", ","), ("WS", " "),
        ("SSA", "%b"), ("WS", " "), ("COLON", ":"), ("WS", " "),
        ("INT_TYPE", "i32"),
        ("RPAR", ")"),
        ("WS", " "), ("__ANON_1", "->"), ("WS", " "),
        ("INT_TYPE", "i32"),
        ("WS", " "), ("LBRACE", "{"), ("WS", " "),
        ("RETURN", "return"), ("WS", " "),
    ]
    for term, val in walk:
        cs = cs.advance(term, val)
    assert cs.classify_ssa() == SSARole.USE
    assert set(cs.candidate_uses()) == {"%a", "%b"}
