"""Unit tests for C2 automata. No model, no tokenizer — pure state-machine logic."""
from __future__ import annotations

from decoder.c2_type_arity import (
    ArityStateMachine,
    TypePrefixAutomaton,
    build_type_automaton,
)


def test_type_prefix_integer_vocab() -> None:
    auto = TypePrefixAutomaton(allowed_types=("i1", "i32", "i64"))
    assert auto.allowed_next_chars() == {"i"}
    assert auto.advance("i")
    # After "i", allowed follow-ups are { "1", "3", "6" }.
    assert auto.allowed_next_chars() == {"1", "3", "6"}
    assert not auto.advance("X")
    assert auto.advance("3")
    assert auto.allowed_next_chars() == {"2"}
    assert auto.advance("2")
    assert auto.is_final()


def test_type_prefix_rejects_invalid() -> None:
    auto = TypePrefixAutomaton(allowed_types=("f32",))
    assert not auto.advance("i")  # "i" is not a prefix of f32


def test_arity_machine_basic() -> None:
    sm = ArityStateMachine(min_arity=2, max_arity=2)
    assert sm.can_emit_operand()
    assert not sm.can_emit_separator()  # no operand yet
    assert not sm.can_terminate()
    sm.on_operand_emitted()
    assert sm.can_emit_separator()
    assert not sm.can_terminate()
    sm.on_operand_emitted()
    assert not sm.can_emit_operand()
    assert not sm.can_emit_separator()
    assert sm.can_terminate()


def test_arity_variadic() -> None:
    sm = ArityStateMachine(min_arity=0, max_arity=10**6)
    assert sm.can_emit_operand()
    assert sm.can_terminate()  # 0 operands is fine


def test_build_automaton_from_any_fallback() -> None:
    from grammar.ods_lattice import ANY, PRIMITIVE_TYPES
    auto = build_type_automaton([ANY])
    # ANY means all primitives are allowed.
    assert set(auto.allowed_types) == set(PRIMITIVE_TYPES)
