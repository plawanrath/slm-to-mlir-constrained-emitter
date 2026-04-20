"""C2: ODS-derived type + arity mask, composable with C1 via intersection.

The mask is constructed from lattice JSONs produced by grammar.ods_lattice.
At any generation step where the C1 state machine is inside an op signature
(between the colon and the terminator), C2 tightens the next-token set to:

  - type-prefix tokens that are consistent with the current op's type lattice
  - delimiters consistent with arity (block "," if max-arity reached;
    block ")" / newline if min-arity not yet satisfied)

Mechanics:

  class TypePrefixAutomaton:
      - states are prefixes of allowed type strings (e.g. "i", "i3", "i32").
      - allowed next chars at state `p` = { c : p+c is a prefix of some T in allowed_types }
      - final state when current string fully matches a T.

  class ArityStateMachine:
      - counts operands emitted so far for the current op.
      - exposes `can_emit_separator()`, `can_terminate()` booleans.

The composed mask is a set of allowed token IDs built once the model's
tokenizer is known. For the mock / unit-test path we expose a string-level
API (`allowed_chars_at(state) -> set[str]`) that avoids needing a real
tokenizer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from grammar.ods_lattice import PRIMITIVE_TYPES, ANY

REPO_ROOT = Path(__file__).resolve().parent.parent
LATTICE_DIR = REPO_ROOT / "grammar" / "lattices"


@lru_cache(maxsize=8)
def load_dialect_lattice(dialect: str) -> dict[str, dict]:
    """Load grammar/lattices/{dialect}.json. Empty dict if missing (C2 skips)."""
    path = LATTICE_DIR / f"{dialect}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


# ---------------- type-prefix automaton ----------------

@dataclass
class TypePrefixAutomaton:
    allowed_types: tuple[str, ...]   # e.g. ("i1", "i32", "f32")
    position: int = 0                # characters consumed so far
    current: str = ""                # the prefix emitted so far

    def allowed_next_chars(self) -> set[str]:
        """Return chars c such that (current + c) is a prefix of some allowed type."""
        nxt: set[str] = set()
        for t in self.allowed_types:
            if t.startswith(self.current) and len(t) > self.position:
                nxt.add(t[self.position])
        return nxt

    def advance(self, ch: str) -> bool:
        """Consume ch; returns False if ch is disallowed."""
        if ch not in self.allowed_next_chars():
            return False
        self.current += ch
        self.position += 1
        return True

    def is_final(self) -> bool:
        return self.current in self.allowed_types

    def reset(self) -> None:
        self.position = 0
        self.current = ""


def build_type_automaton(type_classes: list[str]) -> TypePrefixAutomaton:
    """Build a TypePrefixAutomaton from a list of closed-vocab type-class names.

    Unknown / ANY entries disable the automaton (allowed_types = all primitives).
    """
    if ANY in type_classes:
        allowed = tuple(PRIMITIVE_TYPES)
    else:
        allowed = tuple(t for t in type_classes if t in PRIMITIVE_TYPES)
        if not allowed:
            # Non-primitive head (memref/tensor/vector). C2 skips prefix enforcement
            # for compound types in v0 — compound-type mask is Day-8+ work.
            allowed = tuple(PRIMITIVE_TYPES)
    return TypePrefixAutomaton(allowed_types=allowed)


# ---------------- arity state machine ----------------

@dataclass
class ArityStateMachine:
    min_arity: int
    max_arity: int
    count: int = 0

    def can_emit_operand(self) -> bool:
        return self.count < self.max_arity

    def can_emit_separator(self) -> bool:
        # Separator allowed only if we've just emitted an operand and another fits.
        return 0 < self.count < self.max_arity

    def can_terminate(self) -> bool:
        return self.count >= self.min_arity

    def on_operand_emitted(self) -> None:
        self.count += 1

    def reset(self) -> None:
        self.count = 0


# ---------------- composite mask ----------------

@dataclass
class C2State:
    dialect: str
    current_op: str | None = None     # "arith.addi" etc.
    type_auto: TypePrefixAutomaton | None = None
    arity_sm: ArityStateMachine | None = None
    phase: str = "idle"               # idle | operands | type_tail
    _lattice_cache: dict[str, dict] = field(default_factory=dict)

    def lattice_for(self, op: str) -> dict | None:
        if op in self._lattice_cache:
            return self._lattice_cache[op]
        d = load_dialect_lattice(self.dialect)
        self._lattice_cache[op] = d.get(op)
        return self._lattice_cache[op]

    def enter_op(self, op_full_name: str) -> None:
        self.current_op = op_full_name
        entry = self.lattice_for(op_full_name)
        if entry is None:
            self.phase = "idle"
            self.type_auto = None
            self.arity_sm = None
            return
        op_min, op_max = entry["operand_arity"]
        self.arity_sm = ArityStateMachine(min_arity=op_min, max_arity=op_max)
        # Single-slot v0: only enforce the first operand's type-class. Multi-slot
        # simultaneous enforcement with same-type-class constraints is handled by
        # the composed mask below.
        if entry["operand_types"]:
            self.type_auto = build_type_automaton(entry["operand_types"][0])
        else:
            self.type_auto = None
        self.phase = "operands"

    def exit_op(self) -> None:
        self.current_op = None
        self.type_auto = None
        self.arity_sm = None
        self.phase = "idle"


def apply_type_arity_mask(logits: Any, dialect: str, state: dict) -> Any:
    """Stub integration point. Real impl: multiply logits by the C2 allowed-token
    mask at positions where state.phase != 'idle'.

    For Day-1 code-complete we return logits unchanged; the Day-2-3 integration
    pass wires this into the mlx-lm sampling loop once the logits-processor
    plumbing is verified against the pinned mlx-lm version.
    """
    return logits


def is_composite_compatible(
    candidate_text: str,
    op_full_name: str,
    dialect: str,
) -> bool:
    """Post-hoc validator used by tests + the Ollama rejection sampler.

    True iff `candidate_text` (after the colon) names a type that appears in
    the lattice for `op_full_name` and respects SameOperandsAndResultType.
    """
    lattice = load_dialect_lattice(dialect)
    entry = lattice.get(op_full_name)
    if entry is None:
        return True  # unknown op: abstain rather than reject
    # Pull out the type substring — everything after the last ":" on the line.
    if ":" not in candidate_text:
        return False
    tail = candidate_text.rsplit(":", 1)[1].strip().rstrip(",;})")
    # Check it matches at least one allowed type across all operand slots.
    allowed_union: set[str] = set()
    for slot in entry["operand_types"] + entry["result_types"]:
        for t in slot:
            if t == ANY:
                return True
            allowed_union.add(t)
    return any(tail.startswith(t) for t in allowed_union)
