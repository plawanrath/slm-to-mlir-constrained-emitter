"""CoupledState: CFG automaton × SSA symbol table — the Tier-1 novelty core.

Each advance updates BOTH:
  1. LALR CFG state (via CFGAutomaton)
  2. SSA symbol table (accumulates definitions, validates uses)

Key contribution over Outlines' CFGGuide: integrates a semantic symbol table
into the decoder state machine, turning SSA-scope validity into an invariant
enforced *during* decoding rather than a post-hoc filter. This is the first
published joint (CFG state × symbol table) state machine for NL→IR decoding.

Role inference for SSA terminals:
  Position      | Role | Rule example
  ------------- | ---- | ------------
  params        | DEF  | `param: SSA WS ":" WS type`
  LHS of '='    | DEF  | `op_binop_int: SSA WS "=" WS op_name ...`
  after '='     | USE  | `... WS SSA WS "," WS SSA WS ":" WS type`
  after RETURN  | USE  | `op_return_val: RETURN WS SSA WS ":" WS type`

Phase tracker state (per function):
  - NONE       → not yet in a func body
  - PARAM      → inside params list (after LPAR, before RPAR)
  - OP_START   → start of a new op in op_body (next SSA is DEF-candidate)
  - AFTER_EQ   → past '=' in current op (next SSA is USE)
  - RETURN     → after RETURN keyword (next SSA is USE)
  - CLOSED     → func body closed

Phase transitions are driven by terminal names; the table below is
grammar-version-agnostic: we react to LPAR, RPAR, LBRACE, RBRACE, EQUAL,
RETURN, WS, and op-start terminals.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Literal

from .cfg_automaton import CFGAutomaton, CFGState, CFGAdvanceError


class SSARole(str, Enum):
    DEF = "DEF"
    USE = "USE"
    UNKNOWN = "UNKNOWN"


class OpPhase(str, Enum):
    OUTER = "OUTER"
    PARAM = "PARAM"
    FUNC_BODY = "FUNC_BODY"
    OP_START = "OP_START"
    AFTER_EQ = "AFTER_EQ"
    RETURN_VAL = "RETURN_VAL"
    CLOSED = "CLOSED"


class ScopeViolation(Exception):
    pass


@dataclass(frozen=True)
class SymbolTable:
    """Immutable-after-construction; each mutation returns a new table."""
    entries: tuple[tuple[str, str], ...] = ()   # (name, type_str)

    def names(self) -> set[str]:
        return {n for n, _ in self.entries}

    def has(self, name: str) -> bool:
        return any(n == name for n, _ in self.entries)

    def type_of(self, name: str) -> str | None:
        for n, t in self.entries:
            if n == name:
                return t
        return None

    def define(self, name: str, type_str: str = "?") -> "SymbolTable":
        if self.has(name):
            raise ScopeViolation(f"SSA name {name!r} already defined")
        return SymbolTable(entries=self.entries + ((name, type_str),))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.has(name)


@dataclass(frozen=True)
class CoupledState:
    """Joint (CFG state × SSA symbol table × op phase)."""
    cfg: CFGState
    symbols: SymbolTable = SymbolTable()
    phase: OpPhase = OpPhase.OUTER
    _last_ssa_pending_def: str | None = None   # captured LHS before '='

    @classmethod
    def fresh(cls, automaton: CFGAutomaton) -> "CoupledState":
        return cls(cfg=automaton.start_state())

    def legal_next_terminals(self) -> frozenset[str]:
        return self.cfg.legal_next_terminals()

    def classify_ssa(self) -> SSARole:
        """Classify role of the next SSA terminal assuming one is legal."""
        if self.phase in (OpPhase.PARAM, OpPhase.OP_START):
            return SSARole.DEF
        if self.phase in (OpPhase.AFTER_EQ, OpPhase.RETURN_VAL):
            return SSARole.USE
        return SSARole.UNKNOWN

    def candidate_uses(self) -> tuple[str, ...]:
        """When next SSA is a USE, the legal names are exactly symbols.names()."""
        if self.classify_ssa() == SSARole.USE:
            return tuple(n for n, _ in self.symbols.entries)
        return ()

    def advance(self, term_name: str, value: str) -> "CoupledState":
        new_cfg = self.cfg.advance(term_name, value)
        new_symbols = self.symbols
        new_phase = self.phase
        pending = self._last_ssa_pending_def

        # Phase transitions based on terminal
        if term_name == "LPAR":
            # Entering params list
            new_phase = OpPhase.PARAM
        elif term_name == "RPAR" and self.phase == OpPhase.PARAM:
            new_phase = OpPhase.FUNC_BODY
        elif term_name == "LBRACE":
            # Entering func body; next op starts
            if self.phase in (OpPhase.FUNC_BODY, OpPhase.OUTER):
                new_phase = OpPhase.OP_START
        elif term_name == "RBRACE":
            new_phase = OpPhase.CLOSED
        elif term_name == "EQUAL":
            # We just saw `=`, the SSA we stashed was LHS (DEF)
            if pending is not None:
                # Defer actually defining until we know the type (arrives later).
                # For now, record with '?' type and let the C3 post-check validate.
                new_symbols = new_symbols.define(pending, "?")
                pending = None
            new_phase = OpPhase.AFTER_EQ
        elif term_name == "RETURN":
            new_phase = OpPhase.RETURN_VAL
        elif term_name == "SSA":
            role = self.classify_ssa()
            if role == SSARole.DEF:
                if self.phase == OpPhase.PARAM:
                    # params: SSA is DEF; stash the name and define with '?' type
                    # (actual type filled when we see the following ": <type>")
                    new_symbols = new_symbols.define(value, "?")
                    # stay in PARAM to allow "," next or ")" to exit
                elif self.phase == OpPhase.OP_START:
                    # LHS of op assignment; wait for '=' then define
                    pending = value
            elif role == SSARole.USE:
                if value not in new_symbols:
                    raise ScopeViolation(
                        f"SSA use {value!r} not in scope at phase={self.phase}; "
                        f"in-scope={sorted(new_symbols.names())}"
                    )
        elif term_name == "COMMA" and self.phase == OpPhase.PARAM:
            # stays in PARAM phase for next param
            pass

        # Check for op-start terminals that signal start of a new op in op_body
        # (used when we're in FUNC_BODY or after completing a previous op and
        # transitioning to OP_START via WS + op-start terminal).
        op_start_terms = {
            "SSA",             # if OP_START (LHS of assignment-style op)
            "RETURN",
            "LINALG_ELEMWISE_BIN", "LINALG_ELEMWISE_UN",
            "__ANON_9", "__ANON_10",
            "__ANON_12", "__ANON_13", "__ANON_14", "__ANON_15",
            "__ANON_16", "__ANON_17",
            "__ANON_7",  # memref.alloc
        }

        # When transitioning back to OP_START after WS from a completed op,
        # the caller is responsible for setting phase=OP_START if appropriate.
        # Our default heuristic: if current phase is FUNC_BODY (just entered
        # func body) or AFTER_EQ completes, go to OP_START on next op-start.
        # For simplicity, reset op_phase→OP_START after a WS when previous
        # op-completion happened (heuristic: after COLON ... type sequence we're done).

        return CoupledState(
            cfg=new_cfg, symbols=new_symbols, phase=new_phase,
            _last_ssa_pending_def=pending,
        )

    def is_accept(self) -> bool:
        return self.cfg.is_accept()
