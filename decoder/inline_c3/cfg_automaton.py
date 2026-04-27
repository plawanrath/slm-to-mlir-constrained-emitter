"""CFGAutomaton: wrap a LALR-compiled LARK grammar and expose a token-level
query/advance interface suitable for in-line constrained decoding.

Contract:
  - Built from a LARK grammar file (LALR-compatible). Our
    `grammar/mlir_gen_c1c2.lark` qualifies.
  - `CFGState.legal_next_terminals() -> set[str]` enumerates the terminal
    names (e.g. "MODULE", "SSA", "INT_TYPE") that the grammar would accept
    as the next lexeme given the sequence consumed so far.
  - `CFGState.advance(term_name, value)` consumes one terminal and returns
    a new CFGState. Raises `CFGAdvanceError` if not accepted.
  - `CFGState.is_accept()` returns True iff the empty remainder would
    close the parse.
  - `TerminalSpec` bundles a terminal's regex pattern + priority so a
    caller can build token-id masks by matching vocab text against it
    (Day 24 work).

Thread-safety: CFGState instances are immutable after construction in the
sense that `advance` returns a fresh state. The underlying Lark parser tables
are shared; safe for read-only concurrent use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from lark import Lark, Token
from lark.exceptions import UnexpectedToken
from lark.parsers.lalr_interactive_parser import InteractiveParser


class CFGAutomatonError(Exception):
    pass


class CFGAdvanceError(CFGAutomatonError):
    pass


@dataclass(frozen=True)
class TerminalSpec:
    name: str
    pattern: str          # regex source text, anchored-by-convention at start
    priority: int = 0
    is_literal: bool = False   # True if the pattern is a single quoted literal
    literal: str | None = None


@lru_cache(maxsize=8)
def _load_lalr(grammar_path: str, start: str = "start") -> Lark:
    return Lark.open(
        grammar_path,
        start=start,
        parser="lalr",
        lexer="basic",
        propagate_positions=False,
        maybe_placeholders=False,
    )


class CFGAutomaton:
    """Factory for CFGState instances over a LALR-compiled LARK grammar."""

    def __init__(self, grammar_path: str | Path, start: str = "start") -> None:
        self._grammar_path = str(Path(grammar_path).resolve())
        self._start = start
        self._lark = _load_lalr(self._grammar_path, start)
        self._terminals: dict[str, TerminalSpec] = self._index_terminals()

    def _index_terminals(self) -> dict[str, TerminalSpec]:
        out: dict[str, TerminalSpec] = {}
        for t in self._lark.terminals:
            pat_src = getattr(t.pattern, "value", "") or ""
            pat_type = type(t.pattern).__name__  # PatternStr or PatternRE
            is_literal = pat_type == "PatternStr"
            out[t.name] = TerminalSpec(
                name=t.name,
                pattern=pat_src,
                priority=int(getattr(t, "priority", 0) or 0),
                is_literal=is_literal,
                literal=pat_src if is_literal else None,
            )
        return out

    @property
    def terminals(self) -> dict[str, TerminalSpec]:
        return self._terminals

    def start_state(self) -> "CFGState":
        ip = self._lark.parse_interactive("")
        return CFGState(automaton=self, interactive=ip, depth=0)

    def __repr__(self) -> str:
        return f"CFGAutomaton(grammar={self._grammar_path}, terminals={len(self._terminals)})"


@dataclass
class CFGState:
    """Immutable-by-convention view over a LALR InteractiveParser position.

    `advance(term, value)` returns a NEW CFGState. Do not mutate self.
    """
    automaton: CFGAutomaton
    interactive: InteractiveParser
    depth: int = 0
    _accepts_cache: frozenset[str] | None = field(default=None, repr=False)

    def legal_next_terminals(self) -> frozenset[str]:
        if self._accepts_cache is not None:
            return self._accepts_cache
        acc = frozenset(self.interactive.accepts())
        object.__setattr__(self, "_accepts_cache", acc)
        return acc

    def legal_terminal_specs(self) -> list[TerminalSpec]:
        """Terminal specs for each legal next-terminal. Useful for building
        token-id masks (Day 24)."""
        specs = self.automaton.terminals
        return [specs[n] for n in self.legal_next_terminals() if n in specs]

    def advance(self, term_name: str, value: str) -> "CFGState":
        legal = self.legal_next_terminals()
        if term_name not in legal:
            raise CFGAdvanceError(
                f"terminal {term_name!r} not in legal set at depth={self.depth}; "
                f"legal={sorted(legal)}"
            )
        new_ip = self.interactive.copy()
        try:
            new_ip.feed_token(Token(term_name, value))
        except UnexpectedToken as e:
            raise CFGAdvanceError(f"advance rejected: {e}") from e
        return CFGState(automaton=self.automaton, interactive=new_ip, depth=self.depth + 1)

    def is_accept(self) -> bool:
        """True iff $END is currently acceptable (i.e. parse can close now)."""
        try:
            probe = self.interactive.copy()
            probe.feed_eof()
            return True
        except UnexpectedToken:
            return False
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"CFGState(depth={self.depth}, legal={sorted(self.legal_next_terminals())[:6]}...)"


def ingest_tokens(state: CFGState, pairs: Iterable[tuple[str, str]]) -> CFGState:
    """Advance through a sequence of (term_name, value) pairs. Convenience."""
    s = state
    for term, val in pairs:
        s = s.advance(term, val)
    return s
