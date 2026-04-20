"""LARK parser loader + thin parse/validate helpers."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from lark import Lark, Tree
from lark.exceptions import LarkError

_GRAMMAR_PATH = Path(__file__).resolve().parent / "mlir.lark"


def grammar_path() -> Path:
    return _GRAMMAR_PATH


@lru_cache(maxsize=1)
def load_parser(start: str = "start") -> Lark:
    """Return a cached LALR parser. We use Earley for robustness during grammar
    development (ambiguity-tolerant); switch to LALR once v1 stabilizes."""
    return Lark.open(
        str(_GRAMMAR_PATH),
        start=start,
        parser="earley",
        lexer="dynamic_complete",
        propagate_positions=False,
        maybe_placeholders=False,
    )


def parse_mlir(text: str) -> Tree:
    """Parse an MLIR string. Raises LarkError on invalid input."""
    return load_parser().parse(text)


def is_parse_valid(text: str) -> bool:
    """True iff the grammar accepts `text`. No semantic check."""
    try:
        parse_mlir(text)
        return True
    except LarkError:
        return False
