"""LARK-based MLIR grammar (C1) and ODS-derived type lattice (C2)."""
from .parser import parse_mlir, grammar_path, load_parser

__all__ = ["parse_mlir", "grammar_path", "load_parser"]
