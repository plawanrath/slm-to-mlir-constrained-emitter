"""Constrained decoding (C1 = CFG-GCD, C2 = +type/arity). Entry point: decoder.generate."""
from .generate import generate, ConstraintLevel

__all__ = ["generate", "ConstraintLevel"]
