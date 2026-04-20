"""Unit tests for the generate() dispatcher — mock backend only (no MLX)."""
from __future__ import annotations

from decoder.generate import ConstraintLevel, GenerateRequest, generate


def test_mock_backend_returns_verifying_mlir() -> None:
    req = GenerateRequest(
        prompt="irrelevant",
        model="mock",
        constraint=ConstraintLevel.NONE,
        dialect="arith",
        backend="mock",
    )
    r = generate(req)
    assert "module" in r.text
    assert "arith.constant" in r.text
    assert r.backend_metadata["mock"] is True


def test_constraint_level_enum() -> None:
    assert ConstraintLevel.NONE.value == "none"
    assert ConstraintLevel.C1.value == "c1"
    assert ConstraintLevel.C1_C2.value == "c1_c2"
