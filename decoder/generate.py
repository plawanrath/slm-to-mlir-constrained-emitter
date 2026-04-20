"""Unified generate() API.

Constraint levels (design.md §4.2):
  - NONE: free decoding
  - C1:   CFG-GCD via Outlines + LARK grammar
  - C1_C2: C1 intersected with the ODS-derived type + arity mask (C2)

Backends:
  - "mlx":    MLX-LM (primary SLM path; requires Apple Silicon + `mlx-lm`)
  - "ollama": Ollama REST (30B baselines; C1 applied via post-hoc resampling)
  - "mock":   deterministic stub for unit tests; no model at all
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable


class ConstraintLevel(str, Enum):
    NONE = "none"
    C1 = "c1"
    C1_C2 = "c1_c2"


@dataclass
class GenerateRequest:
    prompt: str
    model: str               # e.g. "microsoft/Phi-3.5-mini-instruct"
    constraint: ConstraintLevel
    dialect: str             # "arith" | "func" | "linalg" | "memref"
    backend: str = "mlx"     # mlx | ollama | mock
    max_tokens: int = 512
    temperature: float = 0.2
    seed: int = 0


@dataclass
class GenerateResult:
    text: str
    num_tokens: int
    backend_metadata: dict


def generate(req: GenerateRequest) -> GenerateResult:
    """Dispatch to the right backend. Import backends lazily so mock tests
    don't require MLX / Outlines / Ollama to be installed."""
    if req.backend == "mock":
        return _mock_generate(req)
    if req.backend == "mlx":
        from .c1_cfg import mlx_generate  # lazy
        return mlx_generate(req)
    if req.backend == "ollama":
        from eval.baselines.run_ollama import ollama_generate  # lazy
        return ollama_generate(req)
    raise ValueError(f"unknown backend: {req.backend}")


def _mock_generate(req: GenerateRequest) -> GenerateResult:
    """Deterministic mock for tests: returns a trivial MLIR that always verifies."""
    text = (
        "module {\n"
        f"  func.func @mock_{req.dialect}() -> i32 {{\n"
        "    %0 = arith.constant 0 : i32\n"
        "    return %0 : i32\n"
        "  }\n"
        "}\n"
    )
    return GenerateResult(text=text, num_tokens=len(text.split()), backend_metadata={"mock": True})
