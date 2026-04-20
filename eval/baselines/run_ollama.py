"""Ollama REST wrapper for 30B baselines.

We do NOT run Ollama in Docker — Docker on macOS has no Metal access and would
force CPU-only inference (unusable for 30B Q4). Ollama is therefore the single
documented host-level dependency (see RUNBOOK.md §0).

Reproducibility for non-macOS reviewers: ship a llama.cpp CPU fallback in the
final reproducibility Dockerfile (Day 16). The paper reports the Metal numbers
as primary; the CPU numbers are a ~20-30× slower sanity check.
"""
from __future__ import annotations

import json
import os
from typing import Any

import requests

from decoder.generate import GenerateRequest, GenerateResult

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_TIMEOUT = 600.0  # 30B Q4 can be slow on first load


def ollama_generate_raw(
    prompt: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    seed: int = 0,
) -> tuple[str, dict]:
    """Single /api/generate call. Returns (text, metadata)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_predict": max_tokens,
        },
    }
    r = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json=payload,
        timeout=DEFAULT_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("response", ""), {
        "eval_count": data.get("eval_count", 0),
        "eval_duration_ns": data.get("eval_duration", 0),
        "total_duration_ns": data.get("total_duration", 0),
        "num_tokens": data.get("eval_count", 0),
    }


def ollama_generate(req: GenerateRequest) -> GenerateResult:
    """Dispatcher-compatible wrapper. Handles {free, C1} via rejection sampling."""
    from decoder.generate import ConstraintLevel
    if req.constraint == ConstraintLevel.NONE:
        text, meta = ollama_generate_raw(
            prompt=req.prompt, model=req.model,
            max_tokens=req.max_tokens, temperature=req.temperature, seed=req.seed,
        )
        return GenerateResult(
            text=text, num_tokens=meta["num_tokens"],
            backend_metadata={"backend": "ollama", **meta},
        )
    if req.constraint == ConstraintLevel.C1:
        from decoder.c1_cfg import rejection_sample_ollama
        return rejection_sample_ollama(req)
    if req.constraint == ConstraintLevel.C1_C2:
        # design.md §5.2: baselines are {free, C1} only. We never call this.
        raise ValueError("C1_C2 not supported on the Ollama backend (design §5.2).")
    raise ValueError(f"unknown constraint: {req.constraint}")


def list_models() -> list[str]:
    r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]
