# SLM-to-MLIR Constrained Emitter

Milestone 1 of a multi-milestone research program on spec-driven programming languages. This repository pursues a NeurIPS 2026 main-track submission on training-free, ODS-aware constrained decoding for NL→MLIR generation with small language models.

**Thesis**: For formal intermediate representations, ODS-derived structural priors applied at inference time close most of the capability gap between 1-3B SLMs and 30B+ models — without fine-tuning, without RL, without distillation.

## Status

Week 0. Design locked 2026-04-17. See `docs/design.md`.

## Key documents

- [`docs/design.md`](docs/design.md) — locked design doc: thesis, contributions, methodology, eval matrix, gates.
- [`docs/execution_plan.md`](docs/execution_plan.md) — active 17-day execution schedule (supersedes `design.md` §6).
- [`docs/decisions/`](docs/decisions/) — architecture decision records (see ADR-0004 for the current timeline lock).

## Hardware target

Single Apple M4 Max 128GB unified memory. All main experiments reproducible on-device.

## Stack

MLX + MLX-LM (SLMs) · llama.cpp via Ollama (30B baselines) · Outlines + LARK (constrained decoding) · Polygeist (C→MLIR lowering) · pinned LLVM mlir-opt (verification).

## Target dialects

`arith`, `func`, `linalg`.
