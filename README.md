# SLM-to-MLIR Constrained Emitter

A NeurIPS 2026 Evaluations & Datasets (E&D) track submission on training-free, ODS-aware constrained decoding for NL→MLIR generation with small language models.

**Thesis**: For formal intermediate representations, ODS-derived structural priors (C1 syntactic + C2 type-domain + C3 SSA-scope) applied at inference time close most of the capability gap between a **1.7B SLM** and 30B+ models — without fine-tuning, without RL, without distillation.

## Status

Week 0. Design locked 2026-04-17. See `docs/design.md`.

## Key documents

- [`docs/design.md`](docs/design.md) — locked design doc: thesis, contributions, methodology, eval matrix, gates.
- [`docs/execution_plan.md`](docs/execution_plan.md) — active 17-day execution schedule (supersedes `design.md` §6).
- [`RUNBOOK.md`](RUNBOOK.md) — **step-by-step commands** to execute the plan: env setup (Docker + venv), day-by-day runs, gates, reproducibility package.
- [`TODO.md`](TODO.md) — **persistent work log**: resumable across sessions; mirrors the RUNBOOK with checkbox state, gate results, and artifact paths.
- [`docs/decisions/`](docs/decisions/) — architecture decision records (see ADR-0004 for the current timeline lock).
- [`docs/daily_log/`](docs/daily_log/) — running research log: per-day measurements, surprises, and narrative material for the paper.

## Hardware target

Single Apple M4 Max 128GB unified memory. All main experiments reproducible on-device.

## Stack

MLX + MLX-LM (SLMs: **SmolLM2-1.7B** primary, Phi-3.5-mini-instruct secondary) · llama.cpp via Ollama (30B baselines: CodeLlama-34B, Granite-Code-34B) · Outlines + llguidance + LARK (constrained decoding, C1+C2+C3) · pinned LLVM mlir-opt (verification).

## Target dialects

`arith`, `func`, `linalg`.
