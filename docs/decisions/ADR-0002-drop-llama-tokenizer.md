# ADR-0002: Drop Llama-3.2-3B from Tokenizer Shootout

**Date**: 2026-04-18
**Status**: Accepted

## Context

The Day-2 tokenizer shootout in design §4.3 listed four candidates: Phi-3.5-mini, SmolLM2-1.7B, Gemma-2-2b, and Llama-3.2-3B. The shootout is a selection step for primary/secondary SLM — no downstream result depends on the full candidate set, only on the chosen model.

## Decision

Drop Llama-3.2-3B. Shootout proceeds with Phi-3.5-mini, SmolLM2-1.7B, Gemma-2-2b.

## Rationale

- Three candidates still yield a valid primary+secondary selection with one comparison point.
- No entry in the §5.2 main-results matrix references Llama-3.2-3B; removing it does not affect any reported number.
- Author preference against Llama-family models for this project.

## Consequences

- CodeLlama-34B remains as a 30B baseline (design §4.3) pending separate review. Candidates if swapped: Qwen2.5-Coder-32B, or running only DeepSeek-Coder-33B as the single 30B baseline.
- No change to timeline, gates, or contributions.
