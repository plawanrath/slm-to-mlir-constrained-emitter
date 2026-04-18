# ADR-0003: Swap DeepSeek-Coder-33B for Granite-Code-34B as Second 30B Baseline

**Date**: 2026-04-18
**Status**: Accepted

## Context

Design §4.3 originally listed DeepSeek-Coder-33B-Instruct (Q4_K_M) as the second 30B baseline alongside CodeLlama-34B. The "30B+ open code-specialized model" framing in §2 requires two comparable open-weight baselines at scale.

## Decision

Replace DeepSeek-Coder-33B with **Granite-Code-34B-Instruct** (IBM, Apache-2.0).

## Rationale

- Author preference for US/EU-provenance models; DeepSeek is PRC-origin.
- Granite-Code-34B is a direct structural analogue: dense ~34B, code-specialized, instruction-tuned, available as GGUF Q4_K_M via Ollama.
- Apache-2.0 license simplifies the reproducibility package.
- Fits comfortably on the M4 Max 128GB alongside CodeLlama-34B.
- Keeps §2's "30B+ open code-specialized" framing intact (unlike Codestral-22B or Mixtral-8x22B, which are either too small or not code-specialized).

## Consequences

- All §5.2 matrix rows and Day-14 / Day-16 timeline entries reading "DeepSeek-Coder-33B" now read "Granite-Code-34B".
- Contribution statement §2.1 updated to name CodeLlama-34B and Granite-Code-34B explicitly.
- No change to timeline, gates, or claims.
- Day-1 `ollama pull` list updated: swap `deepseek-coder:33b-instruct-q4_K_M` for `granite-code:34b-instruct-q4_K_M`.

## Alternatives considered

- **Codestral-22B** (Mistral, EU): only 22B; breaks the "30B+" framing. License constraints on the original release.
- **Mixtral-8x22B-Instruct** (Mistral, EU): 141B-total MoE, not code-specialized; baseline comparison would be weaker.
- **Run only CodeLlama-34B as a single baseline**: reduces the "two independent 30B baselines" evidence in §2.1; rejected.
