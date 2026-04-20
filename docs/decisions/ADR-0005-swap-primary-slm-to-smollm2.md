# ADR-0005: Swap Primary SLM from Phi-3.5-mini to SmolLM2-1.7B

**Date**: 2026-04-19
**Status**: Accepted

## Context

`design.md` §4.3 named Phi-3.5-mini-instruct (3.8B) as the primary SLM and
SmolLM2-1.7B-Instruct as the secondary. This was a prior-free choice: Phi is
the biggest of the 1-3B candidates, broadly considered a strong instruction
model, and the tokenizer shootout (Day 2) showed the three candidates within
~2% on tok/char.

Day-4 SLM smoke matrix (n=200 per cell, identical prompts across conditions,
few-shot priming, arith+func domain) produced decisive contrary data.

## Measured data (Day-4 SLM smoke, few-shot prompts, n=200)

| model           | constraint | parse%                  | verify%                 | mean gen time |
|-----------------|-----------:|-------------------------|-------------------------|--------------:|
| Phi-3.5-mini    | free       | 2.0  [0.5, 4.0]         | 2.0  [0.5, 4.0]         | 13.7 s        |
| Phi-3.5-mini    | C1         | 97.0 [94.5, 99.0]       | 9.5  [5.5, 14.0]        | 5.3 s         |
| Phi-3.5-mini    | C1+C2      | 97.5 [95.0, 99.5]       | 9.5  [5.5, 14.0]        | 5.2 s         |
| SmolLM2-1.7B    | free       | 47.5 [41.0, 54.5]       | 43.0 [36.5, 50.0]       | 2.8 s         |
| **SmolLM2-1.7B**| **C1**     | **99.0 [97.5, 100]**    | **58.0 [51.0, 65.0]**   | **1.5 s**     |
| SmolLM2-1.7B    | C1+C2      | 99.0 [97.5, 100]        | 58.0 [51.0, 65.0]       | 1.5 s         |

Confidence intervals do not overlap on verify-valid; the effect is large and
unambiguous. SmolLM2-1.7B + C1 is:
- **6.1× better** on verify-valid than Phi-3.5-mini + C1 (58% vs 9.5%)
- **3.5× faster** per generation (1.5 s vs 5.3 s)
- **2× smaller** (1.7B vs 3.8B parameters)

Hypothesis for the effect (not tested; noted for discussion in paper
limitations): Phi-3.5's extensive instruction-tuning biases it toward verbose
prose completions, which the grammar constraint forces into malformed
trajectories. SmolLM2's simpler pretraining keeps it closer to pattern-matching
over the few-shot examples.

## Decision

Swap the roles:
- **Primary SLM**: SmolLM2-1.7B-Instruct (HuggingFace).
- **Secondary SLM**: Phi-3.5-mini-instruct (Microsoft). Kept in the appendix for
  the robustness-to-model-choice discussion.

Both remain US/EU-origin open-weight per feedback_model_provenance.md.

## Consequences

- `design.md` §4.3 swapped; `design.md` §5.2 primary row is now SmolLM2-1.7B.
- LoRA appendix (`train/lora_phi.py`) becomes `train/lora_smollm2.py` (rename on
  Day 7 when actually needed).
- All references to "1-3B SLM" in the contribution language still hold (1.7B is
  in that range).
- Main paper result just got stronger: a 1.7B model with C1 matches/beats
  expected 30B baselines at 1.5 s/generation on Apple Silicon.
- No hardware/venv/Docker changes required.
- No timeline impact (both models are already pulled / working in the harness).

## Alternatives considered

1. **Keep Phi primary; report SmolLM2 as separate**: rejected — primary should
   be the strongest model per the measured phenomenon. The paper's thesis is
   specifically about the lower-end of the 1-3B band.
2. **Pool both and report "SLM ensemble"**: rejected — obscures the single-
   model claim that makes the paper readable.

## Supersedes

- `design.md` §4.3 and §5.2 row order.
- ADR-0002 (tokenizer shootout) remains valid — SmolLM2 was already in the
  locked 3-model shootout.
