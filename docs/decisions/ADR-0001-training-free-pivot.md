# ADR-0001: Training-Free Framing for M1

**Date**: 2026-04-17
**Status**: Accepted

## Context

Original framing centered on SFT + GRPO with verifier-shaped rewards (RLVR on formal IR). Two constraints surfaced during planning that made this infeasible for the originally targeted 4-week submission window:

1. **Hardware**: single Apple M4 Max 128GB. No GPU cluster. MLX/llama.cpp work; vLLM and distributed training stacks do not. GRPO iteration loops are 10× slower than on H100 and cannot fit in 4 weeks.
2. **Novelty crowding**: RLVR on structured outputs is already the dominant 2025-2026 framing (AutoTriton, TritonRL, Kevin, KernelEvolve). Yet-another-RLVR-paper-on-a-new-domain risks reviewer fatigue.

## Decision

Pivot M1 to **training-free at inference** for the main empirical claim. The paper demonstrates that ODS-derived structural priors alone close most of the SLM-vs-large-model gap on NL→MLIR. LoRA fine-tuning is reported only as an appendix ablation to show the training-free claim's robustness.

## Consequences

**Positive**:
- Fits M4 Max hardware comfortably (inference only for main experiments).
- Contrarian framing vs. current RLVR zeitgeist — more memorable at review.
- Pre-empts "Hidden Cost of Structure" (RANLP 2025) head-on: show ODS-aware GCD reverses the effect that plain CFG-GCD causes in small models.
- Reproducibility is trivial (laptop-scale), which reviewers favor.
- No compute-fairness critiques possible across baseline comparisons.

**Negative**:
- Sacrifices one of three originally-scoped contributions (structured verifier RL rewards become future work).
- Risk that reviewers consider training-free methods "not a real capability improvement." Mitigated by the entropy-analysis contribution showing mechanism, not just numbers.

**Out of scope for M1** (explicit future-work list):
- C3: SSA/dominance constrained decoding
- GRPO with structured verifier rewards
- Cross-target generalization (LLVM IR, WASM)
- Additional dialects beyond `arith`+`func`+`linalg`

## Alternatives considered

1. **Original RLVR framing on GPU rental**: rejected — user hardware constraint is fixed; GPU rental out of budget.
2. **Cut scope but keep training**: would require 6-month timeline and miss the original 4-week submission window.
3. **Workshop-only submission**: rejected in favor of swinging for main with this tighter scope.
