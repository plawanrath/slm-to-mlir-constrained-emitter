# ADR-0004: Compressed 17-Day Execution Timeline + Eval Methodology Lock

**Date**: 2026-04-18
**Status**: Accepted

## Context

Original timeline (`design.md` §6) was 28 days (Apr 17 – May 15, 2026), with the NeurIPS main-track deadline ~May 20. On review, two separable questions emerged:

1. **Can we finish sooner?** Most components are <300 LoC each; the real time sinks are data-pipeline runtime, 30B baseline inference, grammar iteration, and hand-authoring the gold benchmark. Code-writing is a fraction of wall-clock.
2. **Is the originally-planned eval n adequate for NeurIPS reviewers?** An earlier proposal to run at n=200 per cell was rejected as reviewer-vulnerable: at n=200 with p≈0.5, 95% CI half-width is ±6.9pp — making the paper's claimed 3-5pp effects statistically indistinguishable from noise. The hand-authored MLIR-Spec-50 at n=50 has ±14pp half-width, far below code-gen-benchmark norms (HumanEval=164, MBPP=974).

## Decision

Compress execution to **17 days** (Apr 18 – May 4, 2026) while **increasing** eval sample sizes to reviewer-defensible levels. Details in `docs/execution_plan.md`.

### Timeline

| Phase | Original | Compressed |
|---|---|---|
| Code writing | Distributed across Weeks 1-3 | Day 1 (code-complete target) |
| Data pipelines | Days 3-5 sequential | Days 2-5 parallel + background |
| Grammar + C1 | Days 6-7 | Days 2-5 |
| C2 + entropy | Days 8-11 | Days 4-5 |
| 30B baselines | Days 12-14 sequential | Days 3-7 background |
| Full matrix + ablations | Days 15-21 | Days 5-10 |
| Hand-authored benchmark | Day 18 (single day) | Days 3-7 (spread) |
| Paper | Days 22-28 | Days 12-17 |
| Submission-ready | Day 28 (May 15) | Day 17 (May 4) |

Net buffer before NeurIPS deadline: ~16 days instead of ~5.

### Eval methodology (locked)

- **MLIR-Spec-150** (up from 50): hand-authored, 2-person review, locked Day 7.
- **SLM cells**: n=1000 (L3 held-out) + n=500 (L1 held-out) per `(model × constraint × dialect × seed)`.
- **30B baseline cells**: n=300 per cell (with explicit cost table in paper appendix).
- **Paired bootstrap** across identical prompt sets: 10k resamples, 95% percentile CI.
- **Power analysis** reported in methods: "At n=N, minimum detectable effect at α=0.05 is Δpp."
- **Pre-registered gates**: `design.md` §7 restated verbatim in paper methods as "decision rules fixed before data collection."
- **All generations released** in the reproducibility package.

## Rationale

### Timeline compression

- Original plan budgeted ~5-7 days of code writing spread across 3 weeks. In practice, Day-1 code-complete (scaffolded + unit-tested) is achievable; real-data debugging happens concurrently with data-pipeline runtime.
- Data pipelines (L0, L1, L3) are independent and parallelizable.
- 30B baselines run as background compute across 5 days, not as a sequential 3-day block.
- Hand-authoring MLIR-Spec-150 spread across 5 days (Days 3-7) rather than concentrated on a single day.
- Earlier submission-ready date provides ~2 weeks of buffer before the NeurIPS deadline — useful for polish, rebuttal-prep, and catching missed ablations.

### Eval size increase

- At n=200, the claimed 3-5pp effects are within the 95% CI noise floor. Reviewers would reject or demand more data.
- MLIR-Spec-50 is below current code-gen benchmark norms. MLIR-Spec-150 matches HumanEval scale.
- SLM inference on MLX is cheap (~30 tok/s on M4 Max). n=1000 per SLM cell costs single-digit hours total.
- 30B baseline cost is the binding constraint. n=300 × 2 baselines × 2 constraint levels × 2 dialects × 3 seeds ≈ 96 hours of background compute — absorbable in the Days 3-7 window.
- Paired bootstrap across identical prompt sets increases statistical power materially over independent-sample tests; this is the appropriate test anyway since all systems see the same prompts.

## Consequences

**Positive**:
- Submission-ready by May 4 with ~16 days of buffer before NeurIPS deadline.
- Statistical claims robust to reviewer challenge (n ≥ HumanEval for gold benchmark; paired tests; pre-registered gates).
- Cost transparency (n per cell, wall-clock table) pre-empts "cherry-picked eval" critiques.
- No change to thesis, contributions, methodology, or scope.

**Negative**:
- Day-1 code-complete is aggressive. Realistic slip probability: medium-high. Day 2 buffer absorbs up to 1 day of slip before the critical path is hit.
- Hand-authoring 150 gold pairs (up from 50) costs ~3 additional days of the primary author's time, spread across Days 3-7.
- `linalg` remains a Day-7 go/no-go. If deferred, the main claim holds on `arith + func` alone, but reviewers may note narrower scope.

**Out of scope (unchanged from `design.md` §9)**:
- C3 SSA/dominance
- GRPO
- Cross-target generalization
- Dialects beyond `arith + func + linalg`

## Alternatives considered

1. **Keep original 28-day timeline, keep n=200**: rejected — n=200 is reviewer-vulnerable regardless of timeline.
2. **Compress timeline, keep n=200**: rejected for the same reason. Compression doesn't fix the statistical problem.
3. **Keep original 28-day timeline, raise n**: feasible but wastes ~2 weeks of buffer that can instead go to writing polish and rebuttal prep.
4. **Compress further (e.g., 10-day timeline)**: rejected — LARK grammar iteration and hand-authoring MLIR-Spec-150 do not compress below their current allocation without quality loss.

## Supersedes

- `design.md` §6 (28-day timeline) is superseded by `docs/execution_plan.md` §2.
- `design.md` §5.1 eval sizing is refined by `docs/execution_plan.md` §1 and this ADR.

All other sections of `design.md` remain authoritative.
