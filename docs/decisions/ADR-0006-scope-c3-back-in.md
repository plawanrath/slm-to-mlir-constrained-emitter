# ADR-0006: Scope C3 (SSA/Dominance) Back into M1

**Date**: 2026-04-19
**Status**: Accepted

## Context

`design.md` §3 scoped C3 (SSA / dominance-aware constrained decoding) out of
M1 and into future work: "*Out of scope.*" The rationale was 4-week
infeasibility.

Day-4 empirical results (B2 ablation) show that C3 is **the bottleneck**:

- SmolLM2-1.7B + C1 few-shot: 58% verify-valid, 116 passing / 80 "other" /
  4 type / 0 arity / 2 parse-fail (n=200).
- SmolLM2-1.7B + C1+C2 few-shot: 58% verify-valid, 116 passing / 80 "other" /
  2 type / 0 arity / 2 parse-fail (n=200).

Error categorization confirms:
- C2 reduces "type" errors (4→2) — real but small effect.
- "Other" (78-80) = cross-SSA-value type inconsistency (`%x` declared as index,
  referenced in body as f32) — pure C3 territory.
- Arity errors are already 0 (operand counts are structurally enforced by
  CFG alternatives).

Without C3, the verify-valid ceiling on this benchmark is ~60%. The paper's
§2 contribution 1 ("SLM + ODS-aware constrained decoding matches 30B+ open
models") requires lifting this. Two additional factors make C3 feasible now:

1. Day-4 timeline is ~2 days ahead of the compressed 17-day plan
   (`docs/execution_plan.md`). We have slack to absorb C3 work.
2. Design §4.2 defined C3 as "stateful symbol table" — a modest addition on
   top of the existing decoder integration, not a new research problem.

## Decision

Scope C3 **in** to M1. Implementation approach (finalized on Day 5):

- **Symbol-table tracker** maintained across a single module generation. Each
  SSA definition (`%name = op ... : type`) records `(name, type)` in a scope.
  Function parameters are seeded into the scope at function entry. Lives in
  `decoder/c3_scope.py`.
- **Post-hoc rejection sampling chosen as the M1 production path.** The
  in-line logits-processor route was explored on Day 5 and not pursued:
  Outlines 1.2's `Generator` wraps llguidance behind a single CFG mask with
  no public hook to compose a secondary mask, and reimplementing the MLX
  sampling loop to run our own CFG guide alongside a second mask is ~1 day
  of work for marginal speed win. We defer the in-line path to Milestone 2.
- **Rejection-sampling protocol**: generate under C1+C2 (greedy, temp=0);
  run `decoder.c3_scope.validate` on the output; on reject, resample up to 4
  more times with temp=0.8 (5 attempts total). Zero false rejects measured
  on the Day-4 n=200 SmolLM2+C1 postmortem (`scripts/day5_c3_postmortem.py`),
  so the wrapper cannot lower verify-valid below C1+C2's floor.

## Consequences

- §3 scope table update: C3 moves from "Future work" to "In M1".
- New ablation row added to §5.2: SLM + C1+C2+C3.
- §5.3 ablation item 4 (error-category analysis) gains bite: we can show C3
  specifically collapses the "other" (cross-SSA) bucket.
- `decoder/c2_type_arity.py` gains a sibling `decoder/c3_scope.py` (symbol
  table + logits processor or post-hoc validator, TBD).
- Expected timeline impact: ~2 days for in-line integration (Day 5-6). If
  in-line blocked, same 2 days for rejection-sampling path. Either fits within
  the Week-3 buffer of the 17-day plan.
- Design §2 contribution 2 ("ODS-derived constraint construction") gets richer:
  we now demonstrate type-lattice prefix automata + operand-arity state machine
  + SSA scope tracker — a full stack, not just a slice.

## Alternatives considered

1. **Keep C3 out; reframe contribution 1 as "few-shot + C1"**: rejected. The
   paper's structural-knowledge-beats-scale claim is weakest if we settle at a
   60% ceiling while 30B baselines approach or beat that number. Early Granite
   data (still running) suggests we need C3 to clearly win.
2. **Implement only the post-hoc validator (rejection sampling)**: valid
   fallback; chosen if in-line integration blocks, not the default.
3. **Leave C3 as a stretch for appendix only**: rejected — the phenomenon
   isn't the paper's without it.

## Supersedes

- `design.md` §3 row "C3 future work".
- `design.md` §9 future-work list (drop the C3 entry).
