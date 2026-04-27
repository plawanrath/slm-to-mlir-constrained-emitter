# ADR-0008: Main-track scope expansion (in-line C3 + cross-IR + formal proofs)

**Date**: 2026-04-24
**Status**: Accepted

## Context

The M1 execution plan (ADR-0004, 17 days) delivered a submission-ready paper
on Day 17. The paper's current shape — "training-free method + phenomenon +
mechanism" backed by a strong cross-dialect matrix — is a plausible NeurIPS
Datasets & Benchmarks track submission.

External analysis flagged three gaps that separate D&B-track from main-track:

1. **Missing algorithmic novelty.** Our C3 is a post-hoc validator driving
   5-retry rejection sampling over three known filters (CFG, type-domain,
   scope validator). There is no novel decoder. Outlines, XGrammar,
   llguidance, SynCode, Grammar-Aligned Decoding (NeurIPS 2024), and DOMINO
   all handle static grammars. **Dynamically-bound lexical scopes as a
   first-class constraint at the decoder level is a verified literature gap.**
2. **Missing formal characterization.** Our soundness claim is empirical
   (zero false rejects on n=200). A theorem + experiment would move the
   soundness rubric.
3. **Cross-IR claim is unsupported.** We evaluate on two MLIR dialects;
   the paper's claim that the methodology generalizes to other
   TableGen-defined IRs is unvalidated.

Separate list of reviewer-objection-killers (30B + C3 rejection sampling,
modern baseline, multi-seed, functional equivalence, novelty reframing) is
low-cost and high-value; no reason not to bundle.

Buffer: 26 calendar days before NeurIPS 2026-05-20 deadline. Current M1 is
complete, so every additional day is a net add.

## Decision

Expand scope to main-track target over 21 additional execution days
(Days 18–38), structured in four phases with hard go/no-go gates.

### Phase A — Low-cost high-ROI fills (Days 18–21, 4 days)

- **30B + C3 rejection sampling** on both dialects. Our C3 validator is
  model-agnostic. Kills the constraint-asymmetry objection regardless of
  outcome.
- **Modern baseline**: StarCoder2-15B (2025-vintage) via Ollama, free +
  C1 on both dialects. Closes the "only 2023 models" objection.
- **Novelty reframe**: abstract/intro/contributions lead with mechanical
  ODS→constraint synthesis as the core contribution. Phenomenon (SLM beats
  30B) becomes supporting evidence.
- **Integration**: regenerate matrix + figures, daily logs.

**◆ Gate**: if 30B + C3 raises 30B verify above SmolLM2's, halt and reassess.

### Phase B — In-line C3 (Days 22–28, 7 days) — Tier 1 novelty

The paper's missing technical core. Bypass Outlines 1.2's closed CFG mask
interface; compile LARK directly to a token-level automaton; couple it with
a symbol-table state machine; intersect masks at each decode step.

- Day 22: custom LARK→automaton compilation path
- Day 23: joint (parser_state, symbol_table) state machine
- Day 24: BPE-aware in-scope name trie for SSA use positions
- Day 25: MLX sampling-loop integration + unit tests
- Day 26: equivalence test (in-line vs rejection-sampled, should match)
- Day 27: re-run SmolLM2 matrix with in-line C3 cell
- Day 28: Tier 2 soundness theorems (in parallel starting Day 26)

**Theorem**: under in-line C3, every generated string is in
L(C1) ∩ L(C2) ∩ L(C3). Corollary: zero false rejects by construction.

**Expected payoff**: 2× speedup on arith+func (1.83 attempts → 1); verify
equivalent to rejection-sampled at convergence with tighter CIs; framing
shift from "compose three known filters" to "novel decoder for
dynamically-bound lexical scopes."

**◆ Gate (Day 26)**: if in-line verify-rate is significantly lower than
rejection-sampled at convergence, there's a bug — debug before measurement.

### Phase C — Strong additions (Days 29–35, 7 days)

- **Tier 3 cross-IR generalization (Days 29–33)**: StableHLO. 10 named ops
  in memref/tensor semantics. Same mechanical ODS pipeline. Hand-authored
  StableHLO-Spec-30 benchmark. SmolLM2 + 30B + StarCoder2 matrix.
- **Multi-seed (Day 34)**: seeds 0,1,2 at n=100 on 5 critical cells.
  Tighter CIs on the hero result.
- **Functional equivalence (Day 35)**: 20-sample spot check — lower
  generated MLIR through LLVM pipeline, compile with clang, run on 10
  random inputs, compare vs hand-written reference C. Appendix subsample.

### Phase D — Paper rewrite + submission (Days 36–38, 3 days)

- Day 36: method + results rewrite around in-line C3, regenerate all
  figures with new numbers + StableHLO column
- Day 37: cross-IR section + functional-equivalence appendix integration
- Day 38: rebuild tarball, final internal review, submission-ready

Total extension: 21 days. Buffer after: ~5 days before May 20.

## Consequences

- Paper track target shifts from D&B to **NeurIPS main track**.
- Core contribution reframed: mechanical ODS→constraint synthesis (method)
  + in-line joint CFG + dynamic-scope decoder (algorithm) + phenomenon
  across 3 schema-rich IRs (result).
- `decoder/c3_scope.py` (rejection-sampling) remains as a baseline cell;
  new `decoder/c3_inline.py` is the shipped algorithm.
- Outlines 1.2 is bypassed for in-line C3. Custom automaton compiler or
  direct llguidance lower-level API wired into MLX generate loop.
- StableHLO added as Tier 3 IR (concrete target, not a hypothetical).
- Functional equivalence measured on 20 samples — this is the reviewer-magnet
  metric.
- Fallback plan: if Tier 1 in-line C3 slips >10 days, ship Phase-A-enhanced
  paper (still stronger than current D&B, still viable submission).

## Alternatives considered

1. **Submit current paper as D&B** (0 extra days): tested, strong,
   defensible. Rejected because user explicitly targets main track.
2. **Lean main-track** (Phase A + Tier 1 only, 14 days): strong algorithmic
   contribution + quick wins, skip StableHLO and functional equivalence.
   Rejected by user ("aim for a star") in favor of full push.
3. **Full push with overflow into post-submission revisions**: if Phase C
   runs long, could defer multi-seed / functional-equivalence to
   rebuttal/camera-ready. Retained as a fallback.

## Risks

See Main-track risks table in the plan response.
Key residual risks:
- In-line C3 BPE trie harder than 5-day estimate
- StableHLO pipeline blocks on MLIR dialect registration
- Functional equivalence LLVM lowering finicky
- 30B + C3 flips headline

Mitigation: each phase has a decision gate; Phase A alone is main-track
viable if Tier 1 slips catastrophically.

## Supersedes

- `docs/execution_plan.md` §6 "Days 8-17" — entire post-Day-17 schedule.
- ADR-0006 "Fallback if in-line integration blocked": rejection-sampling
  remains the baseline, but in-line is now the production path for the
  paper's headline system.
- `docs/paper/main.tex` contribution framing — rewrite in Phase A.
