# ADR-0007: Pursue linalg (named ops) in M1

**Date**: 2026-04-20
**Status**: Accepted

## Context

Day-7 go/no-go decision on `linalg` per `docs/execution_plan.md` §2 and
`TODO.md` §6.4. Inputs gathered Day 7:

- L3 corpus has **1881 linalg samples** (22% of 8528 total), of which 733
  use `linalg.generic`. Named ops by frequency:
  matmul (250), fill (217), copy (65), transpose (34), contract (36),
  batch_matmul (28), matvec (17), broadcast (17), exp (19), add / sub /
  mul / abs (~15 each).
- Our `grammar/lattices/linalg.json` already contains 97 named ops with
  operand/result type classes (no generic).
- **5-sample SmolLM2-1.7B free-decoding probe** (linalg few-shot):
  all 5 produced linalg-shaped MLIR, 0/5 verified. Failure modes: scalar
  vs. memref type mismatches (classic C3 territory), invented ops
  (`linalg.slice` — not real), and truncated generation. Consistent with
  Day-4/5 findings — SmolLM2 follows few-shot structure but fails at
  cross-SSA type consistency until C3 is applied.
- **L3 weak-NL is not usable for linalg**: prompts are test-pathname
  fragments ("visitors", "slice", "print use nameloc as prefix"). Any
  linalg benchmark needs hand-authored NL prompts.

Our arith+func+memref result is complete and strong (Week-2 gate met at
+13pp, ADR-0006). The question is whether to strengthen the paper's
generalization claim by adding a second dialect.

## Decision

**Pursue linalg named ops in M1.** Scope strictly limited:

- **In**: linalg.{matmul, matvec, fill, copy, transpose, broadcast, add,
  sub, mul, div, exp, abs} under **memref semantics only**. ~12 ops
  covering ~85% of named-op occurrences in L3.
- **Out**: `linalg.generic` (affine indexing maps + iterator_types + yield
  regions). Tensor semantics (`ins(%t : tensor<...>)` → `-> tensor<...>`).
  Complex convs (`conv_2d_nhwc_hwcf` etc.). These are M2.

Execution plan, slotted into Days 8-10:

1. **Day 8 AM**: extend `grammar/mlir_gen_c1.lark` and `mlir_gen_c1c2.lark`
   with linalg named-op productions. Target: ≥90% round-trip on L3 linalg
   samples filtered to our 12-op subset. Iterate if needed.
2. **Day 8 PM**: extend `decoder/c3_scope.py` with linalg op-pattern
   handlers. Each of the 12 ops gets an ins/outs parser that records def
   types (outs: element type of the memref) and validates use types
   (ins: memref element-type matching plus shape constraints where
   feasible). Unit-test extension.
3. **Day 9 AM**: author a **Linalg-Spec-30** hand-authored mini-benchmark
   (~30 NL→MLIR pairs, all 12 ops covered, easy/medium/hard mix, all
   verify-clean). Stored under
   `eval/benchmarks/linalg_spec_30/examples/`.
4. **Day 9 PM**: measurement. SmolLM2-1.7B × {none, C1, C1+C2, C1+C2+C3}
   × linalg × (Linalg-Spec-30 + filtered L3 held-out n=500).
5. **Day 10**: 30B baselines on linalg. CodeLlama-34B + C1
   (rejection-sampled) and Granite-Code-34B + C1 on the same prompt sets,
   n=200. Error categorization on the new results. Regenerate §5.2 matrix.

Existing arith+func+memref results are **frozen** (see
`results/frozen/week2_gate_day5/`) and do not re-run.

## Consequences

- Paper contribution 1 becomes multi-dialect, strengthening generalization
  claim. Week-3 gate has two chances to pass (either dialect suffices).
- Design §5.2 matrix grows to add a linalg column populated end-to-end.
- `docs/execution_plan.md` Days 8-10 re-scoped (LoRA appendix deferred
  into Day 11 buffer).
- ~2 extra days consumed. Still ~7 days of buffer before submission.
- Scope boundary documented explicitly in methods section:
  "We evaluate `linalg` only under memref semantics with named ops from
  {matmul, matvec, fill, copy, transpose, broadcast, add, sub, mul, div,
  exp, abs}. Tensor-typed `linalg.generic` with affine indexing maps is
  out of M1 scope and is noted as future work." This pre-empts reviewer
  concerns about "you evaluated linalg but didn't include generic."

## Alternatives considered

1. **Freeze at arith+func+memref** — defensible, would put ~2 days into
   paper polish + LoRA + error-bucket taxonomy. Rejected because top-tier
   reviewers may see single-dialect as narrow, especially for a thesis
   about "formal intermediate representations" (plural).
2. **Pursue linalg including generic** — too much work for M1. `linalg.generic`
   alone is probably 1 week of grammar iteration (indexing maps, iterator
   types, block arguments + yield region). Rejected as out of scope.
3. **Add linalg only to MLIR-Spec-150** (no grammar / C3 work, just
   evaluate under `none`) — would show SLM can't do linalg zero-shot but
   wouldn't demonstrate our method generalizes. Rejected as weakening the
   paper.

## Supersedes

- `docs/execution_plan.md` §2 Day 7 ("Go/no-go on linalg: defer
  otherwise") — now resolved as "pursue named ops only."
- `docs/design.md` §9 future-work list — linalg generic + tensor
  semantics move from "post-M1" to "explicitly M2."
