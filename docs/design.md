# Design Doc — SLM-to-MLIR Constrained Emitter (M1)

**Status**: locked 2026-04-17. Changes require an ADR in `docs/decisions/`.

> **2026-04-18 update**: timeline compressed to 17 days and eval sample sizes raised per **ADR-0004**. The authoritative execution schedule is now [`docs/execution_plan.md`](execution_plan.md). §6 (timeline) is superseded by it.
>
> **2026-04-19 updates**:
> - **ADR-0005** — primary SLM swapped from Phi-3.5-mini (3.8B) to **SmolLM2-1.7B**. Day-4 measurement: SmolLM2 + C1 = 58% verify-valid / 1.5 s gen vs Phi + C1 = 9.5% verify-valid / 5.3 s gen (n=200). Phi kept as secondary.
> - **ADR-0006** — **C3 (SSA/dominance) scoped back into M1** after error-category analysis showed C3 is the verify-valid bottleneck ("other" bucket = 80/200 cross-SSA type errors that neither C1 nor C2 can fix). The §5.2 matrix gains a C1+C2+C3 row; §9 future-work loses its C3 entry.

---

## 1. Thesis

For formal intermediate representations, ODS-derived structural priors applied at **inference time** close most of the capability gap between 1-3B small language models and 30B+ models — without fine-tuning, without RL, without distillation. Training adds an incremental lift, not a categorical one. The implication: in this regime, capability is gated by *constraint richness*, not model scale.

## 2. Contributions

1. **Phenomenon (empirical)**: a **1.7B SLM** (SmolLM2-1.7B) + ODS-aware constrained decoding (C1+C2+C3) matches 30B+ open models (CodeLlama-34B, Granite-Code-34B, both quantized Q4) at verifier-pass rate on NL→MLIR across 3 dialects, with **zero gradient updates**.
2. **Algorithm**: ODS-derived constraint construction — type-domain grammar splits + operand-arity structural enforcement + **SSA-scope symbol-table tracker (C3)**, composable with any transformer decoder. Categorically beyond CFG-based GCD.
3. **Mechanism analysis**: quantify MLIR token entropy and error-category distribution under progressive constraint tightening (base → C1 → +C2 → +C3); show entropy collapse and error-bucket collapse match task-accuracy recovery. The "why it works" artifact.

## 3. Scope (locked)

| Dimension | Choice |
|---|---|
| Training regime | **Training-free** for main results. LoRA fine-tune reported as appendix ablation only. |
| Data pipeline | **Frontier-model-free** (no paraphrase-via-LLM). Three existing sources only. |
| Dialects | `arith`, `func`, `linalg` (3). `tosa` stretch goal if week 2 slack allows. |
| Cross-target generalization | **Future work**. Not in this paper. |
| SSA/dominance constraint (C3) | **In M1** (per ADR-0006). Symbol-table tracker; in-line logits processor or post-hoc rejection-sampling fallback. |
| Proper RL (GRPO) | **Future work**. Not in this paper. |
| Hardware | Apple M4 Max 128GB unified memory, single machine. |
| Venue | NeurIPS 2026 main track. Abstract ~May 13, full paper ~May 20. |
| Timeline | 4 weeks (Apr 17 – May 15, 2026). |

## 4. Methodology

### 4.1 Data pipeline (frontier-free)

| Layer | Source | Target pairs | Status |
|---|---|---|---|
| L0 | ODS `.td` `summary`/`description` mining across LLVM monorepo + IREE + StableHLO | ≥15k op-level (NL, op-signature) | Day 3 |
| L1 | CodeSearchNet C subset → Polygeist → MLIR (`func`+`arith`+`memref` lowering at -O0) | ≥50k (docstring, MLIR) | Day 4 |
| L3 | LLVM `mlir/test/**/*.mlir` with FileCheck + pathname as weak NL | ≥5k (weak-NL, MLIR) | Day 5 |

Evaluation gold set: `MLIR-Spec-50`, hand-authored, 2-person review. Day 18.

### 4.2 Constraint engine

- **C1 — Syntactic (LARK → Outlines/llguidance mask)**: explicit-WS generation grammar for target dialects covering the subset we emit. Day 6-7.
- **C2 — Semantic (ODS-derived type+arity)**: type lattice per dialect from `.td` files; type-domain grammar splits (ARITH_BIN_INT vs ARITH_BIN_FLOAT, etc.); operand-arity enforced by production structure. Day 8-9.
- **C3 — SSA/dominance (ADR-0006, scoped in)**: symbol-table tracker seeded with function parameters; SSA operand tokens constrained to names currently in scope. In-line logits processor layered on Outlines CFG, or post-hoc rejection-sampling fallback. Day 5-6 of the 17-day execution plan.

### 4.3 Models

| Role | Model | Quantization | Source |
|---|---|---|---|
| Primary SLM | **SmolLM2-1.7B-Instruct** | fp16 via MLX-LM | HuggingFace (ADR-0005) |
| Secondary SLM | **Phi-3.5-mini-instruct** (3.8B) | fp16 via MLX-LM | Microsoft |
| Tokenizer shootout (day 2) | Phi-3.5-mini, SmolLM2-1.7B, Gemma-2-2b | — | — |
| Large baseline 1 | **CodeLlama-34B-Instruct** | Q4_K_M via llama.cpp | Meta |
| Large baseline 2 | **Granite-Code-34B-Instruct** | Q4_K_M via llama.cpp | IBM |
| Appendix fine-tune | Phi-3.5-mini + LoRA (rank 16) | via MLX-LM | — |

Model selection is final after day-2 tokenizer shootout measures mean-tokens-per-op on a 10k MLIR sample.

### 4.4 Stack

| Component | Tool |
|---|---|
| Model runtime (SLMs) | MLX + MLX-LM |
| Model runtime (30B baselines) | llama.cpp via Ollama |
| Constrained decoding | Outlines (primary), XGrammar (fallback) |
| C→MLIR lowering | Polygeist |
| MLIR verification | `mlir-opt --verify` (LLVM pinned in Docker) |
| Grammar parser | LARK |
| ODS parser | Custom (`.td` → type lattice) |
| Appendix LoRA | MLX-LM LoRA |
| Experiment tracking | Wandb |
| Eval harness | Python + subprocess |

**Not used**: vLLM (no Apple Silicon support), TRL (not needed without RL), GRPO (out of scope), frontier LLM APIs.

## 5. Evaluation

### 5.1 Metrics

- **Primary**: pass@1 under `mlir-opt --verify` on MLIR-Spec-50 and held-out splits of L0/L1.
- **Secondary**: canonicalization-idempotence (`mlir-opt --canonicalize` leaves output semantically unchanged).
- **Tertiary**: normalized edit distance to reference MLIR (for paraphrase robustness).
- **Mechanism**: token entropy per dialect under each constraint level.

### 5.2 Main results matrix

| System | arith+func (n=200, smoke) | arith+func (n=1000, final) | linalg | MLIR-Spec-150 |
|---|---|---|---|---|
| SmolLM2-1.7B, free | 34.5 [28.0, 41.0] | TBD Day 10 | TBD | TBD |
| SmolLM2-1.7B + C1 | 54.5 [47.5, 61.5] | TBD Day 10 | TBD | TBD |
| SmolLM2-1.7B + C1+C2 | 54.5 [47.5, 61.5] | TBD Day 10 | TBD | TBD |
| **SmolLM2-1.7B + C1+C2+C3** (**ours**) | **67.5 [61.0, 74.0]** | TBD Day 10 | TBD | TBD |
| Phi-3.5-mini, free | 2.0 [0, 4] | TBD | TBD | TBD |
| Phi-3.5-mini + C1 | 9.5 [6, 14] | TBD | TBD | TBD |
| Phi-3.5-mini + C1+C2 | 9.5 [6, 14] | TBD | TBD | TBD |
| Phi-3.5-mini + C1+C2+C3 | TBD | TBD | TBD | TBD |
| CodeLlama-34B (Q4), free | 0.8 [0, 2] Day-3 | TBD | TBD | TBD |
| CodeLlama-34B (Q4) + C1 | 82.0 [76, 88] | TBD | TBD | TBD |
| Granite-Code-34B (Q4), free | 17.0 [12, 22] | TBD | TBD | TBD |
| Granite-Code-34B (Q4) + C1 | 27.0 [21, 34] | TBD | TBD | TBD |

All cells: mean + 95% bootstrap CI. "Smoke" column is Day-4/Day-5 n=200 few-shot
arith+func, single seed; final column will be n=1000 L3 held-out × 3 seeds per
the execution plan (Day 10). "C1" on 30B baselines denotes post-hoc
rejection-sampling against the LARK coverage grammar (not token-level masked
decoding, which llama.cpp doesn't support); SLM "C1" is true masked decoding
via Outlines + llguidance. Fairness caveat documented at §8 risks.

### 5.3 Required ablations

1. **Progressive constraint tightening**: base, +C1, +C1+C2, +C1+C2+C3. Entropy + pass@1 at each level.
2. **Hidden-Cost-of-Structure replication**: show plain CFG-GCD hurts Phi-3.5-mini base (per RANLP 2025); show C1+C2 reverses it.
3. **LoRA appendix**: primary SLM + LoRA vs. training-free. Demonstrates claim robustness (training adds <Xpp).
4. **Error-category analysis**: categorize failures (type, arity, dialect-misuse, syntax, SSA/scope); show each constraint layer specifically resolves its error bucket. Day-4/Day-5 data: C2 trims type errors 4→2; C3 collapses the "other" (SSA/scope) bucket from 80/200 to ~22/200 residual.

## 6. Timeline

### Week 1 (Apr 17-24) — Foundations + C1

| Day | Task |
|---|---|
| 1 | design.md (this doc); repo scaffold; Docker + pinned LLVM; MLX + Ollama + llama.cpp installed |
| 2 | Tokenizer shootout (Phi, SmolLM2, Gemma, Llama-3.2). Lock primary + secondary. |
| 3 | L0 ODS miner → 15k pairs |
| 4 | L1 compile-through via Polygeist → 50k pairs |
| 5 | L3 LLVM test-mining → 5k pairs |
| 6 | LARK MLIR grammar for `arith`+`func`+`memref`; 100% round-trip |
| 7 | Outlines + HF-MPS constrained decoding; parse-valid >95% |

**Gate**: base SLM + C1 parse-valid ≥95%. Else stop and diagnose.

### Week 2 (Apr 24 – May 1) — C2 + entropy analysis + first baselines

| Day | Task |
|---|---|
| 8 | ODS parser → type lattice per dialect |
| 9 | C2 type+arity mask integrated with decoder |
| 10 | Extend grammar + C2 to `linalg` |
| 11 | Entropy analysis across constraint levels |
| 12 | Baseline: CodeLlama-34B free decoding via Ollama |
| 13 | Baseline: CodeLlama-34B + C1 (CFG-GCD) |
| 14 | Baseline: Granite-Code-34B × {free, C1} |

**Gate**: C1+C2 beats C1 alone by ≥5pp pass@1. Else pivot framing.

### Week 3 (May 1-8) — Main experiments + ablations + benchmark

| Day | Task |
|---|---|
| 15 | Full matrix: {Phi, SmolLM2} × {none, C1, C1+C2} × {arith+func, linalg} × 3 seeds |
| 16 | 30B baselines: {CodeLlama, Granite-Code} × {free, C1} × 2 dialects × 3 seeds |
| 17 | Hidden-Cost-of-Structure replication + reversal ablation |
| 18 | MLIR-Spec-50 hand-authored benchmark; run all systems |
| 19 | Appendix: Phi-3.5-mini + LoRA vs. training-free lift |
| 20 | Error-category analysis |
| 21 | Bootstrap CIs + significance tests on all cells |

**Gate**: SLM + C1+C2 within 3pp of 30B + C1 on ≥1 dialect. Else workshop-only.

### Week 4 (May 8-15) — Writing

| Day | Task |
|---|---|
| 22 | Abstract + intro + method draft. Figures generated from `results.json`. |
| 23 | Results + ablation + analysis sections. |
| 24 | Related work (map already done). |
| 25 | Limitations + explicit future-work list (SSA/C3, GRPO, cross-target, tosa). |
| 26 | Internal review + ≥1 external reader. |
| 27 | Revisions + reproducibility package (Docker, seeds, eval scripts). |
| 28 | Buffer day. Submit. |

## 7. Go/no-go checkpoints

| Week | Gate | If fails |
|---|---|---|
| 1 | C1 parse-valid ≥95% | Stop, diagnose grammar or tokenizer |
| 2 | C1+C2 > C1 by ≥5pp | Pivot to "types alone suffice" or drop C2 |
| 3 | SLM + C1+C2 within 3pp of 30B+C1 on ≥1 dialect | Workshop-only submission |
| 4 | Writing complete by day 27 | Submit with known gaps; use rebuttal |

## 8. Risks + mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| M4 Max throughput insufficient for 33B baseline eval | Medium | Subsample eval set for large models; report on subset explicitly |
| Polygeist lowering quality insufficient for L1 data | Medium | Fall back to `clang -emit-llvm` + `mlir-translate` |
| Tokenizer chosen handles MLIR poorly | Low | Day 2 shootout precisely to prevent this |
| Outlines/XGrammar mask construction too slow on MPS | Medium | Pre-compile masks; consider XGrammar if Outlines slow |
| Reviewers: "incremental over Mündler" | Medium | Emphasize type-lattice-prefix-automata formalism; Mündler targets TS, not dialect IR |
| Reviewers: "only 3 dialects" | Medium | Pre-emptively frame as depth over breadth; position cross-target as future work |
| Data contamination in open 30B baselines | Medium | Audit Stack v2 `.mlir` overlap; report as caveat |
| "Hidden Cost of Structure" reproduces and we can't reverse it | Low-Medium | Ablation 2 is designed specifically to detect this; if reproduces, paper title changes |

## 9. Future work (explicitly scoped out)

These items are deliberately excluded from M1. They become the basis for subsequent milestones or extended-version papers.

- **Proper RL**: GRPO with structured verifier diagnostics as dense reward.
- **Cross-target generalization**: LLVM IR direct, WebAssembly text.
- **Additional dialects**: `tosa`, `scf`, `affine`, `gpu`.
- **Executable equivalence**: I/O-based correctness on dialects that support it.
- **Iterative repair loop**: bounded multi-round compiler-feedback refinement.

*(C3 — SSA/dominance — was originally listed here but moved back into M1 scope per ADR-0006; see §4.2.)*

## 10. Repository layout

```
docs/            # This doc + ADRs + paper
data/            # Pipelines + raw (gitignored) + processed
grammar/         # LARK grammar + ODS type parser
decoder/         # Constrained decoding integration
eval/            # Harness, benchmarks, baselines
scripts/         # Utility scripts (tokenizer bench, etc.)
train/           # LoRA appendix only
```

## 11. Definitions

- **pass@1**: proportion of one-shot generations that `mlir-opt --verify` accepts.
- **ODS**: Operation Definition Specification. MLIR's TableGen-based op metadata.
- **CFG-GCD**: grammar-constrained decoding using a context-free grammar only (current state of the art in e.g. XGrammar, Outlines).
- **C1/C2/C3**: successive constraint layers — syntactic / semantic (type+arity) / dominance. M1 ships C1+C2 only.
