# Execution Plan — SLM-to-MLIR M1 (17 days)

**Status**: locked 2026-04-18 via ADR-0004.
**Start**: 2026-04-18 (Day 1).
**Target submission-ready**: 2026-05-04 (Day 17).
**NeurIPS main-track deadline**: ~2026-05-20. Buffer: ~16 days.

---

## 0. Relationship to `design.md`

Compressed execution of `design.md`. Scientific content (thesis, contributions, methodology, scope) is **unchanged**. This document supersedes `design.md` §6 (timeline) and refines `design.md` §5.1 (eval n and statistics) per ADR-0004.

---

## 1. Eval methodology (locked)

All numbers below are per-cell, where a cell is `(model × constraint × dialect × seed)`.

| Slice | n | Source |
|---|---|---|
| Hand-authored gold | **MLIR-Spec-150** | Hand-written Days 3-7, 2-person review |
| Held-out L3 per SLM cell | 1000 | Random split from L3 (pathname+FileCheck weak-NL) |
| Held-out L1 per SLM cell | 500 | Random split from L1 (Polygeist docstring+MLIR) |
| Held-out per 30B baseline cell | 300 | Stratified subsample from L1+L3 union, seed-locked |
| Entropy analysis | 500 per (dialect × constraint) | Intrinsic; no pass@1 |

**Statistics**:
- **Paired bootstrap** over identical prompt sets (10,000 resamples, 95% CI, percentile method). All cross-system comparisons use paired tests — systems see the same prompt set per seed.
- **Power analysis reported in methods**: "At n=N and α=0.05, minimum detectable effect is Δpp." Computed per cell size.
- **Pre-registration**: `design.md` §7 gates are restated verbatim in the paper's methods section as "decision rules fixed before data collection."
- **Transparency**: every cell in every table reports its n in the caption. All generations released in the reproducibility package.

**Cost justification for 30B baseline subsampling** (included as a table in the paper appendix):
- Consumer hardware: Apple M4 Max 128GB, single machine.
- Measured wall-clock per generation at Q4_K_M: CodeLlama-34B ≈ 45s, Granite-Code-34B ≈ 50s. (Update with measured values on Day 3.)
- n=300 × 2 baselines × 2 constraint levels × 2 dialects × 3 seeds ≈ 7,200 generations ≈ 96 hours of background compute, runnable across Days 3-7.

---

## 2. Day-by-day schedule

Dates assume Day 1 = 2026-04-18.

| Day | Date | Primary work | Background runs |
|---|---|---|---|
| 1 | 04-18 | **Code-complete everything** (see §3). Scaffolded + unit tests passing. | — |
| 2 | 04-19 | Tokenizer shootout → lock primary+secondary SLM. Run L0 ODS miner. Run L3 test miner. Kick off L1 Polygeist pipeline. Grammar v0 iteration against L3 samples. | L1 Polygeist (long) |
| 3 | 04-20 | Grammar v1; C1 decoder on SLMs parse-valid check. Start MLIR-Spec authoring (target 25 pairs today). Kick off CodeLlama-34B free-decoding baseline on n=300 subsample. | L1 Polygeist; CodeLlama baseline |
| 4 | 04-21 | C2 mask integration (type + arity). C1+C2 sanity on 100 L3 held-out samples. Continue MLIR-Spec (target 50 total). Kick off CodeLlama + C1 baseline. | CodeLlama + C1; Granite free |
| 5 | 04-22 | **Week-1 gate**: base SLM + C1 parse-valid ≥95% on L3 held-out. Entropy analysis. Full SLM matrix on `arith+func` at n=1000. Continue MLIR-Spec (target 100). | Granite + C1 |
| 6 | 04-23 | Hidden-Cost-of-Structure replication + reversal ablation. Continue MLIR-Spec (target 130). | — |
| 7 | 04-24 | **Week-2 gate**: C1+C2 beats C1 alone by ≥5pp (paired bootstrap). MLIR-Spec-150 locked (2-person review). **Go/no-go on `linalg`**: proceed if Week-2 gate passed comfortably, defer otherwise. | LoRA training (overnight) |
| 8 | 04-25 | `linalg` grammar + C2 (if pursuing). LoRA eval. | — |
| 9 | 04-26 | `linalg` full eval (if pursuing). Error-category analysis across all cells. | — |
| 10 | 04-27 | **Week-3 gate**: SLM + C1+C2 within 3pp of 30B + C1 on ≥1 dialect (paired bootstrap). Bootstrap CIs + paired tests on every cell. Regenerate all figures from `results/*.json`. | — |
| 11 | 04-28 | Buffer: re-runs for any cell with anomalous CI width. Freeze numbers in `results/frozen/`. | — |
| 12 | 04-29 | Paper: abstract, intro, method sections. Figures already embedded from Day 10. | — |
| 13 | 04-30 | Paper: results, ablations, mechanism (entropy) analysis. | — |
| 14 | 05-01 | Paper: related work (map already done per `design.md`), limitations, explicit future-work list (C3, GRPO, cross-target, `tosa`). | — |
| 15 | 05-02 | Internal review + ≥1 external reader. | — |
| 16 | 05-03 | Revisions + reproducibility package (Dockerfile.reproduce, seeds, eval scripts, all generations). | — |
| 17 | 05-04 | Final polish. Submission-ready. | — |

---

## 3. Day-1 code-complete target

Every file below scaffolded with unit tests passing on synthetic inputs by EOD Day 1. "Code-complete" here means executable-without-crashing, not real-data-validated. Real-data debugging happens Days 2-5.

### Environment
- `scripts/env/Dockerfile.llvm` — `ubuntu:22.04` + pinned LLVM commit (MLIR+clang+FileCheck) + Polygeist built from source.
- `scripts/env/requirements.txt` — `mlx`, `mlx-lm`, `outlines`, `xgrammar`, `lark`, `transformers`, `wandb`, `pytest`, `numpy`, `scipy`, `matplotlib`.
- `scripts/env/bin/mlir-opt`, `mlir-translate`, `polygeist`, `FileCheck` — thin host wrappers that `docker exec` into a long-lived container (not `docker run` per call).
- `scripts/env/verify_cache.py` — sqlite-backed `sha256(mlir) → verify_result` cache.

### Data miners
- `data/pipelines/l0_ods/parse_td.py` — `.td` AST parser (shared with C2 type lattice).
- `data/pipelines/l0_ods/mine.py` — walks LLVM + IREE + StableHLO, emits `data/processed/l0_ods.jsonl`.
- `data/pipelines/l1_polygeist/driver.py` — drives Polygeist on CodeSearchNet C subset, verifies output, emits `data/processed/l1_polygeist.jsonl`.
- `data/pipelines/l3_tests/mine.py` — walks `mlir/test/**/*.mlir`, uses pathname + FileCheck comments as weak-NL.

### Grammar
- `grammar/mlir.lark` — LARK grammar for `arith + func + memref` (v0).
- `grammar/ods_lattice.py` — consumes `.td` AST, emits per-op `{operands, results, arity}` to `grammar/lattices/{arith,func,linalg}.json`.
- `grammar/tests/test_roundtrip.py` — round-trip test on a sampled L3 corpus.

### Decoder
- `decoder/c1_cfg.py` — MLX-LM + Outlines grammar guide, pre-compiled masks.
- `decoder/c2_type_arity.py` — type-prefix automaton + arity state machine; composable with C1.
- `decoder/generate.py` — unified `generate(prompt, model, constraint_level, seed) -> str`.

### Eval
- `eval/harness.py` — grid runner over `(model × constraint × dialect × seed)`, outputs `results/matrix.jsonl`.
- `eval/baselines/run_ollama.py` — Ollama REST wrapper; `{free, C1}` via post-hoc grammar-guided re-sampling.
- `eval/entropy.py` — token entropy per `(dialect × constraint)`.
- `eval/error_categories.py` — regex `mlir-opt --verify` diagnostics into `{type, arity, dialect-misuse, syntax, other}`.
- `eval/stats.py` — paired bootstrap, independent bootstrap, power analysis, effect-size reporting.
- `eval/ablations/hcs_replication.py` — base vs. +plain-CFG vs. +C1+C2 reversal test.

### Benchmark scaffold
- `eval/benchmarks/mlir_spec_150/authoring_template.md` — the authoring guide.
- `eval/benchmarks/mlir_spec_150/examples/` — empty, populated Days 3-7.
- `eval/benchmarks/mlir_spec_150/validate.py` — asserts all entries verify.

### Training (appendix)
- `train/lora_phi.py` — MLX-LM LoRA rank 16, single-run script.

### Scripts
- `scripts/tokenizer_shootout.py` — 3-way tokenizer comparison, writes `results/tokenizer_shootout.json`.
- `scripts/make_figures.py` — deterministic figure generator from `results/*.json`.

---

## 4. Gates (from `design.md` §7, redated)

| Gate | Date | Criterion | If fails |
|---|---|---|---|
| Week-1 | Day 5 (04-22) | Base SLM + C1 parse-valid ≥95% on L3 held-out | Stop, diagnose grammar or tokenizer |
| Week-2 | Day 7 (04-24) | C1+C2 > C1 by ≥5pp (paired bootstrap, 95% CI lower bound >0) | Pivot to "types alone suffice" or drop C2 |
| Week-3 | Day 10 (04-27) | SLM + C1+C2 within 3pp of 30B + C1 on ≥1 dialect (paired bootstrap) | Workshop-only submission |
| Writing | Day 17 (05-04) | Submission-ready draft | Submit with known gaps; use rebuttal |

---

## 5. Risks specific to the compressed schedule

Beyond the risks in `design.md` §8:

| Risk | Probability | Mitigation |
|---|---|---|
| LARK grammar v0 needs >2 days of iteration to round-trip | Medium | Day-2 buffer built in; grammar v0 intentionally scoped to `arith+func+memref` only |
| Day-1 code-complete slips to Day 2 | Medium-High | Acceptable; Day 2 already has grammar iteration scheduled |
| L1 Polygeist throughput <50k/day | Medium | Subsample target: 20k verified pairs suffice; fall back to `clang -emit-llvm` + `mlir-translate` per `design.md` §8 |
| 30B baseline wall-clock worse than estimated | Medium | n=300 already conservative; if needed, drop to n=200 and widen CI reporting |
| MLIR-Spec-150 authoring slips | Medium | Authoring spread across Days 3-7 rather than a single day; partial progress is usable |
| `linalg` C2 lattice too complex by Day 8 | High | Explicit Day-7 go/no-go; `arith+func` result alone suffices for the main claim |

---

## 6. Run-order pointer

Exact command sequences for Days 2-17 will be appended as `docs/run_order.md` once Day-1 code-complete lands. That doc will contain literal shell commands keyed to the table in §2.
