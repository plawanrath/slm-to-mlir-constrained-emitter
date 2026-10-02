# Datasheet for Datasets — NL→MLIR Benchmark Suite

Following Gebru et al. "Datasheets for Datasets" (2021).

Covered datasets:
- `MLIR-Spec-150` (arith+func+memref, 150 prompts)
- `Linalg-Spec-30` (linalg named ops under memref semantics, 30 prompts)
- `StableHLO-Spec-30` (10 op families, 30 prompts)
- `StableHLO-Held-Out-200` (parametric sweep, 200 prompts)
- `StableHLO-OutOfGrammar-25` (ops outside our 10-op grammar scope, 25 prompts)
- `MLIR-Functional-Reference-30` (execution-based functional references, 10 per dialect, 30 records)

`MLIR-Functional-Reference-30` is released as evaluation evidence for
bounding the verify-valid → functional gap, not as part of the test
benchmarks (see the paper's Experimental Setup and Limitations); where its answers differ from the
other five, they are given separately below.

## Motivation

### Why was this dataset created?

Natural-language-to-MLIR (NL→MLIR) is an under-represented task in
existing code-LM benchmarks. Existing MLIR corpora (LLVM in-tree
`mlir/test/`) are designed for compiler regression testing, not NL
pairing; HumanEval/MBPP/APPS target general-purpose Python. This
suite fills the gap with five aligned NL→MLIR benchmarks spanning
three dialects (arith+func, linalg, StableHLO), plus a small
execution-based functional reference set that measures how often
verifier-valid generations compute the intended function.

### Who created the dataset?

Plawan Kumar Rath (single author).

### Who funded the creation?

Self-funded; no institutional grant.

## Composition

### What do the instances represent?

Each instance is a natural-language task description paired with a
reference MLIR program that implements it. All reference MLIR
programs verify under the intended verifier (`mlir-opt
--verify-diagnostics` for arith+func+linalg; `iree-compile
--compile-to=input` for StableHLO).

`MLIR-Functional-Reference-30`: each instance is a functional test for
one spec prompt: a canonical function name and signature, concrete
inputs, and the expected output under a dialect-specific execution
wrapper (`mlir-cpu-runner` for arith and linalg+memref;
`iree-run-module` for StableHLO).

### How many instances in total?

435 benchmark instances in the five test sets (150 + 30 + 30 + 200 + 25).
`MLIR-Functional-Reference-30` adds 30 records, but each attaches
execution references (inputs and expected output) to an existing prompt in
MLIR-Spec-150, Linalg-Spec-30, or StableHLO-Spec-30 (fields
`source_benchmark`, `source_id`), so it adds no new instances. The six
repositories hold 465 JSON records in total.

### Does the dataset contain all possible instances or a sample?

Samples. `MLIR-Spec-150` is a stratified author-curated sample over
the 12 named ops in scope for arith+func+memref; `Linalg-Spec-30`
covers the 12 linalg named ops we support; `StableHLO-Spec-30`
covers 10 op families the grammar accepts; `StableHLO-Held-Out-200`
is a parametric sweep over 7 op families × 6 dtypes × 3 shape
ranks filtered to verifier-clean (585 raw → 200 kept);
`StableHLO-OutOfGrammar-25` is 25 programs using ops NOT in our
grammar scope; `MLIR-Functional-Reference-30` samples 10 prompts per
dialect from `MLIR-Spec-150`, `Linalg-Spec-30` and `StableHLO-Spec-30`,
biased toward elementary numerical operations.

### What data does each instance consist of?

JSON with fields: `id` (string), `nl` (string, NL description),
`mlir` (string, reference MLIR), `dialect` (string), `difficulty`
(string; "easy"/"medium"/"hard" or "programmatic" or "programmatic-
wide"), `notes` (string, optional), `source` (string, provenance
tag).

`MLIR-Functional-Reference-30` records instead carry: `id`, `dialect`,
`nl`, `source_benchmark` and `source_id` (the spec prompt it tests),
`canonical_fn_name`, `canonical_signature`, `result_type`, the inputs
(`inputs` / `scalar_inputs` for scalars, `memref_inputs` and
`memref_print` for buffers, `iree_inputs` for StableHLO), and the
oracle (`expected_output`, `expected_stdout_regex`,
`expected_output_pattern`).

### Is there a label / target?

The reference `mlir` field is the expected output for pass-rate
evaluation. Verify-valid pass-rate under the target dialect's
verifier is the primary metric. For `MLIR-Functional-Reference-30`
the target is the expected execution output; a candidate is scored
through verify → lower → execute → output-match
(`eval/functional/run_functional.py`).

### Is any information missing from individual instances?

No. Each instance is self-contained.

### Are relationships between individual instances made explicit?

No cross-instance dependencies within a benchmark.
`MLIR-Functional-Reference-30` records point to their source spec
prompt via `source_benchmark` / `source_id`.

### Recommended data splits?

Not pre-split. All instances are intended as a test set for
zero-shot evaluation under the 3-shot in-context priming protocol
described in the paper. Users running methods that require a
training set should hold out programs themselves.

### Errors / redundancies / noise?

No known label errors. Some semantic near-duplicates exist within
the parametric sweep (e.g., `stablehlo.add f32 shape 16` vs
`shape 32`) by design — we sweep shapes to measure shape-robustness.

### Self-contained or links to external data?

Self-contained. No external data fetches required at evaluation
time.

### Confidential data?

No confidential, proprietary, or PII data.

### Offensive / insulting / threatening / anxiety-inducing content?

No. All content is technical MLIR code and NL descriptions of
numerical operations.

### Subpopulations?

Not applicable — no human-subject or demographic data.

### Individuals identifiable?

No.

## Collection

### How was the data acquired?

- `*-Spec-*`: hand-authored by the author, with each
  reference MLIR verified against the target tool before inclusion.
- `Held-Out-200`: generated by a parametric sweep script
  (`scripts/day_f8_held_out_200.py` in the code repository).
  Each candidate verified by `iree-compile --compile-to=input`;
  only clean candidates kept.
- `OutOfGrammar-25`: hand-authored against the StableHLO spec for
  ops our grammar does not cover; each verified by `iree-compile`.
- `Functional-Reference-30`: hand-authored by the author on top of
  prompts from the three `*-Spec-*` benchmarks (10 per dialect).

### Validation mechanisms?

Every reference MLIR is verifier-clean at the time of release.
`eval/benchmarks/mlir_spec_150/validate.py` (mlir-opt verification) and
`scripts/verify_concordance.py` (StableHLO verifier concordance) are
included in the code repository so anyone can re-validate.

### Who was involved?

A single author. No crowdsourcing, no hired labelers, no
LLM-authored reference programs.

### Collection time frame?

2026-04-18 through 2026-04-21 for the `*-Spec-*` benchmarks;
the remaining sets were completed by 2026-04-30.

### Ethical review?

Not applicable — no human-subject data.

## Preprocessing

### Was preprocessing done?

Minimal. Canonicalization via `mlir-opt --canonicalize` was
considered but NOT applied to reference MLIR (we preserve the
author-written form to match the expected model output).

### Was raw data saved?

Yes — the pre-filter candidate list for `Held-Out-200` is in
`scripts/day_f8_held_out_200.py`.

## Uses

### Has the dataset been used already?

Yes — in this paper. Every results table and figure in the paper
uses these benchmarks.

### Leakage and memorization

- **Provenance.** The 210 spec prompts (150 arith+func, 30 linalg,
  30 StableHLO) and their reference modules were hand-authored fresh
  for this benchmark in April 2026, not scraped from any corpus.
  `Held-Out-200` is generated by a parametric sweep and
  `OutOfGrammar-25` is hand-authored; neither is drawn from existing
  code.
- **Timing.** All reference programs post-date the pretraining
  cutoffs of every model evaluated in the paper (2023–2024 releases).
- **No gold reproduction.** No SmolLM2 generation reproduces its
  gold module, even after whitespace normalization (0 of 540
  spec-prompt generations across seeds). The 44 matches among 2,160
  generation–gold pairs are baseline outputs of short canonical
  programs (8 distinct prompts; seven single-op programs and one
  two-op clamp). Recomputed by `scripts/day61_rebuttal_numbers.py`.
- **Direction of bias.** To the extent LLVM/MLIR test files appear
  in pretraining data, they are at least as likely to be in the
  large public-code crawls behind the 15B–34B code baselines as in a
  1.7B SLM's corpus. Memorizing MLIR test idioms would therefore
  inflate the baselines, working against the paper's claim.
- **Padding prompts.** The paper's extended evaluation pools add
  machine-generated weak descriptions of LLVM/MLIR test files. These
  are not part of the released datasets. The paper discloses their
  duplicate structure (193 unique of 200 arith, 75 of 125 linalg;
  disjoint from the spec prompts), and every conclusion holds on
  unique prompts.

Future users should still treat any model trained after this
release as potentially exposed to it.

### Uses the dataset supports?

- Evaluating NL→MLIR generation systems under verify-valid pass-
  rate.
- Ablation studies over constraint-layer stacks (C1/C2/C3).
- Cross-dialect transfer studies.
- `MLIR-Functional-Reference-30`: spot-checking whether verify-valid
  generations compute the intended function, and reporting the
  verify → functional gap on a hand-authored sample.

### Uses the dataset does not support?

- **Functional-correctness testing from verify-valid alone**.
  Verify-valid ≠ functionally correct; a program can pass the
  verifier while computing the wrong function. Use the
  gold-differential component (`eval/functional_differential/`),
  which executes generations against the gold module of all 180
  spec prompts on randomized inputs, alongside the 30 hand-authored
  references of `MLIR-Functional-Reference-30`.
- **Training data**. The benchmarks are intended for evaluation
  only. We ask users not to fine-tune on them, because doing so
  would contaminate future evaluation. This is an intended-use
  request, not a license term: the Apache-2.0 license does not
  restrict training.

### Impact on future uses?

None known. The benchmarks are static.

### Tasks the dataset should not be used for?

See "uses does not support" above.

## Distribution

### Will the dataset be distributed?

Yes. Publicly on Hugging Face and in the code repository
(https://github.com/plawanrath/slm-to-mlir-constrained-emitter).

### Distribution mechanism?

Hugging Face Datasets, one repository per benchmark, grouped in the
collection
https://huggingface.co/collections/plawanrath/cross-dialect-mlir-benchmarks-neurips-2026-e-and-d-6ab64ca74edba2b4380216fd.
Each ships MLCommons Croissant 1.1 metadata (core + RAI + provenance)
as `croissant.json`, mirrored in the code repository at
`eval/benchmarks/<name>/croissant.json` and `eval/functional/croissant.json`.

### When will it be distributed?

Publicly available since the NeurIPS 2026 camera-ready (September 2026).

### License?

**Apache License 2.0** (SPDX: Apache-2.0). Both the data and the
code. No IP-restrictive terms.

### Third-party IP restrictions?

No. The reference MLIR programs are constructed against public
dialect specs (LLVM MLIR, OpenXLA StableHLO); the specs themselves
are separately Apache-2.0.

### Export control / regulatory restrictions?

None known.

## Maintenance

### Who will maintain it?

Plawan Kumar Rath, via the GitHub repository.

### Contact?

GitHub issues at https://github.com/plawanrath/slm-to-mlir-constrained-emitter/issues, or the Hugging Face dataset discussion tabs.

### Erratum / update policy?

Tagged releases on the GitHub repository, starting with
`v1.0-camera-ready`, whose release asset `paper_results.tar.gz` holds the
per-prompt generations behind the paper. Errata filed against the current
tag result in a new tag; the old tag remains available.

Errata for the NeurIPS 2026 camera-ready (no benchmark instance changed):

- `MLIR-Functional-Reference-30`, StableHLO `divide` (f64) reference:
  the reference declares 8xf64 inputs while its gold source declares
  `tensor<32xf64>`, and IREE demotes f64 to f32 by default. The
  record is left unchanged; this reference is excluded from
  gold-gated scoring.
- Harness: the released linalg lowering pipeline lacked
  `--convert-math-to-llvm`, so the linalg elementwise `exp`
  reference failed at execution; fixed in
  `eval/functional/run_functional.py`. The StableHLO 1-D `exp`
  reference's failure is a fixed-input limitation (the candidate
  declares a different static shape); the gold-differential harness
  scores it correctly. The
  released fixed-input results are kept as the released-protocol
  record.

### Newer versions?

Planned extensions in the paper's future-work section: (i) curated
growth of the hand-authored functional reference set (randomized-input
differential testing over all 180 spec prompts already ships as
`eval/functional_differential/`), (ii) cross-target generation for
LLHD/NVVM/SPIR-V.

### Augmentation/extension by others?

Yes. The Croissant metadata declares Apache-2.0; forks and
contributions are welcome under the same license.

### Support / host the extended dataset?

The author, via the same repository.
