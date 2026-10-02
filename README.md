# SLM-to-MLIR Constrained Emitter

> **Accepted to NeurIPS 2026 (Evaluations & Datasets Track)**  
> [Paper (arXiv)](https://arxiv.org/abs/2607.18254) · [Datasets (Hugging Face collection)](https://huggingface.co/collections/plawanrath/cross-dialect-mlir-benchmarks-neurips-2026-e-and-d-6ab64ca74edba2b4380216fd)

Code and benchmarks for "Cross-Dialect Generalization Without Retraining: Benchmarks and Evaluation of Schema-Derived Constrained Decoding for MLIR" — a paper on training-free, ODS-aware constrained decoding for NL→MLIR generation with small language models.

## Datasets

All six datasets are on Hugging Face (Apache-2.0), each with validated MLCommons Croissant 1.1 metadata (core + RAI) in its `croissant.json`. The first five hold the 435 benchmark instances: four NL→MLIR benchmarks plus the StableHLO-OutOfGrammar-25 stress set. MLIR-Functional-Reference-30 attaches hand-authored inputs and expected outputs to 30 existing Spec prompts, so it adds no new instances. Local copies of the records and Croissant files live under `eval/benchmarks/` and `eval/functional/`.

| Dataset | Dialect(s) | n |
|---|---|---|
| [MLIR-Spec-150](https://huggingface.co/datasets/plawanrath/MLIR-Spec-150) | arith + func + memref | 150 |
| [Linalg-Spec-30](https://huggingface.co/datasets/plawanrath/Linalg-Spec-30) | linalg (memref semantics) | 30 |
| [StableHLO-Spec-30](https://huggingface.co/datasets/plawanrath/StableHLO-Spec-30) | StableHLO | 30 |
| [StableHLO-Held-Out-200](https://huggingface.co/datasets/plawanrath/StableHLO-Held-Out-200) | StableHLO (programmatic sweep) | 200 |
| [StableHLO-OutOfGrammar-25](https://huggingface.co/datasets/plawanrath/StableHLO-OutOfGrammar-25) | StableHLO (ops outside the grammar) | 25 |
| [MLIR-Functional-Reference-30](https://huggingface.co/datasets/plawanrath/MLIR-Functional-Reference-30) | all three (execution references on 30 Spec prompts) | 30 |

Datasheet: [`docs/datasheets/datasheet.md`](docs/datasheets/datasheet.md).

## Citation

```bibtex
@inproceedings{rath2026crossdialect,
  title     = {Cross-Dialect Generalization Without Retraining: Benchmarks and Evaluation of Schema-Derived Constrained Decoding for {MLIR}},
  author    = {Rath, Plawan Kumar},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026},
  eprint    = {2607.18254},
  archivePrefix = {arXiv}
}
```

**Thesis**: For formal intermediate representations, ODS-derived structural priors (C1 syntactic + C2 type-domain + C3 SSA-scope) applied at inference time close most of the capability gap between a **1.7B SLM** and 30B+ models — without fine-tuning, without RL, without distillation.

## Repository layout

| Path | Contents |
|---|---|
| `grammar/` | ODS → grammar tooling: `llvm-tblgen --dump-json` extraction (`ods_tblgen.py`), per-dialect type/arity lattices (`lattices/`), LARK grammars |
| `decoder/` | Constrained decoding layers: C1 syntax (`c1_cfg.py`), C2 type/arity (`c2_type_arity.py`), C3 SSA scope (`c3_scope.py`) |
| `eval/` | Benchmarks (`benchmarks/`), functional reference set and runner (`functional/`), harness, statistics, baselines, ablations |
| `scripts/` | Experiment scripts (`dayNN_*.py`, one per experiment), figure generation (`make_paper_figures_final.py`), environment (`env/`) |
| `data/pipelines/` | Corpus pipelines (ODS, Polygeist, MLIR test mining) |
| `docs/` | Datasheet (`datasheets/`) and architecture decision records (`decisions/`) |
| `archive/` | Superseded benchmark versions kept for provenance |

## Reproducing

[`RUNBOOK.md`](RUNBOOK.md) has the step-by-step commands: environment setup (pinned LLVM container + project venv + Ollama), experiment runs, and figure regeneration.

- **Environment**: `scripts/env/` — pinned LLVM/MLIR 19.1.7 container (`Dockerfile.llvm`, `docker-compose.yml`), host wrappers (`bin/`), and locked Python dependencies (`requirements.lock.txt`).
- **Hardware**: all main experiments run on a single Apple M4 Max (128 GB unified memory).
- **Models**: SmolLM2-1.7B-Instruct (primary SLM, MLX); baselines CodeLlama-34B, Granite-Code-34B, StarCoder2-15B (Ollama, Q4) and Granite-Code-8B (fp16 control, MLX).
- **Constrained decoding**: Outlines + llguidance with LARK grammars (C1), plus the C2/C3 layers in `decoder/`.
- **Verifiers**: `mlir-opt --verify-diagnostics` (LLVM 19.1.7) for arith/func/memref/linalg; `stablehlo-opt` v1.4.0 and `iree-compile --compile-to=input` for StableHLO.
- **Baseline runtime pin**: the published Ollama baseline generations reproduce byte-identically under **Ollama 0.32.1** (the official CLI binary can run beside a newer app: `OLLAMA_HOST=127.0.0.1:11435 ollama serve`, then point the client at it with `OLLAMA_HOST=http://127.0.0.1:11435`). Newer Ollama releases change seeded outputs. `scripts/day61_frozen_c1only_baselines.py` gates on this before generating anything.

### Response-period and camera-ready results

`results/` is not tracked in git. It ships as `paper_results.tar.gz`, attached to the [`v1.0-camera-ready` release](https://github.com/plawanrath/slm-to-mlir-constrained-emitter/releases/tag/v1.0-camera-ready): every per-prompt generation and aggregate JSON behind a number, table, or figure in the paper. The Hugging Face datasets hold the benchmarks (inputs); this asset holds model outputs on them. Extract it at the repository root (see the README inside, which maps each paper table and figure to its files). Every number reported during the NeurIPS 2026 E&D author response is then recomputable from `results/`:

| Evidence | Script | Results |
|---|---|---|
| Frozen-pool constraint ladders (arith+func, linalg) | `scripts/day53_e10_ladder_frozen.py`, `scripts/day59_e10b_ladder_linalg_frozen.py` | `results/day53/`, `results/day59/` |
| SmolLM2 under the baselines' post-hoc protocol; 256-token cap | `scripts/day54_e11_smollm2_posthoc.py` | `results/day54/` |
| Baselines rerun at 600 tokens | `scripts/day55_e9_baselines_600.py` | `results/day55/` |
| Gold-differential functional evaluation (180 spec prompts) | `eval/functional_differential/` | `results/day56/`, `results/day61/functional_per_prompt_180.jsonl` |
| Randomized trials on the 30 functional references | `scripts/day57_e12a_randomized_refs.py` | `results/day57/` |
| Signature-adapted wrapper re-execution (linalg) | `scripts/day58_e12b_adapted_wrappers.py` | `results/day58/` |
| StableHLO verifier version drift | `scripts/day60_e15_iree_drift.py` | `results/day60/` |
| Pools, duplicates, 67.5-vs-52.0 forensics, truncation, best configuration, unique-prompt aggregation, leakage | `scripts/day61_rebuttal_numbers.py` | `results/day61/rebuttal_numbers.json` |
| Frozen-pool arith+func baselines without C3 (C1-only, free) | `scripts/day61_frozen_c1only_baselines.py` | `results/day61/frozen_c1only_*` |
| Frozen-pool error categories, paired rung deltas, Figures 5-7, both aggregations, provenance labels | `scripts/day61_ladder_derivatives.py`, `scripts/day61_aggregations.py` | `results/day61/` |
| C3 scope validator vs `mlir-opt` (false rejects per system), Figure 1 | `scripts/day61_c3_false_rejects.py` | `results/day61/c3_false_rejects.json` |
| Hidden Cost of Structure replication on the frozen pool (SmolLM2, Phi-3.5-mini), Figure 8 | `scripts/day61_hcs_frozen.py` | `results/day61/hcs_*` |

Errata (no benchmark instance changed) are listed in the datasheet (`docs/datasheets/datasheet.md`, "Erratum / update policy"); the camera-ready paper corrects them in place.

## Target dialects

`arith`, `func`, `memref`, `linalg`, and `stablehlo`.

## License

Apache-2.0 for both code and data (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). Copyright 2026 Plawan Kumar Rath.
