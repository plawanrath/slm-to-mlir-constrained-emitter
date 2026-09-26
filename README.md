# SLM-to-MLIR Constrained Emitter

> **Accepted to NeurIPS 2026 (Evaluations & Datasets Track)**  
> [Paper (arXiv)](https://arxiv.org/abs/2607.18254) · [Datasets (Hugging Face collection)](https://huggingface.co/collections/plawanrath/cross-dialect-mlir-benchmarks-neurips-2026-e-and-d-6ab64ca74edba2b4380216fd)

Code and benchmarks for "Cross-Dialect Generalization Without Retraining: Benchmarks and Evaluation of Schema-Derived Constrained Decoding for MLIR" — a paper on training-free, ODS-aware constrained decoding for NL→MLIR generation with small language models.

## Datasets

All six benchmarks are on Hugging Face (Apache-2.0), each with validated MLCommons Croissant 1.1 metadata (core + RAI) in its `croissant.json`. Local copies of the records and Croissant files live under `eval/benchmarks/` and `eval/functional/`.

| Dataset | Dialect(s) | n |
|---|---|---|
| [MLIR-Spec-150](https://huggingface.co/datasets/plawanrath/MLIR-Spec-150) | arith + func + memref | 150 |
| [Linalg-Spec-30](https://huggingface.co/datasets/plawanrath/Linalg-Spec-30) | linalg (memref semantics) | 30 |
| [StableHLO-Spec-30](https://huggingface.co/datasets/plawanrath/StableHLO-Spec-30) | StableHLO | 30 |
| [StableHLO-Held-Out-200](https://huggingface.co/datasets/plawanrath/StableHLO-Held-Out-200) | StableHLO (programmatic sweep) | 200 |
| [StableHLO-OutOfGrammar-25](https://huggingface.co/datasets/plawanrath/StableHLO-OutOfGrammar-25) | StableHLO (ops outside the grammar) | 25 |
| [MLIR-Functional-Reference-30](https://huggingface.co/datasets/plawanrath/MLIR-Functional-Reference-30) | all three (execution-based) | 30 |

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

## Target dialects

`arith`, `func`, `memref`, `linalg`, and `stablehlo`.

## License

Apache-2.0 for both code and data (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). Copyright 2026 Plawan Kumar Rath.
