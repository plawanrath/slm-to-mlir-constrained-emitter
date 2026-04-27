# TODO — SLM-to-MLIR M1

**Purpose**: persistent, session-resumable work log keyed to `RUNBOOK.md`.
**Source of truth**: this file. In-session task trackers mirror it but do not replace it.
**Keep in sync with**: `RUNBOOK.md` (how to run each step) and `docs/execution_plan.md` (dates + gates).

---

## How to resume from a fresh session

1. Read **Current state** below. It names the next concrete action.
2. Scan **Open blockers** — if any are present, resolve them before picking up new work.
3. Find the first unchecked (`[ ]`) item and either execute it (see the RUNBOOK §reference) or mark it `[~] in progress` and begin.
4. When done, check it off (`[x]`), write the artifact path under **Artifacts produced**, and move on.
5. **At EOD or work-day end**: write a `docs/daily_log/dayNN.md` entry using the template in `docs/daily_log/README.md`. This is paper-narrative material — don't skip.

**Status legend**
- `[ ]` pending
- `[~]` in progress (with initials + start timestamp in a trailing comment, e.g. `[~] — pl 04-19 09:00`)
- `[x]` completed
- `[!]` blocked (add a one-line reason and link to the blocker in **Open blockers**)
- `[-]` deliberately skipped (add a one-line rationale)

---

## Current state

**As of**: **Day 17 end (2026-04-24) — M1 submission-ready for D&B track.** Scope expanded per ADR-0008 to main-track target over 21 additional days (Days 18-38). Core addition: in-line joint CFG + dynamic symbol-table decoder (Tier 1 novelty) + cross-IR generalization on StableHLO (Tier 3) + formal soundness theorems (Tier 2).
**Next action**: Day 18 — 30B + C3 rejection sampling on both dialects (Phase A item 1). Running in background; see `logs/day18_30b_c3.log`.

**Day-4 numbers locked** (n=200 per cell, few-shot, arith+func):

| model           | constraint | parse%   | verify%  | gen s |
|-----------------|:----------:|---------:|---------:|------:|
| SmolLM2-1.7B    | free       | 47.5     | 43.0     | 2.8   |
| **SmolLM2-1.7B**| **C1**     | **99.0** | **58.0** | **1.5** |
| SmolLM2-1.7B    | C1+C2      | 99.0     | 58.0     | 1.5   |
| Phi-3.5-mini    | free       | 2.0      | 2.0      | 13.7  |
| Phi-3.5-mini    | C1         | 97.0     | 9.5      | 5.3   |
| Phi-3.5-mini    | C1+C2      | 97.5     | 9.5      | 5.2   |

**Decisions taken 2026-04-19**:
- **ADR-0005** primary SLM swap: Phi-3.5-mini → **SmolLM2-1.7B** (6.1× better verify, 3.5× faster, 2× smaller).
- **ADR-0006** C3 scoped back in: previously future work; Day-4 error-category analysis showed 78-80/200 failures are "other" = cross-SSA type mismatches = pure C3 territory. C1 ≈ C1+C2 on verify because few-shot already delivers C2's type-consistency work, leaving only C3 unaddressed.

**Gate status**:
- Grammar coverage gate (95.2% on L3) ✅ **met**
- Week-1 C1 parse-valid gate: ✅ **met Day 3** at 96.0% (Phi); SmolLM2 is 99%
- **Week-2 gate (revised per ADR-0006) ✅ MET Day 5**: C1+C2+C3 − C1+C2 = +13.0pp [+8.5, +18.0], p < 0.0001 on SmolLM2 n=200 arith+func few-shot.

**Baselines snapshot** (still running):
- Granite-Code-34B free ≈ 60% (n=200 complete)
- Granite-Code-34B C1 ≈ 32-48% (~n=90, trending down as L3 prompts get harder)
- CodeLlama-34B free = 0.8% (Day-3 result — pre-arith-dialect model)
- CodeLlama-34B C1 not yet started

## Open blockers

*(none — C1 integration resolved by new `mlir_gen.lark` with explicit WS. See artifact notes.)*

## Artifacts produced

- `scripts/env/requirements.lock.txt` — 93-pkg pinned set from successful install (mlx-0.31.1, mlx-lm-0.31.2, outlines-1.2.12, xgrammar-0.1.33, transformers-5.5.4, torch-2.11.0)
- Docker image `slm-mlir/llvm-pinned:latest` (LLVM 19.1.7, clang, FileCheck; Polygeist build marked optional → clang+mlir-translate fallback)
- Ollama models: `codellama:34b-instruct-q4_K_M` (20 GB), `granite-code:34b-instruct-q4_K_M` (21 GB)
- Source trees cloned: `~/src/{llvm-project,iree,stablehlo}` (depth=1)
- Seeded `results/verify_cache.sqlite` (1 entry from Day-1 smoke)

**Day-1 bug fixes logged**:
- `grammar/ods_parse.py` — ODS regex was over-strict (required `_Op` suffix); now accepts any parent class identifier and derives dialect from the leading underscore-separated token. Caught by `test_parse_two_ops`.
- `scripts/env/requirements.txt` — pins relaxed after `mlx==0.21.0` + `xgrammar==0.1.9` didn't exist on PyPI; real lock now in `.lock.txt`.
- `scripts/env/Dockerfile.llvm` — lowered `BUILD_JOBS` to 4, added `LLVM_PARALLEL_LINK_JOBS=1`, disabled assertions, split DWARF, post-install build-dir cleanup. Requires Docker Desktop memory ≥32 GB (default 8 GB OOMs).

**Day-2 artifacts**:
- `results/tokenizer_shootout.json` — Phi (0.468 tok/char, vocab 32k), SmolLM2 (0.458, 49k), Gemma-2-2b (0.437, 256k). Primary: **Phi-3.5-mini**. Secondary: **SmolLM2-1.7B**. Finding: `clean_mnemonic_rate=0%` across all — no tokenizer encodes `arith.addi` as a single token, strengthening C2's value proposition.
- `data/processed/l0_ods.jsonl` — 276 ops (183 in target dialects, 93 in extras for corpus volume). Switched from regex-based `ods_parse.py` to `mlir-tblgen --dump-json` for full inheritance resolution (`grammar/ods_tblgen.py`).
- `grammar/lattices/{arith,func,linalg,memref}.json` — lattice JSONs per target dialect: arith=49 ops, func=5, linalg=98, memref=31. `arith.addi` correctly shows `SameOperandsAndResultType` expanded and integer-only operand/result types.
- `data/processed/l3_tests.jsonl` — **8528 cases** from 2543 files (target ≥5k ✓). Split at `// -----` now correctly separates multi-case test files. Verifier-clean via `--require-verify`.
- `data/raw/c_corpus.jsonl` — 7573 C files from `bigcode/the-stack-smol` staged for L1 (deferred).
- `grammar/mlir.lark` v9 — **95.2% parse-valid on target-only L3 (n=756 full subset, multi-seed stable, 25 ms/sample)**. Covers arith (42 ops), memref (25 ops), func, generic op syntax, type/attr aliases, block labels, loc suffixes, all common attribute forms. Iteration trajectory: 0.2% → 95.2% across 9 versions. **Parsing/coverage grammar only — do not use for generation.**
- `grammar/mlir_gen.lark` — **Generation grammar for C1 decoding.** Explicit WS productions (no `%ignore`), bounded identifiers (SSA ≤32 chars, SYM ≤48 chars), capped op body (≤6 ops/func). Achieves 94% parse-valid on n=100 L3 prompts with Phi-3.5-mini MLX + outlines/llguidance. Mean gen time 2.8 s/sample. **Use this for all C1/C2 decoding; gen output is a subset of parse grammar so passes coverage gate.**
- Key finding (ADR-worthy): llguidance (outlines 1.2 default CFG backend) silently drops `%ignore WS` enforcement, which lets Phi emit infinite whitespace without advancing grammar state. Mandatory explicit WS in productions plus bounded identifier regex keep llguidance tight. Without these, C1 parse-valid is 0%; with them, 94-98%.
- `scripts/env/bin/llvm-tblgen` wrapper — added for tblgen-based ODS extraction.

**Day-2 bug fixes**:
- L0 miner: regex-based parser missed operand/result types for ops with multi-inheritance class hierarchies (e.g., `Arith_IntBinaryOp<...>, Arguments<(ins X:$lhs, ...)>`). Replaced with `mlir-tblgen --dump-json` consumer (`grammar/ods_tblgen.py`). Coverage went 37→49 arith ops, 2→98 linalg ops.
- L3 miner: treated multi-case `// -----`-separated test files as one sample. Now splits per `// -----` block before filtering. 1920 → 4540+ cases.
- Grammar v0 required `module { ... }` wrapper, rejecting bare top-level `func.func`; no attribute dicts on ops; no `func.func private`; no strided memref layouts. v1 permissive at top level, tight at op level.
- Grammar `any_top_op` fallback caused pathological Earley parses (4+ min per file). Removed; grammar now unambiguous and parses in 4 ms/sample.

**Day-2 deferrals**:
- L1 Polygeist pipeline: Polygeist build failed in container (`CMake Generate step failed`). Not a blocker since L1's role per design.md §5.3 is the LoRA appendix training corpus; main training-free results use L0+L3 only. C corpus staged at `data/raw/c_corpus.jsonl`. Revisit on Day 7 when scheduling LoRA training: options are (a) fix Polygeist cmake, (b) use IREE/XLA lowering, or (c) skip LoRA appendix.

---

## Day 1 — Code-complete (2026-04-18)

All 14 scaffolds already landed. See `RUNBOOK.md §11` for the file map.

- [x] Docker env + wrappers (`scripts/env/Dockerfile.llvm`, `scripts/env/bin/*`)
- [x] venv requirements + verify cache + pyproject
- [x] LARK grammar v0 (arith + func + memref)
- [x] ODS `.td` parser + type-lattice extractor
- [x] C1 decoder (Outlines + MLX-LM scaffold + Ollama rejection sampler)
- [x] C2 type-prefix automaton + arity state machine
- [x] L0 ODS miner
- [x] L1 Polygeist driver (with clang fallback)
- [x] L3 test miner
- [x] Eval harness + Ollama baseline wrapper
- [x] Stats (paired bootstrap + MDE) + entropy + error categorization + HCS ablation
- [x] MLIR-Spec-150 scaffold (validator + seed example) + LoRA script
- [x] Tokenizer shootout + deterministic figure generator
- [x] RUNBOOK.md

---

## §0 — One-time setup (from RUNBOOK §0)

- [ ] **§0.1** Prerequisites installed (Docker Desktop running, Python 3.11+)
- [ ] **§0.2** Build the pinned LLVM container (`docker compose … build && up -d`)
  - [ ] Verify `docker exec slm-mlir-llvm mlir-opt --version` prints a version
  - [ ] Export `PATH="$PWD/scripts/env/bin:$PATH"` and verify `which mlir-opt` points inside `scripts/env/bin/`
- [ ] **§0.3** Install Ollama + pull `codellama:34b-instruct-q4_K_M` + `granite-code:34b-instruct-q4_K_M`
- [ ] **§0.4** Create venv, install `scripts/env/requirements.txt`, `pip install -e .`
  - [ ] `pytest -x -q` — all non-docker/mlx tests green
  - [ ] Docker smoke test (echo a trivial module through `mlir-opt --verify-diagnostics`)
- [ ] **§0.5** Clone `llvm-project`, `iree`, `stablehlo` into `~/src/`

---

## §1 — Day 2 (2026-04-18, actual): tokenizer shootout + data miners

- [x] **§1.1** Tokenizer shootout — `scripts/tokenizer_shootout.py`
  - [x] `results/tokenizer_shootout.json` inspected
  - [x] **Locked**: primary = **Phi-3.5-mini** (tok/char 0.468, vocab 32k), secondary = **SmolLM2-1.7B** (tok/char 0.458, vocab 49k). Gemma-2-2b better compression (0.437) but 8× vocab → bigger output head; not worth it.
- [x] **§1.2** L0 ODS miner → `data/processed/l0_ods.jsonl` — **276 ops** (183 target + 93 extras)
  - [x] `grammar/lattices/{arith,func,linalg,memref}.json` populated (49/5/98/31 ops)
  - Note: ≥15k design target was across all LLVM/IREE/StableHLO dialects; 276 is correct for our 4-target + 4-extra scope.
- [x] **§1.3** L3 test miner → `data/processed/l3_tests.jsonl` — **8528 cases** from 2543 files (design target ≥5k ✓)
  - [x] Verifier-clean via `--require-verify`
  - [x] `// -----` splitting now separates multi-case test files correctly
- [-] **§1.4** L1 Polygeist pipeline — **DEFERRED to Day 7** (Polygeist CMake failed in container; not a blocker since L1 feeds LoRA appendix only)
  - [x] C corpus staged (7573 rows from `bigcode/the-stack-smol` at `data/raw/c_corpus.jsonl`)
- [x] **§1.5** Grammar round-trip on L3 — **95.2% on n=756 target-only samples (full subset), multi-seed stable**, 25ms/sample.
  - [x] ≥95% gate met
  - Iteration trajectory: v0=0.2% → v1=35.6% → v2=67.8% → v3=73.6% → v4=78.0% → v5=83.0% → v6=90.4% → v7=92.6% → v8=93.8% → v9=95.2%
  - Fixes applied across 9 iterations:
    - Top-level: `module { ... }` optional; sequence of ops; type/attr aliases at top (`!type = ...`, `#map0 = ...`)
    - Functions: merged `private_func_decl` into `named_func_op` with optional body; added `attributes { ... }` clause; unnamed positional arg form (`@foo(i16)`)
    - Arith: full binop list incl. `ceildiv*`, `floordiv*`, `max/min{i,u,f}`; `fastmath<>` flag; typed int attributes (`64 : i64`); accepted `constant` alias; multi-result binop (`mului_extended`); quoted cmp predicates; hex literals (`0xFF800000`); `true`/`false` as cmpf predicates
    - Types: arbitrary-bitwidth integers (`i4`, `i11`); low-precision floats (`f8E4M3FN` etc.); unranked memref (`*xf32`); scalable vector (`[N]x`); strided layout; tuple; complex; type aliases
    - Memref ops: 17 ops total — alloc, alloca, dealloc, load (opt result), store (opt result), dim, cast, subview (empty ranges), expand_shape, collapse_shape, reshape, reinterpret_cast, realloc, view, rank, prefetch, transpose (affine map or inline permutation), copy, get_global, global (quoted visibility), extract_aligned_pointer_as_index, memory_space_cast, extract_strided_metadata, dma_start, dma_wait, alloca_scope
    - SSA names: added `-`, `.`, `$` in first and subsequent chars; optional `:N` multi-result suffix; optional `#N` index
    - Blocks: `^bb0(%a: index):` label syntax inside regions
    - Generic op syntax: `"dialect.op"(operands) : func_type`
    - `call_indirect` form
    - Trailing `loc(...)` suffix on ops and top-level items
  - Remaining failures (~5%): custom dialect attributes (`#test<...>`), dense-attr-with-dict (`dense<1.0> {...}`), locality<N> prefetch flag, nested modules, null-byte test files — all extremely low-value edge cases.

---

## §2 — Day 3 (2026-04-18, compressed): C1 sanity + baselines + seeds

- [x] **§2.1** C1 sanity pass@1 on 500 L3 held-out (Phi-3.5-mini) — **96.0%** parse-valid, 1.4% verify-valid, 59 min. Result file: `results/day3/c1_sanity_500.jsonl`. **Gate exceeded.**
  - Week-1 gate (design §7): C1 parse-valid ≥95% → ✅ met at 96.0% on 500 samples.
  - Required fix to get here: built `grammar/mlir_gen.lark` with explicit WS productions + bounded identifiers (0% → 96%).
- [x] **§2.2** CodeLlama-34B free-decoding baseline — **0.8%** parse-valid (7/900), 0% verify, 183 min. Result: `results/day3/baseline_codellama_free.jsonl`.
  - CodeLlama was trained on pre-arith-dialect MLIR (`func @foo`) + emits markdown/prose preambles; fails grammar on almost every sample.
  - This is exactly the evidence the paper's thesis predicts. Small model + constraint beats large model free-decoding by ~120×.
- [x] **§2.3** MLIR-Spec-150 authoring — **15 seed examples authored (10% of 150)**
  - Filenames match id scheme ✓
  - `python -m eval.benchmarks.mlir_spec_150.validate` passes all ✓ (verified against mlir-opt via Docker)
  - Covers arith binops/cmp/select/cast/const/negf, memref alloc/dealloc/load/store, chains, index types, easy+medium difficulty
  - **Remaining 135 need hand-authoring Days 3-7** (target +25/day per original plan). Seed examples in `eval/benchmarks/mlir_spec_150/examples/001-015_*.json` serve as authoring templates.

---

## §3 — Day 4 (2026-04-19, actual): C2 + SLM smoke + more baselines

- [x] **§3.1** C2 integrated via type-domain grammar splits (`mlir_gen_c1c2.lark`)
- [x] **§3.2** SLM smoke n=200 × 6 cells complete. SmolLM2+C1 = 58% verify; Phi+C1 = 9.5%. C1 ≈ C1+C2 under few-shot. Results: `results/day4/slm_smoke.jsonl`.
- [~] **§3.3** Baselines (running in background)
  - [x] CodeLlama-34B free (Day-3 rerun at 0.8% verify)
  - [~] Granite-Code-34B free + C1 — granite-free n=200 complete; granite-c1 at ~n=90 (trending 32-48% verify); CodeLlama-c1 not yet started
- [x] **§3.4** MLIR-Spec-150 authoring at **50/150** ✓ (day target met)
- [x] **B2 ablation** (added mid-day): zero-shot SmolLM2 × {C1, C1+C2}. Both at 0% verify. Error categorization confirms "other" (cross-SSA) is the bottleneck → motivates ADR-0006 (C3 back in).

---

## §4 — Day 5 (2026-04-22, revised per ADR-0006): C3 prototype + measurement

- [x] **§4.0** Design C3 symbol-table tracker — see `decoder/c3_scope.py` docstring.
- [-] **§4.1** In-line logits-processor path **explored, not pursued**. Outlines 1.2 `Generator` wraps llguidance behind a single CFG mask with no hook to compose a secondary mask; reimplementing the whole sampling loop is ~1d for marginal gain. ADR-0006 explicitly accepts the rejection-sampling fallback.
- [x] **§4.2** Post-hoc rejection-sampling path: `decoder/c3_scope.py` + 5-retry loop in `scripts/day5_c3_smoke.py`. First try is greedy (temp=0) so equals C1+C2 exactly on accept; retries use temp=0.8 for diversity.
  - Validator: 19/19 unit tests green (`decoder/tests/test_c3_scope.py`).
  - Postmortem on Day-4 smoke (`scripts/day5_c3_postmortem.py`): on SmolLM2+C1 n=200, scope-pass∩verify-pass = 58.0%, scope-FAIL∩verify-pass = **0/200 false rejects**, scope-FAIL∩verify-FAIL = 31.0% — 62 of 84 verify-failures have SSA-scope/type violations the validator catches.
  - Predicted C3-gated verify-rate: P(verify|scope_pass) = 116/138 = 84.1%, so 5-try rejection sampling is expected to land in the 75–85% range on the same prompt set, a ~20–25pp lift over C1/C1+C2's 58%.
- [x] **§4.3** SmolLM2 × {free, C1, C1+C2, C1+C2+C3} × arith+func × n=200 — DONE (15 min wall-clock). Results: `results/day5/c3_smoke.jsonl`. Analysis: `results/day5/c3_summary.json`.
  - **Day-5 matrix**:
    | constraint   |  parse%               | verify%                | mean_att | mean_gen_s |
    |--------------|-----------------------|------------------------|---------:|-----------:|
    | none         | 38.0 [31.5, 44.5]     | 34.5 [28.0, 41.0]      |     1.00 |       1.19 |
    | C1           | 99.0 [97.5, 100.0]    | 54.5 [47.5, 61.5]      |     1.00 |       0.77 |
    | C1+C2        | 99.0 [97.5, 100.0]    | 54.5 [47.5, 61.5]      |     1.00 |       0.78 |
    | **C1+C2+C3** | 98.0 [96.0, 99.5]     | **67.5 [61.0, 74.0]**  |     1.83 |       1.65 |
  - **Paired Δverify (C1+C2+C3 − C1+C2)** = **+13.0pp** [+8.5, +18.0], p < 0.0001, n_pair=200.
  - **Revised Week-2 gate MET** (CI_low = +8.5pp > 0; Δ = 13pp ≥ 10pp target).
- [x] **§4.4** MLIR-Spec-150 authoring at **75/150** (target met). `python -m eval.benchmarks.mlir_spec_150.validate` → 0 errors.

---

## §5 — Day 6 (2026-04-20, actual): HCS replication + MLIR-Spec push + error-category refinement

- [x] **§5.1** HCS ablation — **reversal NOT detected** on either SmolLM2 or Phi.
  - SmolLM2: C1 − free = +20.0pp [+13.0, +27.0] p<.0001 (C1 monotonically helps).
  - Phi-3.5-mini: C1 − free = +7.5pp [+4.0, +11.5] p<.0001.
  - C1+C2 − C1 = 0.0pp for both; few-shot already delivers C2's verify lift (restates Day-4).
  - Paper framing: HCS is setup-conditional; our few-shot + C2-aware gen grammar pipeline does not suffer it. `results/day6/hcs_*.json`.
- [x] **§5.2** Error-category analysis refined with `type_ssa` split. Hero finding: **C3 collapses `type_ssa` 56 → 17 (−69.6%) on SmolLM2**. C2 collapses `type` 6 → 2. `arity`/`dialect_misuse`/`syntax` = 0 across all SmolLM2 cells. `results/day6/{error_categories,day6_summary}.json`.
- [x] **§5.3** MLIR-Spec-150 authoring at **130/150** (day target +55 met). Validator clean.

---

## §6 — Day 7 (2026-04-20, actual): Week-2 gate confirm + MLIR-Spec-150 lock + linalg decision

- [x] **§6.1** Week-2 gate numbers frozen at `results/frozen/week2_gate_day5/` with README + jsonl + summary + reproducer command. Δ = +13.0pp, CI = [+8.5, +18.0], MET.
- [x] **§6.2** **MLIR-Spec-150 LOCKED**: 150/150 seeds, all `python -m eval.benchmarks.mlir_spec_150.validate` clean, no duplicate ids/nl/mlir, all required keys present.
  - Distribution: 57 easy / 66 medium / 27 hard. 109 arith+func / 14 memref+func / 27 memref+arith+func.
  - Self-review caught 1 duplicate MLIR (`015_neg-float` vs `070_negate-float`); fixed by retargeting 070 to f64.
- [-] **§6.3** LoRA training deferred to Day 11 buffer per ADR-0007 (linalg displaces it).
- [x] **§6.4** **Decision: PURSUE linalg (named ops only)** — ADR-0007 accepted.
  - Scope: 12 ops (matmul, matvec, fill, copy, transpose, broadcast, add, sub, mul, div, exp, abs) under memref semantics.
  - Out of M1: `linalg.generic` + tensor semantics + complex convs → M2.
  - Execution: Days 8-10 (grammar → C3 → benchmark authoring → measurement → 30B baselines). ~2 extra days consumed; ~7 days buffer remain.

---

## §7 — Days 8–10 (revised per ADR-0007): linalg integration + matrix regeneration

Day 7 decision: **pursue** (ADR-0007).

- [ ] **§7.1 Day 8 AM**: extend `grammar/mlir_gen_{c1,c1c2}.lark` with 12 linalg named-op productions under memref semantics. Iterate against L3 linalg subset; target round-trip ≥90%.
- [ ] **§7.2 Day 8 PM**: extend `decoder/c3_scope.py` with handlers for each linalg op's ins()/outs() typing rules. Add ≥12 unit tests. All 31 tests (19 existing + 12 new) pass.
- [ ] **§7.3 Day 9 AM**: author **Linalg-Spec-30** hand-authored benchmark (`eval/benchmarks/linalg_spec_30/`). 30 NL→MLIR pairs, 12 ops covered, all verify clean.
- [ ] **§7.4 Day 9 PM**: measurement. SmolLM2 × {none, C1, C1+C2, C1+C2+C3} × linalg × Linalg-Spec-30 + L3 linalg-subset held-out (n=500). Results → `results/day9/linalg_smoke.jsonl`.
- [ ] **§7.5 Day 10 AM**: 30B baselines on linalg. CodeLlama-34B + C1, Granite-Code-34B + C1, n=200. Results → `results/day10/linalg_baselines.jsonl`.
- [ ] **§7.6 Day 10 PM**: error categorization on linalg results. Regenerate design §5.2 matrix. Re-confirm Week-3 gate across two dialects.

---

## §8 — Day 10 (2026-04-27): Week-3 gate + all figures

- [ ] **§8** **GATE — Week 3**: SLM + C1+C2 within 3pp of 30B + C1 on ≥1 dialect (paired bootstrap).
  - [ ] Dialect that satisfied the gate: _____
  - [ ] Measured Δ: _____ pp. CI: [___, ___].
  - [ ] If gate fails: degrade to workshop-only submission, update ADR, continue to Day 11 (paper work still progresses).
- [ ] **§8** `scripts/make_figures.py` regenerated all figures + LaTeX tables under `docs/paper/figures/`

---

## §9 — Days 11–17: buffer + paper + reproducibility

- [ ] **Day 11 (04-28)** Buffer: re-run any cell with 95% CI half-width > 0.04. Freeze numbers in `results/frozen/`.
- [ ] **Day 12 (04-29)** Paper: abstract + intro + method sections in `docs/paper/main.tex`
- [ ] **Day 13 (04-30)** Paper: results + ablation + mechanism (entropy) sections
- [ ] **Day 14 (05-01)** Paper: related work + limitations + explicit future-work list (C3, GRPO, cross-target, `tosa`)
- [ ] **Day 15 (05-02)** Internal review + ≥1 external reader
- [ ] **Day 16 (05-03)** Revisions + reproducibility package (`submission_artifact.tar.gz` per RUNBOOK §9)
  - [ ] Include `Dockerfile.reproduce` with llama.cpp CPU fallback for non-macOS reviewers
- [ ] **Day 17 (05-04)** Final polish — **submission-ready**

---

## Stretch / post-submission

Not in the M1 submission. Listed so nothing falls out of sight.

- [ ] Switch LARK from Earley → LALR once grammar is stable (speedup for C1 mask construction)
- [ ] Finalize `decoder.c1_cfg.mlx_generate` logits-processor integration once a stable mlx-lm API lands
- [ ] Replace `decoder.c2_type_arity.apply_type_arity_mask` stub with full in-loop logits masking
- [ ] Extend `grammar/mlir.lark` to cover `linalg.generic` indexing-map syntax (currently only op shell)
- [ ] Include `tosa` as a 4th dialect if Week-3 gate passed comfortably
- [ ] Write `docs/paper/figures/` PDF caption-autogen script so captions stay in sync with `results/*.json`

---

## Risks to watch (from `docs/execution_plan.md §5`)

Mark `[!]` if a risk actually materializes.

- [ ] LARK v0 grammar needs >2 days of iteration
- [ ] Day-1 code-complete slipped into Day 2 (low-severity: buffer exists)
- [ ] L1 Polygeist throughput < 50k/day → fall back to `clang -emit-llvm + mlir-translate`
- [ ] 30B baseline wall-clock worse than estimated → drop to n=200 + widen CI reporting
- [ ] MLIR-Spec-150 authoring slipping behind daily targets
- [ ] `linalg` C2 lattice too complex by Day 8 → freeze at arith+func

---

## §10 — Main-track extension (Days 18–38, per ADR-0008)

**Scope**: see `docs/decisions/ADR-0008-main-track-scope-expansion.md`.

### Per-day Definition of Done (applies to EVERY day below)

Every entry in §10 is not complete until all four:

1. Result artifacts written under `results/dayNN/` (or appropriate frozen path)
2. `docs/paper/` figures / tables regenerated if numbers changed
3. `docs/daily_log/dayNN.md` entry written using the template in
   `docs/daily_log/README.md`
4. `docs/daily_log/README.md` index updated with a link to the new entry

Skipping (3) or (4) blocks forward progress — paper-narrative material must
not be reconstructed from memory later.

### Phase A — low-cost high-ROI fills (Days 18–21)

- [ ] **Day 18** (2026-04-25): 30B + C3 rejection sampling on both dialects.
  - `results/day18/{30b_c3_arith,30b_c3_linalg}.jsonl`
  - Update `results/frozen/day10_final_matrix/` if headline changes
  - `docs/daily_log/day18.md` with paired deltas vs C1-only baselines
- [ ] **Day 19** (2026-04-25 end): StarCoder2-15B via Ollama, free + C1 on both dialects.
  - `results/day19/starcoder2_baselines.jsonl`
  - Updated §5.2 matrix with new row
  - `docs/daily_log/day19.md`
- [ ] **Day 20** (2026-04-26): novelty reframe in `main.tex` (abstract, intro, contributions, related work).
  - `docs/paper/main.tex` diff committed
  - `docs/daily_log/day20.md` noting before/after framing
- [ ] **Day 21** (2026-04-26 end): integrate Phase-A results.
  - Regenerate `fig1_main_matrix.pdf`, `fig4_efficiency_frontier.pdf` with new cells
  - Update `docs/paper/figures/FIGURES.md`
  - `docs/daily_log/day21.md` with Phase-A summary table + gate-evaluation
  - ◆ **Gate**: if 30B + C3 flips headline, halt and reassess

### Phase B — Tier 1 in-line C3 (Days 22–28)

- [ ] **Day 22**: LARK → token-automaton compilation bypassing Outlines.
  - New module under `decoder/` (e.g. `decoder/in_line_grammar.py`)
  - Unit tests: token-level acceptance equivalent to Earley parse
  - `docs/daily_log/day22.md`
- [ ] **Day 23**: joint `(parser_state, symbol_table)` state machine.
  - `decoder/c3_inline.py` with reduction hooks + scope lifecycle
  - `docs/daily_log/day23.md`
- [ ] **Day 24**: BPE-aware in-scope name trie.
  - Tokenizer-specific trie construction + per-step mask intersection
  - Edge-case tests on SmolLM2 tokenizer (multi-token SSA names)
  - `docs/daily_log/day24.md`
- [ ] **Day 25**: MLX sampling-loop integration + unit tests.
  - Custom `mlx_generate_c3_inline(prompt, …)` replacing Outlines Generator
  - `docs/daily_log/day25.md`
- [ ] **Day 26**: equivalence test in-line vs rejection-sampled C3.
  - `results/day26/c3_inline_vs_rejection.jsonl`, n=200 paired
  - ◆ **Gate**: if in-line significantly lower than rejection, debug
  - `docs/daily_log/day26.md`
- [ ] **Day 27**: full SmolLM2 × in-line C3 matrix (arith+func + linalg).
  - `results/day27/in_line_c3_matrix.jsonl`
  - Expected: attempts → 1.0, ~2× speedup, tighter CIs
  - `docs/daily_log/day27.md`
- [ ] **Day 28**: Tier 2 — soundness + coverage theorems appendix.
  - `docs/paper/appendix_soundness.tex`
  - `docs/daily_log/day28.md`

### Phase C — strong additions (Days 29–35)

- [ ] **Day 29** (AM): StableHLO grammar extension for 10 named ops.
- [ ] **Day 29** (PM) / **Day 30**: C3 scope handlers for StableHLO ops (tensor semantics).
  - `decoder/c3_scope.py` extension + tests
  - `docs/daily_log/day29.md`, `day30.md`
- [ ] **Day 31**: author **StableHLO-Spec-30** benchmark.
  - `eval/benchmarks/stablehlo_spec_30/` with 30 verify-clean pairs
  - `docs/daily_log/day31.md`
- [ ] **Day 32**: SmolLM2 × 4 constraint cells × StableHLO matrix.
  - `results/day32/stablehlo_smoke.jsonl`
  - `docs/daily_log/day32.md`
- [ ] **Day 33**: 30B + StarCoder2 baselines on StableHLO.
  - `results/day33/stablehlo_baselines.jsonl`
  - `docs/daily_log/day33.md`
- [ ] **Day 34**: seeds 0,1,2 × n=100 on 5 critical cells for multi-seed CIs.
  - `results/day34/multiseed.jsonl`
  - `docs/daily_log/day34.md`
- [ ] **Day 35**: functional-equivalence spot check on 20 arith+func samples.
  - `results/day35/func_equiv.jsonl`
  - `docs/daily_log/day35.md`

### Phase D — paper + submission (Days 36–38)

- [ ] **Day 36**: rewrite §Method + §Results + regenerate all figures.
  - All fig{1..8}_*.pdf + StableHLO cross-IR figure (fig9?) regenerated
  - `docs/daily_log/day36.md`
- [ ] **Day 37**: §Cross-IR-generalization + §Functional-equivalence appendix + theorem integration.
  - `docs/daily_log/day37.md`
- [ ] **Day 38**: rebuild `submission_artifact.tar.gz`, final review, pin SHA256.
  - `docs/daily_log/day38.md` with main-track final state
  - `results/frozen/day38_main_track/` with all pinned numbers

### Main-track decision gates

| Gate | Day | Criterion | Action if fails |
|------|-----|-----------|-----------------|
| Phase A | 21 | 30B + C3 does not flip headline | Halt, reassess paper framing |
| Phase B | 26 | in-line C3 verify ≥ rejection-sampled C3 verify | Debug in-line implementation |
| Phase C | 33 | StableHLO matrix reproduces arith/linalg pattern | Downgrade StableHLO to appendix |
| Submission | 38 | All gates passed; paper integrates all new results | Fall back to Phase-A-enhanced D&B submission |

---

## §11 — Main-track REVISION for reviewer credibility (Days 39–49, post-Day-38 critique)

**Motivation**: Sharp reviewer-style critique after Day 38 flagged 3
main-track showstoppers:

1. **Abstract/data tension**: abstract claimed "ties 15B on arith+func"
   but multi-seed mean shows SmolLM2 trails StarCoder2 by −20pp on
   arith+func. Abstract and Limitations now tell contradictory stories.
2. **Tier-1 novelty is code + theorem, not running algorithm**: the
   in-line coupled decoder's MLX integration has a known race condition
   and never produced valid output on a real model.
3. **StableHLO methodology gaps**: different verify tool
   (iree-compile vs mlir-opt), author-written grammar + benchmark,
   30B baselines via parse-rejection not CFG mask — re-introduces the
   constraint-asymmetry problem we fixed on MLIR.

**Strategy**: honest reframe + fix the fixable. Lead with the two clean
wins (linalg, StableHLO); acknowledge arith+func mean trails modern
mid-size; fix the MLX race; test on held-out StableHLO; install
stablehlo-opt.

### Phase E — honest-paper revision (Days 39–49)

Per-day DoD same as §10 Phase A-D: (1) results artifacts, (2) paper
updates, (3) `docs/daily_log/dayNN.md`, (4) README index.

**P1 — must-do (Days 39–41)**:

- [~] **Day 39 AM**: Fix Day-26 MLX terminal-closure race. Defer
  mask computation until after close-terminals runs. ~2-3 hr.
  Task #80.
- [ ] **Day 39 PM**: Launch full-n multi-seed (n=200/125, seeds 1,2) on
  5 critical cells in background. Task #81.
- [ ] **Day 40**: In-line vs rejection equivalence test on fixed MLX.
  Measure paired verify + wall-clock speedup. Task #82.
- [ ] **Day 41**: Pull IREE held-out StableHLO corpus; filter to our
  10-op scope. Target ~100 held-out prompts. Task #83.

**P2 — strongly recommended (Days 42–43)**:

- [ ] **Day 42**: Re-run StableHLO matrix on held-out corpus. Task #84.
- [ ] **Day 43**: Build `stablehlo-opt` (or scope-downgrade). Task #85.

**P3 — polish (Days 44–47)**:

- [ ] **Day 44**: Rewrite abstract + §Results for multi-seed honesty.
  Task #86.
- [ ] **Day 45 AM**: Finalize `refs.bib` — no placeholders. Task #87.
- [ ] **Day 45 PM**: Add zero-false-rejects caveat + Thm 2 BPE-boundary
  refinement. Task #88.
- [ ] **Day 46**: Regenerate all figures with full-n multi-seed + held-
  out StableHLO data. Task #89.
- [ ] **Day 47**: External reader pass (someone outside the loop).
  Task #90.

**P4 — submission prep (Days 48–49)**:

- [ ] **Day 48**: Copy-edit + formatting pass. Task #91.
- [ ] **Day 49**: Rebuild tarball + pin new SHA256 + submit. Task #92.

### Revision gates

| Gate | Day | Criterion | Action if fails |
|------|-----|-----------|-----------------|
| Race fix | 39 | MLX in-line decoder produces parse-valid output | Revert to theorem-only claim in paper |
| Equivalence | 40 | in-line ≥ rejection verify (paired, CI_low ≥ 0) | Debug (should not happen — theorem) |
| Full-n multi-seed | 41 | seeds 1,2 arith CI matches Day 34 first-100 trend | Widen paper's reported CIs |
| Held-out StableHLO | 42 | Pattern replicates (SmolLM2 beats baselines) | Downgrade StableHLO to "grammar-construction transfer" |
| stablehlo-opt | 43 | build succeeds; numbers match iree-compile | Explicitly scope StableHLO claim |
| Abstract rewrite | 44 | abstract and Limitations tell same story | Iterate |
| External review | 47 | no killshot objections | Address + extend by 2 days |
| Submission | 49 | paper + tarball submission-ready | — |

**Expected outcome**: credible main-track submission with 40-55%
acceptance odds (up from borderline-reject of the current Day-38
state).
