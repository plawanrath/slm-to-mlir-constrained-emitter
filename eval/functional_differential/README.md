# Gold-differential functional evaluation (all 180 spec prompts)

Verify-valid is a structural metric: a program can pass `mlir-opt --verify-diagnostics`
and still compute the wrong function. This component measures functional correctness at
benchmark scale **without changing the released data**. Every MLIR-Spec-150 and
Linalg-Spec-30 example already ships a hand-authored gold module (`mlir` field), so each
generation is executed against its gold on the same randomized inputs.

It extends functional coverage from the 30 hand-authored references in `eval/functional/`
to all 180 spec prompts (150 arith+func, 30 linalg).

## Protocol

Per spec prompt, K = 5 seeded trials. Trial *t* sets every dynamic dimension to 3 + *t*;
static dimensions come from the gold signature. Inputs are seeded and identical for the gold
and the candidate: memref data, in-bounds index scalars, nonzero small integers, and
quarter-step floats.

1. **Gold gate.** A driver is built around the gold function, lowered (linalg to loops,
   scf to cf, math/arith/cf/memref/func to LLVM), and executed **twice** under
   `mlir-cpu-runner` for every trial. The two runs must succeed with exactly equal outputs on
   all 5 trials. A prompt whose gold fails the gate is excluded and its reason is recorded, so
   no generation is scored against an unvalidated reference. Prompts with no observable output
   (void to void) or an unprintable memref dtype are excluded as out of scope.
2. **Differential.** For a verify-valid generation whose signature is compatible with the gold's,
   the same driver bytes are run around the generation. Every memref argument's final state and
   any scalar result are compared with the gold's (integers exact; floats rel 1e-4 / abs 1e-5).
   A generation is **functionally correct** on a prompt iff all 5 trials execute and match.
3. **Signature compatibility.** Scalar argument types and the result type must be identical.
   Memref arguments must match in rank and dtype, and each dimension must equal the gold's or
   be dynamic (`?`). Static-to-dynamic generalization is accepted and flagged `dim_relaxed`.
   Anything else is `signature_mismatch`: interface non-conformance that the textual verifier
   cannot see (dropped output arguments, wrong rank or element type).

## Results on the frozen seed-0 SmolLM2-1.7B C1+C2+C3 generations

(`results/day51_seed0_n200/multiseed_seed0.jsonl`, the generations behind the paper's
verify-valid numbers; no new inference.)

| | arith+func | linalg |
|---|---:|---:|
| gold modules | 150 | 30 |
| pass the gold gate | 143 | 30 |
| verify-valid generations (gate-passed prompts) | 68 | 22 |
| signature-compatible (scored) | 60 | 9 |
| match the gold on all 5 trials | 43 (71.7%, CI95 [60.0, 83.3]) | 8 |
| **correct under the canonical interface** (non-conformers count as failures) | **43/68 = 63.2%** | **8/22 = 36.4%** |
| non-conforming interfaces | 8 | 13 |

All non-matching scored generations lower and execute: their failures are semantic.
Gate exclusions (arith): 4 no observable output, 1 unprintable memref dtype, 2 gate
failures (one nondeterministic, one runner crash).

Interface non-conformance dominates the linalg gap. On the n=10 linalg references in
`eval/functional/`, re-executing the same generations under wrappers adapted to each
generation's declared signature moves the cascade from 8/10 verify, 4/10 lower, 3/10 exec,
2/10 match to 8/10, 8/10, 8/10, 7/10 (`scripts/day58_e12b_adapted_wrappers.py`). That
diagnosis rests on that small experiment.

## Files

| Path | Content |
|---|---|
| `scripts/day56_e12c_differential_frozen.py` | driver: gold gate + differential trials |
| `scripts/day61_functional_per_prompt.py` | rebuilds the 180-row per-prompt table from the driver outputs and checks it against the summary |
| `scripts/day57_e12a_randomized_refs.py` | the same oracle applied to the 30 hand-authored references (5 draws each, 150 trials) |
| `results/day56/e12c_gold_gate.jsonl` | one row per gold prompt (gate outcome, reason, signature) |
| `results/day56/e12c_differential.jsonl` | one row per scored candidate trial |
| `results/day56/e12c_summary.json` | per-dialect counts, rates, bootstrap CIs |
| `results/day61/functional_per_prompt_180.jsonl` | **per-prompt results for all 180 spec prompts** (gate, candidate status, trials matched, first divergence) |

## Running

```bash
docker compose -f scripts/env/docker-compose.yml up -d     # pinned LLVM 19.1.7 container
python -m eval.functional_differential                      # driver, then the per-prompt table
# or step by step:
python scripts/day56_e12c_differential_frozen.py --out-dir results/day56
python scripts/day61_functional_per_prompt.py
```

To score other generations, point `CANDIDATES_JSONL` in the driver at a JSONL with
`model`, `dialect`, `seed`, `prompt_id`, `nl`, `generated`, `verify_valid` rows.
`SLM_MLIR_CONTAINER` selects a different container name; any container used must reproduce
the pinned verifier's output (see `scripts/day61_frozen_c1only_baselines.py::verifier_check`).
