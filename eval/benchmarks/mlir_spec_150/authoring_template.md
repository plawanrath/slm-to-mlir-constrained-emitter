# MLIR-Spec-150 — Authoring Guide

This is the hand-authored gold benchmark, **n = 150**, 2-person review.
Locked Day 7 (2026-04-24).

## Authoring rules

1. **One JSON file per pair**, under `examples/`. Filename: `NNN_short-slug.json` (e.g. `001_add-two-ints.json`).
2. **Every pair must verify** under `mlir-opt --verify-diagnostics`. Use `python -m eval.benchmarks.mlir_spec_150.validate` before committing.
3. **Distribution target**: 60 `arith`, 30 `func`, 60 `linalg`. Adjust if `linalg` is deferred.
4. **Difficulty spread**: aim for 40% easy (single-op), 40% medium (2–5 ops), 20% hard (multi-region or compound types).
5. **Avoid trivial paraphrases** — each NL should capture a meaningfully different task.

## JSON schema

```json
{
  "id": "001_add-two-ints",
  "dialect": "arith+func",
  "difficulty": "easy",
  "nl": "Write a function `add` that takes two i32 values and returns their sum.",
  "mlir": "module { func.func @add(%a: i32, %b: i32) -> i32 { %0 = arith.addi %a, %b : i32 ; return %0 : i32 } }",
  "notes": "tests arith.addi + func.func + return; trivial."
}
```

## Review process

- **Author** writes the pair and runs validate.py.
- **Reviewer** (second author) checks: does the NL unambiguously specify the MLIR? Does the MLIR verify? Is the difficulty label honest?
- **Both sign off** by adding their initials to `notes`: `"notes": "... | reviewed: AB, CD"`.

## Progress tracker

| Date | Target | Actual |
|---|---|---|
| Day 3 (04-20) | 25 |  |
| Day 4 (04-21) | 50 |  |
| Day 5 (04-22) | 100 |  |
| Day 6 (04-23) | 130 |  |
| Day 7 (04-24) | **150 (locked)** |  |
