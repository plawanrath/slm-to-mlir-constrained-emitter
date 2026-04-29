"""Functional-correctness mini-benchmark (n=30).

Closes the largest gap in §10 of the paper: verify-valid is structural,
not functional. The 19/20 mlir-translate lowerability sample didn't move
the needle because lowerability is still structural — it doesn't catch
"wrong algorithm". This sub-package adds 30 hand-authored
(input, expected_output) tuples spanning arith+func / linalg+memref /
stablehlo, lowers each candidate generation through the dialect-specific
LLVM/IREE pipeline, executes it, and compares stdout to the reference.

Constraint (datasheet): these 30 are evaluation evidence, NOT a benchmark
release. They live under `eval/functional/`, not `eval/benchmarks/`.
"""
