#!/usr/bin/env bash
# Day-16 reproducibility package builder. Run from repo root.
#
# Produces `submission_artifact.tar.gz` containing:
#   - frozen/ (pinned numerical results from Day 5 + Day 10)
#   - grammar/ (LARK grammars + ODS lattices)
#   - decoder/ (c1_cfg, c2_type_arity, c3_scope + tests)
#   - eval/ (benchmarks, stats, error categories)
#   - scripts/ (reproducer scripts, Dockerfile.reproduce)
#   - docs/paper/ (figures, LaTeX)
#   - ADRs, daily log
#   - README.md, RUNBOOK.md, TODO.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-submission_artifact}"
TARBALL="${TARBALL:-submission_artifact.tar.gz}"

rm -rf "$OUT_DIR" "$TARBALL"
mkdir -p "$OUT_DIR"

# Frozen results (the paper's pinned numbers)
cp -r results/frozen "$OUT_DIR/frozen"

# Per-prompt sources of the paper's tables and the response-period evidence
mkdir -p "$OUT_DIR/results"
for d in day4 day10 day18 day19 day32 day33 day39pm day50 day51 day51_seed0_n200 day52 \
         day53 day54 day55 day56 day57 day58 day59 day60 day61 day_f6 day_f8; do
    [ -d "results/$d" ] && cp -r "results/$d" "$OUT_DIR/results/$d"
done

# Grammar + lattices
cp -r grammar "$OUT_DIR/grammar"
# Strip __pycache__
find "$OUT_DIR/grammar" -type d -name __pycache__ -exec rm -rf {} +

# Decoder + tests
cp -r decoder "$OUT_DIR/decoder"
find "$OUT_DIR/decoder" -type d -name __pycache__ -exec rm -rf {} +

# Eval harness + benchmarks
cp -r eval "$OUT_DIR/eval"
find "$OUT_DIR/eval" -type d -name __pycache__ -exec rm -rf {} +

# Scripts
cp -r scripts "$OUT_DIR/scripts"
find "$OUT_DIR/scripts" -type d -name __pycache__ -exec rm -rf {} +

# Paper + figures
cp -r docs/paper "$OUT_DIR/paper"

# ADRs + daily log
cp -r docs/decisions "$OUT_DIR/ADRs"
cp -r docs/daily_log "$OUT_DIR/daily_log"
cp docs/design.md       "$OUT_DIR/design.md"
cp docs/execution_plan.md "$OUT_DIR/execution_plan.md"

# Top-level
cp README.md RUNBOOK.md TODO.md "$OUT_DIR/"

# Dockerfile.reproduce pinned LLVM/mlir-opt + CPU-only llama.cpp for non-macOS users
cp scripts/env/Dockerfile.llvm "$OUT_DIR/Dockerfile.reproduce" 2>/dev/null || true

# Hash manifest for deterministic verification
find "$OUT_DIR" -type f -exec sha256sum {} + | LC_ALL=C sort > "$OUT_DIR/SHA256SUMS"

# Build tarball
tar -czf "$TARBALL" "$OUT_DIR"
echo "Wrote $TARBALL ($(du -h "$TARBALL" | cut -f1))"
echo "SHA256: $(shasum -a 256 "$TARBALL" | cut -d' ' -f1)"
