"""Validator for MLIR-Spec-150 entries.

Checks:
  - JSON schema (id, dialect, difficulty, nl, mlir, notes).
  - Filename matches id.
  - `mlir` passes `mlir-opt --verify-diagnostics` (requires Docker container).
  - `dialect` claim matches mnemonics appearing in the MLIR.

Usage:
    python -m eval.benchmarks.mlir_spec_150.validate
    python -m eval.benchmarks.mlir_spec_150.validate --examples some/path --skip-verify
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REQUIRED_FIELDS = ("id", "dialect", "difficulty", "nl", "mlir", "notes")
DIFFICULTIES = {"easy", "medium", "hard"}
KNOWN_DIALECTS = {"arith", "func", "linalg", "memref"}


@dataclass
class ValidationError:
    path: Path
    field: str
    detail: str


def _validate_one(path: Path, skip_verify: bool = False) -> list[ValidationError]:
    errs: list[ValidationError] = []
    try:
        rec = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        return [ValidationError(path, "json", str(e))]

    for f in REQUIRED_FIELDS:
        if f not in rec:
            errs.append(ValidationError(path, f, "missing field"))
    if errs:
        return errs

    if rec["difficulty"] not in DIFFICULTIES:
        errs.append(ValidationError(path, "difficulty", f"not in {DIFFICULTIES}"))

    # Filename must match id.
    expected_stem = rec["id"]
    if path.stem != expected_stem:
        errs.append(ValidationError(path, "id", f"filename stem {path.stem} != id {expected_stem}"))

    # Dialect claim consistency.
    claimed = set(d.strip() for d in rec["dialect"].split("+"))
    mentioned = set(re.findall(r"([a-z_][a-z_0-9]*)\.[a-z_]", rec["mlir"]))
    unknown_claimed = claimed - KNOWN_DIALECTS - {"func"}
    if unknown_claimed:
        errs.append(ValidationError(path, "dialect", f"unknown dialects: {unknown_claimed}"))
    if not (claimed & mentioned):
        errs.append(ValidationError(
            path, "dialect",
            f"none of claimed dialects {claimed} appear in MLIR (saw {mentioned})",
        ))

    if not skip_verify:
        from scripts.env.verify_cache import verify
        v = verify(rec["mlir"])
        if v["returncode"] != 0:
            errs.append(ValidationError(
                path, "mlir", f"mlir-opt --verify failed: {v['stderr'][:300]}",
            ))

    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--examples",
        type=Path,
        default=Path(__file__).resolve().parent / "examples",
    )
    ap.add_argument("--skip-verify", action="store_true",
                    help="Skip the mlir-opt verification step (useful if Docker is offline).")
    args = ap.parse_args()

    files = sorted(args.examples.glob("*.json"))
    if not files:
        print(f"[validate] no example JSONs under {args.examples}", file=sys.stderr)
        return 0

    total_errs = 0
    for path in files:
        errs = _validate_one(path, skip_verify=args.skip_verify)
        for e in errs:
            print(f"[validate] {e.path.name}: {e.field}: {e.detail}", file=sys.stderr)
        total_errs += len(errs)

    print(f"[validate] {len(files)} files, {total_errs} errors", file=sys.stderr)
    return 1 if total_errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
