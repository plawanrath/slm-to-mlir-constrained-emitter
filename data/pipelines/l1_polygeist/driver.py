"""L1 pipeline — CodeSearchNet C subset → MLIR via Polygeist, with clang fallback.

For each C snippet:
  1. Try `polygeist -emit-mlir -O0` (primary).
  2. Fall back to `clang -emit-llvm -O0 -S` + `mlir-translate --import-llvm`.
  3. Verify the result with `mlir-opt --verify-diagnostics`.
  4. Keep only verified (docstring, mlir) pairs.

All toolchain calls go through the Docker wrappers in scripts/env/bin/.

Usage:
    python -m data.pipelines.l1_polygeist.driver \
        --input data/raw/codesearchnet_c.jsonl \
        --out data/processed/l1_polygeist.jsonl \
        --max-pairs 50000 \
        --workers 8
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scripts.env.verify_cache import VerifyCache, verify

REPO_ROOT = Path(__file__).resolve().parents[3]
BIN = REPO_ROOT / "scripts" / "env" / "bin"


@dataclass
class L1Input:
    docstring: str
    code: str
    source_url: str = ""


def _polygeist_emit(code: str, timeout: float = 15.0) -> str | None:
    with tempfile.NamedTemporaryFile("w", suffix=".c", dir=REPO_ROOT, delete=False) as f:
        f.write(code)
        src = Path(f.name)
    try:
        proc = subprocess.run(
            [str(BIN / "polygeist"), str(src.relative_to(REPO_ROOT)),
             "-S", "-emit-mlir", "-O0"],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode == 0 and "module" in proc.stdout:
            return proc.stdout
    except subprocess.TimeoutExpired:
        return None
    finally:
        src.unlink(missing_ok=True)
    return None


def _clang_fallback(code: str, timeout: float = 15.0) -> str | None:
    """clang -emit-llvm → mlir-translate --import-llvm."""
    with tempfile.NamedTemporaryFile("w", suffix=".c", dir=REPO_ROOT, delete=False) as f:
        f.write(code)
        src = Path(f.name)
    try:
        ll = subprocess.run(
            [str(BIN / "clang"), "-emit-llvm", "-O0", "-S",
             str(src.relative_to(REPO_ROOT)), "-o", "-"],
            capture_output=True, text=True, timeout=timeout,
        )
        if ll.returncode != 0:
            return None
        translate = subprocess.run(
            [str(BIN / "mlir-translate"), "--import-llvm"],
            input=ll.stdout, capture_output=True, text=True, timeout=timeout,
        )
        if translate.returncode == 0 and "module" in translate.stdout:
            return translate.stdout
    except subprocess.TimeoutExpired:
        return None
    finally:
        src.unlink(missing_ok=True)
    return None


def lower_one(inp: L1Input, cache: VerifyCache | None = None) -> dict | None:
    mlir = _polygeist_emit(inp.code)
    backend = "polygeist"
    if mlir is None:
        mlir = _clang_fallback(inp.code)
        backend = "clang+mlir-translate"
    if mlir is None:
        return None
    v = verify(mlir, cache=cache)
    if v["returncode"] != 0:
        return None
    return {
        "source": "l1_polygeist",
        "docstring": inp.docstring,
        "code_c": inp.code,
        "mlir": mlir,
        "lower_backend": backend,
        "source_url": inp.source_url,
    }


def stream_inputs(path: Path) -> Iterable[L1Input]:
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            yield L1Input(
                docstring=rec.get("docstring", "") or rec.get("comment", ""),
                code=rec["code"],
                source_url=rec.get("source_url", ""),
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="JSONL file with {code, docstring, source_url}.")
    ap.add_argument("--out", type=Path, default=Path("data/processed/l1_polygeist.jsonl"))
    ap.add_argument("--max-pairs", type=int, default=50_000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache = VerifyCache()
    n_kept = 0
    n_tried = 0

    with args.out.open("w") as f, cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for inp in stream_inputs(args.input):
            if n_kept >= args.max_pairs:
                break
            n_tried += 1
            futures.append(pool.submit(lower_one, inp, cache))
            if len(futures) >= 4 * args.workers:
                for fut in cf.as_completed(futures):
                    r = fut.result()
                    if r is not None:
                        f.write(json.dumps(r) + "\n")
                        n_kept += 1
                futures = []
            if n_tried % 500 == 0:
                print(f"[l1] tried={n_tried} kept={n_kept}", file=sys.stderr)
        for fut in cf.as_completed(futures):
            r = fut.result()
            if r is not None and n_kept < args.max_pairs:
                f.write(json.dumps(r) + "\n")
                n_kept += 1

    print(f"[l1] done: tried={n_tried} kept={n_kept} → {args.out}", file=sys.stderr)
    cache.close()


if __name__ == "__main__":
    main()
