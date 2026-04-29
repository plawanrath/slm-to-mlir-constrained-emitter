"""Phase F F8: expand the held-out sweep to n≥150 for tighter CIs.

F6's held-out-50 has CI half-width ≈±14pp at p=0.5, too wide to
claim replication within ±5pp. We widen the sweep (more shape × dtype
combinations, larger dtype[:4] cap) to push kept ≥ 150, verify each
with iree-compile, and write to
eval/benchmarks/stablehlo_held_out_200/.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
os.chdir(REPO); sys.path.insert(0, REPO)

from scripts.env.verify_stablehlo import verify_stablehlo

import scripts.day41_held_out_stablehlo as d41

OUT = Path("eval/benchmarks/stablehlo_held_out_200/examples")
OUT.mkdir(parents=True, exist_ok=True)

# Widen the sweep: 8 ew_bin ops × more dtypes × more shapes
# plus fuller ew_un × reshape × transpose × multi-op coverage.


def _ew_bin_wide() -> list[d41.Candidate]:
    out = []
    i = 0
    EW_BIN = d41.EW_BIN
    # use all 6 dtypes where the op supports them; float-only ops
    # keep the 3 float dtypes; else all 6
    for op in EW_BIN:
        safe = ["f32", "f16", "f64"] if op in ("divide", "power") else d41.DTYPES
        shapes = [8, 16, 32, 64, (4, 4), (8, 8), (8, 16), (4, 4, 4), (2, 8, 8)]
        for dtype in safe:
            for shape in shapes:
                i += 1
                t = d41._shape_str(shape).format(dtype=dtype)
                shape_desc = f"{shape}" if isinstance(shape, int) else "x".join(str(s) for s in shape)
                nl = (f"Write a function that applies stablehlo.{op} elementwise "
                      f"to two {dtype} tensors of shape {shape_desc}.")
                mlir = (
                    "module {\n"
                    f"  func.func @f(%a: {t}, %b: {t}) -> {t} {{\n"
                    f"    %0 = stablehlo.{op} %a, %b : {t}\n"
                    f"    return %0 : {t}\n"
                    "  }\n"
                    "}"
                )
                out.append(d41.Candidate(
                    id=f"ew_bin_{i:03d}_{op}_{dtype}_{shape_desc}",
                    nl=nl, mlir=mlir, notes=f"{op} {dtype} {shape_desc}"))
    return out


def _ew_un_wide() -> list[d41.Candidate]:
    out = []
    i = 0
    for op in d41.EW_UN:
        for dtype in ["f32", "f16", "f64"]:
            for shape in [16, 32, 64, (4, 8), (8, 16), (4, 4, 4)]:
                i += 1
                t = d41._shape_str(shape).format(dtype=dtype)
                shape_desc = f"{shape}" if isinstance(shape, int) else "x".join(str(s) for s in shape)
                nl = (f"Write a function that applies stablehlo.{op} "
                      f"elementwise to a {dtype} tensor of shape {shape_desc}.")
                mlir = (
                    "module {\n"
                    f"  func.func @f(%a: {t}) -> {t} {{\n"
                    f"    %0 = stablehlo.{op} %a : {t}\n"
                    f"    return %0 : {t}\n"
                    "  }\n"
                    "}"
                )
                out.append(d41.Candidate(
                    id=f"ew_un_{i:03d}_{op}_{dtype}_{shape_desc}",
                    nl=nl, mlir=mlir, notes=f"{op} {dtype} {shape_desc}"))
    return out


def build(target_n: int = 200) -> None:
    candidates: list[d41.Candidate] = []
    candidates += _ew_bin_wide()
    candidates += _ew_un_wide()
    candidates += d41._reshape_candidates()
    candidates += d41._transpose_candidates()
    candidates += d41._multi_op_candidates()
    print(f"[f8] {len(candidates)} candidates (raw)", file=sys.stderr)

    kept = []
    for c in candidates:
        r = verify_stablehlo(c.mlir)
        if r["returncode"] == 0:
            kept.append(c)
        if len(kept) >= target_n:
            break
    print(f"[f8] kept {len(kept)} verify-clean", file=sys.stderr)

    for e in OUT.glob("*.json"):
        e.unlink()
    for i, c in enumerate(kept, start=1):
        (OUT / f"{i:03d}_{c.id}.json").write_text(json.dumps({
            "id": f"{i:03d}_{c.id}",
            "difficulty": "programmatic-wide",
            "nl": c.nl, "mlir": c.mlir, "notes": c.notes,
            "dialect": "stablehlo+func",
            "source": "day_f8_wide_sweep",
        }, indent=2))
    print(f"[f8] wrote {len(kept)} examples to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    build(target_n=int(os.environ.get("F8_N", "200")))
