"""Day-60 (E15): direct StableHLO verifier version-drift measurement.

How sensitive are results to MLIR/StableHLO version drift? The existing
check measures cross-IMPLEMENTATION concordance at the same pin
(stablehlo-opt v1.4.0 vs iree-compile 20241104.1068, 50/50). This script
measures version drift directly: re-verify every released StableHLO
module under a much newer iree-compile (iree-base-compiler 3.11.0rc20260316,
PyPI, ~16 months past the pin) and report per-module agreement with the pinned
verifier, both run through the same harness here.

Population: released gold/reference modules only (no model generations, no LLM
inference): StableHLO-Spec-30 golds (30), StableHLO-Held-Out-200 golds (200,
constructed iree-compile-clean at the pin), and StableHLO-Out-Of-Grammar-25
(25, reported separately: out-of-grammar but hand-authored valid StableHLO).

Verifier call replicates the paper's wrapper contract: `iree-compile
--compile-to=input` on stdin-fed module text, with the `func.func @` substring
guard (the empty-module gotcha disclosed in the paper).

Output: results/day60/e15_iree_drift.jsonl (one row per module x compiler)
        results/day60/e15_summary.json

Setup (one-time; keep the newer compiler out of the pinned .venv):
  python -m venv /path/to/iree_drift_venv
  /path/to/iree_drift_venv/bin/pip install iree-base-compiler==3.11.0rc20260316

Usage:
  python scripts/day60_e15_iree_drift.py \
      --new-compiler /path/to/iree_drift_venv/bin/iree-compile
  (or set IREE_COMPILE_NEW instead of passing --new-compiler)

Writes NEW files only; no released artifact is modified and the pinned .venv
is untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PINNED = REPO / ".venv/bin/iree-compile"

BENCHES = {
    "stablehlo_spec_30": REPO / "eval/benchmarks/stablehlo_spec_30/examples",
    "stablehlo_held_out_200": REPO / "eval/benchmarks/stablehlo_held_out_200/examples",
    "stablehlo_outofgrammar_25": REPO / "eval/benchmarks/stablehlo_outofgrammar_25/examples",
}


def verify(compiler: Path, mlir: str) -> tuple[bool, str]:
    if "func.func @" not in mlir:
        return False, "guard: no func.func"
    p = subprocess.run(
        [str(compiler), "--compile-to=input", "-"],
        input=mlir.encode(), capture_output=True, timeout=120,
    )
    return p.returncode == 0, p.stderr.decode(errors="replace")[-400:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-compiler", default=os.environ.get("IREE_COMPILE_NEW"),
                    help="path to the newer iree-compile (default: $IREE_COMPILE_NEW)")
    args = ap.parse_args()
    if not args.new_compiler:
        ap.error("pass --new-compiler or set IREE_COMPILE_NEW")
    new = Path(args.new_compiler).expanduser().resolve()
    for c in (PINNED, new):
        if not c.is_file():
            ap.error(f"iree-compile not found: {c}")

    out_dir = REPO / "results/day60"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "e15_iree_drift.jsonl"
    summary: dict = {"pinned": str(PINNED), "new": str(new)}
    for tag, compiler in (("pinned", PINNED), ("new", new)):
        v = subprocess.run([str(compiler), "--version"], capture_output=True, text=True)
        summary[f"{tag}_version"] = (v.stdout or v.stderr).strip().splitlines()[0] if (v.stdout or v.stderr) else "?"
    t0 = time.perf_counter()
    with rows_path.open("w") as fout:
        for bench, ex_dir in BENCHES.items():
            files = sorted(ex_dir.glob("*.json"))
            agree = pinned_ok = new_ok = 0
            disagreements = []
            for f in files:
                d = json.loads(f.read_text())
                mlir = d["mlir"]
                ok_p, err_p = verify(PINNED, mlir)
                ok_n, err_n = verify(new, mlir)
                if ok_p == ok_n: agree += 1
                else: disagreements.append({"id": d.get("id", f.stem), "pinned": ok_p, "new": ok_n,
                                            "err": (err_n if ok_p else err_p)})
                pinned_ok += ok_p; new_ok += ok_n
                fout.write(json.dumps({"bench": bench, "id": d.get("id", f.stem),
                                       "pinned_ok": ok_p, "new_ok": ok_n}) + "\n")
            summary[bench] = {
                "n": len(files), "pinned_accept": pinned_ok, "new_accept": new_ok,
                "agree": agree, "agree_rate": agree / max(len(files), 1),
                "disagreements": disagreements,
            }
            print(f"[day60] {bench}: n={len(files)} pinned={pinned_ok} new={new_ok} "
                  f"agree={agree}/{len(files)}", file=sys.stderr)
    summary["wall_s"] = time.perf_counter() - t0
    (out_dir / "e15_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[day60] done in {summary['wall_s']:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
