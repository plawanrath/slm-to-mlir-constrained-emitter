"""Day-35: Lowerability spot check on 20 SmolLM2 arith+func samples.

For each selected generation from Day-5 (SmolLM2 + C1+C2+C3 arith+func),
run:
  1. `mlir-opt --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts`
  2. `mlir-translate --mlir-to-llvmir`

If both stages succeed, the code is LOWERABLE — it passes LLVM dialect
conversion and translation to LLVM IR. This is a STRONGER check than
`mlir-opt --verify` alone (which only validates structural correctness)
because it exercises semantic validity across the lowering.

What it does NOT check: functional correctness (does the code do what
the NL asks?). That would require an executable build + random-input
testing against a reference implementation. We scope Day 35 to
lowerability as a feasible middle ground.

Output: results/day35/lowerability.jsonl
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
os.chdir(REPO); sys.path.insert(0, REPO)

MLIR_OPT = Path("scripts/env/bin/mlir-opt")
MLIR_TRANSLATE = Path("scripts/env/bin/mlir-translate")

N_SAMPLES = 20


def _lower_to_llvm_dialect(mlir_text: str) -> tuple[int, str, str]:
    r = subprocess.run(
        [str(MLIR_OPT),
         "--convert-arith-to-llvm",
         "--convert-func-to-llvm",
         "--reconcile-unrealized-casts"],
        input=mlir_text, capture_output=True, text=True, timeout=20,
    )
    return r.returncode, r.stdout, r.stderr


def _translate_to_llvm_ir(llvm_dialect: str) -> tuple[int, str, str]:
    r = subprocess.run(
        [str(MLIR_TRANSLATE), "--mlir-to-llvmir"],
        input=llvm_dialect, capture_output=True, text=True, timeout=20,
    )
    return r.returncode, r.stdout, r.stderr


def _load_c3_verified() -> list[tuple[str, str]]:
    """Pull SmolLM2 + C1+C2+C3 verify-valid samples from Day-5 frozen data."""
    path = Path("results/frozen/week2_gate_day5/c3_smoke.jsonl")
    if not path.exists():
        path = Path("results/day5/c3_smoke.jsonl")
    if not path.exists():
        return []
    out: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("constraint") == "c1_c2_c3" and r.get("verify_valid"):
            out.append((r.get("nl", ""), r.get("generated", "")))
    return out[:N_SAMPLES]


def run(out_path: Path) -> None:
    samples = _load_c3_verified()
    print(f"[day35] {len(samples)} SmolLM2 + C1+C2+C3 verify-clean samples", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lower_ok = translate_ok = 0
    with out_path.open("w") as f:
        for i, (nl, mlir) in enumerate(samples):
            rc1, out1, err1 = _lower_to_llvm_dialect(mlir)
            stage1 = (rc1 == 0)
            stage2 = False
            ir = ""
            err2 = ""
            if stage1:
                rc2, out2, err2 = _translate_to_llvm_ir(out1)
                stage2 = (rc2 == 0)
                ir = out2[:1000] if stage2 else ""
            if stage1: lower_ok += 1
            if stage2: translate_ok += 1
            f.write(json.dumps({
                "prompt_id": i, "nl": nl,
                "mlir_src": mlir[:500],
                "lowered_to_llvm_dialect": stage1,
                "translated_to_llvm_ir": stage2,
                "llvm_ir_preview": ir[:500],
                "stage1_err": err1[:400] if not stage1 else "",
                "stage2_err": err2[:400] if not stage2 else "",
            }) + "\n")
            f.flush()
            print(f"  [{i+1:2d}/{len(samples)}] stage1={stage1} stage2={stage2} nl={nl[:60]}",
                  file=sys.stderr)
    print(f"\n[day35] Lowerability: stage1={lower_ok}/{len(samples)} "
          f"stage2={translate_ok}/{len(samples)}", file=sys.stderr)


if __name__ == "__main__":
    run(Path("results/day35/lowerability.jsonl"))
