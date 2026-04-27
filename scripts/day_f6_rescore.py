"""Phase F: re-score Day 32 / Day 42 / F6 StableHLO jsonl files using
the patched verify_stablehlo (which rejects empty/no-func inputs).

Day 32/42 results were scored with the pre-patch verify, which treated
empty stdin as a valid empty module. Re-scoring corrects this.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

REPO = "/Users/plawanrath/Documents/GitHub-public/slm-to-mlir-constrained-emitter"
os.chdir(REPO); sys.path.insert(0, REPO)

from scripts.env.verify_stablehlo import verify_stablehlo


def rescore(path: Path) -> dict:
    rows = [json.loads(l) for l in path.open()]
    by_constraint: dict[str, dict[str, int]] = {}
    for r in rows:
        gen = r.get("generated", "")
        vv = verify_stablehlo(gen)["returncode"] == 0
        # For StableHLO, parse_valid == verify_valid (same gate)
        r["parse_valid"] = vv
        r["verify_valid"] = vv
        c = r["constraint"]
        if c not in by_constraint:
            by_constraint[c] = {"n": 0, "pv": 0, "vv": 0, "empty": 0}
        by_constraint[c]["n"] += 1
        by_constraint[c]["pv"] += int(vv)
        by_constraint[c]["vv"] += int(vv)
        if not gen.strip():
            by_constraint[c]["empty"] += 1
    # Write back in-place (non-destructive: new field names)
    fixed = path.with_suffix(".fixed.jsonl")
    with fixed.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return {str(path): by_constraint}


def main():
    targets = [
        Path("results/day32/stablehlo_smoke.jsonl"),
        Path("results/day42/stablehlo_held_out.jsonl"),
        Path("results/day_f6/outofgrammar_matrix.jsonl"),
    ]
    summary = {}
    for t in targets:
        if t.exists():
            summary.update(rescore(t))
    out = Path("results/day_f6/phase_f_rescore_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
