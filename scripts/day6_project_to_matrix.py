"""Project Day-4 / Day-5 smoke JSONLs into the canonical matrix schema
{model, constraint, dialect, seed, prompt_id, passed, verify_stderr}.

The eval ablations (`hcs_replication`, `error_categories`) expect this shape.
We keep Day-5's rich rows (attempts, dt, etc.) intact in their own files and
produce `results/day6/matrix.jsonl` as the ablation-friendly view.

`verify_stderr` is filled from the sqlite verify cache; rows whose generation
was verify-valid get an empty stderr.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = "/Users/plawanrath/Documents/GitHub-public/slm-to-mlir-constrained-emitter"
os.chdir(REPO)
sys.path.insert(0, REPO)

from scripts.env.verify_cache import VerifyCache


SOURCES = [
    # (path, source_tag)
    ("results/day4/slm_smoke.jsonl",     "day4_slm_smoke"),
    ("results/day4/c2_ablation_zeroshot.jsonl", "day4_b2"),
    ("results/day5/c3_smoke.jsonl",      "day5_c3_smoke"),
]

OUT = Path("results/day6/matrix.jsonl")


def main() -> None:
    cache = VerifyCache()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Dedup by (model, constraint, prompt_id, seed). Later sources win —
    # so Day-5 SmolLM2 rows overwrite Day-4 SmolLM2 rows, keeping Phi Day-4
    # cells intact (Day 5 didn't re-run Phi).
    bucket: dict[tuple, dict] = {}
    n_in = 0
    for src_path, source_tag in SOURCES:
        p = Path(src_path)
        if not p.exists():
            print(f"[project] skip missing {p}", file=sys.stderr)
            continue
        for line in p.read_text().splitlines():
            n_in += 1
            row = json.loads(line)
            text = row["generated"]
            parse_valid = bool(row.get("parse_valid", False))
            verify_valid = bool(row.get("verify_valid", False))
            stderr = ""
            if parse_valid:
                cached = cache.get(text, flags=("--verify-diagnostics",))
                if cached is not None:
                    stderr = cached.get("stderr", "") or ""
            key = (row["model"], row["constraint"], int(row["prompt_id"]), 0)
            bucket[key] = {
                "source": source_tag,
                "model": row["model"],
                "constraint": row["constraint"],
                "dialect": "arith+func",
                "seed": 0,
                "prompt_id": int(row["prompt_id"]),
                "passed": verify_valid,
                "parse_valid": parse_valid,
                "verify_valid": verify_valid,
                "verify_stderr": stderr,
            }
    with OUT.open("w") as fout:
        for key in sorted(bucket):
            fout.write(json.dumps(bucket[key]) + "\n")
    print(f"[project] {n_in} → {len(bucket)} rows (deduped) → {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
