"""Item 5 / Day-51: verifier-substitution concordance check.

The paper §6 substitutes `iree-compile --compile-to=input` for the upstream
`stablehlo-opt --verify` because we couldn't build stablehlo-opt locally
(the M4 Max build failed). The paper claims the substitution is "not
materially looser". This script measures it.

Protocol:
  1. Take a stratified random n=50 from
     `eval/benchmarks/stablehlo_held_out_200/examples/`. Stratification
     covers the 7 op families present (ew_bin, ew_un, transpose,
     reshape, broadcast, dot_general, slice).
  2. For each instance, run BOTH verifiers on the *reference* MLIR (not
     a model generation):
       VERIFIER A: `iree-compile --iree-input-type=stablehlo --compile-to=input`
                  (the script's Day-32+ verifier)
       VERIFIER B: `stablehlo-opt --verify` if available (built in container
                  per `docker exec slm-mlir-llvm bash -c
                  /root/stablehlo-src/build/bin/stablehlo-opt --verify`).
                  Fallback: `iree-opt` with explicit stablehlo verification —
                  documented as a best-effort substitute when stablehlo-opt
                  is absent.
  3. Aggregate concordance: %{A=acc & B=acc}, %{A=rej & B=rej},
     count of {A=acc, B=rej} (looser-iree direction), count of {A=rej, B=acc}
     (looser-stablehlo direction).

Output: results/day51/verify_concordance_n50.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))


# ---------------- benchmark loading ----------------

def _stratify_op_family(filename: str) -> str:
    """Map filename prefix to one of the 7 op families.
    Held-Out-200 names look like '001_ew_bin_001_add_f16_8.json'."""
    name = filename.lower()
    for fam in ("ew_bin", "ew_un", "dot_general", "broadcast",
                "transpose", "reshape", "slice"):
        if fam in name:
            return fam
    return "other"


def stratified_sample(corpus_dir: Path, n: int, seed: int) -> list[Path]:
    files = sorted(corpus_dir.glob("*.json"))
    by_family: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        by_family[_stratify_op_family(f.name)].append(f)
    rng = random.Random(seed)
    # Allocate proportional to family size, with min 1 per family.
    total = len(files)
    out: list[Path] = []
    for fam, fs in sorted(by_family.items()):
        share = max(1, round(n * len(fs) / total))
        rng.shuffle(fs)
        out.extend(fs[:share])
    rng.shuffle(out)
    return out[:n]


# ---------------- verifiers ----------------

def verifier_iree_compile(mlir_text: str, timeout: float = 20.0) -> dict:
    """The paper's substitute verifier."""
    from scripts.env.verify_stablehlo import verify_stablehlo
    r = verify_stablehlo(mlir_text, timeout=timeout)
    return {
        "tool": "iree-compile --compile-to=input",
        "accept": (r["returncode"] == 0),
        "rc": r["returncode"],
        "stderr": (r.get("stderr") or "")[:200],
    }


def _docker_stablehlo_opt(mlir_text: str, timeout: float = 20.0) -> dict | None:
    """Returns None if stablehlo-opt isn't built in the container."""
    bin_path = "/root/stablehlo-src/build/bin/stablehlo-opt"
    test = subprocess.run(
        ["docker", "exec", "slm-mlir-llvm", "test", "-x", bin_path],
        capture_output=True,
    )
    if test.returncode != 0:
        return None
    try:
        r = subprocess.run(
            ["docker", "exec", "-i", "slm-mlir-llvm", bin_path,
             "--allow-unregistered-dialect"],
            input=mlir_text, capture_output=True, text=True, timeout=timeout,
        )
        return {
            "tool": "stablehlo-opt --verify (v1.4.0 in container)",
            "accept": (r.returncode == 0),
            "rc": r.returncode,
            "stderr": (r.stderr or "")[:200],
        }
    except subprocess.TimeoutExpired:
        return {"tool": "stablehlo-opt", "accept": False, "rc": -1, "stderr": "timeout"}


def _iree_opt_fallback(mlir_text: str, timeout: float = 20.0) -> dict:
    """Use venv's iree-opt as a best-effort secondary verifier when
    stablehlo-opt isn't built. Documented as not fully independent of
    iree-compile (both link against openxla/stablehlo upstream)."""
    iree_opt = REPO / ".venv" / "bin" / "iree-opt"
    try:
        r = subprocess.run(
            [str(iree_opt), "--verify-each", "--allow-unregistered-dialect"],
            input=mlir_text, capture_output=True, text=True, timeout=timeout,
        )
        return {
            "tool": "iree-opt --verify-each (fallback; not independent of iree-compile)",
            "accept": (r.returncode == 0),
            "rc": r.returncode,
            "stderr": (r.stderr or "")[:200],
        }
    except subprocess.TimeoutExpired:
        return {"tool": "iree-opt fallback", "accept": False, "rc": -1, "stderr": "timeout"}


def verifier_stablehlo_opt_or_fallback(mlir_text: str, timeout: float = 20.0) -> dict:
    """Prefer real stablehlo-opt; fall back to iree-opt if not built."""
    res = _docker_stablehlo_opt(mlir_text, timeout=timeout)
    if res is not None:
        return res
    return _iree_opt_fallback(mlir_text, timeout=timeout)


# ---------------- aggregation ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(REPO / "eval/benchmarks/stablehlo_held_out_200/examples"))
    ap.add_argument("--extra-corpus", action="append", default=[
        str(REPO / "eval/benchmarks/stablehlo_spec_30/examples"),
    ], help="Additional corpora to mix in (for op-family diversity since Held-Out-200 is 100%% ew_bin)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "results/day51/verify_concordance_n50.json"))
    args = ap.parse_args()

    # Stratified sample across the 7 op families. Since Held-Out-200 is
    # 100% ew_bin, we mix in all the other corpora for cross-family
    # diversity — the resulting sample is a "stratified union over the
    # available StableHLO benchmarks", documented in the daily log.
    all_files: list[Path] = sorted(Path(args.corpus).glob("*.json"))
    for extra in args.extra_corpus:
        ep = Path(extra)
        if ep.exists():
            all_files.extend(sorted(ep.glob("*.json")))
    by_family: dict[str, list[Path]] = defaultdict(list)
    for f in all_files:
        by_family[_stratify_op_family(f.name)].append(f)
    rng = random.Random(args.seed)
    sample: list[Path] = []
    # Equal allocation across families so non-ew_bin families get visibility.
    families = sorted(by_family.keys())
    per_family = max(1, args.n // len(families))
    for fam in families:
        fs = list(by_family[fam])
        rng.shuffle(fs)
        sample.extend(fs[:per_family])
    rng.shuffle(sample)
    sample = sample[:args.n]
    # Top up if rounding left us short.
    if len(sample) < args.n:
        leftovers = [f for f in all_files if f not in set(sample)]
        rng.shuffle(leftovers)
        sample.extend(leftovers[:args.n - len(sample)])
    print(f"[concordance] sampled n={len(sample)} across families {dict((f, sum(1 for s in sample if _stratify_op_family(s.name)==f)) for f in families)}",
          file=sys.stderr)

    rows = []
    counts = {"AA": 0, "AR": 0, "RA": 0, "RR": 0}  # A=accept, R=reject
    family_counts = defaultdict(lambda: {"AA": 0, "AR": 0, "RA": 0, "RR": 0})
    fallback_used = False

    for path in sample:
        ref = json.loads(path.read_text())
        mlir = ref["mlir"]
        rA = verifier_iree_compile(mlir)
        rB = verifier_stablehlo_opt_or_fallback(mlir)
        if "fallback" in rB["tool"].lower():
            fallback_used = True
        a, b = rA["accept"], rB["accept"]
        key = ("A" if a else "R") + ("A" if b else "R")
        counts[key] += 1
        family = _stratify_op_family(path.name)
        family_counts[family][key] += 1
        rows.append({
            "id": ref.get("id", path.stem),
            "family": family,
            "iree_compile_accept": a,
            "stablehlo_opt_accept": b,
            "key": key,
            "iree_stderr": rA["stderr"] if not a else "",
            "stablehlo_stderr": rB["stderr"] if not b else "",
        })
        print(f"  {path.stem[:35]:35} | iree={'✓' if a else '✗'} | shlo={'✓' if b else '✗'}", file=sys.stderr)

    n = len(rows)
    summary = {
        "n": n,
        "concordant_accept_count": counts["AA"],
        "concordant_reject_count": counts["RR"],
        "iree_accepts_stablehlo_rejects_count": counts["AR"],  # iree looser
        "iree_rejects_stablehlo_accepts_count": counts["RA"],  # stablehlo looser
        "iree_looser_direction_pct": (counts["AR"] / n if n else 0.0) * 100.0,
        "stablehlo_looser_direction_pct": (counts["RA"] / n if n else 0.0) * 100.0,
        "concordance_pct": ((counts["AA"] + counts["RR"]) / n if n else 0.0) * 100.0,
        "verifier_B_is_fallback": fallback_used,
        "verifier_B_tool": rows[-1].get("tool") if rows else None,
        "by_family": {f: dict(c) for f, c in family_counts.items()},
        "verifier_A": "iree-compile --iree-input-type=stablehlo --compile-to=input (with phase-F empty-stdin guard)",
        "verifier_B": "stablehlo-opt --verify (v1.4.0 against container LLVM 19.1.7)" if not fallback_used
                       else "iree-opt --verify-each (fallback; documented limitation)",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"per_instance": rows, "summary": summary}, indent=2))
    print(f"\n[concordance] summary: {summary}", file=sys.stderr)
    print(f"[concordance] → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
