"""Day-61: per-prompt table for the gold-differential functional evaluation (all 180 spec prompts).

Context (camera-ready, NeurIPS 2026 E&D). The author response commits to
releasing the gold-differential harness and its per-prompt randomized-trial
results as a named functional-evaluation component covering all 180 spec
prompts (150 MLIR-Spec-150 + 30 Linalg-Spec-30). The Day-56 driver
(scripts/day56_e12c_differential_frozen.py) wrote the gold gate and the
per-trial rows, but kept the per-prompt candidate status in memory only. This
script rebuilds that per-prompt view from the released files, using the Day-56
classification functions verbatim (no execution, no inference), and checks the
result against the published Day-56 summary.

Per prompt: gold-gate outcome (pass / fail / excluded, with reason), the
frozen seed-0 SmolLM2 C1+C2+C3 candidate's status (scored, signature_mismatch,
not_verify_valid, ...), and for scored candidates the number of the 5 seeded
randomized trials that match the gold and the first divergence.

Output: results/day61/functional_per_prompt_180.jsonl
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from scripts.day56_e12c_differential_frozen import (  # noqa: E402
    DIALECTS, K_TRIALS, load_candidates, parse_first_fn, signature_compatible,
)

DAY56 = REPO / "results/day56"
OUT = REPO / "results/day61/functional_per_prompt_180.jsonl"


def main() -> int:
    gate = {(r["dialect"], r["id"]): r for r in map(json.loads, (DAY56 / "e12c_gold_gate.jsonl").read_text().splitlines())}
    trials: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in map(json.loads, (DAY56 / "e12c_differential.jsonl").read_text().splitlines()):
        trials[(r["dialect"], r["id"])].append(r)
    summary = json.loads((DAY56 / "e12c_summary.json").read_text())

    rows = []
    for dkey, cfg in DIALECTS.items():
        examples = [json.loads(p.read_text()) for p in sorted((REPO / cfg["examples_dir"]).glob("*.json"))]
        cands = load_candidates(cfg["cand_dialect"])
        for idx, ex in enumerate(examples):
            g = gate[(dkey, ex["id"])]
            cand = cands.get(idx)
            row = {"dialect": dkey, "idx": idx, "id": ex["id"], "gold_gate": g["gate"],
                   "gold_gate_reason": (g["reason"] or "")[:120] or None,
                   "candidate_verify_valid": bool(cand and cand.get("verify_valid"))}
            if g["gate"] != "pass":
                row["cand_status"] = "gold_not_gated_in"
            elif cand is None:
                row["cand_status"] = "no_candidate_row"
            elif cand.get("nl") and cand["nl"] != ex["nl"]:
                row["cand_status"] = "nl_mismatch"
            elif not cand.get("verify_valid"):
                row["cand_status"] = "not_verify_valid"
            else:
                csig = parse_first_fn(cand["generated"])
                compat, dim_relaxed = signature_compatible(g["sig"], csig) if csig else (False, False)
                if csig is None:
                    row["cand_status"] = "no_parseable_fn"
                elif not compat:
                    row["cand_status"] = "signature_mismatch"
                    row["cand_sig"] = csig
                elif csig["name"] == "main":
                    row["cand_status"] = "fn_named_main"
                else:
                    t = sorted(trials[(dkey, ex["id"])], key=lambda r: r["trial"])
                    assert len(t) == K_TRIALS, (dkey, ex["id"], len(t))
                    n = sum(bool(r["match"]) for r in t)
                    first = next((("lower_fail" if not r["lower_ok"] else "exec_fail" if not r["exec_ok"]
                                   else "output_mismatch") for r in t if not r["match"]), "")
                    row.update({"cand_status": "scored", "dim_relaxed": dim_relaxed,
                                "n_trials_matched": n, "functional_match": n == K_TRIALS,
                                "first_divergence": first or None,
                                "all_trials_lowered_and_executed": all(r["lower_ok"] and r["exec_ok"] for r in t)})
            rows.append(row)

    # consistency with the published Day-56 summary
    ok = True
    for dkey in DIALECTS:
        sub = [r for r in rows if r["dialect"] == dkey and r["gold_gate"] == "pass"]
        pub = summary["dialects"][dkey]
        got = dict(Counter(r["cand_status"] for r in sub))
        scored = [r for r in sub if r["cand_status"] == "scored"]
        checks = {
            "gold_gate_pass": (len(sub), pub["gold_gate_pass"]),
            "cand_status_counts": (got, pub["cand_status_counts"]),
            "n_functional_match": (sum(r["functional_match"] for r in scored), pub["n_functional_match"]),
            "n_scored_dim_relaxed": (sum(bool(r["dim_relaxed"]) for r in scored), pub["n_scored_dim_relaxed"]),
        }
        for k, (a, b) in checks.items():
            status = "ok " if a == b else "DIFF"
            ok &= a == b
            print(f"{status} {dkey}.{k}: rebuilt={a} published={b}")
    vv = {d: sum(r["candidate_verify_valid"] for r in rows if r["dialect"] == d and r["gold_gate"] == "pass")
          for d in DIALECTS}
    print(f"verify-valid candidates among gate-passed prompts: {vv} (posted: arith 68, linalg 22)")
    ok &= vv == {"arith": 68, "linalg": 22}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} prompts -> {OUT.relative_to(REPO)}")
    return 0 if ok and len(rows) == 180 else 1


if __name__ == "__main__":
    sys.exit(main())
