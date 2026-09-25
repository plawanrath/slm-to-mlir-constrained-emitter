"""Day-58 (E12b): re-execution of the EXISTING linalg functional
generations under wrappers adapted to each generation's own declared
signature.

Context: the released functional table (results/day51/functional_n30.json)
shows the linalg cascade 8/10 verify -> 4/10 lower -> 3/10 exec -> 2/10
match. We attribute the verify-to-match gap to a test-harness interface
artifact rather than wrong computation; E12b tests that diagnosis
directly: same 10 references (eval/functional/references.json), same
candidate generations (SmolLM2 C1+C2+C3 seed 1,
results/day39pm/multiseed_fulln.jsonl, verified to reproduce the released
per-reference outcomes), same uniform fill-value input protocol; the ONLY
change is that the driver adapts buffer types to the candidate's declared
signature (day-56 build_driver with callee_types) instead of hard-coding the
reference's canonical types, and the lowering pipeline includes math-to-llvm
(the released runner could not execute math.exp, an unrelated harness gap
also fixed here and reported separately).

Oracle: with uniform fills the expected FINAL value of every buffer follows
from the reference semantics (fill c, copy of a fill, x+y on fills, |x|,
exp(x)); the driver prints every memref argument's final state and the
checker requires each buffer to be uniformly its expected constant
(rel 1e-4). Buffer roles follow the reference's canonical argument order
(inputs first, output last; in-place candidates use their single argument as
both). This is deliberately the released protocol's own oracle, not the
randomized-differential one (that is E12a/E12c); E12b isolates the
harness-adaptation variable.

Output: results/day58/e12b_adapted.jsonl
        results/day58/e12b_summary.json

Writes NEW files only; no released artifact is modified.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.day56_e12c_differential_frozen import (  # noqa: E402
    build_driver, lower_and_run, outputs_equal, parse_first_fn,
    parse_memref_type, parse_stdout, strip_outer_module,
)

# Per-reference input fills and expected final constants, from
# eval/functional/references.json fills + operation semantics.
# in_fills: value per candidate arg position (memref args; scalars given
# separately); expect: expected uniform final value per memref arg position.
# Positions follow canonical order: inputs first, output last; single-arg
# (in-place) candidates: the one buffer is input and output.
PLAN = {
    "F11_linalg_fill_zero_1d":   {"fills": [7.0],           "expect": [0.0]},
    "F12_linalg_fill_value_2d":  {"scalars": [5.0], "fills": [0.0], "expect": [5.0]},
    "F13_linalg_fill_i32_const": {"fills": [9],             "expect": [0]},
    "F14_linalg_copy_1d":        {"fills": [3.5, 0.0],      "expect": [3.5, 3.5]},
    "F15_linalg_copy_2d_static": {"fills": [1.25, 0.0],     "expect": [1.25, 1.25]},
    "F16_linalg_add_elemwise":   {"fills": [2.0, 3.0, 0.0], "expect": [2.0, 3.0, 5.0]},
    "F17_linalg_mul_elemwise_2d": {"fills": [3.0, 4.0, 0.0], "expect": [3.0, 4.0, 12.0]},
    # 2x2 all-ones matmul into C=0: every C entry = 2.0
    "F18_linalg_matmul_2x2":     {"fills": [1.0, 1.0, 0.0], "expect": [1.0, 1.0, 2.0]},
    # exp(0) = 1
    "F19_linalg_exp_elemwise":   {"fills": [0.0, 0.0],      "expect": [0.0, 1.0]},
    "F20_linalg_abs_elemwise":   {"fills": [-2.5, 0.0],     "expect": [-2.5, 2.5]},
}

DYN_DIM = 4  # matches the reference shapes ([4] / [2,2]-scale buffers)

REL = 1e-4
ABS = 1e-5


def adapted_trial(csig, ref_id: str, plan: dict):
    """Shapes from the CANDIDATE signature ('?' -> DYN_DIM); uniform data
    from the plan. Returns (trial_data, expected_per_memref) or (None, err)."""
    shapes, values, expect = [], [], []
    fills = list(plan["fills"])
    scalars = list(plan.get("scalars", []))
    exp = list(plan["expect"])
    # In-place candidates (fewer memref args than the canonical protocol):
    # drop leading canonical buffers so the LAST fills/expect align, except
    # the pure-input fill refs (F11/F13) where the single buffer is arg 0.
    n_memref = sum(1 for t in csig["arg_types"] if t.startswith("memref"))
    if n_memref < len(fills):
        fills = fills[:1] if len(fills) == 1 else fills[-n_memref:]
        exp = exp[-n_memref:]
        # in-place: the surviving buffer starts from the ORIGINAL input fill
        if n_memref == 1 and len(plan["fills"]) > 1:
            fills = [plan["fills"][0]]
    if n_memref > len(fills):
        return None, f"more memref args ({n_memref}) than protocol buffers"
    mi = 0
    for t in csig["arg_types"]:
        if t.startswith("memref"):
            pm = parse_memref_type(t)
            if pm is None:
                return None, f"unsupported memref type {t}"
            dims, dtype = pm
            conc = [DYN_DIM if d == "?" else int(d) for d in dims]
            count = 1
            for d in conc:
                count *= d
            shapes.append(conc)
            v = fills[mi]
            values.append([v] * count)
            expect.append(exp[mi])
            mi += 1
        elif t in ("f32", "f64"):
            if not scalars:
                return None, "scalar arg with no protocol scalar"
            shapes.append(None)
            values.append(float(scalars.pop(0)))
        elif t in ("i1", "i8", "i16", "i32", "i64", "index"):
            if not scalars:
                return None, "int arg with no protocol scalar"
            shapes.append(None)
            values.append(int(scalars.pop(0)))
        else:
            return None, f"unsupported arg type {t}"
    return {"n": DYN_DIM, "shapes": shapes, "values": values}, expect


def check_uniform(parsed: dict, expect: list[float]) -> tuple[bool, str]:
    mrs = parsed["memrefs"]
    if len(mrs) != len(expect):
        return False, f"printed {len(mrs)} memrefs, expected {len(expect)}"
    for k, (mr, e) in enumerate(zip(mrs, expect)):
        if not mr["data"]:
            return False, f"memref {k} empty"
        for v in mr["data"]:
            if abs(v - e) > ABS + REL * max(abs(v), abs(e)):
                return False, f"memref {k}: saw {v}, expected uniform {e}"
    return True, ""


def main():
    out_dir = REPO / "results/day58"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_f = (out_dir / "e12b_adapted.jsonl").open("a")
    summary_path = out_dir / "e12b_summary.json"

    refs = [r for r in json.loads(
        (REPO / "eval/functional/references.json").read_text())["references"]
        if r["dialect"] == "linalg+memref"]
    cands = {}
    for line in (REPO / "results/day39pm/multiseed_fulln.jsonl").read_text().splitlines():
        r = json.loads(line)
        if ("smollm2" in str(r.get("model", "")) and r.get("seed") == 1
                and r.get("dialect") == "linalg"):
            cands[int(r["prompt_id"])] = r

    released = {  # V/L/E/M from results/day51/functional_n30.json
        "F11_linalg_fill_zero_1d": [0, 0, 0, 0],
        "F12_linalg_fill_value_2d": [1, 0, 0, 0],
        "F13_linalg_fill_i32_const": [0, 0, 0, 0],
        "F14_linalg_copy_1d": [1, 0, 0, 0],
        "F15_linalg_copy_2d_static": [1, 1, 1, 1],
        "F16_linalg_add_elemwise": [1, 0, 0, 0],
        "F17_linalg_mul_elemwise_2d": [1, 1, 1, 0],
        "F18_linalg_matmul_2x2": [1, 1, 1, 1],
        "F19_linalg_exp_elemwise": [1, 1, 0, 0],
        "F20_linalg_abs_elemwise": [1, 0, 0, 0],
    }

    t0 = time.time()
    rows = []
    for ref in refs:
        rid = ref["id"]
        pid = int(ref["source_id"].split("_", 1)[0]) - 1
        cand = cands.get(pid)
        row = {"id": rid, "pid": pid, "released_VLEM": released[rid]}
        if cand is None:
            row["status"] = "no_candidate_row"
        elif not cand.get("verify_valid"):
            row["status"] = "not_verify_valid"
        else:
            csig = parse_first_fn(cand["generated"])
            if csig is None:
                row["status"] = "no_parseable_fn"
            else:
                row["cand_sig"] = csig["arg_types"]
                td, expect = adapted_trial(csig, rid, PLAN[rid])
                if td is None:
                    row["status"] = "unadaptable"
                    row["reason"] = expect
                else:
                    sig = {"name": csig["name"],
                           "arg_types": csig["arg_types"], "ret": csig["ret"]}
                    drv = build_driver(strip_outer_module(cand["generated"]),
                                       csig["name"], sig, td)
                    r1 = lower_and_run(drv)
                    r2 = lower_and_run(drv) if r1["exec_ok"] else None
                    stable = bool(r2 and r2["exec_ok"] and outputs_equal(
                        parse_stdout(r1["stdout"]), parse_stdout(r2["stdout"]),
                        exact=True))
                    match, why = ((False, "exec failed") if not r1["exec_ok"]
                                  else check_uniform(parse_stdout(r1["stdout"]),
                                                     expect))
                    row.update({
                        "status": "run",
                        "lower_ok": r1["lower_ok"],
                        "exec_ok": r1["exec_ok"] and stable,
                        "output_match": bool(match and stable),
                        "mismatch_reason": why if not match else "",
                        "err": r1["err"][:200],
                    })
        rows.append(row)
        out_f.write(json.dumps(row) + "\n")
        out_f.flush()
        print(f"  {rid}: {row.get('status')} "
              f"L={row.get('lower_ok')} E={row.get('exec_ok')} "
              f"M={row.get('output_match')} {row.get('mismatch_reason','')[:60]}",
              file=sys.stderr)

        n = len(refs)
        vv = sum(1 for r in rows if r.get("status") == "run")
        summary = {
            "n_refs": n,
            "released_cascade": {"verify": "8/10", "lower": "4/10",
                                 "exec": "3/10", "match": "2/10"},
            "adapted_cascade": {
                "verify": f"{vv}/{n}",
                "lower": f"{sum(1 for r in rows if r.get('lower_ok'))}/{n}",
                "exec": f"{sum(1 for r in rows if r.get('exec_ok'))}/{n}",
                "match": f"{sum(1 for r in rows if r.get('output_match'))}/{n}",
            },
            "per_ref": {r["id"]: {
                "released_VLEM": r["released_VLEM"],
                "adapted": [int(bool(r.get(k))) for k in
                            ("lower_ok", "exec_ok", "output_match")]
                if r.get("status") == "run" else r.get("status")}
                for r in rows},
            "elapsed_min": round((time.time() - t0) / 60.0, 1),
        }
        summary_path.write_text(json.dumps(summary, indent=2))

    out_f.close()
    print(f"[day58] DONE in {round((time.time()-t0)/60,1)} min", file=sys.stderr)
    print(summary_path.read_text(), file=sys.stderr)


if __name__ == "__main__":
    main()
