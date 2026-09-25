"""Day-57 (E12a): randomized-input trials on the EXISTING 30
functional references, using the exact candidate generations behind the
released functional table (results/day51/functional_n30.json).

Context: randomized input/output testing. The released
functional table checks each reference once, on one fixed input. E12a turns
those 30 fixed-input checks into 30 x 5 seeded randomized trials with no
change to any released artifact: same references (eval/functional/
references.json), same candidate generations, same gold modules; only the
harness draws randomized inputs and compares candidate vs gold outputs
differentially (the fixed-input expected_stdout_regex oracle does not apply
to random inputs, and the released benchmark ships a gold module for every
reference's source prompt).

Candidate provenance (verified to reproduce the released per-reference
verify/lower/exec/match pattern):
  - arith+func refs F01-F10:  SmolLM2 C1+C2+C3, seed 1,
        results/day39pm/multiseed_fulln.jsonl (dialect arith+func)
  - linalg refs F11-F20:      same file, dialect linalg, seed 1
  - stablehlo refs F21-F30:   SmolLM2 C1+C3 spec-30 cell,
        results/day32/stablehlo_smoke.jsonl (constraint c1_c3)

Protocol per reference, K=5 seeded trials (identical trial keys and input
bytes as Day-56 for the arith/linalg source prompts):
  1. gold-vs-gold gate: the gold module must execute twice with exactly
     equal parsed outputs on every trial (arith/linalg: mlir-cpu-runner in
     the pinned container; stablehlo: iree-compile + iree-run-module from
     the pinned .venv). Ungated references are excluded with reasons.
  2. differential: the signature-compatible verify-valid candidate runs on
     byte-identical inputs; outputs compared to gold (ints exact, floats
     rel 1e-4 / abs 1e-5). Per-reference functional match = all 5 trials.

Output: results/day57/e12a_gold_gate.jsonl
        results/day57/e12a_trials.jsonl
        results/day57/e12a_summary.json  (live-updated)

Writes NEW files only; no released artifact is modified.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.day56_e12c_differential_frozen import (  # noqa: E402
    K_TRIALS, bootstrap_ci, build_driver, classify_signature, gen_trial,
    lower_and_run, outputs_equal, parse_first_fn, parse_stdout,
    signature_compatible, strip_outer_module, _rng_for,
)

IREE_COMPILE = REPO / ".venv/bin/iree-compile"
IREE_RUN = REPO / ".venv/bin/iree-run-module"

REFS = json.loads((REPO / "eval/functional/references.json").read_text())["references"]

SOURCES = {
    "mlir_spec_150": ("arith", REPO / "eval/benchmarks/mlir_spec_150/examples"),
    "linalg_spec_30": ("linalg", REPO / "eval/benchmarks/linalg_spec_30/examples"),
    "stablehlo_spec_30": ("stablehlo", REPO / "eval/benchmarks/stablehlo_spec_30/examples"),
}


def load_candidates() -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {"arith": {}, "linalg": {}, "stablehlo": {}}
    for line in (REPO / "results/day39pm/multiseed_fulln.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "smollm2" not in str(r.get("model", "")) or r.get("seed") != 1:
            continue
        if r.get("dialect") == "arith+func":
            out["arith"][int(r["prompt_id"])] = r
        elif r.get("dialect") == "linalg":
            out["linalg"][int(r["prompt_id"])] = r
    for line in (REPO / "results/day32/stablehlo_smoke.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "smollm2" in str(r.get("model", "")) and r.get("constraint") == "c1_c3":
            out["stablehlo"][int(r["prompt_id"])] = r
    return out


# ---------------- stablehlo (iree) path ----------------

_TENSOR_RE = re.compile(r"^tensor<([0-9?x]*?)x?([a-z]+\d+)>$")


def parse_tensor_type(t: str):
    m = _TENSOR_RE.match(t)
    if not m:
        return None
    dims_src, dtype = m.group(1), m.group(2)
    dims = dims_src.split("x") if dims_src else []
    if any(d != "?" and not d.isdigit() for d in dims):
        return None
    return dims, dtype


def shlo_classify(sig):
    """Exclusion reason or None. Supports functions of static/dynamic ranked
    tensors of f32/f64/i32 returning one tensor."""
    if sig is None:
        return "no_public_fn"
    if not sig["ret"]:
        return "no_result"
    if parse_tensor_type(sig["ret"]) is None:
        return "unsupported_ret_type"
    for t in sig["arg_types"]:
        pt = parse_tensor_type(t)
        if pt is None:
            return "unsupported_arg_type"
        if pt[1] not in ("f32", "f64", "i32"):
            return "unsupported_dtype"
    return None


def shlo_compatible(gold_sig, cand_sig) -> bool:
    if cand_sig is None or len(cand_sig["arg_types"]) != len(gold_sig["arg_types"]):
        return False
    for gt, ct in zip(gold_sig["arg_types"] + [gold_sig["ret"]],
                      cand_sig["arg_types"] + [cand_sig["ret"]]):
        if gt == ct:
            continue
        gm, cm = parse_tensor_type(gt), parse_tensor_type(ct)
        if gm is None or cm is None:
            return False
        (gd, gdt), (cd, cdt) = gm, cm
        if gdt != cdt or len(gd) != len(cd):
            return False
        if not all(c == "?" or c == g for g, c in zip(gd, cd)):
            return False
    return True


def shlo_gen_inputs(sig, prompt_key: str, trial: int) -> list[str]:
    """iree --input strings: '<dims>x<dtype>=v v v ...' (row-major flat)."""
    rng = _rng_for(prompt_key, trial)
    n = 3 + trial
    inputs = []
    for t in sig["arg_types"]:
        dims, dtype = parse_tensor_type(t)
        conc = [n if d == "?" else int(d) for d in dims]
        count = 1
        for d in conc:
            count *= d
        if dtype.startswith("f"):
            vals = [rng.randrange(2, 33) * 0.25 for _ in range(count)]
            body = " ".join(f"{v}" for v in vals)
        else:
            vals = [rng.randint(1, 7) for _ in range(count)]
            body = " ".join(str(v) for v in vals)
        shape = "x".join(str(d) for d in conc)
        inputs.append(f"{shape}x{dtype}={body}" if shape else f"{dtype}={body}")
    return inputs


def shlo_compile(mlir_text: str, tag: str, tmpdir: Path) -> tuple[Path | None, str]:
    src = tmpdir / f"{tag}.mlir"
    vmfb = tmpdir / f"{tag}.vmfb"
    src.write_text(mlir_text)
    r = subprocess.run(
        [str(IREE_COMPILE), "--iree-input-type=stablehlo",
         "--iree-hal-target-backends=llvm-cpu", str(src), "-o", str(vmfb)],
        capture_output=True, text=True, timeout=120.0)
    if r.returncode != 0:
        return None, (r.stderr or "")[:300]
    return vmfb, ""


def shlo_run(vmfb: Path, fn: str, inputs: list[str]) -> dict:
    cmd = [str(IREE_RUN), f"--module={vmfb}", f"--function={fn}",
           "--device=local-task"] + [f"--input={i}" for i in inputs]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60.0)
    except subprocess.TimeoutExpired:
        return {"exec_ok": False, "out": None, "err": "exec_timeout"}
    if r.returncode != 0:
        return {"exec_ok": False, "out": None, "err": (r.stderr or "")[:300]}
    return {"exec_ok": True, "out": shlo_parse(r.stdout or ""), "err": ""}


_NUM_RE = re.compile(r"-?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|nan|inf)")


def shlo_parse(stdout: str) -> dict:
    """Parse iree-run-module results into {'results': [{'shape': str,
    'data': [floats]}]}."""
    results = []
    blocks = re.split(r"result\[\d+\]: hal\.buffer_view", stdout)
    for block in blocks[1:]:
        eq = block.find("=")
        if eq < 0:
            results.append({"shape": "", "data": []})
            continue
        shape = block[:eq].strip().splitlines()[-1].strip()
        data = [float(x) for x in _NUM_RE.findall(block[eq + 1:])]
        results.append({"shape": shape, "data": data})
    return {"results": results}


def shlo_equal(x: dict, y: dict, exact: bool = False,
               rel: float = 1e-4, absr: float = 1e-5) -> bool:
    if len(x["results"]) != len(y["results"]):
        return False
    for a, b in zip(x["results"], y["results"]):
        if a["shape"] != b["shape"] or len(a["data"]) != len(b["data"]):
            return False
        for u, v in zip(a["data"], b["data"]):
            if u != u and v != v:
                continue
            if exact:
                if u != v:
                    return False
            elif abs(u - v) > absr + rel * max(abs(u), abs(v)):
                return False
    return True


# ---------------- main ----------------

def main():
    out_dir = REPO / "results/day57"
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_f = (out_dir / "e12a_gold_gate.jsonl").open("a")
    trial_f = (out_dir / "e12a_trials.jsonl").open("a")
    summary_path = out_dir / "e12a_summary.json"

    cands = load_candidates()
    t0 = time.time()
    ref_rows: list[dict] = []

    for ref in REFS:
        rid = ref["id"]
        dkey, ex_dir = SOURCES[ref["source_benchmark"]]
        src_id = ref["source_id"]
        ex = json.loads((ex_dir / f"{src_id}.json").read_text())
        gold = ex["mlir"]
        pid = int(src_id.split("_", 1)[0]) - 1
        cand = cands[dkey].get(pid)
        sig = parse_first_fn(gold)
        prompt_key = f"{dkey}:{src_id}"  # same trial keys/bytes as Day-56
        row = {"id": rid, "dialect": dkey, "source_id": src_id, "pid": pid}

        if dkey in ("arith", "linalg"):
            reason, detail = classify_signature(sig)
            if reason:
                row.update({"gate": "excluded", "reason": reason, "detail": detail})
                _emit(gate_f, row, ref_rows)
                continue
            gate_ok, gate_err, gold_outs = True, "", {}
            for t in range(K_TRIALS):
                td = gen_trial(sig, prompt_key, t)
                driver = build_driver(strip_outer_module(gold), sig["name"], sig, td)
                ra, rb = lower_and_run(driver), None
                if ra["exec_ok"]:
                    rb = lower_and_run(driver)
                if not (ra["exec_ok"] and rb and rb["exec_ok"] and outputs_equal(
                        parse_stdout(ra["stdout"]), parse_stdout(rb["stdout"]),
                        exact=True)):
                    gate_ok, gate_err = False, f"trial{t}:{(ra.get('err') or 'nondet')[:150]}"
                    break
                gold_outs[t] = parse_stdout(ra["stdout"])
            row.update({"gate": "pass" if gate_ok else "fail",
                        "reason": None if gate_ok else gate_err})
            if not gate_ok:
                _emit(gate_f, row, ref_rows)
                continue
            # candidate
            if cand is None:
                row["cand_status"] = "no_candidate_row"
            elif not cand.get("verify_valid"):
                row["cand_status"] = "not_verify_valid"
            else:
                csig = parse_first_fn(cand["generated"])
                compat, relaxed = (signature_compatible(sig, csig)
                                   if csig else (False, False))
                if not compat:
                    row["cand_status"] = "signature_mismatch"
                    row["cand_sig"] = csig
                else:
                    row["cand_status"] = "scored"
                    row["dim_relaxed"] = relaxed
                    n_match = 0
                    for t in range(K_TRIALS):
                        td = gen_trial(sig, prompt_key, t)
                        drv = build_driver(strip_outer_module(cand["generated"]),
                                           csig["name"], sig, td,
                                           callee_types=csig["arg_types"])
                        rc = lower_and_run(drv)
                        m = rc["exec_ok"] and outputs_equal(
                            parse_stdout(rc["stdout"]), gold_outs[t])
                        n_match += bool(m)
                        trial_f.write(json.dumps({
                            "id": rid, "dialect": dkey, "trial": t,
                            "lower_ok": rc["lower_ok"], "exec_ok": rc["exec_ok"],
                            "match": bool(m), "err": rc["err"][:200]}) + "\n")
                        trial_f.flush()
                    row["n_trials_matched"] = n_match
                    row["functional_match"] = (n_match == K_TRIALS)
            _emit(gate_f, row, ref_rows)

        else:  # stablehlo
            reason = shlo_classify(sig)
            if reason:
                row.update({"gate": "excluded", "reason": reason})
                _emit(gate_f, row, ref_rows)
                continue
            with tempfile.TemporaryDirectory() as td_:
                tmpdir = Path(td_)
                gvmfb, cerr = shlo_compile(gold, "gold", tmpdir)
                if gvmfb is None:
                    row.update({"gate": "fail", "reason": f"gold_compile:{cerr[:150]}"})
                    _emit(gate_f, row, ref_rows)
                    continue
                gate_ok, gate_err, gold_outs = True, "", {}
                for t in range(K_TRIALS):
                    ins = shlo_gen_inputs(sig, prompt_key, t)
                    ra = shlo_run(gvmfb, sig["name"], ins)
                    rb = shlo_run(gvmfb, sig["name"], ins) if ra["exec_ok"] else None
                    if not (ra["exec_ok"] and rb and rb["exec_ok"]
                            and shlo_equal(ra["out"], rb["out"], exact=True)):
                        gate_ok = False
                        gate_err = f"trial{t}:{(ra.get('err') or 'nondet')[:150]}"
                        break
                    gold_outs[t] = ra["out"]
                row.update({"gate": "pass" if gate_ok else "fail",
                            "reason": None if gate_ok else gate_err})
                if not gate_ok:
                    _emit(gate_f, row, ref_rows)
                    continue
                if cand is None:
                    row["cand_status"] = "no_candidate_row"
                elif not cand.get("verify_valid"):
                    row["cand_status"] = "not_verify_valid"
                else:
                    csig = parse_first_fn(cand["generated"])
                    if not shlo_compatible(sig, csig):
                        row["cand_status"] = "signature_mismatch"
                        row["cand_sig"] = csig
                    else:
                        cvmfb, cerr = shlo_compile(cand["generated"], "cand", tmpdir)
                        if cvmfb is None:
                            row["cand_status"] = "scored"
                            row["n_trials_matched"] = 0
                            row["functional_match"] = False
                            row["compile_err"] = cerr[:200]
                            for t in range(K_TRIALS):
                                trial_f.write(json.dumps({
                                    "id": rid, "dialect": dkey, "trial": t,
                                    "lower_ok": False, "exec_ok": False,
                                    "match": False, "err": "compile_fail"}) + "\n")
                            trial_f.flush()
                        else:
                            row["cand_status"] = "scored"
                            n_match = 0
                            for t in range(K_TRIALS):
                                ins = shlo_gen_inputs(sig, prompt_key, t)
                                rc = shlo_run(cvmfb, csig["name"], ins)
                                m = rc["exec_ok"] and shlo_equal(rc["out"], gold_outs[t])
                                n_match += bool(m)
                                trial_f.write(json.dumps({
                                    "id": rid, "dialect": dkey, "trial": t,
                                    "lower_ok": True, "exec_ok": rc["exec_ok"],
                                    "match": bool(m), "err": rc["err"][:200]}) + "\n")
                                trial_f.flush()
                            row["n_trials_matched"] = n_match
                            row["functional_match"] = (n_match == K_TRIALS)
            _emit(gate_f, row, ref_rows)

        _write_summary(summary_path, ref_rows, t0)

    _write_summary(summary_path, ref_rows, t0)
    gate_f.close(); trial_f.close()
    print(f"[day57] DONE in {round((time.time()-t0)/60,1)} min", file=sys.stderr)
    print(summary_path.read_text(), file=sys.stderr)


def _emit(f, row, acc):
    acc.append(row)
    f.write(json.dumps(row) + "\n")
    f.flush()
    status = row.get("cand_status", row.get("gate"))
    extra = (f" match={row.get('n_trials_matched')}/{K_TRIALS}"
             if "n_trials_matched" in row else "")
    print(f"  [{row['dialect']}] {row['id']}: gate={row.get('gate')} "
          f"cand={status}{extra}", file=sys.stderr)


def _write_summary(path: Path, ref_rows: list[dict], t0: float):
    from collections import Counter
    summary = {"k_trials": K_TRIALS,
               "candidates": {
                   "arith": "results/day39pm/multiseed_fulln.jsonl (smollm2 seed 1)",
                   "linalg": "results/day39pm/multiseed_fulln.jsonl (smollm2 seed 1)",
                   "stablehlo": "results/day32/stablehlo_smoke.jsonl (c1_c3)"},
               "released_fixed_input_match": {"arith": "8/10", "linalg": "2/10",
                                              "stablehlo": "5/10"},
               "dialects": {}, "overall": {}}
    for dkey in ("arith", "linalg", "stablehlo"):
        rows = [r for r in ref_rows if r["dialect"] == dkey]
        scored = [r for r in rows if r.get("cand_status") == "scored"]
        matched = [r for r in scored if r.get("functional_match")]
        summary["dialects"][dkey] = {
            "n_refs": len(rows),
            "gate": dict(Counter(r.get("gate") for r in rows)),
            "cand_status": dict(Counter(r.get("cand_status") for r in rows
                                        if r.get("cand_status"))),
            "n_scored": len(scored),
            "n_functional_match": len(matched),
            "per_ref": {r["id"]: r.get("n_trials_matched") for r in scored},
        }
    scored = [r for r in ref_rows if r.get("cand_status") == "scored"]
    matched = [r for r in scored if r.get("functional_match")]
    flags = [bool(r.get("functional_match")) for r in scored]
    summary["overall"] = {
        "n_scored": len(scored), "n_functional_match": len(matched),
        "match_ci95": bootstrap_ci(flags) if flags else None,
        "total_trials": sum(r.get("n_trials_matched", 0) is not None and K_TRIALS
                            for r in scored),
        "trials_matched": sum(r.get("n_trials_matched", 0) for r in scored),
    }
    summary["elapsed_min"] = round((time.time() - t0) / 60.0, 1)
    path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
