"""Day-61: recompute every author-response number that had no committed script.

Context (camera-ready, NeurIPS 2026 E&D). The author responses state that every
reported number is recomputable from released raw generations or
response-period reruns. The values below were computed ad hoc during the
response period; this script recomputes each one from the frozen per-prompt
files and checks it against the value posted to reviewers. No inference, no
verifier calls: pure aggregation over released JSONL files.

  POOLS        pool construction: 95 eligible linalg corpus rows; 193/200 and 75/125
      unique prompts; spec and pad disjoint; counts unchanged under
      lowercase/whitespace/punctuation normalization
  FORENSICS    67.5 vs 52.0 forensics: 93 shared unique prompts, 60/93 (draft run) vs
      62/93 (frozen run); the 100 released-only prompts score 37.0%
  BEST_CONFIG  best-configuration table (C1-only vs C1+C3) with pool provenance, and the
      linalg C3 deltas
  TRUNCATION   truncation signature in baseline outputs and worst-case credit bounds
  UNIQUE       unique-prompt (deduplicated) three-seed means
  LEAKAGE      leakage: SmolLM2 spec-prompt generations reproducing the gold module
      (whitespace-normalized) and coincidences among all generation-gold pairs
  POOL_SHIFT   pool-shift diagnosis for the Day-4 draft-pool baseline cells (C1-only and
      free) vs their frozen-pool reruns (scripts/day61_frozen_c1only_baselines.py).
      On the prompts both pools share, verify outcomes are identical (CodeLlama
      outputs byte-identical 58/58; Granite 54/58 and 57/58, the differing
      outputs have the same outcome), so the rate change is prompt-pool
      composition. Mechanism: only 15 of the draft pool's first 150 prompts are
      released MLIR-Spec-150 prompts; the rest are weak test-file descriptions,
      mostly of three words or fewer ("canonicalize"). Verify-valid does not
      check task fidelity, so copying the few-shot example scores on those
      prompts (CodeLlama does; Granite does not).

Output: results/day61/rebuttal_numbers.json (every recomputed value, the posted
value, and whether they agree). Exit status 1 if any check disagrees.
"""
from __future__ import annotations

import glob
import json
import os
import re
import string
import sys
from collections import defaultdict
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])
os.chdir(REPO)
sys.path.insert(0, REPO)

from scripts.day18_30b_c3_rejection import TARGET_LINALG_OPS, _prompts_arith, _prompts_linalg

MULTISEED = [
    "results/day51_seed0_n200/multiseed_seed0.jsonl",   # seed 0, all systems, both dialects
    "results/day39pm/multiseed_fulln.jsonl",            # seeds 1-2 (arith all; SmolLM2 linalg)
    "results/day50/baseline_linalg_multiseed.jsonl",    # seeds 1-2, baselines on linalg
]
SYSTEMS = ["smollm2-c1c2c3", "codellama-c1c3", "starcoder2-c1c3", "granite-c1c3"]
BASELINES = SYSTEMS[1:]


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def pct(k: float, n: int) -> float:
    return round(100 * k / n, 1)


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, key: str, computed, posted, note: str = "") -> None:
        self.rows.append({"key": key, "computed": computed, "posted": posted,
                          "agree": computed == posted, "note": note})


def e1(c: Checks) -> dict:
    eligible = 0
    for line in open("data/processed/l3_tests.jsonl"):
        r = json.loads(line)
        mlir = r.get("mlir", "")
        if "linalg." not in mlir or "tensor<" in mlir or not r.get("weak_nl", "").strip():
            continue
        ops = {m.group(1) for m in re.finditer(r"linalg\.(\w+)", mlir)}
        if ops and ops <= set(TARGET_LINALG_OPS):
            eligible += 1
    c.add("POOLS.linalg_eligible_corpus_rows", eligible, 95)
    norm = lambda s: " ".join(s.lower().translate(str.maketrans("", "", string.punctuation)).split())
    out = {"linalg_eligible_corpus_rows": eligible}
    for name, pool, n_spec, posted in (("arith", _prompts_arith(200), 150, 193),
                                       ("linalg", _prompts_linalg(125), 30, 75)):
        uniq, uniq_norm = len(set(pool)), len({norm(p) for p in pool})
        disjoint = not (set(pool[:n_spec]) & set(pool[n_spec:]))
        c.add(f"POOLS.{name}_unique", uniq, posted)
        c.add(f"POOLS.{name}_unique_normalized", uniq_norm, posted)
        c.add(f"POOLS.{name}_spec_pad_disjoint", disjoint, True)
        out[name] = {"n": len(pool), "unique": uniq, "unique_normalized": uniq_norm,
                     "spec_pad_disjoint": disjoint}
    return out


def e2(c: Checks) -> dict:
    """One value per unique prompt string, taken from its first occurrence."""
    draft = [r for r in load("results/frozen/week2_gate_day5/c3_smoke.jsonl") if r["constraint"] == "c1_c2_c3"]
    frozen = [r for r in load(MULTISEED[0]) if r["model"] == "smollm2-c1c2c3" and r["dialect"] == "arith+func"]

    def first(rows):
        d = {}
        for r in rows:
            d.setdefault(r["nl"], bool(r["verify_valid"]))
        return d

    o, n = first(draft), first(frozen)
    shared = set(o) & set(n)
    only = [r for r in frozen if r["nl"] not in o]
    res = {
        "draft_rate": pct(sum(r["verify_valid"] for r in draft), len(draft)),
        "frozen_rate": pct(sum(r["verify_valid"] for r in frozen), len(frozen)),
        "shared_unique": len(shared),
        "shared_draft_pass": sum(o[k] for k in shared),
        "shared_frozen_pass": sum(n[k] for k in shared),
        "released_only_rows": len(only),
        "released_only_rate": pct(sum(r["verify_valid"] for r in only), len(only)),
    }
    c.add("FORENSICS.draft_rate", res["draft_rate"], 67.5)
    c.add("FORENSICS.frozen_rate", res["frozen_rate"], 52.0)
    c.add("FORENSICS.shared_unique", res["shared_unique"], 93)
    c.add("FORENSICS.shared_draft_pass", res["shared_draft_pass"], 60)
    c.add("FORENSICS.shared_frozen_pass", res["shared_frozen_pass"], 62)
    c.add("FORENSICS.shared_draft_rate", pct(res["shared_draft_pass"], 93), 64.5)
    c.add("FORENSICS.shared_frozen_rate", pct(res["shared_frozen_pass"], 93), 66.7)
    c.add("FORENSICS.released_only_rows", res["released_only_rows"], 100)
    c.add("FORENSICS.released_only_rate", res["released_only_rate"], 37.0)
    return res


def three_seed_mean(rows, model, dialect) -> float:
    per = []
    for s in (0, 1, 2):
        sub = [r for r in rows if r["model"] == model and r["dialect"] == dialect and r["seed"] == s]
        per.append(100 * sum(bool(r["verify_valid"]) for r in sub) / len(sub))
    return round(sum(per) / 3, 1)


def e3(c: Checks, ms: list[dict]) -> dict:
    arith, linalg = _prompts_arith(200), _prompts_linalg(125)

    def cell(path, model, dialect, pool):
        rows = [r for r in load(path) if r["model"] == model and (r.get("dialect") in (None, dialect))]
        rows.sort(key=lambda r: int(r["prompt_id"]))
        return {"rate": pct(sum(bool(r["verify_valid"]) for r in rows), len(rows)),
                "n": len(rows), "source": path,
                "frozen_pool": [r["nl"] for r in rows] == pool}

    c1 = {
        ("codellama", "arith"): cell("results/day4/baselines_30b.jsonl", "codellama-c1", "arith+func", arith),
        ("granite", "arith"): cell("results/day4/baselines_30b.jsonl", "granite-c1", "arith+func", arith),
        ("starcoder2", "arith"): cell("results/day19/starcoder2_baselines.jsonl", "starcoder2-c1", "arith+func", arith),
        ("codellama", "linalg"): cell("results/day10/linalg_baselines.jsonl", "codellama-c1", "linalg", linalg),
        ("granite", "linalg"): cell("results/day10/linalg_baselines.jsonl", "granite-c1", "linalg", linalg),
        ("starcoder2", "linalg"): cell("results/day19/starcoder2_baselines.jsonl", "starcoder2-c1", "linalg", linalg),
    }
    free = {
        ("granite", "arith"): cell("results/day4/baselines_30b.jsonl", "granite-free", "arith+func", arith),
        ("granite", "linalg"): cell("results/day10/linalg_baselines.jsonl", "granite-free", "linalg", linalg),
    }
    c1c3 = {(m.split("-")[0], d.split("+")[0]): three_seed_mean(ms, m, d)
            for m in BASELINES for d in ("arith+func", "linalg")}
    posted_c1 = {("codellama", "arith"): 82.0, ("starcoder2", "arith"): 67.0, ("granite", "arith"): 27.0,
                 ("codellama", "linalg"): 56.8, ("starcoder2", "linalg"): 47.2, ("granite", "linalg"): 33.6}
    posted_c1c3 = {("codellama", "arith"): 59.8, ("starcoder2", "arith"): 66.8, ("granite", "arith"): 51.5,
                   ("codellama", "linalg"): 58.7, ("starcoder2", "linalg"): 54.9, ("granite", "linalg"): 35.7}
    for k, v in posted_c1.items():
        c.add(f"BEST_CONFIG.c1only.{k[0]}.{k[1]}", c1[k]["rate"], v,
              "" if c1[k]["frozen_pool"] else "PRE-FREEZE DRAFT POOL (see day61_frozen_c1only_baselines.py)")
    for k, v in posted_c1c3.items():
        c.add(f"BEST_CONFIG.c1c3.{k[0]}.{k[1]}", c1c3[k], v)
    for m, posted in (("codellama", 1.9), ("starcoder2", 7.7), ("granite", 2.1)):
        c.add(f"BEST_CONFIG.linalg_c3_delta.{m}", round(c1c3[(m, "linalg")] - c1[(m, "linalg")]["rate"], 1), posted)
    c.add("BEST_CONFIG.codellama_arith_c3_drop", round(c1[("codellama", "arith")]["rate"] - c1c3[("codellama", "arith")], 1),
          22.2, "draft-pool C1-only minus frozen-pool C1+C3")
    return {"c1_only": {f"{a}.{b}": v for (a, b), v in c1.items()},
            "free": {f"{a}.{b}": v for (a, b), v in free.items()},
            "c1_c3_three_seed": {f"{a}.{b}": v for (a, b), v in c1c3.items()}}


def e8(c: Checks, ms: list[dict]) -> dict:
    posted = {("arith+func", "codellama-c1c3"): (0.0, None), ("arith+func", "starcoder2-c1c3"): (1.8, None),
              ("arith+func", "granite-c1c3"): (4.8, None), ("linalg", "codellama-c1c3"): (2.4, 61.1),
              ("linalg", "starcoder2-c1c3"): (7.7, 62.6), ("linalg", "granite-c1c3"): (10.4, 46.1)}
    out = {}
    for (d, m), (p_tr, p_worst) in posted.items():
        sub = [r for r in ms if r["model"] == m and r["dialect"] == d]
        tr = [r for r in sub if r["generated"].count("{") > r["generated"].count("}")]
        valid = sum(bool(r["verify_valid"]) for r in sub)
        worst = pct(valid + len(tr), len(sub))
        out[f"{d}.{m}"] = {"n": len(sub), "truncated": len(tr), "truncated_pct": pct(len(tr), len(sub)),
                           "truncated_and_valid": sum(bool(r["verify_valid"]) for r in tr),
                           "worst_case_credit": worst}
        c.add(f"TRUNCATION.trunc_pct.{d}.{m}", pct(len(tr), len(sub)), p_tr)
        c.add(f"TRUNCATION.trunc_all_invalid.{d}.{m}", out[f"{d}.{m}"]["truncated_and_valid"] == 0, True)
        if p_worst is not None:
            c.add(f"TRUNCATION.worst_case.{d}.{m}", worst, p_worst,
                  "posted value = rounded verify rate + rounded truncation rate" if worst != p_worst else "")
    return out


def e13(c: Checks, ms: list[dict]) -> dict:
    """One value per unique prompt (mean over its duplicates), per seed, then seed-averaged."""
    def dedup(model, dialect):
        per = []
        for s in (0, 1, 2):
            g = defaultdict(list)
            for r in ms:
                if r["model"] == model and r["dialect"] == dialect and r["seed"] == s:
                    g[r["nl"]].append(bool(r["verify_valid"]))
            per.append(100 * sum(sum(v) / len(v) for v in g.values()) / len(g))
        return round(sum(per) / 3, 1)

    posted = {"linalg": (75.7, 63.6, 40.4, 38.6), "arith+func": (52.2, 59.4, 67.2, 53.0)}
    out = {}
    for d, vals in posted.items():
        for m, p in zip(SYSTEMS, vals):
            v = dedup(m, d)
            out[f"{d}.{m}"] = v
            c.add(f"UNIQUE.{d}.{m}", v, p)
    c.add("UNIQUE.linalg_min_margin", round(out["linalg.smollm2-c1c2c3"] - max(out[f"linalg.{m}"] for m in BASELINES), 1), 12.1)
    return out


def leakage(c: Checks, ms: list[dict]) -> dict:
    gold = {}
    for i, p in enumerate(sorted(glob.glob("eval/benchmarks/mlir_spec_150/examples/*.json"))):
        gold[("arith+func", i)] = json.load(open(p))["mlir"]
    for i, p in enumerate(sorted(glob.glob("eval/benchmarks/linalg_spec_30/examples/*.json"))):
        gold[("linalg", i)] = json.load(open(p))["mlir"]
    ws = lambda s: " ".join(s.split())
    pairs = [(r, gold[(r["dialect"], int(r["prompt_id"]))]) for r in ms if (r["dialect"], int(r["prompt_id"])) in gold]
    hits = [(r, g) for r, g in pairs if ws(r["generated"]) == ws(g)]
    by_model = defaultdict(int)
    for r, _ in hits:
        by_model[r["model"]] += 1
    smol = sum(1 for r, _ in pairs if r["model"] == "smollm2-c1c2c3")
    ops_in_hits = sorted({len(re.findall(r"\b[a-z_]+\.[a-z_]+\b(?=[ (])", g.split("func.func", 1)[-1]))
                          for _, g in hits})
    res = {"pairs": len(pairs), "smollm2_generations": smol, "hits_by_model": dict(by_model),
           "hits_total": len(hits), "op_counts_in_hit_golds": ops_in_hits}
    c.add("LEAKAGE.pairs", len(pairs), 2160)
    c.add("LEAKAGE.smollm2_generations", smol, 540)
    c.add("LEAKAGE.smollm2_hits", by_model.get("smollm2-c1c2c3", 0), 0)
    c.add("LEAKAGE.hits_total", len(hits), 44)
    return res


def pool_shift(c: Checks) -> dict:
    """Split each Day-4 vs frozen-pool change into shared prompts vs pool-only prompts."""
    draft_all = load("results/day4/baselines_30b.jsonl")
    released_spec = {json.load(open(p))["nl"] for p in glob.glob("eval/benchmarks/mlir_spec_150/examples/*.json")}
    frozen_path = "results/day61/frozen_c1only_arith.jsonl"
    if not Path(frozen_path).exists():
        return {"skipped": f"{frozen_path} missing"}
    frozen_all = load(frozen_path)
    rate = lambda rows: [sum(bool(r["verify_valid"]) for r in rows), len(rows)]
    out = {}
    for model in ("codellama-c1", "granite-c1", "granite-free"):
        old = [r for r in draft_all if r["model"] == model]
        new = [r for r in frozen_all if r["model"] == model]
        o, n = {}, {}
        for r in old:
            o.setdefault(r["nl"], r)
        for r in new:
            n.setdefault(r["nl"], r)
        shared = sorted(set(o) & set(n))
        ident = sum(o[k]["generated"].strip() == n[k]["generated"].strip() for k in shared)
        res = {
            "draft_overall": rate(old), "frozen_overall": rate(new),
            "shared_unique_prompts": len(shared),
            "shared_draft": rate([o[k] for k in shared]), "shared_frozen": rate([n[k] for k in shared]),
            "shared_outputs_byte_identical": ident,
            "draft_only_rows": rate([r for r in old if r["nl"] not in n]),
            "frozen_only_rows": rate([r for r in new if r["nl"] not in o]),
            "draft_spec": rate([r for r in old if int(r["prompt_id"]) < 150]),
            "draft_pad": rate([r for r in old if int(r["prompt_id"]) >= 150]),
            "frozen_spec": rate([r for r in new if int(r["prompt_id"]) < 150]),
            "frozen_pad": rate([r for r in new if int(r["prompt_id"]) >= 150]),
            "shared_verify_outcomes_identical": sum(bool(o[k]["verify_valid"]) == bool(n[k]["verify_valid"]) for k in shared),
        }
        dspec = [r for r in old if int(r["prompt_id"]) < 150]
        other = [r for r in dspec if r["nl"] not in released_spec]
        res["draft_ids_0_149_in_released_spec"] = rate([r for r in dspec if r["nl"] in released_spec])
        res["draft_ids_0_149_not_released"] = rate(other)
        res["draft_ids_0_149_not_released_le3_words"] = rate([r for r in other if len(r["nl"].split()) <= 3])
        res["draft_ids_0_149_unique_prompts"] = len({r["nl"] for r in dspec})
        out[model] = res
        c.add(f"POOL_SHIFT.{model}.shared_verify_outcomes_identical", res["shared_verify_outcomes_identical"], len(shared),
              f"byte-identical outputs {ident}/{len(shared)}")
    return out


def main() -> int:
    c = Checks()
    ms = sum((load(p) for p in MULTISEED), [])
    result = {"POOLS": e1(c), "FORENSICS": e2(c), "BEST_CONFIG": e3(c, ms), "TRUNCATION": e8(c, ms),
              "UNIQUE": e13(c, ms), "LEAKAGE": leakage(c, ms), "POOL_SHIFT": pool_shift(c)}
    result["checks"] = c.rows
    out = Path("results/day61/rebuttal_numbers.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    bad = [r for r in c.rows if not r["agree"]]
    for r in c.rows:
        flag = "ok " if r["agree"] else "DIFF"
        print(f"{flag} {r['key']:42s} computed={r['computed']!s:8s} posted={r['posted']!s:8s} {r['note']}")
    print(f"\n{len(c.rows) - len(bad)}/{len(c.rows)} agree; written {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
