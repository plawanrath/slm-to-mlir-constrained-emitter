# RUNBOOK — SLM-to-MLIR M1

Step-by-step commands for executing the 17-day plan (`docs/execution_plan.md`).
Everything runs in one of three environments:

- **Docker** (`slm-mlir-llvm` container) — pinned LLVM, `mlir-opt`, Polygeist, `clang`, `FileCheck`.
- **Project venv** (`.venv/`) — all Python code, including MLX-LM inference.
- **Host Ollama** — the one documented host-level exception; required for 30B Q4 baselines (Docker on macOS has no Metal access). See §0.3.

Nothing else is installed on the laptop. Rebuild the container on any fresh machine; the venv + Ollama are both fully scripted below.

---

## §0. One-time setup (Day 1, ~2 hours mostly unattended)

### 0.1. Prerequisites

```bash
# Docker Desktop on macOS (one-time; if you already have it, skip)
brew install --cask docker
open -a Docker            # wait for the whale icon

# Python 3.11+ (for the venv; comes with Homebrew)
brew install python@3.11
```

If you cannot `brew install` and want to stay 100% Docker, use `docker run -it python:3.11 bash` as an alternative runtime — but MLX won't work there, only Ollama baselines would. The recommended path is the project venv.

### 0.2. Build the pinned LLVM container

```bash
cd <repo-root>
docker compose -f scripts/env/docker-compose.yml build
docker compose -f scripts/env/docker-compose.yml up -d
# sanity:
docker exec slm-mlir-llvm mlir-opt --version
docker exec slm-mlir-llvm which polygeist || echo "polygeist optional; clang+mlir-translate fallback ok"
```

Expected cold build: 30-60 min (LLVM + MLIR + Polygeist). Cached afterward.

Add the wrapper directory to your PATH so `mlir-opt`, `mlir-translate`, `polygeist`, `FileCheck`, `clang` all transparently shell into the container:

```bash
export PATH="$PWD/scripts/env/bin:$PATH"
```

Append that line to `~/.zshrc` only if you want it permanent; otherwise re-export per shell session. (This is PATH-only — no binaries are installed on the host.)

### 0.3. Install Ollama + pull the 30B baselines

Ollama is the sole host-level dependency (Metal-only; Docker on macOS can't access the GPU). Everything else stays in venv/Docker.

```bash
brew install --cask ollama
open -a Ollama
ollama pull codellama:34b-instruct-q4_K_M
ollama pull granite-code:34b-instruct-q4_K_M
ollama list
```

For non-macOS users: the reproducibility Dockerfile will include a llama.cpp CPU fallback (~20-30× slower but fully Docker-native).

### 0.4. Project venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r scripts/env/requirements.txt
pip install -e .          # make `grammar`, `decoder`, `eval`, `data`, `train` importable
```

Smoke test the venv + Docker integration:

```bash
pytest -x -q
# should run grammar/tests, decoder/tests, eval/tests (excluding @pytest.mark.docker tests)

# Docker smoke:
echo 'module { func.func @f() -> i32 { %0 = arith.constant 0 : i32; return %0 : i32 } }' \
  | mlir-opt --verify-diagnostics -
# expect exit 0 + the same MLIR printed back
```

### 0.5. Clone the source trees we mine from

We don't vendor these; keep them outside the repo to avoid bloating `git status`.

```bash
mkdir -p ~/src
git clone --depth 1 https://github.com/llvm/llvm-project.git ~/src/llvm-project
git clone --depth 1 https://github.com/iree-org/iree.git ~/src/iree        # optional
git clone --depth 1 https://github.com/openxla/stablehlo.git ~/src/stablehlo  # optional
```

---

## §1. Day 2 — Tokenizer shootout + first data pipelines

Today's output lockdowns: **primary SLM + secondary SLM** (from shootout); **L0 ODS pairs + lattice JSONs**; **L3 pairs**; **L1 Polygeist** kicked off in background.

### 1.1. Tokenizer shootout

```bash
source .venv/bin/activate
python scripts/tokenizer_shootout.py \
  --llvm-src ~/src/llvm-project \
  --out results/tokenizer_shootout.json \
  --sample-lines 10000
cat results/tokenizer_shootout.json | python -m json.tool
```

Decide primary + secondary from the printed table. Default is Phi-3.5-mini (primary) + SmolLM2-1.7B (secondary) unless shootout surprises you.

### 1.2. L0 ODS miner (fast, minutes)

```bash
python -m data.pipelines.l0_ods.mine \
  --llvm-src ~/src/llvm-project \
  --iree-src ~/src/iree \
  --stablehlo-src ~/src/stablehlo \
  --out data/processed/l0_ods.jsonl \
  --target-dialects arith,func,linalg,memref \
  --lattice-out grammar/lattices
wc -l data/processed/l0_ods.jsonl           # expect ≥15k
ls grammar/lattices/                         # expect arith.json func.json linalg.json memref.json
```

### 1.3. L3 test miner (fast, minutes)

```bash
python -m data.pipelines.l3_tests.mine \
  --llvm-src ~/src/llvm-project \
  --out data/processed/l3_tests.jsonl \
  --target-dialects arith,func,linalg,memref \
  --require-verify          # slower but keeps only verifier-clean tests
wc -l data/processed/l3_tests.jsonl          # expect ≥5k
```

### 1.4. L1 Polygeist pipeline (long, background)

CodeSearchNet C subset first (one-time download + JSONL prep). If you already have a JSONL of `{code, docstring}` pairs, skip this step.

```bash
python -c "
from datasets import load_dataset
import json
ds = load_dataset('code_search_net', 'java', split='train[:10000]')
# replace 'java' with 'c' if a C subset is exposed; otherwise use a C corpus of your choice.
with open('data/raw/codesearchnet_c.jsonl', 'w') as f:
    for r in ds:
        f.write(json.dumps({
            'code': r.get('func_code_string', ''),
            'docstring': r.get('func_documentation_string', ''),
        }) + '\n')
"
```

Kick off the lowering run in the background:

```bash
nohup python -m data.pipelines.l1_polygeist.driver \
  --input data/raw/codesearchnet_c.jsonl \
  --out data/processed/l1_polygeist.jsonl \
  --max-pairs 50000 --workers 8 \
  > logs/l1.log 2>&1 &
tail -f logs/l1.log
```

### 1.5. Grammar round-trip check (debug v0 → v1)

```bash
pytest grammar/tests/test_roundtrip.py -v
# now extend to a held-out L3 sample:
python -c "
import json, random
from grammar.parser import is_parse_valid
lines = open('data/processed/l3_tests.jsonl').read().splitlines()
random.seed(0); random.shuffle(lines)
sample = [json.loads(l)['mlir'] for l in lines[:1000]]
ok = sum(is_parse_valid(s) for s in sample)
print(f'parse-valid on 1k L3 sample: {ok/10:.1f}%')
"
```

Target: ≥95%. If below, tighten `grammar/mlir.lark` (attributes, compound types are the usual culprits) and rerun.

---

## §2. Day 3 — C1 sanity + baseline kick-off + MLIR-Spec authoring starts

### 2.1. C1 parse-valid on held-out (Week-1 gate dry run)

```bash
python -m eval.harness \
  --eval-set data/processed/l3_tests.jsonl \
  --models "microsoft/Phi-3.5-mini-instruct" \
  --constraints c1 \
  --dialects arith+func \
  --seeds 0 \
  --n-per-cell 500 \
  --out results/c1_sanity.jsonl
python -c "
import json
from eval.stats import bootstrap_ci, pass_rate
rows = [json.loads(l) for l in open('results/c1_sanity.jsonl')]
passed = [int(r['passed']) for r in rows]
print(f'n={len(passed)}, pass@1={pass_rate(passed):.3f}, 95% CI={bootstrap_ci(passed).as_dict()}')
"
```

### 2.2. CodeLlama-34B free-decoding baseline (background, ~6-8h)

```bash
nohup python -m eval.harness \
  --eval-set data/processed/l3_tests.jsonl \
  --models codellama:34b-instruct-q4_K_M \
  --constraints none \
  --dialects arith+func linalg \
  --seeds 0 1 2 \
  --n-per-cell 300 \
  --backend-override ollama \
  --out results/baseline_codellama_free.jsonl \
  > logs/codellama_free.log 2>&1 &
```

### 2.3. MLIR-Spec-150 authoring

```bash
$EDITOR eval/benchmarks/mlir_spec_150/examples/002_*.json
# ... target 25 pairs today.
python -m eval.benchmarks.mlir_spec_150.validate
```

---

## §3. Day 4 — C2 integration + C1+C2 run + more baselines

### 3.1. C2 correctness on verified-gold set

Before running the expensive eval matrix, make sure C1+C2 never rejects a known-good MLIR:

```bash
python -c "
import json
from grammar.parser import is_parse_valid
from decoder.c2_type_arity import is_composite_compatible
lines = open('data/processed/l1_polygeist.jsonl').read().splitlines()
miss = 0
for line in lines[:500]:
    rec = json.loads(line)
    # C1 must accept every line:
    if not is_parse_valid(rec['mlir']):
        miss += 1
print(f'C1 rejected {miss}/500 verified-gold L1 samples')
# Expect miss ≤ 25 (5%). If higher, fix grammar before proceeding.
"
```

### 3.2. SLM eval: C1+C2 on arith+func, n=500 smoke

```bash
python -m eval.harness \
  --eval-set data/processed/l3_tests.jsonl \
  --models "microsoft/Phi-3.5-mini-instruct" "HuggingFaceTB/SmolLM2-1.7B-Instruct" \
  --constraints none c1 c1_c2 \
  --dialects arith+func \
  --seeds 0 \
  --n-per-cell 500 \
  --out results/day4_smoke.jsonl
```

### 3.3. CodeLlama-34B + C1 and Granite-Code-34B baselines (background)

```bash
nohup python -m eval.harness \
  --eval-set data/processed/l3_tests.jsonl \
  --models codellama:34b-instruct-q4_K_M \
  --constraints c1 \
  --dialects arith+func linalg --seeds 0 1 2 --n-per-cell 300 \
  --backend-override ollama \
  --out results/baseline_codellama_c1.jsonl > logs/codellama_c1.log 2>&1 &

nohup python -m eval.harness \
  --eval-set data/processed/l3_tests.jsonl \
  --models granite-code:34b-instruct-q4_K_M \
  --constraints none c1 \
  --dialects arith+func linalg --seeds 0 1 2 --n-per-cell 300 \
  --backend-override ollama \
  --out results/baseline_granite.jsonl > logs/granite.log 2>&1 &
```

### 3.4. MLIR-Spec-150 (target 50 today)

---

## §4. Day 5 — Week-1 gate + full SLM matrix on arith+func + entropy

### 4.1. Week-1 gate (parse-valid ≥95%)

```bash
python -c "
import json
from grammar.parser import is_parse_valid
matrix = [json.loads(l) for l in open('results/day4_smoke.jsonl')]
c1 = [r for r in matrix if r['constraint'] == 'c1']
ok = sum(is_parse_valid(r.get('generated', '')) for r in c1)
print(f'C1 parse-valid: {ok}/{len(c1)} = {ok/max(len(c1),1):.1%}')
"
# If < 95% on either SLM: STOP, debug grammar. Do not proceed to Day 6.
```

### 4.2. Full SLM matrix on arith+func, n=1000

```bash
python -m eval.harness \
  --eval-set data/processed/l3_tests.jsonl \
  --models "microsoft/Phi-3.5-mini-instruct" "HuggingFaceTB/SmolLM2-1.7B-Instruct" \
  --constraints none c1 c1_c2 \
  --dialects arith+func \
  --seeds 0 1 2 \
  --n-per-cell 1000 \
  --out results/matrix_arith_func.jsonl
```

### 4.3. Entropy analysis

```bash
# Export a prompt file from L3 held-out:
python -c "
import json
lines = open('data/processed/l3_tests.jsonl').read().splitlines()[:500]
with open('data/processed/entropy_prompts.jsonl','w') as f:
    for l in lines:
        r = json.loads(l)
        f.write(json.dumps({'prompt': 'Emit MLIR for: ' + r['weak_nl']}) + '\n')
"
for c in none c1 c1_c2; do
  python -m eval.entropy \
    --model microsoft/Phi-3.5-mini-instruct \
    --prompts data/processed/entropy_prompts.jsonl \
    --dialect arith+func --constraint $c \
    --out results/entropy.json --n-prompts 500
done
```

### 4.4. MLIR-Spec-150 (target 100 today)

---

## §5. Day 6 — HCS replication + MLIR-Spec-150 push to 130

```bash
python -m eval.ablations.hcs_replication \
  --matrix results/matrix_arith_func.jsonl \
  --model microsoft/Phi-3.5-mini-instruct \
  --dialect arith+func \
  --out results/hcs_ablation.json
cat results/hcs_ablation.json | python -m json.tool
```

Expect `reversal_detected: true` and `c1c2_vs_c1_paired_diff.ci_low > 0`.

---

## §6. Day 7 — Week-2 gate + MLIR-Spec-150 lock + linalg go/no-go

### 6.1. Week-2 gate

```bash
python -c "
import json
from eval.ablations.hcs_replication import run
rows = [json.loads(l) for l in open('results/matrix_arith_func.jsonl')]
r = run(rows, 'microsoft/Phi-3.5-mini-instruct', 'arith+func')
print(r)
# Gate: r['c1c2_vs_c1_paired_diff']['ci_low'] must be > 0.
# Equivalent to: C1+C2 beats C1 by at least some margin with 95% confidence.
# Design gate: ≥5pp point estimate. If closer to 0, pivot framing.
"
```

### 6.2. Lock MLIR-Spec-150

```bash
python -m eval.benchmarks.mlir_spec_150.validate
# fix any errors, then commit the JSON files:
ls eval/benchmarks/mlir_spec_150/examples/ | wc -l  # should be 150
git add eval/benchmarks/mlir_spec_150/examples/
git commit -m "lock MLIR-Spec-150 (n=150, 2-person review)"
```

### 6.3. LoRA training (overnight)

```bash
# Split L1 into train/val:
python -c "
import json, random
lines = open('data/processed/l1_polygeist.jsonl').read().splitlines()
random.seed(0); random.shuffle(lines)
val_n = 500
with open('data/processed/l1_polygeist_train.jsonl','w') as f: f.writelines(l+'\n' for l in lines[val_n:])
with open('data/processed/l1_polygeist_val.jsonl','w') as f: f.writelines(l+'\n' for l in lines[:val_n])
"

nohup python -m train.lora_phi \
  --base-model microsoft/Phi-3.5-mini-instruct \
  --train-data data/processed/l1_polygeist_train.jsonl \
  --val-data data/processed/l1_polygeist_val.jsonl \
  --out checkpoints/phi-3.5-mini-lora \
  --rank 16 --iters 2000 \
  > logs/lora.log 2>&1 &
```

### 6.4. Go/no-go on linalg

If Week-2 gate passed comfortably (CI lower bound well above 0), extend `grammar/mlir.lark` + `grammar/lattices/linalg.json` for `linalg`. If barely passed, freeze at arith+func and skip Days 8-9.

---

## §7. Days 8-9 — linalg + error categorization

```bash
# Full matrix on linalg (if pursuing):
python -m eval.harness \
  --eval-set data/processed/l3_tests.jsonl \
  --models "microsoft/Phi-3.5-mini-instruct" "HuggingFaceTB/SmolLM2-1.7B-Instruct" \
  --constraints none c1 c1_c2 \
  --dialects linalg --seeds 0 1 2 --n-per-cell 1000 \
  --out results/matrix_linalg.jsonl

# MLIR-Spec-150 eval:
python -m eval.harness \
  --eval-set eval/benchmarks/mlir_spec_150/examples.jsonl \
  --models "microsoft/Phi-3.5-mini-instruct" "HuggingFaceTB/SmolLM2-1.7B-Instruct" \
    codellama:34b-instruct-q4_K_M granite-code:34b-instruct-q4_K_M \
  --constraints none c1 c1_c2 \
  --dialects arith+func linalg --seeds 0 1 2 --n-per-cell 150 \
  --out results/matrix_spec150.jsonl

# Concatenate everything for downstream analysis:
cat results/matrix_arith_func.jsonl results/matrix_linalg.jsonl results/matrix_spec150.jsonl \
  > results/matrix.jsonl

python -m eval.error_categories --matrix results/matrix.jsonl --out results/error_categories.json
```

---

## §8. Day 10 — Week-3 gate + figure generation

```bash
python -c "
import json
from eval.stats import paired_bootstrap_diff
rows = [json.loads(l) for l in open('results/matrix.jsonl')]
# SLM C1+C2 vs 30B free: pick strongest case per dialect.
def cell(model, constraint, dialect):
    return sorted([(r['prompt_id'], r['seed'], int(r['passed']))
                   for r in rows
                   if r['model']==model and r['constraint']==constraint and r['dialect']==dialect])
a = cell('microsoft/Phi-3.5-mini-instruct', 'c1_c2', 'arith+func')
b = cell('codellama:34b-instruct-q4_K_M', 'c1', 'arith+func')
keys = sorted(set((x[0],x[1]) for x in a) & set((x[0],x[1]) for x in b))
aa = [p for k,s,p in a if (k,s) in keys]
bb = [p for k,s,p in b if (k,s) in keys]
print(paired_bootstrap_diff(aa, bb).as_dict())
# Gate: |diff.point| ≤ 0.03 OR diff.ci_low > -0.03.
"

python scripts/make_figures.py --results results --out docs/paper/figures
ls docs/paper/figures
```

---

## §9. Days 11-17 — buffer + writing + reproducibility

- **Day 11**: re-run any cell whose CI width exceeded 0.08; freeze numbers in `results/frozen/`.
- **Day 12**: abstract + intro + method in `docs/paper/main.tex`. Pull figures from §8.
- **Day 13**: results + ablation sections.
- **Day 14**: related work + limitations + explicit future-work list.
- **Day 15**: internal review + one external reader.
- **Day 16**: revisions + reproducibility package:
  ```bash
  tar czf submission_artifact.tar.gz \
    scripts/env/Dockerfile.llvm scripts/env/docker-compose.yml \
    scripts/env/requirements.txt \
    grammar/ decoder/ eval/ data/pipelines/ train/ scripts/ \
    RUNBOOK.md docs/ \
    results/frozen/ \
    checkpoints/phi-3.5-mini-lora/adapters/
  ```
- **Day 17**: submit.

---

## §10. Common issues

- **`docker exec` hangs**: the long-lived container may have died. `docker compose -f scripts/env/docker-compose.yml up -d` to restart.
- **MLX OOM**: drop `--max-tokens` or set `MLX_METAL_BUFFER_CACHE_LIMIT=0`.
- **Ollama slow first request**: first invocation loads the 20-25GB model into Metal; expect a 60-90s warmup. Subsequent calls are fast.
- **Grammar rejects real MLIR in L3**: usually an attribute or compound-type shape we didn't cover. Add a production; re-run `pytest grammar/tests/`.
- **mlir-opt --verify says "unregistered dialect"**: verify which dialects the pinned LLVM commit includes; bump the ADR if you need a new one.

---

## §11. Reference — which file does what

```
scripts/env/              Docker + venv env
  Dockerfile.llvm           pinned LLVM/MLIR/Polygeist image
  docker-compose.yml        long-lived container
  bin/*                     host wrappers (docker exec passthrough)
  requirements.txt          venv pins
  verify_cache.py           sqlite cache for mlir-opt --verify

grammar/                  C1 + C2 grammar + lattice
  mlir.lark                 LARK grammar (arith+func+memref v0)
  parser.py                 load_parser / parse_mlir / is_parse_valid
  ods_parse.py              .td file parser (shared with L0)
  ods_lattice.py            per-op type + arity extraction
  lattices/*.json           generated; not checked in

decoder/                  constrained decoding
  generate.py               unified generate() API
  c1_cfg.py                 MLX-LM + Outlines CFG guide; Ollama rejection sampler
  c2_type_arity.py          type-prefix automaton + arity state machine

data/pipelines/           data miners
  l0_ods/mine.py            ODS → (NL, signature) pairs + lattices
  l1_polygeist/driver.py    C → MLIR via Polygeist (+ clang fallback)
  l3_tests/mine.py          mlir/test/*.mlir → (weak-NL, MLIR) pairs

eval/                     eval + stats
  harness.py                (model × constraint × dialect × seed) grid runner
  stats.py                  bootstrap + paired bootstrap + MDE
  entropy.py                per-step entropy per (dialect, constraint)
  error_categories.py       verify-stderr → {type, arity, ...}
  baselines/run_ollama.py   Ollama REST wrapper
  ablations/hcs_replication.py  HCS reversal test
  benchmarks/mlir_spec_150/     hand-authored gold benchmark + validator

train/lora_phi.py         LoRA appendix runner

scripts/
  tokenizer_shootout.py     day-2 3-way tokenizer comparison
  make_figures.py           paper figures + LaTeX tables from results/*.json
```
