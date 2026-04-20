"""LoRA fine-tune of Phi-3.5-mini via MLX-LM (appendix ablation only).

Design doc §5.3 item 3: demonstrates the training-free claim's robustness by
showing that training adds a small incremental lift over C1+C2 inference.

Usage:
    python -m train.lora_phi \
        --base-model microsoft/Phi-3.5-mini-instruct \
        --train-data data/processed/l1_polygeist_train.jsonl \
        --val-data   data/processed/l1_polygeist_val.jsonl \
        --out checkpoints/phi-3.5-mini-lora \
        --rank 16 --iters 2000

This is a single-shot wrapper around the `mlx_lm.lora` CLI. We keep it in-repo
so the exact hyperparams used in the paper appendix are version-controlled.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _prep_jsonl(records_path: Path, out_path: Path) -> int:
    """Convert our (docstring, mlir) JSONL into the mlx-lm LoRA format.

    mlx-lm expects `{"prompt": "...", "completion": "..."}` per line.
    """
    n = 0
    with records_path.open() as src, out_path.open("w") as dst:
        for line in src:
            rec = json.loads(line)
            prompt = rec.get("docstring") or rec.get("weak_nl") or rec.get("nl") or ""
            completion = rec.get("mlir") or rec.get("op_signature") or ""
            if not prompt or not completion:
                continue
            dst.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--train-data", type=Path, required=True)
    ap.add_argument("--val-data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stage = args.out / "stage"
    stage.mkdir(exist_ok=True)
    train_jsonl = stage / "train.jsonl"
    val_jsonl = stage / "valid.jsonl"
    n_train = _prep_jsonl(args.train_data, train_jsonl)
    n_val = _prep_jsonl(args.val_data, val_jsonl)
    print(f"[lora] train={n_train} val={n_val}", file=sys.stderr)

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", args.base_model,
        "--data", str(stage),
        "--train",
        "--iters", str(args.iters),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.lr),
        "--lora-layers", "16",
        "--adapter-path", str(args.out / "adapters"),
    ]
    print(f"[lora] running: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
