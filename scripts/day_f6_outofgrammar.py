"""Phase F (F6): out-of-grammar StableHLO held-out corpus.

The Phase-E held-out-50 corpus sampled the SAME 10 ops that our
grammar covers (add/subtract/multiply/divide/abs/exponential/transpose/
broadcast_in_dim/reshape/dot_general). A reviewer-grade concern is
``that's a coverage tautology --- your grammar was authored for those
ops, so of course constraints hit 100\%.''

This script authors 25 StableHLO programs using ops OUTSIDE our
10-op grammar scope:
  - stablehlo.reduce (reductions)
  - stablehlo.convolution (N-D convs)
  - stablehlo.pad (with edge_padding)
  - stablehlo.dynamic_slice / stablehlo.dynamic_update_slice
  - stablehlo.iota (index-generating)
  - stablehlo.clamp (3-operand)
  - stablehlo.select (3-operand)
  - stablehlo.compare (with comparison_direction)
  - stablehlo.scatter / stablehlo.gather (indirect access)
  - stablehlo.while (control flow)
  - stablehlo.concatenate

Expected failure mode:
  - C1 grammar: 0% verify (grammar doesn't accept these ops; the
    model CAN'T emit them regardless of sampling)
  - free decoding: whatever the model naturally produces (variable)

The point is to document *graceful degradation*: the constraint
stack transparently refuses to emit programs outside its declared
scope, rather than silently emitting malformed ops.

Output: results/day_f6/outofgrammar.jsonl + summary.json
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

REPO = "/Users/plawanrath/Documents/GitHub-public/slm-to-mlir-constrained-emitter"
os.chdir(REPO); sys.path.insert(0, REPO)
os.environ["PATH"] = f"{REPO}/scripts/env/bin:" + os.environ["PATH"]

from scripts.env.verify_stablehlo import verify_stablehlo

OUT = Path("eval/benchmarks/stablehlo_outofgrammar_25/examples")
OUT.mkdir(parents=True, exist_ok=True)


def _prog(id_, nl, mlir, op):
    return {"id": id_, "nl": nl, "mlir": mlir, "op": op,
            "dialect": "stablehlo+func", "source": "outofgrammar_phase_f"}


PROGRAMS = [
    # reduce
    _prog("01_reduce_sum_1d",
          "Sum a 1-D f32 tensor of 16 elements using stablehlo.reduce with an add body.",
          """module {
  func.func @f(%a: tensor<16xf32>) -> tensor<f32> {
    %init = stablehlo.constant dense<0.0> : tensor<f32>
    %0 = stablehlo.reduce(%a init: %init) across dimensions = [0] : (tensor<16xf32>, tensor<f32>) -> tensor<f32>
      reducer(%arg0: tensor<f32>, %arg1: tensor<f32>) {
        %sum = stablehlo.add %arg0, %arg1 : tensor<f32>
        stablehlo.return %sum : tensor<f32>
      }
    return %0 : tensor<f32>
  }
}""", "reduce"),
    # convolution
    _prog("02_conv_2d",
          "Apply a 2-D convolution of an input tensor 1x4x4x3 (NHWC) with a filter 3x3x3x8.",
          """module {
  func.func @f(%a: tensor<1x4x4x3xf32>, %w: tensor<3x3x3x8xf32>) -> tensor<1x2x2x8xf32> {
    %0 = stablehlo.convolution(%a, %w)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
      : (tensor<1x4x4x3xf32>, tensor<3x3x3x8xf32>) -> tensor<1x2x2x8xf32>
    return %0 : tensor<1x2x2x8xf32>
  }
}""", "convolution"),
    # pad
    _prog("03_pad_2d",
          "Pad a 4x4 f32 tensor with 1-pixel borders of zero on each side, producing 6x6.",
          """module {
  func.func @f(%a: tensor<4x4xf32>) -> tensor<6x6xf32> {
    %pad = stablehlo.constant dense<0.0> : tensor<f32>
    %0 = stablehlo.pad %a, %pad, low = [1, 1], high = [1, 1], interior = [0, 0]
      : (tensor<4x4xf32>, tensor<f32>) -> tensor<6x6xf32>
    return %0 : tensor<6x6xf32>
  }
}""", "pad"),
    # dynamic_slice
    _prog("04_dynamic_slice_1d",
          "Slice 8 elements from a 1-D f32 tensor of 32 starting at dynamic index.",
          """module {
  func.func @f(%a: tensor<32xf32>, %i: tensor<i32>) -> tensor<8xf32> {
    %0 = stablehlo.dynamic_slice %a, %i, sizes = [8] : (tensor<32xf32>, tensor<i32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}""", "dynamic_slice"),
    # iota
    _prog("05_iota_1d",
          "Produce a 1-D i32 iota tensor of 16 elements (values 0..15).",
          """module {
  func.func @f() -> tensor<16xi32> {
    %0 = stablehlo.iota dim = 0 : tensor<16xi32>
    return %0 : tensor<16xi32>
  }
}""", "iota"),
    # clamp
    _prog("06_clamp_f32",
          "Clamp a 1-D f32 tensor of 16 elements to the [0, 1] range.",
          """module {
  func.func @f(%a: tensor<16xf32>) -> tensor<16xf32> {
    %lo = stablehlo.constant dense<0.0> : tensor<16xf32>
    %hi = stablehlo.constant dense<1.0> : tensor<16xf32>
    %0 = stablehlo.clamp %lo, %a, %hi : tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}""", "clamp"),
    # select
    _prog("07_select_f32",
          "Elementwise select between two f32 tensors of shape 16 based on a bool mask.",
          """module {
  func.func @f(%p: tensor<16xi1>, %x: tensor<16xf32>, %y: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.select %p, %x, %y : (tensor<16xi1>, tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}""", "select"),
    # compare
    _prog("08_compare_lt",
          "Elementwise less-than comparison of two f32 tensors of shape 16.",
          """module {
  func.func @f(%a: tensor<16xf32>, %b: tensor<16xf32>) -> tensor<16xi1> {
    %0 = stablehlo.compare LT, %a, %b, FLOAT : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xi1>
    return %0 : tensor<16xi1>
  }
}""", "compare"),
    # concatenate
    _prog("09_concatenate",
          "Concatenate two 4x4 f32 tensors along axis 0 to produce an 8x4 tensor.",
          """module {
  func.func @f(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<8x4xf32> {
    %0 = stablehlo.concatenate %a, %b, dim = 0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
  }
}""", "concatenate"),
    # reduce max
    _prog("10_reduce_max_2d",
          "Reduce max across rows of a 4x8 f32 tensor, producing a 4-element vector.",
          """module {
  func.func @f(%a: tensor<4x8xf32>) -> tensor<4xf32> {
    %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%a init: %init) across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
      reducer(%arg0: tensor<f32>, %arg1: tensor<f32>) {
        %m = stablehlo.maximum %arg0, %arg1 : tensor<f32>
        stablehlo.return %m : tensor<f32>
      }
    return %0 : tensor<4xf32>
  }
}""", "reduce"),
    # conv 1D
    _prog("11_conv_1d",
          "Apply a 1-D convolution to an input tensor of shape 1x16x4 (NWC) with filter 3x4x8.",
          """module {
  func.func @f(%a: tensor<1x16x4xf32>, %w: tensor<3x4x8xf32>) -> tensor<1x14x8xf32> {
    %0 = stablehlo.convolution(%a, %w)
      dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f],
      window = {stride = [1], pad = [[0, 0]]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
      : (tensor<1x16x4xf32>, tensor<3x4x8xf32>) -> tensor<1x14x8xf32>
    return %0 : tensor<1x14x8xf32>
  }
}""", "convolution"),
    # gather (simple)
    _prog("12_gather_simple",
          "Gather 4 rows from an 8x16 f32 tensor using a 4-element i32 index vector.",
          """module {
  func.func @f(%a: tensor<8x16xf32>, %idx: tensor<4x1xi32>) -> tensor<4x16xf32> {
    %0 = "stablehlo.gather"(%a, %idx) {
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1],
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1
      >,
      slice_sizes = array<i64: 1, 16>,
      indices_are_sorted = false
    } : (tensor<8x16xf32>, tensor<4x1xi32>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
  }
}""", "gather"),
    # slice (static)
    _prog("13_slice_static",
          "Take the first 4x4 block of an 8x8 f32 tensor using stablehlo.slice.",
          """module {
  func.func @f(%a: tensor<8x8xf32>) -> tensor<4x4xf32> {
    %0 = stablehlo.slice %a [0:4, 0:4] : (tensor<8x8xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}""", "slice"),
    # rsqrt (elementwise unary op not in our 10)
    _prog("14_rsqrt",
          "Compute elementwise reciprocal square root of a 16-element f32 tensor.",
          """module {
  func.func @f(%a: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.rsqrt %a : tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}""", "rsqrt"),
    # sign
    _prog("15_sign",
          "Compute elementwise sign of a 16-element f32 tensor.",
          """module {
  func.func @f(%a: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.sign %a : tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}""", "sign"),
    # tanh
    _prog("16_tanh",
          "Compute elementwise hyperbolic tangent of a 16-element f32 tensor.",
          """module {
  func.func @f(%a: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.tanh %a : tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}""", "tanh"),
    # batch norm inference (compound op not in our grammar)
    _prog("17_sort_1d",
          "Sort a 1-D f32 tensor of 16 elements in ascending order using stablehlo.sort.",
          """module {
  func.func @f(%a: tensor<16xf32>) -> tensor<16xf32> {
    %0 = "stablehlo.sort"(%a) ({
      ^bb0(%x: tensor<f32>, %y: tensor<f32>):
        %cmp = stablehlo.compare LT, %x, %y, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %cmp : tensor<i1>
    }) {dimension = 0 : i64, is_stable = true} : (tensor<16xf32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}""", "sort"),
    # power-of-element (pow)
    _prog("18_power",
          "Compute elementwise power a^b of two 16-element f32 tensors.",
          """module {
  func.func @f(%a: tensor<16xf32>, %b: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.power %a, %b : tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}""", "power"),
    # dynamic_update_slice
    _prog("19_dynamic_update_slice",
          "Update a 4-element slice of a 32-element f32 tensor at a dynamic index.",
          """module {
  func.func @f(%a: tensor<32xf32>, %u: tensor<4xf32>, %i: tensor<i32>) -> tensor<32xf32> {
    %0 = stablehlo.dynamic_update_slice %a, %u, %i : (tensor<32xf32>, tensor<4xf32>, tensor<i32>) -> tensor<32xf32>
    return %0 : tensor<32xf32>
  }
}""", "dynamic_update_slice"),
    # reverse
    _prog("20_reverse",
          "Reverse the order of elements in a 1-D f32 tensor of 16 elements.",
          """module {
  func.func @f(%a: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.reverse %a, dims = [0] : tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}""", "reverse"),
    # round_nearest_even
    _prog("21_round",
          "Round elementwise to nearest even integer on a 16-element f32 tensor.",
          """module {
  func.func @f(%a: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.round_nearest_even %a : tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}""", "round"),
    # is_finite
    _prog("22_is_finite",
          "Elementwise is_finite on a 16-element f32 tensor, returning a bool tensor.",
          """module {
  func.func @f(%a: tensor<16xf32>) -> tensor<16xi1> {
    %0 = stablehlo.is_finite %a : (tensor<16xf32>) -> tensor<16xi1>
    return %0 : tensor<16xi1>
  }
}""", "is_finite"),
    # shift_left (integer ops out of grammar)
    _prog("23_shift_left",
          "Elementwise left-shift of two 16-element i32 tensors.",
          """module {
  func.func @f(%a: tensor<16xi32>, %b: tensor<16xi32>) -> tensor<16xi32> {
    %0 = stablehlo.shift_left %a, %b : tensor<16xi32>
    return %0 : tensor<16xi32>
  }
}""", "shift_left"),
    # xor
    _prog("24_xor",
          "Elementwise bitwise xor of two 16-element i32 tensors.",
          """module {
  func.func @f(%a: tensor<16xi32>, %b: tensor<16xi32>) -> tensor<16xi32> {
    %0 = stablehlo.xor %a, %b : tensor<16xi32>
    return %0 : tensor<16xi32>
  }
}""", "xor"),
    # popcnt
    _prog("25_popcnt",
          "Elementwise population-count of a 16-element i32 tensor.",
          """module {
  func.func @f(%a: tensor<16xi32>) -> tensor<16xi32> {
    %0 = stablehlo.popcnt %a : tensor<16xi32>
    return %0 : tensor<16xi32>
  }
}""", "popcnt"),
]


def build():
    print(f"[f6] authoring {len(PROGRAMS)} out-of-grammar programs", file=sys.stderr)
    kept = []
    dropped = []
    for p in PROGRAMS:
        r = verify_stablehlo(p["mlir"])
        if r["returncode"] == 0:
            kept.append(p)
        else:
            dropped.append((p["id"], r["stderr"][:200]))
    print(f"[f6] verify-clean: {len(kept)}/{len(PROGRAMS)}", file=sys.stderr)
    for ex in OUT.glob("*.json"):
        ex.unlink()
    for i, p in enumerate(kept, start=1):
        (OUT / f"{i:02d}_{p['id']}.json").write_text(json.dumps(p, indent=2))
    summary = {"n_authored": len(PROGRAMS),
               "n_verify_clean": len(kept),
               "n_rejected_by_iree": len(dropped),
               "rejected_ids": [d[0] for d in dropped]}
    (Path("results/day_f6")).mkdir(parents=True, exist_ok=True)
    Path("results/day_f6/corpus_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[f6] wrote {len(kept)} examples to {OUT}", file=sys.stderr)
    print(f"[f6] rejected by iree-compile: {len(dropped)}", file=sys.stderr)
    for id_, err in dropped[:3]:
        print(f"  {id_}: {err[:120]}", file=sys.stderr)


if __name__ == "__main__":
    build()
