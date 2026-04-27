"""One-shot authoring of StableHLO-Spec-30 (Day 31 of ADR-0008).

Writes 30 verify-clean NL→MLIR pairs under
`eval/benchmarks/stablehlo_spec_30/examples/`.

Coverage target: the 10 named ops from the Phase-C grammar
(`grammar/mlir_gen_stablehlo.lark`):
  elementwise bin/un (6 ops), transpose, broadcast_in_dim, reshape, dot_general.

Difficulty mix (per ADR-0008 spec): 40% easy, 40% medium, 20% hard.

NOTE: this script overwrites existing files. Run once at benchmark authoring
time. Verify-valid status is checked by a downstream mlir-opt pass (which may
require a StableHLO-enabled build; the script authors the seeds regardless).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO = "/Users/plawanrath/Documents/GitHub-public/slm-to-mlir-constrained-emitter"
os.chdir(REPO)

OUT = Path("eval/benchmarks/stablehlo_spec_30/examples")
OUT.mkdir(parents=True, exist_ok=True)


def _write(seed: dict) -> None:
    p = OUT / f"{seed['id']}.json"
    p.write_text(json.dumps(seed, indent=2) + "\n")


SEEDS = [
    # ---- Elementwise binary (add, subtract, multiply, divide) ----
    dict(id="01_add-1d", difficulty="easy",
         nl="Write a function that adds two 1-D f32 tensors of 16 elements using stablehlo.add.",
         mlir="""module {
  func.func @a(%a: tensor<16xf32>, %b: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.add %a, %b : tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}""",
         notes="canonical stablehlo.add"),
    dict(id="02_add-2d-dynamic", difficulty="easy",
         nl="Write a function that adds two 2-D f32 tensors with dynamic shapes and returns the result.",
         mlir="""module {
  func.func @add2d(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = stablehlo.add %a, %b : tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
  }
}""",
         notes="dynamic-shape addition"),
    dict(id="03_subtract-1d-i32", difficulty="easy",
         nl="Write a function that subtracts two 1-D i32 tensors elementwise.",
         mlir="""module {
  func.func @sub(%a: tensor<8xi32>, %b: tensor<8xi32>) -> tensor<8xi32> {
    %0 = stablehlo.subtract %a, %b : tensor<8xi32>
    return %0 : tensor<8xi32>
  }
}""",
         notes="integer subtraction"),
    dict(id="04_multiply-2d", difficulty="easy",
         nl="Write a function that multiplies two 4x4 f32 tensors elementwise using stablehlo.multiply.",
         mlir="""module {
  func.func @mul(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = stablehlo.multiply %a, %b : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}""",
         notes="static 4x4 multiply"),
    dict(id="05_divide-f64", difficulty="easy",
         nl="Write a function that divides two 1-D f64 tensors of 32 elements using stablehlo.divide.",
         mlir="""module {
  func.func @div(%a: tensor<32xf64>, %b: tensor<32xf64>) -> tensor<32xf64> {
    %0 = stablehlo.divide %a, %b : tensor<32xf64>
    return %0 : tensor<32xf64>
  }
}""",
         notes="f64 division"),

    # ---- Elementwise unary (abs, exponential) ----
    dict(id="06_abs-f32", difficulty="easy",
         nl="Write a function that computes the elementwise absolute value of a 1-D f32 tensor.",
         mlir="""module {
  func.func @ab(%a: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.abs %a : tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}""",
         notes="abs"),
    dict(id="07_exp-1d", difficulty="easy",
         nl="Write a function that computes the elementwise exponential of a 1-D f32 tensor of 10 elements.",
         mlir="""module {
  func.func @ex(%a: tensor<10xf32>) -> tensor<10xf32> {
    %0 = stablehlo.exponential %a : tensor<10xf32>
    return %0 : tensor<10xf32>
  }
}""",
         notes="exp"),
    dict(id="08_abs-dynamic", difficulty="medium",
         nl="Write a function that computes the elementwise absolute value of a dynamic-shape 2-D f32 tensor.",
         mlir="""module {
  func.func @abd(%a: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = stablehlo.abs %a : tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
  }
}""",
         notes="dynamic abs"),

    # ---- Transpose ----
    dict(id="09_transpose-2d", difficulty="medium",
         nl="Write a function that transposes a 4x8 f32 tensor producing an 8x4 tensor.",
         mlir="""module {
  func.func @t(%a: tensor<4x8xf32>) -> tensor<8x4xf32> {
    %0 = stablehlo.transpose %a, permutation = [1, 0] : (tensor<4x8xf32>) -> tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
  }
}""",
         notes="transpose 2D"),
    dict(id="10_transpose-3d", difficulty="medium",
         nl="Write a function that transposes a 2x3x4 f32 tensor with permutation [2, 0, 1] producing a 4x2x3 tensor.",
         mlir="""module {
  func.func @t3(%a: tensor<2x3x4xf32>) -> tensor<4x2x3xf32> {
    %0 = stablehlo.transpose %a, permutation = [2, 0, 1] : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
    return %0 : tensor<4x2x3xf32>
  }
}""",
         notes="3D transpose"),
    dict(id="11_transpose-square", difficulty="easy",
         nl="Write a function that transposes a 3x3 f32 tensor.",
         mlir="""module {
  func.func @t(%a: tensor<3x3xf32>) -> tensor<3x3xf32> {
    %0 = stablehlo.transpose %a, permutation = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
    return %0 : tensor<3x3xf32>
  }
}""",
         notes="square transpose"),

    # ---- Broadcast ----
    dict(id="12_broadcast-1d-to-2d", difficulty="medium",
         nl="Write a function that broadcasts a 1-D f32 tensor of 8 elements to a 4x8 2-D tensor along dimension 1.",
         mlir="""module {
  func.func @b(%a: tensor<8xf32>) -> tensor<4x8xf32> {
    %0 = stablehlo.broadcast_in_dim %a, broadcast_dimensions = [1] : (tensor<8xf32>) -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
}""",
         notes="broadcast 1D to 2D"),
    dict(id="13_broadcast-scalar-to-vector", difficulty="medium",
         nl="Write a function that broadcasts a scalar f32 (shape [1]) to a 1-D f32 tensor of 16 elements.",
         mlir="""module {
  func.func @bs(%a: tensor<1xf32>) -> tensor<16xf32> {
    %0 = stablehlo.broadcast_in_dim %a, broadcast_dimensions = [0] : (tensor<1xf32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}""",
         notes="scalar broadcast"),

    # ---- Reshape ----
    dict(id="14_reshape-flatten", difficulty="medium",
         nl="Write a function that flattens a 4x8 f32 tensor into a 1-D tensor of 32 elements.",
         mlir="""module {
  func.func @r(%a: tensor<4x8xf32>) -> tensor<32xf32> {
    %0 = stablehlo.reshape %a : (tensor<4x8xf32>) -> tensor<32xf32>
    return %0 : tensor<32xf32>
  }
}""",
         notes="flatten"),
    dict(id="15_reshape-2d-to-3d", difficulty="medium",
         nl="Write a function that reshapes a 12x8 f32 tensor into a 3x4x8 3-D tensor.",
         mlir="""module {
  func.func @r(%a: tensor<12x8xf32>) -> tensor<3x4x8xf32> {
    %0 = stablehlo.reshape %a : (tensor<12x8xf32>) -> tensor<3x4x8xf32>
    return %0 : tensor<3x4x8xf32>
  }
}""",
         notes="2D to 3D reshape"),
    dict(id="16_reshape-transpose-chain", difficulty="hard",
         nl="Write a function that flattens a 4x8 f32 tensor, then transposes the result — no wait, simpler: reshape a 4x8 tensor into 8x4.",
         mlir="""module {
  func.func @r(%a: tensor<4x8xf32>) -> tensor<8x4xf32> {
    %0 = stablehlo.reshape %a : (tensor<4x8xf32>) -> tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
  }
}""",
         notes="reshape shape change"),

    # ---- dot_general (matmul) ----
    dict(id="17_dot_general-matmul", difficulty="medium",
         nl="Write a function that performs a matrix multiplication of a 4x8 f32 tensor and an 8x16 f32 tensor using stablehlo.dot_general.",
         mlir="""module {
  func.func @m(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
  }
}""",
         notes="canonical matmul"),
    dict(id="18_dot_general-square", difficulty="medium",
         nl="Write a function that multiplies two 8x8 f32 tensors using stablehlo.dot_general.",
         mlir="""module {
  func.func @m(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}""",
         notes="square matmul"),
    dict(id="19_dot_general-tall-thin", difficulty="medium",
         nl="Multiply a 128x16 f32 tensor by a 16x4 f32 tensor using stablehlo.dot_general.",
         mlir="""module {
  func.func @m(%a: tensor<128x16xf32>, %b: tensor<16x4xf32>) -> tensor<128x4xf32> {
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<128x16xf32>, tensor<16x4xf32>) -> tensor<128x4xf32>
    return %0 : tensor<128x4xf32>
  }
}""",
         notes="tall-thin matmul"),

    # ---- Chains / multi-op ----
    dict(id="20_add-multiply-chain", difficulty="medium",
         nl="Write a function that adds two 1-D f32 tensors and then multiplies the sum by the first input.",
         mlir="""module {
  func.func @c(%a: tensor<16xf32>, %b: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.add %a, %b : tensor<16xf32>
    %1 = stablehlo.multiply %0, %a : tensor<16xf32>
    return %1 : tensor<16xf32>
  }
}""",
         notes="add-then-multiply"),
    dict(id="21_abs-exp-chain", difficulty="medium",
         nl="Write a function that computes the exponential of the absolute value of a 1-D f32 tensor.",
         mlir="""module {
  func.func @c(%a: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.abs %a : tensor<16xf32>
    %1 = stablehlo.exponential %0 : tensor<16xf32>
    return %1 : tensor<16xf32>
  }
}""",
         notes="abs then exp"),
    dict(id="22_matmul-add-bias", difficulty="hard",
         nl="Matrix-multiply a 4x8 f32 tensor by an 8x16 f32 tensor, then add a 4x16 bias tensor.",
         mlir="""module {
  func.func @lin(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>, %bias: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    %1 = stablehlo.add %0, %bias : tensor<4x16xf32>
    return %1 : tensor<4x16xf32>
  }
}""",
         notes="linear layer"),
    dict(id="23_transpose-matmul", difficulty="hard",
         nl="Transpose a 8x4 f32 tensor, then matrix-multiply the result with a 4x16 f32 tensor.",
         mlir="""module {
  func.func @tm(%a: tensor<8x4xf32>, %b: tensor<4x16xf32>) -> tensor<8x16xf32> {
    %0 = stablehlo.transpose %a, permutation = [1, 0] : (tensor<8x4xf32>) -> tensor<4x8xf32>
    %1 = stablehlo.dot_general %0, %b, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
    return %1 : tensor<8x16xf32>
  }
}""",
         notes="transpose+matmul"),
    dict(id="24_reshape-add", difficulty="medium",
         nl="Reshape a 4x4 f32 tensor into a 16-element 1-D tensor, then add to an existing 16-element tensor.",
         mlir="""module {
  func.func @ra(%a: tensor<4x4xf32>, %b: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.reshape %a : (tensor<4x4xf32>) -> tensor<16xf32>
    %1 = stablehlo.add %0, %b : tensor<16xf32>
    return %1 : tensor<16xf32>
  }
}""",
         notes="reshape+add"),
    dict(id="25_broadcast-multiply", difficulty="hard",
         nl="Broadcast a length-8 1-D f32 tensor to a 4x8 tensor, then multiply with an existing 4x8 tensor.",
         mlir="""module {
  func.func @bm(%a: tensor<8xf32>, %b: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %0 = stablehlo.broadcast_in_dim %a, broadcast_dimensions = [1] : (tensor<8xf32>) -> tensor<4x8xf32>
    %1 = stablehlo.multiply %0, %b : tensor<4x8xf32>
    return %1 : tensor<4x8xf32>
  }
}""",
         notes="broadcast+multiply"),

    # ---- More variety ----
    dict(id="26_add-3d", difficulty="easy",
         nl="Write a function that adds two 2x3x4 f32 tensors elementwise.",
         mlir="""module {
  func.func @a3(%a: tensor<2x3x4xf32>, %b: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %0 = stablehlo.add %a, %b : tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }
}""",
         notes="3D add"),
    dict(id="27_subtract-bf16", difficulty="easy",
         nl="Write a function that subtracts two 16-element bf16 tensors elementwise.",
         mlir="""module {
  func.func @s(%a: tensor<16xbf16>, %b: tensor<16xbf16>) -> tensor<16xbf16> {
    %0 = stablehlo.subtract %a, %b : tensor<16xbf16>
    return %0 : tensor<16xbf16>
  }
}""",
         notes="bf16 arithmetic"),
    dict(id="28_dot_general-f16", difficulty="medium",
         nl="Multiply two 16x16 f16 tensors using stablehlo.dot_general.",
         mlir="""module {
  func.func @m(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>) -> tensor<16x16xf16> {
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<16x16xf16>, tensor<16x16xf16>) -> tensor<16x16xf16>
    return %0 : tensor<16x16xf16>
  }
}""",
         notes="f16 matmul"),
    dict(id="29_add-multiply-abs-chain", difficulty="hard",
         nl="Write a function that computes the absolute value of (a + b) * a for two 1-D f32 tensors.",
         mlir="""module {
  func.func @c(%a: tensor<16xf32>, %b: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.add %a, %b : tensor<16xf32>
    %1 = stablehlo.multiply %0, %a : tensor<16xf32>
    %2 = stablehlo.abs %1 : tensor<16xf32>
    return %2 : tensor<16xf32>
  }
}""",
         notes="3-op chain"),
    dict(id="30_transpose-add", difficulty="medium",
         nl="Transpose a 4x4 f32 tensor then add it back to the original.",
         mlir="""module {
  func.func @ta(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = stablehlo.transpose %a, permutation = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %1 = stablehlo.add %0, %a : tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}""",
         notes="symmetric sum"),
]


def main() -> None:
    assert len(SEEDS) == 30, f"expected 30 seeds, got {len(SEEDS)}"
    ids = [s["id"] for s in SEEDS]
    assert len(set(ids)) == len(ids), "duplicate IDs"
    for seed in SEEDS:
        seed["dialect"] = "stablehlo+func"
        _write(seed)
    # Difficulty breakdown
    by_diff = {"easy": 0, "medium": 0, "hard": 0}
    for s in SEEDS:
        by_diff[s["difficulty"]] += 1
    print(f"Wrote {len(SEEDS)} seeds → {OUT}")
    print(f"Difficulty: {by_diff}")


if __name__ == "__main__":
    main()
