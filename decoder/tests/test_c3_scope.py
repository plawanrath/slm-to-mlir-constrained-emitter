"""Unit tests for decoder.c3_scope.

Run: pytest decoder/tests/test_c3_scope.py -q
"""
from __future__ import annotations

import textwrap

from decoder.c3_scope import (
    _memref_element_type,
    accept_or_reject,
    summarize_violations,
    validate,
)


# ---------------- passing cases ----------------

def test_simple_addi_passes():
    src = textwrap.dedent("""\
        module {
          func.func @add(%a: i32, %b: i32) -> i32 {
            %0 = arith.addi %a, %b : i32
            return %0 : i32
          }
        }
    """)
    rep = validate(src)
    assert rep.passed, rep.violations


def test_chain_passes():
    src = textwrap.dedent("""\
        module {
          func.func @fma(%a: i32, %b: i32, %c: i32) -> i32 {
            %0 = arith.addi %a, %b : i32
            %1 = arith.muli %0, %c : i32
            return %1 : i32
          }
        }
    """)
    rep = validate(src)
    assert rep.passed, rep.violations


def test_memref_load_passes():
    src = textwrap.dedent("""\
        module {
          func.func @load(%m: memref<?xf32>, %i: index) -> f32 {
            %0 = memref.load %m[%i] : memref<?xf32>
            return %0 : f32
          }
        }
    """)
    rep = validate(src)
    assert rep.passed, rep.violations


def test_cast_chain_passes():
    # i32 -> i64 via extsi, then used as i64.
    src = textwrap.dedent("""\
        module {
          func.func @widen(%a: i32) -> i64 {
            %0 = arith.extsi %a : i32 to i64
            return %0 : i64
          }
        }
    """)
    rep = validate(src)
    assert rep.passed, rep.violations


def test_cmp_select_passes():
    src = textwrap.dedent("""\
        module {
          func.func @clamp(%a: i32, %b: i32) -> i32 {
            %0 = arith.cmpi slt, %a, %b : i32
            %1 = arith.select %0, %a, %b : i32
            return %1 : i32
          }
        }
    """)
    rep = validate(src)
    assert rep.passed, rep.violations


def test_const_and_addf_passes():
    src = textwrap.dedent("""\
        module {
          func.func @addc(%a: f32) -> f32 {
            %0 = arith.constant 1.0 : f32
            %1 = arith.addf %a, %0 : f32
            return %1 : f32
          }
        }
    """)
    rep = validate(src)
    assert rep.passed, rep.violations


# ---------------- failing cases: undefined use ----------------

def test_undef_use_rejected():
    src = textwrap.dedent("""\
        module {
          func.func @bad(%a: i32) -> i32 {
            %0 = arith.addi %a, %b : i32
            return %0 : i32
          }
        }
    """)
    rep = validate(src)
    assert not rep.passed
    kinds = summarize_violations(rep.violations)
    assert kinds.get("undef_use", 0) >= 1


def test_use_before_def_rejected():
    # %1 used on line 1, defined on line 2.
    src = textwrap.dedent("""\
        module {
          func.func @bad(%a: i32) -> i32 {
            %0 = arith.addi %a, %1 : i32
            %1 = arith.constant 7 : i32
            return %0 : i32
          }
        }
    """)
    rep = validate(src)
    assert not rep.passed


# ---------------- failing cases: type mismatch ----------------

def test_param_type_mismatch_rejected():
    # Param %b is f32 but used under : i32.
    src = textwrap.dedent("""\
        module {
          func.func @bad(%a: i32, %b: f32) -> i32 {
            %0 = arith.addi %a, %b : i32
            return %0 : i32
          }
        }
    """)
    rep = validate(src)
    assert not rep.passed
    kinds = summarize_violations(rep.violations)
    assert kinds.get("type_mismatch", 0) >= 1


def test_defined_value_type_mismatch_rejected():
    # %0 defined as f32, used in an i32 op.
    src = textwrap.dedent("""\
        module {
          func.func @bad(%a: f32) -> i32 {
            %0 = arith.addf %a, %a : f32
            %1 = arith.addi %0, %0 : i32
            return %1 : i32
          }
        }
    """)
    rep = validate(src)
    assert not rep.passed


def test_memref_store_type_mismatch_rejected():
    # %v is f32 but stored into a memref<?xi32>.
    src = textwrap.dedent("""\
        module {
          func.func @bad(%m: memref<?xi32>, %i: index, %v: f32) {
            memref.store %v, %m[%i] : memref<?xi32>
            return
          }
        }
    """)
    rep = validate(src)
    assert not rep.passed


def test_cmp_result_is_i1():
    # %0 : i1 from cmpi; reusing it under : i32 must fail.
    src = textwrap.dedent("""\
        module {
          func.func @bad(%a: i32, %b: i32) -> i32 {
            %0 = arith.cmpi slt, %a, %b : i32
            %1 = arith.addi %0, %a : i32
            return %1 : i32
          }
        }
    """)
    rep = validate(src)
    assert not rep.passed


# ---------------- element-type helper ----------------

def test_memref_element_type_1d():
    assert _memref_element_type("memref<?xf32>") == "f32"
    assert _memref_element_type("memref<4x8xi32>") == "i32"


def test_memref_element_type_0d():
    # Scalar memref has no x-separator.
    assert _memref_element_type("memref<f64>") == "f64"


def test_memref_element_type_nested():
    # Element type itself contains < > — e.g., tuple or vector element.
    # Our walker goes right-to-left on top-level 'x' boundaries.
    assert _memref_element_type("memref<4xvector<8xf32>>") == "vector<8xf32>"


# ---------------- multiple functions ----------------

def test_scope_does_not_leak_between_functions():
    # %x defined in @f1; used without definition in @f2 — must be rejected.
    src = textwrap.dedent("""\
        module {
          func.func @f1(%a: i32) -> i32 {
            %x = arith.addi %a, %a : i32
            return %x : i32
          }
          func.func @f2(%a: i32) -> i32 {
            %0 = arith.addi %a, %x : i32
            return %0 : i32
          }
        }
    """)
    rep = validate(src)
    assert not rep.passed


# ---------------- accept_or_reject wrapper ----------------

def test_accept_or_reject_true_for_good():
    src = "module { func.func @f(%a: i32) -> i32 { return %a : i32 } }"
    ok, _ = accept_or_reject(src)
    assert ok


def test_accept_or_reject_false_for_undef():
    src = "module { func.func @f(%a: i32) -> i32 { return %b : i32 } }"
    ok, rep = accept_or_reject(src)
    assert not ok
    assert rep.violations


# ---------------- no-func abstain ----------------

def test_no_func_returns_passed():
    # Abstain when we can't find any func.func — we don't want to reject
    # generations that produce top-level attribute/type aliases only.
    rep = validate("#map = affine_map<(d0) -> (d0)>\n")
    assert rep.passed


# ---------------- linalg named ops (ADR-0007 scope) ----------------

def test_linalg_matmul_passes():
    src = textwrap.dedent("""\
        module {
          func.func @m(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
            linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>) outs(%C : memref<?x?xf32>)
            return
          }
        }
    """)
    rep = validate(src)
    assert rep.passed, rep.violations


def test_linalg_fill_passes():
    src = textwrap.dedent("""\
        module {
          func.func @f(%v: f32, %m: memref<?xf32>) {
            linalg.fill ins(%v : f32) outs(%m : memref<?xf32>)
            return
          }
        }
    """)
    rep = validate(src)
    assert rep.passed, rep.violations


def test_linalg_copy_passes():
    src = textwrap.dedent("""\
        module {
          func.func @c(%s: memref<?xf32>, %d: memref<?xf32>) {
            linalg.copy ins(%s : memref<?xf32>) outs(%d : memref<?xf32>)
            return
          }
        }
    """)
    rep = validate(src)
    assert rep.passed, rep.violations


def test_linalg_transpose_passes():
    src = textwrap.dedent("""\
        module {
          func.func @t(%s: memref<?x?xf32>, %d: memref<?x?xf32>) {
            linalg.transpose ins(%s : memref<?x?xf32>) outs(%d : memref<?x?xf32>) permutation = [1, 0]
            return
          }
        }
    """)
    rep = validate(src)
    assert rep.passed, rep.violations


def test_linalg_elemwise_bin_passes():
    for op in ("add", "sub", "mul", "div"):
        src = textwrap.dedent(f"""\
            module {{
              func.func @f(%a: memref<?xf32>, %b: memref<?xf32>, %c: memref<?xf32>) {{
                linalg.{op} ins(%a, %b : memref<?xf32>, memref<?xf32>) outs(%c : memref<?xf32>)
                return
              }}
            }}
        """)
        rep = validate(src)
        assert rep.passed, (op, rep.violations)


def test_linalg_elemwise_un_passes():
    for op in ("exp", "abs"):
        src = textwrap.dedent(f"""\
            module {{
              func.func @f(%a: memref<?xf32>, %b: memref<?xf32>) {{
                linalg.{op} ins(%a : memref<?xf32>) outs(%b : memref<?xf32>)
                return
              }}
            }}
        """)
        rep = validate(src)
        assert rep.passed, (op, rep.violations)


def test_linalg_undef_ssa_rejected():
    # %Z is not a parameter, so ins(%A, %Z ...) should be flagged.
    src = textwrap.dedent("""\
        module {
          func.func @m(%A: memref<?x?xf32>, %C: memref<?x?xf32>) {
            linalg.matmul ins(%A, %Z : memref<?x?xf32>, memref<?x?xf32>) outs(%C : memref<?x?xf32>)
            return
          }
        }
    """)
    rep = validate(src)
    assert not rep.passed
    assert any(v.kind == "undef_use" for v in rep.violations)


def test_linalg_type_mismatch_rejected():
    # %A is memref<?x?xf32> in params; the op declares it as memref<?x?xi32>.
    src = textwrap.dedent("""\
        module {
          func.func @m(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
            linalg.matmul ins(%A, %B : memref<?x?xi32>, memref<?x?xf32>) outs(%C : memref<?x?xf32>)
            return
          }
        }
    """)
    rep = validate(src)
    assert not rep.passed
    assert any(v.kind == "type_mismatch" for v in rep.violations)


def test_linalg_fill_scalar_memref_element_mismatch_rejected():
    # %v is f32 scalar; %m is memref<?xi32> — element-type mismatch should fire.
    src = textwrap.dedent("""\
        module {
          func.func @f(%v: f32, %m: memref<?xi32>) {
            linalg.fill ins(%v : f32) outs(%m : memref<?xi32>)
            return
          }
        }
    """)
    rep = validate(src)
    assert not rep.passed
    assert any(v.kind == "type_mismatch" for v in rep.violations)


def test_linalg_multiline_parses():
    # ins/outs clauses split across newlines (the gen grammar permits this).
    src = textwrap.dedent("""\
        module {
          func.func @m(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
            linalg.matmul ins(%A, %B : memref<?x?xf32>,
              memref<?x?xf32>) outs(%C :
              memref<?x?xf32>)
            return
          }
        }
    """)
    rep = validate(src)
    assert rep.passed, rep.violations


def test_extract_clause_helper():
    from decoder.c3_scope import _extract_clause
    line = "linalg.add ins(%a, %b : memref<?xf32>, memref<?xf32>) outs(%c : memref<?xf32>)"
    assert _extract_clause(line, "ins")  == "%a, %b : memref<?xf32>, memref<?xf32>"
    assert _extract_clause(line, "outs") == "%c : memref<?xf32>"


def test_parse_linalg_clause_two_ins():
    from decoder.c3_scope import _parse_linalg_clause
    pairs = _parse_linalg_clause("%a, %b : memref<?xf32>, memref<?xf32>")
    assert pairs == [("a", "memref<?xf32>"), ("b", "memref<?xf32>")]
