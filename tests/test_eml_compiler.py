"""Tests for ``eml_compiler`` (issue #30)."""

from __future__ import annotations

import math
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

import eml_compiler as C
from eml_compiler import Leaf, Node


REPO_ROOT = Path(__file__).parent.parent


# ─── Primitive identities (§3 / Table 4 of the paper) ──────────────
#
# Each identity has a canonical tree shape we can check by size/depth
# AND numerical equivalence to the reference python expression.

@pytest.mark.parametrize(
    "expr, bindings, py_fn, expected_size, expected_depth",
    [
        # (formula, bindings, numpy reference, canonical size, canonical depth)
        ("exp(x)",     {"x": 2.0},           lambda x: np.exp(x),    3,  1),
        ("ln(x)",      {"x": 2.0},           lambda x: np.log(x),    7,  3),
        ("log(x)",     {"x": 2.0},           lambda x: np.log(x),    7,  3),
        ("x - y",      {"x": 5.0, "y": 3.0}, lambda x, y: x - y,     11, 4),
        ("-x",         {"x": 2.0},           lambda x: -x,           11, 4),
        ("x + y",      {"x": 2.0, "y": 3.0}, lambda x, y: x + y,     21, 6),
        ("1 / x",      {"x": 4.0},           lambda x: 1 / x,        53, 14),
        ("x * y",      {"x": 2.0, "y": 3.0}, lambda x, y: x * y,     35, 8),
        ("x / y",      {"x": 6.0, "y": 4.0}, lambda x, y: x / y,     53, 14),
        ("x^y",        {"x": 2.0, "y": 3.0}, lambda x, y: x**y,      43, 12),
    ],
)
def test_primitive_sizes_and_values(expr, bindings, py_fn, expected_size, expected_depth):
    """Each primitive produces the canonical tree shape and evaluates correctly."""
    tree = C.compile_expr(expr)
    assert C.tree_size(tree) == expected_size
    assert C.tree_depth(tree) == expected_depth
    val = C.eval_eml(tree, bindings)
    expected = py_fn(**bindings)
    assert abs(val - expected) < 1e-10, f"{expr}: got {val}, want {expected}"


def test_sqrt():
    """sqrt is compiled as pow(a, 0.5)."""
    tree = C.compile_expr("sqrt(x)")
    v = C.eval_eml(tree, x=9.0)
    assert abs(v.real - 3.0) < 1e-9
    assert abs(v.imag) < 1e-9


def test_constants_e_and_pi():
    tree_e = C.compile_expr("e")
    v = C.eval_eml(tree_e)
    assert abs(v.real - math.e) < 1e-10
    # e = eml(1, 1) — size 3 depth 1
    assert C.tree_size(tree_e) == 3
    assert C.tree_depth(tree_e) == 1

    tree_pi = C.compile_expr("pi")
    v = C.eval_eml(tree_pi)
    assert abs(v.real - math.pi) < 1e-10
    # pi is embedded as a numeric leaf (per grammar relaxation)
    assert C.tree_size(tree_pi) == 1


def test_numeric_literals_non_strict():
    # The grammar relaxation: arbitrary constants ride as leaves.
    tree = C.compile_expr("2 * x")
    assert abs(C.eval_eml(tree, x=3.5) - 7.0) < 1e-9

    tree = C.compile_expr("0.125")
    assert abs(C.eval_eml(tree) - 0.125) < 1e-12


def test_bare_eml_syntax():
    """The parser accepts ``eml(a, b)`` directly — this is what EMLTree1D.to_expr emits."""
    tree = C.compile_expr("eml(x, 1)")
    # eml(x, 1) = exp(x) - ln(1) = exp(x) - 0 = exp(x)
    v = C.eval_eml(tree, x=2.0)
    assert abs(v.real - math.exp(2.0)) < 1e-10
    # should be a single eml node with leaves x, 1 (no bootstrap overhead)
    assert C.tree_size(tree) == 3
    assert C.tree_depth(tree) == 1


def test_nested_eml_syntax():
    # ln(x) = eml(1, eml(eml(1, x), 1)) per §3
    tree = C.compile_expr("eml(1, eml(eml(1, x), 1))")
    v = C.eval_eml(tree, x=2.0)
    assert abs(v.real - math.log(2.0)) < 1e-10


# ─── Round-trip: to_string → parse → compile → evaluate ────────────

ROUND_TRIP_EXPRS = [
    "exp(x)",
    "ln(x)",
    "-x",
    "x + y",
    "x * y",
    "x - y",
    "x / y",
    "x ^ y",
    "sqrt(x + 1)",
    "ln(x) + exp(y)",
    "(x + y) * z",
    "exp(-x) + 1",
    "pi * x",
]


@pytest.mark.parametrize("expr", ROUND_TRIP_EXPRS)
def test_round_trip_self(expr):
    """to_string → parse → compile → same value as the original compiled tree."""
    b = {"x": 1.5, "y": 2.5, "z": 3.5}
    t1 = C.compile_expr(expr)
    t2 = C.compile_expr(C.to_string(t1))
    v1 = C.eval_eml(t1, b)
    v2 = C.eval_eml(t2, b)
    assert abs(v1 - v2) < 1e-10, f"{expr}: v1={v1} v2={v2}"


# ─── Round-trip against a reference python eval ────────────────────

def _reference_eval(expr: str, bindings: dict) -> complex:
    """Use python's eval() with math symbols as the gold reference."""
    env = {
        "exp": np.exp, "log": np.log, "ln": np.log, "sqrt": np.sqrt,
        "pi": math.pi, "e": math.e, "__builtins__": {},
    }
    env.update(bindings)
    # JS-style ^ for power → python **
    py_expr = expr.replace("^", "**")
    return complex(eval(py_expr, env))  # noqa: S307 — controlled env


@pytest.mark.parametrize(
    "expr, bindings",
    [
        ("exp(x) + ln(y)",         {"x": 0.5, "y": 3.0}),
        ("sqrt(x*x + y*y)",        {"x": 3.0, "y": 4.0}),
        ("(x + 1) / (x - 1)",      {"x": 2.5}),
        ("x^3 - 2*x + 1",          {"x": 1.5}),
        ("ln(exp(x))",             {"x": 2.3}),
        ("exp(ln(x))",             {"x": 7.1}),
        ("e^x",                    {"x": 1.5}),
    ],
)
def test_compile_matches_reference(expr, bindings):
    tree = C.compile_expr(expr)
    got = C.eval_eml(tree, bindings)
    want = _reference_eval(expr, bindings)
    # Generous tolerance — the bootstrap chain accumulates some rounding
    # through repeated exp/ln and through IEEE-inf cancellation in add/neg.
    assert abs(got - want) < 1e-6, f"{expr}: got {got}, want {want}, diff {got - want}"


# ─── Strict mode ───────────────────────────────────────────────────

STRICT_ACCEPTED = ["x", "x + x", "x * x", "exp(x)", "ln(x)", "e", "-x", "x^x", "1", "0"]


@pytest.mark.parametrize("expr", STRICT_ACCEPTED)
def test_strict_accepts(expr):
    tree = C.compile_expr(expr, strict=True)
    # All numeric leaves must be 0 or 1 in strict mode
    for leaf in _walk_leaves(tree):
        if not leaf.is_symbolic:
            assert leaf.label in ("0", "1"), f"{expr}: forbidden leaf {leaf.label!r}"


@pytest.mark.parametrize(
    "expr, substring_in_error",
    [
        ("2",          "numeric literal"),
        ("0.5",        "numeric literal"),
        ("pi",         "π"),
        ("3 * x",      "numeric literal"),
    ],
)
def test_strict_rejects(expr, substring_in_error):
    with pytest.raises(ValueError, match=substring_in_error):
        C.compile_expr(expr, strict=True)


def test_strict_neg_uses_eml_ln_one_not_literal_zero():
    """Paper-faithful strict mode routes negation through eml_ln(1) instead
    of a literal 0 leaf. Verify no '0' leaves appear."""
    tree = C.compile_expr("-x", strict=True)
    for leaf in _walk_leaves(tree):
        assert leaf.label != "0", "strict -x should not contain a literal 0 leaf"
    # Still evaluates correctly
    assert abs(C.eval_eml(tree, x=2.0) - (-2.0)) < 1e-10


def test_strict_sqrt_rewrites_half_as_one_over_two():
    """sqrt in strict mode must not introduce a 0.5 leaf — it rewrites as
    x^(1/(1+1))."""
    tree = C.compile_expr("sqrt(x)", strict=True)
    for leaf in _walk_leaves(tree):
        if not leaf.is_symbolic:
            assert leaf.label in ("0", "1"), f"strict sqrt leaked leaf {leaf.label!r}"
    val = C.eval_eml(tree, x=16.0).real
    assert abs(val - 4.0) < 1e-8


def test_trig_rejected():
    for fn in ("sin", "cos", "tan"):
        with pytest.raises(ValueError, match="out of scope"):
            C.compile_expr(f"{fn}(x)")


# ─── Variables whitelist ───────────────────────────────────────────

def test_variables_whitelist_accepts_listed():
    tree = C.compile_expr("a + b", variables=["a", "b"])
    assert abs(C.eval_eml(tree, a=2.0, b=3.0) - 5.0) < 1e-9


def test_variables_whitelist_rejects_unlisted():
    with pytest.raises(ValueError, match="unknown identifier 'z'"):
        C.compile_expr("a + z", variables=["a", "b"])


# ─── Tree utilities ────────────────────────────────────────────────

def test_tree_size_and_depth_leaf():
    leaf = Leaf(value=1.0, label="1")
    assert C.tree_size(leaf) == 1
    assert C.tree_depth(leaf) == 0


def test_free_variables():
    tree = C.compile_expr("x * y + z")
    assert C.free_variables(tree) == {"x", "y", "z"}


def test_missing_binding_raises():
    tree = C.compile_expr("x + 1")
    with pytest.raises(ValueError, match="no binding for variable 'x'"):
        C.eval_eml(tree)


# ─── EMLTree1D round-trip: compile(to_expr(tree)) ≈ tree(x) ────────

@pytest.mark.parametrize("seed", list(range(5)))
def test_round_trip_through_emltree1d(seed):
    """For a snapped EMLTree1D, compile(tree.to_expr()) should evaluate
    to the same number as the tree itself at a test point. This is the
    compatibility contract mentioned in the issue.

    The tree must be *snapped* first — ``to_expr`` reads argmax over
    leaf/gate logits, but unsnapped ``forward`` returns the softmax
    mixture. Snapping aligns them.
    """
    import eml_sr

    torch.manual_seed(seed)
    tree = eml_sr.EMLTree1D(depth=2).snap()
    expr = tree.to_expr()

    compiled = C.compile_expr(expr)
    x_test = 1.5
    y_compiled = C.eval_eml(compiled, x=x_test).real

    with torch.no_grad():
        y_tree, _, _ = tree(torch.tensor([[x_test]], dtype=torch.complex128))
    y_tree = y_tree.real.item()

    # Accept inf/nan matches; otherwise require tight numerical agreement.
    if math.isnan(y_tree) or math.isnan(y_compiled):
        assert math.isnan(y_tree) and math.isnan(y_compiled), (
            f"seed={seed}: one is nan, other isn't — expr={expr!r}"
        )
        return
    if math.isinf(y_tree) or math.isinf(y_compiled):
        assert math.isinf(y_tree) and math.isinf(y_compiled), (
            f"seed={seed}: one is inf, other isn't — expr={expr!r}"
        )
        assert (y_tree > 0) == (y_compiled > 0), (
            f"seed={seed}: inf sign mismatch — expr={expr!r}"
        )
        return
    assert abs(y_tree - y_compiled) < 1e-3, (
        f"seed={seed} expr={expr!r}: tree={y_tree} compiled={y_compiled}"
    )


# ─── CLI ───────────────────────────────────────────────────────────

def test_cli_prints_tree_size_depth():
    result = subprocess.run(
        [sys.executable, "-m", "eml_compiler", "ln(x) + exp(y)"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "eml(" in out
    assert "size:" in out
    assert "depth:" in out


def test_cli_eval_flag():
    result = subprocess.run(
        [sys.executable, "-m", "eml_compiler", "x + y", "--eval", "x=2", "y=3"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "value:" in result.stdout
    # 2 + 3 = 5; value line should contain "5"
    assert "5" in result.stdout.split("value:", 1)[1]


def test_cli_strict_rejects_literal():
    result = subprocess.run(
        [sys.executable, "-m", "eml_compiler", "--strict", "2 * x"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 2
    assert "compile error" in result.stderr


# ─── Helpers ───────────────────────────────────────────────────────

def _walk_leaves(tree):
    if isinstance(tree, Leaf):
        yield tree
        return
    yield from _walk_leaves(tree.left)
    yield from _walk_leaves(tree.right)
