"""Tests for the ternary-operator experiment (issue #37).

Covers:
    * Formula verification — ``T(x, x, x) = 1`` under the paper's
      left-to-right parsing; the issue's original ``exp(x/ln x)`` parsing
      does NOT satisfy the identity.
    * Hand-derived tree primitives (``1``, ``0``, ``exp(x-1)``, ``exp(x)``)
      evaluate to the expected values at numerical probes.
    * Enumerative search at size ≤ 13 reproduces the reachable-primitive
      set and confirms the unreachable ones (``e``, ``ln x``, ``-x``,
      ``1/x``, ``x*x``, ``sqrt x``).
    * TernaryTree1D forward pass runs and produces finite output under
      both pure and relaxed grammars.

The tests are numerical, not symbolic — they check the operator
behaves as documented, not that the grammar is complete.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ternary.bootstrap import PRIMITIVES, T, X, exp_x, exp_x_minus_1, one, zero
from ternary.enumerate_search import (
    DEFAULT_PROBES,
    default_targets,
    enumerate_trees,
    search_targets,
)
from ternary.operator import t_np
from ternary.tree import TernaryTree1D
from ternary.verify_formula import symbolic_check


# ─── A. Formula verification ─────────────────────────────────────

def test_paper_formula_satisfies_identity():
    res = symbolic_check()
    assert res["paper_parse"]["is_one"], (
        "Paper's left-to-right parse (exp(x)/ln(x) * ln(z)/exp(y)) "
        "must satisfy T(x,x,x) = 1 symbolically."
    )


def test_issue_initial_parse_does_not_satisfy_identity():
    res = symbolic_check()
    assert not res["issue_parse"]["is_one"], (
        "Issue #37's original transcription (exp(x/ln x)) should NOT "
        "satisfy the identity — that confirms the discrepancy was a "
        "transcription typo, not a mathematical one."
    )


def test_operator_T_of_xxx_is_one():
    for x in (0.3, 0.577, 1.414, math.pi / 4, 2.5, 5.0, 10.0):
        v = complex(t_np(x, x, x))
        assert abs(v - 1.0) < 1e-12, f"T({x},{x},{x}) = {v}, expected 1"


# ─── B. Hand-derived primitives ──────────────────────────────────

_PROBE_XS = [0.3, 0.577, 1.414, math.pi / 4, 2.5, 5.0]


@pytest.mark.parametrize("primitive", PRIMITIVES, ids=[p.name for p in PRIMITIVES])
def test_hand_derived_primitive(primitive):
    tree = primitive.tree
    for x in _PROBE_XS:
        got = tree.eval(complex(x))
        expected = primitive.target(complex(x))
        err = abs(got - expected)
        assert err < 1e-10, (
            f"{primitive.name}: tree={tree} at x={x} gave {got}, "
            f"expected {expected}, |err|={err:.2e}"
        )


def test_tree_sizes_as_documented():
    assert one().size() == 4
    assert one().depth() == 1
    assert zero().size() == 7
    assert zero().depth() == 2
    assert exp_x_minus_1().size() == 7
    assert exp_x_minus_1().depth() == 2
    assert exp_x().size() == 10
    assert exp_x().depth() == 3


# ─── C. Enumerative VerifyBaseSet ────────────────────────────────

@pytest.fixture(scope="module")
def pool_size_13():
    return enumerate_trees(max_size=13)


def test_reachable_primitives_at_size_13(pool_size_13):
    found = search_targets(pool_size_13, default_targets())
    # These four primitives must be reachable at size ≤ 13.
    for name in ("1", "0", "exp(x-1)", "exp(x)"):
        assert found[name] is not None, (
            f"{name} should be reachable at size ≤ 13 but enumeration "
            f"didn't find it"
        )


def test_unreachable_primitives_at_size_13(pool_size_13):
    found = search_targets(pool_size_13, default_targets())
    # These must NOT be reachable at size ≤ 13 — pure S → x | T(S,S,S)
    # grammar cannot construct them at practical depths. Guarded as a
    # regression: if a future change to the search or operator makes
    # them reachable, the test flags it for investigation.
    for name in ("e", "ln(x)", "-x", "1/x", "x*x", "sqrt(x)", "2"):
        assert found[name] is None, (
            f"{name} unexpectedly reachable at size ≤ 13 — "
            f"investigate."
        )


# ─── D. TernaryTree1D forward pass ───────────────────────────────

def test_tree_pure_forward_runs():
    torch.manual_seed(0)
    tree = TernaryTree1D(depth=2, allow_terminal=False)
    x = torch.linspace(0.3, 2.0, 16, dtype=torch.float64)
    y, leaf_p, gate_p = tree(x, tau=1.0)
    assert y.shape == (16,)
    assert torch.isfinite(y.real).all()
    assert torch.isfinite(y.imag).all()


def test_tree_relaxed_forward_runs():
    torch.manual_seed(0)
    tree = TernaryTree1D(depth=2, allow_terminal=True)
    x = torch.linspace(0.3, 2.0, 16, dtype=torch.float64)
    y, leaf_p, gate_p = tree(x, tau=1.0)
    assert y.shape == (16,)
    # Finite is all we guarantee on random init; values can be wild.
    assert torch.isfinite(y.real).all() or torch.isnan(y.real).any() is False


def test_tree_parameter_counts():
    """Relaxed tree should have strictly more parameters than pure tree
    of the same depth (learnable per-leaf and per-gate complex constants).
    """
    pure = TernaryTree1D(depth=2, allow_terminal=False)
    relaxed = TernaryTree1D(depth=2, allow_terminal=True)
    assert relaxed.n_params() > pure.n_params()
