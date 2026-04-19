"""Numerical verification of EDL and NEG_EML canonical identities.

Each identity in :mod:`eml_compiler` for EDL and NEG_EML is hand-derived
from the operator definition, and the paper does not list them. This
test suite plugs concrete sample points into each identity and checks
that the EML compiler's evaluator produces the same value as Python's
:mod:`math` reference, within machine epsilon for the well-conditioned
ones and within ``1e-9`` for the ones that route through complex
intermediates (NEG_EML's ``exp``).

If a future change to the primitive tables drifts an identity, the
tests here are the canary.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from eml_compiler import (
    Leaf, Node, compile_expr, eval_eml, tree_size, tree_depth,
    _e_leaf, _ninf_leaf, _eml_exp, _eml_ln,
    _edl_exp, _edl_ln, _edl_one_tree,
    _ne_exp, _ne_ln,
)
from eml_operators import EML, EDL, NEG_EML

# Sample points chosen to be (a) positive, (b) far from boundary x=1
# (where EDL's ln(y) denominator goes to zero), and (c) spread enough
# that an accidentally-constant identity would be caught.
SAMPLES = [0.5, 1.7, 2.3, 3.9, 5.1]


# ─── EDL identities ───────────────────────────────────────────────

class TestEDLIdentities:
    def test_exp_identity(self):
        """exp(x) = edl(x, e); size 3, depth 1."""
        from eml_compiler import Leaf as _L
        x = _L(value=None, label="x")
        tree = _edl_exp(x)
        assert tree_size(tree) == 3
        assert tree_depth(tree) == 1
        for xv in SAMPLES:
            got = complex(eval_eml(tree, x=xv, op_config=EDL))
            want = math.exp(xv)
            assert abs(got - want) < 1e-12, f"exp({xv}): got {got}, want {want}"

    def test_ln_identity(self):
        """ln(x) = edl(e, edl(edl(e, x), e)); size 7, depth 3."""
        from eml_compiler import Leaf as _L
        x = _L(value=None, label="x")
        tree = _edl_ln(x)
        assert tree_size(tree) == 7
        assert tree_depth(tree) == 3
        for xv in SAMPLES:
            got = complex(eval_eml(tree, x=xv, op_config=EDL))
            want = math.log(xv)
            assert abs(got - want) < 1e-12, f"ln({xv}): got {got}, want {want}"

    def test_one_constant(self):
        """1 = edl(e, edl(edl(e, e), e)); size 7, depth 3."""
        tree = _edl_one_tree()
        assert tree_size(tree) == 7
        assert tree_depth(tree) == 3
        got = complex(eval_eml(tree, op_config=EDL))
        assert abs(got - 1.0) < 1e-12

    def test_compile_division(self):
        """Compiled a/b matches numerical a/b for compiled trees."""
        tree = compile_expr("x / y", op_config=EDL, variables=("x", "y"))
        for xv, yv in [(2.0, 3.0), (5.0, 1.5), (7.0, 0.4)]:
            got = complex(eval_eml(tree, x=xv, y=yv, op_config=EDL))
            want = xv / yv
            assert abs(got - want) < 1e-9, f"{xv}/{yv}: got {got}, want {want}"

    def test_compile_exp(self):
        tree = compile_expr("exp(x)", op_config=EDL, variables=("x",))
        for xv in SAMPLES:
            got = complex(eval_eml(tree, x=xv, op_config=EDL))
            want = math.exp(xv)
            assert abs(got - want) < 1e-12


# ─── NEG_EML identities ───────────────────────────────────────────

class TestNegEMLIdentities:
    def test_ln_identity(self):
        """ln(x) = ne(x, -inf); size 3, depth 1."""
        from eml_compiler import Leaf as _L
        x = _L(value=None, label="x")
        tree = _ne_ln(x)
        assert tree_size(tree) == 3
        assert tree_depth(tree) == 1
        for xv in SAMPLES:
            got = complex(eval_eml(tree, x=xv, op_config=NEG_EML))
            want = math.log(xv)
            assert abs(got - want) < 1e-9, f"ln({xv}): got {got}, want {want}"

    def test_exp_identity(self):
        """exp(x) = ne(x, ne(ne(x, x), -inf)); size 7, depth 3.

        Routes through complex intermediates because ``ln(x) - exp(x)``
        is negative for x > 0, and the outer ``ne`` then takes ``ln``
        of that negative real (principal branch).
        """
        from eml_compiler import Leaf as _L
        x = _L(value=None, label="x")
        tree = _ne_exp(x)
        assert tree_size(tree) == 7
        assert tree_depth(tree) == 3
        for xv in SAMPLES:
            got = complex(eval_eml(tree, x=xv, op_config=NEG_EML))
            want = complex(math.exp(xv), 0.0)
            # Loose tolerance: the principal-branch path has a tiny
            # imaginary residual from accumulated ln of negatives.
            assert abs(got - want) < 1e-8, f"exp({xv}): got {got}, want {want}"


# ─── EML regression (sanity) ──────────────────────────────────────

class TestEMLStillWorks:
    """The cousin refactor must not have drifted EML's behaviour."""

    def test_exp(self):
        from eml_compiler import Leaf as _L
        x = _L(value=None, label="x")
        tree = _eml_exp(x)
        for xv in SAMPLES:
            got = complex(eval_eml(tree, x=xv, op_config=EML))
            assert abs(got - math.exp(xv)) < 1e-12

    def test_ln(self):
        from eml_compiler import Leaf as _L
        x = _L(value=None, label="x")
        tree = _eml_ln(x)
        for xv in SAMPLES:
            got = complex(eval_eml(tree, x=xv, op_config=EML))
            assert abs(got - math.log(xv)) < 1e-12

    def test_compile_division(self):
        tree = compile_expr("x / y", op_config=EML, variables=("x", "y"))
        for xv, yv in [(2.0, 3.0), (5.0, 1.5)]:
            got = complex(eval_eml(tree, x=xv, y=yv, op_config=EML))
            assert abs(got - xv / yv) < 1e-9
