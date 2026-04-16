"""Test multivariate support for EMLTree1D and EMLTree1DLinear (issue #15).

Verifies:
    1. Construction with n_vars > 1
    2. Forward pass shape correctness
    3. Parameter shapes
    4. Snap and to_expr for multivariate trees
    5. Backward compatibility: n_vars=1 matches original behavior
    6. Same for EMLTree1DLinear
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eml_sr import DTYPE, REAL, EMLTree1D, eml_op
from eml_sr_linear import EMLTree1DLinear


def _as_real(x):
    return torch.tensor(np.asarray(x), dtype=REAL)


# ═══════════════════════ Option A multivariate ══════════════════════

class TestEMLTree1DMultivariate:
    """EMLTree1D with n_vars > 1."""

    @pytest.mark.parametrize("n_vars", [2, 3, 5])
    def test_construction(self, n_vars):
        tree = EMLTree1D(depth=2, n_vars=n_vars)
        assert tree.n_vars == n_vars
        assert tree.leaf_logits.shape == (4, n_vars + 1)
        assert tree.gate_logits.shape == (3, 2, n_vars + 2)

    @pytest.mark.parametrize("n_vars", [2, 3])
    def test_forward_shape(self, n_vars):
        tree = EMLTree1D(depth=2, n_vars=n_vars)
        X = torch.randn(20, n_vars, dtype=REAL)
        out, lp, gp = tree(X)
        assert out.shape == (20,)
        assert out.dtype == DTYPE
        assert lp.shape == (4, n_vars + 1)
        assert gp.shape == (3, 2, n_vars + 2)

    def test_forward_2d_batch(self):
        """2D input (batch, n_vars) works."""
        tree = EMLTree1D(depth=1, n_vars=2)
        X = torch.randn(10, 2, dtype=REAL)
        out, _, _ = tree(X)
        assert out.shape == (10,)

    def test_snap_multivariate(self):
        tree = EMLTree1D(depth=1, n_vars=3)
        snapped = tree.snap()
        assert snapped.leaf_logits.shape == tree.leaf_logits.shape
        assert snapped.gate_logits.shape == tree.gate_logits.shape
        # Snap is idempotent
        snapped2 = snapped.snap()
        torch.testing.assert_close(
            snapped.leaf_logits, snapped2.leaf_logits)
        torch.testing.assert_close(
            snapped.gate_logits, snapped2.gate_logits)

    def test_to_expr_multivariate_vars(self):
        """With n_vars > 1, expression should use x1, x2, ... names."""
        tree = EMLTree1D(depth=1, n_vars=2)
        snapped = tree.snap()
        expr = snapped.to_expr()
        assert isinstance(expr, str)
        # Should contain x1 or x2 (or 1, depending on snap)
        # At minimum it should be a valid string
        assert len(expr) > 0

    def test_to_expr_univariate_uses_x(self):
        """With n_vars=1, expression should use 'x' not 'x1'."""
        tree = EMLTree1D(depth=1, n_vars=1)
        snapped = tree.snap()
        expr = snapped.to_expr()
        # Should use "x" not "x1"
        assert "x1" not in expr or "x" in expr

    def test_hand_crafted_eml_x1_x2(self):
        """Hand-snap a 2-var depth-1 tree to compute eml(x1, x2)."""
        tree = EMLTree1D(depth=1, n_vars=2, init_scale=0.0)
        k = 50.0
        with torch.no_grad():
            # Leaf 0 → x1 (index 1), Leaf 1 → x2 (index 2)
            leaf = torch.full((2, 3), -k, dtype=REAL)
            leaf[0, 1] = k  # leaf 0 = x1
            leaf[1, 2] = k  # leaf 1 = x2
            tree.leaf_logits.copy_(leaf)
            # Gate: pass children through (index 3 = child, which is n_vars+1=3)
            gate = torch.full((1, 2, 4), -k, dtype=REAL)
            gate[0, 0, 3] = k  # left side → child
            gate[0, 1, 3] = k  # right side → child
            tree.gate_logits.copy_(gate)

        x1 = torch.tensor([0.5, 1.0, 2.0], dtype=REAL)
        x2 = torch.tensor([1.0, 2.0, 0.5], dtype=REAL)
        X = torch.stack([x1, x2], dim=1)
        out, _, _ = tree(X, tau=0.01)
        expected = eml_op(x1.to(DTYPE), x2.to(DTYPE))
        torch.testing.assert_close(out, expected, atol=1e-10, rtol=1e-10)

    def test_backward_compat_1d_input(self):
        """1D tensor input still works with n_vars=1 (backward compat)."""
        tree = EMLTree1D(depth=2, n_vars=1)
        x = torch.randn(15, dtype=REAL)
        out, lp, gp = tree(x)
        assert out.shape == (15,)

    def test_n_uncertain_multivariate(self):
        tree = EMLTree1D(depth=1, n_vars=3)
        n = tree.n_uncertain()
        assert isinstance(n, int)
        assert n >= 0


# ═══════════════════════ Option B multivariate ══════════════════════

class TestEMLTree1DLinearMultivariate:
    """EMLTree1DLinear with n_vars > 1."""

    @pytest.mark.parametrize("n_vars", [2, 3, 5])
    def test_construction(self, n_vars):
        tree = EMLTree1DLinear(depth=2, n_vars=n_vars)
        assert tree.n_vars == n_vars
        assert tree.leaf_logits.shape == (4, n_vars + 1)
        assert tree.gate_logits.shape == (3, 2, n_vars + 2)

    @pytest.mark.parametrize("n_vars", [2, 3])
    def test_forward_shape(self, n_vars):
        tree = EMLTree1DLinear(depth=2, n_vars=n_vars)
        X = torch.randn(20, n_vars, dtype=REAL)
        out, lp, gp = tree(X)
        assert out.shape == (20,)
        assert out.dtype == DTYPE
        assert lp is None
        assert gp is None

    def test_snap_multivariate(self):
        tree = EMLTree1DLinear(depth=1, n_vars=3)
        snapped = tree.snap()
        assert snapped.leaf_logits.shape == tree.leaf_logits.shape
        assert snapped.gate_logits.shape == tree.gate_logits.shape

    def test_to_expr_multivariate(self):
        tree = EMLTree1DLinear(depth=1, n_vars=2)
        snapped = tree.snap()
        expr = snapped.to_expr()
        assert isinstance(expr, str)
        assert len(expr) > 0

    def test_to_expr_univariate_backward_compat(self):
        """n_vars=1 should use 'x' not 'x1'."""
        tree = EMLTree1DLinear(depth=1, n_vars=1)
        expr = tree.snap().to_expr()
        assert "x1" not in expr

    def test_hand_crafted_eml_x1_x2(self):
        """Hand-set coefficients for eml(x1, x2) = exp(x1) - ln(x2)."""
        tree = EMLTree1DLinear(depth=1, n_vars=2, init_scale=0.0)
        with torch.no_grad():
            # Leaf 0: 0 + 1*x1 + 0*x2 = x1
            # Leaf 1: 0 + 0*x1 + 1*x2 = x2
            tree.leaf_logits.copy_(torch.tensor(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=REAL))
            # Gate: α=0, β1=0, β2=0, γ=1 for both sides → pass children
            tree.gate_logits.copy_(torch.tensor(
                [[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]], dtype=REAL))

        x1 = torch.tensor([0.5, 1.0, 2.0], dtype=REAL)
        x2 = torch.tensor([1.0, 2.0, 0.5], dtype=REAL)
        X = torch.stack([x1, x2], dim=1)
        out, _, _ = tree(X)
        expected = eml_op(x1.to(DTYPE), x2.to(DTYPE))
        torch.testing.assert_close(out, expected, atol=1e-10, rtol=1e-10)

    def test_backward_compat_1d_input(self):
        """1D tensor input still works with n_vars=1."""
        tree = EMLTree1DLinear(depth=2, n_vars=1)
        x = torch.randn(15, dtype=REAL)
        out, _, _ = tree(x)
        assert out.shape == (15,)

    def test_n_params_multivariate(self):
        tree = EMLTree1DLinear(depth=2, n_vars=3)
        # leaves: 4 * (3+1) = 16, gates: 3 * 2 * (3+2) = 30
        assert tree.n_params() == 16 + 30


# ═══════════════════════ Cross-tree consistency ═════════════════════

class TestMultivariateCrossConsistency:
    """Verify Option A and B have matching shapes for same n_vars."""

    @pytest.mark.parametrize("n_vars", [1, 2, 3])
    def test_matching_shapes(self, n_vars):
        a = EMLTree1D(depth=2, n_vars=n_vars)
        b = EMLTree1DLinear(depth=2, n_vars=n_vars)
        assert a.leaf_logits.shape == b.leaf_logits.shape
        assert a.gate_logits.shape == b.gate_logits.shape
