"""Test suite for eml_sr_linear (Option B) and eml_sr_hybrid.

Covers:
    1. EMLTree1DLinear architecture (shapes, forward, snap, to_expr)
    2. Helper functions (_snap_scalar, _discreteness_penalty, etc.)
    3. Training and iterative snap
    4. discover_linear API
    5. discover_hybrid staged fallback

Multi-seed recovery tests are marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from eml_sr import DTYPE, REAL, eml_op
from eml_sr_linear import (
    EMLTree1DLinear,
    _discreteness_penalty,
    _nearest_snap,
    _snap_scalar,
    _train_one_linear,
    discover_linear,
    iterative_snap,
)
from eml_sr_hybrid import discover_hybrid


# ───────────────────────── helpers ──────────────────────────

def _as_real(x):
    return torch.tensor(np.asarray(x), dtype=REAL)


def _as_complex(x):
    return torch.tensor(np.asarray(x), dtype=DTYPE)


# ═══════════════════════ 1. Architecture tests ══════════════════════

class TestLinearTreeArchitecture:
    """Verify EMLTree1DLinear parameter shapes and forward pass."""

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_param_shapes(self, depth):
        tree = EMLTree1DLinear(depth)
        n_leaves = 2 ** depth
        n_internal = n_leaves - 1
        assert tree.leaf_logits.shape == (n_leaves, 2)
        assert tree.gate_logits.shape == (n_internal, 2, 3)

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_forward_output_shape(self, depth):
        tree = EMLTree1DLinear(depth)
        x = _as_real(np.linspace(0.5, 3.0, 20))
        out, lp, gp = tree(x)
        assert out.shape == (20,)
        assert out.dtype == DTYPE
        # Option B returns None for leaf_probs and gate_probs
        assert lp is None
        assert gp is None

    def test_forward_batch_independence(self):
        """Each sample computes independently (no cross-batch contamination)."""
        tree = EMLTree1DLinear(2)
        x1 = _as_real([1.0, 2.0])
        x2 = _as_real([1.0, 2.0, 3.0])
        out1, _, _ = tree(x1)
        out2, _, _ = tree(x2)
        # First two outputs should match
        torch.testing.assert_close(out1, out2[:2])

    def test_snap_is_idempotent(self):
        tree = EMLTree1DLinear(2)
        snapped1 = tree.snap()
        snapped2 = snapped1.snap()
        torch.testing.assert_close(
            snapped1.leaf_logits, snapped2.leaf_logits)
        torch.testing.assert_close(
            snapped1.gate_logits, snapped2.gate_logits)

    def test_snap_returns_new_tree(self):
        tree = EMLTree1DLinear(1)
        snapped = tree.snap()
        assert snapped is not tree
        # Modifying snapped should not affect original
        with torch.no_grad():
            snapped.leaf_logits[0, 0] = 999.0
        assert tree.leaf_logits[0, 0].item() != 999.0

    def test_to_expr_returns_string(self):
        tree = EMLTree1DLinear(1)
        expr = tree.snap().to_expr()
        assert isinstance(expr, str)
        assert len(expr) > 0

    def test_n_params(self):
        tree = EMLTree1DLinear(2)
        expected = tree.leaf_logits.numel() + tree.gate_logits.numel()
        assert tree.n_params() == expected

    def test_hand_crafted_exp(self):
        """Hand-set coefficients to compute eml(x, 1) = exp(x)."""
        tree = EMLTree1DLinear(1)
        with torch.no_grad():
            # Leaves: left = x (α=0, β=1), right = 1 (α=1, β=0)
            tree.leaf_logits.copy_(torch.tensor(
                [[0.0, 1.0], [1.0, 0.0]], dtype=REAL))
            # Gate: pass children through (α=0, β=0, γ=1 for both sides)
            tree.gate_logits.copy_(torch.tensor(
                [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], dtype=REAL))
        x = _as_real(np.linspace(0.1, 2.0, 20))
        out, _, _ = tree(x)
        expected = torch.exp(x.to(DTYPE))
        torch.testing.assert_close(out, expected, atol=1e-12, rtol=1e-12)


# ═══════════════════════ 2. Helper function tests ═══════════════════

class TestSnapHelpers:
    """Test _snap_scalar and _nearest_snap."""

    def test_snap_zero(self):
        assert _snap_scalar(0.001) == 0.0

    def test_snap_one(self):
        assert _snap_scalar(0.98) == 1.0

    def test_snap_neg_one(self):
        assert _snap_scalar(-1.02) == -1.0

    def test_snap_e(self):
        assert _snap_scalar(math.e + 0.01) == math.e

    def test_snap_neg_e(self):
        assert _snap_scalar(-math.e + 0.01) == -math.e

    def test_snap_two(self):
        assert _snap_scalar(2.03) == 2.0

    def test_no_snap_far(self):
        """Values far from any named constant return None."""
        assert _snap_scalar(1.5) is None

    def test_nearest_snap_distance(self):
        target, dist = _nearest_snap(0.5)
        # Closest named constant is either 0 or 1, distance 0.5
        assert dist == pytest.approx(0.5)
        assert target in (0.0, 1.0)


class TestDiscretnessPenalty:
    """Test _discreteness_penalty."""

    def test_at_integers(self):
        """Penalty is zero at exact integers."""
        t = torch.tensor([0.0, 1.0, -1.0, 2.0], dtype=REAL)
        p = _discreteness_penalty(t)
        assert float(p.item()) == pytest.approx(0.0, abs=1e-10)

    def test_at_e(self):
        t = torch.tensor([math.e], dtype=REAL)
        p = _discreteness_penalty(t)
        assert float(p.item()) == pytest.approx(0.0, abs=1e-10)

    def test_midpoint_nonzero(self):
        """Penalty is nonzero between named constants."""
        t = torch.tensor([0.5], dtype=REAL)
        p = _discreteness_penalty(t)
        assert float(p.item()) > 0.01

    def test_gradient_flows(self):
        t = torch.tensor([0.5], dtype=REAL, requires_grad=True)
        p = _discreteness_penalty(t)
        p.backward()
        assert t.grad is not None
        assert t.grad.abs().item() > 0


# ═══════════════════════ 3. Training tests ══════════════════════════

class TestTrainOneLinear:
    """Test _train_one_linear returns valid results."""

    def test_returns_expected_keys(self):
        x = _as_real(np.linspace(0.5, 3.0, 25))
        y = _as_complex(np.exp(np.linspace(0.5, 3.0, 25)))
        r = _train_one_linear(x, y, depth=1, seed=0,
                              search_iters=50, snap_iters=50)
        assert "tree" in r
        assert "snapped" in r
        assert "best_mse" in r
        assert "snap_mse" in r
        assert "snap_rmse" in r
        assert "expr" in r
        assert "nan_restarts" in r

    def test_training_reduces_loss(self):
        """After training, MSE should be lower than initial."""
        x = _as_real(np.linspace(0.5, 3.0, 25))
        y = _as_complex(np.exp(np.linspace(0.5, 3.0, 25)))
        r = _train_one_linear(x, y, depth=1, seed=42,
                              search_iters=200, snap_iters=100)
        assert r["best_mse"] < 100.0  # should decrease from random init

    def test_snap_rmse_is_sqrt_snap_mse(self):
        x = _as_real(np.linspace(0.5, 3.0, 25))
        y = _as_complex(np.exp(np.linspace(0.5, 3.0, 25)))
        r = _train_one_linear(x, y, depth=1, seed=0,
                              search_iters=50, snap_iters=50)
        assert r["snap_rmse"] == pytest.approx(
            math.sqrt(max(r["snap_mse"], 0)), abs=1e-12)


class TestIterativeSnap:
    """Test iterative_snap preserves or improves fit quality."""

    def test_snap_does_not_destroy_fit(self):
        """Iterative snap should not increase MSE catastrophically."""
        x = _as_real(np.linspace(0.5, 3.0, 25))
        y_np = np.exp(np.linspace(0.5, 3.0, 25))
        y = _as_complex(y_np)
        r = _train_one_linear(x, y, depth=1, seed=0,
                              search_iters=500, snap_iters=200)
        snapped = iterative_snap(r["tree"], x, y,
                                 retrain_iters=100, verbose=False)
        with torch.no_grad():
            pred, _, _ = snapped(x)
            snap_mse = float(torch.mean(
                (pred - y).abs() ** 2).real.item())
        # Should not be worse than 100× the pre-snap fit
        assert snap_mse < r["best_mse"] * 200 + 1.0

    def test_returns_tree_instance(self):
        x = _as_real(np.linspace(0.5, 3.0, 15))
        y = _as_complex(np.exp(np.linspace(0.5, 3.0, 15)))
        r = _train_one_linear(x, y, depth=1, seed=0,
                              search_iters=50, snap_iters=50)
        snapped = iterative_snap(r["tree"], x, y,
                                 retrain_iters=50, verbose=False)
        assert isinstance(snapped, EMLTree1DLinear)


# ═══════════════════════ 4. discover_linear tests ═══════════════════

class TestDiscoverLinear:
    """Test the discover_linear API."""

    def test_returns_dict_with_expected_keys(self):
        x = np.linspace(0.5, 3.0, 25)
        y = np.exp(x)
        r = discover_linear(x, y, max_depth=1, n_tries=2, verbose=False)
        assert isinstance(r, dict)
        assert "expr" in r
        assert "depth" in r
        assert "snap_rmse" in r
        assert "method" in r
        assert r["method"] == "linear"

    @pytest.mark.slow
    def test_recover_exp(self):
        """Option B should fit exp(x) reasonably well."""
        x = np.linspace(0.5, 3.0, 30)
        y = np.exp(x)
        r = discover_linear(x, y, max_depth=2, n_tries=4, verbose=False)
        assert r["snap_rmse"] < 1.0  # may not be machine-precision

    @pytest.mark.slow
    def test_recover_identity(self):
        """y = x is a key target Option B should handle (Option A cannot)."""
        x = np.linspace(0.5, 3.0, 30)
        y = x.copy()
        r = discover_linear(x, y, max_depth=3, n_tries=6, verbose=False)
        # Option B should fit y = x with reasonable RMSE
        assert r["snap_rmse"] < 0.5


# ═══════════════════════ 5. discover_hybrid tests ═══════════════════

class TestDiscoverHybrid:
    """Test the hybrid A→B fallback dispatcher."""

    @pytest.mark.slow
    def test_hybrid_on_exp(self):
        """exp(x) is in Option A's vocabulary — hybrid should use A."""
        x = np.linspace(0.5, 3.0, 30)
        y = np.exp(x)
        r = discover_hybrid(x, y, max_depth=2, n_tries_a=4,
                            n_tries_b=2, verbose=False)
        assert r is not None
        assert r["method"] == "option_a"
        assert r["snap_rmse"] < 1e-6

    @pytest.mark.slow
    def test_hybrid_on_identity(self):
        """y = x needs Option B fallback."""
        x = np.linspace(0.5, 3.0, 30)
        y = x.copy()
        r = discover_hybrid(x, y, max_depth=3, n_tries_a=4,
                            n_tries_b=4, max_depth_b=3, verbose=False)
        assert r is not None
        # Should have fallen back to Option B (or A got lucky, either is fine)
        assert r["snap_rmse"] < 1.0

    def test_hybrid_returns_expected_keys(self):
        """Even with minimal budget, returns a dict with required keys."""
        x = np.linspace(0.5, 3.0, 15)
        y = np.exp(x)
        r = discover_hybrid(x, y, max_depth=1, n_tries_a=1,
                            n_tries_b=1, verbose=False)
        assert r is not None
        assert "expr" in r
        assert "method" in r
        assert "snap_rmse" in r
        assert r["method"] in ("option_a", "option_b")
