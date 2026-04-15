"""Comprehensive Python test suite for eml_sr.

Replaces the legacy Mojo stack-machine tests with pytest coverage of the
PyTorch symbolic regression engine:

    1. Unit tests for the EML operator                       (::TestEmlOp)
    2. Tree architecture tests                               (::TestTreeArchitecture)
    3. Known-target recovery tests                           (::TestRecovery)
    4. Regression tests against numpy / reference values     (::TestRegression)
    5. Property tests (idempotence, depth monotonicity, ...) (::TestProperties)
    6. Expression simplifier tests                           (::TestSimplifier)

Multi-seed recovery tests are marked ``@pytest.mark.slow`` so the default run
stays fast. Run everything with ``pytest -m "not slow or slow"`` or simply
``pytest``; skip the slow ones with ``pytest -m "not slow"``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import eml_sr
from eml_sr import (
    DTYPE,
    REAL,
    EMLTree1D,
    _simplify,
    _train_one,
    discover,
    eml_op,
)


# ───────────────────────── helpers ──────────────────────────

def _as_complex(x):
    return torch.tensor(np.asarray(x), dtype=DTYPE)


def _as_real(x):
    return torch.tensor(np.asarray(x), dtype=REAL)


def _fast_train(x, y, depth, seed=0, search=250, hard=100):
    """Tiny training budget for property / regression tests."""
    x_t = _as_real(x)
    y_t = _as_complex(y)
    return _train_one(
        x_t, y_t, depth=depth, seed=seed,
        search_iters=search, hard_iters=hard, verbose=False,
    )


# ═══════════════════════ 1. eml_op unit tests ═══════════════════════

class TestEmlOp:
    """eml_op(x, y) = exp(x) - ln(y) on torch.complex128."""

    def test_known_values(self):
        # eml(0, 1) = exp(0) - ln(1) = 1 - 0 = 1
        r = eml_op(_as_complex([0.0]), _as_complex([1.0]))
        assert torch.allclose(r.real, torch.tensor([1.0], dtype=REAL))
        assert torch.allclose(r.imag, torch.tensor([0.0], dtype=REAL))

    def test_eml_x_1_is_exp(self):
        x = _as_real(np.linspace(-2.0, 2.0, 11))
        r = eml_op(x.to(DTYPE), _as_complex(np.ones(11)))
        expected = torch.exp(x)
        assert torch.allclose(r.real, expected, atol=1e-12)
        assert torch.allclose(r.imag, torch.zeros_like(r.imag), atol=1e-12)

    def test_eml_0_x_is_neg_ln(self):
        x = _as_real(np.linspace(0.1, 5.0, 10))
        r = eml_op(_as_complex(np.zeros(10)), x.to(DTYPE))
        # eml(0, x) = 1 - ln(x)
        expected = 1.0 - torch.log(x)
        assert torch.allclose(r.real, expected, atol=1e-12)

    def test_eml_x_e_equals_exp_minus_1(self):
        e = _as_complex(np.full(5, math.e))
        x = _as_real(np.linspace(0.0, 2.0, 5))
        r = eml_op(x.to(DTYPE), e)
        expected = torch.exp(x) - 1.0
        assert torch.allclose(r.real, expected, atol=1e-12)

    def test_eml_is_exp_minus_ln(self):
        # Pure definition check over random points.
        torch.manual_seed(0)
        x = torch.randn(20, dtype=REAL)
        y = torch.rand(20, dtype=REAL) + 0.1  # keep y > 0
        lhs = eml_op(x.to(DTYPE), y.to(DTYPE))
        rhs = torch.exp(x) - torch.log(y)
        assert torch.allclose(lhs.real, rhs, atol=1e-12)

    def test_complex_branch_cut_negative_real(self):
        # ln of a negative real enters the upper branch: ln(-1) = iπ.
        r = eml_op(_as_complex([0.0]), _as_complex([-1.0 + 0j]))
        # eml(0, -1) = 1 - ln(-1) = 1 - iπ
        assert abs(r[0].real.item() - 1.0) < 1e-12
        assert abs(r[0].imag.item() + math.pi) < 1e-12

    def test_complex_input(self):
        # Both inputs complex: eml(i, 1) = exp(i) - 0 = cos(1) + i sin(1)
        r = eml_op(_as_complex([1j]), _as_complex([1.0 + 0j]))
        assert abs(r[0].real.item() - math.cos(1.0)) < 1e-12
        assert abs(r[0].imag.item() - math.sin(1.0)) < 1e-12

    def test_ln_zero_gives_neg_inf(self):
        r = eml_op(_as_complex([0.0]), _as_complex([0.0]))
        # ln(0) = -inf → eml = 1 - (-inf) = +inf
        assert torch.isinf(r.real).item()
        assert r.real.item() > 0

    def test_inf_inputs_produce_nonfinite(self):
        r = eml_op(_as_complex([float("inf")]), _as_complex([1.0]))
        assert not torch.isfinite(r.real).item()


# ═══════════════════════ 2. Tree architecture tests ═══════════════════════

class TestTreeArchitecture:

    @pytest.mark.parametrize("depth", [1, 2, 3, 4])
    def test_leaf_and_internal_counts(self, depth):
        tree = EMLTree1D(depth)
        assert tree.n_leaves == 2 ** depth
        assert tree.n_internal == 2 ** depth - 1

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_parameter_count(self, depth):
        tree = EMLTree1D(depth)
        # leaf_logits: n_leaves * 2, gate_logits: n_internal * 2 * 3
        expected = (2 ** depth) * 2 + (2 ** depth - 1) * 6
        got = sum(p.numel() for p in tree.parameters())
        assert got == expected

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_forward_shapes(self, depth):
        tree = EMLTree1D(depth)
        x = _as_real(np.linspace(0.5, 2.5, 7))
        pred, leaf_probs, gate_probs = tree(x, tau=1.0)
        assert pred.shape == (7,)
        assert pred.dtype == DTYPE
        assert leaf_probs.shape == (2 ** depth, 2)
        assert gate_probs.shape == (2 ** depth - 1, 2, 3)

    def test_forward_finite(self):
        # Even with extreme input magnitudes, forward must stay finite
        # (clamp/scrub logic is part of the contract).
        tree = EMLTree1D(3)
        x = _as_real(np.linspace(-10.0, 10.0, 25))
        pred, _, _ = tree(x, tau=1.0)
        assert torch.isfinite(pred.real).all()
        assert torch.isfinite(pred.imag).all()

    def test_snap_returns_deep_copy(self):
        tree = EMLTree1D(2)
        snapped = tree.snap()
        assert snapped is not tree
        # Mutating original shouldn't affect snapped.
        with torch.no_grad():
            tree.leaf_logits.add_(100.0)
        assert not torch.allclose(tree.leaf_logits, snapped.leaf_logits)

    def test_snap_produces_hard_onehots(self):
        tree = EMLTree1D(3)
        snapped = tree.snap()
        # Every softmax row should have max prob ~= 1.0.
        with torch.no_grad():
            lp = torch.softmax(snapped.leaf_logits, dim=1)
            gp = torch.softmax(snapped.gate_logits, dim=-1)
        assert (lp.max(dim=1).values > 0.999).all()
        assert (gp.max(dim=-1).values > 0.999).all()
        assert snapped.n_uncertain() == 0

    def test_snap_is_idempotent(self):
        torch.manual_seed(7)
        tree = EMLTree1D(3)
        snapped1 = tree.snap()
        snapped2 = snapped1.snap()
        assert torch.allclose(snapped1.leaf_logits, snapped2.leaf_logits)
        assert torch.allclose(snapped1.gate_logits, snapped2.gate_logits)

    def test_to_expr_parseable(self):
        torch.manual_seed(0)
        tree = EMLTree1D(2).snap()
        expr = tree.to_expr()
        assert isinstance(expr, str) and len(expr) > 0
        # Simplification must not crash on its own output.
        assert _simplify(expr) == expr or _simplify(_simplify(expr)) == _simplify(expr)

    def test_n_uncertain_bounds(self):
        tree = EMLTree1D(3)
        # Random init: at most n_leaves + 2*n_internal parameters can be uncertain.
        max_nu = tree.n_leaves + 2 * tree.n_internal
        assert 0 <= tree.n_uncertain() <= max_nu


# ═══════════════════════ 3. Known-target recovery ═══════════════════════

class TestRecovery:
    """The key tests: can training recover known targets?

    Fast recovery tests (depth 1) run by default. The ≥8-seed ladder tests
    are gated behind ``-m slow`` so a default ``pytest`` stays snappy.
    """

    def test_snap_of_manual_exp_tree(self):
        """A depth-1 tree hand-snapped to eml(x, 1) = exp(x) should fit exactly."""
        tree = EMLTree1D(1)
        with torch.no_grad():
            k = 50.0
            # leaf 0 → x (index 1), leaf 1 → 1 (index 0)
            tree.leaf_logits.copy_(torch.tensor([[-k, k], [k, -k]], dtype=REAL))
            # gate: route both sides to child (index 2)
            g = torch.full((1, 2, 3), -k, dtype=REAL)
            g[0, :, 2] = k
            tree.gate_logits.copy_(g)
        x = _as_real(np.linspace(0.5, 2.5, 15))
        pred, _, _ = tree(x, tau=0.01)
        expected = torch.exp(x)
        assert torch.allclose(pred.real, expected, atol=1e-9)
        assert tree.to_expr() == "exp(x)"

    def test_snap_of_constant_e(self):
        """Default init's biased toward constant 1 — and snaps to 'e'."""
        torch.manual_seed(0)
        tree = EMLTree1D(1).snap()
        assert tree.to_expr() == "e"
        x = _as_real(np.linspace(0.5, 2.5, 10))
        pred, _, _ = tree(x, tau=0.01)
        assert torch.allclose(pred.real, torch.full((10,), math.e, dtype=REAL), atol=1e-9)

    def test_recover_exp_depth1_fast(self):
        """exp(x) at depth 1 — should succeed on seed 0 alone."""
        x = np.linspace(0.5, 2.5, 20)
        y = np.exp(x)
        out = _fast_train(x, y, depth=1, seed=0, search=700, hard=250)
        assert out["snap_rmse"] < 1e-8
        assert out["expr"] == "exp(x)"

    def test_recover_constant_e_depth1_fast(self):
        """Constant `e` is depth 1 — routing picks child = eml(1,1) = exp(1) - ln(1) = e."""
        x = np.linspace(0.5, 2.5, 20)
        y = np.full_like(x, np.e)
        out = _fast_train(x, y, depth=1, seed=0, search=700, hard=250)
        assert out["snap_rmse"] < 1e-8
        assert out["expr"] == "e"

    @pytest.mark.slow
    def test_recover_exp_depth1_ladder(self):
        """≤4 seeds at depth 1 must recover exp(x)."""
        x = np.linspace(0.5, 2.5, 25)
        y = np.exp(x)
        r = discover(x, y, max_depth=1, n_tries=4, verbose=False)
        assert r is not None
        assert r["snap_rmse"] < 1e-8
        assert r["expr"] == "exp(x)"

    @pytest.mark.slow
    def test_recover_constant_e(self):
        """Constant e at depth 1 — 100% success within a few seeds."""
        x = np.linspace(0.5, 2.5, 25)
        y = np.full_like(x, np.e)
        r = discover(x, y, max_depth=1, n_tries=4, verbose=False)
        assert r is not None
        assert r["snap_rmse"] < 1e-8
        assert r["expr"] == "e"

    @pytest.mark.slow
    def test_recover_ln_depth3(self):
        """ln(x) lives at depth 3: ln(x) = -eml(0, x) + 1 via several nested forms.

        Requires 8 seeds for ≥75% aggregate success — we just need ≥1 hit.
        """
        x = np.linspace(0.5, 5.0, 30)
        y = np.log(x)
        r = discover(x, y, max_depth=3, n_tries=8, verbose=False)
        assert r is not None
        # ln is hard from random init; accept either exact or near-exact fit.
        assert r["snap_rmse"] < 1e-4

    @pytest.mark.slow
    def test_recover_exp_minus_ln(self):
        """exp(x) - ln(x) = eml(x, x) at depth 1 — exercises the gate routing."""
        x = np.linspace(0.5, 3.0, 25)
        y = np.exp(x) - np.log(x)
        r = discover(x, y, max_depth=2, n_tries=8, verbose=False)
        assert r is not None
        # Accept any formula within tight RMSE — exact form may differ.
        assert r["snap_rmse"] < 1e-6


# ═══════════════════════ 4. Regression tests ═══════════════════════

class TestRegression:
    """Numerical regression: snapped trees match numpy reference values."""

    def test_exp_snap_matches_numpy(self):
        tree = EMLTree1D(1)
        # Hand-snap to exp(x).
        with torch.no_grad():
            k = 50.0
            tree.leaf_logits.copy_(torch.tensor([[-k, k], [k, -k]], dtype=REAL))
            g = torch.full((1, 2, 3), -k, dtype=REAL)
            g[0, :, 2] = k
            tree.gate_logits.copy_(g)
        x_np = np.linspace(-1.0, 2.0, 15)
        pred, _, _ = tree(_as_real(x_np), tau=0.01)
        np.testing.assert_allclose(pred.real.detach().numpy(), np.exp(x_np), atol=1e-10)

    def test_training_does_not_leak_nan(self):
        """Training on a sane target must not end with a NaN MSE."""
        out = _fast_train(
            np.linspace(0.5, 2.0, 15), np.exp(np.linspace(0.5, 2.0, 15)),
            depth=1, seed=0, search=200, hard=100,
        )
        assert np.isfinite(out["best_mse"])
        assert np.isfinite(out["snap_mse"])

    def test_training_with_harsh_target_recovers_gracefully(self):
        """Even if the target is out of reach, training must still return finite stats."""
        # sin(x) is NOT expressible as eml(x,1) alone at depth 1 — fit will be poor
        # but must not crash and must not return NaN.
        x = np.linspace(-1.0, 1.0, 15)
        y = np.sin(x)
        out = _fast_train(x, y, depth=1, seed=0, search=200, hard=100)
        assert np.isfinite(out["best_mse"])
        assert np.isfinite(out["snap_mse"])
        assert out["nan_restarts"] <= 20


# ═══════════════════════ 5. Property tests ═══════════════════════

class TestProperties:

    def test_snap_is_idempotent_on_trained_tree(self):
        out = _fast_train(
            np.linspace(0.5, 2.0, 15), np.exp(np.linspace(0.5, 2.0, 15)),
            depth=1, seed=0, search=200, hard=100,
        )
        snapped1 = out["snapped"]
        snapped2 = snapped1.snap()
        assert torch.allclose(snapped1.leaf_logits, snapped2.leaf_logits)
        assert torch.allclose(snapped1.gate_logits, snapped2.gate_logits)
        # Same forward output.
        x = _as_real(np.linspace(0.5, 2.0, 15))
        p1, _, _ = snapped1(x, tau=0.01)
        p2, _, _ = snapped2(x, tau=0.01)
        assert torch.allclose(p1.real, p2.real, atol=1e-12)

    @pytest.mark.parametrize("tau", [0.01, 0.1, 1.0, 2.0])
    def test_snapped_tree_is_tau_invariant(self, tau):
        """Once snapped, output should be (nearly) independent of tau."""
        torch.manual_seed(1)
        tree = EMLTree1D(2).snap()
        x = _as_real(np.linspace(0.5, 2.0, 10))
        base, _, _ = tree(x, tau=0.01)
        other, _, _ = tree(x, tau=tau)
        # At hard snap levels the softmax is ~one-hot regardless of tau.
        assert torch.allclose(base.real, other.real, atol=1e-8)
        assert torch.allclose(base.imag, other.imag, atol=1e-8)

    def test_deeper_trees_can_express_shallower(self):
        """A depth-2 tree, when appropriately snapped, can realize exp(x) —
        just like a depth-1 tree. Verify by matching outputs."""
        # Depth 1: exp(x) = eml(x, 1).
        shallow = EMLTree1D(1)
        with torch.no_grad():
            k = 50.0
            shallow.leaf_logits.copy_(torch.tensor([[-k, k], [k, -k]], dtype=REAL))
            g = torch.full((1, 2, 3), -k, dtype=REAL)
            g[0, :, 2] = k
            shallow.gate_logits.copy_(g)

        # Depth 2: a subtree passing x through, then eml(x, 1) at the root.
        # The lower-level gate routes both sides to the LEAF (value x, 1)
        # but a simpler approach is to bypass the lower subtree via gate
        # choices [x, 1] at the root, which is exactly eml(x, 1) = exp(x).
        deep = EMLTree1D(2)
        with torch.no_grad():
            k = 50.0
            # Leaves don't matter — root gate bypasses them.
            deep.leaf_logits.copy_(
                torch.tensor([[k, -k], [k, -k], [k, -k], [k, -k]], dtype=REAL))
            # Internal nodes: root is index 2 in bottom-up order; first 2 are
            # the lower gates. Set all gates to something, then the root to
            # route side 0 → x (index 1), side 1 → 1 (index 0).
            g = torch.full((3, 2, 3), -k, dtype=REAL)
            g[0, :, 0] = k  # lower left: both sides → 1
            g[1, :, 0] = k  # lower right: both sides → 1
            g[2, 0, 1] = k  # root left → x
            g[2, 1, 0] = k  # root right → 1
            deep.gate_logits.copy_(g)

        x = _as_real(np.linspace(0.5, 2.0, 10))
        ps, _, _ = shallow(x, tau=0.01)
        pd, _, _ = deep(x, tau=0.01)
        assert torch.allclose(ps.real, pd.real, atol=1e-9)

    def test_forward_determinism(self):
        """Given a fixed state, repeated forward calls produce identical outputs."""
        torch.manual_seed(42)
        tree = EMLTree1D(2)
        x = _as_real(np.linspace(0.5, 2.0, 10))
        a, _, _ = tree(x, tau=1.0)
        b, _, _ = tree(x, tau=1.0)
        assert torch.allclose(a.real, b.real)
        assert torch.allclose(a.imag, b.imag)


# ═══════════════════════ 6. Expression simplifier ═══════════════════════

class TestSimplifier:

    def test_eml_x_1_is_exp_x(self):
        assert _simplify("eml(x, 1)") == "exp(x)"

    def test_eml_1_1_is_e(self):
        # eml(1, 1) = exp(1) - ln(1) = e - 0 = e
        assert _simplify("eml(1, 1)") == "e"

    def test_eml_0_1_is_1(self):
        # eml(0, 1) = exp(0) - ln(1) = 1 - 0 = 1
        assert _simplify("eml(0, 1)") == "1"

    def test_eml_x_x_is_exp_minus_ln(self):
        # eml(x, x) = exp(x) - ln(x); simplifier shouldn't collapse this (non-trivial).
        got = _simplify("eml(x, x)")
        assert "exp(x)" in got and "ln(x)" in got

    def test_atom_passthrough(self):
        assert _simplify("x") == "x"
        assert _simplify("1") == "1"

    def test_malformed_expression_passthrough(self):
        # Simplifier has a defensive try/except — malformed input returns as-is.
        bad = "eml(x"
        assert _simplify(bad) == bad

    def test_nested_eml_simplifies(self):
        # eml(eml(x, 1), 1) = eml(exp(x), 1) = exp(exp(x)) - ln(1) = exp(exp(x))
        got = _simplify("eml(eml(x, 1), 1)")
        assert got == "exp(exp(x))"

    def test_simplify_is_fixed_point(self):
        expr = "eml(eml(x, 1), eml(1, 1))"
        once = _simplify(expr)
        twice = _simplify(once)
        assert once == twice
