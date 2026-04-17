"""Tests for vectorized ``Normalizer`` (issue #17, Direction E step 3/6).

Covers the acceptance criteria from
https://github.com/oaustegard/eml-sr/issues/17:

  1. ``Normalizer.fit(x_1d, y)`` unchanged behaviour (byte-for-byte vs. scalar).
  2. ``Normalizer.fit(X_2d, y)`` computes per-column normalization.
  3. ``transform_x(X_2d)`` returns correct shapes.
  4. ``inverse_x(X_2d)`` round-trips.
  5. ``describe()`` and ``to_dict()`` report per-column stats.

Plus: constant-column handling, round-trips in both modes, and verification
that the 1D/scalar public surface is unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from eml_sr import Normalizer


# ─── 1D / backward-compat ────────────────────────────────────────────────

class TestNormalizer1DBackwardCompat:
    """Issue #17 acceptance criterion 1: 1D input behaves exactly as before."""

    @pytest.mark.parametrize("mode", ["minmax", "standard", "none"])
    def test_fit_stores_scalars(self, mode):
        rng = np.random.default_rng(0)
        x = rng.normal(size=50)
        y = rng.normal(size=50)
        norm = Normalizer.fit(x, y, mode=mode)
        assert isinstance(norm.x_a, float)
        assert isinstance(norm.x_b, float)
        assert isinstance(norm.y_a, float)
        assert isinstance(norm.y_b, float)
        assert norm.n_vars == 1

    def test_minmax_values_match_legacy(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        norm = Normalizer.fit(x, y, mode="minmax")
        # legacy: target [-1, 1] over [0, 4] → a=0.5, b=-1
        assert norm.x_a == pytest.approx(0.5)
        assert norm.x_b == pytest.approx(-1.0)
        # legacy: target [-1, 1] over [10, 50] → a=0.05, b=-1.5
        assert norm.y_a == pytest.approx(0.05)
        assert norm.y_b == pytest.approx(-1.5)

    def test_standard_values_match_legacy(self):
        rng = np.random.default_rng(42)
        x = rng.normal(loc=3.0, scale=2.0, size=200)
        y = rng.normal(loc=-5.0, scale=0.5, size=200)
        norm = Normalizer.fit(x, y, mode="standard")
        # x_a = 1/sd, x_b = -mu/sd  (scalar)
        assert norm.x_a == pytest.approx(1.0 / x.std())
        assert norm.x_b == pytest.approx(-x.mean() / x.std())

    def test_transform_inverse_roundtrip_1d(self):
        x = np.linspace(-3.0, 5.0, 40)
        y = np.linspace(100.0, 200.0, 40)
        norm = Normalizer.fit(x, y, mode="minmax")
        np.testing.assert_allclose(norm.inverse_x(norm.transform_x(x)), x)
        np.testing.assert_allclose(norm.inverse_y(norm.transform_y(y)), y)

    def test_constant_1d_x(self):
        x = np.full(10, 7.0)
        y = np.linspace(0.0, 1.0, 10)
        norm = Normalizer.fit(x, y, mode="minmax")
        assert norm.x_a == 0.0
        assert norm.x_b == 0.0
        # inverse_x with x_a == 0 returns zeros
        out = norm.inverse_x(np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(out, np.zeros(3))


# ─── 2D / per-column ─────────────────────────────────────────────────────

class TestNormalizer2D:
    """Issue #17 acceptance criteria 2–4: per-column normalization for 2D X."""

    @pytest.mark.parametrize("n_vars", [2, 3, 5])
    def test_fit_2d_minmax_shapes(self, n_vars):
        rng = np.random.default_rng(n_vars)
        X = rng.normal(size=(100, n_vars))
        y = rng.normal(size=100)
        norm = Normalizer.fit(X, y, mode="minmax")
        assert isinstance(norm.x_a, np.ndarray)
        assert isinstance(norm.x_b, np.ndarray)
        assert norm.x_a.shape == (n_vars,)
        assert norm.x_b.shape == (n_vars,)
        # y stays scalar
        assert isinstance(norm.y_a, float)
        assert isinstance(norm.y_b, float)
        assert norm.n_vars == n_vars

    def test_fit_2d_minmax_values(self):
        # Three columns with very different ranges.
        X = np.column_stack([
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),      # col 0: [0, 4]
            np.array([10.0, 20.0, 30.0, 40.0, 50.0]), # col 1: [10, 50]
            np.array([-1.0, -0.5, 0.0, 0.5, 1.0]),    # col 2: [-1, 1]
        ])
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        norm = Normalizer.fit(X, y, mode="minmax")
        # Each column maps its own [lo, hi] to [-1, 1].
        np.testing.assert_allclose(norm.x_a, [0.5, 0.05, 1.0])
        np.testing.assert_allclose(norm.x_b, [-1.0, -1.5, 0.0])

    def test_fit_2d_standard_values(self):
        rng = np.random.default_rng(7)
        X = np.column_stack([
            rng.normal(loc=0.0, scale=1.0, size=500),
            rng.normal(loc=10.0, scale=3.0, size=500),
            rng.normal(loc=-5.0, scale=0.1, size=500),
        ])
        y = rng.normal(size=500)
        norm = Normalizer.fit(X, y, mode="standard")
        np.testing.assert_allclose(norm.x_a, 1.0 / X.std(axis=0))
        np.testing.assert_allclose(norm.x_b, -X.mean(axis=0) / X.std(axis=0))

    def test_fit_2d_none_mode(self):
        X = np.random.default_rng(0).normal(size=(10, 4))
        y = np.random.default_rng(1).normal(size=10)
        norm = Normalizer.fit(X, y, mode="none")
        np.testing.assert_array_equal(norm.x_a, np.ones(4))
        np.testing.assert_array_equal(norm.x_b, np.zeros(4))
        assert norm.y_a == 1.0
        assert norm.y_b == 0.0

    def test_transform_x_2d_shape(self):
        X = np.random.default_rng(0).normal(size=(30, 3))
        y = np.random.default_rng(0).normal(size=30)
        norm = Normalizer.fit(X, y, mode="minmax")
        Xp = norm.transform_x(X)
        assert Xp.shape == X.shape

    def test_transform_x_2d_minmax_in_range(self):
        X = np.random.default_rng(0).uniform(-10.0, 10.0, size=(50, 4))
        y = np.random.default_rng(0).normal(size=50)
        norm = Normalizer.fit(X, y, mode="minmax", target_lo=-1.0, target_hi=1.0)
        Xp = norm.transform_x(X)
        # Each column should fill [-1, 1] exactly (bounded by min/max rows).
        np.testing.assert_allclose(Xp.min(axis=0), -1.0)
        np.testing.assert_allclose(Xp.max(axis=0), 1.0)

    def test_transform_x_2d_standard_moments(self):
        rng = np.random.default_rng(11)
        X = np.column_stack([
            rng.normal(loc=100.0, scale=50.0, size=1000),
            rng.normal(loc=-20.0, scale=3.0, size=1000),
        ])
        y = rng.normal(size=1000)
        norm = Normalizer.fit(X, y, mode="standard")
        Xp = norm.transform_x(X)
        np.testing.assert_allclose(Xp.mean(axis=0), np.zeros(2), atol=1e-12)
        np.testing.assert_allclose(Xp.std(axis=0), np.ones(2), atol=1e-12)

    @pytest.mark.parametrize("mode", ["minmax", "standard", "none"])
    @pytest.mark.parametrize("n_vars", [2, 3, 5])
    def test_roundtrip_2d(self, mode, n_vars):
        rng = np.random.default_rng(mode.__hash__() % (1 << 31) + n_vars)
        X = rng.normal(size=(80, n_vars))
        y = rng.normal(size=80)
        norm = Normalizer.fit(X, y, mode=mode)
        X_rt = norm.inverse_x(norm.transform_x(X))
        np.testing.assert_allclose(X_rt, X, atol=1e-10)


# ─── Constant columns (2D) ──────────────────────────────────────────────

class TestNormalizerConstantColumns:
    """A constant column should collapse to (0, 0) without breaking others."""

    def test_mixed_constant_and_varying_columns_minmax(self):
        X = np.column_stack([
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),  # varies
            np.full(5, 7.0),                       # constant
            np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), # varies
        ])
        y = np.linspace(0.0, 1.0, 5)
        norm = Normalizer.fit(X, y, mode="minmax")
        # Constant column → (0, 0); others have valid affine params.
        np.testing.assert_allclose(norm.x_a, [0.5, 0.0, 0.5])
        np.testing.assert_allclose(norm.x_b, [-1.0, 0.0, 0.0])

        Xp = norm.transform_x(X)
        # Constant column transformed to all zeros.
        np.testing.assert_array_equal(Xp[:, 1], np.zeros(5))
        # Others map to [-1, 1].
        np.testing.assert_allclose(Xp[:, 0].min(), -1.0)
        np.testing.assert_allclose(Xp[:, 0].max(), 1.0)
        np.testing.assert_allclose(Xp[:, 2].min(), -1.0)
        np.testing.assert_allclose(Xp[:, 2].max(), 1.0)

    def test_mixed_constant_and_varying_columns_standard(self):
        rng = np.random.default_rng(0)
        X = np.column_stack([
            rng.normal(size=20),
            np.full(20, -3.14),  # constant
        ])
        y = rng.normal(size=20)
        norm = Normalizer.fit(X, y, mode="standard")
        assert norm.x_a[1] == 0.0
        assert norm.x_b[1] == 0.0
        # Column 0 normalizes as usual.
        assert norm.x_a[0] == pytest.approx(1.0 / X[:, 0].std())

    def test_inverse_x_with_constant_column_returns_zero_for_that_col(self):
        X = np.column_stack([
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.full(4, 5.0),  # constant
        ])
        y = np.zeros(4)
        norm = Normalizer.fit(X, y, mode="minmax")
        Xp = norm.transform_x(X)
        X_back = norm.inverse_x(Xp)
        # Non-constant column round-trips.
        np.testing.assert_allclose(X_back[:, 0], X[:, 0])
        # Constant column: inverse returns 0 (can't recover the constant —
        # matches the documented 1D behavior when x_a == 0).
        np.testing.assert_array_equal(X_back[:, 1], np.zeros(4))


# ─── describe() / to_dict() ──────────────────────────────────────────────

class TestNormalizerReporting:
    """Issue #17 acceptance criterion 5."""

    def test_describe_1d_unchanged(self):
        x = np.linspace(0.0, 10.0, 11)
        y = np.linspace(0.0, 1.0, 11)
        norm = Normalizer.fit(x, y, mode="minmax")
        s = norm.describe()
        assert "x' =" in s
        assert "y' =" in s
        assert "mode=minmax" in s
        # Should NOT mention per-column notation.
        assert "x1'" not in s
        assert "x2'" not in s

    def test_describe_2d_contains_per_column(self):
        X = np.random.default_rng(0).normal(size=(20, 3))
        y = np.random.default_rng(0).normal(size=20)
        norm = Normalizer.fit(X, y, mode="minmax")
        s = norm.describe()
        for i in range(1, 4):
            assert f"x{i}'" in s, f"describe() missing x{i}': {s}"
        assert "mode=minmax" in s

    def test_to_dict_1d_scalars(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        d = Normalizer.fit(x, y, mode="minmax").to_dict()
        assert isinstance(d["x_a"], float)
        assert isinstance(d["x_b"], float)
        assert isinstance(d["y_a"], float)
        assert isinstance(d["y_b"], float)
        assert d["mode"] == "minmax"

    def test_to_dict_2d_lists(self):
        X = np.random.default_rng(0).normal(size=(20, 3))
        y = np.random.default_rng(0).normal(size=20)
        d = Normalizer.fit(X, y, mode="standard").to_dict()
        assert isinstance(d["x_a"], list)
        assert isinstance(d["x_b"], list)
        assert len(d["x_a"]) == 3
        assert len(d["x_b"]) == 3
        # y still scalar.
        assert isinstance(d["y_a"], float)
        assert isinstance(d["y_b"], float)

    def test_to_dict_2d_json_serializable(self):
        import json
        X = np.random.default_rng(0).normal(size=(10, 2))
        y = np.random.default_rng(0).normal(size=10)
        d = Normalizer.fit(X, y, mode="minmax").to_dict()
        s = json.dumps(d)  # must not raise
        restored = json.loads(s)
        assert restored["mode"] == "minmax"
        assert len(restored["x_a"]) == 2


# ─── Input validation ────────────────────────────────────────────────────

class TestNormalizerValidation:

    def test_fit_rejects_3d_x(self):
        X = np.zeros((4, 2, 3))
        y = np.zeros(4)
        with pytest.raises(ValueError):
            Normalizer.fit(X, y, mode="minmax")

    def test_fit_accepts_column_vector_y(self):
        # (n, 1) y should be accepted and treated as (n,).
        x = np.linspace(0.0, 1.0, 10)
        y = np.linspace(0.0, 10.0, 10).reshape(-1, 1)
        norm = Normalizer.fit(x, y, mode="minmax")
        assert isinstance(norm.y_a, float)

    def test_init_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError):
            Normalizer(np.array([1.0, 2.0]), np.array([3.0]), 1.0, 0.0, "minmax")

    def test_init_rejects_2d_x_params(self):
        with pytest.raises(ValueError):
            Normalizer(np.zeros((2, 2)), np.zeros((2, 2)), 1.0, 0.0, "minmax")

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            Normalizer.fit(np.zeros(5), np.zeros(5), mode="quantile")
