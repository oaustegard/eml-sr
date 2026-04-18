"""Test EMLRegressor native multivariate support (issue #19).

#19 changes:
    * `auto_project` default flips to `False`; native multivariate is the
      new default path.
    * `fit(X_2d, y)` routes all columns through the multivariate forward
      pass + per-column Normalizer.
    * `predict(X_2d)` routes through the snapped multivariate tree.
    * `model_` is the raw expression string (unchanged shape).
    * `original_model_` back-substitutes `xi → (a_i * xi + b_i)` for each
      column.
    * `n_vars_` records the fit dimensionality.
    * Legacy `auto_project=True` path preserves the old single-column
      behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from eml_sr_sklearn import EMLRegressor


# ─── Backward compatibility: univariate still works ─────────────────────

class TestUnivariateBackcompat:
    """1D x and single-column 2D X should behave as before #19."""

    def test_1d_x_fit_predict(self):
        x = np.linspace(0.2, 2.0, 24)
        y = np.exp(x)
        est = EMLRegressor(max_depth=2, n_tries=2, method="curriculum",
                           normalize="none", verbose=False)
        est.fit(x, y)
        assert est.n_vars_ == 1
        assert est.active_col_ is None  # native path, no projection
        yhat = est.predict(x)
        assert yhat.shape == (24,)
        # Univariate expr should use 'x' (not 'x1').
        assert "x1" not in est.expr_
        assert "x" in est.expr_ or "1" == est.expr_

    def test_single_column_2d_x(self):
        """X shape (n, 1) is also univariate; same result as (n,)."""
        x = np.linspace(0.2, 2.0, 24)
        y = np.exp(x)
        est = EMLRegressor(max_depth=2, n_tries=2, method="curriculum",
                           normalize="none", verbose=False)
        est.fit(x.reshape(-1, 1), y)
        assert est.n_vars_ == 1
        yhat = est.predict(x.reshape(-1, 1))
        assert yhat.shape == (24,)


# ─── Native multivariate (default path) ─────────────────────────────────

class TestNativeMultivariate:
    """auto_project=False (the new default) — all columns used."""

    def test_default_is_native(self):
        est = EMLRegressor()
        assert est.auto_project is False

    def test_multicol_fit_uses_all_columns(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0.3, 1.5, size=(32, 3))
        y = X[:, 1]  # y depends on x2 only; depth-0 atom should nail it.
        est = EMLRegressor(max_depth=1, n_tries=2, method="curriculum",
                           normalize="none", verbose=False)
        est.fit(X, y)
        assert est.n_vars_ == 3
        assert est.active_col_ is None  # not projected down

    def test_depth0_x2_recovery(self):
        """Native path: curriculum finds y = x2 at depth 0."""
        rng = np.random.default_rng(1)
        X = rng.uniform(0.3, 1.5, size=(32, 2))
        y = X[:, 1]
        est = EMLRegressor(max_depth=1, n_tries=1, method="curriculum",
                           normalize="none", verbose=False)
        est.fit(X, y)
        assert est.n_vars_ == 2
        assert est.exact_ is True
        # Expression should reference x2.
        assert "x2" in est.expr_
        # Prediction should match y within normalizer round-trip.
        yhat = est.predict(X)
        np.testing.assert_allclose(yhat, y, atol=1e-8)

    def test_predict_shape(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(0.3, 1.5, size=(20, 2))
        y = X[:, 0] + 0.5 * X[:, 1]
        est = EMLRegressor(max_depth=1, n_tries=1, method="curriculum",
                           normalize="none", verbose=False)
        est.fit(X, y)
        yhat = est.predict(X)
        assert yhat.shape == (20,)
        # score() returns a float.
        assert isinstance(est.score(X, y), float)

    def test_predict_rejects_wrong_width(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(0.3, 1.5, size=(16, 3))
        y = X[:, 0]
        est = EMLRegressor(max_depth=1, n_tries=1, method="curriculum",
                           normalize="none", verbose=False)
        est.fit(X, y)
        wrong = rng.uniform(0.3, 1.5, size=(10, 2))
        with pytest.raises(ValueError, match="columns"):
            est.predict(wrong)

    def test_model_has_multivar_expr(self):
        rng = np.random.default_rng(4)
        X = rng.uniform(0.3, 1.5, size=(24, 2))
        y = X[:, 1]
        est = EMLRegressor(max_depth=1, n_tries=1, method="curriculum",
                           normalize="none", verbose=False)
        est.fit(X, y)
        assert "x2" in est.model_


# ─── Legacy auto_project=True path ─────────────────────────────────────

class TestAutoProjectLegacy:
    """auto_project=True should reproduce the pre-#19 single-column behaviour."""

    def test_auto_project_picks_best_column(self):
        rng = np.random.default_rng(5)
        x1 = rng.uniform(0.3, 1.5, 40)
        x2 = rng.uniform(0.3, 1.5, 40)
        X = np.column_stack([x1, x2])
        # y correlates strongly with x2, nothing with x1.
        y = np.exp(x2)
        est = EMLRegressor(max_depth=2, n_tries=2, method="curriculum",
                           normalize="none", auto_project=True, verbose=False)
        est.fit(X, y)
        assert est.n_vars_ == 1
        assert est.active_col_ == 1  # picked x2
        # Expression should use 'x' (univariate).
        assert "x1" not in est.expr_
        assert "x2" not in est.expr_

    def test_auto_project_predict_subsets(self):
        """predict() honours the projection stored at fit time."""
        rng = np.random.default_rng(6)
        X = np.column_stack([rng.uniform(0.5, 2.0, 30),
                             rng.uniform(0.5, 2.0, 30)])
        y = np.exp(X[:, 1])
        est = EMLRegressor(max_depth=1, n_tries=2, method="curriculum",
                           normalize="none", auto_project=True, verbose=False)
        est.fit(X, y)
        yhat = est.predict(X)
        assert yhat.shape == (30,)


# ─── original_model_ per-column affine back-substitution ────────────────

class TestOriginalModel:
    """original_model_ replaces each xi with its per-column affine."""

    def test_univariate_original_model_uses_x(self):
        x = np.linspace(0.5, 2.0, 24)
        y = np.exp(x)
        est = EMLRegressor(max_depth=1, n_tries=2, method="curriculum",
                           normalize="minmax", verbose=False)
        est.fit(x, y)
        s = est.original_model_
        assert s is not None
        # Should contain a scalar affine for x: "(a * x + b)" pattern.
        assert " * x " in s or "* x +" in s

    def test_multivariate_original_model_per_column(self):
        """Each xi gets its own (a_i * xi + b_i) substitution."""
        rng = np.random.default_rng(7)
        X = rng.uniform(0.3, 1.5, size=(32, 2))
        y = X[:, 1]
        est = EMLRegressor(max_depth=1, n_tries=1, method="curriculum",
                           normalize="minmax", verbose=False)
        est.fit(X, y)
        s = est.original_model_
        assert s is not None
        # The discovered expr is literally 'x2' (depth-0 atom). After
        # substitution, original_model_ should embed (a2 * x2 + b2) then
        # apply the inverse y-scale around that.
        assert "* x2" in s
        # x1 may or may not appear depending on the formula — here the
        # formula uses only x2, so x1 shouldn't appear in the substitution.
        # But we only guarantee: every xi that appeared in expr_ was
        # substituted.
        for i in range(1, est.n_vars_ + 1):
            name = f"x{i}"
            if name in est.expr_:
                assert f"* {name}" in s, f"{name} not substituted in {s!r}"

    def test_longest_first_substitution(self):
        """x1 substitution must not partially match x10 or similar."""
        # Fit on 10 columns to ensure x10 appears.
        rng = np.random.default_rng(8)
        X = rng.uniform(0.3, 1.5, size=(40, 10))
        y = X[:, 9]  # x10
        est = EMLRegressor(max_depth=1, n_tries=1, method="curriculum",
                           normalize="none", verbose=False)
        est.fit(X, y)
        # Expression should contain x10. After substitution, there must
        # not be an orphan 'x1' remaining unsubstituted inside the x10
        # substitution (i.e. no bare 'x1' left outside a '* x1 +' pattern).
        assert "x10" in est.expr_
        s = est.original_model_
        assert "x10" in s
        # Sanity: the substitution for x10 should appear.
        assert "* x10" in s


# ─── get_params / set_params ────────────────────────────────────────────

class TestParams:
    def test_get_params_has_auto_project(self):
        est = EMLRegressor()
        p = est.get_params()
        assert "auto_project" in p
        assert p["auto_project"] is False  # new default

    def test_set_params_auto_project(self):
        est = EMLRegressor()
        est.set_params(auto_project=True)
        assert est.auto_project is True
