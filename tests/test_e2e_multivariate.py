"""End-to-end multivariate tests for issue #20 (Direction E [6/6]).

These tests complement the existing multivariate suites:

    - tests/test_multivariate.py (#15)          — tree construction, shapes,
                                                   snap, hand-crafted leaves
    - tests/test_curriculum_multivariate.py     — GrowingEMLTree, per-variable
                                                   split routing, curriculum
                                                   recovery of eml(x1, x2)
    - tests/test_api_2d.py (#16)                — public API 2D shim
                                                   (x → X, shape validation,
                                                   n_vars in result dict)
    - tests/test_normalizer_multivariate.py(#17) — per-column Normalizer
    - tests/test_multivariate.py (sklearn, #19) — see test_eml_sr_linear.py
                                                   / tests for sklearn wrapper

What's new here (issue #20):

    1. End-to-end recovery via ``discover`` (Option A, not curriculum)
       on a genuinely bivariate target — training run, not hand-crafted.
    2. End-to-end recovery via ``discover_hybrid`` on the same target,
       verifying the hybrid dispatcher picks Option A and returns a
       symbolically-clean expression.
    3. Snap-tree callability and prediction fidelity on 2D input —
       a recovered tree, re-run against fresh X, matches y to machine
       precision.
    4. Smoke test for the multivariate Feynman benchmark runner
       (``_run_one_mv`` in ``benchmarks.feynman``): the dataclass +
       runner accept multivariate problems without exploding.

These tests train real models. They are not marked ``slow`` because
depth-1 recovery of ``eml(x1, x2)`` fits comfortably in a single-digit
number of seconds on CPU; empirically ~7s with ``n_tries=4``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from eml_sr import DTYPE, REAL, discover
from eml_sr_hybrid import discover_hybrid
from eml_sr_linear import discover_linear


# ═══════════════════════ Helpers ══════════════════════════════════

def _bivariate_eml_target(n=100, seed=0):
    """eml(x1, x2) = exp(x1) - ln(x2) — the depth-1 bivariate atom.

    Ranges chosen so that exp(x1) stays modest (x1 ≤ 2.5 ⇒ exp ≤ 12.2)
    and ln(x2) stays well-defined (x2 ∈ [0.5, 3.0]).
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.5, 2.5, n)
    x2 = rng.uniform(0.5, 3.0, n)
    X = np.stack([x1, x2], axis=1)
    y = np.exp(x1) - np.log(x2)
    return X, y


# ═══════════════════════ discover() recovery ═══════════════════════

class TestDiscoverRecoveryMultivariate:
    """Option A end-to-end recovery on bivariate inputs."""

    def test_discover_recovers_eml_x1_x2_depth1(self):
        """discover(X, y) with n_vars=2 recovers eml(x1, x2) at depth 1."""
        X, y = _bivariate_eml_target()
        result = discover(X, y, max_depth=1, n_tries=4, verbose=False)
        assert result is not None, "discover returned None on bivariate eml"
        assert result["depth"] == 1
        assert result["n_vars"] == 2
        # Machine-precision recovery (eml(x1, x2) is the literal depth-1 atom).
        assert result["snap_rmse"] < 1e-10

    def test_discover_snapped_tree_predicts_on_fresh_2d(self):
        """Snapped tree from discover() is callable on fresh 2D input."""
        X, y = _bivariate_eml_target()
        result = discover(X, y, max_depth=1, n_tries=4, verbose=False)
        assert result is not None
        snapped = result["snapped_tree"]

        # Re-evaluate on a different 2D tensor.
        X2, y2 = _bivariate_eml_target(n=50, seed=1)
        X2_t = torch.tensor(X2, dtype=REAL)
        with torch.no_grad():
            pred, _, _ = snapped(X2_t, tau=0.01)
        pred_np = pred.real.detach().numpy()
        rmse = float(np.sqrt(np.mean((pred_np - y2) ** 2)))
        assert rmse < 1e-10


# ═══════════════════════ discover_hybrid() dispatch ════════════════

class TestDiscoverHybridMultivariate:
    """Hybrid dispatcher on bivariate inputs: Option A should win here."""

    def test_hybrid_picks_option_a_on_depth1_bivariate(self):
        """For a clean depth-1 target, hybrid returns via option_a."""
        X, y = _bivariate_eml_target()
        result = discover_hybrid(
            X, y,
            max_depth=1, n_tries_a=4, n_tries_b=2,
            verbose=False,
        )
        assert result is not None
        # Option A is supposed to succeed cleanly on this; falling back
        # would signal a regression in the dispatcher's threshold logic.
        assert result["method"] == "option_a"
        assert result["n_vars"] == 2
        assert result["snap_rmse"] < 1e-10
        # Expression should reference both variables.
        assert "x1" in result["expr"]
        assert "x2" in result["expr"]

    def test_hybrid_result_has_multivariate_contract(self):
        """Hybrid result dict has all required multivariate fields."""
        X, y = _bivariate_eml_target(n=50)
        result = discover_hybrid(
            X, y,
            max_depth=1, n_tries_a=2, n_tries_b=1,
            verbose=False,
        )
        assert result is not None
        for key in ("expr", "depth", "snap_rmse", "snapped_tree",
                    "n_vars", "method"):
            assert key in result, f"missing result key: {key}"
        assert result["method"] in ("option_a", "warm_start_a", "option_b")


# ═══════════════════════ Linear tree on bivariate addition ═════════

class TestDiscoverLinearMultivariate:
    """Option B on a target that depends on a linear combination of inputs.

    ``exp(x1 + x2)`` is NOT in Option A's depth-1 vocabulary — there is no
    addition atom. Option B's affine leaves (α + β₁x₁ + β₂x₂) *can*
    represent the inner sum, so a depth-1 B tree should reach it.
    """

    def test_linear_fits_exp_of_sum(self):
        """Option B depth-1 should achieve low snap RMSE on exp(x1+x2)."""
        rng = np.random.default_rng(0)
        n = 100
        x1 = rng.uniform(0.0, 1.0, n)
        x2 = rng.uniform(0.0, 1.0, n)
        X = np.stack([x1, x2], axis=1)
        y = np.exp(x1 + x2)

        result = discover_linear(
            X, y, max_depth=1, n_tries=4, verbose=False,
        )
        assert result is not None
        assert result["n_vars"] == 2
        # Option B is numerical, not symbolic. We accept a loose RMSE
        # because the affine leaves should close on x1+x2 under gradient
        # descent but need not hit float64 precision in 4 tries. The
        # point is it's *reachable* by Option B; a catastrophic failure
        # (rmse ~ O(y)) signals a regression in the 2D linear path.
        # y has range exp(0)..exp(2) ≈ 1..7.4; 0.15 is ~2% of signal.
        assert result["snap_rmse"] < 0.15


# ═══════════════════════ Feynman benchmark (multivariate) ══════════

class TestFeynmanMultivariateRunner:
    """Smoke tests for the multivariate additions to benchmarks.feynman."""

    def test_feynman_problem_multivariate_dataclass(self):
        """FeynmanProblem accepts n_vars + x_ranges for multivariate."""
        from benchmarks.feynman import FeynmanProblem

        p = FeynmanProblem(
            feynman_id="TEST.mv",
            name="bivariate-eml",
            formula="exp(x1) - ln(x2)",
            fn=lambda X: np.exp(X[:, 0]) - np.log(X[:, 1]),
            x_ranges=[(0.5, 2.5), (0.5, 3.0)],
            n_vars=2,
            n_samples=50,
        )
        assert p.n_vars == 2
        assert len(p.x_ranges) == 2

    def test_feynman_problem_univariate_backward_compat(self):
        """Univariate FeynmanProblem (x_range tuple) still works."""
        from benchmarks.feynman import FeynmanProblem

        p = FeynmanProblem(
            feynman_id="TEST.uv",
            name="univariate",
            formula="exp(x)",
            fn=lambda x: np.exp(x),
            x_range=(0.5, 2.5),
        )
        # n_vars defaults to 1; x_ranges is synthesized from x_range.
        assert p.n_vars == 1
        # Sampling (done via _sample_X) produces 1D for backward compat.

    def test_feynman_mv_sampling_shape(self):
        """Multivariate sampler returns X of shape (n_samples, n_vars)."""
        from benchmarks.feynman import FeynmanProblem, _sample_X

        p = FeynmanProblem(
            feynman_id="TEST.mv",
            name="bivariate",
            formula="exp(x1) - ln(x2)",
            fn=lambda X: np.exp(X[:, 0]) - np.log(X[:, 1]),
            x_ranges=[(0.5, 2.5), (0.5, 3.0)],
            n_vars=2,
            n_samples=40,
        )
        X = _sample_X(p)
        assert X.shape == (40, 2)
        # Columns should lie in the declared ranges.
        assert 0.5 <= X[:, 0].min() <= X[:, 0].max() <= 2.5
        assert 0.5 <= X[:, 1].min() <= X[:, 1].max() <= 3.0

    def test_feynman_uv_sampling_shape(self):
        """Univariate sampler returns 1D x (backward compat)."""
        from benchmarks.feynman import FeynmanProblem, _sample_X

        p = FeynmanProblem(
            feynman_id="TEST.uv",
            name="univariate",
            formula="exp(x)",
            fn=lambda x: np.exp(x),
            x_range=(0.5, 2.5),
            n_samples=30,
        )
        x = _sample_X(p)
        assert x.ndim == 1
        assert x.shape == (30,)

    def test_run_one_mv_executes(self):
        """_run_one on a bivariate problem returns a result dict."""
        from benchmarks.feynman import FeynmanProblem, _run_one

        p = FeynmanProblem(
            feynman_id="TEST.mv.eml",
            name="depth1-bivariate",
            formula="exp(x1) - ln(x2)",
            fn=lambda X: np.exp(X[:, 0]) - np.log(X[:, 1]),
            x_ranges=[(0.5, 2.5), (0.5, 3.0)],
            n_vars=2,
            n_samples=40,
        )
        r = _run_one(
            p,
            max_depth=1, n_tries=2, method="curriculum",
            normalize="none", n_workers=1, threshold=1e-6,
        )
        assert "ok" in r
        assert "elapsed" in r
        assert "rmse" in r
        assert "depth" in r
        # Depth-1 eml(x1, x2) should actually recover with only n_tries=2
        # since it is the literal leaf atom; if this flakes in CI we'd
        # bump n_tries or mark slow. Empirically reliable on CPU.
        assert r["ok"] is True, f"expected recovery, got {r}"
        assert math.isfinite(r["rmse"])
        assert r["rmse"] < 1e-6
