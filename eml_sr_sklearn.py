"""sklearn-compatible wrapper around eml-sr for SRBench integration.

SRBench (La Cava et al. 2021) drives every regressor through a uniform
fit/predict interface and reads the discovered model as a string. This
module exposes `EMLRegressor`, a thin sklearn-style estimator that wraps
`eml_sr.discover` / `discover_curriculum` and stores the snapped torch
tree for prediction.

Limitation: eml-sr is *univariate* (Direction D from the Odrzywolek paper).
SRBench problems are typically multivariate. `fit(X, y)` will accept a
single column, or — when `auto_project=True` — pick the column with the
highest absolute Spearman correlation with y and treat it as the active
input. This is enough to put eml-sr on the leaderboard for univariate /
near-univariate problems and to flag the limitation cleanly for the rest.

Usage (SRBench-style)::

    est = EMLRegressor(max_depth=4, n_tries=16, n_workers=8)
    est.fit(X, y)
    yhat = est.predict(X_test)
    print(est.model_)            # symbolic expression in normalized coords
    print(est.original_model_)   # with the affine substitution applied
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from eml_sr import (
    DTYPE,
    REAL,
    Normalizer,
    discover,
    discover_curriculum,
)


def _spearman_abs(col: np.ndarray, y: np.ndarray) -> float:
    """Absolute Spearman rank correlation. Numpy-only, no scipy dependency."""
    if col.std() == 0 or y.std() == 0:
        return 0.0
    rc = np.argsort(np.argsort(col))
    ry = np.argsort(np.argsort(y))
    rc = rc - rc.mean()
    ry = ry - ry.mean()
    denom = math.sqrt((rc * rc).sum() * (ry * ry).sum())
    if denom == 0:
        return 0.0
    return abs(float((rc * ry).sum() / denom))


class EMLRegressor:
    """sklearn-style univariate symbolic regressor backed by eml-sr.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth to search.
    n_tries : int
        Random seeds per depth (or curriculum runs for `method='curriculum'`).
    method : {"discover", "curriculum"}
        Search strategy. "discover" = fixed-depth ladder. "curriculum" =
        leaf-splitting growing tree (better for deep formulas).
    normalize : {"minmax", "standard", "none"}
        Affine pre-scaling. EML overflows easily on raw data; "minmax" is
        the safest default for arbitrary CSVs.
    n_workers : int
        Parallel workers used by `discover` (curriculum is serial).
    auto_project : bool
        If `X` has more than one column, pick the column with the highest
        |Spearman| against `y` and use that as the univariate input.
    success_threshold : float
        MSE threshold below which a fit is reported as exact.
    verbose : bool
        Print progress to stdout.
    """

    def __init__(
        self,
        max_depth: int = 4,
        n_tries: int = 16,
        method: str = "discover",
        normalize: str = "minmax",
        n_workers: int = 1,
        auto_project: bool = True,
        success_threshold: float = 1e-10,
        verbose: bool = False,
    ):
        self.max_depth = max_depth
        self.n_tries = n_tries
        self.method = method
        self.normalize = normalize
        self.n_workers = n_workers
        self.auto_project = auto_project
        self.success_threshold = success_threshold
        self.verbose = verbose

    # --- sklearn API ---

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y row counts differ: {X.shape[0]} vs {y.shape[0]}")

        # Pick active column (univariate restriction).
        if X.shape[1] == 1:
            active = 0
        elif self.auto_project:
            scores = [_spearman_abs(X[:, j], y) for j in range(X.shape[1])]
            active = int(np.argmax(scores))
            if self.verbose:
                print(f"EMLRegressor: projecting onto column {active} "
                      f"(|spearman|={scores[active]:.3f})")
        else:
            raise ValueError(
                f"EMLRegressor is univariate; got X with {X.shape[1]} columns "
                "and auto_project=False. Set auto_project=True to pick the "
                "best column automatically, or pass a single column."
            )

        x = X[:, active]
        self.active_col_ = active
        self.normalizer_ = Normalizer.fit(x, y, mode=self.normalize)

        x_n = self.normalizer_.transform_x(x)
        y_n = self.normalizer_.transform_y(y)

        if self.method == "curriculum":
            result = discover_curriculum(
                x_n, y_n,
                max_depth=self.max_depth, n_tries=self.n_tries,
                verbose=self.verbose,
                success_threshold=self.success_threshold,
            )
        elif self.method == "discover":
            result = discover(
                x_n, y_n,
                max_depth=self.max_depth, n_tries=self.n_tries,
                verbose=self.verbose,
                success_threshold=self.success_threshold,
                n_workers=self.n_workers,
            )
        else:
            raise ValueError(f"unknown method {self.method!r}")

        self.result_ = result
        self.snapped_tree_ = result["snapped_tree"]
        self.depth_ = result["depth"]
        self.expr_ = result["expr"]
        self.snap_rmse_ = result["snap_rmse"]
        self.exact_ = bool(result.get("exact", True))
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("EMLRegressor: call fit() before predict()")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        x = X[:, self.active_col_]
        x_n = self.normalizer_.transform_x(x)
        x_t = torch.tensor(x_n, dtype=REAL)
        with torch.no_grad():
            pred, _, _ = self.snapped_tree_(x_t, tau=0.01)
            yp_n = pred.real.detach().cpu().numpy()
        return self.normalizer_.inverse_y(yp_n)

    def score(self, X, y):
        """R² coefficient of determination (sklearn convention)."""
        y = np.asarray(y, dtype=np.float64)
        yhat = self.predict(X)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0:
            return 0.0
        return 1.0 - ss_res / ss_tot

    # --- SRBench-style introspection ---

    @property
    def model_(self) -> str:
        """Symbolic expression as discovered, in normalized coordinates.

        SRBench reads `model_` (or `est.model_`) to evaluate solution
        complexity and for symbolic comparisons with the ground truth.
        """
        return getattr(self, "expr_", None)

    @property
    def original_model_(self) -> Optional[str]:
        """Symbolic expression with the normalizer's affine transform stitched in.

        Returns a string like `(formula(0.5*x - 1.0) - 2.3) / 0.7`. Useful
        for quick reading; not algebraically simplified.
        """
        if not getattr(self, "is_fitted_", False):
            return None
        import re
        n = self.normalizer_
        x_sub = f"({n.x_a:.6g} * x + {n.x_b:.6g})"
        # Word-boundary replace so we hit the standalone identifier `x`
        # without touching `exp` or future identifiers that contain x.
        formula = re.sub(r"\bx\b", x_sub, self.expr_)
        if n.y_a == 1.0 and n.y_b == 0.0:
            return formula
        return f"(({formula}) - {n.y_b:.6g}) / {n.y_a:.6g}"

    def get_params(self, deep: bool = True) -> dict:
        return {
            "max_depth": self.max_depth,
            "n_tries": self.n_tries,
            "method": self.method,
            "normalize": self.normalize,
            "n_workers": self.n_workers,
            "auto_project": self.auto_project,
            "success_threshold": self.success_threshold,
            "verbose": self.verbose,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"unknown parameter {k!r}")
            setattr(self, k, v)
        return self
