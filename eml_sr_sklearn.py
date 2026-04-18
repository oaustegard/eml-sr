"""sklearn-compatible wrapper around eml-sr for SRBench integration.

SRBench (La Cava et al. 2021) drives every regressor through a uniform
fit/predict interface and reads the discovered model as a string. This
module exposes `EMLRegressor`, a thin sklearn-style estimator that wraps
`eml_sr.discover` / `discover_curriculum` and stores the snapped torch
tree for prediction.

eml-sr natively supports multivariate inputs via the Odrzywolek §4.3
master formula (PR #15 for the forward pass, #18 for curriculum growing,
#17 for per-column normalization). `EMLRegressor` passes `X` with any
number of columns straight through by default.

The legacy univariate-projection behaviour is still available via
``auto_project=True``: when set, `EMLRegressor` picks the single column
of `X` with the highest absolute Spearman correlation against `y` and
discards the rest. This used to be the default; as of #19 it is off by
default but available for explicit univariate-only runs.

Usage (SRBench-style)::

    est = EMLRegressor(max_depth=4, n_tries=16, n_workers=8)
    est.fit(X, y)                # X can be (n,) or (n, n_vars)
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
    """sklearn-style symbolic regressor backed by eml-sr.

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
        the safest default for arbitrary CSVs. Per-column affines are used
        for multi-variable `X` (see `Normalizer` and PR #17).
    n_workers : int
        Parallel workers used by `discover` (curriculum is serial).
    auto_project : bool
        Legacy univariate mode. If `True` and `X` has more than one column,
        pick the column with the highest |Spearman| against `y` and use
        that single column as the input. Default is `False` — `X` is
        passed through natively as a multivariate input, using the
        multivariate forward pass added in #15.
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
        auto_project: bool = False,
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

        # Legacy path: project multi-column X down to a single column via
        # |Spearman| against y. Off by default; kept for explicit runs.
        if X.shape[1] > 1 and self.auto_project:
            scores = [_spearman_abs(X[:, j], y) for j in range(X.shape[1])]
            active = int(np.argmax(scores))
            if self.verbose:
                print(f"EMLRegressor: auto_project → column {active} "
                      f"(|spearman|={scores[active]:.3f})")
            X = X[:, active:active + 1]
            self.active_col_ = active
        else:
            # Native multivariate path. active_col_ set to None to signal
            # "used all columns"; predict() keys off its type to decide
            # whether to subset.
            self.active_col_ = None

        self.n_vars_ = X.shape[1]
        # Fit a normalizer over either (n,) (n_vars=1, preserving the
        # legacy scalar surface) or (n, n_vars). The #17 Normalizer picks
        # the right internal representation.
        if self.n_vars_ == 1:
            self.normalizer_ = Normalizer.fit(X[:, 0], y, mode=self.normalize)
            x_n = self.normalizer_.transform_x(X[:, 0])
        else:
            self.normalizer_ = Normalizer.fit(X, y, mode=self.normalize)
            x_n = self.normalizer_.transform_x(X)
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

        # Route through the stored normalizer + snapped tree. If fit()
        # projected down to a single column (auto_project=True), honour
        # that projection; otherwise pass all columns through.
        if self.active_col_ is not None:
            x_n = self.normalizer_.transform_x(X[:, self.active_col_])
            x_t = torch.tensor(x_n, dtype=REAL)
        else:
            if X.shape[1] != self.n_vars_:
                raise ValueError(
                    f"predict: X has {X.shape[1]} columns, "
                    f"fit was on {self.n_vars_}"
                )
            if self.n_vars_ == 1:
                x_n = self.normalizer_.transform_x(X[:, 0])
                x_t = torch.tensor(x_n, dtype=REAL)
            else:
                x_n = self.normalizer_.transform_x(X)
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

        For a univariate fit (one active column), returns a string like
        ``(formula(0.5*x - 1.0) - 2.3) / 0.7``. For a multivariate fit,
        each ``x_k`` is replaced with its per-column affine
        ``(a_k*x_k + b_k)``. Not algebraically simplified — useful for
        quick reading rather than downstream symbolic work.
        """
        if not getattr(self, "is_fitted_", False):
            return None
        import re
        n = self.normalizer_

        # Build per-variable substitution map.
        #
        # Univariate fit: the model uses `x`. Substitute `x` with the
        # scalar affine. (Legacy surface, byte-for-byte.)
        #
        # Multivariate fit: the model uses `x1, x2, ...`. Each is
        # substituted with its own (a_k, b_k). Regex is applied
        # longest-name-first so `x10` doesn't match `x1` mid-token.
        subs = {}
        if self.n_vars_ == 1:
            # x_a/x_b are scalars in the 1D Normalizer path (see #17).
            a = float(n.x_a if isinstance(n.x_a, float) else n.x_a[0])
            b = float(n.x_b if isinstance(n.x_b, float) else n.x_b[0])
            subs["x"] = f"({a:.6g} * x + {b:.6g})"
        else:
            # x_a/x_b are 1D arrays of length n_vars_.
            for i in range(self.n_vars_):
                a = float(n.x_a[i])
                b = float(n.x_b[i])
                subs[f"x{i+1}"] = f"({a:.6g} * x{i+1} + {b:.6g})"

        formula = self.expr_
        # Longest-first substitution prevents "x1" from partially matching
        # "x10" (or "x" from matching "x1"/"exp").
        for name in sorted(subs, key=len, reverse=True):
            pattern = r"\b" + re.escape(name) + r"\b"
            formula = re.sub(pattern, subs[name], formula)

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
