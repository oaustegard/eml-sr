"""Feynman-equation benchmark for eml-sr.

The AI Feynman dataset (Udrescu & Tegmark 2020,
https://github.com/SJ001/AI-Feynman) is a collection of ~100 physics
formulas culled from the Feynman Lectures. Most are multivariate. eml-sr
is univariate (Direction D from the Odrzywolek paper), so this benchmark
covers two slices:

  1. Genuinely univariate Feynman equations (e.g. I.6.2 — Gaussian).
  2. Univariate *projections* of multivariate equations: fix all variables
     except one to representative constants and let eml-sr discover the
     resulting one-dimensional shape.

Each test case is generated synthetically rather than downloaded — the
real Feynman CSVs are 1M+ rows each and most pertain to the multivariate
case. The synthetic generators here use the identical functional forms
listed in the Feynman Lectures, drawn over physically reasonable ranges.

The catalogue was retuned in issue #11 after the original (issue #6) run
revealed that polynomial/rational targets like 1/x², 2x/(1+x²), and the
gaussian sit far outside EML's reachable vocabulary at depth ≤ 5 from
random init. The current selection keeps:

  - Linear targets (which expose the depth-1 identity-gap finding —
    EML cannot output `y = x` until depth ≥ 4 because every output is
    wrapped in an outer `eml(·,·)`)
  - exp / ln chains (EML's natural wheelhouse)
  - The depth-1 EML shape eml(x, x) = exp(x) - ln(x)

The defaults were also flipped to `--method curriculum --normalize none`.
Both `minmax` and `standard` normalization destroy symbolic recoverability
for nonlinear targets — the affine transform of an elementary function is
generally not itself elementary in the EML vocabulary, so the engine ends
up fitting an unreachable shape. The runner now reports RMSE in *original*
coordinate space (not normalized space, as the original implementation
did, which was misleading). See issue #11 for the full diagnosis.

Usage::

    python -m benchmarks.feynman                  # quick: 8 problems
    python -m benchmarks.feynman --all            # all problems
    python -m benchmarks.feynman --workers 8      # parallel seeds
    python -m benchmarks.feynman --normalize minmax  # for huge y ranges
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

# Allow running as a script from the repo root or as `python -m benchmarks.feynman`.
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eml_sr import REAL, discover, discover_curriculum, Normalizer  # noqa: E402


# ─── Problem catalogue ─────────────────────────────────────────
#
# Each problem captures: the Feynman ID, a generator (x → y), the sample
# range, and the human-readable target formula. We avoid fancy multivariate
# projections because most Feynman equations have several active variables
# whose constants are not standardized — instead we list ones with clean
# univariate slices.

@dataclass
class FeynmanProblem:
    feynman_id: str
    name: str           # short readable label
    formula: str        # human-readable formula in terms of x
    fn: Callable[[np.ndarray], np.ndarray]
    x_range: tuple      # (lo, hi)
    n_samples: int = 100
    notes: str = ""


# Helpers used by several problems ------------------------------------
_SQRT_TWO_PI = math.sqrt(2 * math.pi)


def _gaussian(theta):
    return np.exp(-(theta ** 2) / 2) / _SQRT_TWO_PI


def _shifted_gaussian(theta, sigma=1.0, theta1=0.0):
    return (1.0 / (sigma * _SQRT_TWO_PI)) * np.exp(
        -((theta - theta1) ** 2) / (2 * sigma ** 2)
    )


def _projection_const(c):
    """Constant projection — useful for testing 'recover a constant'."""
    return lambda x: np.full_like(x, c, dtype=np.float64)


# Problem catalogue --------------------------------------------------
# References are to the equation numbers in the original Feynman Lectures
# / AI Feynman repo. "Univariate" means the equation has one active var;
# "Projection" means a multivariate equation evaluated at fixed constants.

PROBLEMS: list[FeynmanProblem] = [
    # ── EML's natural vocabulary (depth 1) ──────────────────────
    FeynmanProblem(
        "eml.exp", "exp", "exp(x)",
        lambda x: np.exp(x), x_range=(0.5, 2.5),
        notes="depth-1 atom: eml(x, 1) = exp(x)",
    ),
    FeynmanProblem(
        "eml.eml", "exp-minus-ln", "exp(x) - ln(x)",
        lambda x: np.exp(x) - np.log(x), x_range=(0.5, 3.0),
        notes="depth-1 atom: the raw eml(x, x) shape",
    ),
    FeynmanProblem(
        "eml.const-e", "constant-e", "e",
        _projection_const(math.e), x_range=(0.5, 2.5),
        notes="depth-1 atom: eml(1, 1) = e",
    ),
    FeynmanProblem(
        "eml.e-ln", "e-minus-ln", "e - ln(x)",
        lambda x: math.e - np.log(x), x_range=(0.5, 3.0),
        notes="depth-1 atom: eml(1, x)",
    ),

    # ── exp / ln chains (EML's wheelhouse) ──────────────────────
    FeynmanProblem(
        "eml.lnx", "ln", "ln(x)",
        lambda x: np.log(x), x_range=(0.5, 5.0),
        notes="known to require depth 3 for exact recovery",
    ),
    FeynmanProblem(
        "eml.expexp", "exp-of-exp", "exp(exp(x))",
        lambda x: np.exp(np.exp(x)), x_range=(-1.0, 0.5),
        notes="depth-2 nested exp; curriculum often beats fixed search",
    ),
    FeynmanProblem(
        "eml.expexpexp", "exp-of-exp-of-exp", "exp(exp(exp(x)))",
        lambda x: np.exp(np.exp(np.exp(x))), x_range=(-1.0, 0.3),
        notes="depth-3 nested exp; curriculum-only territory",
    ),
    FeynmanProblem(
        "eml.exp-1", "exp-minus-1", "exp(x) - 1",
        lambda x: np.exp(x) - 1.0, x_range=(0.0, 2.0),
        notes="depth-2: eml(x, e) — also the dominant local minimum",
    ),

    # ── Linear targets (Feynman: friction, potential, photon) ───
    # These exist primarily to exercise the depth-1 identity-gap finding
    # from issue #11: EML cannot emit `y = x` until depth ≥ 4 because
    # every output is wrapped in an outer eml(·,·). When these "succeed"
    # it is at depth 4 and only because the simplifier collapses a
    # multi-eml chain back to `x`.
    FeynmanProblem(
        "I.34.8", "doppler-low", "x", lambda x: x.copy(),
        x_range=(0.5, 3.0),
        notes="identity baseline; expected to need depth ≥ 4 (issue #11)",
    ),
    FeynmanProblem(
        "I.12.1", "friction", "0.3 * x", lambda x: 0.3 * x,
        x_range=(0.5, 5.0),
        notes="μ*F_n with μ=0.3 fixed; relies on normalizer to absorb scale",
    ),
    FeynmanProblem(
        "I.14.3", "potential-mgz", "9.81 * x", lambda x: 9.81 * x,
        x_range=(0.5, 10.0),
        notes="gravitational PE, m=1 kg",
    ),
]


# ─── Runner ────────────────────────────────────────────────────


def _run_one(prob: FeynmanProblem, *, max_depth: int, n_tries: int,
             method: str, normalize: str, n_workers: int,
             threshold: float) -> dict:
    x = np.linspace(prob.x_range[0], prob.x_range[1], prob.n_samples)
    y = prob.fn(x)

    norm = Normalizer.fit(x, y, mode=normalize)
    x_n = norm.transform_x(x)
    y_n = norm.transform_y(y)

    t0 = time.time()
    if method == "curriculum":
        result = discover_curriculum(
            x_n, y_n, max_depth=max_depth, n_tries=n_tries,
            verbose=False, success_threshold=threshold,
        )
    else:
        result = discover(
            x_n, y_n, max_depth=max_depth, n_tries=n_tries,
            verbose=False, success_threshold=threshold,
            n_workers=n_workers,
        )
    elapsed = time.time() - t0

    if result is None:
        return {"prob": prob, "ok": False, "expr": None, "elapsed": elapsed,
                "rmse": float("inf"), "rmse_norm": float("inf"), "depth": None}

    # Recompute RMSE in *original* coordinate space — `result["snap_rmse"]`
    # is reported in normalized space, which is misleading: an affine
    # transform of an elementary target (minmax/standard) is generally
    # not itself elementary in the EML vocabulary, so a perfect-RMSE fit
    # in normalized space corresponds to a *bad* fit in original space.
    # Issue #11 documents the implications.
    snapped = result.get("snapped_tree")
    rmse_orig = float("inf")
    if snapped is not None:
        with torch.no_grad():
            pred_n, _, _ = snapped(torch.tensor(x_n, dtype=REAL), tau=0.01)
            pred_n_np = pred_n.real.detach().numpy()
            pred_orig = norm.inverse_y(pred_n_np)
            rmse_orig = float(np.sqrt(np.mean((pred_orig - y) ** 2)))

    return {
        "prob": prob,
        "ok": bool(result.get("exact", True)),
        "expr": result["expr"],
        "rmse": rmse_orig,
        "rmse_norm": result["snap_rmse"],
        "depth": result["depth"],
        "elapsed": elapsed,
        "normalizer": norm,
    }


def _fmt_row(r: dict, threshold: float) -> str:
    p = r["prob"]
    # Recovery is "ok" only when the original-space RMSE is below threshold.
    # The discover()-internal "exact" flag is computed against the
    # normalized-space MSE which can be misleading (issue #11).
    is_ok = math.isfinite(r["rmse"]) and r["rmse"] < threshold
    ok = "✓" if is_ok else " "
    rmse = "—" if not math.isfinite(r["rmse"]) else f"{r['rmse']:.2e}"
    depth = "—" if r["depth"] is None else str(r["depth"])
    expr = (r["expr"] or "<no formula>")[:42]
    return (f"  {ok} {p.feynman_id:9s} {p.name:18s} d={depth:>2s} "
            f"rmse={rmse:>9s} t={r['elapsed']:5.1f}s  → {expr}")


def run(quick: bool = True, **kwargs) -> list:
    """Run the Feynman benchmark and return a list of result dicts."""
    problems = PROBLEMS[:8] if quick else PROBLEMS
    threshold = kwargs.get("threshold", 1e-6)

    print(f"═══ Feynman benchmark — {len(problems)} problems "
          f"({kwargs.get('method', 'discover')}, "
          f"normalize={kwargs.get('normalize', 'minmax')}, "
          f"workers={kwargs.get('n_workers', 1)}) ═══\n")
    print("  ID         name               depth rmse        time   formula")
    print("  ─────────────────────────────────────────────────────────────────────")

    results = []
    t_total = time.time()
    for prob in problems:
        r = _run_one(prob, **kwargs)
        results.append(r)
        # RMSE in the printed row is now in *original* coordinate space
        # (issue #11). Recovery counts only succeed when that original-space
        # RMSE is below the user-supplied threshold.
        print(_fmt_row(r, threshold=threshold))

    n_ok = sum(1 for r in results
               if math.isfinite(r["rmse"]) and r["rmse"] < threshold)
    print()
    print(f"  exact recovery:  {n_ok}/{len(results)}  "
          f"({100*n_ok/len(results):.0f}%)")
    print(f"  total time:      {time.time() - t_total:.1f}s")
    return results


def main():
    p = argparse.ArgumentParser(description="Feynman benchmark for eml-sr")
    p.add_argument("--all", action="store_true", help="Run all problems (else first 8)")
    p.add_argument("--max-depth", type=int, default=4)
    # Issue #11 recommended ≥16 seeds for the multi-depth ladder in CI runs.
    # Depth-3 targets (e.g. ln(x)) need ≥8 reliably; deeper or noisier
    # targets benefit from the extra headroom.
    p.add_argument("--tries", type=int, default=16)
    # Defaults retuned in issue #11. Key finding: ANY normalization (minmax
    # or standard) destroys symbolic recoverability for nonlinear targets,
    # because the affine transform of an elementary function is generally
    # not itself elementary in the EML vocabulary. The trimmed catalog uses
    # modest x ranges so `none` does not overflow the EML clamp. Use
    # `--normalize minmax` only when the y range spans many decades.
    p.add_argument("--method", choices=["discover", "curriculum"], default="curriculum")
    p.add_argument("--normalize", choices=["minmax", "standard", "none"], default="none")
    p.add_argument("--workers", type=int, default=1)
    # Threshold is on *original-space* RMSE, not normalized-space MSE.
    # 1e-6 is the threshold for "exactly recovered" given float64 precision
    # over reasonable y-magnitudes; anything higher is an approximation.
    p.add_argument("--threshold", type=float, default=1e-6)
    args = p.parse_args()

    run(
        quick=not args.all,
        max_depth=args.max_depth,
        n_tries=args.tries,
        method=args.method,
        normalize=args.normalize,
        n_workers=args.workers,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
