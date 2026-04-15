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

Usage::

    python -m benchmarks.feynman                  # quick: 8 problems
    python -m benchmarks.feynman --all            # all problems
    python -m benchmarks.feynman --workers 8      # parallel seeds
    python -m benchmarks.feynman --method curriculum
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

# Allow running as a script from the repo root or as `python -m benchmarks.feynman`.
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eml_sr import discover, discover_curriculum, Normalizer  # noqa: E402


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
    # ── Genuinely univariate ────────────────────────────────────
    FeynmanProblem(
        "I.6.2", "gaussian", "exp(-x^2 / 2) / sqrt(2*pi)",
        _gaussian, x_range=(-2.0, 2.0),
        notes="standard normal pdf, depth ≥ 4 (squaring + exp + scale)",
    ),
    FeynmanProblem(
        "I.6.20a", "gaussian-sigma1", "exp(-x^2/2)",
        lambda x: np.exp(-x ** 2 / 2), x_range=(-2.0, 2.0),
    ),
    FeynmanProblem(
        "I.27.6", "lens-thin", "1/x", lambda x: 1.0 / x,
        x_range=(0.5, 3.0),
    ),
    FeynmanProblem(
        "I.34.8", "doppler-low", "x", lambda x: x.copy(), x_range=(0.5, 3.0),
        notes="trivial identity baseline",
    ),
    FeynmanProblem(
        "I.34.27", "photon-momentum", "1.0545718e-34 * x",
        lambda x: 1.0545718e-34 * x, x_range=(1e14, 1e15),
        notes="extreme-scale linear; tests normalization",
    ),

    # ── Projections of common multivariate Feynman eqs ──────────
    FeynmanProblem(
        "I.12.1", "friction", "0.3 * x", lambda x: 0.3 * x,
        x_range=(0.0, 5.0),
        notes="μ*F_n with μ=0.3 fixed",
    ),
    FeynmanProblem(
        "I.12.4", "coulomb-r", "1 / x^2", lambda x: 1.0 / (x ** 2),
        x_range=(0.5, 3.0),
        notes="Coulomb law in r with q1*q2/(4πε0)=1",
    ),
    FeynmanProblem(
        "I.14.3", "potential-mgz", "9.81 * x", lambda x: 9.81 * x,
        x_range=(0.0, 10.0),
        notes="gravitational PE, m=1 kg",
    ),
    FeynmanProblem(
        "I.16.6", "rel-velocity", "2*x / (1 + x*x)",
        lambda x: 2 * x / (1 + x * x),
        x_range=(-0.9, 0.9),
        notes="velocity addition (v1=v2=x in c=1 units)",
    ),
    FeynmanProblem(
        "I.25.13", "voltage-q", "x / 1e-6", lambda x: x / 1e-6,
        x_range=(1e-9, 1e-7),
        notes="V = q/C, C=1μF, extreme-scale",
    ),
    FeynmanProblem(
        "I.29.4", "wavevector", "x / 3e8", lambda x: x / 3e8,
        x_range=(1e6, 1e9),
        notes="ω/c → wavenumber",
    ),
    FeynmanProblem(
        "I.34.10", "doppler-freq", "1 / (1 - x)",
        lambda x: 1.0 / (1.0 - x),
        x_range=(-0.5, 0.5),
        notes="non-relativistic Doppler shift",
    ),
    FeynmanProblem(
        "I.39.10", "kinetic-T", "1.5 * 1.380649e-23 * x",
        lambda x: 1.5 * 1.380649e-23 * x,
        x_range=(100.0, 600.0),
        notes="thermal energy at temperature T",
    ),
    FeynmanProblem(
        "I.41.16", "planck-rj", "2 * x", lambda x: 2 * x,
        x_range=(1e10, 1e14),
        notes="Rayleigh-Jeans low-freq limit (linear in ν)",
    ),

    # ── Stress tests (deeper formulas) ──────────────────────────
    FeynmanProblem(
        "stress.1", "exp-of-exp", "exp(exp(x))",
        lambda x: np.exp(np.exp(x)), x_range=(-1.0, 0.5),
        notes="depth-2 nested exp, curriculum often beats fixed search",
    ),
    FeynmanProblem(
        "stress.2", "exp-minus-ln", "exp(x) - ln(x)",
        lambda x: np.exp(x) - np.log(x), x_range=(0.5, 3.0),
        notes="raw eml shape — should be depth 1",
    ),
    FeynmanProblem(
        "stress.3", "ln-x", "ln(x)", lambda x: np.log(x),
        x_range=(0.5, 5.0),
        notes="known to require depth 3 for exact recovery",
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
                "rmse": float("inf"), "depth": None}

    return {
        "prob": prob,
        "ok": bool(result.get("exact", True)),
        "expr": result["expr"],
        "rmse": result["snap_rmse"],
        "depth": result["depth"],
        "elapsed": elapsed,
        "normalizer": norm,
    }


def _fmt_row(r: dict) -> str:
    p = r["prob"]
    ok = "✓" if r["ok"] else " "
    rmse = "—" if not math.isfinite(r["rmse"]) else f"{r['rmse']:.2e}"
    depth = "—" if r["depth"] is None else str(r["depth"])
    expr = (r["expr"] or "<no formula>")[:42]
    return (f"  {ok} {p.feynman_id:9s} {p.name:18s} d={depth:>2s} "
            f"rmse={rmse:>9s} t={r['elapsed']:5.1f}s  → {expr}")


def run(quick: bool = True, **kwargs) -> list:
    """Run the Feynman benchmark and return a list of result dicts."""
    problems = PROBLEMS[:8] if quick else PROBLEMS

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
        print(_fmt_row(r))

    n_ok = sum(1 for r in results if r["ok"])
    print()
    print(f"  exact recovery:  {n_ok}/{len(results)}  "
          f"({100*n_ok/len(results):.0f}%)")
    print(f"  total time:      {time.time() - t_total:.1f}s")
    return results


def main():
    p = argparse.ArgumentParser(description="Feynman benchmark for eml-sr")
    p.add_argument("--all", action="store_true", help="Run all problems (else first 8)")
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--tries", type=int, default=8)
    p.add_argument("--method", choices=["discover", "curriculum"], default="discover")
    p.add_argument("--normalize", choices=["minmax", "standard", "none"], default="minmax")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threshold", type=float, default=1e-10)
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
