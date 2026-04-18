"""Multivariate validation for issue #20 (Direction E [6/6]).

Runs ``discover_hybrid(X, y)`` on three multivariate targets spanning
the expected difficulty spectrum:

    1. ``eml(x1, x2) = exp(x1) - ln(x2)`` — the depth-1 bivariate atom;
       must recover at depth 1 with machine precision.
    2. ``exp(x1) * exp(x2) = exp(x1 + x2)`` — needs an addition inside
       an exp; depth 2+ territory. Option A cannot represent x1+x2
       natively (no addition atom), so this probes the hybrid's
       Option-B fallback path.
    3. ``x1 / x2^2`` — a genuinely hard bivariate Feynman-style target
       (Coulomb-like). Establishes the depth/search frontier for
       n_vars=2: where the search budget starts to bite.

Each run reports: method dispatch (option_a / warm_start_a / option_b),
discovered depth, original-space RMSE, and wall clock.

Univariate baselines for calibration:

    discover_hybrid on exp(x) at depth 1 → option_a, <1e-10, ~a few seconds.
    discover_hybrid on ln(x) at depth 3  → option_a, <1e-10, ~tens of s.

Usage::

    python -m benchmarks.validate_multivariate              # default budget
    python -m benchmarks.validate_multivariate --tries-a 8  # harder budget
"""

from __future__ import annotations

import argparse
import math
import sys
import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eml_sr_hybrid import discover_hybrid  # noqa: E402


@dataclass
class ValidationTarget:
    name: str
    formula: str
    fn: Callable[[np.ndarray], np.ndarray]
    x_ranges: list      # list of (lo, hi) per column
    expected_depth: int | None    # None means "unknown / probing frontier"
    must_recover: bool            # if True, recovery failure is a test failure
    notes: str = ""


TARGETS: list[ValidationTarget] = [
    ValidationTarget(
        name="eml(x1, x2)",
        formula="exp(x1) - ln(x2)",
        fn=lambda X: np.exp(X[:, 0]) - np.log(X[:, 1]),
        x_ranges=[(0.5, 2.5), (0.5, 3.0)],
        expected_depth=1,
        must_recover=True,
        notes="depth-1 bivariate atom; reference for Option A recovery",
    ),
    ValidationTarget(
        name="exp(x1) * exp(x2)",
        formula="exp(x1 + x2)",
        fn=lambda X: np.exp(X[:, 0]) * np.exp(X[:, 1]),
        x_ranges=[(0.0, 1.0), (0.0, 1.0)],
        expected_depth=None,
        must_recover=False,
        notes="needs addition inside exp; probes Option-B fallback",
    ),
    ValidationTarget(
        name="x1 / x2^2",
        formula="x1 / x2^2",
        fn=lambda X: X[:, 0] / (X[:, 1] ** 2),
        x_ranges=[(0.5, 3.0), (0.5, 3.0)],
        expected_depth=None,
        must_recover=False,
        notes="Coulomb-like; establishes multivariate search frontier",
    ),
]


def _sample_X(target: ValidationTarget, n_samples: int = 100) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(target.name)) % (2**32))
    cols = [rng.uniform(lo, hi, n_samples) for (lo, hi) in target.x_ranges]
    return np.stack(cols, axis=1)


def _run_target(target: ValidationTarget, *, max_depth: int,
                n_tries_a: int, n_tries_b: int) -> dict:
    X = _sample_X(target)
    y = target.fn(X)

    t0 = time.time()
    result = discover_hybrid(
        X, y,
        max_depth=max_depth,
        n_tries_a=n_tries_a,
        n_tries_b=n_tries_b,
        verbose=False,
    )
    elapsed = time.time() - t0

    if result is None:
        return {
            "target": target, "ok": False, "method": None,
            "depth": None, "expr": None, "rmse": float("inf"),
            "elapsed": elapsed,
        }

    # RMSE in original space — discover_hybrid reports snap_rmse which
    # for hybrid runs is already in original space when no Normalizer
    # is applied (the caller is responsible for normalization).
    rmse = result["snap_rmse"]
    return {
        "target": target,
        "ok": rmse < 1e-6,
        "method": result["method"],
        "depth": result["depth"],
        "expr": result["expr"],
        "rmse": rmse,
        "elapsed": elapsed,
    }


def _fmt(r: dict) -> str:
    t = r["target"]
    ok = "✓" if r["ok"] else "✗" if t.must_recover else "·"
    rmse = "—" if not math.isfinite(r["rmse"]) else f"{r['rmse']:.2e}"
    depth = "—" if r["depth"] is None else str(r["depth"])
    method = r["method"] or "—"
    expr = (r["expr"] or "<no formula>")[:48]
    return (f"  {ok} {t.name:22s} d={depth:>2s} rmse={rmse:>9s} "
            f"t={r['elapsed']:6.1f}s  [{method:13s}]  → {expr}")


def main():
    p = argparse.ArgumentParser(
        description="Multivariate validation for discover_hybrid (issue #20)")
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--tries-a", type=int, default=4)
    p.add_argument("--tries-b", type=int, default=2)
    args = p.parse_args()

    print("═══ Multivariate validation — discover_hybrid ═══")
    print(f"  max_depth={args.max_depth}  "
          f"n_tries_a={args.tries_a}  n_tries_b={args.tries_b}\n")
    print("  target                 depth rmse        time     method         → expr")
    print("  " + "─" * 88)

    t_total = time.time()
    results = []
    n_must = 0
    n_must_ok = 0
    for tgt in TARGETS:
        r = _run_target(
            tgt, max_depth=args.max_depth,
            n_tries_a=args.tries_a, n_tries_b=args.tries_b,
        )
        results.append(r)
        print(_fmt(r))
        if tgt.must_recover:
            n_must += 1
            if r["ok"]:
                n_must_ok += 1

    print()
    print(f"  required recoveries: {n_must_ok}/{n_must}")
    print(f"  total wall clock:    {time.time() - t_total:.1f}s")

    # Non-zero exit if any must-recover target failed.
    if n_must_ok < n_must:
        sys.exit(1)


if __name__ == "__main__":
    main()
