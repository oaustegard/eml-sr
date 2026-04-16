"""discover_hybrid: staged Option A → Option B formula discovery.

Tries Option A (softmax/argmax, clean symbolic output) first. If A's
post-snap RMSE exceeds a threshold, falls back to Option B (linear
coefficients, more expressive, lossy symbolic snap).

This is Strategy 1 from the issue #11 / Option B investigation:

    (x, y)  →  Option A (fast, clean symbolic)
            →  Option B fallback (slower, numerical fit)
            →  return best result with a flag showing which stage won

The user sees one API (`discover_hybrid`) and gets:
  - Clean symbolic output when Option A succeeds
  - A numerical near-fit when Option A fails architecturally
  - A `method` field indicating which stage produced the answer

Usage::

    from eml_sr_hybrid import discover_hybrid

    result = discover_hybrid(x, y)
    print(result["expr"], result["method"])
    # → 'exp(x)'  'option_a'          (clean symbolic)
    # → 'eml(...)'  'option_b'        (numerical fit)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from eml_sr import DTYPE, REAL, discover
from eml_sr_linear import discover_linear, iterative_snap, _train_one_linear


def discover_hybrid(
    x: np.ndarray,
    y: np.ndarray,
    max_depth: int = 4,
    n_tries_a: int = 8,
    n_tries_b: int = 4,
    max_depth_b: int = 3,
    fallback_threshold: float = 1e-6,
    verbose: bool = True,
) -> Optional[dict]:
    """Discover a formula relating x → y, using Option A first with
    Option B as a fallback.

    Args:
        x: input values (1D numpy array)
        y: output values (1D numpy array)
        max_depth: maximum tree depth for Option A (default 4)
        n_tries_a: random seeds per depth for Option A (default 8)
        n_tries_b: random seeds per depth for Option B (default 4)
        max_depth_b: maximum tree depth for Option B (default 3)
        fallback_threshold: RMSE below which Option A is accepted
            without falling back to Option B (default 1e-6)
        verbose: print progress

    Returns:
        dict with keys:
            expr: symbolic expression string
            depth: tree depth used
            snap_rmse: RMSE of the snapped tree
            snapped_tree: callable nn.Module
            method: 'option_a' or 'option_b'
            fit_rmse: (Option B only) pre-snap RMSE showing the
                      architecture's true fitting power
    """
    # ── Stage 1: Option A ──────────────────────────────────────
    if verbose:
        print("═══ Stage 1: Option A (softmax / clean symbolic) ═══")

    result_a = discover(
        x, y,
        max_depth=max_depth,
        n_tries=n_tries_a,
        verbose=verbose,
    )

    if result_a is not None and result_a["snap_rmse"] < fallback_threshold:
        if verbose:
            print(f"\n  ✓ Option A succeeded: {result_a['expr']}")
            print(f"    depth={result_a['depth']} "
                  f"rmse={result_a['snap_rmse']:.2e}")
        result_a["method"] = "option_a"
        return result_a

    a_rmse = result_a["snap_rmse"] if result_a else float("inf")
    a_expr = result_a["expr"] if result_a else None
    if verbose:
        print(f"\n  Option A best: rmse={a_rmse:.2e} → {a_expr}")
        print(f"  Above threshold {fallback_threshold:.0e}, "
              f"falling back to Option B.")

    # ── Stage 2: Option B (train + iterative snap) ─────────────
    if verbose:
        print("\n═══ Stage 2: Option B (linear coefficients "
              "+ iterative snap) ═══")

    x_t = torch.tensor(x, dtype=REAL)
    y_t = torch.tensor(y, dtype=DTYPE)

    best_b = None
    for depth in range(1, max_depth_b + 1):
        for seed in range(n_tries_b):
            r = _train_one_linear(x_t, y_t, depth=depth, seed=seed,
                                  search_iters=3000, snap_iters=10,
                                  lam_disc_max=0.0)
            r["depth"] = depth
            if best_b is None or r["best_mse"] < best_b["best_mse"]:
                best_b = r
        if best_b and best_b["best_mse"] < 1e-5:
            break

    if best_b is None:
        if result_a is not None:
            result_a["method"] = "option_a"
            return result_a
        return None

    # Iterative snap: prune one coefficient at a time, retrain.
    if verbose:
        print(f"\n  B free fit: mse={best_b['best_mse']:.2e} "
              f"(depth {best_b['depth']})")
        print("  Running iterative snap...")

    snapped_tree = iterative_snap(
        best_b["tree"], x_t, y_t,
        retrain_iters=300, lr=0.005,
        verbose=verbose,
    )

    with torch.no_grad():
        pred, _, _ = snapped_tree(x_t)
        b_rmse = float(np.sqrt(np.mean(
            (pred.real.detach().numpy() - y) ** 2)))

    b_expr = snapped_tree.to_expr()

    if verbose:
        print(f"\n  B iterative snap: rmse={b_rmse:.2e}")
        print(f"  expr: {b_expr[:80]}")

    # Pick the better result between A and B.
    if a_rmse < b_rmse and result_a is not None:
        if verbose:
            print(f"\n  Option A still wins on snap RMSE "
                  f"({a_rmse:.2e} < {b_rmse:.2e})")
        result_a["method"] = "option_a"
        return result_a

    if verbose:
        print(f"\n  ✓ Option B wins: rmse={b_rmse:.2e}")

    return {
        "expr": b_expr,
        "depth": best_b["depth"],
        "snap_rmse": b_rmse,
        "snapped_tree": snapped_tree,
        "method": "option_b",
        "fit_rmse": best_b["best_mse"] ** 0.5,
    }
