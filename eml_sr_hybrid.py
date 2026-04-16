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

from eml_sr import DTYPE, REAL, EMLTree1D, _train_one, discover
from eml_sr_linear import (
    EMLTree1DLinear, discover_linear, iterative_snap, _train_one_linear,
)


def warm_start_a_from_b(
    b_tree: EMLTree1DLinear,
    bias: float = 4.0,
) -> EMLTree1D:
    """Create an Option A tree with logits biased from Option B's coefficients.

    For each leaf ``(α, β)`` in B, the A leaf logit for ``{1, x}`` is biased
    toward whichever has the larger absolute coefficient. For each gate side
    ``(α, β, γ)`` in B, the A gate logit for ``{1, x, child}`` is biased
    toward the dominant contributor.

    This gives Option A a warm start from B's structural exploration, so
    A's tau-anneal training can commit to a discrete structure faster and
    with fewer random seeds (issue #12).

    Args:
        b_tree: a trained EMLTree1DLinear (not modified)
        bias: strength of the logit bias (default 4.0; higher = more
            confident warm start, lower = more exploration room)

    Returns:
        A new EMLTree1D with biased logits, ready for training.
    """
    depth = b_tree.depth
    a_tree = EMLTree1D(depth, init_scale=0.0)

    with torch.no_grad():
        # ── Leaves: (n_leaves, 2) ──
        # B stores [α (constant), β (x)]. Bias A toward whichever is larger.
        b_leaf = b_tree.leaf_logits.detach()  # (n_leaves, 2)
        abs_leaf = b_leaf.abs()
        dominant = torch.argmax(abs_leaf, dim=1)  # 0 → constant, 1 → x
        new_leaf = torch.full_like(a_tree.leaf_logits, -bias)
        new_leaf[torch.arange(a_tree.n_leaves), dominant] = bias
        a_tree.leaf_logits.copy_(new_leaf)

        # ── Gates: (n_internal, 2, 3) ──
        # B stores [α (constant), β (x), γ (child)] per side.
        # Bias A's 3-way softmax logit toward the dominant contributor.
        b_gate = b_tree.gate_logits.detach()  # (n_internal, 2, 3)
        abs_gate = b_gate.abs()
        dominant_gate = torch.argmax(abs_gate, dim=-1)  # (n_internal, 2)
        new_gate = torch.full_like(a_tree.gate_logits, -bias)
        idx = torch.arange(a_tree.n_internal)
        for side in range(2):
            new_gate[idx, side, dominant_gate[:, side]] = bias
        a_tree.gate_logits.copy_(new_gate)

    return a_tree


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
              f"trying warm-start from Option B.")

    x_t = torch.tensor(x, dtype=REAL)
    y_t = torch.tensor(y, dtype=DTYPE)

    # ── Stage 1.5: Warm-start A from B (issue #12) ────────────
    # Train a quick Option B to discover structure, then bias
    # Option A's logits from B's dominant coefficients. This gives
    # A a better starting point without changing its architecture.
    if verbose:
        print("\n═══ Stage 1.5: Warm-start Option A from Option B ═══")

    best_warm = None
    for depth in range(1, max_depth_b + 1):
        # Quick B exploration: short budget, no snap penalty.
        for seed in range(n_tries_b):
            b_result = _train_one_linear(
                x_t, y_t, depth=depth, seed=seed,
                search_iters=1500, snap_iters=0, lam_disc_max=0.0)

            # Convert B's structure to A's logits and train A.
            a_init = warm_start_a_from_b(b_result["tree"])
            a_ws = _train_one(x_t, y_t, depth=depth, seed=seed,
                              init_tree=a_init, verbose=False)

            if verbose and a_ws["snap_rmse"] < 1e-3:
                print(f"  d={depth} s={seed}: "
                      f"rmse={a_ws['snap_rmse']:.2e} "
                      f"→ {a_ws['expr'][:60]}")

            if best_warm is None or a_ws["snap_mse"] < best_warm["snap_mse"]:
                best_warm = a_ws
                best_warm["depth"] = depth

            if a_ws["snap_mse"] < fallback_threshold ** 2:
                break
        if best_warm and best_warm["snap_mse"] < fallback_threshold ** 2:
            break

    ws_rmse = best_warm["snap_rmse"] if best_warm else float("inf")
    if best_warm is not None and ws_rmse < fallback_threshold:
        if verbose:
            print(f"\n  ✓ Warm-start A succeeded: {best_warm['expr']}")
            print(f"    depth={best_warm['depth']} rmse={ws_rmse:.2e}")
        return {
            "expr": best_warm["expr"],
            "depth": best_warm["depth"],
            "snap_rmse": ws_rmse,
            "snapped_tree": best_warm["snapped"],
            "method": "warm_start_a",
        }

    if verbose:
        print(f"\n  Warm-start A best: rmse={ws_rmse:.2e}")
        print(f"  Falling back to full Option B.")

    # Update a_rmse to include warm-start result for final comparison.
    if ws_rmse < a_rmse:
        a_rmse = ws_rmse
        result_a = {
            "expr": best_warm["expr"],
            "depth": best_warm["depth"],
            "snap_rmse": ws_rmse,
            "snapped_tree": best_warm["snapped"],
            "n_uncertain": best_warm.get("n_uncertain", 0),
        }

    # ── Stage 2: Option B (train + iterative snap) ─────────────
    if verbose:
        print("\n═══ Stage 2: Option B (linear coefficients "
              "+ iterative snap) ═══")

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

    # Pick the better result between A (or warm-start A) and B.
    if a_rmse < b_rmse and result_a is not None:
        if verbose:
            print(f"\n  Option A still wins on snap RMSE "
                  f"({a_rmse:.2e} < {b_rmse:.2e})")
        result_a["method"] = result_a.get("method", "option_a")
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
