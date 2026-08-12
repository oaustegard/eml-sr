"""Test suite for eml_dca (issue #58 DC/DCA training for EMLTree1DLinear).

Covers:
    1. `_gate_level_slices` bookkeeping
    2. Exactness of the DC pair reconstruction (G - H == true objective)
    3. Convexity of G and H under in-block perturbations within the
       trust region
    4. The depth-1 "affine leaves are convex" anchor claim from the
       issue, plus a hand-computed value check
    5. The squared-loss monotone-split correction (why the issue's
       sketch split was replaced)
    6. `dca_train` end-to-end behavior (monotone descent, depth-0 edge
       case, result shape, determinism)

All tests are float64, seeded, and budgeted to keep the whole file
well under a minute.
"""

from __future__ import annotations

import math

import numpy as np
import itertools

import pytest
import torch

from eml_dca import (
    DCPair,
    _gate_level_slices,
    _true_mse,
    dc_forward,
    dc_objective,
    dca_train,
    sq_loss_pair,
)
from eml_sr import REAL
from eml_sr_linear import EMLTree1DLinear

# ───────────────────────── helpers ──────────────────────────

def _perturb_block(leaf_base, gate_base, block, depth, delta, rng):
    """Return (leaf, gate) equal to the base tensors everywhere except
    the rows belonging to `block`, which are perturbed by a uniform
    draw in [-delta, delta] (per-coordinate, inf-norm bounded by delta
    by construction)."""
    leaf = leaf_base.clone()
    gate = gate_base.clone()
    if block == "leaves":
        d = torch.tensor(rng.uniform(-delta, delta, size=tuple(leaf.shape)),
                          dtype=REAL)
        leaf = leaf + d
    else:
        k = int(block.split(":", 1)[1])
        start, count = _gate_level_slices(depth)[k]
        shape = (count,) + tuple(gate.shape[1:])
        d = torch.tensor(rng.uniform(-delta, delta, size=shape), dtype=REAL)
        gate[start:start + count] = gate[start:start + count] + d
    return leaf, gate


# ═══════════════════ 1. _gate_level_slices ═══════════════════

class TestGateLevelSlices:
    """`_gate_level_slices(d)` walks gate_logits rows bottom-up, one
    slice per tree level."""

    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_partitions_range(self, d):
        slices = _gate_level_slices(d)
        n_internal = 2 ** d - 1
        covered = []
        for start, count in slices:
            covered.extend(range(start, start + count))
        assert sorted(covered) == list(range(n_internal))

    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_counts_halve_bottom_up(self, d):
        slices = _gate_level_slices(d)
        counts = [c for _, c in slices]
        for prev, nxt in itertools.pairwise(counts):
            assert nxt == prev // 2

    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_last_count_is_one(self, d):
        slices = _gate_level_slices(d)
        assert slices[-1][1] == 1


# ═══════════════════ 2. Exactness of G - H ═══════════════════

class TestExactness:
    """G - H, evaluated at a perturbed point via the DC-pair machinery
    (base != active, delta > 0), must equal a from-scratch evaluation
    of the same objective at that point (base == active, delta == 0):
    the Lipschitz correction terms cancel in the difference by
    construction, for every block and every in-block perturbation."""

    depth = 2
    n_vars = 2
    delta = 0.5
    n_trials = 20

    def _setup(self, seed):
        torch.manual_seed(seed)
        # Leaf init like EMLTree1DLinear: small randn, alpha biased +1.
        # Gate init: randn*0.1 with gamma biased +1. Building the tree
        # directly reproduces that exact recipe.
        tree = EMLTree1DLinear(self.depth, n_vars=self.n_vars)
        leaf_base = tree.leaf_logits.detach().clone()
        gate_base = tree.gate_logits.detach().clone()
        rng = np.random.default_rng(seed + 1000)
        x = torch.tensor(rng.uniform(0.5, 2.5, size=(32, self.n_vars)),
                          dtype=REAL)
        y = torch.tensor(rng.uniform(-2.0, 2.0, size=32), dtype=REAL)
        return leaf_base, gate_base, x, y, rng

    @pytest.mark.parametrize("block", ["leaves", "gates:0", "gates:1"])
    def test_g_minus_h_matches_scratch(self, block):
        leaf_base, gate_base, x, y, rng = self._setup(seed=0)
        for _ in range(self.n_trials):
            leaf_p, gate_p = _perturb_block(
                leaf_base, gate_base, block, self.depth, self.delta, rng)

            G, H = dc_objective(leaf_p, gate_p, leaf_base, gate_base,
                                 x, y, block, self.delta, self.depth)
            G0, H0 = dc_objective(leaf_p, gate_p, leaf_p, gate_p,
                                   x, y, "leaves", 0.0, self.depth)

            lhs = (G - H).item()
            rhs = (G0 - H0).item()
            scale = max(abs(float(G.item())), abs(float(H.item())), 1.0)
            assert abs(lhs - rhs) <= 1e-9 * scale, (
                f"block={block}: G-H={lhs!r} vs scratch={rhs!r}")


# ═══════════════════ 3. Convexity of G and H ═══════════════════

class TestConvexity:
    """G and H are each convex in the active block over the trust
    region: for any two in-block perturbations, the objective at their
    midpoint is <= the average of the objectives at the two points."""

    depth = 2
    n_vars = 2
    delta = 0.5
    n_trials = 40

    def _setup(self, seed):
        torch.manual_seed(seed)
        tree = EMLTree1DLinear(self.depth, n_vars=self.n_vars)
        leaf_base = tree.leaf_logits.detach().clone()
        gate_base = tree.gate_logits.detach().clone()
        rng = np.random.default_rng(seed + 2000)
        x = torch.tensor(rng.uniform(0.5, 2.5, size=(32, self.n_vars)),
                          dtype=REAL)
        y = torch.tensor(rng.uniform(-2.0, 2.0, size=32), dtype=REAL)
        return leaf_base, gate_base, x, y, rng

    @pytest.mark.parametrize("block", ["leaves", "gates:0", "gates:1"])
    def test_midpoint_convexity(self, block):
        leaf_base, gate_base, x, y, rng = self._setup(seed=1)
        for _ in range(self.n_trials):
            leaf1, gate1 = _perturb_block(
                leaf_base, gate_base, block, self.depth, self.delta, rng)
            leaf2, gate2 = _perturb_block(
                leaf_base, gate_base, block, self.depth, self.delta, rng)
            # Midpoint of two points within an inf-ball of radius delta
            # around the base stays within the same ball.
            leaf_mid = 0.5 * (leaf1 + leaf2)
            gate_mid = 0.5 * (gate1 + gate2)

            G1, H1 = dc_objective(leaf1, gate1, leaf_base, gate_base,
                                   x, y, block, self.delta, self.depth)
            G2, H2 = dc_objective(leaf2, gate2, leaf_base, gate_base,
                                   x, y, block, self.delta, self.depth)
            Gm, Hm = dc_objective(leaf_mid, gate_mid, leaf_base, gate_base,
                                   x, y, block, self.delta, self.depth)

            for name, mid, a, b in [("G", Gm, G1, G2), ("H", Hm, H1, H2)]:
                mid_v = float(mid.item())
                avg_v = float(((a + b) / 2).item())
                scale = max(abs(mid_v), abs(avg_v), 1.0)
                assert mid_v <= avg_v + 1e-9 * scale, (
                    f"block={block} {name}: midpoint {mid_v!r} > "
                    f"average {avg_v!r}")


# ═══════════════════ 4. Depth-1 convexity anchor ═══════════════════

class TestDepth1Anchor:
    """The issue-#58 depth-1 claim: an eml node over affine leaves is
    convex in the leaf params (q_zero==True throughout), independent of
    delta. Also checks the forward value against a hand computation."""

    def test_root_is_exactly_convex_and_matches_handcomputed(self):
        depth = 1
        n_vars = 1

        # Affine leaves: both slots = 1 + 0.5*x, safely positive on
        # x in [0.5, 2.5] (both eml(.) inputs stay well above 0).
        leaf_coeffs = torch.tensor([[1.0, 0.5], [1.0, 0.5]], dtype=REAL)

        # Pass-through gates: alpha = beta = 0, gamma = 1, so the gate
        # forwards its child unchanged.
        gate_coeffs = torch.zeros(1, 2, n_vars + 2, dtype=REAL)
        gate_coeffs[..., -1] = 1.0

        rng = np.random.default_rng(7)
        x = torch.tensor(rng.uniform(0.5, 2.5, size=(16, n_vars)),
                          dtype=REAL)

        root, barrier = dc_forward(leaf_coeffs, gate_coeffs,
                                    leaf_coeffs, gate_coeffs,
                                    x, "leaves", 0.3, depth)

        assert root.q_zero is True

        u = 1.0 + 0.5 * x[:, 0]
        v = 1.0 + 0.5 * x[:, 0]
        expected = torch.exp(u) - torch.log(v)
        torch.testing.assert_close(root.value, expected,
                                    atol=1e-10, rtol=1e-10)

        # Sanity: barrier pair is well-formed (no NaN/Inf) even though
        # it isn't exercised much by this pass-through construction.
        assert torch.isfinite(barrier.value).all()


# ═══════════════════ 5. sq_loss_pair correction ═══════════════════

class TestSqLossCorrection:
    """Pins the correction described in the module docstring. The
    issue's sketch was g = f**2 + 2*max(0,-y)*f, h = 2*max(0,y)*f; at
    y=0 that g reduces to plain f**2, which is NOT convex whenever f
    dips negative (f**2 has negative curvature there). The replacement
    monotone split r**2 = max(r,0)**2 + min(r,0)**2 stays convex.

    Verified numerically at three collinear points t = -1, 0, 1 through
    f(t) = e^t - 5 (f is convex in t; f(t) < 0 throughout this range,
    so f**2 = -f * -f folds the wrong way and midpoint convexity
    fails)."""

    @staticmethod
    def _f(t: float) -> float:
        return math.exp(t) - 5.0

    def test_raw_square_violates_convexity(self):
        f_neg, f_0, f_pos = self._f(-1.0), self._f(0.0), self._f(1.0)
        lhs = f_0 ** 2
        rhs = (f_neg ** 2 + f_pos ** 2) / 2.0
        # Convexity would require lhs <= rhs; it doesn't hold here.
        # (2*max(0,-y)*f == 0 at y=0, so this is also exactly the
        # issue sketch's g, demonstrating its failure mode directly.)
        assert lhs > rhs

    def test_sq_loss_pair_halves_are_convex(self):
        ts = [-1.0, 0.0, 1.0]
        vals = torch.tensor([self._f(t) for t in ts], dtype=REAL)
        # f is monotone increasing on [-1, 1] (f' = e^t > 0), so the
        # certified interval is exactly [f(-1), f(1)].
        lo = torch.full((3,), self._f(-1.0), dtype=REAL)
        hi = torch.full((3,), self._f(1.0), dtype=REAL)

        # r is a valid DC pair over theta=t: q == 0 identically
        # (trivially convex), and p = f(t) is itself convex in t
        # (f''(t) = e^t > 0) -- exactly the hypothesis rule 4/5 need.
        # affine=False forces sq_loss_pair's non-exact (Lipschitz
        # corrected) branch for the decreasing half.
        r = DCPair(vals, torch.zeros(3, dtype=REAL), lo, hi, True, False)
        out = sq_loss_pair(r)

        # p - q reconstructs r**2 exactly regardless of the split.
        torch.testing.assert_close(out.value, vals ** 2,
                                    atol=1e-10, rtol=1e-10)

        g_neg, g_0, g_pos = out.p.tolist()
        h_neg, h_0, h_pos = out.q.tolist()
        assert g_0 <= (g_neg + g_pos) / 2.0 + 1e-9
        assert h_0 <= (h_neg + h_pos) / 2.0 + 1e-9


# ═══════════════════ 6. dca_train end-to-end ═══════════════════

class TestDcaTrain:

    def test_history_monotone_and_improves(self):
        rng = np.random.default_rng(0)
        x = torch.tensor(rng.uniform(0.5, 2.5, 64), dtype=REAL)
        y = 2.0 * x
        seed = 123

        # Initial mse before any training, replicated with the same
        # seed/init contract dca_train uses internally.
        torch.manual_seed(seed)
        tree0 = EMLTree1DLinear(1, n_vars=1)
        initial = _true_mse(tree0, x.unsqueeze(1), y)

        result = dca_train(x, y, depth=1, seed=seed,
                            outer_iters=12, inner_iters=15)
        hist = result["history"]

        assert len(hist) >= 1
        for a, b in itertools.pairwise(hist):
            assert b <= a + 1e-9, "history is not monotone non-increasing"
        assert hist[-1] < initial
        assert result["best_mse"] < initial
        assert result["outer_iters_used"] >= 1

    def test_depth_zero_runs_without_crashing(self):
        rng = np.random.default_rng(1)
        x = torch.tensor(rng.uniform(0.5, 2.5, 32), dtype=REAL)
        y = torch.tensor(rng.uniform(-1.0, 1.0, 32), dtype=REAL)
        result = dca_train(x, y, depth=0, seed=5,
                            outer_iters=4, inner_iters=5)
        assert math.isfinite(result["best_mse"])
        assert result["tree"].depth == 0
        assert result["tree"].n_internal == 0

    def test_returns_documented_keys(self):
        rng = np.random.default_rng(2)
        x = torch.tensor(rng.uniform(0.5, 2.5, 20), dtype=REAL)
        y = torch.tensor(rng.uniform(-1.0, 1.0, 20), dtype=REAL)
        result = dca_train(x, y, depth=1, seed=3,
                            outer_iters=3, inner_iters=4)
        for key in ("tree", "best_mse", "outer_iters_used", "history"):
            assert key in result

    def test_determinism_same_seed(self):
        rng = np.random.default_rng(3)
        x = torch.tensor(rng.uniform(0.5, 2.5, 24), dtype=REAL)
        y = torch.tensor(rng.uniform(-1.0, 1.0, 24), dtype=REAL)

        r1 = dca_train(x.clone(), y.clone(), depth=1, seed=9,
                        outer_iters=5, inner_iters=6)
        r2 = dca_train(x.clone(), y.clone(), depth=1, seed=9,
                        outer_iters=5, inner_iters=6)

        assert r1["best_mse"] == r2["best_mse"]
        assert r1["history"] == r2["history"]
