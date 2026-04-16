"""eml_sr_linear: Option B (paper-faithful) tree for EML symbolic regression.

This is an experimental alternative to `EMLTree1D` in `eml_sr.py`. Both trees
have identical parameter shapes and the same outer EML algebra, but they
differ in how a gate combines its three input candidates `{1, x, child}`:

  * **Option A** (current default, `EMLTree1D`): the gate applies a softmax
    over three logits, producing a *convex combination* of `{1, x, child}`.
    The achievable input value is constrained to the simplex spanned by
    those three sources — every coefficient is non-negative and they sum
    to 1. This is computationally cheap and snaps cleanly via argmax, but
    it is strictly weaker than the paper's parameterization.

  * **Option B** (this module, `EMLTree1DLinear`): the gate uses an
    *unconstrained linear combination*

        input = α·1 + β·x + γ·child

    with `α, β, γ ∈ ℝ` learned via gradient descent. This matches Section
    4.3 equation (6) of Odrzywolek (2026). Three things change as a result:

    1. Constants outside `[0, max(1, x)]` (notably `e`, `π`, fractional
       slopes like `0.3`) live in the depth-1 vocabulary directly, since
       a leaf or gate side can learn an arbitrary real constant.
    2. **Subtraction of children is finally available**: setting `γ = -1`
       lets a parent invert its child's contribution. This unlocks
       depth-2 recoveries such as `eml(ln(x), 1) = x` via the construction
       `gate_l = 1 - eml(0, x) = 1 - (1 - ln(x)) = ln(x)`, which is
       *impossible* under Option A because softmax forbids negative
       coefficients.
    3. The loss landscape is smooth everywhere — there is no temperature
       schedule, no entropy penalty, and no simplex corners producing
       discontinuous gradients. This eliminates a large class of bad
       local minima (including the `(e − ln(0))` snap that Option A's
       curriculum mode exhibits on `exp(exp(exp(x)))`).

## Why we built it

After issue #6's Feynman benchmark and issue #11's diagnosis, we observed
that Option A failed on every nonlinear target outside its tightest
vocabulary — `ln(x)` snapped to `e`, `exp(x) - 1` snapped to `exp(x)`,
`exp(exp(exp(x)))` produced `(e − ln(0))`, and all linear targets
(`y = x`, `0.3·x`, `9.81·x`) bottomed out as constants. Issue #4
explicitly listed Option A as the "minimal" fix and Option B as the
"paper's approach", and we picked A in commit `9363cb9` because it was
sufficient for the test case in #4 (`eml(x, x) = exp(x) - ln(x)` at
depth 1). We never circled back to validate that Option A could handle
the rest of the paper's bootstrap chain. This module addresses that gap.

## Snap to symbolic expressions

The tree's expressivity gain comes at a cost: snapping real-valued
coefficients to a clean symbolic form is harder than argmax. The
prototype here uses a simple recipe:

    if |α| < SNAP_EPS:                    α := 0
    elif |α - round(α)| < SNAP_EPS:       α := round(α)   # nearest int
    elif |α - e| < SNAP_EPS:              α := e
    elif |α + e| < SNAP_EPS:              α := -e

Anything else stays as a learned real and is printed numerically. This
catches the integer/`e` cases that arise from EML identities; richer
constant recognition (`π`, `1/e`, `ln(2)`, etc.) is a follow-up.

The expression simplifier in `eml_sr.py` only handles the Option-A
vocabulary `{1, x, 0, e}`. We do not extend it here — instead we emit
expressions in a separate "linear-combination" form that the snap
function can pretty-print but the simplifier won't recursively reduce.
That's intentional scope control: the experiment's job is to answer
"can the architecture fit these targets at all?", not to deliver the
full symbolic-recovery pipeline. If the answer is yes, simplifier
extension is a follow-up issue.

## Empirical findings from the prototype

Runs via `benchmarks/option_ab_compare.py` and direct `_train_one_linear`
calls surfaced three separable phenomena:

### 1. Architectural expressivity: Option B wins where it should

On targets Option A cannot express at a given depth, Option B fits them
by orders of magnitude better even with *free* coefficients (before any
snap attempt):

    exp(x) - 1   (depth 1, 6 seeds)
      Option A: rmse 1.00   (snaps to `exp(x)`, off by 1 constant)
      Option B: best_mse 1.17e-06   (4 of 6 seeds converge)

    y = x        (identity target, which Option A cannot reach at d≤4)
      Option A: rmse 0.56  at d=3   (snaps to `e - ln(e - ln(x))`)
      Option B: best_mse 2.4e-07 at d=2   (5 of 6 seeds converge)
      This confirms the theoretical prediction that Option B unlocks
      `eml(ln(x), 1) = x` via the construction
          gate_l = 1 − eml(0, x)
      which requires γ = −1 on the child — impossible under softmax.

The 6-order-of-magnitude improvement on `exp(x) - 1` and ~6-order
improvement on `y = x` prove the architectural claim. Option B
strictly dominates Option A in representational power on the targets
where Option A fails.

### 2. Fitting regression on targets Option A handles well

On targets Option A already nails to machine precision (`exp(x)`, `e`,
`ln(x)`, `exp(x) - ln(x)`), Option B is *slower* and fits *worse*:

    target         Option A (rmse)   Option B (best_mse)
    exp(x)         1.99e-16           ~1e-06    (stops at near-fit)
    ln(x)          7.35e-17           ~1e-05    (stops at near-fit)
    e              0.00               ~1e-13    (only this one matches)

Root cause: **Option B's free parameterization admits infinite
equivalent representations of the same function**. With leaves
`α + βx` and gates `α + βx + γ·child`, there are more free parameters
than degrees of freedom in the target. The optimizer converges
polynomially to near-fits but never commits to the canonical
sparse representation. This is the dual of Option A's problem —
Option A has a discrete, one-hot snap at the cost of expressivity;
Option B has continuous expressivity at the cost of committing to
any specific discrete structure.

### 3. Symbolic snap is broken for Option B

Per-coefficient rounding to `{0, ±1, ±2, ±e}` destroys the fit in
every non-trivial case, because individual coefficients are not
*independent* carriers of meaning — their *combined* value is what
matters. A linear combo like `0.5·1 + 0.5·x + 0.7·child` may fit
perfectly but no single coefficient is close to a named constant.

A **discreteness penalty** (nearest-point loss toward the lattice
`{0, ±1, ±2, ±e}`, applied in the second training phase) failed to
fix this even at λ=100. The optimizer sits at saddle points on the
Voronoi boundaries between constants; no single coefficient wants
to move because any movement makes the penalty worse.

## Implications

Option B is the correct **architecture** (matches the paper, unlocks
the targets Option A fails on) but the naive snap recipe is the
wrong **algorithm**. Three paths forward, in increasing order of
ambition:

  * **Iterative magnitude pruning**: snap one coefficient at a time,
    smallest-magnitude first, retrain briefly after each snap to
    absorb the discontinuity. Classic neural-network pruning
    technique. Probably tractable but has N×M outer iterations.

  * **Brute-force discrete search**: at training's end, enumerate
    all 2^k snap configurations over the k coefficients that are
    "close enough" to multiple constants, evaluate MSE on each,
    pick the best. Tractable for depth ≤ 3 (k ≈ 10–30).

  * **Learned snap network**: train a separate head that, given
    the final tree's coefficients, predicts which configuration
    of discrete choices minimizes MSE. More complex but
    differentiable end-to-end.

The empirical conclusion: **Option B is worth pursuing**, but not
by simply flipping the default in `eml_sr.py`. It needs either
(a) its own training + snap pipeline that matches the paper
properly, or (b) a hybrid that uses Option A's argmax discipline
for targets it can handle and falls back to Option B only for
targets where A provably cannot express the answer.

## Limitations of this prototype

  * No multi-process worker support (single-threaded only).
  * No curriculum learning. Random init at fixed depths.
  * Snap-to-symbolic is broken for non-trivial cases; see §3 above.
  * Only tested on a small set of targets from issue #11. Broader
    regression coverage lives in `tests/test_eml_sr.py` and pertains
    to Option A.

## Reference

  Odrzywolek, A. (2026). "All elementary functions from a single operator."
  arXiv:2603.21852. Section 4.3, equation (6).
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from eml_sr import DTYPE, REAL, _CLAMP, eml_op  # noqa: F401  (re-exported)


# ─── Constants ─────────────────────────────────────────────────

# Snap tolerance for recognizing integer / `e` / 0 coefficients.
SNAP_EPS = 0.05

# Famous constants the snap step recognizes.
_NAMED_CONSTANTS: list[tuple[float, str]] = [
    (0.0, "0"),
    (1.0, "1"),
    (-1.0, "-1"),
    (2.0, "2"),
    (-2.0, "-2"),
    (math.e, "e"),
    (-math.e, "-e"),
]


# ─── Tree ──────────────────────────────────────────────────────

class EMLTree1DLinear(nn.Module):
    """Option B EML tree with unconstrained linear gate combinations.

    Same shape contract as `EMLTree1D` but with paper-faithful
    parameterization:

      Leaves: input = α + β·x   (2 reals per leaf)
      Gates:  input = α + β·x + γ·child  (3 reals per side, 2 sides per node)

    Identical parameter tensor *shapes* to `EMLTree1D`. They are *not*
    interchangeable — the values are coefficients here, not pre-softmax
    logits.
    """

    def __init__(self, depth: int, init_scale: float = 0.1):
        super().__init__()
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.n_internal = self.n_leaves - 1

        # Leaf coefficients: (n_leaves, 2) — [α (constant), β (x)]
        # Initialize close to the depth-1 atom `α=1, β=0` (constant 1) so
        # the tree starts as a stable constant-1 fit and the optimizer
        # discovers structure from there.
        leaf_init = torch.randn(self.n_leaves, 2, dtype=REAL) * init_scale
        leaf_init[:, 0] += 1.0  # bias α toward 1
        self.leaf_logits = nn.Parameter(leaf_init)

        # Gate coefficients: (n_internal, 2, 3) — [α (constant), β (x), γ (child)]
        # Bias γ toward 1.0 so the child contribution flows up by default;
        # this matches Option A's behavior of "child" being the most
        # informative source after random init.
        gate_init = torch.randn(self.n_internal, 2, 3, dtype=REAL) * init_scale
        gate_init[..., 2] += 1.0  # bias γ toward 1
        self.gate_logits = nn.Parameter(gate_init)

    def forward(self, x, tau: float = 1.0):  # tau accepted for API parity, unused
        x = x.to(DTYPE)
        batch = x.shape[0]
        ones = torch.ones(batch, dtype=DTYPE)

        # Leaf values: α + β·x (no softmax, no temperature)
        a = self.leaf_logits[:, 0].to(DTYPE)  # (n_leaves,)
        b = self.leaf_logits[:, 1].to(DTYPE)  # (n_leaves,)
        # broadcast: (batch, n_leaves)
        level = a.unsqueeze(0) * ones.unsqueeze(1) + b.unsqueeze(0) * x.unsqueeze(1)

        # Bottom-up: pair children, apply gate linear combos, compute eml
        node_idx = 0
        x_b = x.unsqueeze(1)  # (batch, 1)
        while level.shape[1] > 1:
            n_pairs = level.shape[1] // 2
            left = level[:, 0::2]   # (batch, n_pairs), complex
            right = level[:, 1::2]

            g = self.gate_logits[node_idx:node_idx + n_pairs]  # (n_pairs, 2, 3) real
            g_c = g.to(DTYPE)
            # α, β, γ for left and right sides
            a_l = g_c[:, 0, 0].unsqueeze(0)  # (1, n_pairs)
            b_l = g_c[:, 0, 1].unsqueeze(0)
            c_l = g_c[:, 0, 2].unsqueeze(0)
            a_r = g_c[:, 1, 0].unsqueeze(0)
            b_r = g_c[:, 1, 1].unsqueeze(0)
            c_r = g_c[:, 1, 2].unsqueeze(0)

            # Free linear combinations — no clamping of coefficients.
            left_in = a_l + b_l * x_b + c_l * left
            right_in = a_r + b_r * x_b + c_r * right

            level = eml_op(left_in, right_in)

            # Same numerical clamp as Option A — exp/ln overflow is
            # independent of the parameterization.
            level = torch.complex(
                torch.nan_to_num(level.real, nan=0.0,
                                 posinf=_CLAMP, neginf=-_CLAMP).clamp(-_CLAMP, _CLAMP),
                torch.nan_to_num(level.imag, nan=0.0,
                                 posinf=_CLAMP, neginf=-_CLAMP).clamp(-_CLAMP, _CLAMP),
            )
            node_idx += n_pairs

        return level.squeeze(1), None, None  # API-compatible 3-tuple

    def snap(self) -> "EMLTree1DLinear":
        """Snap each coefficient to the nearest recognized constant.

        Modifies a deep-copied tree in place; returns it. Coefficients
        that don't match any named constant are left as-is and printed
        numerically by `to_expr()`.
        """
        tree = copy.deepcopy(self)
        with torch.no_grad():
            for tensor in (tree.leaf_logits, tree.gate_logits):
                flat = tensor.view(-1)
                for i in range(flat.numel()):
                    v = float(flat[i].item())
                    snapped = _snap_scalar(v)
                    if snapped is not None:
                        flat[i] = snapped
        return tree

    def to_expr(self) -> str:
        """Pretty-print the tree as a linear-combination eml expression.

        This does *not* run through the recursive simplifier in
        `eml_sr.py`, since that simplifier only knows about Option A's
        atomic vocabulary. The printed form is therefore pre-simplification:
        readers can verify structure but the surface syntax is verbose.
        """
        leaf_a = self.leaf_logits[:, 0].tolist()
        leaf_b = self.leaf_logits[:, 1].tolist()
        gate = self.gate_logits.tolist()

        # Pretty-print each leaf as α + β·x.
        exprs = [_lin_expr(a, b, "x") for a, b in zip(leaf_a, leaf_b)]

        node_idx = 0
        while len(exprs) > 1:
            new_exprs = []
            for i in range(0, len(exprs), 2):
                left_child, right_child = exprs[i], exprs[i + 1]
                gl = gate[node_idx]  # [[αl, βl, γl], [αr, βr, γr]]
                left_in = _lin_expr3(gl[0][0], gl[0][1], gl[0][2],
                                     "x", left_child)
                right_in = _lin_expr3(gl[1][0], gl[1][1], gl[1][2],
                                      "x", right_child)
                new_exprs.append(f"eml({left_in}, {right_in})")
                node_idx += 1
            exprs = new_exprs
        return exprs[0]

    def n_params(self) -> int:
        return self.leaf_logits.numel() + self.gate_logits.numel()


# ─── Snap & pretty-print helpers ───────────────────────────────

def _snap_scalar(v: float) -> Optional[float]:
    """Return the snapped value if it matches a named constant within
    SNAP_EPS, else None to indicate "leave it as a learned real"."""
    # Named constants first (covers e, integers in [-2, 2]).
    for c, _ in _NAMED_CONSTANTS:
        if abs(v - c) < SNAP_EPS:
            return c
    # Nearest integer for larger magnitudes.
    if abs(v - round(v)) < SNAP_EPS:
        return float(round(v))
    return None


def _fmt_coef(v: float) -> str:
    """Pretty-print a coefficient: recognize named constants, else
    print as a 3-significant-figure float."""
    for c, name in _NAMED_CONSTANTS:
        if abs(v - c) < SNAP_EPS:
            return name
    if abs(v - round(v)) < SNAP_EPS:
        return str(int(round(v)))
    return f"{v:.3g}"


def _lin_expr(a: float, b: float, var: str) -> str:
    """Pretty-print α + β·var with sane elision of zeros and ones."""
    parts = []
    a_str = _fmt_coef(a) if abs(a) >= SNAP_EPS else None
    if a_str is not None and a_str != "0":
        parts.append(a_str)
    if abs(b) >= SNAP_EPS:
        if abs(b - 1.0) < SNAP_EPS:
            parts.append(var)
        elif abs(b + 1.0) < SNAP_EPS:
            parts.append(f"-{var}")
        else:
            parts.append(f"{_fmt_coef(b)}*{var}")
    if not parts:
        return "0"
    return " + ".join(parts).replace("+ -", "- ")


def _lin_expr3(a: float, b: float, c: float, var: str, child: str) -> str:
    """Pretty-print α + β·var + γ·child."""
    parts = []
    if abs(a) >= SNAP_EPS:
        parts.append(_fmt_coef(a))
    if abs(b) >= SNAP_EPS:
        if abs(b - 1.0) < SNAP_EPS:
            parts.append(var)
        elif abs(b + 1.0) < SNAP_EPS:
            parts.append(f"-{var}")
        else:
            parts.append(f"{_fmt_coef(b)}*{var}")
    if abs(c) >= SNAP_EPS:
        if abs(c - 1.0) < SNAP_EPS:
            parts.append(child)
        elif abs(c + 1.0) < SNAP_EPS:
            parts.append(f"-({child})")
        else:
            parts.append(f"{_fmt_coef(c)}*({child})")
    if not parts:
        return "0"
    return " + ".join(parts).replace("+ -", "- ")


# ─── Training ──────────────────────────────────────────────────

def _discreteness_penalty(tensor: torch.Tensor) -> torch.Tensor:
    """Penalty pushing each coefficient toward the nearest named constant.

    For each value v, returns min over `c ∈ {0, ±1, ±2, ±e}` of `(v - c)²`,
    summed over all elements. Differentiable (min-of-squares is smooth
    almost everywhere; the kink at the bisector between two constants
    is on a measure-zero set and Adam's momentum smooths it out).

    This is the Option-B counterpart to Option A's entropy penalty:
    instead of pushing softmax probabilities toward one-hot, it pushes
    free coefficients toward an integer / `e` lattice that we know the
    snap step recognizes.
    """
    constants = torch.tensor(
        [c for c, _ in _NAMED_CONSTANTS], dtype=tensor.dtype
    )  # (n_const,)
    flat = tensor.view(-1, 1)  # (n, 1)
    diffs = (flat - constants.unsqueeze(0)) ** 2  # (n, n_const)
    nearest = diffs.min(dim=1).values  # (n,)
    return nearest.sum()


def _train_one_linear(
    x_data: torch.Tensor,
    targets: torch.Tensor,
    depth: int,
    seed: int,
    search_iters: int = 2000,
    snap_iters: int = 1500,
    lr: float = 0.01,
    lam_disc_max: float = 0.5,
    verbose: bool = False,
) -> dict:
    """Train one Option-B tree from a single random seed.

    Two phases (analogous to Option A's tau anneal but additive
    rather than temperature-based):

      1. Search (it < search_iters): MSE only. Free coefficients
         explore the full real-valued parameter space.
      2. Snap (it >= search_iters): MSE + λ * discreteness_penalty,
         where λ ramps quadratically from 0 to `lam_disc_max`. The
         penalty pulls each coefficient toward the nearest member of
         {0, ±1, ±2, ±e}, which the snap step recognizes.

    There is no temperature schedule and no entropy penalty — both are
    softmax-specific. There is also no L1 penalty; sparsity emerges
    naturally from the discreteness penalty pulling toward 0 when no
    nonzero constant fits the local landscape.
    """
    torch.manual_seed(seed)
    tree = EMLTree1DLinear(depth)
    opt = torch.optim.Adam(tree.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = None
    nan_restarts = 0
    total = search_iters + snap_iters

    for it in range(1, total + 1):
        if nan_restarts > 20:
            break

        opt.zero_grad()
        pred, _, _ = tree(x_data)
        mse = torch.mean((pred - targets).abs() ** 2).real

        if it >= search_iters:
            t = (it - search_iters) / max(1, snap_iters)
            lam = lam_disc_max * (t ** 2)
            disc = (_discreteness_penalty(tree.leaf_logits)
                    + _discreteness_penalty(tree.gate_logits))
            loss = mse + lam * disc
        else:
            loss = mse

        if not torch.isfinite(loss):
            nan_restarts += 1
            if best_state is not None:
                tree.load_state_dict(best_state)
                opt = torch.optim.Adam(tree.parameters(), lr=lr)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(tree.parameters(), 1.0)
        opt.step()

        val = float(mse.item())
        if math.isfinite(val) and val < best_loss:
            best_loss = val
            best_state = {k: v.clone() for k, v in tree.state_dict().items()}

        if verbose and it % 500 == 0:
            print(f"  it={it:5d} mse={val:.3e} best={best_loss:.3e}")

    if best_state is not None:
        tree.load_state_dict(best_state)

    snapped = tree.snap()
    with torch.no_grad():
        pred_snap, _, _ = snapped(x_data)
        snap_mse = torch.mean((pred_snap - targets).abs() ** 2).real.item()

    return {
        "tree": tree,
        "snapped": snapped,
        "best_mse": best_loss,
        "snap_mse": snap_mse,
        "snap_rmse": math.sqrt(max(snap_mse, 0)),
        "expr": snapped.to_expr(),
        "nan_restarts": nan_restarts,
    }


# ─── Iterative snap (pruning) ────────────────────────────────
#
# The naive all-at-once snap destroys the fit because individual
# coefficients are not independent carriers of meaning. Iterative
# pruning snaps one coefficient at a time — smallest distance to
# nearest named constant first — and retrains the remaining free
# coefficients after each snap to absorb the discontinuity.
#
# This is the standard neural-network magnitude-pruning algorithm
# applied to the coefficient lattice {0, ±1, ±2, ±e}.

def _nearest_snap(v: float) -> tuple[float, float]:
    """Return (snap_target, distance) for the nearest named constant."""
    best_c, best_d = 0.0, abs(v)
    for c, _ in _NAMED_CONSTANTS:
        d = abs(v - c)
        if d < best_d:
            best_c, best_d = c, d
    # Also check nearest integer for larger values.
    ri = float(round(v))
    d = abs(v - ri)
    if d < best_d:
        best_c, best_d = ri, d
    return best_c, best_d


def iterative_snap(
    tree: EMLTree1DLinear,
    x_data: torch.Tensor,
    targets: torch.Tensor,
    retrain_iters: int = 300,
    lr: float = 0.005,
    max_mse_ratio: float = 100.0,
    verbose: bool = False,
) -> EMLTree1DLinear:
    """Iteratively snap a trained Option-B tree's coefficients.

    Algorithm:
      1. Collect all un-snapped coefficients
      2. Find the one closest to a named constant
      3. Snap it (set value, freeze via requires_grad mask)
      4. Retrain remaining free coefficients for `retrain_iters` steps
      5. If MSE blew up beyond `max_mse_ratio × initial_mse`, undo and
         try the next-closest coefficient
      6. Repeat until all coefficients are snapped or no more can be
         snapped without blowing up MSE

    Args:
        tree: a trained EMLTree1DLinear (will be deep-copied)
        x_data: training inputs (float64 tensor)
        targets: training targets (complex128 tensor)
        retrain_iters: gradient steps after each snap
        lr: learning rate for retraining
        max_mse_ratio: if MSE after retrain exceeds this × baseline,
            reject the snap and try the next candidate
        verbose: print each snap step

    Returns:
        A new EMLTree1DLinear with as many coefficients snapped as
        possible without destroying the fit.
    """
    tree = copy.deepcopy(tree)

    # Baseline MSE before any snapping.
    with torch.no_grad():
        pred, _, _ = tree(x_data)
        baseline_mse = float(torch.mean((pred - targets).abs() ** 2).real.item())

    if not math.isfinite(baseline_mse):
        return tree  # nothing we can do

    mse_ceiling = baseline_mse * max_mse_ratio

    # Build a flat index of all coefficients: (param_name, flat_idx).
    def _all_indices():
        indices = []
        for name, param in [("leaf", tree.leaf_logits),
                            ("gate", tree.gate_logits)]:
            flat = param.view(-1)
            for i in range(flat.numel()):
                indices.append((name, i))
        return indices

    frozen = set()  # (name, idx) pairs that are snapped and frozen
    n_total = tree.leaf_logits.numel() + tree.gate_logits.numel()

    for round_num in range(n_total):
        # Collect candidates: un-frozen coefficients with their snap distance.
        candidates = []
        for name, idx in _all_indices():
            if (name, idx) in frozen:
                continue
            param = tree.leaf_logits if name == "leaf" else tree.gate_logits
            v = float(param.view(-1)[idx].item())
            snap_target, dist = _nearest_snap(v)
            candidates.append((dist, name, idx, snap_target))

        if not candidates:
            break  # all snapped

        # Sort by distance — snap the "easiest" one first.
        candidates.sort()

        snapped_one = False
        for dist, name, idx, snap_target in candidates:
            # Save state in case we need to undo.
            saved_state = {k: v.clone() for k, v in tree.state_dict().items()}

            # Snap this coefficient.
            param = tree.leaf_logits if name == "leaf" else tree.gate_logits
            with torch.no_grad():
                param.view(-1)[idx] = snap_target

            frozen.add((name, idx))

            # Retrain remaining free coefficients.
            _retrain_free(tree, x_data, targets, frozen, retrain_iters, lr)

            # Check MSE.
            with torch.no_grad():
                pred, _, _ = tree(x_data)
                new_mse = float(torch.mean(
                    (pred - targets).abs() ** 2).real.item())

            if not math.isfinite(new_mse) or new_mse > mse_ceiling:
                # Reject — undo and try next candidate.
                tree.load_state_dict(saved_state)
                frozen.discard((name, idx))
                if verbose:
                    print(f"  reject: {name}[{idx}]→{snap_target:.3g} "
                          f"(mse {new_mse:.2e} > ceiling {mse_ceiling:.2e})")
                continue

            # Accept.
            if verbose:
                print(f"  snap: {name}[{idx}] {dist:.3g}→{snap_target:.3g} "
                      f"mse={new_mse:.2e}")
            # Update ceiling: allow gradual degradation but not catastrophic.
            mse_ceiling = max(mse_ceiling, new_mse * max_mse_ratio)
            snapped_one = True
            break  # restart scan with updated tree

        if not snapped_one:
            if verbose:
                print(f"  no more coefficients can be snapped "
                      f"(round {round_num})")
            break

    if verbose:
        n_snapped = len(frozen)
        print(f"  iterative snap: {n_snapped}/{n_total} coefficients "
              f"snapped")

    return tree


def _retrain_free(
    tree: EMLTree1DLinear,
    x_data: torch.Tensor,
    targets: torch.Tensor,
    frozen: set,
    iters: int,
    lr: float,
):
    """Retrain only the un-frozen coefficients for `iters` steps."""
    opt = torch.optim.Adam(tree.parameters(), lr=lr)

    for it in range(iters):
        opt.zero_grad()
        pred, _, _ = tree(x_data)
        mse = torch.mean((pred - targets).abs() ** 2).real
        if not torch.isfinite(mse):
            break
        mse.backward()
        torch.nn.utils.clip_grad_norm_(tree.parameters(), 1.0)

        # Zero out gradients on frozen coefficients so they don't move.
        with torch.no_grad():
            for name, idx in frozen:
                param = tree.leaf_logits if name == "leaf" else tree.gate_logits
                if param.grad is not None:
                    param.grad.view(-1)[idx] = 0.0

        opt.step()

        # Re-enforce exact frozen values (Adam momentum can drift them).
        with torch.no_grad():
            for name, idx in frozen:
                param = tree.leaf_logits if name == "leaf" else tree.gate_logits
                v = float(param.view(-1)[idx].item())
                snap_target, _ = _nearest_snap(v)
                param.view(-1)[idx] = snap_target


def discover_linear(
    x: np.ndarray,
    y: np.ndarray,
    max_depth: int = 4,
    n_tries: int = 8,
    verbose: bool = True,
    success_threshold: float = 1e-10,
) -> Optional[dict]:
    """Option-B counterpart of `eml_sr.discover()`.

    Tries depths 1 through `max_depth`, each with `n_tries` independent
    seeds. Returns the simplest (shallowest) formula that fits within
    `success_threshold`. API mirrors `discover()` for drop-in use in
    benchmarks.
    """
    x_t = torch.tensor(x, dtype=REAL)
    y_t = torch.tensor(y, dtype=DTYPE) if y.dtype.kind == "c" \
        else torch.tensor(y, dtype=DTYPE)

    best_overall = None
    for depth in range(1, max_depth + 1):
        if verbose:
            print(f"\n─── depth {depth} (linear / Option B) ───")
        best_at_depth = None
        for seed in range(n_tries):
            result = _train_one_linear(x_t, y_t, depth, seed)
            if verbose and (seed < 2 or result["snap_rmse"] < 1e-5):
                print(f"  seed {seed:2d}: snap_rmse={result['snap_rmse']:.3e} "
                      f"expr={result['expr'][:60]}")
            if best_at_depth is None or result["snap_mse"] < best_at_depth["snap_mse"]:
                best_at_depth = result
            if result["snap_mse"] < success_threshold:
                break

        if best_at_depth and best_at_depth["snap_mse"] < success_threshold:
            return {
                "expr": best_at_depth["expr"],
                "depth": depth,
                "snap_rmse": best_at_depth["snap_rmse"],
                "snapped_tree": best_at_depth["snapped"],
                "method": "linear",
            }
        if best_overall is None or (best_at_depth
                                    and best_at_depth["snap_mse"] < best_overall["snap_mse"]):
            best_overall = best_at_depth

    return {
        "expr": best_overall["expr"],
        "depth": best_overall["snapped"].depth,
        "snap_rmse": best_overall["snap_rmse"],
        "snapped_tree": best_overall["snapped"],
        "exact": False,
        "method": "linear",
    }
