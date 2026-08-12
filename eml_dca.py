"""eml_dca: Difference-of-convex (DC) training for EML trees (issue #58).

`eml(x, y) = exp(x) - ln(y)` is jointly convex on its domain (y > 0):
the Hessian is `diag(e^x, 1/y**2)`, positive definite everywhere. This
module exploits that structure to train `EMLTree1DLinear` (Option B,
affine leaves + linear gates) with a DCA/majorization outer loop
instead of running Adam blind on the nonconvex landscape.

## The DC recursion, derived

Fix the tree structure and freeze all parameters except one *block*
theta (either the leaf coefficients, or the gate coefficients of one
tree level). Every per-sample scalar in the forward pass is then a DC
function of theta, tracked as a pair (p, q) of convex functions with
value p - q, alongside an interval [lo, hi] certified to contain the
value while theta stays inside an infinity-norm trust region of radius
delta around the current iterate.

Blocks are single levels because a gate input `a + b.x + g*child`
multiplies the in-block parameter g by a child value that would itself
depend on theta if lower gates shared the block; a product of two
block variables is not DC-trackable by pair propagation. With
one-level blocks, `g*child` always has exactly one factor in the
block, so every rule below applies.

Composition rules (h denotes a scalar convex function, L a Lipschitz
bound for h on the argument's certified interval):

1.  affine(theta)                 -> (affine, 0)
2.  (p1,q1) + (p2,q2)             -> (p1+p2, q1+q2)
3.  c*(p,q), c >= 0               -> (c*p, c*q);  c < 0 -> (-c*q, -c*p)
4.  h convex INCREASING of (p,q)  -> (h(p-q) + L*q, L*q)
5.  h convex DECREASING of (p,q)  -> (h(p-q) + L*p, L*p)

Rule 4 (rule 5 is the mirror image via h(t) = h_inc(-t)): with p, q
convex and h convex increasing L-Lipschitz on the reachable range,
`g = h(p-q) + L*q` is convex. Proof: at theta = c*t1 + (1-c)*t2,
p - q <= c(p1-q1) + (1-c)(p2-q2) + d where d = c*q1 + (1-c)*q2 - q(theta)
>= 0 by convexity of q. Monotonicity and Lipschitzness give
h(p-q) <= h(c(p1-q1) + (1-c)(p2-q2)) + L*d <= c*h1 + (1-c)*h2 + L*d,
and adding L*q(theta) cancels the L*d slack exactly.

Special exact cases (no Lipschitz slack): q == 0 makes rule 4 exact
for `exp` -- exp(convex) is convex; and an *affine* argument makes
`-ln(affine)` convex outright. Both are tracked with flags so shallow
trees keep exact convex parts.

The eml node itself is rule 2 over `exp(u)` (rule 4, L = e^{u_hi}) and
`-ln(v)` (rule 5, L = 1/v_lo, domain v_lo > 0).

## Squared loss: correction to the issue-#58 sketch

The issue sketches a depth-1 split `g = f**2 + 2*max(0,-y)*f`,
`h = 2*max(0,y)*f`. That g is NOT convex in general: for convex f,
f**2 has (along direction d) second variation
`2*(grad_f . d)**2 + 2*f*(d' Hess_f d)`, and the second term is negative
wherever f < 0 -- e.g. f(t) = e^t - 5 gives (f**2)'' = e^t(4e^t - 10) < 0
for t < ln(2.5). The correct split uses the monotone decomposition of
the square: with r = f - y,

    r**2 = max(r,0)**2 + min(r,0)**2

where `max(.,0)**2` is convex increasing (rule 4, L = 2*max(r_hi, 0))
and `min(.,0)**2` is convex decreasing (rule 5, L = 2*max(-r_lo, 0)).
The depth-1 *convexity* claim in the issue survives -- a node over
affine leaves is convex in theta (q == 0 throughout) -- and the unit
tests anchor exactly that, plus numerical convexity of both halves of
the corrected loss split.

## Domain handling

The DCA arm runs in REAL arithmetic on the branch v > 0 (the convexity
statement is a real-domain statement; the complex harness has no
convexity to exploit). Two guards keep iterates on-domain:

* a DC-representable hinge barrier `mu * max(0, v_min - v)**2` on every
  right-slot input v (rule 4 on the DC pair of `v_min - v`), and
* a conditional clamp `v <- max(v, v_floor)` inserted only when the
  certified interval reaches below `v_floor` (the clamp is convex
  increasing, rule 4 with L = 1; when the interval is safely positive
  it is skipped so exactness flags survive).

## Outer loop

DCA: at iterate theta_k, linearize H (the sum of q parts) and solve

    min_theta  G(theta) - <grad H(theta_k), theta>,   ||theta - theta_k||_inf <= delta

approximately (Adam + projection, per issue #58 "convex subproblem may
be solved approximately"). L constants and intervals are recomputed
each outer iteration from the current iterate and trust radius --
strictly this is majorization-minimization with an adaptive surrogate
rather than textbook global DCA, so every outer step is validated
against the exact loss (G - H reconstructs it exactly: the L terms
cancel in the difference by construction). Non-descent steps are
rejected and the trust region shrinks; descent steps grow it. Blocks
cycle leaves-first, then gate levels bottom-up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from eml_sr import REAL
from eml_sr_linear import EMLTree1DLinear

# Domain floor for right-slot (ln) inputs, and the barrier pushing
# iterates away from it.
V_MIN = 1e-3
BARRIER_MU = 10.0


# ---- DC pair -------------------------------------------------------------

@dataclass
class DCPair:
    """Per-sample DC decomposition value = p - q, with a certified
    interval [lo, hi] for the value over the current trust region.

    p, q: (batch,) float64 tensors inside the autograd graph of the
    active block. lo, hi: (batch,) tensors detached from the graph
    (interval bounds are constants of the subproblem).

    q_zero: q is identically 0 (value is certified convex).
    affine: value is affine in the block (implies q_zero).
    """

    p: torch.Tensor
    q: torch.Tensor
    lo: torch.Tensor
    hi: torch.Tensor
    q_zero: bool
    affine: bool

    @property
    def value(self) -> torch.Tensor:
        return self.p - self.q

    def __add__(self, other: "DCPair") -> "DCPair":
        return DCPair(
            self.p + other.p, self.q + other.q,
            self.lo + other.lo, self.hi + other.hi,
            self.q_zero and other.q_zero,
            self.affine and other.affine,
        )

    def add_const(self, c: torch.Tensor) -> "DCPair":
        """Add a per-sample constant (batch,) tensor (detached)."""
        return DCPair(self.p + c, self.q, self.lo + c, self.hi + c,
                      self.q_zero, self.affine)

    def scale(self, c: float) -> "DCPair":
        if c >= 0:
            return DCPair(c * self.p, c * self.q, c * self.lo, c * self.hi,
                          self.q_zero, self.affine)
        return DCPair((-c) * self.q, (-c) * self.p, c * self.hi, c * self.lo,
                      self.q_zero and _is_zero_tensor(self.p), self.affine)


def _is_zero_tensor(t: torch.Tensor) -> bool:
    with torch.no_grad():
        return bool(torch.all(t == 0).item())


def _const_pair(v: torch.Tensor) -> DCPair:
    """A per-sample constant (no block dependence)."""
    z = torch.zeros_like(v)
    return DCPair(v, z, v.detach().clone(), v.detach().clone(), True, True)


def exp_pair(a: DCPair) -> DCPair:
    """exp of a DC pair (rule 4; exact when q == 0)."""
    lo, hi = torch.exp(a.lo), torch.exp(a.hi)
    if a.q_zero:
        return DCPair(torch.exp(a.p), torch.zeros_like(a.p), lo, hi,
                      True, False)
    L = hi  # sup of exp' = exp on (-inf, hi], per sample
    return DCPair(torch.exp(a.value) + L * a.q, L * a.q, lo, hi,
                  False, False)


def clamp_floor_pair(a: DCPair, floor: float) -> DCPair:
    """max(value, floor) -- convex increasing, L = 1 (rule 4).

    Only called when the certified interval dips below `floor`.
    """
    fl = torch.as_tensor(floor, dtype=a.p.dtype)
    lo = torch.clamp(a.lo, min=floor)
    hi = torch.clamp(a.hi, min=floor)
    if a.q_zero:
        return DCPair(torch.clamp(a.p, min=floor), torch.zeros_like(a.p),
                      lo, hi, True, False)
    return DCPair(torch.clamp(a.value, min=fl) + a.q, a.q, lo, hi,
                  False, False)


def neg_ln_pair(a: DCPair) -> DCPair:
    """-ln of a DC pair (rule 5; exact-convex when the argument is
    affine). Caller must guarantee a.lo > 0 (use clamp_floor_pair)."""
    lo, hi = -torch.log(a.hi), -torch.log(a.lo)
    if a.affine:
        return DCPair(-torch.log(a.p), torch.zeros_like(a.p), lo, hi,
                      True, False)
    L = 1.0 / a.lo  # sup |d/dt (-ln t)| on [lo, hi]
    return DCPair(-torch.log(a.value) + L * a.p, L * a.p, lo, hi,
                  False, False)


def sq_loss_pair(r: DCPair) -> DCPair:
    """Squared residual r**2 as a DC pair via the monotone split
    r**2 = max(r,0)**2 + min(r,0)**2 (see module docstring for why the
    issue's sketch split is replaced)."""
    val = r.value
    zero = torch.zeros_like(r.p)
    out_lo = torch.where((r.lo <= 0) & (r.hi >= 0),
                         torch.zeros_like(r.lo),
                         torch.minimum(r.lo ** 2, r.hi ** 2))
    out_hi = torch.maximum(r.lo ** 2, r.hi ** 2)

    # Increasing half: max(r, 0)**2, L+ = 2*max(hi, 0).
    inc_val = torch.clamp(val, min=0) ** 2
    if r.q_zero:
        p_inc, q_inc = inc_val, zero
        inc_exact = True
    else:
        Lp = 2.0 * torch.clamp(r.hi, min=0)
        p_inc, q_inc = inc_val + Lp * r.q, Lp * r.q
        inc_exact = False

    # Decreasing half: min(r, 0)**2, L- = 2*max(-lo, 0).
    dec_val = torch.clamp(val, max=0) ** 2
    if r.affine:
        p_dec, q_dec = dec_val, zero
        dec_exact = True
    else:
        Lm = 2.0 * torch.clamp(-r.lo, min=0)
        p_dec, q_dec = dec_val + Lm * r.p, Lm * r.p
        dec_exact = False

    return DCPair(p_inc + p_dec, q_inc + q_dec, out_lo, out_hi,
                  inc_exact and dec_exact, False)


def hinge_sq_pair(a: DCPair, threshold: float) -> DCPair:
    """max(0, threshold - value)**2 -- the domain barrier. Built as the
    convex-increasing map max(t, 0)**2 (rule 4) applied to the DC pair
    of (threshold - value), whose roles are (q + threshold, p)."""
    neg = a.scale(-1.0).add_const(
        torch.full_like(a.p, threshold).detach())
    val = torch.clamp(neg.value, min=0) ** 2
    lo = torch.clamp(neg.lo, min=0) ** 2
    hi = torch.clamp(neg.hi, min=0) ** 2
    if neg.q_zero:
        return DCPair(val, torch.zeros_like(val), lo, hi, True, False)
    L = 2.0 * torch.clamp(neg.hi, min=0)
    return DCPair(val + L * neg.q, L * neg.q, lo, hi, False, False)


# ---- Tree evaluation as DC pairs ----------------------------------------

def _gate_level_slices(depth: int) -> list:
    """Index ranges of gate_logits rows per tree level, bottom-up,
    matching EMLTree1DLinear.forward's node_idx walk. Element k is the
    (start, count) slice for the k-th processed level (k = 0 is the
    level just above the leaves; the last element is the root)."""
    slices = []
    idx = 0
    n_pairs = 2 ** depth // 2
    while n_pairs >= 1:
        slices.append((idx, n_pairs))
        idx += n_pairs
        n_pairs //= 2
    return slices


def dc_forward(
    leaf_coeffs: torch.Tensor,
    gate_coeffs: torch.Tensor,
    leaf_base: torch.Tensor,
    gate_base: torch.Tensor,
    x: torch.Tensor,
    block: str,
    delta: float,
    depth: int,
) -> tuple:
    """Propagate DC pairs through a fixed-structure linear EML tree.

    The pair values (p, q) are functions of the *active* parameters
    (`leaf_coeffs` / `gate_coeffs`); the certified intervals and every
    Lipschitz constant derive from the *base* parameters (theta_k) and
    the trust radius `delta`, so G and H are fixed convex functions of
    the active block over `||theta - theta_k||_inf <= delta`. Callers
    must keep the active block inside that region (the inner solver
    projects onto it).

    Args:
        leaf_coeffs: (n_leaves, n_vars+1). Requires grad iff block=="leaves".
        gate_coeffs: (n_internal, 2, n_vars+2). Rows of the active gate
            level require grad iff block=="gates:<k>".
        leaf_base / gate_base: detached copies of theta_k (out-of-block
            tensors must equal their active counterparts).
        x: (batch, n_vars) float64.
        block: "leaves" or "gates:<k>" with k indexing
            `_gate_level_slices` (0 = bottom level).
        delta: trust-region radius (inf-norm) for the active block.
        depth: tree depth.

    Returns:
        (root DCPair, barrier DCPair) -- barrier sums the right-slot
        hinge penalties (already scaled by BARRIER_MU).
    """
    batch, n_vars = x.shape
    ones = torch.ones(batch, dtype=REAL)
    feats = torch.cat([ones.unsqueeze(1), x], dim=1)  # (batch, n_vars+1)
    abs_feats = feats.abs()
    gate_slices = _gate_level_slices(depth)
    active_gate_level = None
    if block.startswith("gates:"):
        active_gate_level = int(block.split(":", 1)[1])

    # Leaves: value = leaf_coeffs @ [1, x]; affine in the leaf block,
    # constant otherwise. Intervals center on the BASE leaf values.
    leaves = []
    n_leaves = 2 ** depth
    for i in range(n_leaves):
        if block == "leaves":
            v = feats @ leaf_coeffs[i]  # (batch,)
            v_base = (feats @ leaf_base[i]).detach()
            rad = delta * abs_feats.sum(dim=1)
            leaves.append(DCPair(v, torch.zeros_like(v),
                                 v_base - rad, v_base + rad, True, True))
        else:
            v = (feats @ leaf_base[i]).detach()
            leaves.append(_const_pair(v))

    barrier_terms = []
    level = leaves
    for lvl, (start, count) in enumerate(gate_slices):
        next_level = []
        in_block = (active_gate_level == lvl)
        for pair_i in range(count):
            left_child = level[2 * pair_i]
            right_child = level[2 * pair_i + 1]

            sides = []
            for side, child in ((0, left_child), (1, right_child)):
                if in_block:
                    # Affine in the gate block, features (1, x, child_value).
                    # The child is a frozen constant here (lower levels are
                    # out of block): child.q == 0 and child.p is detached.
                    gs = gate_coeffs[start + pair_i, side]
                    gs_base = gate_base[start + pair_i, side].detach()
                    cv = child.value.detach()
                    v = feats @ gs[: n_vars + 1] + gs[n_vars + 1] * cv
                    v_base = (feats @ gs_base[: n_vars + 1]
                              + gs_base[n_vars + 1] * cv).detach()
                    rad = delta * (abs_feats.sum(dim=1) + cv.abs())
                    sides.append(DCPair(v, torch.zeros_like(v),
                                        v_base - rad, v_base + rad,
                                        True, True))
                else:
                    gs_c = gate_base[start + pair_i, side].detach()
                    aff = (feats @ gs_c[: n_vars + 1]).detach()
                    scaled = child.scale(float(gs_c[n_vars + 1].item()))
                    sides.append(scaled.add_const(aff))

            u, v = sides
            if bool((v.lo <= V_MIN).any().item()):
                v = clamp_floor_pair(v, V_MIN)
            barrier_terms.append(hinge_sq_pair(v, 2.0 * V_MIN))
            node = exp_pair(u) + neg_ln_pair(v)
            next_level.append(node)
        level = next_level

    root = level[0]
    if not barrier_terms:  # depth-0 tree: single leaf, no ln slots
        zero = torch.zeros(batch, dtype=REAL)
        return root, _const_pair(zero)
    barrier = barrier_terms[0]
    for b in barrier_terms[1:]:
        barrier = barrier + b
    return root, barrier.scale(BARRIER_MU)


def dc_objective(
    leaf_coeffs: torch.Tensor,
    gate_coeffs: torch.Tensor,
    leaf_base: torch.Tensor,
    gate_base: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    block: str,
    delta: float,
    depth: int,
) -> tuple:
    """Full-objective DC pair: mean squared residual + domain barrier.

    Returns (G, H) scalar tensors with true_objective = G - H exactly
    (Lipschitz corrections cancel in the difference by construction).
    G and H are convex in the active block while it stays within
    `delta` of the base parameters.
    """
    root, barrier = dc_forward(leaf_coeffs, gate_coeffs, leaf_base,
                               gate_base, x, block, delta, depth)
    resid = root.add_const((-y).detach())
    loss = sq_loss_pair(resid)
    total = loss + barrier
    G = total.p.mean()
    H = total.q.mean()
    return G, H


# ---- DCA outer loop ------------------------------------------------------

def _clone_params(tree: EMLTree1DLinear) -> dict:
    return {k: v.detach().clone() for k, v in tree.state_dict().items()}


def _true_mse(tree: EMLTree1DLinear, x: torch.Tensor,
              y: torch.Tensor) -> float:
    """Exact real-domain objective (mse + barrier) at the current
    parameters, via G - H (exact for any block; delta = 0 keeps the
    intervals tight so clamp decisions match the evaluated point)."""
    with torch.no_grad():
        lf = tree.leaf_logits.detach()
        gt = tree.gate_logits.detach()
        G, H = dc_objective(lf, gt, lf, gt, x, y, "leaves", 0.0,
                            tree.depth)
        return float((G - H).item())


def dca_train(
    x_data: torch.Tensor,
    targets: torch.Tensor,
    depth: int,
    seed: int,
    n_vars: Optional[int] = None,
    outer_iters: int = 40,
    inner_iters: int = 60,
    inner_lr: float = 0.02,
    delta0: float = 0.5,
    verbose: bool = False,
) -> dict:
    """Block-cyclic DCA search for an EMLTree1DLinear.

    Mirrors `_train_one_linear`'s phase-1 search (same init, same seed
    contract) but replaces free-fall Adam with the DCA/MM outer loop.
    Returns the same result-dict shape minus snap fields; callers run
    the shared snap pipeline afterwards so the search optimizer is the
    only treatment difference between arms.
    """
    if x_data.dim() == 1:
        x_data = x_data.unsqueeze(1)
    x_real = x_data.real.to(REAL) if x_data.is_complex() else x_data.to(REAL)
    y_real = (targets.real if targets.is_complex() else targets).to(REAL)
    if n_vars is None:
        n_vars = x_real.shape[1]

    torch.manual_seed(seed)
    tree = EMLTree1DLinear(depth, n_vars=n_vars)

    blocks = ["leaves"] + [f"gates:{k}" for k in range(depth)]
    delta = {b: delta0 for b in blocks}
    hist = []
    cur = _true_mse(tree, x_real, y_real)
    n_outer_used = 0
    stall = 0

    for outer in range(outer_iters):
        improved_any = False
        for block in blocks:
            leaf_k = tree.leaf_logits.detach().clone()
            gate_k = tree.gate_logits.detach().clone()

            # grad of H at theta_k for the active block
            leaf_v = leaf_k.clone().requires_grad_(block == "leaves")
            gate_v = gate_k.clone().requires_grad_(block != "leaves")
            _, H = dc_objective(leaf_v, gate_v, leaf_k, gate_k,
                                x_real, y_real, block,
                                delta[block], depth)
            active_k = leaf_v if block == "leaves" else gate_v
            if H.requires_grad:
                (gH,) = torch.autograd.grad(H, active_k)
            else:
                gH = torch.zeros_like(active_k)

            # inner solve: min G(theta) - <gH, theta>, ||theta-theta_k||_inf <= delta
            theta = (leaf_k.clone() if block == "leaves"
                     else gate_k.clone()).requires_grad_(True)
            opt = torch.optim.Adam([theta], lr=inner_lr)
            base_k = leaf_k if block == "leaves" else gate_k
            for _ in range(inner_iters):
                opt.zero_grad()
                if block == "leaves":
                    G, _ = dc_objective(theta, gate_k, leaf_k, gate_k,
                                        x_real, y_real, block,
                                        delta[block], depth)
                else:
                    G, _ = dc_objective(leaf_k, theta, leaf_k, gate_k,
                                        x_real, y_real, block,
                                        delta[block], depth)
                obj = G - (gH * theta).sum()
                if not torch.isfinite(obj):
                    break
                obj.backward()
                opt.step()
                with torch.no_grad():
                    theta.clamp_(base_k - delta[block], base_k + delta[block])

            # descent check on the exact objective
            cand = theta.detach()
            with torch.no_grad():
                if block == "leaves":
                    tree.leaf_logits.copy_(cand)
                else:
                    tree.gate_logits.copy_(cand)
            new = _true_mse(tree, x_real, y_real)
            if math.isfinite(new) and new < cur - 1e-15:
                cur = new
                improved_any = True
                delta[block] = min(delta[block] * 1.5, 4.0)
            else:
                with torch.no_grad():
                    if block == "leaves":
                        tree.leaf_logits.copy_(leaf_k)
                    else:
                        tree.gate_logits.copy_(gate_k)
                delta[block] = max(delta[block] * 0.5, 1e-4)

        n_outer_used = outer + 1
        hist.append(cur)
        if verbose and (outer % 5 == 0 or outer == outer_iters - 1):
            print(f"  dca outer={outer:3d} obj={cur:.6e} "
                  f"delta_leaves={delta['leaves']:.3g}")
        stall = 0 if improved_any else stall + 1
        if stall >= 3:
            break

    return {
        "tree": tree,
        "best_mse": cur,
        "outer_iters_used": n_outer_used,
        "history": hist,
    }
