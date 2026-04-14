"""eml_sr: Discover elementary formulas from univariate data.

Feed it (x, y) pairs. It finds the formula.

Architecture: a full binary tree where every internal node computes
eml(left, right) = exp(left) - ln(right). Leaves soft-route between
the constant 1 and the variable x. Training snaps the routing weights
to hard 0/1, recovering an exact symbolic expression.

Based on Odrzywolek (2026), "All elementary functions from a single
operator," Section 4.3 — adapted from the bivariate PyTorch v16 trainer
to the univariate case.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

DTYPE = torch.complex128
REAL = torch.float64
_CLAMP = 1e300
_BYPASS = 1.0 - torch.finfo(torch.float64).eps


# ─── EML operator ──────────────────────────────────────────────

def eml_op(x, y):
    """EML(x, y) = exp(x) - log(y), complex plane."""
    return torch.exp(x) - torch.log(y)


# ─── Tree ──────────────────────────────────────────────────────

class EMLTree1D(nn.Module):
    """Univariate EML tree of given depth.

    Leaves: soft choice between 1 and x  (2 logits per leaf)
    Gates:  soft bypass to 1 for each child input  (2 logits per node)
    """

    def __init__(self, depth: int, init_scale: float = 1.0):
        super().__init__()
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.n_internal = self.n_leaves - 1

        # Leaf logits: (n_leaves, 2) — [weight_for_1, weight_for_x]
        leaf_init = torch.randn(self.n_leaves, 2, dtype=REAL) * init_scale
        leaf_init[:, 0] += 2.0  # bias toward constant 1
        self.leaf_logits = nn.Parameter(leaf_init)

        # Gate logits: (n_internal, 2) — [left_bypass, right_bypass]
        # High value = bypass child, use 1 instead
        gate_init = torch.randn(self.n_internal, 2, dtype=REAL) * init_scale + 4.0
        self.gate_logits = nn.Parameter(gate_init)

    def forward(self, x, tau: float = 1.0):
        x = x.to(DTYPE)
        batch = x.shape[0]
        ones = torch.ones(batch, dtype=DTYPE)

        # Leaf values: soft mixture of 1 and x
        w = torch.softmax(self.leaf_logits / tau, dim=1).to(DTYPE)  # (n_leaves, 2)
        candidates = torch.stack([ones, x], dim=1)  # (batch, 2)
        level = candidates @ w.T  # (batch, n_leaves)

        # Bottom-up: pair children, apply gates, compute eml
        node_idx = 0
        while level.shape[1] > 1:
            n_pairs = level.shape[1] // 2
            left = level[:, 0::2]   # (batch, n_pairs)
            right = level[:, 1::2]

            s = torch.sigmoid(
                self.gate_logits[node_idx:node_idx + n_pairs] / tau
            )  # (n_pairs, 2)
            sl = s[:, 0].unsqueeze(0)  # (1, n_pairs)
            sr = s[:, 1].unsqueeze(0)

            # Blend: s=1 → use 1, s=0 → use child
            # Careful inf handling: blend real/imag separately
            bl = sl > _BYPASS
            br = sr > _BYPASS
            oml = 1.0 - sl
            omr = 1.0 - sr

            lr = torch.where(bl, 1.0, sl + oml * left.real)
            li = torch.where(bl, 0.0, oml * left.imag)
            rr = torch.where(br, 1.0, sr + omr * right.real)
            ri = torch.where(br, 0.0, omr * right.imag)

            left_in = torch.complex(lr, li)
            right_in = torch.complex(rr, ri)

            level = eml_op(left_in, right_in)

            # Clamp and scrub NaN
            level = torch.complex(
                torch.nan_to_num(level.real, nan=0.0, posinf=_CLAMP, neginf=-_CLAMP)
                    .clamp(-_CLAMP, _CLAMP),
                torch.nan_to_num(level.imag, nan=0.0, posinf=_CLAMP, neginf=-_CLAMP)
                    .clamp(-_CLAMP, _CLAMP),
            )
            node_idx += n_pairs

        leaf_probs = torch.softmax(self.leaf_logits / tau, dim=1)
        gate_probs = torch.sigmoid(self.gate_logits / tau)
        return level.squeeze(1), leaf_probs, gate_probs

    def snap(self):
        """Hard-snap all weights to 0/1. Returns a detached copy."""
        import copy
        tree = copy.deepcopy(self)
        with torch.no_grad():
            k = 50.0
            lc = torch.argmax(tree.leaf_logits, dim=1)
            new_leaf = torch.full_like(tree.leaf_logits, -k)
            new_leaf[torch.arange(tree.n_leaves), lc] = k
            tree.leaf_logits.copy_(new_leaf)

            gc = (tree.gate_logits >= 0).to(tree.gate_logits.dtype)
            tree.gate_logits.copy_(
                torch.where(gc > 0.5,
                            torch.full_like(tree.gate_logits, k),
                            torch.full_like(tree.gate_logits, -k))
            )
        return tree

    def to_expr(self) -> str:
        """Extract symbolic expression from snapped tree."""
        leaf_choices = torch.argmax(self.leaf_logits, dim=1).tolist()
        gate_choices = (self.gate_logits >= 0).tolist()

        labels = {0: "1", 1: "x"}
        # Build leaf expressions
        exprs = [labels[c] for c in leaf_choices]

        node_idx = 0
        while len(exprs) > 1:
            new_exprs = []
            for i in range(0, len(exprs), 2):
                left_bypass, right_bypass = gate_choices[node_idx]
                left = "1" if left_bypass else exprs[i]
                right = "1" if right_bypass else exprs[i + 1]
                new_exprs.append(f"eml({left}, {right})")
                node_idx += 1
            exprs = new_exprs

        return _simplify(exprs[0])

    def n_uncertain(self, threshold: float = 0.01) -> int:
        """Count weights that don't cleanly snap."""
        n = 0
        with torch.no_grad():
            lp = torch.softmax(self.leaf_logits, dim=1)
            if (lp.max(dim=1).values < 1.0 - threshold).any():
                n += int((lp.max(dim=1).values < 1.0 - threshold).sum())
            gp = torch.sigmoid(self.gate_logits)
            flat = gp.flatten()
            n += int(((flat > threshold) & (flat < 1.0 - threshold)).sum())
        return n


# ─── Expression simplification ─────────────────────────────────

def _simplify(expr: str) -> str:
    """Apply known EML identities to make expressions readable."""
    import re
    for _ in range(30):
        prev = expr
        # Atomic: eml(x, 1) = exp(x), eml(1, 1) = e
        expr = re.sub(r'eml\(([^,()]+), 1\)', r'exp(\1)', expr)
        expr = expr.replace("exp(1)", "e")

        # ln(x) = eml(1, exp(eml(1, x)))
        expr = re.sub(r'eml\(1, exp\(eml\(1, ([^()]+)\)\)\)', r'ln(\1)', expr)

        # eml(ln(a), exp(b)) = a - b
        expr = re.sub(r'eml\(ln\(([^()]+)\), exp\(([^()]+)\)\)', r'(\1 - \2)', expr)

        # exp(ln(a)) = a (for simple a)
        expr = re.sub(r'exp\(ln\(([^()]+)\)\)', r'\1', expr)

        # Clean up
        expr = expr.replace("(x - 0)", "x")
        expr = expr.replace("(0 - x)", "(-x)")

        if expr == prev:
            break
    return expr


# ─── Training ──────────────────────────────────────────────────

def _train_one(
    x_data: torch.Tensor,
    targets: torch.Tensor,
    depth: int,
    seed: int,
    search_iters: int = 2000,
    hard_iters: int = 800,
    lr: float = 0.01,
    tau_search: float = 1.0,
    tau_hard: float = 0.01,
    verbose: bool = False,
) -> dict:
    """Train one EML tree from a single random seed."""
    torch.manual_seed(seed)
    tree = EMLTree1D(depth)
    opt = torch.optim.Adam(tree.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = None
    nan_restarts = 0

    total = search_iters + hard_iters
    for it in range(1, total + 1):
        if nan_restarts > 20:
            break

        # Phase
        if it <= search_iters:
            tau = tau_search
            lam_ent = 0.0
        else:
            t = (it - search_iters) / max(1, hard_iters)
            tau = tau_search * (tau_hard / tau_search) ** (t ** 2)
            lam_ent = t * 0.01  # gentle entropy penalty

        opt.zero_grad()
        pred, leaf_p, gate_p = tree(x_data, tau=tau)
        mse = torch.mean((pred - targets).abs() ** 2).real

        # Entropy penalty encourages snapping
        leaf_ent = -(leaf_p * (leaf_p + 1e-12).log()).sum(dim=1).mean()
        loss = mse + lam_ent * leaf_ent

        if not torch.isfinite(loss):
            nan_restarts += 1
            if best_state is not None:
                tree.load_state_dict(best_state)
                opt = torch.optim.Adam(tree.parameters(), lr=lr)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(tree.parameters(), 1.0)
        opt.step()

        val = mse.item()
        if np.isfinite(val) and val < best_loss:
            best_loss = val
            best_state = {k: v.clone() for k, v in tree.state_dict().items()}

        if verbose and it % 500 == 0:
            print(f"  it={it:5d} tau={tau:.4f} mse={val:.3e} best={best_loss:.3e}")

    # Restore best and snap
    if best_state is not None:
        tree.load_state_dict(best_state)

    snapped = tree.snap()
    with torch.no_grad():
        pred_snap, _, _ = snapped(x_data, tau=0.01)
        snap_mse = torch.mean((pred_snap - targets).abs() ** 2).real.item()

    return {
        "tree": tree,
        "snapped": snapped,
        "best_mse": best_loss,
        "snap_mse": snap_mse,
        "snap_rmse": math.sqrt(max(snap_mse, 0)),
        "n_uncertain": tree.n_uncertain(),
        "expr": snapped.to_expr(),
        "nan_restarts": nan_restarts,
    }


def discover(
    x: np.ndarray,
    y: np.ndarray,
    max_depth: int = 4,
    n_tries: int = 16,
    verbose: bool = True,
    success_threshold: float = 1e-10,
) -> Optional[dict]:
    """Discover a formula relating x → y.

    Tries depths 2 through max_depth, multiple seeds per depth.
    Returns the simplest (shallowest) formula that fits within threshold.

    Args:
        x: input values (1D numpy array)
        y: output values (1D numpy array)
        max_depth: maximum tree depth to try (default 4)
        n_tries: random seeds per depth (default 16)
        verbose: print progress
        success_threshold: MSE threshold for "exact" recovery

    Returns:
        dict with keys: expr, depth, snap_rmse, snapped_tree
        or None if no formula found
    """
    # Prepare data
    x_t = torch.tensor(x, dtype=REAL)
    y_t = torch.tensor(y, dtype=DTYPE)

    best_overall = None

    for depth in range(1, max_depth + 1):
        n_leaves = 2 ** depth
        n_params = n_leaves * 2 + (n_leaves - 1) * 2
        # Scale effort by depth: shallow = fast, deep = more budget
        s_iters = 300 + depth * 400
        h_iters = 100 + depth * 150
        if verbose:
            print(f"\n─── depth {depth} ({n_leaves} leaves, {n_params} params, {n_tries} seeds) ───")

        best_at_depth = None
        n_success = 0

        for seed in range(n_tries):
            result = _train_one(
                x_t, y_t, depth, seed,
                search_iters=s_iters,
                hard_iters=h_iters,
                verbose=False,
            )

            if verbose and (seed < 3 or result["snap_rmse"] < 1e-5):
                tag = " ✓" if result["snap_mse"] < success_threshold else ""
                print(f"  seed {seed:2d}: snap_rmse={result['snap_rmse']:.3e} "
                      f"uncertain={result['n_uncertain']} "
                      f"expr={result['expr'][:60]}{tag}")

            if result["snap_mse"] < success_threshold:
                n_success += 1

            if best_at_depth is None or result["snap_mse"] < best_at_depth["snap_mse"]:
                best_at_depth = result

        if verbose:
            rate = n_success / n_tries * 100
            print(f"  success rate: {n_success}/{n_tries} ({rate:.0f}%)")
            if best_at_depth:
                print(f"  best: rmse={best_at_depth['snap_rmse']:.3e} → {best_at_depth['expr'][:80]}")

        if best_at_depth and best_at_depth["snap_mse"] < success_threshold:
            if verbose:
                print(f"\n  ✓ Found exact formula at depth {depth}")
            return {
                "expr": best_at_depth["expr"],
                "depth": depth,
                "snap_rmse": best_at_depth["snap_rmse"],
                "snapped_tree": best_at_depth["snapped"],
                "n_uncertain": best_at_depth["n_uncertain"],
            }

        if best_overall is None or (best_at_depth and best_at_depth["snap_mse"] < best_overall["snap_mse"]):
            best_overall = best_at_depth

    if verbose:
        print(f"\n  No exact formula found. Best: rmse={best_overall['snap_rmse']:.3e}")
        print(f"  → {best_overall['expr'][:120]}")

    return {
        "expr": best_overall["expr"],
        "depth": best_overall["snapped"].depth,
        "snap_rmse": best_overall["snap_rmse"],
        "snapped_tree": best_overall["snapped"],
        "n_uncertain": best_overall["n_uncertain"],
        "exact": False,
    }


# ─── CLI ───────────────────────────────────────────────────────

def _demo():
    """Demo: discover some known functions."""
    print("═══ EML Symbolic Regression — Univariate ═══\n")

    demos = [
        ("exp(x)", lambda x: np.exp(x), (0.5, 3.0)),
        ("e (constant)", lambda x: np.full_like(x, np.e), (0.5, 3.0)),
        ("ln(x)",  lambda x: np.log(x), (0.5, 5.0)),
        ("exp(x) - ln(x)", lambda x: np.exp(x) - np.log(x), (0.5, 3.0)),
    ]

    for name, fn, (lo, hi) in demos:
        print(f"\n{'='*50}")
        print(f"Target: y = {name}")
        print(f"{'='*50}")
        x = np.linspace(lo, hi, 30)
        y = fn(x)
        result = discover(x, y, max_depth=4, n_tries=4, verbose=True)
        if result:
            print(f"\nResult: {result['expr']}")
            print(f"  depth={result['depth']} rmse={result['snap_rmse']:.3e}")


if __name__ == "__main__":
    _demo()
