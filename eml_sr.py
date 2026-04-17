"""eml_sr: Discover elementary formulas from univariate data.

Feed it (x, y) pairs. It finds the formula.

Architecture: a full binary tree where every internal node computes
eml(left, right) = exp(left) - ln(right). Leaves soft-route between
the constant 1 and the variable x. Each internal node's gate soft-routes
each child input between [1, x, child-output]. Training snaps the
routing weights to hard choices, recovering an exact symbolic expression.

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
_CHILD_EPS = torch.finfo(torch.float64).eps


# ─── EML operator ──────────────────────────────────────────────

def eml_op(x, y):
    """EML(x, y) = exp(x) - log(y), complex plane."""
    return torch.exp(x) - torch.log(y)


# ─── Tree ──────────────────────────────────────────────────────

class EMLTree1D(nn.Module):
    """EML tree of given depth.

    Supports n_vars input variables (default 1 for backward compat).

    Leaves: soft choice over {1, x₁, ..., xₙ}  (n_vars+1 logits per leaf)
    Gates:  each child input soft-routes over {1, x₁, ..., xₙ, child}
            (n_vars+2 logits per side, 2 sides per node)
    """

    def __init__(self, depth: int, n_vars: int = 1, init_scale: float = 1.0):
        super().__init__()
        self.depth = depth
        self.n_vars = n_vars
        self.n_leaves = 2 ** depth
        self.n_internal = self.n_leaves - 1

        # Leaf logits: (n_leaves, n_vars+1) — [weight_for_1, weight_for_x1, ...]
        leaf_init = torch.randn(self.n_leaves, n_vars + 1, dtype=REAL) * init_scale
        leaf_init[:, 0] += 2.0  # bias toward constant 1
        self.leaf_logits = nn.Parameter(leaf_init)

        # Gate logits: (n_internal, 2, n_vars+2) — softmax over [1, x1, ..., xn, child]
        # Bias toward constant 1 (safe default that triggers leaf usage below).
        gate_init = torch.randn(self.n_internal, 2, n_vars + 2, dtype=REAL) * init_scale
        gate_init[..., 0] += 4.0
        self.gate_logits = nn.Parameter(gate_init)

    def forward(self, x, tau: float = 1.0):
        # Handle both 1D (batch,) and 2D (batch, n_vars) input.
        if x.dim() == 1:
            x = x.unsqueeze(1)  # (batch,) → (batch, 1)
        x = x.to(DTYPE)
        batch = x.shape[0]
        ones = torch.ones(batch, 1, dtype=DTYPE)

        # Leaf values: soft mixture of {1, x₁, ..., xₙ}
        w = torch.softmax(self.leaf_logits / tau, dim=1).to(DTYPE)  # (n_leaves, n_vars+1)
        candidates = torch.cat([ones, x], dim=1)  # (batch, n_vars+1)
        level = candidates @ w.T  # (batch, n_leaves)

        # Gate probabilities: (n_vars+2)-way softmax per side
        gate_probs = torch.softmax(self.gate_logits / tau, dim=-1)  # (n_internal, 2, n_vars+2)

        # Precompute x real/imag for gate blending: (batch, n_vars)
        x_r_all = x.real  # (batch, n_vars)
        x_i_all = x.imag  # (batch, n_vars)

        # Bottom-up: pair children, apply gates, compute eml
        node_idx = 0
        while level.shape[1] > 1:
            n_pairs = level.shape[1] // 2
            left = level[:, 0::2]   # (batch, n_pairs)
            right = level[:, 1::2]

            pg = gate_probs[node_idx:node_idx + n_pairs]  # (n_pairs, 2, n_vars+2)

            # Build gate inputs for left and right sides.
            # For each side s ∈ {0=left, 1=right}:
            #   input = p[0]*1 + p[1]*x₁ + ... + p[n_vars]*xₙ + p[n_vars+1]*child
            child_idx = self.n_vars + 1  # last index is child

            for side, child in [(0, left), (1, right)]:
                ps = pg[:, side, :]  # (n_pairs, n_vars+2)

                # Child contribution with safe zeroing
                p_child = ps[:, child_idx].unsqueeze(0)  # (1, n_pairs)
                mask_c = p_child > _CHILD_EPS
                safe_child_r = torch.where(mask_c, child.real,
                                           torch.zeros_like(child.real))
                safe_child_i = torch.where(mask_c, child.imag,
                                           torch.zeros_like(child.imag))
                p_child_safe = torch.where(mask_c, p_child,
                                           torch.zeros_like(p_child))

                # Start with constant term
                p_const = ps[:, 0].unsqueeze(0)  # (1, n_pairs)
                blend_r = p_const + p_child_safe * safe_child_r
                blend_i = p_child_safe * safe_child_i

                # Add variable contributions
                for v in range(self.n_vars):
                    p_v = ps[:, v + 1].unsqueeze(0)  # (1, n_pairs)
                    blend_r = blend_r + p_v * x_r_all[:, v:v+1]  # (batch, n_pairs)
                    blend_i = blend_i + p_v * x_i_all[:, v:v+1]

                # Clean bypass when a single choice has all the mass.
                b_to_1 = p_const > _BYPASS
                blend_r = torch.where(b_to_1, torch.ones_like(blend_r), blend_r)
                blend_i = torch.where(b_to_1, torch.zeros_like(blend_i), blend_i)
                for v in range(self.n_vars):
                    b_to_xv = ps[:, v + 1].unsqueeze(0) > _BYPASS
                    xv_r = x_r_all[:, v:v+1].expand_as(blend_r)
                    xv_i = x_i_all[:, v:v+1].expand_as(blend_i)
                    blend_r = torch.where(b_to_xv, xv_r, blend_r)
                    blend_i = torch.where(b_to_xv, xv_i, blend_i)

                if side == 0:
                    left_in = torch.complex(blend_r, blend_i)
                else:
                    right_in = torch.complex(blend_r, blend_i)

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
        return level.squeeze(1), leaf_probs, gate_probs

    def snap(self):
        """Hard-snap all weights to single-choice. Returns a detached copy."""
        import copy
        tree = copy.deepcopy(self)
        with torch.no_grad():
            k = 50.0

            # Leaves: argmax over n_vars+1 options
            lc = torch.argmax(tree.leaf_logits, dim=1)
            new_leaf = torch.full_like(tree.leaf_logits, -k)
            new_leaf[torch.arange(tree.n_leaves), lc] = k
            tree.leaf_logits.copy_(new_leaf)

            # Gates: argmax over n_vars+2 options, per side
            gc = torch.argmax(tree.gate_logits, dim=-1)  # (n_internal, 2)
            new_gate = torch.full_like(tree.gate_logits, -k)
            idx = torch.arange(tree.n_internal)
            for side in range(2):
                new_gate[idx, side, gc[:, side]] = k
            tree.gate_logits.copy_(new_gate)
        return tree

    def _var_labels(self) -> list:
        """Variable names for expression printing."""
        if self.n_vars == 1:
            return ["1", "x"]
        return ["1"] + [f"x{i+1}" for i in range(self.n_vars)]

    def to_expr(self) -> str:
        """Extract symbolic expression from snapped tree."""
        leaf_choices = torch.argmax(self.leaf_logits, dim=1).tolist()
        gate_choices = torch.argmax(self.gate_logits, dim=-1).tolist()  # (n_internal, 2)

        labels = self._var_labels()
        exprs = [labels[c] for c in leaf_choices]

        node_idx = 0
        while len(exprs) > 1:
            new_exprs = []
            for i in range(0, len(exprs), 2):
                lc, rc = gate_choices[node_idx]
                left = _resolve_gate(lc, exprs[i], self.n_vars)
                right = _resolve_gate(rc, exprs[i + 1], self.n_vars)
                new_exprs.append(f"eml({left}, {right})")
                node_idx += 1
            exprs = new_exprs

        return _simplify(exprs[0])

    def n_uncertain(self, threshold: float = 0.01) -> int:
        """Count weights that don't cleanly snap."""
        n = 0
        with torch.no_grad():
            lp = torch.softmax(self.leaf_logits, dim=1)
            n += int((lp.max(dim=1).values < 1.0 - threshold).sum())
            gp = torch.softmax(self.gate_logits, dim=-1)
            max_gp = gp.max(dim=-1).values  # (n_internal, 2)
            n += int((max_gp < 1.0 - threshold).sum())
        return n


def _resolve_gate(choice: int, child_expr: str, n_vars: int = 1) -> str:
    """Map a gate choice index to the input expression.

    Indices: 0 → "1", 1..n_vars → variable names, n_vars+1 → child.
    """
    if choice == 0:
        return "1"
    if choice <= n_vars:
        return "x" if n_vars == 1 else f"x{choice}"
    return child_expr


# ─── Expression simplification ─────────────────────────────────
#
# Recursive tree-walking simplifier. Parses an eml(...) string into
# an AST, rewrites bottom-up with known identities, and pretty-prints.
#
# AST node forms:
#   ('atom', s)       — leaf symbol ('1', 'x', '0', 'e', ...)
#   ('eml', l, r)     — unsimplified eml(l, r)
#   ('exp', a)        — exp(a)
#   ('ln',  a)        — ln(a)
#   ('sub', a, b)     — a - b
#   ('neg', a)        — -a

def _parse_eml(s: str):
    """Parse a string of eml(..., ...) and atoms into an AST."""
    s = s.strip()
    if s.startswith('eml(') and s.endswith(')'):
        inner = s[4:-1]
        depth = 0
        for i, c in enumerate(inner):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif c == ',' and depth == 0:
                return ('eml', _parse_eml(inner[:i]), _parse_eml(inner[i + 1:]))
        raise ValueError(f"Malformed eml expression: {s!r}")
    return ('atom', s)


def _simplify_ast(node):
    """Apply identities bottom-up. Returns simplified AST."""
    kind = node[0]

    if kind == 'atom':
        return node

    if kind == 'eml':
        # eml(a, b) = exp(a) - ln(b)
        left = _simplify_ast(node[1])
        right = _simplify_ast(node[2])
        return _simplify_ast(('sub', ('exp', left), ('ln', right)))

    if kind == 'exp':
        a = _simplify_ast(node[1])
        if a == ('atom', '1'):
            return ('atom', 'e')
        if a == ('atom', '0'):
            return ('atom', '1')
        if a[0] == 'ln':
            return a[1]
        return ('exp', a)

    if kind == 'ln':
        a = _simplify_ast(node[1])
        if a == ('atom', '1'):
            return ('atom', '0')
        if a == ('atom', 'e'):
            return ('atom', '1')
        if a[0] == 'exp':
            return a[1]
        return ('ln', a)

    if kind == 'sub':
        a = _simplify_ast(node[1])
        b = _simplify_ast(node[2])
        # a - 0 = a
        if b == ('atom', '0'):
            return a
        # 0 - a = -a
        if a == ('atom', '0'):
            return _simplify_ast(('neg', b))
        # a - a = 0
        if a == b:
            return ('atom', '0')
        # a - (a - c) = c
        if b[0] == 'sub' and b[1] == a:
            return b[2]
        # (a - c) - a = -c
        if a[0] == 'sub' and a[1] == b:
            return _simplify_ast(('neg', a[2]))
        # a - (-b) = a + b  (leave as sub(a, neg(b)) → print as a + b)
        return ('sub', a, b)

    if kind == 'neg':
        a = _simplify_ast(node[1])
        if a == ('atom', '0'):
            return ('atom', '0')
        if a[0] == 'neg':
            return a[1]
        return ('neg', a)

    return node


def _ast_to_str(node) -> str:
    """Pretty-print an AST node back to a string."""
    kind = node[0]
    if kind == 'atom':
        return node[1]
    if kind == 'exp':
        return f"exp({_ast_to_str(node[1])})"
    if kind == 'ln':
        return f"ln({_ast_to_str(node[1])})"
    if kind == 'sub':
        a = _ast_to_str(node[1])
        b = node[2]
        if b[0] == 'neg':
            return f"({a} + {_ast_to_str(b[1])})"
        return f"({a} - {_ast_to_str(b)})"
    if kind == 'neg':
        return f"(-{_ast_to_str(node[1])})"
    if kind == 'eml':
        return f"eml({_ast_to_str(node[1])}, {_ast_to_str(node[2])})"
    return str(node)


def _simplify(expr: str) -> str:
    """Apply known EML identities to make expressions readable."""
    try:
        ast = _parse_eml(expr)
        simplified = _simplify_ast(ast)
        return _ast_to_str(simplified)
    except Exception:
        return expr


def reachable_exprs(depth: int, n_vars: int = 1) -> set:
    """Enumerate all simplified expressions reachable by a snapped EML tree.

    Useful for vocabulary-reachability sanity checks (issue #11): if a
    target formula is not in `reachable_exprs(depth)`, no amount of seed
    budget will recover it at that depth.

    Args:
        depth: tree depth (0 = single leaf, 1 = one eml node, …)
        n_vars: number of input variables (default 1 for univariate)

    Returns:
        set of simplified expression strings

    Notes:
        - Depth grows the enumeration exponentially: at depth 3 with n_vars=1,
          we enumerate 2^(2^3) · 3^(2^3 · 2) = 256 · 6561 = ~1.7M raw trees
          before simplification. Call with caution for depth ≥ 3.
        - The simplifier collapses many trees to the same expression
          (e.g. depth-1 has 2² · 3² = 36 raw trees → 4 simplified atoms).
    """
    import itertools

    # Leaves: {1, x₁, …, xₙ}. Pre-snap labels.
    leaf_labels = ["1"] + ([f"x{i+1}" for i in range(n_vars)] if n_vars > 1 else ["x"])

    # Depth 0: single leaf, no gates.
    if depth == 0:
        return set(leaf_labels)

    n_leaves = 2 ** depth
    n_internal = n_leaves - 1

    # Gate options: {1, x₁, …, xₙ, child}
    # For enumeration purposes, we index 0=1, 1..n_vars=vars, n_vars+1=child.
    n_gate_options = n_vars + 2

    results: set = set()

    # Iterate all leaf assignments × all gate assignments.
    # leaves: n_leaves × (n_vars + 1) options
    # gates: n_internal × 2 × n_gate_options options
    leaf_space = itertools.product(range(n_vars + 1), repeat=n_leaves)
    for leaves in leaf_space:
        gate_space = itertools.product(
            range(n_gate_options), repeat=n_internal * 2
        )
        for flat_gates in gate_space:
            # Reshape flat_gates to (n_internal, 2)
            gates = [
                (flat_gates[2 * i], flat_gates[2 * i + 1])
                for i in range(n_internal)
            ]
            exprs = [leaf_labels[c] for c in leaves]
            node_idx = 0
            while len(exprs) > 1:
                new_exprs = []
                for i in range(0, len(exprs), 2):
                    lc, rc = gates[node_idx]
                    left = _resolve_gate(lc, exprs[i], n_vars)
                    right = _resolve_gate(rc, exprs[i + 1], n_vars)
                    new_exprs.append(f"eml({left}, {right})")
                    node_idx += 1
                exprs = new_exprs
            results.add(_simplify(exprs[0]))
    return results


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
    init_tree: Optional[nn.Module] = None,
) -> dict:
    """Train one EML tree from a single random seed.

    If ``init_tree`` is provided, it is used instead of creating a fresh
    ``EMLTree1D(depth)`` from the random seed. The seed still sets the
    torch RNG for reproducibility of the training loop itself.
    """
    torch.manual_seed(seed)
    tree = init_tree if init_tree is not None else EMLTree1D(depth)
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
    n_workers: int = 1,
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
        n_workers: parallel worker processes per depth (default 1 = serial).
            Set to a value <= os.cpu_count() to fan out independent seeds
            across cores. ~Linear speedup; the per-depth phase still has
            to wait for its slowest seed before moving to the next depth.

    Returns:
        dict with keys: expr, depth, snap_rmse, snapped_tree
        or None if no formula found
    """
    # Prepare data
    x_t = torch.tensor(x, dtype=REAL)
    y_t = torch.tensor(y, dtype=DTYPE)

    best_overall = None

    # The ladder starts at depth 0 — a single-leaf tree that emits either
    # the constant 1 or the variable x. Issue #11 diagnosed that the
    # absence of this "passthrough leaf" made the identity `y = x`
    # unreachable until depth ≥ 4 (every gated output is wrapped in an
    # outer `eml(·,·)`). The leaf-only tree closes that gap.
    for depth in range(0, max_depth + 1):
        n_leaves = 2 ** depth
        n_internal = n_leaves - 1
        n_params = n_leaves * 2 + n_internal * 2 * 3
        # Scale effort by depth: shallow = fast, deep = more budget
        s_iters = 300 + depth * 400
        h_iters = 100 + depth * 150
        if verbose:
            wtag = f", workers={n_workers}" if n_workers > 1 else ""
            print(f"\n─── depth {depth} ({n_leaves} leaves, {n_params} params, "
                  f"{n_tries} seeds{wtag}) ───")

        best_at_depth = None
        n_success = 0

        train_kwargs = dict(
            search_iters=s_iters,
            hard_iters=h_iters,
            verbose=False,
        )
        for seed, result in _run_seeds(x_t, y_t, depth, n_tries,
                                       train_kwargs, n_workers):

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


# ─── Growing tree (curriculum learning) ──────────────────────
#
# Rather than searching a fixed depth-D tree from scratch — a needle in a
# 2^D-sized haystack at deep depths — we grow the tree one leaf at a time.
# Start at depth 1, train to convergence, then split the leaf whose gradient
# magnitude is largest (the one most "wanting" to change). Each split replaces
# a leaf L with an eml subtree whose parameters are initialized so the subtree
# approximates exp(x) via eml(x, 1) — the optimizer adjusts surrounding weights
# to absorb the discrepancy. Analogous to Net2Net / progressive growing.

class GrowingEMLTree(nn.Module):
    """EML tree that grows by splitting leaves.

    Starts as a depth-1 tree (1 internal node, 2 leaves). Leaves can be
    split into eml subtrees to grow the tree incrementally, providing a
    warm start for deeper formulas.

    Nodes are stored in a flat list; each entry is a dict with 'type'
    ('leaf' or 'internal'), 'key' (into self._params), and for internal
    nodes 'left'/'right' (indices into self.nodes). When a leaf is split,
    its params are orphaned (unreachable from root) but not deleted, so
    optimizer state stays consistent across grow steps.
    """

    def __init__(self, init_scale: float = 1.0):
        super().__init__()
        self._params = nn.ParameterDict()
        self.nodes: list = []
        self.init_scale = init_scale
        self._next_key = 0
        # Build initial: two leaves + one internal root (depth 1)
        l0 = self._new_leaf()
        l1 = self._new_leaf()
        self.root = self._new_internal(l0, l1)

    # --- construction ---

    def _fresh_key(self, prefix: str) -> str:
        k = f"{prefix}_{self._next_key}"
        self._next_key += 1
        return k

    def _new_leaf(self) -> int:
        key = self._fresh_key("leaf")
        init = torch.randn(2, dtype=REAL) * self.init_scale
        init[0] += 2.0  # bias toward constant 1
        self._params[key] = nn.Parameter(init)
        idx = len(self.nodes)
        self.nodes.append({"type": "leaf", "key": key})
        return idx

    def _new_internal(self, left: int, right: int) -> int:
        key = self._fresh_key("gate")
        init = torch.randn(2, 3, dtype=REAL) * self.init_scale
        init[..., 0] += 4.0  # bias toward constant 1
        self._params[key] = nn.Parameter(init)
        idx = len(self.nodes)
        self.nodes.append({"type": "internal", "key": key,
                           "left": left, "right": right})
        return idx

    # --- topology ---

    def parent_of(self, idx: int):
        for i, n in enumerate(self.nodes):
            if n["type"] == "internal" and (n["left"] == idx or n["right"] == idx):
                return i
        return None

    def active_nodes(self) -> list:
        """Nodes reachable from root, pre-order."""
        seen = []
        seen_set = set()
        stack = [self.root]
        while stack:
            i = stack.pop()
            if i in seen_set:
                continue
            seen_set.add(i)
            seen.append(i)
            n = self.nodes[i]
            if n["type"] == "internal":
                stack.append(n["right"])
                stack.append(n["left"])
        return seen

    def active_leaves(self) -> list:
        return [i for i in self.active_nodes() if self.nodes[i]["type"] == "leaf"]

    def current_depth(self) -> int:
        def _d(i):
            n = self.nodes[i]
            if n["type"] == "leaf":
                return 0
            return 1 + max(_d(n["left"]), _d(n["right"]))
        return _d(self.root)

    def depth_of_node(self, target: int) -> int:
        """Depth (distance from root) of a given node index; -1 if unreachable."""
        def _d(i, cur):
            if i == target:
                return cur
            n = self.nodes[i]
            if n["type"] == "leaf":
                return -1
            dl = _d(n["left"], cur + 1)
            if dl >= 0:
                return dl
            return _d(n["right"], cur + 1)
        return _d(self.root, 0)

    def n_internal_active(self) -> int:
        return sum(1 for i in self.active_nodes() if self.nodes[i]["type"] == "internal")

    # --- forward ---

    def forward(self, x, tau: float = 1.0):
        x_c = x.to(DTYPE)
        cache: dict = {}
        val = self._eval(self.root, x_c, tau, cache)
        # Collect active softmaxed probs for entropy regularization / reporting.
        leaf_probs: list = []
        gate_probs: list = []
        for i in self.active_nodes():
            n = self.nodes[i]
            p = self._params[n["key"]]
            if n["type"] == "leaf":
                leaf_probs.append(torch.softmax(p / tau, dim=0))
            else:
                gate_probs.append(torch.softmax(p / tau, dim=-1))
        return val, leaf_probs, gate_probs

    def _eval(self, idx: int, x_c, tau: float, cache: dict):
        if idx in cache:
            return cache[idx]
        n = self.nodes[idx]
        if n["type"] == "leaf":
            logits = self._params[n["key"]]
            w = torch.softmax(logits / tau, dim=0).to(DTYPE)
            ones = torch.ones_like(x_c)
            val = w[0] * ones + w[1] * x_c
        else:
            gate = self._params[n["key"]]
            p = torch.softmax(gate / tau, dim=-1)  # (2, 3), real
            left_val = self._eval(n["left"], x_c, tau, cache)
            right_val = self._eval(n["right"], x_c, tau, cache)

            p0l, p1l, p2l = p[0, 0], p[0, 1], p[0, 2]
            p0r, p1r, p2r = p[1, 0], p[1, 1], p[1, 2]

            # Mask out child contribution when its probability is negligible,
            # to avoid 0*inf = nan (child values may be clamped-inf).
            mask_l = p2l > _CHILD_EPS
            mask_r = p2r > _CHILD_EPS
            zero_r = torch.zeros_like(left_val.real)
            zero_i = torch.zeros_like(left_val.imag)
            left_r = torch.where(mask_l, left_val.real, zero_r)
            left_i = torch.where(mask_l, left_val.imag, zero_i)
            right_r = torch.where(mask_r, right_val.real, zero_r)
            right_i = torch.where(mask_r, right_val.imag, zero_i)
            p2l_s = torch.where(mask_l, p2l, torch.zeros_like(p2l))
            p2r_s = torch.where(mask_r, p2r, torch.zeros_like(p2r))

            x_r = x_c.real
            x_i = x_c.imag
            lr = p0l + p1l * x_r + p2l_s * left_r
            li = p1l * x_i + p2l_s * left_i
            rr = p0r + p1r * x_r + p2r_s * right_r
            ri = p1r * x_i + p2r_s * right_i

            l_in = torch.complex(lr, li)
            r_in = torch.complex(rr, ri)
            val = eml_op(l_in, r_in)
            val = torch.complex(
                torch.nan_to_num(val.real, nan=0.0, posinf=_CLAMP, neginf=-_CLAMP)
                    .clamp(-_CLAMP, _CLAMP),
                torch.nan_to_num(val.imag, nan=0.0, posinf=_CLAMP, neginf=-_CLAMP)
                    .clamp(-_CLAMP, _CLAMP),
            )
        cache[idx] = val
        return val

    # --- growing ---

    def split_leaf(self, leaf_idx: int) -> int:
        """Replace `leaf_idx` with an eml subtree approximating exp(x)=eml(x, 1).

        Returns the new internal node index. The old leaf's parameters are
        orphaned (unreachable) but remain in the parameter dict so that any
        live optimizer state referencing them stays valid.
        """
        assert self.nodes[leaf_idx]["type"] == "leaf", "can only split leaves"
        # New leaves: left biased toward x, right biased toward constant 1
        new_l = self._new_leaf()
        new_r = self._new_leaf()
        with torch.no_grad():
            k = 4.0
            self._params[self.nodes[new_l]["key"]].copy_(
                torch.tensor([-k, k], dtype=REAL))  # → "x"
            self._params[self.nodes[new_r]["key"]].copy_(
                torch.tensor([k, -k], dtype=REAL))  # → "1"
        # New internal: gate routes both sides to child, giving eml(x, 1) = exp(x)
        new_int = self._new_internal(new_l, new_r)
        with torch.no_grad():
            k = 4.0
            g = torch.full((2, 3), -k, dtype=REAL)
            g[:, 2] = k  # "child"
            self._params[self.nodes[new_int]["key"]].copy_(g)
        # Rewire parent (or root) to point at new_int instead of leaf_idx.
        # Also bias the parent gate side toward "child" so the new subtree
        # is visible to the parent — otherwise a gate that had hard-snapped
        # to "1" or "x" would leave the split inert.
        if leaf_idx == self.root:
            self.root = new_int
        else:
            parent = self.parent_of(leaf_idx)
            assert parent is not None, "orphan leaf has no parent"
            pn = self.nodes[parent]
            if pn["left"] == leaf_idx:
                pn["left"] = new_int
                side = 0
            else:
                pn["right"] = new_int
                side = 1
            with torch.no_grad():
                pg = self._params[pn["key"]]
                # Reset the side's logits to bias toward "child" (index 2).
                pg[side, 0] = -2.0
                pg[side, 1] = -2.0
                pg[side, 2] = 2.0
        return new_int

    def leaf_gradient_magnitudes(self, x_data, y_target, tau: float) -> dict:
        """Compute ||∂MSE/∂leaf_logits|| for each active leaf.

        Used as the split-selection heuristic: the leaf most "wanting" to
        change is the one with the largest gradient magnitude.
        """
        self.zero_grad()
        pred, _, _ = self(x_data, tau=tau)
        mse = torch.mean((pred - y_target).abs() ** 2).real
        if not torch.isfinite(mse):
            return {}
        mse.backward()
        grads = {}
        for i in self.active_leaves():
            key = self.nodes[i]["key"]
            g = self._params[key].grad
            grads[i] = float(g.norm().item()) if g is not None else 0.0
        self.zero_grad()
        return grads

    # --- snapping & extraction ---

    def snap(self):
        """Hard-snap all active weights to single-choice. Returns detached copy."""
        import copy
        tree = copy.deepcopy(self)
        k = 50.0
        with torch.no_grad():
            for i in tree.active_nodes():
                n = tree.nodes[i]
                p = tree._params[n["key"]]
                if n["type"] == "leaf":
                    c = int(torch.argmax(p).item())
                    new = torch.full_like(p, -k)
                    new[c] = k
                    p.copy_(new)
                else:
                    choices = torch.argmax(p, dim=-1)
                    new = torch.full_like(p, -k)
                    new[0, int(choices[0].item())] = k
                    new[1, int(choices[1].item())] = k
                    p.copy_(new)
        return tree

    def n_uncertain(self, threshold: float = 0.01) -> int:
        n = 0
        with torch.no_grad():
            for i in self.active_nodes():
                node = self.nodes[i]
                p = self._params[node["key"]]
                if node["type"] == "leaf":
                    probs = torch.softmax(p, dim=0)
                    if probs.max().item() < 1.0 - threshold:
                        n += 1
                else:
                    probs = torch.softmax(p, dim=-1)
                    mx = probs.max(dim=-1).values
                    n += int((mx < 1.0 - threshold).sum().item())
        return n

    def to_expr(self) -> str:
        return _simplify(self._expr_at(self.root))

    def _expr_at(self, idx: int) -> str:
        n = self.nodes[idx]
        if n["type"] == "leaf":
            c = int(torch.argmax(self._params[n["key"]]).item())
            return "1" if c == 0 else "x"
        g = self._params[n["key"]]
        lc = int(torch.argmax(g[0]).item())
        rc = int(torch.argmax(g[1]).item())
        left_expr = self._expr_at(n["left"])
        right_expr = self._expr_at(n["right"])
        return f"eml({_resolve_gate(lc, left_expr)}, {_resolve_gate(rc, right_expr)})"


def _train_growing(
    tree: "GrowingEMLTree",
    x_data: torch.Tensor,
    y_target: torch.Tensor,
    opt: torch.optim.Optimizer,
    search_iters: int,
    hard_iters: int,
    lr: float,
    tau_search: float = 1.0,
    tau_hard: float = 0.01,
) -> float:
    """Train a growing tree for a given iteration budget.

    Mirrors `_train_one`'s two-phase schedule (search with fixed tau, then
    hard anneal with entropy penalty) but operates on the dynamic tree.
    Returns best MSE seen. Restores the best-seen state in-place.
    """
    best_loss = float("inf")
    best_state = None
    nan_restarts = 0
    total = search_iters + hard_iters
    for it in range(1, total + 1):
        if nan_restarts > 20:
            break

        if it <= search_iters:
            tau = tau_search
            lam_ent = 0.0
        else:
            t = (it - search_iters) / max(1, hard_iters)
            tau = tau_search * (tau_hard / tau_search) ** (t ** 2)
            lam_ent = t * 0.01

        opt.zero_grad()
        pred, leaf_ps, _ = tree(x_data, tau=tau)
        mse = torch.mean((pred - y_target).abs() ** 2).real

        ent = torch.zeros((), dtype=REAL)
        for lp in leaf_ps:
            ent = ent - (lp * (lp + 1e-12).log()).sum()
        if leaf_ps:
            ent = ent / len(leaf_ps)
        loss = mse + lam_ent * ent

        if not torch.isfinite(loss):
            nan_restarts += 1
            if best_state is not None:
                tree.load_state_dict(best_state)
                opt = torch.optim.Adam(tree.parameters(), lr=lr)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(tree.parameters(), 1.0)
        opt.step()

        v = mse.item()
        if np.isfinite(v) and v < best_loss:
            best_loss = v
            best_state = {k: t.clone() for k, t in tree.state_dict().items()}

    if best_state is not None:
        tree.load_state_dict(best_state)
    return best_loss


def discover_curriculum(
    x: np.ndarray,
    y: np.ndarray,
    max_depth: int = 6,
    n_tries: int = 8,
    lr: float = 0.01,
    verbose: bool = True,
    success_threshold: float = 1e-10,
) -> Optional[dict]:
    """Discover a formula via curriculum learning: grow the tree incrementally.

    Starts from a depth-1 tree. Trains to convergence. If the fit is not good
    enough, splits the active leaf whose gradient magnitude is largest — the
    split replaces it with `eml(x, 1) = exp(x)` — and resumes training. Repeats
    until `max_depth` is reached or an exact formula is found.

    This provides a warm start for depth-5 / depth-6 formulas that random
    initialization cannot easily find (the paper reports 0% recovery at
    depth 6 from random init).

    Args:
        x: input values (1D numpy array)
        y: output values (1D numpy array, real or complex)
        max_depth: maximum tree depth to grow to (default 6)
        n_tries: independent curriculum runs with different seeds (default 8)
        lr: Adam learning rate
        verbose: print progress
        success_threshold: MSE threshold for "exact" recovery

    Returns:
        dict with keys: expr, depth, snap_rmse, snapped_tree, n_splits, exact
    """
    x_t = torch.tensor(x, dtype=REAL)
    y_t = torch.tensor(y, dtype=DTYPE)

    best_overall = None

    # Depth-0 pre-check: the curriculum's growing tree starts at depth 1,
    # but the atoms `y = 1` and `y = x` are reachable one depth shallower
    # (a single-leaf tree, no gates). Try those two snaps explicitly before
    # kicking off the (more expensive) growing loop. See issue #11.
    with torch.no_grad():
        for leaf_choice, label in ((0, "1"), (1, "x")):
            probe = EMLTree1D(depth=0)
            new_leaf = torch.full_like(probe.leaf_logits, -50.0)
            new_leaf[0, leaf_choice] = 50.0
            probe.leaf_logits.copy_(new_leaf)
            pred_s, _, _ = probe(x_t, tau=0.01)
            mse_s = torch.mean((pred_s - y_t).abs() ** 2).real.item()
            if mse_s < success_threshold:
                if verbose:
                    print(f"  ✓ Depth-0 atom matches: y = {label} "
                          f"(rmse={math.sqrt(max(mse_s, 0)):.3e})")
                return {
                    "expr": label,
                    "depth": 0,
                    "snap_rmse": math.sqrt(max(mse_s, 0)),
                    "snapped_tree": probe,
                    "n_uncertain": 0,
                    "n_splits": 0,
                    "seed": 0,
                    "exact": True,
                }

    for seed in range(n_tries):
        torch.manual_seed(seed)
        tree = GrowingEMLTree()
        opt = torch.optim.Adam(tree.parameters(), lr=lr)

        if verbose:
            print(f"\n─── curriculum seed {seed} ───")

        def _finalize_and_check():
            """Run one hard-anneal pass on a fresh copy and evaluate snap MSE.

            Hard-annealing is destructive to the soft state we want to keep
            growing from, so we do it on a deepcopy. Only the snapped tree
            and its MSE are returned; `tree` itself stays soft.
            """
            import copy as _copy
            scratch = _copy.deepcopy(tree)
            scratch_opt = torch.optim.Adam(scratch.parameters(), lr=lr)
            _train_growing(scratch, x_t, y_t, scratch_opt,
                           search_iters=0, hard_iters=400, lr=lr)
            snp = scratch.snap()
            with torch.no_grad():
                pred_s, _, _ = snp(x_t, tau=0.01)
                mse_s = torch.mean((pred_s - y_t).abs() ** 2).real.item()
            return snp, mse_s

        # Initial train at depth 1 (search phase only — keep soft).
        _train_growing(tree, x_t, y_t, opt,
                       search_iters=600, hard_iters=0, lr=lr)

        snapped, snap_mse = _finalize_and_check()
        if verbose:
            print(f"  depth {tree.current_depth()}: "
                  f"snap_rmse={math.sqrt(max(snap_mse, 0)):.3e} "
                  f"expr={snapped.to_expr()[:60]}")

        grow_step = 0
        # Cap to prevent runaway. A fully-grown depth-D tree has 2^D - 1
        # internal nodes, so at most 2^D - 2 splits from the initial 1-node tree.
        max_grow_steps = max(1, 2 ** max_depth - 2)

        while (snap_mse >= success_threshold
               and tree.current_depth() < max_depth
               and grow_step < max_grow_steps):
            grow_step += 1
            # Pick leaf to split: largest gradient magnitude among leaves
            # whose depth is still below max_depth.
            grads = tree.leaf_gradient_magnitudes(x_t, y_t, tau=1.0)
            splittable = {
                i: g for i, g in grads.items()
                if tree.depth_of_node(i) + 1 <= max_depth
            }
            if not splittable:
                break
            leaf_to_split = max(splittable, key=splittable.get)
            tree.split_leaf(leaf_to_split)

            # New params → fresh optimizer state. Old Adam state referenced
            # dead params; re-initializing avoids stale-param issues.
            opt = torch.optim.Adam(tree.parameters(), lr=lr)

            # Search-phase training only. The tree stays soft until the
            # final finalize call, so premature commitments to "1" or "x"
            # don't lock down branches before they've been grown.
            d = tree.current_depth()
            _train_growing(
                tree, x_t, y_t, opt,
                search_iters=400 + 200 * d,
                hard_iters=0,
                lr=lr,
            )

            snapped, snap_mse = _finalize_and_check()
            if verbose:
                print(f"  split #{grow_step} → depth {d}: "
                      f"snap_rmse={math.sqrt(max(snap_mse, 0)):.3e} "
                      f"expr={snapped.to_expr()[:60]}")

        result = {
            "snapped": snapped,
            "snap_mse": snap_mse,
            "snap_rmse": math.sqrt(max(snap_mse, 0)),
            "depth": tree.current_depth(),
            "expr": snapped.to_expr(),
            "n_uncertain": tree.n_uncertain(),
            "n_splits": grow_step,
            "seed": seed,
        }

        if snap_mse < success_threshold:
            if verbose:
                print(f"\n  ✓ Found exact formula "
                      f"(seed {seed}, depth {result['depth']}, "
                      f"splits={grow_step}): {result['expr']}")
            return {
                "expr": result["expr"],
                "depth": result["depth"],
                "snap_rmse": result["snap_rmse"],
                "snapped_tree": result["snapped"],
                "n_uncertain": result["n_uncertain"],
                "n_splits": result["n_splits"],
                "seed": seed,
                "exact": True,
            }

        if best_overall is None or result["snap_mse"] < best_overall["snap_mse"]:
            best_overall = result

    if verbose:
        print(f"\n  No exact formula found. "
              f"Best: rmse={best_overall['snap_rmse']:.3e}")
        print(f"  → {best_overall['expr'][:120]}")

    return {
        "expr": best_overall["expr"],
        "depth": best_overall["depth"],
        "snap_rmse": best_overall["snap_rmse"],
        "snapped_tree": best_overall["snapped"],
        "n_uncertain": best_overall["n_uncertain"],
        "n_splits": best_overall["n_splits"],
        "seed": best_overall["seed"],
        "exact": False,
    }


# ─── Data normalization ────────────────────────────────────────
#
# EML chains overflow easily — a single eml(x, y) is exp(x) − ln(y), so any
# input outside roughly [-30, 30] sends the partial down the clamp path
# (1e300) and kills gradients. For real-world CSV data we need to map x and
# y into a numerically friendly range first, train in that space, then
# (optionally) report the transformations alongside the discovered formula.
#
# Two transforms are supported:
#   "minmax"   — affine map onto [target_lo, target_hi] (default [-1, 1])
#   "standard" — zero-mean, unit-variance
#   "none"     — pass-through (use only if data is already well scaled)
#
# Both are *affine*, x' = a·x + b, y' = c·y + d, so the discovered formula
# in primed coordinates can be back-substituted by hand: y = (f(a·x + b) − d)/c.
# We don't try to do that algebraically — most non-linear EML expressions
# don't simplify cleanly under affine substitution.


class Normalizer:
    """Affine normalizer with invertible transform/inverse for x and y.

    Stored as a plain dict-friendly object so it can be pickled into pool
    workers and serialized alongside results.
    """

    def __init__(self, x_a: float, x_b: float, y_a: float, y_b: float, mode: str):
        # x' = x_a * x + x_b   ;   y' = y_a * y + y_b
        self.x_a = float(x_a)
        self.x_b = float(x_b)
        self.y_a = float(y_a)
        self.y_b = float(y_b)
        self.mode = mode

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, mode: str = "minmax",
            target_lo: float = -1.0, target_hi: float = 1.0) -> "Normalizer":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if mode == "none":
            return cls(1.0, 0.0, 1.0, 0.0, mode)
        if mode == "minmax":
            def _ab(v):
                lo, hi = float(v.min()), float(v.max())
                if hi == lo:
                    return 0.0, 0.0  # constant column collapses to 0
                a = (target_hi - target_lo) / (hi - lo)
                b = target_lo - a * lo
                return a, b
            xa, xb = _ab(x)
            ya, yb = _ab(y)
            return cls(xa, xb, ya, yb, mode)
        if mode == "standard":
            def _ab(v):
                mu = float(v.mean())
                sd = float(v.std())
                if sd < 1e-12:
                    return 0.0, 0.0
                return 1.0 / sd, -mu / sd
            xa, xb = _ab(x)
            ya, yb = _ab(y)
            return cls(xa, xb, ya, yb, mode)
        raise ValueError(f"unknown normalization mode: {mode!r}")

    def transform_x(self, x):
        return self.x_a * x + self.x_b

    def transform_y(self, y):
        return self.y_a * y + self.y_b

    def inverse_x(self, xp):
        return (xp - self.x_b) / self.x_a if self.x_a != 0 else np.zeros_like(xp)

    def inverse_y(self, yp):
        return (yp - self.y_b) / self.y_a if self.y_a != 0 else np.zeros_like(yp)

    def describe(self) -> str:
        return (f"x' = {self.x_a:.6g} * x + {self.x_b:.6g}   "
                f"y' = {self.y_a:.6g} * y + {self.y_b:.6g}   "
                f"(mode={self.mode})")

    def to_dict(self) -> dict:
        return {"x_a": self.x_a, "x_b": self.x_b,
                "y_a": self.y_a, "y_b": self.y_b, "mode": self.mode}


# ─── Parallel seed training ────────────────────────────────────
#
# The biggest perf opportunity in `discover()` is that all `n_tries` seeds
# at a given depth are independent. The single-process loop runs them
# sequentially even though a modern box has 8+ cores sitting idle. Running
# each seed in its own worker process gives near-linear speedup as long as
# we (a) cap each worker to a single torch thread to avoid oversubscription,
# and (b) use `fork` so we don't pay the import-torch tax in every worker.
#
# Tradeoffs:
#   - Workers can't short-circuit each other when one finds an exact fit;
#     the depth still completes its full batch. Net effect is still a big
#     win because the cost-per-seed is what dominates.
#   - The result dict (which embeds two nn.Module trees) pickles cleanly,
#     just bulkier than a scalar payload — usually a few hundred KB.
#   - If `n_workers <= 1`, we run inline and skip the pool entirely.


def _train_one_worker(packed):
    """Pickle-safe trampoline so a multiprocessing.Pool can call _train_one.

    Each worker is its own Python interpreter; we cap torch threads to 1 so
    `n_workers` workers don't all try to grab every core at once.
    """
    import torch as _torch
    _torch.set_num_threads(1)
    x_arr, y_arr, depth, seed, kwargs = packed
    x_t = _torch.tensor(x_arr, dtype=REAL)
    y_t = _torch.tensor(y_arr, dtype=DTYPE)
    return _train_one(x_t, y_t, depth, seed, **kwargs)


def _run_seeds(x_t, y_t, depth, n_tries, train_kwargs, n_workers):
    """Run `n_tries` independent seeds at `depth`, optionally in parallel.

    Yields each (seed, result) in submission order so the caller can apply
    its own early-stop logic.
    """
    if n_workers and n_workers > 1:
        import multiprocessing as mp
        # `fork` is much faster than `spawn` on Linux — children inherit the
        # parent process image instead of re-importing torch from scratch.
        # Caveat: forking after libgomp/libmkl have spun up worker threads
        # is occasionally flaky, hence OMP_NUM_THREADS=1 + torch.set_num_threads(1)
        # in the worker.
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")
        x_arr = x_t.detach().cpu().numpy()
        y_arr = y_t.detach().cpu().numpy()
        packed = [(x_arr, y_arr, depth, s, train_kwargs) for s in range(n_tries)]
        with ctx.Pool(min(n_workers, n_tries)) as pool:
            for s, result in enumerate(pool.imap(_train_one_worker, packed)):
                yield s, result
    else:
        for s in range(n_tries):
            yield s, _train_one(x_t, y_t, depth, s, **train_kwargs)


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


def _demo_curriculum():
    """Demo: discover a deep formula via curriculum learning."""
    print("═══ EML Symbolic Regression — Curriculum Growing ═══\n")

    demos = [
        ("exp(exp(x))",      lambda x: np.exp(np.exp(x)),         (-0.5, 1.0)),
        ("exp(exp(exp(x)))", lambda x: np.exp(np.exp(np.exp(x))), (-1.0, 0.5)),
        ("exp(x) - ln(x)",   lambda x: np.exp(x) - np.log(x),     (0.5, 3.0)),
    ]

    for name, fn, (lo, hi) in demos:
        print(f"\n{'='*50}")
        print(f"Target: y = {name}")
        print(f"{'='*50}")
        x = np.linspace(lo, hi, 30)
        y = fn(x)
        result = discover_curriculum(x, y, max_depth=4, n_tries=3, verbose=True)
        if result:
            print(f"\nResult: {result['expr']}")
            print(f"  depth={result['depth']} splits={result['n_splits']} "
                  f"rmse={result['snap_rmse']:.3e}")


def discover_csv(
    csv_path: str,
    x_col: str,
    y_col: str,
    max_depth: int = 4,
    n_tries: int = 16,
    method: str = "discover",
    normalize: str = "minmax",
    n_workers: int = 1,
    verbose: bool = True,
    success_threshold: float = 1e-10,
) -> dict:
    """Read (x, y) from a CSV and run symbolic regression.

    The discovered formula is in *normalized* coordinates — see the
    `normalizer` field of the returned dict for the affine transform that
    was applied. Predictions on new data must be normalized first.

    Args:
        csv_path: path to a CSV file
        x_col: column name for the independent variable
        y_col: column name for the dependent variable
        max_depth: maximum tree depth to search
        n_tries: random seeds per depth (or per curriculum run)
        method: "discover" (fixed-depth ladder) or "curriculum" (growing tree)
        normalize: "minmax" | "standard" | "none"
        n_workers: parallel seed workers (only used by `discover`)
        verbose: print progress
        success_threshold: MSE threshold for "exact" recovery

    Returns:
        Result dict from discover()/discover_curriculum() augmented with:
          - normalizer: the Normalizer used
          - x_col, y_col: column names
          - n_samples: number of rows
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    if x_col not in df.columns:
        raise ValueError(f"x column {x_col!r} not in {list(df.columns)}")
    if y_col not in df.columns:
        raise ValueError(f"y column {y_col!r} not in {list(df.columns)}")
    sub = df[[x_col, y_col]].dropna()
    x = sub[x_col].to_numpy(dtype=np.float64)
    y = sub[y_col].to_numpy(dtype=np.float64)

    norm = Normalizer.fit(x, y, mode=normalize)
    x_n = norm.transform_x(x)
    y_n = norm.transform_y(y)

    if verbose:
        print(f"Loaded {len(x)} rows from {csv_path}")
        print(f"  x={x_col}: [{x.min():.6g}, {x.max():.6g}]   "
              f"y={y_col}: [{y.min():.6g}, {y.max():.6g}]")
        print(f"Normalizer: {norm.describe()}")
        print(f"  x' range: [{x_n.min():.6g}, {x_n.max():.6g}]   "
              f"y' range: [{y_n.min():.6g}, {y_n.max():.6g}]")

    if method == "curriculum":
        result = discover_curriculum(
            x_n, y_n,
            max_depth=max_depth, n_tries=n_tries,
            verbose=verbose, success_threshold=success_threshold,
        )
    elif method == "discover":
        result = discover(
            x_n, y_n,
            max_depth=max_depth, n_tries=n_tries,
            verbose=verbose, success_threshold=success_threshold,
            n_workers=n_workers,
        )
    else:
        raise ValueError(f"unknown method {method!r}; pick 'discover' or 'curriculum'")

    if result is None:
        return {"normalizer": norm, "x_col": x_col, "y_col": y_col,
                "n_samples": len(x), "expr": None}

    result["normalizer"] = norm
    result["x_col"] = x_col
    result["y_col"] = y_col
    result["n_samples"] = len(x)
    return result


def _cli_csv(args):
    """Handler for the `csv` subcommand."""
    import time
    t0 = time.time()
    result = discover_csv(
        csv_path=args.csv,
        x_col=args.x_col,
        y_col=args.y_col,
        max_depth=args.max_depth,
        n_tries=args.tries,
        method=args.method,
        normalize=args.normalize,
        n_workers=args.workers,
        verbose=not args.quiet,
        success_threshold=args.threshold,
    )
    elapsed = time.time() - t0

    print()
    print("═" * 60)
    print("Result")
    print("═" * 60)
    if result.get("expr") is None:
        print("No formula found.")
        return
    print(f"  formula (in normalized coords):  y' = {result['expr']}")
    print(f"  depth                            {result['depth']}")
    print(f"  snap rmse                        {result['snap_rmse']:.3e}")
    print(f"  exact recovery                   {result.get('exact', True)}")
    print(f"  uncertain weights                {result.get('n_uncertain', 0)}")
    print(f"  elapsed                          {elapsed:.2f}s")
    norm = result["normalizer"]
    print(f"  normalizer                       {norm.describe()}")
    print()
    print("To recover the original formula, substitute:")
    print(f"  x' = {norm.x_a:.6g} * {args.x_col} + {norm.x_b:.6g}")
    print(f"  y' = {norm.y_a:.6g} * {args.y_col} + {norm.y_b:.6g}")
    print(f"  → {args.y_col} = (formula({args.x_col}) - {norm.y_b:.6g}) / {norm.y_a:.6g}")


def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="eml_sr",
        description="EML symbolic regression — discover elementary formulas from data.",
    )
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("demo", help="Run the built-in fixed-depth demo")
    sub.add_parser("curriculum", help="Run the built-in curriculum-growing demo")

    pc = sub.add_parser("csv", help="Discover a formula from a CSV file")
    pc.add_argument("csv", help="Path to CSV file")
    pc.add_argument("--x-col", required=True, help="Column name for the input variable")
    pc.add_argument("--y-col", required=True, help="Column name for the output variable")
    pc.add_argument("--max-depth", type=int, default=4, help="Maximum tree depth (default 4)")
    pc.add_argument("--tries", type=int, default=16, help="Seeds per depth (default 16)")
    pc.add_argument("--method", choices=["discover", "curriculum"], default="discover",
                    help="Search method (default discover)")
    pc.add_argument("--normalize", choices=["minmax", "standard", "none"],
                    default="minmax", help="Data normalization (default minmax)")
    pc.add_argument("--workers", type=int, default=1,
                    help="Parallel seed workers (default 1; set to ncpu for speedup)")
    pc.add_argument("--threshold", type=float, default=1e-10,
                    help="MSE threshold for 'exact' recovery (default 1e-10)")
    pc.add_argument("--quiet", action="store_true", help="Suppress progress output")
    pc.set_defaults(func=_cli_csv)
    return p


if __name__ == "__main__":
    import sys
    parser = _build_parser()
    args = parser.parse_args()
    if args.cmd == "demo" or args.cmd is None:
        _demo()
    elif args.cmd == "curriculum":
        _demo_curriculum()
    else:
        args.func(args)
