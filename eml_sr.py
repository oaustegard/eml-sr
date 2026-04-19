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

from eml_operators import EML, OperatorConfig

DTYPE = torch.complex128
REAL = torch.float64
_CLAMP = 1e300
_BYPASS = 1.0 - torch.finfo(torch.float64).eps
_CHILD_EPS = torch.finfo(torch.float64).eps
_TERM_EPS = torch.finfo(torch.float64).eps


# ─── EML operator ──────────────────────────────────────────────

def eml_op(x, y):
    """EML(x, y) = exp(x) - log(y), complex plane.

    Kept as a module-level function for backward compatibility with
    callers that import it directly. For operator-parameterised trees,
    the tree's ``op_config.op`` is called in place of this function.
    """
    return torch.exp(x) - torch.log(y)


# ─── Tree ──────────────────────────────────────────────────────

class EMLTree1D(nn.Module):
    """EML tree of given depth.

    Supports n_vars input variables (default 1 for backward compat).

    Leaves: soft choice over {1, x₁, ..., xₙ}  (n_vars+1 logits per leaf)
    Gates:  each child input soft-routes over {1, x₁, ..., xₙ, child}
            (n_vars+2 logits per side, 2 sides per node)
    """

    def __init__(self, depth: int, n_vars: int = 1, init_scale: float = 1.0,
                 op_config: OperatorConfig = EML):
        super().__init__()
        self.depth = depth
        self.n_vars = n_vars
        self.n_leaves = 2 ** depth
        self.n_internal = self.n_leaves - 1
        self.op_config = op_config

        # Leaf logits: (n_leaves, n_vars+1) — [weight_for_terminal, weight_for_x1, ...]
        leaf_init = torch.randn(self.n_leaves, n_vars + 1, dtype=REAL) * init_scale
        leaf_init[:, 0] += 2.0  # bias toward the operator's terminal constant
        self.leaf_logits = nn.Parameter(leaf_init)

        # Gate logits: (n_internal, 2, n_vars+2) — softmax over [terminal, x1, ..., xn, child]
        # Bias toward the terminal (safe default that triggers leaf usage below).
        gate_init = torch.randn(self.n_internal, 2, n_vars + 2, dtype=REAL) * init_scale
        gate_init[..., 0] += 4.0
        self.gate_logits = nn.Parameter(gate_init)

    def forward(self, x, tau: float = 1.0):
        # Handle both 1D (batch,) and 2D (batch, n_vars) input.
        if x.dim() == 1:
            x = x.unsqueeze(1)  # (batch,) → (batch, 1)
        x = x.to(DTYPE)
        batch = x.shape[0]
        term_val = self.op_config.terminal_numeric
        term_col = torch.full((batch, 1), term_val, dtype=DTYPE)

        # Leaf values: soft mixture of {terminal, x₁, ..., xₙ}. When the
        # terminal is a large finite stand-in for -inf, zero-weighted
        # routes must be clamped to avoid 0 * (-1e30) = -0 NaN cascades
        # after the first op.
        w = torch.softmax(self.leaf_logits / tau, dim=1).to(DTYPE)  # (n_leaves, n_vars+1)
        candidates = torch.cat([term_col, x], dim=1)  # (batch, n_vars+1)
        if self.op_config.is_neg_inf_terminal:
            # Mask near-zero terminal weights so the product collapses
            # cleanly to 0 instead of generating junk.
            term_w = w[:, 0]
            mask_t = term_w.abs() > _TERM_EPS
            w_safe = w.clone()
            w_safe[:, 0] = torch.where(mask_t, term_w,
                                        torch.zeros_like(term_w))
            level = candidates @ w_safe.T
        else:
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

                # Start with constant term (the operator's terminal).
                p_const = ps[:, 0].unsqueeze(0)  # (1, n_pairs)
                if self.op_config.is_neg_inf_terminal:
                    # Mask zero-weight -inf contributions to avoid
                    # 0 * (-1e30) poisoning intermediate values.
                    mask_t = p_const.abs() > _TERM_EPS
                    p_term = torch.where(mask_t, p_const,
                                          torch.zeros_like(p_const))
                else:
                    p_term = p_const
                term_r = float(self.op_config.terminal_numeric.real)
                term_i = float(self.op_config.terminal_numeric.imag)
                blend_r = p_term * term_r + p_child_safe * safe_child_r
                blend_i = p_term * term_i + p_child_safe * safe_child_i

                # Add variable contributions
                for v in range(self.n_vars):
                    p_v = ps[:, v + 1].unsqueeze(0)  # (1, n_pairs)
                    blend_r = blend_r + p_v * x_r_all[:, v:v+1]  # (batch, n_pairs)
                    blend_i = blend_i + p_v * x_i_all[:, v:v+1]

                # Clean bypass when a single choice has all the mass.
                b_to_1 = p_const > _BYPASS
                blend_r = torch.where(b_to_1,
                                       torch.full_like(blend_r, term_r),
                                       blend_r)
                blend_i = torch.where(b_to_1,
                                       torch.full_like(blend_i, term_i),
                                       blend_i)
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

            level = self.op_config.op(left_in, right_in)

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
        """Variable names for expression printing. Index 0 is the terminal."""
        term = self.op_config.terminal_label
        if self.n_vars == 1:
            return [term, "x"]
        return [term] + [f"x{i+1}" for i in range(self.n_vars)]

    def to_expr(self) -> str:
        """Extract symbolic expression from snapped tree."""
        leaf_choices = torch.argmax(self.leaf_logits, dim=1).tolist()
        gate_choices = torch.argmax(self.gate_logits, dim=-1).tolist()  # (n_internal, 2)

        labels = self._var_labels()
        exprs = [labels[c] for c in leaf_choices]
        op_name = self.op_config.name
        term = self.op_config.terminal_label

        node_idx = 0
        while len(exprs) > 1:
            new_exprs = []
            for i in range(0, len(exprs), 2):
                lc, rc = gate_choices[node_idx]
                left = _resolve_gate(lc, exprs[i], self.n_vars, term)
                right = _resolve_gate(rc, exprs[i + 1], self.n_vars, term)
                new_exprs.append(f"{op_name}({left}, {right})")
                node_idx += 1
            exprs = new_exprs

        if self.op_config.simplifier_enabled:
            return _simplify(exprs[0])
        return exprs[0]

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


def _resolve_gate(choice: int, child_expr: str, n_vars: int = 1,
                  term_label: str = "1") -> str:
    """Map a gate choice index to the input expression.

    Indices: 0 → terminal, 1..n_vars → variable names, n_vars+1 → child.
    """
    if choice == 0:
        return term_label
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
    n_vars: Optional[int] = None,
    op_config: OperatorConfig = EML,
) -> dict:
    """Train one EML tree from a single random seed.

    ``x_data`` is either 1D ``(batch,)`` or 2D ``(batch, n_vars)``. The
    ``EMLTree1D`` is constructed with the matching ``n_vars`` so leaf/gate
    logits have the right width. If ``n_vars`` is passed explicitly it
    overrides shape inference (useful when callers have already committed
    to a dimensionality).

    If ``init_tree`` is provided, it is used instead of creating a fresh
    ``EMLTree1D(depth, n_vars=...)`` from the random seed. The seed still
    sets the torch RNG for reproducibility of the training loop itself.
    """
    if n_vars is None:
        n_vars = x_data.shape[1] if x_data.dim() == 2 else 1
    torch.manual_seed(seed)
    tree = init_tree if init_tree is not None else EMLTree1D(
        depth, n_vars=n_vars, op_config=op_config
    )
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
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int = 4,
    n_tries: int = 16,
    verbose: bool = True,
    success_threshold: float = 1e-10,
    n_workers: int = 1,
    op_config: OperatorConfig = EML,
) -> Optional[dict]:
    """Discover a formula relating X → y.

    Tries depths 2 through max_depth, multiple seeds per depth.
    Returns the simplest (shallowest) formula that fits within threshold.

    Args:
        X: input values. Either 1D ``(n_samples,)`` for the univariate
            case or 2D ``(n_samples, n_vars)`` for multivariate. 1D input
            is auto-promoted to ``(n_samples, 1)`` for backward compat.
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
        dict with keys: expr, depth, snap_rmse, snapped_tree, n_uncertain,
        n_vars — or None if no formula found. ``n_vars`` reports the input
        dimensionality the tree was trained on.
    """
    # 2D-input shim: accept (n,) or (n, n_vars). The tree forward handles
    # both, but `EMLTree1D` must be constructed with matching n_vars so
    # leaf/gate logits have the right width. (Issue #16.)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim != 2:
        raise ValueError(f"X must be 1D or 2D, got shape {X.shape}")
    n_vars = X.shape[1]

    # Prepare data
    x_t = torch.tensor(X, dtype=REAL)
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
            n_vars=n_vars,
            op_config=op_config,
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
                "n_vars": n_vars,
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
        "n_vars": n_vars,
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

    Supports n_vars input variables (default 1 for backward compat).

    Leaves: soft choice over {1, x₁, ..., xₙ}  (n_vars+1 logits per leaf)
    Gates:  each child input soft-routes over {1, x₁, ..., xₙ, child}
            (n_vars+2 logits per side, 2 sides per node)

    Nodes are stored in a flat list; each entry is a dict with 'type'
    ('leaf' or 'internal'), 'key' (into self._params), and for internal
    nodes 'left'/'right' (indices into self.nodes). When a leaf is split,
    its params are orphaned (unreachable from root) but not deleted, so
    optimizer state stays consistent across grow steps.
    """

    def __init__(self, init_scale: float = 1.0, n_vars: int = 1,
                 op_config: OperatorConfig = EML):
        super().__init__()
        self._params = nn.ParameterDict()
        self.nodes: list = []
        self.init_scale = init_scale
        self.n_vars = n_vars
        self.op_config = op_config
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
        # (n_vars+1,) logits: [weight_for_1, weight_for_x1, ..., weight_for_xn]
        init = torch.randn(self.n_vars + 1, dtype=REAL) * self.init_scale
        init[0] += 2.0  # bias toward constant 1
        self._params[key] = nn.Parameter(init)
        idx = len(self.nodes)
        self.nodes.append({"type": "leaf", "key": key})
        return idx

    def _new_internal(self, left: int, right: int) -> int:
        key = self._fresh_key("gate")
        # (2, n_vars+2) logits per gate: each side softmaxes over
        # [1, x1, ..., xn, child].
        init = torch.randn(2, self.n_vars + 2, dtype=REAL) * self.init_scale
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
        # Accept (batch,) for n_vars=1 back-compat or (batch, n_vars).
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if x.shape[1] != self.n_vars:
            raise ValueError(
                f"x has {x.shape[1]} columns but tree is n_vars={self.n_vars}"
            )
        x_c = x.to(DTYPE)                      # (batch, n_vars), complex
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
        batch = x_c.shape[0]
        term_val = self.op_config.terminal_numeric
        term_r = float(term_val.real)
        term_i = float(term_val.imag)
        if n["type"] == "leaf":
            logits = self._params[n["key"]]                     # (n_vars+1,)
            w = torch.softmax(logits / tau, dim=0).to(DTYPE)    # (n_vars+1,)
            ones = torch.ones(batch, dtype=DTYPE)
            if self.op_config.is_neg_inf_terminal:
                # Mask zero-weight -inf contributions.
                wt = w[0]
                wt = torch.where(wt.abs() > _TERM_EPS, wt,
                                  torch.zeros_like(wt))
                val = wt * term_val * ones
            else:
                val = w[0] * term_val * ones
            for v in range(self.n_vars):
                val = val + w[v + 1] * x_c[:, v]
        else:
            gate = self._params[n["key"]]                       # (2, n_vars+2)
            p = torch.softmax(gate / tau, dim=-1)               # (2, n_vars+2), real
            child_idx = self.n_vars + 1
            left_val = self._eval(n["left"], x_c, tau, cache)
            right_val = self._eval(n["right"], x_c, tau, cache)

            is_neg_inf = self.op_config.is_neg_inf_terminal

            def _blend(side: int, child_val):
                ps = p[side]                                    # (n_vars+2,)
                p_child = ps[child_idx]                         # scalar

                # Mask out child contribution when its probability is negligible,
                # to avoid 0*inf = nan (child values may be clamped-inf).
                zero_r = torch.zeros_like(child_val.real)
                zero_i = torch.zeros_like(child_val.imag)
                mask_c = p_child > _CHILD_EPS
                child_r = torch.where(mask_c, child_val.real, zero_r)
                child_i = torch.where(mask_c, child_val.imag, zero_i)
                p_c_s = torch.where(mask_c, p_child, torch.zeros_like(p_child))

                # Mask tiny terminal weights the same way when the
                # terminal is a large finite stand-in for -inf.
                p_term = ps[0]
                if is_neg_inf:
                    p_term = torch.where(p_term.abs() > _TERM_EPS,
                                          p_term,
                                          torch.zeros_like(p_term))

                # α·term + Σ βᵢ xᵢ + γ child — computed separately for real/imag
                # so downstream clamping can scrub NaN per-part cleanly.
                blend_r = p_term * term_r + p_c_s * child_r
                blend_i = p_term * term_i + p_c_s * child_i
                for v in range(self.n_vars):
                    p_v = ps[v + 1]
                    blend_r = blend_r + p_v * x_c[:, v].real
                    blend_i = blend_i + p_v * x_c[:, v].imag
                return torch.complex(blend_r, blend_i)

            l_in = _blend(0, left_val)
            r_in = _blend(1, right_val)
            val = self.op_config.op(l_in, r_in)
            val = torch.complex(
                torch.nan_to_num(val.real, nan=0.0, posinf=_CLAMP, neginf=-_CLAMP)
                    .clamp(-_CLAMP, _CLAMP),
                torch.nan_to_num(val.imag, nan=0.0, posinf=_CLAMP, neginf=-_CLAMP)
                    .clamp(-_CLAMP, _CLAMP),
            )
        cache[idx] = val
        return val

    # --- growing ---

    def split_leaf(self, leaf_idx: int, var_idx: int = 0) -> int:
        """Replace ``leaf_idx`` with an eml subtree approximating
        ``exp(x_{var_idx+1}) = eml(x_{var_idx+1}, 1)``.

        Args:
            leaf_idx: index of the leaf to split.
            var_idx: which input variable (0..n_vars-1) to route through
                the new subtree. For n_vars=1 this is always 0, matching
                the previous behavior. Caller is responsible for picking
                this — typically the variable with the largest gradient
                component at the splitting leaf (see
                ``leaf_gradients()``).

        Returns:
            The new internal node index. The old leaf's parameters are
            orphaned (unreachable) but remain in the parameter dict so
            that any live optimizer state referencing them stays valid.
        """
        assert self.nodes[leaf_idx]["type"] == "leaf", "can only split leaves"
        if not (0 <= var_idx < self.n_vars):
            raise ValueError(
                f"var_idx={var_idx} out of range for n_vars={self.n_vars}"
            )
        # Logit indices: 0 = "1", 1..n_vars = x1..xn, n_vars+1 = "child" (gate only)
        var_logit_idx = var_idx + 1
        child_logit_idx = self.n_vars + 1

        # New leaves: left biased toward x_{var_idx+1}, right biased toward "1".
        new_l = self._new_leaf()
        new_r = self._new_leaf()
        with torch.no_grad():
            k = 4.0
            # Left leaf → x_{var_idx+1}: strong weight on var_logit_idx, weak on rest.
            left_init = torch.full((self.n_vars + 1,), -k, dtype=REAL)
            left_init[var_logit_idx] = k
            self._params[self.nodes[new_l]["key"]].copy_(left_init)
            # Right leaf → "1": strong weight on index 0.
            right_init = torch.full((self.n_vars + 1,), -k, dtype=REAL)
            right_init[0] = k
            self._params[self.nodes[new_r]["key"]].copy_(right_init)
        # New internal gate: both sides route to "child", giving
        # eml(left_leaf, right_leaf) = eml(x_{var_idx+1}, 1) = exp(x_{var_idx+1}).
        new_int = self._new_internal(new_l, new_r)
        with torch.no_grad():
            k = 4.0
            g = torch.full((2, self.n_vars + 2), -k, dtype=REAL)
            g[:, child_logit_idx] = k
            self._params[self.nodes[new_int]["key"]].copy_(g)
        # Rewire parent (or root) to point at new_int instead of leaf_idx.
        # Also bias the parent gate side toward "child" so the new subtree
        # is visible to the parent — otherwise a gate that had hard-snapped
        # to a non-child option would leave the split inert.
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
                # Reset the side's logits to bias toward "child" (last index).
                pg[side, :] = -2.0
                pg[side, child_logit_idx] = 2.0
        return new_int

    def leaf_gradient_magnitudes(self, x_data, y_target, tau: float) -> dict:
        """Compute ||∂MSE/∂leaf_logits|| for each active leaf.

        Used as the split-selection heuristic: the leaf most "wanting" to
        change is the one with the largest gradient magnitude.

        For per-variable gradient components (used to pick *which* variable
        to route through a split subtree), use ``leaf_gradients()``.
        """
        grads = self.leaf_gradients(x_data, y_target, tau)
        return {i: float(g.norm().item()) for i, g in grads.items()}

    def leaf_gradients(self, x_data, y_target, tau: float) -> dict:
        """Compute ∂MSE/∂leaf_logits for each active leaf.

        Returns ``{leaf_idx: tensor}`` where each tensor has shape
        ``(n_vars+1,)`` and entry ``v`` (for v=1..n_vars) is the partial
        derivative of MSE with respect to the logit that routes the leaf
        toward ``x_v``. Entry 0 is the partial for the constant ``1``.

        The split heuristic in ``discover_curriculum`` picks the variable
        with the largest |partial| among entries ``1..n_vars`` to route
        through the new subtree.
        """
        self.zero_grad()
        pred, _, _ = self(x_data, tau=tau)
        mse = torch.mean((pred - y_target).abs() ** 2).real
        if not torch.isfinite(mse):
            self.zero_grad()
            return {}
        mse.backward()
        grads = {}
        for i in self.active_leaves():
            key = self.nodes[i]["key"]
            g = self._params[key].grad
            if g is not None:
                grads[i] = g.detach().clone()
            else:
                grads[i] = torch.zeros(self.n_vars + 1, dtype=REAL)
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
        raw = self._expr_at(self.root)
        if self.op_config.simplifier_enabled:
            return _simplify(raw)
        return raw

    def _expr_at(self, idx: int) -> str:
        n = self.nodes[idx]
        term = self.op_config.terminal_label
        if n["type"] == "leaf":
            c = int(torch.argmax(self._params[n["key"]]).item())
            if c == 0:
                return term
            return "x" if self.n_vars == 1 else f"x{c}"
        g = self._params[n["key"]]
        lc = int(torch.argmax(g[0]).item())
        rc = int(torch.argmax(g[1]).item())
        left_expr = self._expr_at(n["left"])
        right_expr = self._expr_at(n["right"])
        op_name = self.op_config.name
        return (f"{op_name}({_resolve_gate(lc, left_expr, self.n_vars, term)}, "
                f"{_resolve_gate(rc, right_expr, self.n_vars, term)})")


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
    op_config: OperatorConfig = EML,
) -> Optional[dict]:
    """Discover a formula via curriculum learning: grow the tree incrementally.

    Starts from a depth-1 tree. Trains to convergence. If the fit is not good
    enough, splits the active leaf whose gradient magnitude is largest — the
    split replaces it with ``eml(x_v, 1) = exp(x_v)``, where ``x_v`` is the
    variable with the largest gradient component at the splitting leaf — and
    resumes training. Repeats until ``max_depth`` is reached or an exact
    formula is found.

    This provides a warm start for depth-5 / depth-6 formulas that random
    initialization cannot easily find (the paper reports 0% recovery at
    depth 6 from random init).

    Args:
        x: input values. Either 1D ``(n_samples,)`` for the univariate
            case or 2D ``(n_samples, n_vars)`` for multivariate. 1D input
            is auto-promoted for backward compat.
        y: output values (1D numpy array, real or complex)
        max_depth: maximum tree depth to grow to (default 6)
        n_tries: independent curriculum runs with different seeds (default 8)
        lr: Adam learning rate
        verbose: print progress
        success_threshold: MSE threshold for "exact" recovery

    Returns:
        dict with keys: expr, depth, snap_rmse, snapped_tree, n_splits,
        n_vars, exact
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim != 2:
        raise ValueError(f"x must be 1D or 2D, got shape {x.shape}")
    n_vars = x.shape[1]

    x_t = torch.tensor(x, dtype=REAL)
    y_t = torch.tensor(y, dtype=DTYPE)

    best_overall = None

    # Depth-0 pre-check: the curriculum's growing tree starts at depth 1,
    # but the atoms ``y = 1`` and ``y = x_v`` are reachable one depth shallower
    # (a single-leaf tree, no gates). Try those n_vars+1 snaps explicitly
    # before kicking off the (more expensive) growing loop. See issue #11.
    with torch.no_grad():
        n_atoms = n_vars + 1
        for leaf_choice in range(n_atoms):
            label = op_config.terminal_label if leaf_choice == 0 else (
                "x" if n_vars == 1 else f"x{leaf_choice}"
            )
            probe = EMLTree1D(depth=0, n_vars=n_vars, op_config=op_config)
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
                    "n_vars": n_vars,
                    "exact": True,
                }

    for seed in range(n_tries):
        torch.manual_seed(seed)
        tree = GrowingEMLTree(n_vars=n_vars, op_config=op_config)
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
            # whose depth is still below max_depth. Then pick *which variable*
            # to route through the new subtree: the one with the largest
            # gradient component at that leaf (issue #18 option a).
            grads = tree.leaf_gradients(x_t, y_t, tau=1.0)
            if not grads:
                break
            norms = {i: float(g.norm().item()) for i, g in grads.items()}
            splittable = {
                i: v for i, v in norms.items()
                if tree.depth_of_node(i) + 1 <= max_depth
            }
            if not splittable:
                break
            leaf_to_split = max(splittable, key=splittable.get)
            # Variable selection: argmax |grad| among the n_vars variable
            # logits (indices 1..n_vars of the leaf's gradient vector).
            var_grads = grads[leaf_to_split][1:].abs()
            var_idx = int(torch.argmax(var_grads).item()) if n_vars > 1 else 0
            tree.split_leaf(leaf_to_split, var_idx=var_idx)

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
                print(f"  split #{grow_step} (var=x{var_idx+1}) → "
                      f"depth {d}: snap_rmse={math.sqrt(max(snap_mse, 0)):.3e} "
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
                "n_vars": n_vars,
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
        "n_vars": n_vars,
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

    Supports both univariate (1D) and multivariate (2D) inputs for x.
    The y target is always treated as scalar — regression problems here
    are single-output.

    For 1D input, ``x_a`` and ``x_b`` are Python floats (byte-for-byte
    backward compatible with the pre-#17 scalar surface). For 2D input
    of shape ``(n_samples, n_vars)``, they are 1D numpy arrays of length
    ``n_vars`` — one affine pair per column. ``transform_x`` / ``inverse_x``
    rely on NumPy broadcasting to handle both cases through the same code.

    Stored as a plain dict-friendly object so it can be pickled into pool
    workers and serialized alongside results. See ``to_dict()`` for the
    JSON-safe form (arrays become lists).
    """

    def __init__(self, x_a, x_b, y_a: float, y_b: float, mode: str):
        # x' = x_a * x + x_b   ;   y' = y_a * y + y_b
        #
        # x_a/x_b may be scalar (Python number, 0-d array) or 1D array
        # (per-column affines for 2D inputs). y is always scalar.
        x_a_arr = np.asarray(x_a, dtype=np.float64)
        x_b_arr = np.asarray(x_b, dtype=np.float64)
        if x_a_arr.ndim > 1 or x_b_arr.ndim > 1:
            raise ValueError(
                f"x_a and x_b must be scalar or 1D; got shapes "
                f"{x_a_arr.shape} and {x_b_arr.shape}"
            )
        if x_a_arr.shape != x_b_arr.shape:
            raise ValueError(
                f"x_a and x_b must have matching shapes; got "
                f"{x_a_arr.shape} vs {x_b_arr.shape}"
            )

        if x_a_arr.ndim == 0:
            # Scalar path: preserve the Python-float surface.
            self.x_a = float(x_a_arr)
            self.x_b = float(x_b_arr)
        else:
            self.x_a = x_a_arr
            self.x_b = x_b_arr

        self.y_a = float(y_a)
        self.y_b = float(y_b)
        self.mode = mode

    @property
    def n_vars(self) -> int:
        """Number of input variables (1 for scalar/1D inputs)."""
        if isinstance(self.x_a, float):
            return 1
        return int(self.x_a.shape[0])

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, mode: str = "minmax",
            target_lo: float = -1.0, target_hi: float = 1.0) -> "Normalizer":
        """Fit a normalizer to ``X`` and ``y``.

        ``X`` may be 1D ``(n_samples,)`` — the legacy univariate surface —
        or 2D ``(n_samples, n_vars)`` for multivariate inputs. ``y`` is
        flattened to 1D (so column-vector targets like ``(n, 1)`` are
        also accepted).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim > 2:
            raise ValueError(
                f"X must be 1D or 2D; got shape {X.shape} (ndim={X.ndim})"
            )

        # y affine — always scalar. Shared helper.
        def _ab_scalar(v: np.ndarray, mode_: str):
            if mode_ == "none":
                return 1.0, 0.0
            if mode_ == "minmax":
                lo, hi = float(v.min()), float(v.max())
                if hi == lo:
                    return 0.0, 0.0
                a = (target_hi - target_lo) / (hi - lo)
                b = target_lo - a * lo
                return a, b
            if mode_ == "standard":
                mu = float(v.mean())
                sd = float(v.std())
                if sd < 1e-12:
                    return 0.0, 0.0
                return 1.0 / sd, -mu / sd
            raise ValueError(f"unknown normalization mode: {mode_!r}")

        # x affine — vectorized when X is 2D. Constant columns collapse
        # to (0, 0), matching the scalar semantics element-wise.
        def _ab_vec(V: np.ndarray, mode_: str):
            n_vars = V.shape[1]
            if mode_ == "none":
                return np.ones(n_vars), np.zeros(n_vars)
            if mode_ == "minmax":
                lo = V.min(axis=0)
                hi = V.max(axis=0)
                rng = hi - lo
                safe_rng = np.where(rng == 0, 1.0, rng)  # guard /0
                a = np.where(rng == 0, 0.0,
                             (target_hi - target_lo) / safe_rng)
                b = np.where(rng == 0, 0.0, target_lo - a * lo)
                return a, b
            if mode_ == "standard":
                mu = V.mean(axis=0)
                sd = V.std(axis=0)
                safe_sd = np.where(sd < 1e-12, 1.0, sd)  # guard /0
                a = np.where(sd < 1e-12, 0.0, 1.0 / safe_sd)
                b = np.where(sd < 1e-12, 0.0, -mu / safe_sd)
                return a, b
            raise ValueError(f"unknown normalization mode: {mode_!r}")

        if mode not in {"none", "minmax", "standard"}:
            raise ValueError(f"unknown normalization mode: {mode!r}")

        if X.ndim == 1:
            xa, xb = _ab_scalar(X, mode)
        else:
            xa, xb = _ab_vec(X, mode)
        ya, yb = _ab_scalar(y, mode)
        return cls(xa, xb, ya, yb, mode)

    def transform_x(self, x):
        # Broadcasting handles both (batch,) and (batch, n_vars) —
        # scalar x_a/x_b multiply elementwise, 1D x_a/x_b broadcast
        # over the row axis of 2D x.
        return self.x_a * x + self.x_b

    def transform_y(self, y):
        return self.y_a * y + self.y_b

    def inverse_x(self, xp):
        if isinstance(self.x_a, float):
            # Scalar path — byte-for-byte the legacy behaviour.
            return (xp - self.x_b) / self.x_a if self.x_a != 0 \
                else np.zeros_like(xp)
        # Vector path: per-column constant columns (x_a == 0) collapse
        # to zero in the inverse, matching the scalar case element-wise.
        xp = np.asarray(xp)
        a = self.x_a            # (n_vars,)
        b = self.x_b            # (n_vars,)
        safe_a = np.where(a == 0, 1.0, a)
        result = (xp - b) / safe_a          # broadcasts over rows
        return np.where(a == 0, 0.0, result)  # zero-out dead columns

    def inverse_y(self, yp):
        return (yp - self.y_b) / self.y_a if self.y_a != 0 \
            else np.zeros_like(yp)

    def describe(self) -> str:
        if isinstance(self.x_a, float):
            return (f"x' = {self.x_a:.6g} * x + {self.x_b:.6g}   "
                    f"y' = {self.y_a:.6g} * y + {self.y_b:.6g}   "
                    f"(mode={self.mode})")
        # Per-column format for 2D: x1' = ..., x2' = ..., then y'.
        parts = [f"x{i+1}' = {float(a):.6g} * x{i+1} + {float(b):.6g}"
                 for i, (a, b) in enumerate(zip(self.x_a, self.x_b))]
        parts.append(f"y' = {self.y_a:.6g} * y + {self.y_b:.6g}")
        return "   ".join(parts) + f"   (mode={self.mode})"

    def to_dict(self) -> dict:
        # JSON-serializable: arrays → lists.
        xa = self.x_a.tolist() if isinstance(self.x_a, np.ndarray) \
            else self.x_a
        xb = self.x_b.tolist() if isinstance(self.x_b, np.ndarray) \
            else self.x_b
        return {"x_a": xa, "x_b": xb,
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
