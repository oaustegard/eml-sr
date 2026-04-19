"""Trainable ternary tree — analogue of :class:`eml_sr.EMLTree1D`.

Mirrors the EML tree's soft-routing architecture with three children per
internal node instead of two. Two grammars are supported:

* ``allow_terminal=False`` (default, "pure" form): ``S → x | T(S, S, S)``.
  Leaves route over ``{x₁, ..., xₙ, child}`` only — no distinguished
  constants. This is the grammar §5 of the paper calls an "open
  question". Because constructing ``1`` requires a full ``T(x, x, x)``
  subtree, a fixed shallow tree cannot express ``1`` at every leaf
  unless it has room to spawn that subtree.
* ``allow_terminal=True`` ("relaxed practical" form):
  ``S → c | x | T(S, S, S)``. Leaves also route to a learnable complex
  constant ``c``, one per leaf. This is the gradient-friendly analogue
  — matches the relaxation noted in issue #37's design section.

This module is intentionally minimal — enough to test gradient-based
recovery on a handful of small targets. It deliberately does not
reimplement :class:`eml_sr.EMLTree1D`'s curriculum growing, multivariate
support, or snap-simplifier; adding those in parallel would double the
engine surface area. The verdict in ``report.md`` is about feasibility,
not production-readiness.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .operator import t_clamped


DTYPE = torch.complex128
REAL = torch.float64
_CLAMP = 1e300


class TernaryTree1D(nn.Module):
    """Fixed-depth ternary tree, one input variable, soft-routed.

    Layout
    ------
    A full ternary tree of depth ``D`` has ``3**D`` leaves and
    ``(3**D - 1) // 2`` internal nodes.

    Leaves soft-mix over ``{x}`` (pure) or ``{c, x}`` (with learnable
    constants). Internal nodes have three child inputs; each child's
    gate soft-routes over ``{x, child}`` (pure) or ``{c, x, child}``
    (with terminal). ``c`` is a per-leaf / per-gate learnable complex
    constant when ``allow_terminal=True``.

    The forward computes bottom-up, grouping siblings into triples.
    """

    def __init__(self, depth: int, allow_terminal: bool = False,
                 init_scale: float = 1.0):
        super().__init__()
        assert depth >= 1, "ternary tree must have at least depth 1"
        self.depth = depth
        self.allow_terminal = allow_terminal
        self.n_leaves = 3 ** depth
        self.n_internal = (3 ** depth - 1) // 2

        n_leaf_opts = 2 if allow_terminal else 1   # {c?, x}
        n_gate_opts = 3 if allow_terminal else 2   # {c?, x, child}

        leaf_init = torch.randn(self.n_leaves, n_leaf_opts, dtype=REAL) * init_scale
        self.leaf_logits = nn.Parameter(leaf_init)
        if allow_terminal:
            # One complex terminal per leaf and per gate-side; initialised
            # around 1 so zero-weight paths don't blow up ``ln(c)``.
            c_leaf = torch.zeros(self.n_leaves, 2, dtype=REAL)  # (real, imag)
            c_leaf[:, 0] = 1.0
            self.leaf_c = nn.Parameter(c_leaf)

        gate_init = torch.randn(self.n_internal, 3, n_gate_opts, dtype=REAL) * init_scale
        self.gate_logits = nn.Parameter(gate_init)
        if allow_terminal:
            c_gate = torch.zeros(self.n_internal, 3, 2, dtype=REAL)
            c_gate[..., 0] = 1.0
            self.gate_c = nn.Parameter(c_gate)

    # --------------------------------------------------------------

    def _leaf_values(self, x_c: torch.Tensor, tau: float) -> torch.Tensor:
        """Return (batch, n_leaves) leaf-level complex values."""
        w = torch.softmax(self.leaf_logits / tau, dim=1).to(DTYPE)
        batch = x_c.shape[0]
        if self.allow_terminal:
            # candidates: (batch, n_leaves, 2) — [c_leaf, x]
            c = torch.complex(self.leaf_c[:, 0], self.leaf_c[:, 1])  # (n_leaves,)
            c_col = c.unsqueeze(0).expand(batch, -1)                  # (batch, n_leaves)
            x_col = x_c.unsqueeze(1).expand(-1, self.n_leaves)        # (batch, n_leaves)
            vals = w[:, 0].unsqueeze(0) * c_col + w[:, 1].unsqueeze(0) * x_col
        else:
            # Pure grammar: leaf is always x (only one choice).
            vals = x_c.unsqueeze(1).expand(-1, self.n_leaves)
        return vals

    def _blend(self, side_probs: torch.Tensor, side_c: Optional[torch.Tensor],
               x_c: torch.Tensor, child_val: torch.Tensor) -> torch.Tensor:
        """Combine ``{c?, x, child}`` into one complex tensor per triplet.

        ``side_probs`` is ``(n_triples, n_opts)`` (real). ``side_c`` (if
        ``allow_terminal``) is ``(n_triples, 2)`` — the per-gate terminal
        in ``(real, imag)``.
        """
        # p_child, p_x, (p_c)
        n_triples = side_probs.shape[0]
        if self.allow_terminal:
            p_c = side_probs[:, 0].unsqueeze(0)      # (1, n_triples)
            p_x = side_probs[:, 1].unsqueeze(0)
            p_ch = side_probs[:, 2].unsqueeze(0)
            c = torch.complex(side_c[:, 0], side_c[:, 1]).unsqueeze(0).expand_as(child_val)
        else:
            p_x = side_probs[:, 0].unsqueeze(0)
            p_ch = side_probs[:, 1].unsqueeze(0)
            p_c = None
            c = None

        x_ = x_c.unsqueeze(1).expand(-1, n_triples)
        blend = p_x.to(DTYPE) * x_ + p_ch.to(DTYPE) * child_val
        if p_c is not None:
            blend = blend + p_c.to(DTYPE) * c
        return blend

    def forward(self, x: torch.Tensor, tau: float = 1.0):
        """Evaluate the tree. ``x`` is a 1D real tensor (batch,)."""
        x_c = x.to(DTYPE)
        level = self._leaf_values(x_c, tau)             # (batch, n_leaves)
        gate_probs_all = torch.softmax(self.gate_logits / tau, dim=-1)

        node_idx = 0
        while level.shape[1] > 1:
            n_triples = level.shape[1] // 3
            a_col = level[:, 0::3]
            b_col = level[:, 1::3]
            c_col = level[:, 2::3]

            gp = gate_probs_all[node_idx:node_idx + n_triples]  # (n_tri, 3, opts)
            gc = (self.gate_c[node_idx:node_idx + n_triples]
                  if self.allow_terminal else None)              # (n_tri, 3, 2)

            a_in = self._blend(gp[:, 0, :], gc[:, 0, :] if gc is not None else None,
                               x_c, a_col)
            b_in = self._blend(gp[:, 1, :], gc[:, 1, :] if gc is not None else None,
                               x_c, b_col)
            c_in = self._blend(gp[:, 2, :], gc[:, 2, :] if gc is not None else None,
                               x_c, c_col)

            level = t_clamped(a_in, b_in, c_in)
            node_idx += n_triples

        leaf_probs = torch.softmax(self.leaf_logits / tau, dim=1)
        return level.squeeze(1), leaf_probs, gate_probs_all

    # --------------------------------------------------------------

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def snap_choices(self):
        """Return ``(leaf_choices, gate_choices)`` argmaxed, for reporting."""
        with torch.no_grad():
            lc = torch.argmax(self.leaf_logits, dim=1).tolist()
            gc = torch.argmax(self.gate_logits, dim=-1).tolist()
        return lc, gc


# ─── Training ────────────────────────────────────────────────────

def train_one(x_data: torch.Tensor, y_data: torch.Tensor, depth: int,
              seed: int = 0, search_iters: int = 1000, hard_iters: int = 400,
              lr: float = 0.01, allow_terminal: bool = False,
              tau_search: float = 1.0, tau_hard: float = 0.05,
              verbose: bool = False) -> dict:
    """Train one ternary tree. Returns a dict with best MSE and tree."""
    torch.manual_seed(seed)
    tree = TernaryTree1D(depth=depth, allow_terminal=allow_terminal)
    opt = torch.optim.Adam(tree.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = None
    nan_count = 0
    total = search_iters + hard_iters
    for it in range(1, total + 1):
        if nan_count > 20:
            break
        if it <= search_iters:
            tau = tau_search
        else:
            t = (it - search_iters) / max(1, hard_iters)
            tau = tau_search * (tau_hard / tau_search) ** (t ** 2)
        opt.zero_grad()
        pred, _, _ = tree(x_data, tau=tau)
        mse = torch.mean((pred - y_data).abs() ** 2).real
        if not torch.isfinite(mse):
            nan_count += 1
            if best_state is not None:
                tree.load_state_dict(best_state)
                opt = torch.optim.Adam(tree.parameters(), lr=lr)
            continue
        mse.backward()
        torch.nn.utils.clip_grad_norm_(tree.parameters(), 1.0)
        opt.step()
        val = mse.item()
        if np.isfinite(val) and val < best_loss:
            best_loss = val
            best_state = {k: v.clone() for k, v in tree.state_dict().items()}
        if verbose and it % 200 == 0:
            print(f"    it={it:5d} tau={tau:.3f} mse={val:.3e} best={best_loss:.3e}")
    if best_state is not None:
        tree.load_state_dict(best_state)
    with torch.no_grad():
        pred_final, _, _ = tree(x_data, tau=tau_hard)
        final_mse = torch.mean((pred_final - y_data).abs() ** 2).real.item()
    return {
        "tree": tree,
        "best_mse": best_loss,
        "final_mse": final_mse,
        "nan_count": nan_count,
    }
