"""Chain-skeleton enumeration + masked-DCA refinement (issue #62 lite).

#58/#60 measured that continuous search (Adam or DCA) finds good
approximants whose basins are far from any lattice point, while the
DC repair machinery recovers exact structure when started near one.
The missing piece is a structure proposer. This benchmark supplies the
smallest useful one: enumerate *chain* topologies — the family the
hand-built lattice-exact x0*x1 construction lives in — and let masked
block-cyclic DCA refine each skeleton's few free coefficients with
everything off-chain frozen.

A chain skeleton of depth D is D eml nodes N_D (root) .. N_1, where
node N_k feeds exactly one slot (u = exp side / v = ln side) of
N_{k+1}; the other slot reads a terminal: a free constant, or a free
affine a + b*x_i of one variable. The deepest node has two terminal
slots. Per skeleton the free coefficients are the chain-link gate
entries (alpha, gamma) and the terminal entries (alpha, beta_i) — a
dozen-odd reals; every other coefficient in the full EMLTree1DLinear
is frozen at template values (0, or 1 in unused v-slot constants so
they sit at ln(1) = 0 instead of the domain-barrier floor).

Skeleton count: (2*(n_vars+1))^(D-1) * (n_vars+1)^2 — 1944 for the
x0*x1 case (D=4, n=2), 256 for x0**2 (D=4, n=1).

Protocol: screen every skeleton with a short masked sweep over a few
sign-randomized seeds; fully refine the top survivors and dc_snap
them; structural check on the same held-out extrapolation band as
benchmarks/dca_recovery.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eml_dca import _dca_sweep, _gate_level_slices, dc_snap
from eml_sr import REAL
from eml_sr_linear import EMLTree1DLinear

TRAIN_N = 256
TRAIN_DOMAIN = (0.5, 2.5)
HELD_N = 512
HELD_DOMAIN = (0.25, 4.0)
STRUCT_TOL = 1e-6


# ─── Skeleton space ─────────────────────────────────────────────────

def terminal_kinds(n_vars: int) -> list:
    """'const', or 'var:i' (free affine of one variable)."""
    return ["const"] + [f"var:{i}" for i in range(n_vars)]


def chain_skeletons(depth: int, n_vars: int):
    """Yield skeleton specs.

    A spec is (links, deepest) where links[t] = (child_side,
    other_kind) for the node at top-down level t (t = 0 is the root,
    len == depth-1), child_side in {'u','v'}, and deepest =
    (kind_u, kind_v) for the terminal-only deepest node.
    """
    kinds = terminal_kinds(n_vars)
    link_choices = [(side, k) for side in ("u", "v") for k in kinds]
    for links in itertools.product(link_choices, repeat=depth - 1):
        for deepest in itertools.product(kinds, repeat=2):
            yield (links, deepest)


def _slot_indices(depth: int):
    """gate index of node at top-down level t, position pos.

    `_gate_level_slices` orders levels bottom-up; top-down level t
    corresponds to slice depth-1-t.
    """
    slices = _gate_level_slices(depth)

    def gate_index(t: int, pos: int) -> int:
        start, count = slices[depth - 1 - t]
        assert 0 <= pos < count
        return start + pos

    return gate_index


def build_template(spec, depth: int, n_vars: int):
    """Materialize a skeleton: (tree, leaf_frozen, gate_frozen, free_slots).

    free_slots lists (gate_row, side, col) for reporting. The chain
    zigzags: the child side chosen at level t decides whether the chain
    continues into the left (u) or right (v) physical child.
    """
    links, deepest = spec
    tree = EMLTree1DLinear(depth, n_vars=n_vars)
    with torch.no_grad():
        tree.leaf_logits.zero_()
        tree.gate_logits.zero_()
        # Unused v-slots read the constant 1 (ln 1 = 0): no output
        # contribution, no domain-barrier floor.
        tree.gate_logits[:, 1, 0] = 1.0

    leaf_frozen = torch.ones_like(tree.leaf_logits, dtype=torch.bool)
    gate_frozen = torch.ones_like(tree.gate_logits, dtype=torch.bool)
    gate_index = _slot_indices(depth)
    free_slots = []
    gamma_col = n_vars + 1

    def open_terminal(row: int, side: int, kind: str):
        with torch.no_grad():
            if side == 1:
                tree.gate_logits[row, 1, 0] = 0.0  # clear the ln(1) filler
        gate_frozen[row, side, 0] = False  # alpha free
        free_slots.append((row, side, 0))
        if kind.startswith("var:"):
            v = int(kind.split(":")[1])
            gate_frozen[row, side, v + 1] = False  # beta_v free
            free_slots.append((row, side, v + 1))

    pos = 0
    for t, (child_side, other_kind) in enumerate(links):
        row = gate_index(t, pos)
        child_side_idx = 0 if child_side == "u" else 1
        other_side_idx = 1 - child_side_idx
        # chain link: alpha + gamma free on the child side
        with torch.no_grad():
            if child_side_idx == 1:
                tree.gate_logits[row, 1, 0] = 0.0
        gate_frozen[row, child_side_idx, 0] = False
        gate_frozen[row, child_side_idx, gamma_col] = False
        free_slots.append((row, child_side_idx, 0))
        free_slots.append((row, child_side_idx, gamma_col))
        open_terminal(row, other_side_idx, other_kind)
        # descend: u-side reads the LEFT child (2*pos), v-side the RIGHT
        pos = 2 * pos + child_side_idx

    row = gate_index(depth - 1, pos)
    open_terminal(row, 0, deepest[0])
    open_terminal(row, 1, deepest[1])

    return tree, leaf_frozen, gate_frozen, free_slots


# ─── Refinement ─────────────────────────────────────────────────────

def refine(tree, leaf_frozen, gate_frozen, x, y, *, seed: int,
           outer_iters: int, inner_iters: int) -> float:
    """Init the free slots (sign-randomized gammas), run a masked
    sweep in place, return the exact objective."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        free = ~gate_frozen
        noise = torch.randn(tree.gate_logits.shape, generator=g,
                            dtype=REAL) * 0.3
        tree.gate_logits[free] = noise[free]
        # gammas start at +-1: crossing 0 stalls the chain, so the
        # screen enumerates the sign by seed instead
        gamma_free = free[:, :, -1]
        signs = torch.where(
            torch.rand(gamma_free.shape, generator=g) < 0.5, -1.0, 1.0
        ).to(REAL)
        tree.gate_logits[:, :, -1][gamma_free] = signs[gamma_free]
    res = _dca_sweep(tree, x, y, outer_iters=outer_iters,
                     inner_iters=inner_iters, inner_lr=0.03,
                     delta0=0.5, leaf_frozen=leaf_frozen,
                     gate_frozen=gate_frozen, stall_limit=2)
    return res["best_obj"]


def structural_check(tree, X_held, y_held) -> tuple:
    with torch.no_grad():
        pred, _, _ = tree(torch.tensor(X_held, dtype=REAL))
        pred = pred.real.numpy() if np.iscomplexobj(pred.numpy()) else pred.numpy()
    if not np.all(np.isfinite(pred)):
        return False, float("inf")
    scale = max(1.0, float(np.abs(y_held).max()))
    err = float(np.abs(pred - y_held).max())
    return err < STRUCT_TOL * scale, err


# ─── Driver ─────────────────────────────────────────────────────────

TARGETS = {
    "native_multiply": dict(n_vars=2, fn=lambda X: X[:, 0] * X[:, 1],
                            expr="x0*x1"),
    "square_via_dupmul": dict(n_vars=1, fn=lambda X: X[:, 0] ** 2,
                              expr="x0**2"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--screen-seeds", type=int, default=2)
    ap.add_argument("--screen-outers", type=int, default=3)
    ap.add_argument("--screen-inners", type=int, default=15)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--full-seeds", type=int, default=3)
    ap.add_argument("--full-outers", type=int, default=12)
    ap.add_argument("--full-inners", type=int, default=40)
    ap.add_argument("--shard", default="0/1",
                    help="i/n: process skeletons with index %% n == i")
    ap.add_argument("--out-dir", default="benchmarks/results/skeleton_enum")
    args = ap.parse_args()

    cfg = TARGETS[args.target]
    n_vars = cfg["n_vars"]
    rng = np.random.default_rng(abs(hash(args.target)) % 2**32)
    X = rng.uniform(*TRAIN_DOMAIN, size=(TRAIN_N, n_vars))
    y = cfg["fn"](X)
    X_held = rng.uniform(*HELD_DOMAIN, size=(HELD_N, n_vars))
    y_held = cfg["fn"](X_held)
    x_t = torch.tensor(X, dtype=REAL)
    y_t = torch.tensor(y, dtype=REAL)

    shard_i, shard_n = (int(v) for v in args.shard.split("/"))
    skeletons = [s for idx, s in
                 enumerate(chain_skeletons(args.depth, n_vars))
                 if idx % shard_n == shard_i]
    print(f"# {args.target}: {len(skeletons)} skeletons in shard "
          f"{args.shard} (depth {args.depth}, n_vars {n_vars})",
          flush=True)

    t0 = time.time()
    screened = []
    for si, spec in enumerate(skeletons):
        best = float("inf")
        for seed in range(args.screen_seeds):
            tree, lf, gf, _ = build_template(spec, args.depth, n_vars)
            obj = refine(tree, lf, gf, x_t, y_t, seed=seed,
                         outer_iters=args.screen_outers,
                         inner_iters=args.screen_inners)
            if math.isfinite(obj):
                best = min(best, obj)
        screened.append((best, si, spec))
        if si % 50 == 49:
            done = sorted(s[0] for s in screened)[:3]
            print(f"  screened {si + 1}/{len(skeletons)} "
                  f"({time.time() - t0:.0f}s) best3={done}", flush=True)

    screened.sort(key=lambda r: r[0])
    survivors = screened[: args.top_k]
    print(f"# screening done in {time.time() - t0:.0f}s; "
          f"refining top {len(survivors)}", flush=True)

    results = []
    for rank, (screen_obj, si, spec) in enumerate(survivors):
        best = None
        for seed in range(args.full_seeds):
            tree, lf, gf, _ = build_template(spec, args.depth, n_vars)
            obj = refine(tree, lf, gf, x_t, y_t, seed=seed,
                         outer_iters=args.full_outers,
                         inner_iters=args.full_inners)
            if not math.isfinite(obj):
                continue
            snap = dc_snap(tree, x_t, y_t, repair_outers=3)
            rec, err = structural_check(snap["tree"], X_held, y_held)
            row = dict(obj=obj, snap_mse=snap["snap_mse"],
                       stuck=snap["n_stuck"], recovered=bool(rec),
                       held_err=err, seed=seed,
                       expr=snap["tree"].to_expr()[:160])
            if best is None or (row["recovered"], -row["snap_mse"]) > (
                    best["recovered"], -best["snap_mse"]):
                best = row
            if rec:
                break
        results.append(dict(rank=rank, screen_obj=screen_obj,
                            spec=repr(spec), best=best))
        tag = ""
        if best and best["recovered"]:
            tag = f"  *** STRUCTURAL RECOVERY: {best['expr']}"
        print(f"  refine {rank}: screen={screen_obj:.3e} "
              f"snap={best['snap_mse'] if best else None}{tag}",
              flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.target}_shard{shard_i}.json"
    json.dump(dict(target=args.target, depth=args.depth,
                   shard=args.shard, n_skeletons=len(skeletons),
                   wall_s=time.time() - t0, survivors=results),
              out.open("w"), indent=1)
    n_rec = sum(1 for r in results if r["best"] and r["best"]["recovered"])
    print(f"# DONE {args.target} shard {args.shard}: "
          f"{n_rec} structural recoveries -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
