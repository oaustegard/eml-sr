"""Exact discrete enumeration over chain skeletons (issue #62 lite, v2).

skeleton_enum.py's probe run showed that even *given* the true chain
skeleton for x0*x1, short masked-DCA refinement lands at objective
~0.9 from 30 random or lattice inits: the exact solution's basin is
tiny (consistent with the dc_snap perturbation control on #60). With a
dozen free slots and lattice-valued solutions, the honest inner solver
is exact enumeration, not continuous descent.

This benchmark scores EVERY lattice assignment of a chain skeleton's
free slots in one vectorized pass per chain level:

  value_k = exp(u_k) - ln(v_k),  u/v in {alpha + beta*x_i, alpha + gamma*child}

with candidate grids alpha in {0,1,2,-1}, gamma in {1,-1}, beta = 1.
Chains never need the full-tree forward, so a level is one broadcast
op over (n_assignments, n_samples). Assignments containing ln of a
non-positive value die naturally (NaN -> rejected). Survivors on the
16-sample screen are re-scored on 256 samples and finally checked on
the held-out extrapolation band.

Success criterion: structural discovery of x0*x1 (and x0**2) from
enumeration alone — no seeding, no knowledge of the target's form.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.skeleton_enum import (  # noqa: E402
    HELD_DOMAIN,
    HELD_N,
    TRAIN_DOMAIN,
    TRAIN_N,
    chain_skeletons,
)

ALPHAS = (0.0, 1.0, 2.0, -1.0)
GAMMAS = (1.0, -1.0)
STRUCT_TOL = 1e-6
SCREEN_N = 16
SCREEN_TOL = 1e-9  # mse on the screen batch to qualify for re-scoring


def _slot_grids(kind: str):
    """Candidate (alpha, beta) pairs for a terminal slot."""
    if kind == "const":
        return [(a, None) for a in ALPHAS]
    return [(0.0, 1.0)]  # var slot: unit slope, zero offset


def enumerate_assignments(spec):
    """Yield dicts of slot values for one skeleton."""
    links, deepest = spec
    slot_choices = []
    for (child_side, other_kind) in links:
        slot_choices.append([(a, g) for a in ALPHAS for g in GAMMAS])  # link
        slot_choices.append(_slot_grids(other_kind))                   # other
    slot_choices.append(_slot_grids(deepest[0]))
    slot_choices.append(_slot_grids(deepest[1]))
    yield from itertools.product(*slot_choices)


def eval_chain(spec, assign, X):
    """Evaluate the chain bottom-up for one assignment. X: (n, n_vars)."""
    links, deepest = spec

    def term(kind, ab):
        a, b = ab
        if kind == "const":
            return np.full(X.shape[0], a)
        v = int(kind.split(":")[1])
        return a + b * X[:, v]

    # deepest node
    u = term(deepest[0], assign[-2])
    v = term(deepest[1], assign[-1])
    with np.errstate(all="ignore"):
        child = np.exp(u) - np.log(v)
        # walk links bottom-up (links[] is top-down)
        for t in range(len(links) - 1, -1, -1):
            (child_side, other_kind) = links[t]
            a, g = assign[2 * t]
            fed = a + g * child
            other = term(other_kind, assign[2 * t + 1])
            u, v = (fed, other) if child_side == "u" else (other, fed)
            child = np.exp(u) - np.log(v)
    return child


def eval_chain_batch(spec, assigns, X):
    """Vectorized eval of many assignments: returns (n_assign, n_samples)."""
    links, deepest = spec
    n_a, n_s = len(assigns), X.shape[0]
    A = np.array([[a for pair in row for a in
                   (pair if pair[1] is not None else (pair[0], 0.0))]
                  for row in assigns])  # (n_a, 2*n_slots)

    def term_batch(kind, col):
        a = A[:, col][:, None]
        if kind == "const":
            return np.broadcast_to(a, (n_a, n_s))
        b = A[:, col + 1][:, None]
        v = int(kind.split(":")[1])
        return a + b * X[None, :, v]

    n_link_slots = 2 * len(links)
    with np.errstate(all="ignore"):
        u = term_batch(deepest[0], n_link_slots * 2)
        v = term_batch(deepest[1], n_link_slots * 2 + 2)
        child = np.exp(u) - np.log(v)
        for t in range(len(links) - 1, -1, -1):
            a = A[:, 4 * t][:, None]
            g = A[:, 4 * t + 1][:, None]
            fed = a + g * child
            other = term_batch(links[t][1], 4 * t + 2)
            if links[t][0] == "u":
                u, v = fed, other
            else:
                u, v = other, fed
            child = np.exp(u) - np.log(v)
    return child


def spec_to_expr(spec, assign) -> str:
    """Readable expression for a recovered assignment."""
    links, deepest = spec

    def term(kind, ab):
        a, b = ab
        if kind == "const":
            return f"{a:g}"
        v = kind.split(":")[1]
        pre = f"{a:g} + " if a else ""
        return f"{pre}x{v}"

    expr = (f"eml({term(deepest[0], assign[-2])}, "
            f"{term(deepest[1], assign[-1])})")
    for t in range(len(links) - 1, -1, -1):
        a, g = assign[2 * t]
        fed = f"{a:g} + {g:g}*({expr})" if a else f"{g:g}*({expr})"
        other = term(links[t][1], assign[2 * t + 1])
        u, v = (fed, other) if links[t][0] == "u" else (other, fed)
        expr = f"eml({u}, {v})"
    return expr


TARGETS = {
    "native_multiply": dict(n_vars=2, fn=lambda X: X[:, 0] * X[:, 1]),
    "square_via_dupmul": dict(n_vars=1, fn=lambda X: X[:, 0] ** 2),
    "sum_of_squares": dict(n_vars=4, fn=lambda X: X[:, 0] ** 2 + X[:, 3] ** 2),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--out-dir", default="benchmarks/results/skeleton_enum")
    args = ap.parse_args()

    cfg = TARGETS[args.target]
    n_vars = cfg["n_vars"]
    rng = np.random.default_rng(abs(hash(args.target)) % 2**32)
    X = rng.uniform(*TRAIN_DOMAIN, size=(TRAIN_N, n_vars))
    y = cfg["fn"](X)
    X_scr, y_scr = X[:SCREEN_N], y[:SCREEN_N]
    X_held = rng.uniform(*HELD_DOMAIN, size=(HELD_N, n_vars))
    y_held = cfg["fn"](X_held)

    t0 = time.time()
    n_skel = n_assign_total = 0
    found = []
    for spec in chain_skeletons(args.depth, n_vars):
        n_skel += 1
        assigns = list(enumerate_assignments(spec))
        n_assign_total += len(assigns)
        pred = eval_chain_batch(spec, assigns, X_scr)
        mse = np.nanmean((pred - y_scr[None, :]) ** 2, axis=1)
        mse = np.where(np.isfinite(mse), mse, np.inf)
        hits = np.nonzero(mse < SCREEN_TOL)[0]
        for h in hits:
            full = eval_chain(spec, assigns[h], X)
            if not np.all(np.isfinite(full)):
                continue
            if np.mean((full - y) ** 2) > SCREEN_TOL:
                continue
            held = eval_chain(spec, assigns[h], X_held)
            scale = max(1.0, float(np.abs(y_held).max()))
            err = float(np.abs(held - y_held).max())
            if np.all(np.isfinite(held)) and err < STRUCT_TOL * scale:
                found.append(dict(spec=repr(spec),
                                  expr=spec_to_expr(spec, assigns[h]),
                                  held_max_err=err))
        if n_skel % 400 == 0:
            print(f"  {n_skel} skeletons, {n_assign_total} assignments, "
                  f"{len(found)} exact, {time.time() - t0:.0f}s",
                  flush=True)

    wall = time.time() - t0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.target}_exact.json"
    json.dump(dict(target=args.target, depth=args.depth,
                   n_skeletons=n_skel, n_assignments=n_assign_total,
                   wall_s=wall, n_found=len(found), found=found[:50]),
              out.open("w"), indent=1)
    print(f"# DONE {args.target}: {len(found)} exact structural "
          f"discoveries from {n_assign_total} assignments over {n_skel} "
          f"skeletons in {wall:.0f}s -> {out}", flush=True)
    for f in found[:5]:
        print("  ", f["expr"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
