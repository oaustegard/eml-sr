"""Branched-topology enumeration via meet-in-the-middle (issue #62, PR #64
follow-up).

PR #64's chain family (skeleton_exact.py / skeleton_enum.py) enumerates
*single* nested chains of eml nodes and structurally discovers
multiplicative targets (x0*x1, x0**2) — but scored 0/134,217,728 on
sum_of_squares (x0**2 + x3**2): a sum of two independent nonlinear terms
needs a BRANCHED topology, not one chain.

Family here: one root eml node

    y = exp(u_in) - ln(v_in),   u_in = a_u + g_u*U,   v_in = a_v + g_v*V

where U and V are each *independently* either a bare terminal (a
constant in {0,1,2,-1}, or a single variable x_i) or the raw output of
a depth <= max_side_depth chain skeleton from skeleton_exact.py's
family, and (a_u, g_u), (a_v, g_v) range over the same lattice
(ALPHAS x GAMMAS) skeleton_exact.py uses for its own chain links.

Method: meet-in-the-middle. Stage 1 builds one cache of every side's
raw value (deduped by rounded value, evaluated on a 16-sample screen);
stage 2 treats every cache entry as a candidate U, algebraically solves
for the *required* V given the target, and looks it up in the same
cache. A hit is confirmed exactly on the full train set, then checked
on the held-out extrapolation band exactly as skeleton_exact.py does.

Known depth-cap limitation (analytically, not just empirically): getting
x_i**2 to appear via the exp side needs the U-chain to raise its own
internal (alpha, gamma) chain-link lattice to produce 2*ln(x_i) as its
RAW output, which the interior of PR #64's x0**2 discovery shows is
reachable at chain depth 3 (see square_via_dupmul's discovered expr:
its 2 - 2*ln(x0) sub-term is depth 3, then the *root's own* (a_u=2,
g_u=-1) join turns it into 2*ln(x0)). But getting a matching x_j**2 to
appear via the ln side needs the V-chain's raw output to already equal
exp(-x_j**2) itself (v_in is used bare, never re-exponentiated at the
root) — that in turn needs an extra node wrapping a depth-3 "2*ln(x_j)"
sub-chain in one more exp, i.e. depth 4 on that side. The same
depth-4-on-one-side requirement blocks native_multiply's PR #64 chain
solutions from reappearing here (their own root join is itself the 4th
link). Rather than raise --max-side-depth to route around this, the
default stays 3 and the search is run as specified; whether some OTHER
depth<=3 branched construction reaches these targets is an empirical
question the search answers directly. Run --self-test to check the
join algebra's sign conventions in isolation before trusting a
zero-discovery result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.skeleton_exact import (
    ALPHAS,
    GAMMAS,
    HELD_DOMAIN,
    HELD_N,
    STRUCT_TOL,
    TRAIN_DOMAIN,
    TRAIN_N,
    chain_skeletons,
    enumerate_assignments,
    eval_chain,
    eval_chain_batch,
    spec_to_expr,
)

SCREEN_N = 16
TRAIN_MSE_TOL = 1e-9
BARE_CONSTS = (0.0, 1.0, 2.0, -1.0)


# ─── Cache entries ──────────────────────────────────────────────────
# entry = (kind, spec, assign, val16)
#   kind == "var":   spec is None, assign is the variable index i
#   kind == "const": spec is None, assign is the constant value
#   kind == "chain": spec/assign as produced by skeleton_exact.py

def eval_entry_full(entry, X: np.ndarray) -> np.ndarray:
    """Evaluate a cache entry's raw value on arbitrary samples X."""
    kind, spec, assign, _ = entry
    if kind == "var":
        return X[:, assign].astype(np.float64)
    if kind == "const":
        return np.full(X.shape[0], assign, dtype=np.float64)
    return eval_chain(spec, assign, X)


def entry_to_expr(entry) -> str:
    kind, spec, assign, _ = entry
    if kind == "var":
        return f"x{assign}"
    if kind == "const":
        return f"{assign:g}"
    return spec_to_expr(spec, assign)


def _affine_expr(a: float, g: float, inner: str) -> str:
    if a:
        return f"{a:g} + {g:g}*({inner})"
    return f"{g:g}*({inner})"


# ─── Stage 1: side cache ────────────────────────────────────────────

def build_side_cache(n_vars: int, max_side_depth: int, X_scr: np.ndarray):
    """Every depth 0..max_side_depth side value, deduped by rounded bytes.

    Streams per skeleton: a skeleton's assignment list is materialized,
    scored, and discarded before the next skeleton starts — assignments
    across skeletons are never held at once.
    """
    cache: dict = {}
    n_skel = 0
    n_assign = 0

    with np.errstate(all="ignore"):
        for i in range(n_vars):
            val = X_scr[:, i].astype(np.float64)
            key = np.round(val, 8).tobytes()
            cache.setdefault(key, ("var", None, i, val))
            n_assign += 1
        for c in BARE_CONSTS:
            val = np.full(X_scr.shape[0], c, dtype=np.float64)
            key = np.round(val, 8).tobytes()
            cache.setdefault(key, ("const", None, c, val))
            n_assign += 1
    n_skel += n_vars + len(BARE_CONSTS)

    for depth in range(1, max_side_depth + 1):
        for spec in chain_skeletons(depth, n_vars):
            n_skel += 1
            assigns = list(enumerate_assignments(spec))
            n_assign += len(assigns)
            with np.errstate(all="ignore"):
                pred = eval_chain_batch(spec, assigns, X_scr)
            finite = np.all(np.isfinite(pred), axis=1)
            with np.errstate(all="ignore"):
                for h in np.nonzero(finite)[0]:
                    val = pred[h].astype(np.float64)
                    key = np.round(val, 8).tobytes()
                    cache.setdefault(key, ("chain", spec, assigns[h], val))
            if n_skel % 2000 == 0:
                print(f"  cache: {n_skel} side-skeletons, {n_assign} "
                      f"assignments, {len(cache)} unique values", flush=True)

    return cache, n_skel, n_assign


# ─── Stage 2: join search ───────────────────────────────────────────

def _confirm(u_entry, v_entry, a_u, g_u, a_v, g_v,
             X_train, y_train, X_held, y_held):
    """Exact confirmation on the full train set, then held-out extrapolation."""
    U_full = eval_entry_full(u_entry, X_train)
    V_full = eval_entry_full(v_entry, X_train)
    with np.errstate(all="ignore"):
        y_hat = np.exp(a_u + g_u * U_full) - np.log(a_v + g_v * V_full)
    if not np.all(np.isfinite(y_hat)):
        return None
    mse = float(np.mean((y_hat - y_train) ** 2))
    if mse >= TRAIN_MSE_TOL:
        return None

    U_held = eval_entry_full(u_entry, X_held)
    V_held = eval_entry_full(v_entry, X_held)
    with np.errstate(all="ignore"):
        y_hat_held = np.exp(a_u + g_u * U_held) - np.log(a_v + g_v * V_held)
    if not np.all(np.isfinite(y_hat_held)):
        return None
    scale = max(1.0, float(np.abs(y_held).max()))
    err = float(np.abs(y_hat_held - y_held).max())
    if err >= STRUCT_TOL * scale:
        return None

    expr = (f"eml({_affine_expr(a_u, g_u, entry_to_expr(u_entry))}, "
            f"{_affine_expr(a_v, g_v, entry_to_expr(v_entry))})")
    return {"expr": expr, "a_u": a_u, "g_u": g_u, "a_v": a_v, "g_v": g_v,
            "train_mse": mse, "held_max_err": err}


def join_search(cache: dict, y_scr: np.ndarray,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_held: np.ndarray, y_held: np.ndarray):
    """For every cached U-entry and every (a_u, g_u), algebraically solve
    for the required v_in / V and look it up in the same cache."""
    entries = list(cache.values())
    joins_tested = 0
    hits = 0
    discoveries = []
    t0 = time.time()

    for entry_idx, u_entry in enumerate(entries):
        U_scr = u_entry[3]
        for a_u in ALPHAS:
            for g_u in GAMMAS:
                with np.errstate(all="ignore"):
                    P = np.exp(a_u + g_u * U_scr)
                if not np.all(np.isfinite(P)):
                    continue
                # required ln(v_in) = P - y_screen
                with np.errstate(all="ignore"):
                    v_in_req = np.exp(P - y_scr)
                for a_v in ALPHAS:
                    for g_v in GAMMAS:
                        joins_tested += 1
                        with np.errstate(all="ignore"):
                            V_req = (v_in_req - a_v) / g_v
                        if not np.all(np.isfinite(V_req)):
                            continue
                        with np.errstate(all="ignore"):
                            key2 = np.round(V_req, 8).tobytes()
                        v_entry = cache.get(key2)
                        if v_entry is None:
                            continue
                        hits += 1
                        disc = _confirm(u_entry, v_entry, a_u, g_u, a_v, g_v,
                                        X_train, y_train, X_held, y_held)
                        if disc is not None:
                            discoveries.append(disc)
        if (entry_idx + 1) % 50000 == 0:
            print(f"  join: {entry_idx + 1}/{len(entries)} U-entries, "
                  f"{joins_tested} joins tested, {hits} hash hits, "
                  f"{len(discoveries)} discoveries, "
                  f"{time.time() - t0:.0f}s", flush=True)

    return joins_tested, hits, discoveries


# ─── Self-test ───────────────────────────────────────────────────────

def self_test() -> bool:
    """Hand-assemble x0**2 + x3**2 through the join formula's algebra and
    assert it matches to 1e-9 on train-domain samples — BEFORE trusting
    any search result. Exercises both directions:

    forward: given known U, V and (a_u,g_u,a_v,g_v), does
             exp(u_in) - ln(v_in) reproduce the target?
    inverse: given the SAME U and the target y, does stage 2's own
             "required v_in / V" derivation recover the SAME V used to
             build the target? (This is the algebra actually exercised
             during the search; a sign error here would silently zero
             out every discovery.)
    """
    rng = np.random.default_rng(12345)
    n = 64
    x0 = rng.uniform(*TRAIN_DOMAIN, n)
    x3 = rng.uniform(*TRAIN_DOMAIN, n)
    y = x0 ** 2 + x3 ** 2

    # known pieces: U = 2*ln(x0) so exp(0 + 1*U) = x0**2;
    # V = exp(-x3**2) so 0 + 1*V = exp(-x3**2), i.e. -ln(V) = x3**2.
    a_u, g_u, a_v, g_v = 0.0, 1.0, 0.0, 1.0
    U = 2.0 * np.log(x0)
    V = np.exp(-x3 ** 2)

    u_in = a_u + g_u * U
    v_in = a_v + g_v * V
    y_hat = np.exp(u_in) - np.log(v_in)
    ok_forward = bool(np.allclose(y_hat, y, atol=1e-9))

    # inverse: stage 2's own derivation, run on the *same* U/target,
    # should recover the same V.
    P = np.exp(u_in)
    v_in_req = np.exp(P - y)
    V_req = (v_in_req - a_v) / g_v
    ok_inverse = bool(np.allclose(V_req, V, atol=1e-6))

    print(f"self-test: forward assembly {'PASS' if ok_forward else 'FAIL'}, "
          f"inverse join derivation {'PASS' if ok_inverse else 'FAIL'}",
          flush=True)
    return ok_forward and ok_inverse


# ─── Driver ─────────────────────────────────────────────────────────

TARGETS = {
    "sum_of_squares": {"n_vars": 4, "fn": lambda X: X[:, 0] ** 2 + X[:, 3] ** 2},
    "native_multiply": {"n_vars": 2, "fn": lambda X: X[:, 0] * X[:, 1]},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=sorted(TARGETS))
    ap.add_argument("--max-side-depth", type=int, default=3)
    ap.add_argument("--out-dir", default="benchmarks/results/skeleton_enum")
    ap.add_argument("--self-test", action="store_true",
                    help="run the join-algebra self-test and exit")
    args = ap.parse_args()

    if args.self_test:
        return 0 if self_test() else 1

    if not args.target:
        ap.error("--target is required unless --self-test")

    if not self_test():
        print("# WARNING: self-test FAILED; search results below are "
              "suspect (join algebra sign convention is broken)",
              flush=True)

    cfg = TARGETS[args.target]
    n_vars = cfg["n_vars"]
    rng = np.random.default_rng(abs(hash(args.target)) % 2**32)
    X = rng.uniform(*TRAIN_DOMAIN, size=(TRAIN_N, n_vars))
    y = cfg["fn"](X)
    X_scr, y_scr = X[:SCREEN_N], y[:SCREEN_N]
    X_held = rng.uniform(*HELD_DOMAIN, size=(HELD_N, n_vars))
    y_held = cfg["fn"](X_held)

    t0 = time.time()
    cache, n_cache_skel, n_cache_assign = build_side_cache(
        n_vars, args.max_side_depth, X_scr)
    cache_wall = time.time() - t0
    print(f"# cache built: {len(cache)} unique values from "
          f"{n_cache_assign} assignments over {n_cache_skel} side-skeletons "
          f"in {cache_wall:.0f}s", flush=True)

    t1 = time.time()
    joins_tested, hits, discoveries = join_search(
        cache, y_scr, X, y, X_held, y_held)
    join_wall = time.time() - t1
    wall = time.time() - t0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.target}_branch.json"
    json.dump({
        "target": args.target, "max_side_depth": args.max_side_depth,
        "n_vars": n_vars, "cache_size": len(cache),
        "n_cache_skeletons": n_cache_skel, "n_cache_assignments": n_cache_assign,
        "cache_wall_s": cache_wall, "joins_tested": joins_tested, "hash_hits": hits,
        "join_wall_s": join_wall, "wall_s": wall,
        "n_discoveries": len(discoveries), "discoveries": discoveries[:50],
    }, out.open("w"), indent=1)

    print(f"# DONE {args.target}: {len(discoveries)} structural "
          f"discoveries from {joins_tested} joins ({hits} hash hits) over "
          f"a {len(cache)}-entry side cache in {wall:.0f}s -> {out}",
          flush=True)
    for d in discoveries[:5]:
        print("  ", d["expr"])
    if not discoveries:
        print("# NOTE: zero discoveries here does not indict the search — "
              "see the module docstring's depth-cap analysis for a known "
              "reason at least one hand construction is unreachable at "
              "max-side-depth=3.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
