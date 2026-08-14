"""Branched-skeleton join v2: frontier-expansion cache + quantized keys.

Two compounding lessons from the depth-4 attempts:

* Raw chain-assignment enumeration is exponentially redundant — at
  n=3, depth 4, gamma {+-1,+-2} it is ~3.4e10 assignments whose
  deduped value set is a few million. Every depth-(k+1) chain value is
  one eml node applied to a depth-<=k value, so the cache builds by
  LEVEL-WISE FRONTIER EXPANSION: extend the deduped frontier with all
  single-node steps, batch-quantize, dedupe, repeat. Cost per level
  ~ |frontier| x |steps|; deeper sides get cheaper, not exponentially
  worse.

* Exact float64 join keys have false negatives (PR #66: 32 vs 473);
  keys here are int16 cells on arcsinh(V) (KEY_RES resolution) in a
  lexsorted matrix searched by searchsorted — no dict, multi-occupancy
  by construction. A batched 16-sample pre-screen gates the exact
  float64 confirm.

Provenance is a parent-pointer DAG: row -> (parent_row, step_id),
terminals (-1, term_id). Exact values on arbitrary samples and
readable expressions are reconstructed by walking parents. The
expansion family uses bare terminals in the non-child slot (matching
every ground-truth discovery so far); affine-terminal steps are a
straightforward widening if a target demands them.
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

from benchmarks.skeleton_branch import TARGETS, TRAIN_MSE_TOL
from benchmarks.skeleton_exact import (
    ALPHAS,
    GAMMAS,
    HELD_DOMAIN,
    HELD_N,
    SCREEN_N,
    STRUCT_TOL,
    TRAIN_DOMAIN,
    TRAIN_N,
)

KEY_RES = 0.01
BARE_CONSTS = (0.0, 1.0, 2.0, -1.0)


# ─── Keys ───────────────────────────────────────────────────────────

def quantize_keys(V: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.clip(np.rint(np.arcsinh(V) / KEY_RES),
                       -32000, 32000).astype(np.int16)


def _void_view(codes: np.ndarray) -> np.ndarray:
    c = np.ascontiguousarray(codes)
    return c.view([("", c.dtype)] * c.shape[1]).ravel()


# ─── Steps (single-node chain extensions) ───────────────────────────
# step = (child_side, a, g, term_id); term_id < n_vars -> variable,
# else BARE_CONSTS[term_id - n_vars].

def make_steps(n_vars: int) -> list:
    return [(side, float(a), float(g), t)
            for side in (0, 1)
            for a in ALPHAS
            for g in GAMMAS
            for t in range(n_vars + len(BARE_CONSTS))]


def term_matrix(n_vars: int, X: np.ndarray) -> np.ndarray:
    """(n_terms, n_samples) values of each terminal on samples X."""
    rows = [X[:, i].astype(np.float64) for i in range(n_vars)]
    rows += [np.full(X.shape[0], float(c)) for c in BARE_CONSTS]
    return np.stack(rows)


# ─── Cache build ────────────────────────────────────────────────────

def build_cache(n_vars: int, max_side_depth: int, X_scr: np.ndarray):
    """Returns (values32 (N,s), prov (N,2) int32, steps)."""
    steps = make_steps(n_vars)
    terms = term_matrix(n_vars, X_scr)
    seen = set()
    blocks, prov_blocks = [], []
    n_rows = 0
    t0 = time.time()

    def add_block(vals64, prov_rows):
        nonlocal n_rows
        if vals64.shape[0] == 0:
            return None
        cv = _void_view(quantize_keys(vals64))
        _, first = np.unique(cv, return_index=True)
        keep = []
        for fi in first:
            k = cv[fi].tobytes()
            if k not in seen:
                seen.add(k)
                keep.append(int(fi))
        if not keep:
            return None
        keep = np.array(keep)
        blocks.append(vals64[keep].astype(np.float32))
        prov_blocks.append(prov_rows[keep])
        n_rows += len(keep)
        return vals64[keep]

    with np.errstate(all="ignore"):
        n_terms = terms.shape[0]
        tp = np.stack([np.array([-1, i], dtype=np.int32)
                       for i in range(n_terms)])
        frontier = add_block(terms.copy(), tp)

        for depth in range(1, max_side_depth + 1):
            if frontier is None:
                break
            f = frontier.shape[0]
            g_idx = np.arange(n_rows - f, n_rows, dtype=np.int32)
            # Per-step (and per-frontier-slice) dedupe: a 10M frontier x
            # 512 steps would otherwise concatenate ~5e9 candidate rows
            # before deduping. Kept rows per step are tiny; stream them.
            kept_v = []
            FCH = 2_000_000
            for sid, (side, a, g, t) in enumerate(steps):
                other = terms[t][None, :]
                for fs in range(0, f, FCH):
                    F = frontier[fs:fs + FCH]
                    with np.errstate(all="ignore"):
                        fed = a + g * F
                        val = (np.exp(fed) - np.log(other) if side == 0
                               else np.exp(other) - np.log(fed))
                    oi = np.nonzero(np.isfinite(val).all(axis=1))[0]
                    if oi.size == 0:
                        continue
                    pr = np.empty((oi.size, 2), dtype=np.int32)
                    pr[:, 0] = g_idx[fs + oi]
                    pr[:, 1] = sid
                    kept = add_block(val[oi], pr)
                    if kept is not None:
                        kept_v.append(kept)
            frontier = np.concatenate(kept_v) if kept_v else None
            print(f"  build: depth {depth}, {n_rows} rows, "
                  f"{time.time() - t0:.0f}s", flush=True)

    return (np.concatenate(blocks), np.concatenate(prov_blocks), steps)


# ─── Provenance reconstruction ──────────────────────────────────────

def eval_row(row: int, prov: np.ndarray, steps, n_vars: int,
             X: np.ndarray) -> np.ndarray:
    """Exact float64 value of cache row on arbitrary samples X."""
    terms = term_matrix(n_vars, X)
    chain = []
    r = row
    while prov[r, 0] >= 0:
        chain.append(int(prov[r, 1]))
        r = int(prov[r, 0])
    val = terms[int(prov[r, 1])].copy()
    with np.errstate(all="ignore"):
        for sid in reversed(chain):
            side, a, g, t = steps[sid]
            fed = a + g * val
            val = (np.exp(fed) - np.log(terms[t]) if side == 0
                   else np.exp(terms[t]) - np.log(fed))
    return val


def row_expr(row: int, prov: np.ndarray, steps, n_vars: int) -> str:
    names = [f"x{i}" for i in range(n_vars)] + [f"{c:g}" for c in BARE_CONSTS]
    chain = []
    r = row
    while prov[r, 0] >= 0:
        chain.append(int(prov[r, 1]))
        r = int(prov[r, 0])
    e = names[int(prov[r, 1])]
    for sid in reversed(chain):
        side, a, g, t = steps[sid]
        fed = f"{a:g} + {g:g}*({e})" if a else f"{g:g}*({e})"
        e = (f"eml({fed}, {names[t]})" if side == 0
             else f"eml({names[t]}, {fed})")
    return e


# ─── Join ───────────────────────────────────────────────────────────

def join_search(values32, prov, steps, n_vars, y_scr,
                X_train, y_train, X_held, y_held):
    codes = quantize_keys(values32.astype(np.float64))
    order = np.argsort(_void_view(codes), kind="stable")
    sorted_codes = _void_view(codes[order])

    N = values32.shape[0]
    hits = screened = 0
    discoveries, seen_forms = [], set()
    t0 = time.time()
    CHUNK = 65536
    n_chunks = (N + CHUNK - 1) // CHUNK
    for ci, start in enumerate(range(0, N, CHUNK)):
        U = values32[start:start + CHUNK].astype(np.float64)
        for a_u in ALPHAS:
            for g_u in GAMMAS:
                with np.errstate(all="ignore"):
                    P = np.exp(a_u + g_u * U)
                    v_in_req = np.exp(P - y_scr[None, :])
                for a_v in ALPHAS:
                    for g_v in GAMMAS:
                        with np.errstate(all="ignore"):
                            V_req = (v_in_req - a_v) / g_v
                        ok = np.isfinite(V_req).all(axis=1)
                        idx = np.nonzero(ok)[0]
                        if idx.size == 0:
                            continue
                        q = _void_view(quantize_keys(V_req[idx]))
                        lo = np.searchsorted(sorted_codes, q, "left")
                        hi = np.searchsorted(sorted_codes, q, "right")
                        for r in np.nonzero(hi > lo)[0]:
                            for s in range(int(lo[r]), int(hi[r])):
                                vi = int(order[s])
                                hits += 1
                                j = start + int(idx[r])
                                with np.errstate(all="ignore"):
                                    y_s = (np.exp(a_u + g_u * U[idx[r]])
                                           - np.log(a_v + g_v
                                                    * values32[vi]
                                                    .astype(np.float64)))
                                    d = y_s - y_scr
                                if not (np.all(np.isfinite(d)) and
                                        float(np.mean(d ** 2)) < 1e-7):
                                    continue
                                screened += 1
                                disc = _confirm(j, vi, a_u, g_u, a_v, g_v,
                                                prov, steps, n_vars,
                                                X_train, y_train,
                                                X_held, y_held)
                                if disc and disc["expr"] not in seen_forms:
                                    seen_forms.add(disc["expr"])
                                    discoveries.append(disc)
        if (ci + 1) % 20 == 0 or ci == n_chunks - 1:
            print(f"  join: chunk {ci + 1}/{n_chunks}, {hits} hits, "
                  f"{screened} screened, {len(discoveries)} confirmed, "
                  f"{time.time() - t0:.0f}s", flush=True)
    return hits, screened, discoveries


def _confirm(u_row, v_row, a_u, g_u, a_v, g_v, prov, steps, n_vars,
             X_train, y_train, X_held, y_held):
    U = eval_row(u_row, prov, steps, n_vars, X_train)
    V = eval_row(v_row, prov, steps, n_vars, X_train)
    with np.errstate(all="ignore"):
        y_hat = np.exp(a_u + g_u * U) - np.log(a_v + g_v * V)
    if not (np.all(np.isfinite(y_hat))
            and float(np.mean((y_hat - y_train) ** 2)) < TRAIN_MSE_TOL):
        return None
    U_h = eval_row(u_row, prov, steps, n_vars, X_held)
    V_h = eval_row(v_row, prov, steps, n_vars, X_held)
    with np.errstate(all="ignore"):
        y_hh = np.exp(a_u + g_u * U_h) - np.log(a_v + g_v * V_h)
    scale = max(1.0, float(np.abs(y_held).max()))
    if not (np.all(np.isfinite(y_hh))
            and float(np.abs(y_hh - y_held).max()) < STRUCT_TOL * scale):
        return None
    ue = row_expr(u_row, prov, steps, n_vars)
    ve = row_expr(v_row, prov, steps, n_vars)
    fu = f"{a_u:g} + {g_u:g}*({ue})" if a_u else f"{g_u:g}*({ue})"
    fv = f"{a_v:g} + {g_v:g}*({ve})" if a_v else f"{g_v:g}*({ve})"
    return {"expr": f"eml({fu}, {fv})"}


def self_test() -> bool:
    """Verify the join algebra on the known x0^2 + x3^2 construction."""
    rng = np.random.default_rng(7)
    X = rng.uniform(0.5, 2.5, size=(64, 4))
    y = X[:, 0] ** 2 + X[:, 3] ** 2
    U = 1 - np.log(X[:, 0])                      # eml(0, x0)
    C = np.exp(2 - 2 * (1 - np.log(X[:, 3])))    # x3^2
    V = np.exp(-C)                               # eml(-1*C, 1) chain value
    y_hat = np.exp(2.0 - 2.0 * U) - np.log(V)
    ok = bool(np.max(np.abs(y_hat - y)) < 1e-9)
    print(f"self-test: join algebra {'PASS' if ok else 'FAIL'}",
          flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=sorted(TARGETS))
    ap.add_argument("--max-side-depth", type=int, default=3)
    ap.add_argument("--out-dir", default="benchmarks/results/skeleton_enum")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return 0 if self_test() else 1
    if not args.target:
        ap.error("--target required unless --self-test")

    import zlib
    cfg = TARGETS[args.target]
    n_vars = cfg["n_vars"]
    rng = np.random.default_rng(zlib.crc32(args.target.encode()))
    X = rng.uniform(*TRAIN_DOMAIN, size=(TRAIN_N, n_vars))
    y = cfg["fn"](X)
    X_scr, y_scr = X[:SCREEN_N], y[:SCREEN_N]
    X_held = rng.uniform(*HELD_DOMAIN, size=(HELD_N, n_vars))
    y_held = cfg["fn"](X_held)

    if not self_test():
        return 1
    t0 = time.time()
    values32, prov, steps = build_cache(n_vars, args.max_side_depth, X_scr)
    print(f"# cache: {values32.shape[0]} rows in {time.time() - t0:.0f}s",
          flush=True)
    hits, screened, discoveries = join_search(
        values32, prov, steps, n_vars, y_scr, X, y, X_held, y_held)
    wall = time.time() - t0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.target}_branch_v2.json"
    json.dump(dict(target=args.target,
                   max_side_depth=args.max_side_depth,
                   cache_rows=int(values32.shape[0]),
                   hash_hits=int(hits), screened=int(screened),
                   n_discoveries=len(discoveries), wall_s=wall,
                   discoveries=discoveries[:50]),
              out.open("w"), indent=1)
    print(f"# DONE {args.target} v2-expansion: {len(discoveries)} forms, "
          f"{hits} hits / {screened} screened, {wall:.0f}s -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
