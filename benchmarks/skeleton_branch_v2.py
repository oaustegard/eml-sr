"""Branched-skeleton join v2: columnar cache + quantized sorted-array keys.

Supersedes the dict-of-tuples join in skeleton_branch.py for large
side caches. Three measured lessons drive the design (PR #66):

* Exact float64 keys have FALSE NEGATIVES — the algebraically-derived
  V differs from the cached V by round-off past any fixed rounding, so
  byte-exact lookup misses true joins (32 vs 473 discoveries on
  sum_of_squares). Keys here are uniform int16 cells on arcsinh(V)
  (0.01 resolution): deterministic, injective on exact duplicates,
  tolerant of cross-route float noise.
* Lossy keys REQUIRE multi-occupancy — implemented not as dict buckets
  but as a lexsorted key matrix + binary search (searchsorted on a
  void view): no per-entry dict node overhead at all.
* A 16-sample pre-screen must gate the exact confirm, or collision
  storms run unbounded.

Memory, per cache row: 64 B float32 screen values + 32 B int16 codes
+ 12 B provenance (depth, skeleton_idx, assignment_idx as int32; spec
and assignment are re-enumerated on demand only for rows that survive
the pre-screen) + 8 B sort index = 116 B -> a 40M-row depth-4 cache
fits in ~5 GB where the v1 dict needed ~15.
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

from benchmarks.skeleton_branch import (
    BARE_CONSTS,
    TARGETS,
    TRAIN_MSE_TOL,
    _affine_expr,
    entry_to_expr,
    eval_entry_full,
)
from benchmarks.skeleton_exact import (
    ALPHAS,
    GAMMAS,
    HELD_DOMAIN,
    HELD_N,
    SCREEN_N,
    STRUCT_TOL,
    TRAIN_DOMAIN,
    TRAIN_N,
    chain_skeletons,
    enumerate_assignments,
    eval_chain_batch,
)

KEY_RES = 0.01  # arcsinh-space cell size


def quantize_keys(V: np.ndarray) -> np.ndarray:
    """(n, d) float -> (n, d) int16 codes. Deterministic; identical
    vectors get identical codes; cross-route float noise (<< cell size)
    lands in the same cell."""
    with np.errstate(all="ignore"):
        return np.clip(np.rint(np.arcsinh(V) / KEY_RES),
                       -32000, 32000).astype(np.int16)


def _void_view(codes: np.ndarray) -> np.ndarray:
    c = np.ascontiguousarray(codes)
    return c.view([("", c.dtype)] * c.shape[1]).ravel()


class SortedJoinIndex:
    """Lexsorted quantized-key index over the columnar cache."""

    def __init__(self, values32: np.ndarray, prov: np.ndarray):
        self.values = values32          # (N, s) float32 screen values
        self.prov = prov                # (N, 3) int32
        codes = quantize_keys(values32.astype(np.float64))
        order = np.argsort(_void_view(codes), kind="stable")
        self.sorted_codes = _void_view(codes[order])
        self.order = order

    def lookup(self, Vq: np.ndarray) -> list:
        """For each query row, the range of matching cache indices.
        Returns list of (query_row, cache_index) pairs."""
        q = _void_view(quantize_keys(Vq))
        lo = np.searchsorted(self.sorted_codes, q, side="left")
        hi = np.searchsorted(self.sorted_codes, q, side="right")
        out = []
        for r in np.nonzero(hi > lo)[0]:
            for s in range(lo[r], hi[r]):
                out.append((r, int(self.order[s])))
        return out


def build_columnar_cache(n_vars: int, max_side_depth: int,
                         X_scr: np.ndarray):
    """Columnar side cache: float32 screen values + int32 provenance.

    Provenance rows: (depth, skeleton_idx, assignment_idx); terminals
    use depth 0 with skeleton_idx = -1 (variables: assignment_idx = var
    index; constants: assignment_idx = index into BARE_CONSTS + n_vars).
    Exact duplicates (8-decimal float64 bytes) are dropped on sight.
    """
    vals, prov = [], []
    seen = set()

    def add(v64, p):
        # Dedupe on the QUANTIZED code (32 B/key): near-duplicates
        # collapse to one representative. Alternative provenances of
        # ~equal values are lost, but join recall survives (any
        # representative confirms), and build-time memory stays
        # bounded — the exact-bytes seen-set was ~130 B/row and the
        # per-row float32 list another ~180: prohibitive at 40M rows.
        k = quantize_keys(v64[None, :]).tobytes()
        if k in seen:
            return
        seen.add(k)
        vals.append(v64.astype(np.float32))
        prov.append(p)

    with np.errstate(all="ignore"):
        for i in range(n_vars):
            add(X_scr[:, i].astype(np.float64), (0, -1, i))
        for ci, c in enumerate(BARE_CONSTS):
            add(np.full(X_scr.shape[0], c), (0, -1, n_vars + ci))
        for depth in range(1, max_side_depth + 1):
            for si, spec in enumerate(chain_skeletons(depth, n_vars)):
                assigns = list(enumerate_assignments(spec))
                pred = eval_chain_batch(spec, assigns, X_scr)
                finite = np.isfinite(pred).all(axis=1)
                for h in np.nonzero(finite)[0]:
                    add(pred[h].astype(np.float64), (depth, si, int(h)))
    return (np.stack(vals), np.array(prov, dtype=np.int32),
            len(seen))


# NOTE: vals still accumulates per-row float32 arrays; at very large
# caches the Python-list overhead (~180 B/row) dominates the arrays.
# Block-wise accumulation would cut peak build memory ~2x more if the
# quantized dedupe proves insufficient.


def prov_to_entry(p, n_vars: int, X_scr: np.ndarray):
    """Reconstruct a skeleton_branch-style cache entry from provenance."""
    depth, si, ai = int(p[0]), int(p[1]), int(p[2])
    if depth == 0:
        if ai < n_vars:
            return ("var", None, ai,
                    X_scr[:, ai].astype(np.float64))
        c = BARE_CONSTS[ai - n_vars]
        return ("const", None, c, np.full(X_scr.shape[0], c))
    spec = next(itertools.islice(chain_skeletons(depth, n_vars), si, None))
    assign = next(itertools.islice(enumerate_assignments(spec), ai, None))
    return ("chain", spec, assign, None)


def _confirm_v2(u_entry, v_entry, a_u, g_u, a_v, g_v,
                X_train, y_train, X_held, y_held):
    U_full = eval_entry_full(u_entry, X_train)
    V_full = eval_entry_full(v_entry, X_train)
    with np.errstate(all="ignore"):
        y_hat = np.exp(a_u + g_u * U_full) - np.log(a_v + g_v * V_full)
    if not np.all(np.isfinite(y_hat)):
        return None
    mse = float(np.mean((y_hat - y_train) ** 2))
    if mse >= TRAIN_MSE_TOL:
        return None
    U_h = eval_entry_full(u_entry, X_held)
    V_h = eval_entry_full(v_entry, X_held)
    with np.errstate(all="ignore"):
        y_hh = np.exp(a_u + g_u * U_h) - np.log(a_v + g_v * V_h)
    scale = max(1.0, float(np.abs(y_held).max()))
    if not (np.all(np.isfinite(y_hh))
            and float(np.abs(y_hh - y_held).max()) < STRUCT_TOL * scale):
        return None
    expr = (f"eml({_affine_expr(a_u, g_u, entry_to_expr(u_entry))}, "
            f"{_affine_expr(a_v, g_v, entry_to_expr(v_entry))})")
    return {"expr": expr, "train_mse": mse}


def join_search_v2(index: SortedJoinIndex, n_vars: int,
                   X_scr: np.ndarray, y_scr: np.ndarray,
                   X_train, y_train, X_held, y_held,
                   progress_every: int = 20):
    values = index.values
    N = values.shape[0]
    hits = screened = confirmed = 0
    discoveries = []
    seen_forms = set()
    t0 = time.time()
    CHUNK = 65536
    n_chunks = (N + CHUNK - 1) // CHUNK
    for ci, start in enumerate(range(0, N, CHUNK)):
        U = values[start:start + CHUNK].astype(np.float64)
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
                        pairs = index.lookup(V_req[idx])
                        if not pairs:
                            continue
                        hits += len(pairs)
                        qr_a = np.fromiter((p[0] for p in pairs),
                                           dtype=np.int64, count=len(pairs))
                        vi_a = np.fromiter((p[1] for p in pairs),
                                           dtype=np.int64, count=len(pairs))
                        # batched 16-sample pre-screen over all hits
                        with np.errstate(all="ignore"):
                            y_s = (np.exp(a_u + g_u * U[idx[qr_a]])
                                   - np.log(a_v + g_v
                                            * values[vi_a]
                                            .astype(np.float64)))
                            d = y_s - y_scr[None, :]
                            ok_s = (np.isfinite(d).all(axis=1)
                                    & (np.mean(d ** 2, axis=1) < 1e-7))
                        for w in np.nonzero(ok_s)[0]:
                            screened += 1
                            j = start + int(idx[qr_a[w]])
                            vi = int(vi_a[w])
                            ue = prov_to_entry(index.prov[j], n_vars, X_scr)
                            ve = prov_to_entry(index.prov[vi], n_vars, X_scr)
                            disc = _confirm_v2(ue, ve, a_u, g_u, a_v, g_v,
                                               X_train, y_train,
                                               X_held, y_held)
                            if disc is not None and disc["expr"] not in seen_forms:
                                seen_forms.add(disc["expr"])
                                confirmed += 1
                                discoveries.append(disc)
        if (ci + 1) % progress_every == 0 or ci == n_chunks - 1:
            print(f"  join: chunk {ci + 1}/{n_chunks}, {hits} hits, "
                  f"{screened} screened, {confirmed} confirmed, "
                  f"{time.time() - t0:.0f}s", flush=True)
    return hits, screened, discoveries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--max-side-depth", type=int, default=3)
    ap.add_argument("--out-dir", default="benchmarks/results/skeleton_enum")
    args = ap.parse_args()

    import zlib
    cfg = TARGETS[args.target]
    n_vars = cfg["n_vars"]
    rng = np.random.default_rng(zlib.crc32(args.target.encode()))
    X = rng.uniform(*TRAIN_DOMAIN, size=(TRAIN_N, n_vars))
    y = cfg["fn"](X)
    X_scr, y_scr = X[:SCREEN_N], y[:SCREEN_N]
    X_held = rng.uniform(*HELD_DOMAIN, size=(HELD_N, n_vars))
    y_held = cfg["fn"](X_held)

    t0 = time.time()
    values, prov, n_unique = build_columnar_cache(
        n_vars, args.max_side_depth, X_scr)
    per_row = values.itemsize * values.shape[1] + 12 + 32 + 8
    print(f"# cache: {values.shape[0]} rows, ~{per_row} B/row "
          f"(~{values.shape[0] * per_row / 1e9:.1f} GB), "
          f"{time.time() - t0:.0f}s", flush=True)
    index = SortedJoinIndex(values, prov)
    print(f"# index sorted in {time.time() - t0:.0f}s total", flush=True)

    hits, screened, discoveries = join_search_v2(
        index, n_vars, X_scr, y_scr, X, y, X_held, y_held)
    wall = time.time() - t0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.target}_branch_v2.json"
    json.dump(dict(target=args.target,
                   max_side_depth=args.max_side_depth,
                   cache_rows=int(values.shape[0]),
                   hash_hits=int(hits), screened=int(screened),
                   n_discoveries=len(discoveries), wall_s=wall,
                   discoveries=discoveries[:50]),
              out.open("w"), indent=1)
    print(f"# DONE {args.target} v2: {len(discoveries)} distinct forms, "
          f"{hits} hits / {screened} screened, {wall:.0f}s -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
