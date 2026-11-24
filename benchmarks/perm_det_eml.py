"""Smallest pure-EML trees for the 2x2 permanent and determinant (and pieces).

Grammar: S -> 1 | x0 | ... | x_{n-1} | eml(S, S), eml(u, v) = exp(u) - ln(v).
Size = number of eml nodes. No other constants, no affine leaves.

Method (benchmarks/eml_complexity.py generalised to n inputs):

* Bottom-up enumeration by size with observational dedupe. Every tree is
  evaluated at K random sample points; trees whose K values agree to SIG
  significant digits are one function and only the first (smallest) is kept.
  A target found on the frontier at size s is minimal.
* Sizes 0..N-1 are stored with values. Size N is streamed: values exist only
  chunk by chunk, keys and parent pointers are kept.
* Root join: eml(L, R) = T  <=>  R = exp(exp(L) - T). With L over sizes
  0..N-1 looked up against 0..N, and L at size N looked up against 0..N-1,
  every tree of size <= N+1 is covered, so a miss proves size > N+1.
* Shape joins for upper bounds: one of the root's grandchildren Z is taken
  from the small sizes 0..zmax and the other grandchild Y is looked up:
      S1  eml(L, eml(Y, Z))     S2  eml(L, eml(Z, Y))
      S3  eml(eml(Y, Z), R)     S4  eml(eml(Z, Y), R)
  A hit is a verified tree; a miss proves nothing.

Branches:
  real     inputs log-uniform on [0.2, 5]; every ln argument must be a positive
           real at every sample point.
  complex  inputs uniform on [-3, 3] (both signs, |x| >= 0.1); principal
           branch, near-real values snapped to the axis.

Hits are verified on the K sample points during search, then re-checked on
1000 fresh points from the same distribution. The fresh-point check is what
separates trees that are exact on the whole sampling region from trees that
are exact only on part of it (their ln arguments leave the domain elsewhere).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SIG = 11
JSIG = 8  # coarser key for join lookups, so rounding error rarely straddles a cell
MAG = 1e200
EXACT = 1e-9
TOL = 1e-8
MAXHITS = 400


def splitmix(h: np.ndarray) -> np.ndarray:
    """Splitmix64 finalizer on int64 arrays (wrapping arithmetic)."""
    h = h ^ (h >> np.int64(30))
    h = h * np.int64(-4658895280553007687)
    h = h ^ (h >> np.int64(27))
    h = h * np.int64(-7723592293110705685)
    return h ^ (h >> np.int64(31))


def quant(t: np.ndarray, sig: int) -> np.ndarray:
    """Round float64 values to `sig` significant digits (tiny values to 0)."""
    out = np.zeros_like(t)
    nz = np.abs(t) >= 1e-12
    e = np.floor(np.log10(np.abs(t[nz])))
    out[nz] = np.round(t[nz] * 10.0 ** (sig - 1 - e)) / 10.0 ** (sig - 1 - e)
    return out


def keys_of(v: np.ndarray, sig: int) -> np.ndarray:
    """Hash rows of v (m, K) to int64 keys at `sig` significant digits."""
    with np.errstate(all="ignore"):
        parts = [quant(v.real, sig), quant(v.imag, sig)] if np.iscomplexobj(v) else [quant(v, sig)]
        h = np.zeros(v.shape[0], dtype=np.int64)
        salt = np.int64(7150367640395946793)
        for p in parts:
            bits = p.view(np.int64)
            for k in range(bits.shape[1]):
                h = splitmix(h * salt + bits[:, k] + np.int64(k + 1))
    return h


class Engine:
    """Bottom-up enumeration of EML trees over n input variables."""

    def __init__(self, nvars: int, branch: str, K: int, seed: int, lo: float = 0.2, hi: float = 5.0):
        self.n = nvars
        self.branch = branch
        self.cplx = branch == "complex"
        self.dt = np.complex128 if self.cplx else np.float64
        self.K = K
        self.lo, self.hi = lo, hi
        self.X = self.sample(np.random.default_rng(seed), K)
        self.names = ["1"] + [f"x{i}" for i in range(nvars)]
        v0 = np.array([np.ones(K)] + [self.X[:, i] for i in range(nvars)], dtype=self.dt)
        self.vals = [v0]
        self.par = [np.zeros((len(v0), 3), dtype=np.int32)]
        self.keys = [keys_of(v0, SIG)]
        self.Eidx, self.Ev, self.Lidx, self.Lv = [], [], [], []
        self._prep(0)
        self.gsorted = np.sort(self.keys[0])
        self.hits: dict[str, list[tuple]] = {}
        self.stats: list[dict] = []

    def sample(self, rng, m: int) -> np.ndarray:
        if self.cplx:
            x = rng.uniform(-3.0, 3.0, size=(m, self.n))
            return np.where(np.abs(x) < 0.1, np.sign(x) * 0.1 + x, x)
        return np.exp(rng.uniform(math.log(self.lo), math.log(self.hi), size=(m, self.n)))

    # --- elementwise helpers ---------------------------------------------
    def snap(self, v):
        if not self.cplx:
            return v
        return np.where(np.abs(v.imag) < 1e-11 * np.abs(v), v.real + 0j, v)

    def safe_log(self, v):
        """ln with validity mask (rows valid only if all K entries valid)."""
        with np.errstate(all="ignore"):
            if self.cplx:
                v = self.snap(v)
                Lg = np.log(v)
                ok = (v != 0) & np.isfinite(v)
            else:
                ok = (v > 0) & np.isfinite(v)
                Lg = np.log(np.where(ok, v, 1.0))
            ok &= np.isfinite(Lg)
        return Lg, np.all(ok, axis=-1)

    def safe_exp(self, v):
        with np.errstate(all="ignore"):
            E = np.exp(v)
            ok = np.isfinite(E) & (np.abs(E) < MAG)
        return E, np.all(ok, axis=-1)

    def _prep(self, lvl: int) -> None:
        v = self.vals[lvl]
        E, eok = self.safe_exp(v)
        Lg, lok = self.safe_log(v)
        self.Eidx.append(np.nonzero(eok)[0])
        self.Ev.append(E[eok])
        self.Lidx.append(np.nonzero(lok)[0])
        self.Lv.append(Lg[lok])

    def node(self, E: np.ndarray, Lg: np.ndarray):
        """exp(L) - ln(R) from cached blocks, with the exactness filter."""
        with np.errstate(all="ignore"):
            v = self.snap(E - Lg)
            ok = np.isfinite(v.real) & np.isfinite(v.imag) if self.cplx else np.isfinite(v)
            ok &= np.abs(v) < MAG
            ea, lb = np.abs(E), np.abs(Lg)
            mag = np.maximum(ea, lb)
            ok &= ~(ea < EXACT * mag) & ~((lb > 0) & (lb < EXACT * mag))
        return v, np.all(ok, axis=-1)

    def match(self, v: np.ndarray, T: np.ndarray) -> np.ndarray:
        return np.all(np.abs(v - T) <= TOL * (1 + np.abs(T)), axis=-1)

    def add_hit(self, name: str, size: int, wit: tuple) -> None:
        lst = self.hits.setdefault(name, [])
        if len(lst) < MAXHITS:
            lst.append((size, wit))

    # --- enumeration -------------------------------------------------------
    def pairs(self, n: int, chunk_pairs: int):
        """Yield (i, v, li, ri) chunks of all valid eml(L, R) with |L|+|R| = n-1."""
        for i in range(n):
            j = n - 1 - i
            EL, Li = self.Ev[i], self.Eidx[i]
            LR, Ri = self.Lv[j], self.Lidx[j]
            if len(EL) == 0 or len(LR) == 0:
                continue
            bl = max(1, chunk_pairs // len(LR))
            for s in range(0, len(EL), bl):
                e = EL[s:s + bl]
                v, ok = self.node(e[:, None, :], LR[None, :, :])
                v = v.reshape(-1, self.K)[ok.reshape(-1)]
                li = np.repeat(Li[s:s + bl], len(LR))[ok.reshape(-1)]
                ri = np.tile(Ri, len(e))[ok.reshape(-1)]
                yield i, v, li, ri

    def expand(self, targets: dict, stream: bool = False, join_targets: dict | None = None,
               zmax: int = 1, chunk_pairs: int = 3_000_000, log=print) -> int:
        """Build size n = len(self.vals). If stream, keep keys/parents only and
        run the joins with L (or R) at size n against sizes < n on the fly."""
        n = len(self.vals)
        t0 = time.time()
        nk, nv, npar = [], [], []
        cand = 0
        for i, v, li, ri in self.pairs(n, chunk_pairs):
            cand += len(v)
            for tn, T in targets.items():
                for h in np.nonzero(self.match(v, T[None, :]))[0][:8]:
                    self.add_hit(tn, n, ("e", (i, int(li[h])), (n - 1 - i, int(ri[h]))))
            k = keys_of(v, SIG)
            k, first = np.unique(k, return_index=True)
            pos = np.minimum(np.searchsorted(self.gsorted, k), len(self.gsorted) - 1)
            fresh = self.gsorted[pos] != k
            sel = first[fresh]
            nk.append(k[fresh])
            npar.append(np.stack([np.full(len(sel), i), li[sel], ri[sel]], axis=1).astype(np.int32))
            if stream:
                if join_targets:
                    self._chunk_joins(v[sel], n, np.arange(len(sel)) + sum(len(x) for x in nk[:-1]),
                                      join_targets, zmax, upto=n - 1, raw=True)
            else:
                nv.append(v[sel])
        k = np.concatenate(nk) if nk else np.zeros(0, np.int64)
        k, first = np.unique(k, return_index=True)
        p = (np.concatenate(npar) if npar else np.zeros((0, 3), np.int32))[first]
        if stream:
            self.rawpar = np.concatenate(npar) if npar else np.zeros((0, 3), np.int32)
            self.vals.append(None)
        else:
            v = (np.concatenate(nv) if nv else np.zeros((0, self.K), self.dt))[first]
            self.vals.append(v)
        self.par.append(p)
        self.keys.append(k)
        self.gsorted = np.sort(np.concatenate([self.gsorted, k]))
        if not stream:
            self._prep(n)
        st = {"size": n, "candidates": int(cand), "distinct": len(k), "secs": round(time.time() - t0, 1)}
        self.stats.append(st)
        log(json.dumps(st))
        return len(k)

    # --- value / witness reconstruction -----------------------------------
    def value(self, lvl: int, idx: int) -> np.ndarray:
        if self.vals[lvl] is not None:
            return self.vals[lvl][idx]
        i, li, ri = (int(t) for t in self.par[lvl][idx])
        a, b = self.value(i, li), self.value(lvl - 1 - i, ri)
        with np.errstate(all="ignore"):
            return self.snap(np.exp(a) - np.log(b.astype(self.dt) if self.cplx else b))

    def witness(self, lvl: int, idx: int) -> str:
        if lvl == 0:
            return self.names[idx]
        i, li, ri = (int(t) for t in self.par[lvl][idx])
        return f"e({self.witness(i, li)},{self.witness(lvl - 1 - i, ri)})"

    def wstr(self, w) -> str:
        if w[0] == "e":
            return f"e({self.wstr(w[1])},{self.wstr(w[2])})"
        lvl, idx = w
        if isinstance(idx, tuple):  # ("raw", j): streamed size, pre-dedupe parent row
            i, li, ri = (int(t) for t in self.rawpar[idx[1]])
            return f"e({self.witness(i, li)},{self.witness(lvl - 1 - i, ri)})"
        return self.witness(lvl, idx)

    # --- joins -------------------------------------------------------------
    def build_index(self, upto: int) -> None:
        ks, ls, ids = [], [], []
        for lvl in range(upto + 1):
            if self.vals[lvl] is not None:
                ks.append(keys_of(self.vals[lvl], JSIG))
            else:
                ks.append(self._coarse_streamed(lvl))
            ls.append(np.full(len(ks[-1]), lvl, dtype=np.int8))
            ids.append(np.arange(len(ks[-1]), dtype=np.int64))
        k = np.concatenate(ks)
        order = np.argsort(k, kind="stable")
        self.ix_k, self.ix_l, self.ix_i = k[order], np.concatenate(ls)[order], np.concatenate(ids)[order]
        self.ix_upto = upto

    def _coarse_streamed(self, lvl: int) -> np.ndarray:
        out = np.zeros(len(self.par[lvl]), dtype=np.int64)
        B = 200_000
        for s in range(0, len(out), B):
            p = self.par[lvl][s:s + B]
            v = np.empty((len(p), self.K), self.dt)
            for i in np.unique(p[:, 0]):
                m = p[:, 0] == i
                a = self.vals[i][p[m, 1]]
                b = self.vals[lvl - 1 - i][p[m, 2]]
                with np.errstate(all="ignore"):
                    v[m] = self.snap(np.exp(a) - np.log(b))
            out[s:s + B] = keys_of(v, JSIG)
        return out

    def lookup(self, req: np.ndarray, ok: np.ndarray, maxlvl: int):
        """Rows of req whose value is (coarsely) in the index at size <= maxlvl."""
        with np.errstate(all="ignore"):
            mag = np.abs(req)
            # required values that underflow or blow up cannot be matched reliably
            ok = ok & np.all((mag > 1e-11) & (mag < MAG), axis=-1)
        rows = np.nonzero(ok)[0]
        if len(rows) == 0:
            return []
        k = keys_of(req[rows], JSIG)
        pos = np.minimum(np.searchsorted(self.ix_k, k), len(self.ix_k) - 1)
        m = (self.ix_k[pos] == k) & (self.ix_l[pos] <= maxlvl)
        return [(int(r), int(self.ix_l[p]), int(self.ix_i[p])) for r, p in zip(rows[m], pos[m])]

    def _chunk_joins(self, A: np.ndarray, lvl: int, idx: np.ndarray, jt: dict, zmax: int, upto: int,
                     raw: bool = False):
        """Joins with anchor values A (rows of size `lvl`) against the index (<= upto)."""
        if len(A) == 0:
            return
        if getattr(self, "ix_upto", -1) < upto:
            self.build_index(upto)
        EA, eok = self.safe_exp(A)
        LA, lok = self.safe_log(A)
        anc = raw
        for tn, T in jt.items():
            # S0 root: anchor L, need R = exp(e^L - T)
            with np.errstate(all="ignore"):
                R, rok = self.safe_exp(EA - T[None, :])
                _rl, rlok = self.safe_log(R)
            rok &= eok & rlok
            self._emit(tn, "S0", A, lvl, idx, R, rok, upto, lambda a, y: ("e", a, y), anc)
            # W for anchor R: e^{left} = T + ln R, left = Log(T + ln R)
            with np.errstate(all="ignore"):
                W, wok = self.safe_log(T[None, :] + LA)
            wok &= lok
            if raw or tn not in SHAPE_TARGETS:
                continue  # streamed anchors and minor targets: root join only
            for zl in range(zmax + 1):
                for zi in range(len(self.vals[zl])):
                    Zv = self.vals[zl][zi]
                    zw = (zl, zi)
                    lz, lzok = self.safe_log(Zv[None, :])
                    ez, ezok = self.safe_exp(Zv[None, :])
                    with np.errstate(all="ignore"):
                        if lzok[0]:
                            # S1 eml(L, eml(Y, Z)): e^Y = R + ln Z
                            Y, yok = self.safe_log(R + lz)
                            self._emit(tn, "S1", A, lvl, idx, Y, yok & rok, upto,
                                       lambda a, y, zw=zw: ("e", a, ("e", y, zw)), anc)
                            # S3 eml(eml(Y, Z), R): e^Y = W + ln Z
                            Y, yok = self.safe_log(W + lz)
                            self._emit(tn, "S3", A, lvl, idx, Y, yok & wok, upto,
                                       lambda a, y, zw=zw: ("e", ("e", y, zw), a), anc)
                        if ezok[0]:
                            # S2 eml(L, eml(Z, Y)): ln Y = e^Z - R
                            Y, yok = self.safe_exp(ez - R)
                            self._emit(tn, "S2", A, lvl, idx, Y, yok & rok, upto,
                                       lambda a, y, zw=zw: ("e", a, ("e", zw, y)), anc)
                            # S4 eml(eml(Z, Y), R): ln Y = e^Z - W
                            Y, yok = self.safe_exp(ez - W)
                            self._emit(tn, "S4", A, lvl, idx, Y, yok & wok, upto,
                                       lambda a, y, zw=zw: ("e", ("e", zw, y), a), anc)

    def _emit(self, tn, shape, A, lvl, idx, req, ok, upto, build, anc):
        for r, yl, yi in self.lookup(req, ok, upto):
            aref = (lvl, ("raw", int(idx[r]))) if anc else (lvl, int(idx[r]))
            w = build(aref, (yl, yi))
            self.pending.append((tn, shape, w, A[r]))

    def run_joins(self, jt: dict, zmax: int, log=print) -> None:
        """Joins with anchors at stored sizes against the full index."""
        N = len(self.vals) - 1
        self.build_index(N)
        for lvl in range(N + 1):
            if self.vals[lvl] is None:
                continue
            B = 400_000
            for s in range(0, len(self.vals[lvl]), B):
                A = self.vals[lvl][s:s + B]
                self._chunk_joins(A, lvl, np.arange(s, s + len(A)), jt, zmax, upto=N)
            log(f"  joins anchored at size {lvl} done; pending {len(self.pending)}")

    def verify_pending(self, jt: dict, log=print) -> None:
        """Turn pending join candidates into verified hits."""
        seen = set()
        for tn, shape, w, _ in self.pending:
            s = self.wstr(w)
            if (tn, s) in seen:
                continue
            seen.add((tn, s))
            t = parse(s)
            env = {f"x{i}": self.X[:, i].astype(self.dt) for i in range(self.n)}
            v, okm = evaluate(t, env, self.cplx)
            if np.all(okm) and self.match(self.snap(v), jt[tn]):
                self.add_hit(tn, size(t), ("str", s, shape))
        self.pending = []


# --- tree utilities ------------------------------------------------------
def parse(w: str):
    """Parse 'e(L,R)' witness strings into nested tuples; leaves are names."""
    pos = 0

    def at():
        nonlocal pos
        if w.startswith("e(", pos):
            pos += 2
            a = at()
            assert w[pos] == ","
            pos += 1
            b = at()
            assert w[pos] == ")"
            pos += 1
            return (a, b)
        j = pos
        while pos < len(w) and w[pos] not in ",)":
            pos += 1
        return w[j:pos]

    t = at()
    assert pos == len(w)
    return t


def size(t) -> int:
    return 0 if isinstance(t, str) else 1 + size(t[0]) + size(t[1])


def depth(t) -> int:
    return 0 if isinstance(t, str) else 1 + max(depth(t[0]), depth(t[1]))


def evaluate(t, env: dict, cplx: bool):
    """Evaluate a tree elementwise; returns (value, mask of valid ln arguments)."""
    if isinstance(t, str):
        v = np.ones_like(next(iter(env.values()))) if t == "1" else env[t]
        return v, np.ones(v.shape, bool)
    a, oka = evaluate(t[0], env, cplx)
    b, okb = evaluate(t[1], env, cplx)
    with np.errstate(all="ignore"):
        if cplx:
            b = b.astype(np.complex128)
            b = np.where(np.abs(b.imag) < 1e-11 * np.abs(b), b.real + 0j, b)
            v = np.exp(a) - np.log(b)
            v = np.where(np.abs(v.imag) < 1e-11 * np.abs(v), v.real + 0j, v)
            return v, oka & okb & (b != 0) & np.isfinite(v)
        ok = oka & okb & (b > 0)
        v = np.exp(a) - np.log(np.where(b > 0, b, 1.0))
        return v, ok & np.isfinite(v)


def check_points(w: str, target_fn, nvars: int, cplx: bool, X: np.ndarray) -> dict:
    t = parse(w)
    env = {f"x{i}": X[:, i].astype(np.complex128 if cplx else np.float64) for i in range(nvars)}
    v, ok = evaluate(t, env, cplx)
    T = target_fn(X)
    with np.errstate(all="ignore"):
        err = np.abs(v - T) / (1 + np.abs(T))
    good = ok & np.isfinite(err) & (err < 1e-9)
    return {"points": len(X), "exact": int(good.sum()),
            "max_err_on_exact": float(err[good].max()) if good.any() else None}


# --- targets -------------------------------------------------------------
def _c(z):
    return np.asarray(z, dtype=np.complex128)


TARGETS_2x2 = {
    # matrix [[x0, x1], [x2, x3]]
    "perm": ("x0*x3 + x1*x2", lambda X: X[:, 0] * X[:, 3] + X[:, 1] * X[:, 2]),
    "det": ("x0*x3 - x1*x2", lambda X: X[:, 0] * X[:, 3] - X[:, 1] * X[:, 2]),
    "mul": ("x0*x1", lambda X: X[:, 0] * X[:, 1]),
    "add": ("x0 + x1", lambda X: X[:, 0] + X[:, 1]),
    "sub": ("x0 - x1", lambda X: X[:, 0] - X[:, 1]),
    "neg": ("-x0", lambda X: -X[:, 0]),
    "inv": ("1/x0", lambda X: 1 / X[:, 0]),
    "div": ("x0/x1", lambda X: X[:, 0] / X[:, 1]),
    "lnmul": ("log(x0) + log(x1)", lambda X: np.log(_c(X[:, 0])) + np.log(_c(X[:, 1]))),
    "expmul": ("exp(x0*x1)", lambda X: np.exp(X[:, 0] * X[:, 1])),
    "expnegmul": ("exp(-x0*x1)", lambda X: np.exp(-X[:, 0] * X[:, 1])),
    "negmul": ("-x0*x1", lambda X: -X[:, 0] * X[:, 1]),
    "e_minus_mul": ("E - x0*x1", lambda X: np.e - X[:, 0] * X[:, 1]),
    "mul3": ("x0*x1*x2", lambda X: X[:, 0] * X[:, 1] * X[:, 2]),
    "lnmul3": ("log(x0)+log(x1)+log(x2)", lambda X: np.log(_c(X[:, 0])) + np.log(_c(X[:, 1])) + np.log(_c(X[:, 2]))),
}
JOIN_2x2 = ("perm", "det", "add", "expmul", "expnegmul", "negmul", "mul3")
SHAPE_TARGETS = ("perm", "det")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--branch", choices=["real", "complex"], default="real")
    ap.add_argument("--nmax", type=int, default=8, help="last size; it is streamed")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--zmax", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args(argv)
    log = lambda s: print(s, flush=True)

    eng = Engine(4, args.branch, args.K, args.seed)
    eng.pending = []
    conv = (lambda a: a.astype(np.complex128)) if eng.cplx else (lambda a: np.real(a).astype(np.float64))
    tv = {k: conv(f(eng.X)) for k, (_, f) in TARGETS_2x2.items()}
    jt = {k: tv[k] for k in JOIN_2x2}
    for n in range(1, args.nmax + 1):
        last = n == args.nmax
        if last:
            eng.build_index(n - 1)
        eng.expand(tv, stream=last, join_targets=jt if last else None, zmax=args.zmax, log=log)
    log(f"streamed-anchor join candidates: {len(eng.pending)}")
    eng.run_joins(jt, args.zmax, log=log)
    eng.verify_pending(jt, log=log)

    rng = np.random.default_rng(12345)
    Xc = eng.sample(rng, 1000)
    N = args.nmax
    res = {"branch": args.branch, "K": args.K, "seed": args.seed, "enumerated_to": N,
           "root_join_exact_to": N + 1, "zmax": args.zmax, "levels": eng.stats, "targets": {}}
    for tn, (texpr, f) in TARGETS_2x2.items():
        cands = []
        for s, w in eng.hits.get(tn, []):
            if w[0] == "str":
                ws, how = w[1], w[2]
            else:
                ws, how = eng.wstr(w), "frontier"
            cands.append((s, ws, how))
        cands.sort(key=lambda c: (c[0], c[1]))
        entry = {"target": texpr}
        if not cands:
            entry["status"] = f"> {N + 1} (frontier to {N}, root join to {N + 1}, shape joins empty)" \
                if tn in jt else f"> {N} (frontier only)"
        else:
            checked = []
            for s, ws, how in cands[:60]:
                c = check_points(ws, lambda X, f=f: conv(f(X)), 4, eng.cplx, Xc)
                checked.append({"size": s, "depth": depth(parse(ws)), "how": how, "witness": ws, "check": c})
            best = checked[0]
            full = [c for c in checked if c["check"]["exact"] == c["check"]["points"]]
            smin = best["size"]
            exact = best["how"] == "frontier" or (best["how"] == "S0" and smin <= N + 1) or smin <= N + 1
            entry["status"] = "minimal" if exact else "upper bound"
            entry["smallest"] = best
            entry["smallest_exact_on_all_fresh_points"] = full[0] if full else None
        res["targets"][tn] = entry
        log(f"{tn}: {json.dumps(entry)}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
