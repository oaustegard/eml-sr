"""Hand-built EML trees for the 2x2 and 3x3 permanent and determinant.

Upper bounds by construction, from small gadgets whose sizes are minimal in
benchmarks/perm_det_eml.py (sub 5, lnmul 7, mul 8 are frontier results):

  EXP(x)   = e(x, 1)                                   1 + |x|
  LN(x)    = e(1, e(e(1, x), 1))                       3 + |x|
  SUB(x,y) = e(LN(x), EXP(y))                          5 + |x| + |y|
  F_k      = e - ln w_1 - ... - ln w_k   (accumulator)
             F_1 = e(1, w_1),  F_j = e(LN(F_{j-1}), w_j)
  LNPROD(w_1..w_k) = e(1, EXP(F_k)) = sum ln w_i       4k - 1 + sum |w_i|
  PROD(w_1..w_k)   = EXP(LNPROD)                       4k     + sum |w_i|
  SUM(v_1..v_k)    = LNPROD(EXP v_1, ..., EXP v_k)     5k - 1 + sum |v_i|

LNPROD(x, y) is the 7-node lnmul tree the enumeration finds, and PROD(x, y)
the 8-node mul. Peephole: LN(PROD(...)) is LNPROD(...), and SUB(PROD(..), y)
uses it, which is where det2 = 17 comes from.

Exactness: on the complex principal branch every tree below equals its target
for all nonzero real inputs, because each gadget uses ln only in the form
exp(ln w) = w or ln(exp v) = v with v real (the 2*pi*i ambiguity of a sum of
logs is removed by the final exp). On the real branch (all ln arguments must
be positive reals) the accumulator needs F_j > 0, which restricts the domain;
check_real() reports how much of a positive box survives.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.perm_det_eml import depth, evaluate, parse, size


def EXP(x):
    return f"e({x},1)"


def LN(x):
    if x.startswith("PROD:"):
        return x[5:]
    return f"e(1,e(e(1,{x}),1))"


def F(ws):
    acc = f"e(1,{ws[0]})"
    for w in ws[1:]:
        acc = f"e({LN(acc)},{w})"
    return acc


def LNPROD(*ws):
    return f"e(1,{EXP(F(list(ws)))})"


def PROD(*ws):
    """Product, tagged so LN() can strip the outer exp."""
    return "PROD:" + LNPROD(*ws)


def S(x):
    """Untag a PROD for use as a plain subtree."""
    return EXP(x[5:]) if x.startswith("PROD:") else x


def SUB(x, y):
    return f"e({LN(x)},{EXP(S(y))})"


def SUM(*vs):
    return LNPROD(*[EXP(S(v)) for v in vs])


# matrix entries: 2x2 [[x0, x1], [x2, x3]], 3x3 row-major x0..x8
def det2(a, b, c, d):
    return SUB(PROD(a, d), PROD(b, c))


def perm2(a, b, c, d):
    return SUM(PROD(a, d), PROD(b, c))


def cands_2x2():
    v = [f"x{i}" for i in range(4)]
    return {
        "det2": {"sub(prod, prod)": det2(*v)},
        "perm2": {"sum(prod, prod)": perm2(*v)},
    }


def cands_3x3():
    x = [f"x{i}" for i in range(9)]
    a, b, c, d, e, f, g, h, i = x
    m1, m2, m3 = det2(e, f, h, i), det2(d, f, g, i), det2(d, e, g, h)
    p1, p2, p3 = perm2(e, f, h, i), perm2(d, f, g, i), perm2(d, e, g, h)
    P = [PROD(a, e, i), PROD(b, f, g), PROD(c, d, h)]
    Q = [PROD(a, f, h), PROD(b, d, i), PROD(c, e, g)]
    out = {
        "det3": {
            # a*m1 - (b*m2 - c*m3)
            "cofactor a*m1 - (b*m2 - c*m3)": SUB(PROD(a, m1), SUB(PROD(b, m2), PROD(c, m3))),
            # (aei + bfg + cdh) - (afh + bdi + ceg)
            "sum3(P) - sum3(Q)": SUB(SUM(*P), SUM(*Q)),
        },
        "perm3": {
            "cofactor sum3(a*p1, b*p2, c*p3)": SUM(PROD(a, p1), PROD(b, p2), PROD(c, p3)),
            "sum6 of triple products": SUM(*(P + Q)),
        },
    }
    return out


def target_fns():
    def det3(X):
        return np.linalg.det(X[:, :9].reshape(-1, 3, 3))

    def perm3(X):
        A = X[:, :9].reshape(-1, 3, 3)
        from itertools import permutations
        return sum(A[:, 0, p[0]] * A[:, 1, p[1]] * A[:, 2, p[2]] for p in permutations(range(3)))

    return {
        "det2": (4, lambda X: X[:, 0] * X[:, 3] - X[:, 1] * X[:, 2]),
        "perm2": (4, lambda X: X[:, 0] * X[:, 3] + X[:, 1] * X[:, 2]),
        "det3": (9, det3),
        "perm3": (9, perm3),
    }


def check(w: str, fn, nvars: int, cplx: bool, X: np.ndarray) -> dict:
    t = parse(w)
    env = {f"x{k}": X[:, k].astype(np.complex128 if cplx else np.float64) for k in range(nvars)}
    v, ok = evaluate(t, env, cplx)
    T = fn(X)
    with np.errstate(all="ignore"):
        err = np.abs(v - T) / (1 + np.abs(T))
    good = ok & np.isfinite(err) & (err < 1e-8)
    return {"points": len(X), "exact": int(good.sum()),
            "max_rel_err": float(err[good].max()) if good.any() else None}


def mp_check(w: str, fn, nvars: int, X: np.ndarray, dps: int = 40) -> float:
    """Max |tree - target| at `dps` digits over the rows of X (complex principal branch)."""
    import mpmath as mp
    mp.mp.dps = dps
    t = parse(w)

    def ev(node, env):
        if isinstance(node, str):
            return mp.mpc(1) if node == "1" else env[node]
        return mp.exp(ev(node[0], env)) - mp.log(ev(node[1], env))

    worst = 0.0
    for row in X:
        env = {f"x{k}": mp.mpc(float(row[k])) for k in range(nvars)}
        tv = fn(row[None, :])[0]
        worst = max(worst, float(abs(ev(t, env) - tv) / (1 + abs(tv))))
    return worst


def sympy_identity(w: str, nvars: int, target) -> bool:
    """True if sympy reduces tree - target to 0 with positive real symbols.

    sympy rewrites exp(log(z)) -> z (true for every z != 0) and log(exp(v)) -> v
    for real v, which are exactly the identities the gadgets rely on, so a 0
    here is a symbolic proof for the complex principal branch on nonzero reals.
    """
    import sympy as sp
    xs = sp.symbols(f"x0:{nvars}", positive=True)
    env = {f"x{k}": xs[k] for k in range(nvars)}

    def build(t):
        if isinstance(t, str):
            return sp.Integer(1) if t == "1" else env[t]
        return sp.exp(build(t[0])) - sp.log(build(t[1]))

    return sp.simplify(sp.expand(build(parse(w)) - target(xs))) == 0


def sym_targets():
    import sympy as sp

    def M(xs, n):
        return sp.Matrix(n, n, list(xs[:n * n]))

    return {"det2": lambda xs: M(xs, 2).det(), "perm2": lambda xs: M(xs, 2).per(),
            "det3": lambda xs: M(xs, 3).det(), "perm3": lambda xs: M(xs, 3).per()}


def main() -> None:
    rng = np.random.default_rng(7)
    tf = target_fns()
    res = {}
    allc = {**cands_2x2(), **cands_3x3()}
    for tn, cs in allc.items():
        nv, fn = tf[tn]
        Xc = rng.uniform(-3, 3, size=(4000, nv))
        Xc = np.where(np.abs(Xc) < 0.05, 0.05, Xc)
        Xr = np.exp(rng.uniform(np.log(0.2), np.log(5), size=(4000, nv)))
        Xr_small = np.exp(rng.uniform(np.log(0.2), np.log(1.5), size=(4000, nv)))
        res[tn] = {}
        for name, w in cs.items():
            w = S(w)
            t = parse(w)
            entry = {"size": size(t), "depth": depth(t), "witness": w,
                     "complex_all_real_inputs": check(w, fn, nv, True, Xc),
                     "complex_mp40_max_rel_err_30pts": mp_check(w, fn, nv, Xc[:30]),
                     "real_box_0.2_5": check(w, fn, nv, False, Xr),
                     "real_box_0.2_1.5": check(w, fn, nv, False, Xr_small)}
            entry["sympy_identity"] = sympy_identity(w, nv, sym_targets()[tn])
            res[tn][name] = entry
            print(tn, name, json.dumps({k: v for k, v in entry.items() if k != "witness"}), flush=True)
    out = ROOT / "benchmarks" / "results" / "perm_det_eml" / "gadgets.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
