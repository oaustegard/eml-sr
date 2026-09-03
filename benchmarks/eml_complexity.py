"""Vectorized EML constant complexity (real branch or complex principal branch).

Per size n: values array, quantized int64 key, parent pointers (split a, ia, ib).
Global dedupe via a sorted int64 key array (observational equivalence at 11 sig
digits). Witness trees reconstructed only for targets.
"""
import sys, math, cmath, time, json
import numpy as np

BRANCH = sys.argv[1] if len(sys.argv) > 1 else "real"
NMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 18
SIG = 11
MAG = 1e300
EXACT = 1e-9
CPLX = BRANCH == "complex"


def qkey(v):
    """Quantize to SIG significant digits per component; hash to int64."""
    def q(t):
        t = np.asarray(t, dtype=np.float64)
        out = np.zeros_like(t)
        nz = np.abs(t) >= 1e-12
        e = np.floor(np.log10(np.abs(t[nz])))
        scale = 10.0 ** (SIG - 1 - e)
        out[nz] = np.round(t[nz] * scale) / scale
        return out
    def mix(h):
        h = h ^ (h >> np.int64(30)); h = h * np.int64(-4658895280553007687)
        h = h ^ (h >> np.int64(27)); h = h * np.int64(-7723592293110705685)
        return h ^ (h >> np.int64(31))
    if CPLX:
        r, i = q(v.real), q(v.imag)
        return mix(mix(r.view(np.int64)) + np.int64(7150367640395946793) * i.view(np.int64))
    else:
        return mix(q(v).view(np.int64))


def eml_vec(a, b):
    with np.errstate(all="ignore"):
        if CPLX:
            # canonicalize: snap near-real values to the real axis so ln on the
            # negative axis takes the principal value +i*pi deterministically
            b = np.where(np.abs(b.imag) < 1e-11 * np.abs(b), b.real + 0j, b)
            a = np.where(np.abs(a.imag) < 1e-11 * np.abs(a), a.real + 0j, a)
            v = np.exp(a) - np.log(b)
            v = np.where(np.abs(v.imag) < 1e-11 * np.abs(v), v.real + 0j, v)
            ok = np.isfinite(v.real) & np.isfinite(v.imag) & (np.abs(v) < MAG) & (b != 0)
        else:
            v = np.exp(a) - np.log(b)
            ok = np.isfinite(v) & (b > 0) & (np.abs(v) < MAG)
        # exactness: neither term may be numerically swallowed by the other
        ea = np.abs(np.exp(a)); lb = np.abs(np.log(b)); mag = np.maximum(ea, lb)
        ok &= ~(ea < EXACT * mag) & ~((lb > 0) & (lb < EXACT * mag))   # exp never truly 0
    return v, ok


dt = np.complex128 if CPLX else np.float64
vals = {0: np.array([1.0], dtype=dt)}
par = {0: None}
allkeys = np.sort(qkey(vals[0]))
counts, catalan = [1], [1]
t0 = time.time()
for n in range(1, NMAX + 1):
    parts_v, parts_p = [], []
    for a in range(n):
        b = n - 1 - a
        va, vb = vals[a], vals[b]
        if len(va) == 0 or len(vb) == 0: continue
        # chunk over a to bound memory
        step = max(1, int(2e7 // max(1, len(vb))))
        for s in range(0, len(va), step):
            A = va[s:s + step]
            ia = np.repeat(np.arange(s, s + len(A), dtype=np.int32), len(vb))
            ib = np.tile(np.arange(len(vb), dtype=np.int32), len(A))
            v, ok = eml_vec(np.repeat(A, len(vb)), np.tile(vb, len(A)))
            v, ia, ib = v[ok], ia[ok], ib[ok]
            k = qkey(v)
            # within-chunk dedupe, then global
            k, first = np.unique(k, return_index=True)
            v, ia, ib = v[first], ia[first], ib[first]
            pos = np.searchsorted(allkeys, k)
            pos[pos >= len(allkeys)] = 0
            new = allkeys[pos] != k
            if new.any():
                parts_v.append(v[new])
                parts_p.append(np.stack([np.full(new.sum(), a, dtype=np.int32), ia[new], ib[new]], 1))
                allkeys = np.sort(np.concatenate([allkeys, k[new]]))
    if parts_v:
        V = np.concatenate(parts_v); P = np.concatenate(parts_p)
        # cross-chunk dedupe (a value may have been new in two chunks before allkeys updated) — handled since allkeys updated per chunk
    else:
        V = np.array([], dtype=dt); P = np.zeros((0, 3), np.int32)
    vals[n], par[n] = V, P
    counts.append(len(V)); catalan.append(math.comb(2 * n, n) // (n + 1))
    print(f"size {n:2d}: new {len(V):>10d}  catalan {catalan[-1]:>12d}  ratio {len(V)/catalan[-1]:.4f}  "
          f"cum {len(allkeys):>10d}  {time.time()-t0:6.1f}s", flush=True)


def witness(n, i):
    if n == 0: return "1"
    a, ia, ib = par[n][i]
    return f"e({witness(a, ia)},{witness(n-1-a, ib)})"


import sympy as sp
E, PI, I = sp.E, sp.pi, sp.I
targets_sym = {str(k): sp.Integer(k) for k in range(-6, 25)}
targets_sym.update({"1/2": sp.Rational(1,2), "1/3": sp.Rational(1,3), "2/3": sp.Rational(2,3), "3/2": sp.Rational(3,2),
    "e": E, "pi": PI, "sqrt2": sp.sqrt(2), "ln2": sp.log(2), "phi": (1+sp.sqrt(5))/2,
    "e^2": E**2, "1/e": 1/E, "e-1": E-1, "e-2": E-2, "e-3": E-3, "e-4": E-4, "sqrt(e)": sp.sqrt(E), "ln pi": sp.log(PI),
    "2e": 2*E, "e/2": E/2, "e^e": E**E, "ln(e-1)": sp.log(E-1), "-e": -E, "1/4": sp.Rational(1,4), "e^3": E**3})
if CPLX:
    targets_sym.update({"i": I, "-i": -I, "i*pi": I*PI, "-i*pi": -I*PI, "1+i": 1+I, "e^i": sp.exp(I), "2i": 2*I, "e^(ipi/2)?": sp.exp(I*PI/2)})
targets = {k: complex(sp.N(v, 20)) for k, v in targets_sym.items()}
res = {}
for name, tv in targets.items():
    if not CPLX and abs(tv.imag) > 0: res[name] = None; continue
    tv = np.array([tv], dtype=np.complex128) if CPLX else np.array([tv.real])
    k = qkey(tv)[0]
    hit = None
    for n in range(NMAX + 1):
        kn = qkey(vals[n])
        idx = np.nonzero(kn == k)[0]
        if len(idx):
            hit = (n, witness(n, int(idx[0]))); break
    res[name] = hit
    print(f"{name:>8s}: " + (f"size {hit[0]:2d}  {hit[1]}" if hit else f"not found <= {NMAX}"))

json.dump({"branch": BRANCH, "nmax": NMAX, "counts": counts, "catalan": catalan,
           "targets": res}, open(f"/home/claude/eml_complexity_{BRANCH}_{NMAX}.json", "w"), indent=1)

# --- self-check: recompute each stored value from its parents ---
bad = 0
for n in range(1, NMAX + 1):
    P = par[n]
    if len(P) == 0: continue
    a = P[:, 0]; ia = P[:, 1]; ib = P[:, 2]
    for aa in np.unique(a):
        m = a == aa
        v, ok = eml_vec(vals[aa][ia[m]], vals[n - 1 - aa][ib[m]])
        d = np.abs(v - vals[n][m]) / np.maximum(1, np.abs(vals[n][m]))
        bad += int((~ok).sum() + (d > 1e-9).sum())
print("parent self-check mismatches:", bad)

import sympy as sp
def build(w):
    return eval(w.replace("e(", "eml("), {"eml": lambda a, b: sp.exp(a) - sp.log(b)})
ver = {}
for name, hit in res.items():
    if not hit: continue
    x = build(hit[1]); tv = targets_sym[name]
    try:
        d = complex(sp.N(x - tv, 30))
        ok = abs(d) < 1e-20
        sym = sp.simplify(x) if ok else None
    except Exception as ex:
        ok, sym = False, str(ex)[:40]
    ver[name] = ok
    print(f"verify {name:>8s} size {hit[0]:2d}: {'OK' if ok else 'FAIL'}  {sym}")
json.dump({"branch": BRANCH, "nmax": NMAX, "counts": counts, "catalan": catalan,
           "targets": res, "verified": ver}, open(f"/home/claude/eml_complexity_{BRANCH}_{NMAX}.json", "w"), indent=1)
