"""Small, fast checks for benchmarks/perm_det_eml.py and perm_det_gadgets.py."""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks import perm_det_eml as P
from benchmarks import perm_det_gadgets as G


def _run(branch, nmax):
    eng = P.Engine(4, branch, 6, 1)
    eng.pending = []
    conv = (lambda a: a.astype(np.complex128)) if eng.cplx else (lambda a: np.real(a).astype(np.float64))
    tv = {k: conv(f(eng.X)) for k, (_, f) in P.TARGETS_2x2.items()}
    jt = {k: tv[k] for k in ("sub", "lnmul")}
    for n in range(1, nmax + 1):
        last = n == nmax
        if last:
            eng.build_index(n - 1)
        eng.expand(tv, stream=last, join_targets=jt if last else None, log=lambda *_: None)
    eng.run_joins(jt, 0, log=lambda *_: None)
    eng.verify_pending(jt)
    return eng


def test_real_counts_and_sub_minimal():
    eng = _run("real", 5)
    assert [s["distinct"] for s in eng.stats] == [25, 200, 1443, 10195, 72616]
    hits = eng.hits["sub"]
    assert min(s for s, _ in hits) == 5


def test_root_join_reaches_one_past_frontier():
    # lnmul is minimal at 7; enumerating to 6 and joining must find it at 7
    eng = _run("real", 6)
    sizes = [s for s, _ in eng.hits.get("lnmul", [])]
    assert sizes and min(sizes) == 7


def test_gadget_trees_are_identities():
    tf = G.target_fns()
    st = G.sym_targets()
    rng = np.random.default_rng(0)
    for tn, cs in G.cands_2x2().items():
        nv, fn = tf[tn]
        X = rng.uniform(-3, 3, size=(500, nv))
        X = np.where(np.abs(X) < 0.05, 0.05, X)
        for w in cs.values():
            w = G.S(w)
            assert G.check(w, fn, nv, True, X)["exact"] == 500
            assert G.sympy_identity(w, nv, st[tn])
    assert P.size(P.parse(G.S(G.cands_2x2()["det2"]["sub(prod, prod)"]))) == 17
    assert P.size(P.parse(G.S(G.cands_2x2()["perm2"]["sum(prod, prod)"]))) == 25
