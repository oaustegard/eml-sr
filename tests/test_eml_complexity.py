"""Small-size regression for benchmarks/eml_complexity.py and its join module.

Runs in RAM at sizes the claude.ai container could reach in seconds, checks
the distinct-per-size sequence and known minimal sizes from PR #69, and checks
that the inverse joins recover trees whose root children are enumerated.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH = os.path.join(ROOT, "benchmarks")
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)

import eml_complexity as ec  # noqa: E402
from eml_complexity_join import root_join, two_level_join  # noqa: E402

REAL_COUNTS = [1, 1, 2, 5, 10, 27, 73, 197, 545, 1518, 4326, 12455]
CPLX_COUNTS = [1, 1, 2, 5, 10, 28, 79, 228, 676, 2034, 6242, 19388]
REAL_SIZES = {"0": 3, "-1": 8, "2": 9, "1/e": 9, "e^2": 10, "-e": 11}
CPLX_SIZES = {"0": 3, "-1": 8, "2": 9, "-i*pi": 11}


def _expand(branch, nmax):
    ec.set_branch(branch)
    levels, info = ec.expand(branch, nmax, spill_dir=None, chunk=20_000_000,
                             ram_gb=4, resume=False, log=lambda *a, **k: None)
    return levels, info


def _target(levels, name):
    sym = ec.targets(levels.branch)[name]
    tv = complex(__import__("sympy").N(sym, 20))
    arr = np.array([tv], dtype=np.complex128) if ec.CPLX else np.array([tv.real])
    return tv, int(ec.qkey(arr)[0]), sym


@pytest.mark.parametrize("branch,counts,sizes", [
    ("real", REAL_COUNTS, REAL_SIZES),
    ("complex", CPLX_COUNTS, CPLX_SIZES),
])
def test_frontier_matches_pr69(branch, counts, sizes):
    levels, info = _expand(branch, len(counts) - 1)
    # Levels are stored sorted by key, so pair traversal order differs from the
    # PR #69 script and a different float can represent an 11-digit class; a
    # child near a rounding boundary then lands in a neighbouring class. The
    # complex branch drifts by one value at size 11 for that reason; the real
    # branch drifts by ~1e-5 of the level at sizes 13-17 (50 at size 17).
    assert info["counts"][:11] == counts[:11]
    for got, want in zip(info["counts"][11:], counts[11:]):
        assert abs(got - want) <= max(2, 1e-4 * want), (got, want)
    assert ec.selfcheck(levels, log=lambda *a, **k: None) == 0
    for name, want in sizes.items():
        tv, _, sym = _target(levels, name)
        hit = ec.frontier_lookup(levels, tv)
        assert hit is not None, name
        assert hit[0] == want, (name, hit)
        assert ec.verify(hit[1], sym), (name, hit)


def test_lookup_keys_land_in_one_level():
    levels, _ = _expand("real", 8)
    allk = np.concatenate(levels.keys)
    assert len(np.unique(allk)) == len(allk)
    size, pos = levels.lookup(np.sort(allk))
    assert (size >= 0).all()


def test_root_join_reaches_past_frontier():
    # -1 = e(<size 5>, <size 2>): both children enumerated at N=6, root size 8 > N+1
    levels, _ = _expand("real", 6)
    tv, tk, sym = _target(levels, "-1")
    assert ec.frontier_lookup(levels, tv) is None
    hit = root_join(levels, tv, tk, log=lambda *a, **k: None)
    assert hit is not None
    assert hit[0] == 8
    assert ec.verify(hit[1], sym)


def test_two_level_join_form2():
    # -1 = e(e(1, <size 4>), <size 2>): c=1, d at level 4, b at level 2, N=4
    levels, _ = _expand("real", 4)
    tv, tk, sym = _target(levels, "-1")
    assert root_join(levels, tv, tk, log=lambda *a, **k: None) is None
    hit = two_level_join(levels, tv, tk, K=2, budget=1e9, log=lambda *a, **k: None)
    assert hit not in (None, "skipped")
    assert hit[0] == 8
    assert ec.verify(hit[1], sym)


def test_two_level_join_budget():
    levels, _ = _expand("real", 4)
    tv, tk, _ = _target(levels, "-1")
    assert two_level_join(levels, tv, tk, K=2, budget=1, log=lambda *a, **k: None) == "skipped"


def test_mult_flag_on_product_tree():
    # x*y = exp(ln x + ln y); build e^2 * e = e^3 as exp(ln(e^2) + ln(e)) is not
    # what the enumerator emits, so check the heuristic on a hand-built string
    # and on a witness that has no product structure.
    assert ec.mult_shaped("e(1,e(e(1,1),1))") is False        # 0
    prod = "e(e(1,e(e(1,e(e(1,1),1)),1)),1)"                    # exp(ln(e^e)) = e^e: exp of a log, no product
    assert ec.mult_shaped(prod) in (False, None)


def test_cli_join_runs_on_the_complex_branch(tmp_path):
    # As a script the engine is __main__; the join module must see the same
    # module (branch globals included), or complex lookups silently miss.
    import json
    import subprocess
    script = os.path.join(BENCH, "eml_complexity.py")
    out = tmp_path / "out"
    subprocess.run([sys.executable, script, "complex", "6", "--spill", "none",
                    "--out", str(out), "--join2", "0", "--no-selfcheck"],
                   check=True, capture_output=True, text=True, timeout=600)
    d = json.load(open(out / "complex_6.json"))
    hit = d["targets"]["-1"]
    assert hit is not None and hit["source"] == "join" and hit["size"] == 8 and hit["verified"]


def test_resume_stops_at_requested_size(tmp_path):
    spill = str(tmp_path / "spill")
    ec.set_branch("real")
    ec.expand("real", 6, spill_dir=spill, chunk=2_000_000, ram_gb=4, resume=False, log=lambda *a, **k: None)
    levels, info = ec.expand("real", 4, spill_dir=spill, chunk=2_000_000, ram_gb=4, resume=True, log=lambda *a, **k: None)
    assert levels.N == 4
    assert info["counts"] == REAL_COUNTS[:5]
