"""benchmarks/compiler_vs_minimal.py: every chosen compilation evaluates to its constant."""
import os
import sys

import sympy as sp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks.compiler_vs_minimal import TABLE, best_compilation, run  # noqa: E402


def test_every_constant_has_a_strict_compilation():
    rows = {r["constant"]: r for r in run()}
    assert set(rows) == set(TABLE)
    assert all(r["compiler"] is not None for r in rows.values())


def test_known_overheads():
    rows = {r["constant"]: r for r in run()}
    assert rows["0"]["compiler"] == 3          # ln(1) is the enumerator's 3-node zero
    assert rows["e-1"]["compiler"] == 6        # generic subtraction; the minimum is eml(1, eml(1, 1)) at 2
    assert rows["2"]["compiler"] == 13         # 1+1 through neg/sub; the minimum is 9
    assert rows["-e"]["compiler"] == 9         # ln(0) = -inf route; the finite minimum is 11
    assert rows["-e"]["nonfinite_intermediates"] == 3
    assert rows["2"]["nonfinite_intermediates"] == 3


def test_best_compilation_rejects_wrong_values():
    assert best_compilation(["1+1"], sp.Integer(3)) is None
