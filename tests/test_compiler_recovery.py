"""Compiler-backed recovery tests (issue #32).

The existing recovery tests in ``tests/test_eml_sr.py`` and
``tests/test_e2e_multivariate.py`` hand-craft target ``y`` arrays with
``np.exp(x)``, ``np.log(x)``, etc., then check that ``discover()``
names the same formula. That pattern assumes the test author and the
searcher agree on what e.g. ``exp(x)`` means numerically — a fair
assumption, but it doesn't exercise the principled claim that the
paper makes:

    for any formula expressible as an EML tree, the searcher should
    find a tree of comparable (or smaller) size, within machine
    epsilon RMSE.

Now that ``eml_compiler.compile_expr()`` exists (issue #30/#31), we
have ground-truth EML trees for arbitrary elementary expressions. This
module uses those trees to generate target values — so the test spec
IS the formula string — and compares the discovered tree to the
reference tree by size.

Design notes
------------
- ``discover()`` returns ``result["depth"]``, the shallowest exact
  depth on the ladder. We compare that against ``tree_depth()`` of
  the reference tree from the compiler. The searcher should never
  return a *deeper* tree than the compiler's bootstrap, because
  ``discover()`` walks the ladder bottom-up and returns at the first
  exact fit. (An earlier draft of this file tried to compare node
  counts via ``tree_size()`` round-trip, but the compiler's ``sub``
  bootstrap expands ``exp(x) - ln(y)`` into a 19-node tree while the
  atom ``eml(x, y)`` is 3 nodes — same function, different
  representations, incomparable node counts. Depth is the principled
  metric.)
- We gate on ``result.get("exact", True)`` — ``discover()`` omits the
  ``"exact"`` key when it hits the success threshold, and sets it to
  ``False`` otherwise.
- Settings (``max_depth``, ``n_tries``) are tuned per formula depth to
  keep wall-clock in CI reasonable. The paper-recovery sweep belongs
  in issue #33 (PySR benchmark).
- Non-goal: trig. The compiler refuses ``sin``/``cos``/``tan`` per
  paper §4.1.
"""

from __future__ import annotations

import numpy as np
import pytest

from eml_compiler import compile_expr, tree_depth
from eml_sr import discover

from tests._helpers import eval_formula_multivariate, eval_formula_univariate


# ─── Univariate recovery ───────────────────────────────────────────
#
# Catalog covers atoms (``x``, ``1``), depth-1 EML primitives
# (``exp(x)``, ``e``), and a depth-1 subtract (``exp(x) - x``). Each
# should recover exactly on the configured ladder. Deeper primitives
# (``ln(x)`` → depth 3, ``1/x`` → depth 14 in the bootstrap) are
# covered in ``tests/test_eml_sr.py`` and ``tests/test_eml_compiler.py``
# under ``@pytest.mark.slow`` — no point duplicating them here.

UNIVARIATE_CATALOG = [
    # (formula, max_depth, n_tries) — settings scale with expected difficulty
    ("x",            1, 2),
    ("1",            1, 2),
    ("exp(x)",       2, 4),
    ("e",            1, 2),
    ("exp(x) - x",   2, 4),
]


@pytest.mark.parametrize("formula, max_depth, n_tries", UNIVARIATE_CATALOG)
def test_recovery_univariate(formula, max_depth, n_tries):
    """Compiler-generated targets must be recovered exactly by the searcher.

    Reference tree comes from ``compile_expr(formula)``; the target
    array is ``eval_eml`` at each input point. ``discover()`` runs on
    that target and must find an exact formula on the ladder. The
    discovered depth must not exceed the reference depth (the searcher
    should never need a deeper tree than the compiler's bootstrap to
    represent an expression the compiler can build).
    """
    ref_tree = compile_expr(formula)
    ref_depth = tree_depth(ref_tree)

    x = np.linspace(0.5, 3.0, 40)
    y = eval_formula_univariate(formula, x)

    result = discover(x, y, max_depth=max_depth, n_tries=n_tries, verbose=False)
    assert result is not None, f"discover() returned None on {formula!r}"
    assert result.get("exact", True), (
        f"{formula!r}: searcher did not hit success threshold "
        f"(snap_rmse={result['snap_rmse']:.2e}, expr={result['expr']!r})"
    )

    assert result["depth"] <= ref_depth, (
        f"{formula!r}: discovered depth {result['depth']} exceeds "
        f"reference depth {ref_depth} (expr={result['expr']!r})"
    )


# ─── Multivariate catalog (deliverable B) ──────────────────────────
#
# Keep this conservative: the depth-1 EML atoms ``eml(x1, x2)``,
# ``eml(x2, x1)``, and an identity ``eml(x1, 1) = exp(x1) - 0`` cover
# the axes the searcher needs to exercise at n_vars=2. Deeper
# multivariate targets (``x + y``, ``x * y`` — 21 and 35 nodes
# respectively per ``test_primitive_sizes_and_values``) are out of
# scope for a recovery test at the depths we can afford in CI; the
# paper-rate sweep lives in issue #33.

MULTIVARIATE_CATALOG = [
    # (formula, n_vars, var_names, max_depth, n_tries)
    ("eml(x1, x2)",       2, ("x1", "x2"), 1, 4),
    ("eml(x2, x1)",       2, ("x1", "x2"), 1, 4),
]


@pytest.mark.parametrize(
    "formula, n_vars, var_names, max_depth, n_tries", MULTIVARIATE_CATALOG
)
def test_recovery_multivariate(formula, n_vars, var_names, max_depth, n_tries):
    """Multivariate compiler targets must be recovered exactly on the ladder.

    The depth-1 ``eml(xi, xj)`` atoms are the smallest multivariate
    trees expressible in the basis and should be found immediately.
    """
    ref_tree = compile_expr(formula, variables=var_names)
    ref_depth = tree_depth(ref_tree)

    rng = np.random.default_rng(0)
    X = rng.uniform(0.5, 3.0, size=(40, n_vars))
    y = eval_formula_multivariate(formula, X, var_names)

    result = discover(X, y, max_depth=max_depth, n_tries=n_tries, verbose=False)
    assert result is not None, f"discover() returned None on {formula!r}"
    assert result["n_vars"] == n_vars, (
        f"{formula!r}: expected n_vars={n_vars}, got {result['n_vars']}"
    )
    assert result.get("exact", True), (
        f"{formula!r}: searcher did not hit success threshold "
        f"(snap_rmse={result['snap_rmse']:.2e}, expr={result['expr']!r})"
    )

    assert result["depth"] <= ref_depth, (
        f"{formula!r}: discovered depth {result['depth']} exceeds "
        f"reference depth {ref_depth} (expr={result['expr']!r})"
    )


# ─── Harness sanity ────────────────────────────────────────────────


def test_compiler_target_is_numerically_consistent():
    """Compiler-generated targets for an identity formula equal the input.

    A sanity check that the eval_eml bindings wiring is correct — if
    this breaks, every recovery test above becomes meaningless.
    """
    x = np.linspace(0.5, 3.0, 20)
    y_compiler = eval_formula_univariate("x", x)
    np.testing.assert_allclose(y_compiler, x, atol=1e-12)


def test_round_trip_discovered_expr_is_parseable():
    """``discover()``'s ``expr`` output must round-trip through ``compile_expr``.

    The searcher's simplifier can produce strings like ``"exp(x)"``,
    ``"eml(x, 1)"``, or ``"(exp(x) - ln(y))"`` — all of which should
    be parseable by the compiler. Breaking this contract would
    invalidate any downstream tool that wants to re-compile discovered
    expressions (e.g. size/depth analysis, simplification, export).
    """
    x = np.linspace(0.5, 3.0, 20)
    y = x.copy()
    result = discover(x, y, max_depth=1, n_tries=2, verbose=False)
    assert result is not None
    # Must not raise:
    compile_expr(result["expr"])
