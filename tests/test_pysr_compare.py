"""Smoke test for the PySR head-to-head benchmark (issue #33).

Skipped unless ``PYSR_ENABLED=1`` is set and :mod:`pysr` imports cleanly.
This exists to catch bit-rot in the benchmark wiring — the full run is
too slow for CI (and PySR needs a Julia toolchain, which we don't ship).

Run locally with::

    PYSR_ENABLED=1 pytest -m pysr tests/test_pysr_compare.py -v
"""
from __future__ import annotations

import math
import os

import pytest

pytestmark = pytest.mark.pysr


def _pysr_import_ok() -> bool:
    if os.environ.get("PYSR_ENABLED", "") != "1":
        return False
    try:
        import pysr  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _pysr_import_ok(),
    reason="PYSR_ENABLED=1 not set or pysr not installed",
)
def test_pysr_compare_smoke_one_target():
    """One trivial target (exp(x)) runs through both engines without crashing.

    We don't assert PySR recovers anything in particular — PySR behavior
    is version-dependent and this test is a smoke test, not a ground
    truth. We only assert the benchmark harness returns a well-formed
    result dict for both engines.
    """
    from benchmarks.pysr_compare import (
        FEYNMAN_PROBLEMS, _run_eml_sr, _run_pysr,
    )

    by_id = {p.feynman_id: p for p in FEYNMAN_PROBLEMS}
    prob = by_id["eml.exp"]

    # eml-sr side — already covered by other tests but verify the
    # pysr_compare wrapper contract.
    r_eml = _run_eml_sr(prob, method="discover", max_depth=1, n_tries=4,
                        threshold=1e-6)
    assert r_eml.engine == "eml-sr/discover"
    assert math.isfinite(r_eml.wall_time) and r_eml.wall_time >= 0
    assert isinstance(r_eml.expr, str)
    assert isinstance(r_eml.exact, bool)

    # PySR side — minimal niterations through the permissive config path
    # takes ~20s cold, much less warm. A smoke test, not a tuning target.
    r_pysr = _run_pysr(prob, config="default", threshold=1e-6)
    assert r_pysr.engine == "pysr/default"
    assert math.isfinite(r_pysr.wall_time) and r_pysr.wall_time >= 0
    assert isinstance(r_pysr.expr, str)
    assert isinstance(r_pysr.exact, bool)
