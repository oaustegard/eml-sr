"""Pytest config for eml-sr tests."""
import os
import sys

# Ensure the repo root (where eml_sr.py lives) is importable.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks multi-seed recovery tests that run a full training ladder",
    )
    config.addinivalue_line(
        "markers",
        "pysr: head-to-head PySR benchmark smoke tests; require PYSR_ENABLED=1 "
        "and a working pysr + julia install. Skipped by default "
        "(pysr is a heavy dependency). See benchmarks/pysr_compare.py.",
    )
