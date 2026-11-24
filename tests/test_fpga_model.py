"""Checks for the fixed-point EML cell (fpga/).

The model tests always run.  The RTL test runs only when Icarus Verilog
(iverilog + vvp) is on PATH; it streams vectors through rtl/eml_cell.v and
requires a bit-exact match with the model.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

FPGA = Path(__file__).resolve().parent.parent / "fpga"
sys.path.insert(0, str(FPGA))

import eml_model as M
from gen_tree import TREES, eval_tree_q

from benchmarks.perm_det_eml import parse


def test_cell_error_on_analog_box_is_about_one_lsb():
    rng = np.random.default_rng(0)
    xq = M.to_q(rng.uniform(-1, 1, 100_000))
    yq = M.to_q(rng.uniform(0.1, 2, 100_000))
    out, inv = M.eml_q(xq, yq)
    assert not inv.any()
    ref = np.exp(M.from_q(xq)) - np.log(M.from_q(yq))
    assert np.max(np.abs(M.from_q(out) - ref)) < 1.5 / M.ONE


def test_exp_relative_error_large_outputs():
    x = np.linspace(4.2, 10.39, 50_000)
    xq = M.to_q(x)
    e = M.from_q(M.exp_q(xq))
    ref = np.exp(M.from_q(xq))
    assert np.max(np.abs(e - ref) / ref) < 1e-6


def test_out_of_range_behaviour():
    out, inv = M.eml_q(np.array([0, 0, 20 * M.ONE]), np.array([0, -M.ONE, M.ONE]))
    assert inv.tolist() == [True, True, False]
    assert out.tolist() == [M.INT_MAX] * 3


@pytest.mark.parametrize("name,fn", [("mul", np.multiply), ("add", np.add),
                                     ("sub", np.subtract)])
def test_trees_compute_their_targets(name, fn):
    a = M.to_q(np.array([0.5, 1.25, 2.0, 0.3]))
    b = M.to_q(np.array([0.75, 0.6, 0.4, 1.1]))
    out, inv = eval_tree_q(parse(TREES[name]), {"x0": a, "x1": b})
    assert not inv.any()
    exact = fn(M.from_q(a), M.from_q(b))
    assert np.max(np.abs(M.from_q(out) - exact)) < 1e-3


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog not installed")
def test_rtl_bit_exact(tmp_path):
    from tb.make_vectors import vectors

    x, y, out, inv = vectors(5_000, seed=3)
    m = 0xFFFFFFFF
    (tmp_path / "vectors.txt").write_text("".join(
        f"{int(a) & m:08x} {int(b) & m:08x} {int(c) & m:08x} {int(d)}\n"
        for a, b, c, d in zip(x, y, out, inv)))
    for h in (FPGA / "rtl").glob("*.hex"):
        shutil.copy(h, tmp_path)
    subprocess.run(["iverilog", "-g2012", "-o", str(tmp_path / "sim"),
                    str(FPGA / "rtl" / "eml_cell.v"), str(FPGA / "tb" / "tb_eml_cell.v")],
                   check=True)
    r = subprocess.run(["vvp", "-n", str(tmp_path / "sim")], cwd=tmp_path,
                       capture_output=True, text=True, check=True)
    assert f"vectors={x.size} mismatches=0" in r.stdout
