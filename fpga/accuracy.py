"""Accuracy of the bit-exact model (== RTL) against float64; results/accuracy.json.

Errors are measured against the exact function of the *quantized* inputs, so
they are the cell's own error, not the Q16.16 input rounding.

    python fpga/accuracy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eml_model as M
from gen_tree import TREES, eval_tree_q

from benchmarks.perm_det_eml import parse

LSB = 1.0 / M.ONE


def stats(err: np.ndarray) -> dict:
    a = np.abs(err)
    return {"n": int(a.size), "rmse": float(np.sqrt(np.mean(err**2))),
            "mean_abs": float(a.mean()), "max_abs": float(a.max()),
            "max_abs_lsb": float(a.max() / LSB), "rmse_lsb": float(np.sqrt(np.mean(err**2)) / LSB)}


def eval_flags(t, env):
    """Tree eval that also flags any node whose exp term or result saturated."""
    if isinstance(t, str):
        v = np.full_like(next(iter(env.values())), M.ONE) if t == "1" else env[t]
        z = np.zeros(v.shape, bool)
        return v, z, z
    a, ia, sa = eval_flags(t[0], env)
    b, ib, sb = eval_flags(t[1], env)
    v, inv = M.eml_q(a, b)
    lg, _ = M.ln_q(b)
    sat = (M.exp_q(a) >= M.INT_MAX) | (M.exp_q(a) - lg > M.INT_MAX) | (M.exp_q(a) - lg < M.INT_MIN)
    return v, ia | ib | inv, sa | sb | sat


def cell(xs, ys) -> np.ndarray:
    xq, yq = M.to_q(xs), M.to_q(ys)
    out, inv = M.eml_q(xq, yq)
    assert not inv.any()
    ref = np.exp(M.from_q(xq)) - np.log(M.from_q(yq))
    return M.from_q(out) - ref, ref


def main() -> None:
    rng = np.random.default_rng(7)
    res: dict = {}

    # 1. the analog study's grid (analog/spice/README.md: 20 x 41 over
    #    V_x in [-1, 1], V_y in [0.1, 2]) plus a dense random fill
    gx, gy = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(0.1, 2, 41))
    err, _ = cell(gx.ravel(), gy.ravel())
    res["cell_analog_grid_20x41"] = stats(err)
    err, _ = cell(rng.uniform(-1, 1, 10**6), rng.uniform(0.1, 2, 10**6))
    res["cell_analog_box_random_1e6"] = stats(err)

    # 2. whole representable domain: exp(x) < 32768, y in [2^-16, 32767]
    n = 2 * 10**6
    xs = rng.uniform(-11.0, np.log(32767.0) - 1e-3, n)
    ys = np.exp(rng.uniform(np.log(2.0**-16), np.log(32767.0), n))
    err, ref = cell(xs, ys)
    res["cell_full_domain_2e6"] = stats(err)
    big = np.abs(ref) > 1.0
    res["cell_full_domain_rel_err_max_where_abs_out_gt_1"] = float(
        np.max(np.abs(err[big]) / np.abs(ref[big])))
    e_q = M.exp_q(M.to_q(xs))
    ex = np.exp(M.from_q(M.to_q(xs)))
    res["exp_term_rel_err_max_where_exp_gt_64"] = float(
        np.max(np.abs(M.from_q(e_q) - ex)[ex > 64] / ex[ex > 64]))
    lq, _ = M.ln_q(M.to_q(ys))
    res["ln_term"] = stats(M.from_q(lq) - np.log(M.from_q(M.to_q(ys))))

    # 3. trees vs native ops, positive inputs over [1e-2, 8]
    a = np.exp(rng.uniform(np.log(1e-2), np.log(8.0), n))
    b = np.exp(rng.uniform(np.log(1e-2), np.log(8.0), n))
    aq, bq = M.to_q(a), M.to_q(b)
    af, bf = M.from_q(aq), M.from_q(bq)
    native = {
        "mul": (M.from_q(np.round(af * bf * M.ONE)), af * bf),
        "add": (af + bf, af + bf),
        "sub": (af - bf, af - bf),
    }
    for name, w in TREES.items():
        out, inv, sat = eval_flags(parse(w), {"x0": aq, "x1": bq})
        out2, _ = eval_tree_q(parse(w), {"x0": aq, "x1": bq})
        assert np.array_equal(out, out2)
        nat, exact = native[name]
        ok = ~inv & ~sat
        e_t = M.from_q(out)[ok] - exact[ok]
        big = np.abs(exact[ok]) > 0.01
        rel = np.abs(e_t[big]) / np.abs(exact[ok][big])
        res[f"tree_{name}"] = {
            "domain": "x0, x1 log-uniform in [0.01, 8]",
            "invalid_fraction_real_branch": float(inv.mean()),
            "saturated_fraction": float((sat & ~inv).mean()),
            "eml_tree": stats(e_t),
            "native": stats(nat[ok] - exact[ok]),
            "rel_err_p50": float(np.median(rel)),
            "rel_err_p99": float(np.quantile(rel, 0.99)),
            "rel_err_max": float(rel.max()),
        }

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "accuracy.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
