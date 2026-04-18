#!/usr/bin/env python3
"""analog/spice/cascade_driver.py - depth-2 cascade validation.

Runs cascade.cir over the valid input envelope, compares measured
V_out against the ideal depth-2 eml cascade, and cross-checks the
measured cascade RMSE against ``analog.noise_sim.simulate`` running
the same depth-2 tree with per-cell noise calibrated from the single-
cell DC transfer results.

Go/no-go:  SPICE cascade RMSE must agree with noise_sim's depth-2
prediction within +-30%.  This is the headline check in issue #35.

Outputs (under ``analog/spice/results/``):
    cascade_dc.csv          per-point (va, vb, vc, v_int, v_out, err)
    cascade_validation.csv  single-row summary + verdict
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SPICE_DIR = Path(__file__).parent
RESULTS_DIR = SPICE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
REPO_ROOT = SPICE_DIR.parent.parent
ANALOG_DIR = SPICE_DIR.parent

sys.path.insert(0, str(SPICE_DIR))
sys.path.insert(0, str(ANALOG_DIR))
sys.path.insert(0, str(REPO_ROOT))

from driver import run_ngspice_wrdata  # noqa: E402
from noise_sim import simulate, AdditiveGaussian, MultiplicativeGaussian  # noqa: E402
from eml_compiler import compile_expr  # noqa: E402


def ideal_cascade(va, vb, vc):
    """Depth-2 eml cascade: eml(eml(va, vb), vc)."""
    inner = np.exp(va) - np.log(vb)
    return np.exp(inner) - np.log(vc)


def run_cascade_spice(va_pts: int = 8, vb_pts: int = 7,
                      vc_vals=(0.5, 1.0, 1.5)) -> pd.DataFrame:
    """Sweep (va, vb) via ngspice .dc at each vc, filter to valid envelope."""
    records = []
    va_step = 1.4 / (va_pts - 1)       # va in [-1.0, 0.4]
    vb_step = 2.4 / (vb_pts - 1)       # vb in [0.1, 2.5]
    vb_vals_grid = np.linspace(0.1, 2.5, vb_pts)

    for vc in vc_vals:
        body = f""".include cell_opamp_sub.cir
Vcc vcc 0 15
Vee vee 0 -15
Va va 0 0
Vb vb 0 1
Vc vc 0 {vc}
Xcell1 va vb vint vcc vee eml_cell_opamp
Xcell2 vint vc vout vcc vee eml_cell_opamp
"""
        data = run_ngspice_wrdata(
            body,
            analysis=f"dc Va -1.0 0.4 {va_step:.6f} "
                     f"Vb 0.1 2.5 {vb_step:.6f}",
            nodes=["vout", "vint"],
        )
        if data.size == 0:
            continue
        for i in range(len(data)):
            va = float(data[i, 0])
            vout = float(data[i, 1])
            vint = float(data[i, 2])
            vb_idx = i // va_pts
            vb = float(vb_vals_grid[min(vb_idx, vb_pts - 1)])
            ideal_int = float(np.exp(va) - np.log(vb))
            # Envelope: intermediate must sit safely inside cell 2's input range
            if not (-0.9 < ideal_int < 0.9):
                continue
            ideal_out = float(ideal_cascade(va, vb, vc))
            if not np.isfinite(ideal_out):
                continue
            records.append({
                "va": va, "vb": vb, "vc": vc,
                "v_int": vint, "v_out": vout,
                "ideal_int": ideal_int,
                "ideal_out": ideal_out,
                "err": vout - ideal_out,
                "int_err": vint - ideal_int,
            })
    return pd.DataFrame(records)


def main():
    print("Running SPICE cascade DC sweep...")
    df = run_cascade_spice()
    if len(df) == 0:
        print("  no valid cascade points captured — check cascade.cir/envelope")
        return
    df.to_csv(RESULTS_DIR / "cascade_dc.csv", index=False)
    spice_rmse = float(np.sqrt(np.mean(df["err"] ** 2)))
    spice_int_rmse = float(np.sqrt(np.mean(df["int_err"] ** 2)))
    max_err = float(df["err"].abs().max())
    print(f"  {len(df)} valid points")
    print(f"  SPICE cascade RMSE vs ideal   : {spice_rmse:.4g} V")
    print(f"  SPICE intermediate RMSE       : {spice_int_rmse:.4g} V")
    print(f"  max |err|                     : {max_err:.4g} V")

    # Calibrate per-cell noise from single-cell DC RMSE
    sc_path = RESULTS_DIR / "opamp_dc.csv"
    if sc_path.exists():
        sc_df = pd.read_csv(sc_path)
        single_cell_rmse = float(np.sqrt(np.mean(sc_df["err"] ** 2)))
    else:
        print("  WARNING: opamp_dc.csv missing, using fallback sigma=0.010")
        single_cell_rmse = 0.010

    print(f"  Single-cell DC RMSE (opamp)   : {single_cell_rmse:.4g} V")

    # Build depth-2 tree using eml_compiler
    tree = compile_expr(
        "eml(eml(va, vb), vc)",
        variables=["va", "vb", "vc"],
    )
    x_samples = {
        "va": df["va"].to_numpy(),
        "vb": df["vb"].to_numpy(),
        "vc": df["vc"].to_numpy(),
    }

    # Primary prediction: additive Gaussian calibrated to single-cell RMSE
    noise_model = AdditiveGaussian(sigma=single_cell_rmse)
    result = simulate(tree, noise_model, x_samples, n_trials=500, seed=0)
    predicted_rmse = float(result["rmse"])
    predicted_std = float(result["rmse_std"])
    bits = float(result["bits_of_precision"])
    print(f"  noise_sim (AdditiveGaussian sigma={single_cell_rmse:.3g}):")
    print(f"    predicted depth-2 RMSE      : {predicted_rmse:.4g} +- "
          f"{predicted_std:.4g} V")
    print(f"    bits_of_precision           : {bits:.2f}")

    # Secondary check: multiplicative Gaussian as a floor on tolerance
    mult_model = MultiplicativeGaussian(sigma=0.01)
    mult_res = simulate(tree, mult_model, x_samples, n_trials=500, seed=1)
    print(f"  noise_sim (MultiplicativeGaussian sigma=1%):")
    print(f"    predicted depth-2 RMSE      : {float(mult_res['rmse']):.4g} V")

    # Go/no-go
    ratio = spice_rmse / predicted_rmse if predicted_rmse > 0 else float("inf")
    margin = abs(ratio - 1.0)
    verdict = "PASS" if margin <= 0.30 else "FAIL"
    print(f"  Ratio SPICE / noise_sim       : {ratio:.3f}")
    print(f"  Agreement margin              : {margin*100:.1f}%")
    print(f"  Verdict (+-30%)               : {verdict}")

    summary = {
        "n_points": len(df),
        "spice_cascade_rmse": spice_rmse,
        "spice_intermediate_rmse": spice_int_rmse,
        "spice_max_abs_err": max_err,
        "single_cell_rmse": single_cell_rmse,
        "noise_sim_additive_rmse": predicted_rmse,
        "noise_sim_additive_std": predicted_std,
        "noise_sim_bits": bits,
        "noise_sim_mult_rmse": float(mult_res["rmse"]),
        "ratio_spice_over_sim": ratio,
        "margin_pct": margin * 100,
        "verdict_within_30pct": verdict,
    }
    pd.DataFrame([summary]).to_csv(
        RESULTS_DIR / "cascade_validation.csv", index=False
    )
    print(f"  wrote {RESULTS_DIR / 'cascade_validation.csv'}")


if __name__ == "__main__":
    main()
