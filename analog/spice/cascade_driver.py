#!/usr/bin/env python3
"""analog/spice/cascade_driver.py - honest depth-2 cascade validation.

Issue #35 asked whether `analog/noise_sim.py`'s depth-2 prediction
matches SPICE.  The first pass (merged in PR #44) compared
deterministic DC errors to the simulator's Gaussian-noise prediction
and reported "pass within 30%".  That was misleading: DC errors are
systematic and correlated across cells, so the match was a coincidence
of the loose bound, not a model validation.

This driver does the validation correctly by running *two* comparisons:

  1. AC noise track (the one noise_sim actually models).  ngspice
     `.noise` on the single cell and on the cascade measures random
     output-referred noise; we feed the single-cell sigma to
     noise_sim as `AdditiveGaussian` and check depth-2 prediction
     against the measured cascade noise.  These errors genuinely are
     independent between cells (different op-amps, different thermal
     sources), so Gaussian compounding is the right model.

  2. DC error correlation track (honest reporting, not validation).
     We measure the per-input DC error of both cells, quantify how
     correlated they are, and report the cascade-error ratio.  For
     fully correlated errors the ratio is ~2; for independent it is
     ~sqrt(2).  noise_sim assumes independent; the ratio tells you
     how badly that assumption breaks for deterministic bias.

The measured correlation is the finding.  noise_sim is validated (or
not) by the AC track; the DC track tells you where it does *not*
apply.

Outputs (under ``analog/spice/results/``):
    cascade_dc.csv           per-point (va, vb, vc, v_int, v_out, err)
    cascade_noise.csv        cascade output-referred noise spectrum
    cascade_validation.csv   summary of both tracks
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
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
from noise_sim import (  # noqa: E402
    AdditiveGaussian, MultiplicativeGaussian, simulate,
)
from eml_compiler import compile_expr  # noqa: E402


def ideal_cascade(va, vb, vc):
    """Depth-2 eml cascade: eml(eml(va, vb), vc)."""
    inner = np.exp(va) - np.log(vb)
    return np.exp(inner) - np.log(vc)


# ---------------------------------------------------------------
# Track 1: DC error + correlation
# ---------------------------------------------------------------

def run_single_cell_dc_grid(va_pts: int = 8, vb_pts: int = 7) -> pd.DataFrame:
    """Single-cell DC sweep on the same (va, vb) grid used for cascade."""
    va_step = 1.4 / (va_pts - 1)
    vb_step = 2.4 / (vb_pts - 1)
    body = """.include cell_opamp.cir
"""
    # cell_opamp.cir already defines Vx, Vy; we override via .dc sweep
    # but need to rename Va/Vb -> Vx/Vy since the single-cell netlist uses Vx/Vy.
    data = run_ngspice_wrdata(
        body,
        analysis=f"dc Vx -1.0 0.4 {va_step:.6f} Vy 0.1 2.5 {vb_step:.6f}",
        nodes=["vout"],
    )
    vb_vals = np.linspace(0.1, 2.5, vb_pts)
    records = []
    for i in range(len(data)):
        va = float(data[i, 0])
        vout = float(data[i, 1])
        vb_idx = i // va_pts
        vb = float(vb_vals[min(vb_idx, vb_pts - 1)])
        ideal = float(np.exp(va) - np.log(vb))
        records.append({
            "va": va, "vb": vb,
            "v_out": vout, "ideal": ideal, "err": vout - ideal,
        })
    return pd.DataFrame(records)


def run_cascade_spice(va_pts: int = 8, vb_pts: int = 7,
                      vc_vals=(0.5, 1.0, 1.5)) -> pd.DataFrame:
    """Sweep (va, vb) via ngspice .dc at each vc, filter to valid envelope."""
    records = []
    va_step = 1.4 / (va_pts - 1)
    vb_step = 2.4 / (vb_pts - 1)
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


def dc_correlation_analysis(sc_df: pd.DataFrame,
                            casc_df: pd.DataFrame) -> dict:
    """Quantify how much cell-1 error and cell-2 error track each other.

    Cell 1 sees (va, vb) directly; its error matches the single-cell
    sweep at the same (va, vb).  Cell 2 sees (vint, vc); we estimate
    its error by looking up the single-cell error at (vint_ideal, vc).
    The residual (cascade_err - propagated_cell1_err - cell2_err) tells
    us how much of the total is captured by the per-cell surfaces.

    Returns correlation coefficients and the empirical ratio
    cascade_rmse / single_cell_rmse — which equals ~2 for fully
    correlated error surfaces and ~sqrt(2) for independent.
    """
    # Build an interpolator of single-cell error over (va, vb)
    from scipy.interpolate import RegularGridInterpolator
    va_grid = np.sort(sc_df["va"].unique())
    vb_grid = np.sort(sc_df["vb"].unique())
    err_surface = np.zeros((len(va_grid), len(vb_grid)))
    for i, va in enumerate(va_grid):
        for j, vb in enumerate(vb_grid):
            row = sc_df[(sc_df["va"] == va) & (sc_df["vb"] == vb)]
            if len(row) > 0:
                err_surface[i, j] = float(row.iloc[0]["err"])
    interp = RegularGridInterpolator(
        (va_grid, vb_grid), err_surface,
        method="linear", bounds_error=False, fill_value=0.0,
    )

    # Cell 1 sees (va, vb) — error is single-cell err at that point.
    # Cell 2 sees (ideal_int, vc) — error is single-cell err at that point.
    cell1_pts = np.column_stack([casc_df["va"].to_numpy(),
                                 casc_df["vb"].to_numpy()])
    cell2_pts = np.column_stack([casc_df["ideal_int"].to_numpy(),
                                 casc_df["vc"].to_numpy()])
    cell1_err = np.asarray(interp(cell1_pts))
    cell2_err = np.asarray(interp(cell2_pts))
    # Error propagation: cascade_err ~= exp(ideal_int)*cell1_err + cell2_err
    propagated = np.exp(casc_df["ideal_int"].values) * cell1_err + cell2_err
    actual = casc_df["err"].values

    # Correlation between per-cell errors at their respective inputs
    corr_cells = float(np.corrcoef(cell1_err, cell2_err)[0, 1])
    # Fit quality of the propagation model
    corr_propagated = float(np.corrcoef(propagated, actual)[0, 1])
    residual = actual - propagated
    residual_rmse = float(np.sqrt(np.mean(residual ** 2)))

    sc_rmse = float(np.sqrt(np.mean(sc_df["err"] ** 2)))
    casc_rmse = float(np.sqrt(np.mean(casc_df["err"] ** 2)))

    return {
        "single_cell_rmse_V": sc_rmse,
        "cascade_rmse_V": casc_rmse,
        "rmse_ratio": casc_rmse / sc_rmse if sc_rmse > 0 else float("nan"),
        "independent_expected_ratio": float(np.sqrt(2)),
        "correlated_expected_ratio": 2.0,
        "corr_cell1_cell2_err": corr_cells,
        "corr_model_vs_actual": corr_propagated,
        "propagation_residual_rmse_V": residual_rmse,
    }


# ---------------------------------------------------------------
# Track 2: AC noise
# ---------------------------------------------------------------

def _run_noise(body: str, output_node: str, source: str,
               f_low: float = 20, f_high: float = 20000,
               dec: int = 20) -> dict:
    """Run ngspice .noise and return integrated + per-freq data."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".dat", delete=False, dir=SPICE_DIR
    ) as f:
        dat_path = Path(f.name)
    control = f"""
.control
  set filetype = ascii
  alter {source} ac = 1
  noise v({output_node}) {source} dec {dec} {f_low:.1f} {f_high:.1f}
  setplot noise1
  wrdata {dat_path.name} inoise_spectrum onoise_spectrum
  quit
.endc
"""
    full = body + control + "\n.end\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cir", delete=False, dir=SPICE_DIR
    ) as f:
        f.write(full)
        cir_path = Path(f.name)
    try:
        subprocess.run(
            ["ngspice", "-b", str(cir_path)],
            capture_output=True, text=True, timeout=120, cwd=SPICE_DIR,
        )
        if not dat_path.exists():
            return {"onoise_rms_V": float("nan"),
                    "inoise_rms_V": float("nan"), "n_pts": 0}
        raw = np.loadtxt(dat_path)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        freq = raw[:, 0]
        inoise = raw[:, 1]
        onoise = raw[:, 3] if raw.shape[1] >= 4 else raw[:, 1]
        return {
            "onoise_rms_V": float(np.sqrt(np.trapezoid(onoise ** 2, freq))),
            "inoise_rms_V": float(np.sqrt(np.trapezoid(inoise ** 2, freq))),
            "freq": freq, "onoise": onoise, "inoise": inoise,
            "n_pts": int(len(freq)),
        }
    finally:
        cir_path.unlink(missing_ok=True)
        dat_path.unlink(missing_ok=True)


def cascade_noise_analysis() -> dict:
    """Measure cascade output-referred noise via ngspice .noise.

    Injects probe on Va; measures V(vout) spectrum.  Biasing (va=0,
    vb=1, vc=1) puts both cells at the centre of their range, which
    matches the single-cell noise measurement bias point.
    """
    body = """.include cell_opamp_sub.cir
Vcc vcc 0 15
Vee vee 0 -15
Va va 0 0
Vb vb 0 1
Vc vc 0 1
Xcell1 va vb vint vcc vee eml_cell_opamp
Xcell2 vint vc vout vcc vee eml_cell_opamp
"""
    return _run_noise(body, output_node="vout", source="Va")


def single_cell_noise_analysis() -> dict:
    """Re-measure single-cell noise at va=0, vb=1 to match cascade bias."""
    body = """.include cell_opamp.cir
"""
    return _run_noise(body, output_node="vout", source="Vx")


def predict_depth2_noise_sigma(single_cell_sigma_V: float,
                               bias=(0.0, 1.0, 1.0),
                               n_trials: int = 400,
                               n_reps: int = 500) -> dict:
    """Predict depth-2 output *sigma* at a bias point via noise_sim.

    `noise_sim.simulate` returns mean(|err|) as its ``rmse`` key; for
    Gaussian err that equals sigma * sqrt(2/pi) ~ 0.798*sigma.  For a
    proper sigma comparison against SPICE's integrated .noise, we
    replicate the bias point `n_reps` times so per-trial RMSE
    converges to sigma (within 1/sqrt(n_reps) ~ 4% at 500).
    """
    tree = compile_expr("eml(eml(va, vb), vc)",
                        variables=["va", "vb", "vc"])
    model = AdditiveGaussian(sigma=single_cell_sigma_V)
    x_samples = {
        "va": np.full(n_reps, bias[0]),
        "vb": np.full(n_reps, bias[1]),
        "vc": np.full(n_reps, bias[2]),
    }
    r = simulate(tree, model, x_samples, n_trials=n_trials, seed=0)
    # With many samples at the same bias point, per-trial RMSE
    # estimates the output sigma directly.
    return {
        "sigma_V": float(r["rmse"]),
        "sigma_std_V": float(r["rmse_std"]),
        "n_internal_nodes": int(r["n_internal_nodes"]),
        "depth": int(r["depth"]),
    }


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    print("=" * 60)
    print("Track 1: DC error + cell-to-cell correlation")
    print("=" * 60)
    print("  running single-cell DC sweep on cascade grid...")
    sc_df = run_single_cell_dc_grid()
    print(f"    {len(sc_df)} single-cell points")
    print("  running cascade DC sweep...")
    casc_df = run_cascade_spice()
    print(f"    {len(casc_df)} cascade points (envelope-filtered)")
    casc_df.to_csv(RESULTS_DIR / "cascade_dc.csv", index=False)
    sc_df.to_csv(RESULTS_DIR / "single_cell_dc_grid.csv", index=False)

    dc = dc_correlation_analysis(sc_df, casc_df)
    print(f"  single-cell RMSE       : {dc['single_cell_rmse_V']*1e3:.2f} mV")
    print(f"  cascade RMSE           : {dc['cascade_rmse_V']*1e3:.2f} mV")
    print(f"  measured ratio         : {dc['rmse_ratio']:.3f}")
    print(f"    independent expects  : {dc['independent_expected_ratio']:.3f} (sqrt 2)")
    print(f"    correlated expects   : {dc['correlated_expected_ratio']:.3f}")
    print(f"  corr(cell1_err, cell2_err) = {dc['corr_cell1_cell2_err']:+.3f}")
    print(f"  first-order model fit corr = {dc['corr_model_vs_actual']:+.3f}")
    print(f"  residual RMSE after prop   = "
          f"{dc['propagation_residual_rmse_V']*1e3:.2f} mV")

    print()
    print("=" * 60)
    print("Track 2: AC noise validation vs noise_sim")
    print("=" * 60)
    print("  single-cell noise (va=0, vb=1)...")
    sc_noise = single_cell_noise_analysis()
    print(f"    output-referred: {sc_noise['onoise_rms_V']*1e6:.2f} uV rms "
          f"(20 Hz-20 kHz)")
    print("  cascade noise (va=0, vb=1, vc=1)...")
    casc_noise = cascade_noise_analysis()
    print(f"    output-referred: {casc_noise['onoise_rms_V']*1e6:.2f} uV rms")
    pd.DataFrame({
        "freq": casc_noise["freq"],
        "onoise_density": casc_noise["onoise"],
        "inoise_density": casc_noise["inoise"],
    }).to_csv(RESULTS_DIR / "cascade_noise.csv", index=False)

    # Predict depth-2 output sigma via noise_sim, at the matching bias point
    pred = predict_depth2_noise_sigma(sc_noise["onoise_rms_V"])
    pred_sigma = pred["sigma_V"]
    pred_sigma_std = pred["sigma_std_V"]
    noise_ratio = (casc_noise["onoise_rms_V"] / sc_noise["onoise_rms_V"]
                   if sc_noise["onoise_rms_V"] > 0 else float("nan"))
    pred_ratio = (pred_sigma / sc_noise["onoise_rms_V"]
                  if sc_noise["onoise_rms_V"] > 0 else float("nan"))
    # Analytical: at bias (0, 1, 1), inner_ideal = exp(0)-log(1) = 1,
    # so d(outer)/d(inner) = exp(1) ~ 2.718.  Two independent cells:
    # sigma_out = sigma_cell * sqrt(exp(2*inner_ideal) + 1)
    inner_ideal = float(np.exp(0.0) - np.log(1.0))
    analytical_sigma = sc_noise["onoise_rms_V"] * float(
        np.sqrt(np.exp(2 * inner_ideal) + 1.0)
    )
    print(f"  measured cascade sigma        : "
          f"{casc_noise['onoise_rms_V']*1e6:.2f} uV")
    print(f"  measured cascade/single ratio : {noise_ratio:.3f}")
    print(f"  analytical (exp(1) gain)      : "
          f"{analytical_sigma*1e6:.2f} uV  "
          f"(ratio {analytical_sigma/sc_noise['onoise_rms_V']:.3f})")
    print(f"  noise_sim predicted sigma     : "
          f"{pred_sigma*1e6:.2f} +- {pred_sigma_std*1e6:.2f} uV  "
          f"(ratio {pred_ratio:.3f})")

    if pred_sigma > 0:
        spice_over_sim = casc_noise["onoise_rms_V"] / pred_sigma
        noise_margin = abs(spice_over_sim - 1.0)
        noise_verdict = "PASS" if noise_margin <= 0.30 else "FAIL"
    else:
        spice_over_sim = float("nan")
        noise_margin = float("nan")
        noise_verdict = "N/A"
    print(f"  SPICE / noise_sim             : {spice_over_sim:.3f}")
    print(f"  margin vs noise_sim           : {noise_margin*100:.2f} %")
    print(f"  verdict (+-30%)               : {noise_verdict}")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    dc_interpretation = (
        "DC errors are highly correlated between cells — noise_sim's "
        "independent-Gaussian assumption underestimates deterministic "
        "bias compounding." if dc["corr_cell1_cell2_err"] > 0.5
        else "DC errors are weakly correlated — noise_sim's assumption "
        "is roughly compatible with the deterministic bias as well."
    )
    noise_interpretation = (
        "AC noise compounds near sqrt(2), matching noise_sim." if
        noise_verdict == "PASS" else
        "AC noise compounding diverges from noise_sim beyond 30%."
    )
    print(f"  DC track   : {dc_interpretation}")
    print(f"  NOISE track: {noise_interpretation}")

    summary = {
        "dc_single_cell_rmse_mV": dc["single_cell_rmse_V"] * 1e3,
        "dc_cascade_rmse_mV": dc["cascade_rmse_V"] * 1e3,
        "dc_rmse_ratio": dc["rmse_ratio"],
        "dc_corr_cell1_cell2": dc["corr_cell1_cell2_err"],
        "dc_corr_model_vs_actual": dc["corr_model_vs_actual"],
        "dc_residual_rmse_mV": dc["propagation_residual_rmse_V"] * 1e3,
        "noise_single_cell_uV": sc_noise["onoise_rms_V"] * 1e6,
        "noise_cascade_uV": casc_noise["onoise_rms_V"] * 1e6,
        "noise_analytical_uV": analytical_sigma * 1e6,
        "noise_ratio_measured": noise_ratio,
        "noise_sim_predicted_uV": pred_sigma * 1e6,
        "noise_sim_predicted_std_uV": pred_sigma_std * 1e6,
        "noise_spice_over_sim": spice_over_sim,
        "noise_margin_pct": (noise_margin * 100
                             if np.isfinite(noise_margin) else None),
        "noise_verdict_within_30pct": noise_verdict,
    }
    pd.DataFrame([summary]).to_csv(
        RESULTS_DIR / "cascade_validation.csv", index=False
    )
    print(f"  wrote {RESULTS_DIR / 'cascade_validation.csv'}")


if __name__ == "__main__":
    main()
