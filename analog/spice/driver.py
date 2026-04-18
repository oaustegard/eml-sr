#!/usr/bin/env python3
"""analog/spice/driver.py - SPICE harness for EML analog cells.

Runs DC, AC, transient, noise, temperature-sweep, and Monte Carlo
simulations against the netlists in this directory and emits results
as CSVs under ``analog/spice/results/``.

Two cell topologies are characterized:
  - cell_opamp      : op-amp transdiode log, op-amp antilog, subtract
  - cell_translinear: op-amp log, passive-load translinear antilog

Run everything:
    python -m analog.spice.driver --all

Per-analysis:
    python -m analog.spice.driver --dc --topology opamp
    python -m analog.spice.driver --ac --topology translinear
    python -m analog.spice.driver --noise --topology opamp
    python -m analog.spice.driver --temp --topology opamp
    python -m analog.spice.driver --mc --topology opamp --n-trials 100

All analyses write to results/<topology>_<analysis>.csv and print a
one-line summary to stdout.  The cascade driver lives in
``cascade_driver.py`` and depends on results from this module plus the
depth-2 prediction from ``analog.noise_sim``.
"""

from __future__ import annotations

import argparse
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
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

TOPOLOGIES = {
    "opamp": "cell_opamp.cir",
    "translinear": "cell_translinear.cir",
}


# --------------------------------------------------------------
# ngspice runner
# --------------------------------------------------------------

def run_ngspice(netlist: str, timeout: int = 120) -> tuple[str, str]:
    """Run ngspice in batch mode on `netlist` (a string).  Returns stdout/stderr."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cir", delete=False, dir=SPICE_DIR
    ) as f:
        f.write(netlist)
        path = Path(f.name)
    try:
        result = subprocess.run(
            ["ngspice", "-b", str(path)],
            capture_output=True, text=True, timeout=timeout,
            cwd=SPICE_DIR,
        )
        return result.stdout, result.stderr
    finally:
        path.unlink()


def run_ngspice_wrdata(
    netlist_body: str, analysis: str, nodes: list[str],
    pre: str = "", timeout: int = 120,
) -> np.ndarray:
    """Run ngspice with a .control block that writes wrdata to a tmp file.

    `netlist_body` should include the device definitions (.include etc.)
    and any `.param` overrides but NOT `.end`.
    `analysis` is a single analysis line (e.g. 'dc Vx -1 1 0.05 Vy 0.1 2 0.1').
    `nodes` is the list of node/branch names to capture.
    `pre` is additional commands to run inside .control before the analysis
      (e.g. ``alter vx ac = 1`` to switch a source to AC mode).

    Returns a 2-D numpy array of shape (n_points, 1 + len(nodes)).
    The first column is the sweep variable (Vx for dc, frequency for ac, ...).
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".dat", delete=False, dir=SPICE_DIR
    ) as f:
        dat_path = Path(f.name)
    node_spec = " ".join(f"v({n})" for n in nodes)
    pre_lines = pre.strip()
    control = f"""
.control
  set filetype = ascii
  {pre_lines}
  {analysis}
  wrdata {dat_path.name} {node_spec}
  quit
.endc
"""
    netlist = netlist_body + control + "\n.end\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cir", delete=False, dir=SPICE_DIR
    ) as f:
        f.write(netlist)
        cir_path = Path(f.name)
    try:
        subprocess.run(
            ["ngspice", "-b", str(cir_path)],
            capture_output=True, text=True, timeout=timeout, cwd=SPICE_DIR,
        )
        if not dat_path.exists():
            return np.zeros((0, 1 + len(nodes)))
        try:
            raw = np.loadtxt(dat_path)
        except ValueError:
            return np.zeros((0, 1 + len(nodes)))
        if raw.size == 0:
            return np.zeros((0, 1 + len(nodes)))
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        # wrdata writes each variable as its own x/y pair(s).
        # For real-valued analyses (dc, tran, noise) each node is 2 cols
        # (sweep, v).  For AC each node is 3 cols (sweep, re, im).
        total_cols = raw.shape[1]
        n = len(nodes)
        cols_per_node = total_cols // n if n > 0 else 1
        sweep = raw[:, 0]
        if cols_per_node == 2:
            values = raw[:, 1::cols_per_node]
            return np.column_stack([sweep, values])
        elif cols_per_node == 3:
            # AC: combine re+im into complex magnitude per node
            out = [sweep]
            for i in range(n):
                re = raw[:, 1 + i * 3]
                im = raw[:, 2 + i * 3]
                out.append(np.abs(re + 1j * im))
            return np.column_stack(out)
        else:
            return raw
    finally:
        cir_path.unlink(missing_ok=True)
        dat_path.unlink(missing_ok=True)


_PRINT_TABLE_RE = re.compile(
    r"^-{5,}\s*\nIndex\s+(.+?)\n-{5,}\s*\n(.+?)(?=\n\s*\n|\Z)",
    re.DOTALL | re.MULTILINE,
)


def parse_print_tables(stdout: str) -> list[pd.DataFrame]:
    """Parse all `.print` result tables from an ngspice stdout dump."""
    tables = []
    blocks = re.split(r"-{5,}\s*\n", stdout)
    i = 0
    while i < len(blocks) - 2:
        hdr = blocks[i + 1]
        body = blocks[i + 2]
        if hdr.strip().startswith("Index"):
            cols = hdr.split()[1:]
            rows = []
            for line in body.splitlines():
                if not line.strip() or line.lstrip().startswith("*"):
                    break
                parts = line.split()
                if len(parts) >= len(cols) + 1:
                    try:
                        row = [float(p) for p in parts[1:1 + len(cols)]]
                        rows.append(row)
                    except ValueError:
                        break
            if rows and cols:
                tables.append(pd.DataFrame(rows, columns=cols))
            i += 2
        else:
            i += 1
    return tables


# --------------------------------------------------------------
# Topology helpers
# --------------------------------------------------------------

def _topology_netlist_header(topology: str) -> str:
    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown topology: {topology}")
    return f".include {TOPOLOGIES[topology]}\n"


def _ideal_eml(vx: float, vy: float) -> float:
    """Reference EML operator in math units."""
    return float(np.exp(vx) - np.log(vy))


# --------------------------------------------------------------
# DC transfer
# --------------------------------------------------------------

def dc_transfer(topology: str, vx_pts: int = 41, vy_pts: int = 20) -> pd.DataFrame:
    """Sweep V_x and V_y, compare V_out against ideal eml_op."""
    vx_step = 2.0 / (vx_pts - 1)
    vy_step = 1.9 / (vy_pts - 1)
    data = run_ngspice_wrdata(
        _topology_netlist_header(topology),
        analysis=f"dc Vx -1.0 1.0 {vx_step:.6f} Vy 0.1 2.0 {vy_step:.6f}",
        nodes=["vout"],
    )
    if data.size == 0:
        raise RuntimeError("no DC data captured")
    # Inner sweep: Vx; outer: Vy.  data[:, 0] is Vx, cycling vx_pts
    # times per Vy step.  Reconstruct Vy from row index.
    vy_vals = np.linspace(0.1, 2.0, vy_pts)
    records = []
    for i in range(len(data)):
        vx = float(data[i, 0])
        vout = float(data[i, 1])
        vy_idx = i // vx_pts
        vy = float(vy_vals[min(vy_idx, vy_pts - 1)])
        records.append({
            "vx": vx, "vy": vy,
            "v_out": vout,
            "ideal": _ideal_eml(vx, vy),
            "err": vout - _ideal_eml(vx, vy),
        })
    return pd.DataFrame(records)


# --------------------------------------------------------------
# AC / Bode
# --------------------------------------------------------------

def ac_bandwidth(topology: str) -> dict:
    """AC small-signal: find -3 dB bandwidth, injected at V_x."""
    data = run_ngspice_wrdata(
        _topology_netlist_header(topology),
        pre="alter vx ac = 1",
        analysis="ac dec 20 1 1e8", nodes=["vout"],
    )
    if data.size == 0:
        return {"bandwidth_hz": float("nan"), "dc_gain_db": float("nan"), "n_pts": 0}
    freq = data[:, 0]
    mag = np.abs(data[:, 1])
    with np.errstate(divide="ignore"):
        mag_db = 20 * np.log10(np.maximum(mag, 1e-30))
    dc_gain_db = float(mag_db[0])
    threshold = dc_gain_db - 3.0
    bw_idx = np.argmax(mag_db < threshold)
    bandwidth = float(freq[bw_idx]) if bw_idx > 0 else float(freq[-1])
    return {
        "bandwidth_hz": bandwidth,
        "dc_gain_db": dc_gain_db,
        "n_pts": int(len(freq)),
        "freq": freq, "mag_db": mag_db,
    }


# --------------------------------------------------------------
# Transient step
# --------------------------------------------------------------

def transient_step(topology: str) -> dict:
    """Step-response: V_x steps from 0 -> 0.5 at t=1us; measure settling."""
    # ngspice's `alter` can't retype a DC source to PULSE inside .control.
    # Instead, inline the cell file and rewrite Vx to a pulsed source.
    cell_path = SPICE_DIR / TOPOLOGIES[topology]
    body = cell_path.read_text()
    body = re.sub(
        r"^Vx\s+vx\s+0\s+\{VX_DC\}.*$",
        "Vx vx 0 DC 0 PULSE(0 0.5 1u 10n 10n 100u 200u)",
        body, flags=re.MULTILINE,
    )
    body = re.sub(r"\n\.end\s*$", "\n", body.rstrip()) + "\n"
    data = run_ngspice_wrdata(
        body, analysis="tran 50n 20u", nodes=["vout"],
    )
    if data.size == 0:
        return {"settle_1pct_us": float("nan"), "n_pts": 0}
    t = data[:, 0]
    vout = data[:, 1]
    # Final value = value from second half of window (>= 15us)
    final = float(np.mean(vout[t >= 15e-6])) if np.any(t >= 15e-6) else float(vout[-1])
    initial = float(np.mean(vout[t < 1e-6])) if np.any(t < 1e-6) else float(vout[0])
    step = final - initial
    if abs(step) < 1e-6:
        return {"settle_1pct_us": float("nan"), "initial": initial,
                "final": final, "step": step}
    # Find first t where |vout - final| < 1% |step| for all subsequent points
    mask = np.abs(vout - final) < 0.01 * abs(step)
    settled = (
        np.all(mask[i:]) for i in range(len(mask))
    )
    idx = next((i for i, s in enumerate(settled) if s), len(t) - 1)
    return {
        "settle_1pct_us": float(t[idx] * 1e6 - 1.0),
        "initial": initial,
        "final": final,
        "step": step,
        "t": t, "v": vout,
    }


# --------------------------------------------------------------
# Noise analysis
# --------------------------------------------------------------

def noise_analysis(topology: str, f_low: float = 20, f_high: float = 20000) -> dict:
    """ngspice .noise: integrate input-referred noise in [f_low, f_high]."""
    netlist = _topology_netlist_header(topology)
    # For .noise, wrdata captures `inoise_spectrum` and `onoise_spectrum`
    # as pseudo-vectors.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".dat", delete=False, dir=SPICE_DIR
    ) as f:
        dat_path = Path(f.name)
    control = f"""
.control
  set filetype = ascii
  alter vx ac = 1
  noise v(vout) Vx dec 20 {f_low:.1f} {f_high:.1f}
  setplot noise1
  wrdata {dat_path.name} inoise_spectrum onoise_spectrum
  quit
.endc
"""
    full = netlist + control + "\n.end\n"
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
            return {"integrated_noise_rms": float("nan"),
                    "input_referred_noise_rms": float("nan"), "n_pts": 0}
        raw = np.loadtxt(dat_path)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        freq = raw[:, 0]
        inoise = raw[:, 1]
        onoise = raw[:, 3] if raw.shape[1] >= 4 else raw[:, 1]
        int_onoise = float(np.sqrt(np.trapezoid(onoise ** 2, freq)))
        int_inoise = float(np.sqrt(np.trapezoid(inoise ** 2, freq)))
        return {
            "integrated_noise_rms": int_onoise,
            "input_referred_noise_rms": int_inoise,
            "n_pts": int(len(freq)),
            "freq": freq, "onoise": onoise, "inoise": inoise,
        }
    finally:
        cir_path.unlink(missing_ok=True)
        dat_path.unlink(missing_ok=True)


# --------------------------------------------------------------
# Temperature sweep
# --------------------------------------------------------------

def temperature_sweep(topology: str, temps_c=(-40, 0, 25, 55, 85)) -> pd.DataFrame:
    """DC transfer at multiple temperatures; report RMSE vs ideal."""
    records = []
    for T in temps_c:
        netlist = _topology_netlist_header(topology) + f"""
.options TEMP={T}
"""
        data = run_ngspice_wrdata(
            netlist, analysis="dc Vx -1.0 1.0 0.1", nodes=["vout"],
        )
        if data.size == 0:
            continue
        vx = data[:, 0]
        vout = data[:, 1]
        # Default Vy = 1 so ideal = exp(Vx) - 0 = exp(Vx)
        ideal = np.exp(vx)
        err = vout - ideal
        records.append({
            "temp_C": T,
            "rmse_vs_ideal": float(np.sqrt(np.mean(err ** 2))),
            "max_abs_err": float(np.max(np.abs(err))),
            "vout_at_vx0": float(np.interp(0.0, vx, vout)),
        })
    return pd.DataFrame(records)


# --------------------------------------------------------------
# Monte Carlo on component tolerance
# --------------------------------------------------------------

def monte_carlo(topology: str, n_trials: int = 50, tol: float = 0.01) -> pd.DataFrame:
    """Resistor 1% tolerance Monte Carlo.

    Reads the base netlist, injects per-resistor random scaling, runs DC
    sweep, records RMSE vs ideal for each trial.
    """
    netlist_path = SPICE_DIR / TOPOLOGIES[topology]
    base = netlist_path.read_text()
    # strip trailing .end so run_ngspice_wrdata can append its own
    base = re.sub(r"\n\.end\s*$", "\n", base.rstrip()) + "\n"
    rng = np.random.default_rng(42)
    records = []
    r_pattern = re.compile(r"^(R\w+)\s+(\S+)\s+(\S+)\s+(\S+)(.*)$", re.MULTILINE)
    resistors = [
        (m.group(1), m.group(4)) for m in r_pattern.finditer(base)
        if not m.group(4).startswith("{")
    ]

    for trial in range(n_trials):
        mods = {}
        for name, value in resistors:
            try:
                if value.endswith("k"):
                    v = float(value[:-1]) * 1e3
                elif value.endswith("Meg"):
                    v = float(value[:-3]) * 1e6
                elif value.endswith("M"):
                    v = float(value[:-1]) * 1e6
                else:
                    v = float(value)
            except ValueError:
                continue
            scale = 1.0 + rng.normal(0, tol)
            mods[name] = (value, f"{v * scale:.6g}")
        trial_netlist = base
        for name, (old, new) in mods.items():
            trial_netlist = re.sub(
                rf"^({name}\s+\S+\s+\S+\s+){re.escape(old)}",
                rf"\g<1>{new}",
                trial_netlist,
                count=1,
                flags=re.MULTILINE,
            )
        # Run DC Vx sweep at Vy = 1
        data = run_ngspice_wrdata(
            trial_netlist, analysis="dc Vx -1.0 1.0 0.1", nodes=["vout"],
        )
        if data.size == 0:
            records.append({"trial": trial, "rmse": float("nan")})
            continue
        vx = data[:, 0]
        vout = data[:, 1]
        ideal = np.exp(vx)  # Vy = 1
        err = vout - ideal
        records.append({
            "trial": trial,
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "max_abs_err": float(np.max(np.abs(err))),
        })
    return pd.DataFrame(records)


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topology", choices=list(TOPOLOGIES), default="opamp")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dc", action="store_true")
    ap.add_argument("--ac", action="store_true")
    ap.add_argument("--tran", action="store_true")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--temp", action="store_true")
    ap.add_argument("--mc", action="store_true")
    ap.add_argument("--n-trials", type=int, default=30)
    args = ap.parse_args()

    topos = list(TOPOLOGIES) if args.topology is None else [args.topology]

    for topo in topos:
        print(f"\n=== Topology: {topo} ===")
        if args.all or args.dc:
            df = dc_transfer(topo)
            df.to_csv(RESULTS_DIR / f"{topo}_dc.csv", index=False)
            rmse = float(np.sqrt(np.mean(df["err"] ** 2)))
            print(f"  DC : {len(df)} pts, RMSE = {rmse:.4g}, "
                  f"max |err| = {df['err'].abs().max():.4g}")

        if args.all or args.ac:
            r = ac_bandwidth(topo)
            print(f"  AC : -3dB BW = {r['bandwidth_hz']:.3g} Hz, "
                  f"DC gain = {r['dc_gain_db']:.2f} dB")
            pd.DataFrame({"freq": r["freq"], "mag_db": r["mag_db"]})\
              .to_csv(RESULTS_DIR / f"{topo}_ac.csv", index=False)

        if args.all or args.tran:
            r = transient_step(topo)
            print(f"  TRAN : 1% settle = {r['settle_1pct_us']:.3g} us, "
                  f"step = {r['step']:.3g} V")
            if "t" in r:
                pd.DataFrame({"t": r["t"], "v_out": r["v"]})\
                  .to_csv(RESULTS_DIR / f"{topo}_tran.csv", index=False)

        if args.all or args.noise:
            r = noise_analysis(topo)
            print(f"  NOISE : output RMS noise (20Hz-20kHz) = "
                  f"{r['integrated_noise_rms']:.3g} V, "
                  f"input-referred = {r['input_referred_noise_rms']:.3g} V")
            if "freq" in r:
                pd.DataFrame({
                    "freq": r["freq"],
                    "onoise_density": r["onoise"],
                    "inoise_density": r["inoise"],
                }).to_csv(RESULTS_DIR / f"{topo}_noise.csv", index=False)

        if args.all or args.temp:
            df = temperature_sweep(topo)
            df.to_csv(RESULTS_DIR / f"{topo}_temp.csv", index=False)
            print(f"  TEMP: {df.to_string(index=False)}")

        if args.all or args.mc:
            df = monte_carlo(topo, n_trials=args.n_trials)
            df.to_csv(RESULTS_DIR / f"{topo}_mc.csv", index=False)
            rmses = df["rmse"].dropna()
            print(f"  MC ({len(df)} trials): RMSE mean={rmses.mean():.4g}, "
                  f"std={rmses.std():.4g}, max={rmses.max():.4g}")


if __name__ == "__main__":
    main()
