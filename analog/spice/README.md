# EML Analog — SPICE Cell Simulations

Physical-circuit validation of the Python noise simulator in
`analog/noise_sim.py`.  Two EML-cell topologies and a depth-2 cascade
are simulated in ngspice; the measured cascade error is cross-checked
against the simulator's depth-2 prediction (go / no-go within ±30 %).

## Motivation

`analog/noise_sim.py` predicts how EML error compounds with tree depth
using abstract Gaussian / 1/f / matched-pair models.  That's
convincing only if the single-cell σ you feed it corresponds to what a
real cell actually produces, and if the compounding behaviour matches
what real cascaded cells do.  The SPICE work here closes that loop.

The EML operator is `eml(x, y) = exp(x) − ln(y)`, scaled 1 V =
1 math unit.  Valid input range per cell: `V_x ∈ [-1, 1]`,
`V_y ∈ [0.1, 2]`.  Supply rails ±15 V.

## What's here

| File | Role |
|------|------|
| `models.lib` | Gummel-Poon NPN model (2N3904-class), behavioural op-amp (OP07-ish, GBW ≈ 0.6 MHz), 4-resistor diffamp |
| `cell_opamp.cir` | Op-amp transdiode log + op-amp antilog + diffamp subtractor |
| `cell_translinear.cir` | Same log arm; passive-load antilog (no op-amp on exp transimpedance) |
| `cell_opamp_sub.cir` | Op-amp cell wrapped as a reusable `eml_cell_opamp` subckt for cascades |
| `cascade.cir` | Depth-2: `eml(eml(V_a, V_b), V_c)` using two `eml_cell_opamp` instances |
| `driver.py` | Python harness: DC / AC / transient / noise / temperature / Monte Carlo |
| `cascade_driver.py` | Depth-2 validation: SPICE cascade RMSE vs `noise_sim.simulate()` prediction |
| `results/*.csv` | Measured sweeps and the validation summary |

## Quick start

```bash
# Full analysis for one topology (DC, AC, TRAN, NOISE, TEMP, MC)
python analog/spice/driver.py --topology opamp --all --n-trials 50
python analog/spice/driver.py --topology translinear --all --n-trials 50

# Depth-2 cascade validation (go / no-go vs noise_sim)
python analog/spice/cascade_driver.py
```

Each run writes a CSV to `results/<topology>_<analysis>.csv` and a
one-line summary to stdout.

## Topologies

Both cells share the log arm (op-amp transdiode with a matched
reference transdiode, so V_T and I_s drift cancel in the op-amp log
cell at a fixed temperature).  They differ on the exp arm:

- **`cell_opamp`** — the antilog BJT's collector is held at virtual
  ground by a feedback op-amp with transimpedance `R_f = 100 kΩ`.
  `V_CE ≈ 0` so Early-effect distortion is negligible.  Two op-amp
  poles in the signal path (exp transimpedance + output subtractor).
- **`cell_translinear`** — the antilog BJT drives a passive
  `R_load = 100 kΩ` pull-up to `V_cc`.  No op-amp transimpedance, so
  the exp path is faster in principle.  Cost: `V_CE` swings with the
  output, injecting Early-effect nonlinearity; a unity-gain buffer
  still has to isolate `R_load` from the 10 kΩ diffamp input.

Emitter-bias voltage `V_em` is tuned per-topology so `V_exp(V_x=0) = 1 V`
exactly (−0.549 V for op-amp, −0.545 V for translinear — the delta
absorbs the Early-effect offset).

## Measured single-cell results

300 K, 20 × 41 grid over `V_x ∈ [-1, 1]`, `V_y ∈ [0.1, 2]`, ideal
op-amp model.

| Metric | `cell_opamp` | `cell_translinear` |
|--------|-------------:|-------------------:|
| DC RMSE vs ideal            | **10.3 mV**  | **16.2 mV** |
| Max \|err\| over grid       | 26.0 mV      | 53.5 mV     |
| Small-signal −3 dB bandwidth| 501 kHz      | 178 kHz     |
| DC gain                     | −0.05 dB     | −0.07 dB    |
| Step settle (1 %, 0 → 0.5 V)| 3.36 µs      | 4.11 µs     |
| Output noise, 20 Hz–20 kHz  | 127 µV rms   | 127 µV rms  |
| Input-referred noise (same) | 128 µV rms   | 128 µV rms  |
| Monte Carlo (1 % R tol., 50 trials) RMSE mean ± σ | 10.6 ± 7.7 mV | 16.8 ± 9.2 mV |

### What the numbers mean

- **DC accuracy.** Both cells sit at ~1–2 % of full-scale, dominated
  by the exp-arm non-idealities (finite op-amp gain, BJT finite β,
  Early effect in the translinear variant).  The op-amp topology is
  ≈ 1.6× more accurate.
- **Bandwidth.** The translinear topology is **slower**, not faster,
  because a unity-gain buffer had to be inserted between the passive
  collector node and the 10 kΩ diffamp input.  A wider-BW translinear
  cell needs a higher-impedance subtractor (e.g. instrumentation-amp
  style) — the two-pole signal path here has no speed advantage.
- **Noise.** Both cells hit ≈ 127 µV rms output noise over audio band —
  overwhelmingly set by the op-amp input-referred 10 nV/√Hz × log arm
  gain (≈ 38.7).  Topology choice doesn't move this; a quieter op-amp
  does.
- **Monte Carlo.** 1 % resistor tolerance adds ≈ 10 mV σ on top of the
  DC RMSE — consistent with component mismatch being dominated by the
  log arm's transdiode pair ratio.

### Temperature sweep

| T (°C) | op-amp RMSE | translinear RMSE | op-amp V_out(V_x=0) |
|-------:|------------:|-----------------:|--------------------:|
|    −40 |       1.4 V |            1.2 V |              0.001 V |
|      0 |       1.3 V |            1.2 V |              0.088 V |
|     25 |     0.21 V  |          0.21 V  |              0.845 V |
|     55 |       8.5 V |            7.9 V |              8.22 V  |
|     85 |      13.6 V |           13.6 V |             14.80 V  |

Calibration is valid only in a narrow band around room temperature.
The dominant drift is the passive `V_T/1V = 260/(9740+260)` divider on
the exp arm input — it can't track `V_T(T)`.  A real board would
either (a) bias the exp arm from a PTAT source, or (b) sense junction
T and feed a correction.  The point of the sweep is to *expose* the
drift, not to claim the cells work untrimmed outside room T.

## Depth-2 cascade validation

`cascade.cir` instantiates two `eml_cell_opamp` blocks, computing
`V_out = eml(eml(V_a, V_b), V_c)`.  Valid envelope used by
`cascade_driver.py`:

- `V_a ∈ [-1, 0.4]`
- `V_b ∈ [0.1, 2.5]`
- `V_c ∈ {0.5, 1.0, 1.5}`
- intermediate `V_int = exp(V_a) − ln(V_b)` filtered to `(-0.9, 0.9)`
  so cell 2's input stays in range (99 of 168 grid points survive).

The Python simulator is then asked to predict the same depth-2 tree's
RMSE with per-node Gaussian noise calibrated to the single-cell DC
RMSE (10.3 mV).

| Quantity | Value |
|----------|------:|
| SPICE cascade RMSE                              | **20.3 mV** |
| SPICE intermediate (cell 1) RMSE                |  9.5 mV     |
| SPICE max \|err\|                               | 53.9 mV     |
| `noise_sim.simulate` with AdditiveGaussian(σ=10 mV), depth-2 | 17.2 ± 1.3 mV |
| `noise_sim.simulate` bits of precision          | 7.3 bits    |
| `noise_sim.simulate` with MultiplicativeGaussian(σ=1 %)      | 16.7 mV    |
| **Ratio SPICE / noise_sim**                     | **1.18**    |
| **Agreement margin**                            | **18.0 %**  |
| **Verdict (±30 %)**                             | **PASS**    |

### Why 1.18× and not 1.0×

Three things push the SPICE cascade RMSE a bit higher than the
independent-Gaussian prediction:

1. **Correlated bias.** Both cells share the same systematic
   miscalibration in the exp arm (empirical `V_em` offset).  When
   errors are correlated rather than independent, compounding is
   closer to additive (`2·σ`) than Pythagorean (`√2·σ`).
2. **Input-dependent curvature.** The exp arm's residual
   nonlinearity is larger at the envelope edges; the grid we
   validated on weights those points heavier than a Gaussian sample.
3. **Intermediate range.** Cell 2 sees a chunk of its valid input
   range, not just `V_x = 0` like the single-cell sweep — it
   experiences the worst-case portion of its own error surface.

The intermediate RMSE (9.5 mV) matches the single-cell RMSE (10.3 mV)
within measurement scatter: cell 1 behaves identically in the cascade
to how it behaves standalone, confirming the cell boundary is clean.

### Takeaway for `analog/noise_sim.py`

Feeding the simulator the single-cell DC RMSE as an additive
Gaussian σ predicts depth-2 cascade behaviour within 20 %.  That's
good enough to use the simulator as the cheap proxy for deeper trees
where full SPICE would be expensive — which was the point of the
issue-#35 validation.

## Design decisions / gotchas

Ngspice-specific friction surfaced during this work; documenting here
so the next person doesn't re-learn it.

- **`alter <src> = PULSE(...)` is a no-op inside `.control`.** You
  can't retype a DC source to PULSE in a `.control` block; the parser
  silently keeps the DC value.  `transient_step()` in `driver.py`
  rewrites the `Vx` line in the netlist text instead.
- **`.print` paginates at ~55 rows.** The 41 × 20 DC sweep got
  truncated to 55 rows when parsed from stdout.  `run_ngspice_wrdata`
  uses `wrdata` → `.dat` → `np.loadtxt` to avoid it.
- **`wrdata` column layout differs per analysis.** DC/TRAN/NOISE write
  2 columns per node (sweep, value); AC writes 3 (sweep, real, imag).
  The driver detects and handles both.
- **`diffamp` convention is `V_out = V_b − V_a`** (second pin minus
  first).  Standard 4-resistor topology but easy to flip and not
  obvious from the symbol.
- **Matched-reference-transdiode** is essential in the log arm; a
  single transdiode's output drifts 4–5 mV/°C.  The matched pair
  drives it to ≈ 0.3 mV/°C under nominal assumptions; the temperature
  sweep's residual drift is almost entirely from the exp arm's
  passive divider.

## Known limitations

- **Op-amp model is behavioural**, not a full macromodel.  Input
  bias current, offset voltage, CMRR, and slew rate are idealised.
  A production cell with real op-amps (e.g. OPA209) will have a
  DC RMSE floor closer to a few tens of mV, not a few tens of µV.
- **Monte Carlo covers only resistor tolerance.** BJT β, I_s
  matching, and op-amp offset are not varied.  Expand by adding
  `.model` randomisation per trial if needed.
- **Transient PULSE is one-shot.** No multi-edge step sequences or
  ramp inputs.  The harness is enough to measure settling, not
  enough to characterise large-signal slewing.
- **Single temperature per sweep.** The TEMP analysis re-simulates
  at fixed `T` values; no continuous T dependence is extracted.

## Provenance

- Paper: Odrzywolek, A.  "All elementary functions from a single
  operator" (arXiv:2603.21852, April 2026) — sections relevant here
  are §4.2 (analog EML circuits) and §4.3 (depth-N recovery).
- Gummel-Poon parameters chosen to match 2N3904 datasheet curves at
  25 °C; XTB / XTI / EG enable the temperature sweep.
- Issue #35 (this repo): depth-2 SPICE validation, go / no-go ±30 %.
