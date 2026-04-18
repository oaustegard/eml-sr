# EML Analog — SPICE Cell Simulations

Physical-circuit validation of the Python noise simulator in
`analog/noise_sim.py`.  Two EML-cell topologies and a depth-2
cascade are simulated in ngspice; measured cascade noise and DC
error are cross-checked against `noise_sim`'s predictions.

> **Note on this writeup.**  An earlier version (PR #44) compared
> deterministic DC errors to `noise_sim`'s Gaussian-noise output and
> declared "pass within ±30%".  That comparison was loose and the
> framing was wrong in an interesting way — the deeper investigation
> below replaces it.  The code from the earlier PR is mostly kept;
> `cascade_driver.py` and this README were rewritten.

## What `noise_sim` actually models

Given an EML tree like `eml(eml(va, vb), vc)` and a per-node noise
model (e.g. `AdditiveGaussian(σ)`), `noise_sim.simulate` Monte-Carlos
the tree forward in float64, injecting independent noise at each
internal node's output, and returns the output error statistics.

The critical detail: it **doesn't multiply σ by √N or √depth**.  It
evaluates the tree forward with noisy intermediates, which means the
noise automatically propagates through each node's *local
derivative* — for example, noise on the inner node of
`eml(inner, c) = exp(inner) − ln(c)` gets amplified by `exp(inner)`
at the outer node before being added to the outer node's own noise.

That's the right model if (and only if) per-cell noise really is
independent between cells in a physical cascade.  The SPICE work
here checks two things: is that assumption true (AC track), and
how close is it to the (deterministic) DC behaviour (DC track).

## What's here

| File | Role |
|------|------|
| `models.lib` | Gummel-Poon NPN model (2N3904-class), behavioural op-amp (OP07-ish, GBW ≈ 0.6 MHz), 4-resistor diffamp |
| `cell_opamp.cir` | Op-amp transdiode log + op-amp antilog + diffamp subtractor |
| `cell_translinear.cir` | Same log arm; passive-load antilog (no op-amp on exp transimpedance) |
| `cell_opamp_sub.cir` | Op-amp cell wrapped as a reusable `eml_cell_opamp` subckt for cascades |
| `cascade.cir` | Depth-2: `eml(eml(V_a, V_b), V_c)` using two `eml_cell_opamp` instances |
| `driver.py` | Python harness: DC / AC / transient / noise / temperature / Monte Carlo |
| `cascade_driver.py` | Two-track validation: AC noise (validates `noise_sim`) + DC correlation (characterises it) |
| `results/*.csv` | Measured sweeps and the validation summary |

## Quick start

```bash
python analog/spice/driver.py --topology opamp --all --n-trials 50
python analog/spice/driver.py --topology translinear --all --n-trials 50
python analog/spice/cascade_driver.py
```

Each analysis writes a CSV under `results/` and a one-line summary.

## Topologies

Both cells share the log arm (op-amp transdiode with a matched
reference transdiode).  They differ on the exp arm:

- **`cell_opamp`** — antilog BJT's collector held at virtual ground
  by a feedback op-amp with transimpedance `R_f = 100 kΩ`.
  `V_CE ≈ 0`, Early-effect negligible.
- **`cell_translinear`** — passive `R_load` pull-up instead of an
  op-amp transimpedance.  `V_CE` swings with the output; Early
  effect is now visible.  Needed a unity-gain buffer anyway to keep
  the 10 kΩ diffamp from loading the collector node.

## Single-cell DC / AC / noise

300 K, 20 × 41 grid over `V_x ∈ [-1, 1]`, `V_y ∈ [0.1, 2]`.

| Metric | `cell_opamp` | `cell_translinear` |
|--------|-------------:|-------------------:|
| DC RMSE vs ideal                               | **10.3 mV**  | **16.2 mV** |
| Max \|err\| over grid                          | 26.0 mV      | 53.5 mV     |
| Small-signal −3 dB bandwidth                   | 501 kHz      | 178 kHz     |
| DC gain                                        | −0.05 dB     | −0.07 dB    |
| Step settle (1 %, 0 → 0.5 V)                   | 3.36 µs      | 4.11 µs     |
| Output noise, 20 Hz–20 kHz                     | 127 µV rms   | 127 µV rms  |
| Input-referred noise (same)                    | 128 µV rms   | 128 µV rms  |
| Monte Carlo (1 % R tol., 50 trials) RMSE ± σ   | 10.6 ± 7.7 mV | 16.8 ± 9.2 mV |

### Interpretation

- **DC accuracy** sits at ~1–2 % full-scale, dominated by the exp-arm
  calibration (empirical `V_em` tuning, BJT non-idealities, finite
  op-amp gain).  The op-amp topology is ≈ 1.6× more accurate.
- **Bandwidth.**  The translinear topology is **slower**, not faster
  — the unity-gain buffer inserted to isolate the passive collector
  from the 10 kΩ diffamp reintroduces a pole.  A genuinely faster
  translinear cell needs a higher-input-impedance subtractor.
- **Noise** is dominated by the op-amp's input-referred 10 nV/√Hz
  through the log arm's ×38.7 post-gain; topology choice is
  irrelevant to this number.

### Temperature sweep

Calibration is valid only in a narrow band around room temperature
— RMSE is 200 mV at 25 °C but 1.2 V at 0 °C and 8.5 V at 55 °C.
The dominant drift is the passive `V_T/1V = 260/(9740+260)` divider
on the exp arm input: it can't track `V_T(T)`.  A real board needs a
PTAT-biased divider or a junction-temperature correction loop.

## Depth-2 cascade validation

`cascade.cir` = two `eml_cell_opamp` blocks computing
`V_out = eml(eml(V_a, V_b), V_c)`.  The valid envelope keeps cell 2's
input inside the tested single-cell range:

- `V_a ∈ [-1, 0.4]`, `V_b ∈ [0.1, 2.5]`, `V_c ∈ {0.5, 1.0, 1.5}`
- filter on intermediate `exp(V_a) − ln(V_b) ∈ (-0.9, 0.9)` → 99
  points retained out of the original 168.

### Track 1 — AC noise (the validating comparison)

Both cells' thermal noise sources are genuinely independent —
different op-amps, different BJT shot noise.  `ngspice .noise` gives
the output-referred RMS; `noise_sim` with an `AdditiveGaussian`
model should match.

At bias point (`V_a=0, V_b=1, V_c=1`):

|  | Value |
|---|------:|
| Single-cell output noise (20 Hz–20 kHz)                 | **127 µV rms** |
| Cascade output noise at same bias                       | **362 µV rms** |
| Measured cascade / single-cell ratio                    | 2.85           |
| Analytical prediction: `σ · √(exp(2·V_int) + 1)` with `V_int=1` | 368 µV rms |
| `noise_sim.simulate(AdditiveGaussian(σ=127 µV))` sigma  | 368 ± 11 µV    |
| **SPICE / `noise_sim`**                                 | **0.985**      |
| **Margin**                                              | **1.5 %**      |
| **Verdict (±30 %)**                                     | **PASS**       |

The measured ratio is `e ≈ 2.72`, not `√2 ≈ 1.41`, because at this
bias the inner node evaluates to `exp(0) − ln(1) = 1` and the outer
node's `exp(inner)` derivative is `e`.  `noise_sim` captures this
automatically by running the tree forward with noisy intermediates;
the 1.5 % agreement is the real validation.

### Track 2 — DC error (characterisation, not validation)

DC errors aren't random — at a given `(V_x, V_y)` the single cell
produces the same error every simulation.  But we can still ask:
how well does the per-cell error surface predict the cascade error?

From 56 single-cell and 99 cascade points on a shared grid:

| Quantity | Value |
|----------|------:|
| Single-cell DC RMSE                                        | 10.9 mV |
| Cascade DC RMSE                                            | 20.3 mV |
| Cascade / single-cell ratio                                | 1.86    |
| `corr(cell1_err_at_(V_a,V_b), cell2_err_at_(V_int,V_c))`   | +0.05   |
| `corr( exp(V_int)·cell1_err + cell2_err ,  cascade_err )`  | +0.77   |
| Residual RMSE after first-order propagation                | 9.5 mV  |

Reading the numbers:

- Cell errors at their respective inputs are essentially
  **uncorrelated** (r = 0.05).  That surprised me — I had expected
  a shared calibration bias to dominate — but the cells operate on
  different parts of their domain, and the exp-arm error surface has
  sign-changing regions that wash out into a near-zero cross-input
  correlation.
- The first-order analytical propagation
  `cascade_err ≈ exp(V_int_ideal) · cell1_err(V_a, V_b) + cell2_err(V_int, V_c)`
  explains 59 % of the variance (r ≈ 0.77).  The other 41 % is
  nonlinear propagation (the exp gain isn't a constant over the
  envelope) plus second-order BJT curvature not captured by a single
  derivative.
- `noise_sim` with `AdditiveGaussian(σ=10.9 mV)` predicts ~17 mV
  cascade RMSE.  SPICE measures 20.3 mV.  That's 1.18× higher; the
  gap is the nonlinear residual.

The DC track is **not** a clean validation of `noise_sim` for
deterministic bias — systematic per-input bias is not a Gaussian
random variable, and treating it as one washes out any shape
information in the bias surface.  But the order of magnitude is
right, and the 9.5 mV first-order residual tells you where the
model stops being linearisable (at the envelope edges).

### Verdict on `noise_sim`

- **For AC / stochastic noise**: validated within 1.5 % at depth 2.
  Trust it for depth-3 and deeper estimates, understanding that real
  deeper cascades will pick up additional op-amp noise contributions
  the single-cell σ didn't see.
- **For DC / systematic bias**: it gets the order of magnitude
  right (within ~20 %) because tree-forward evaluation carries the
  correct local derivatives, but it doesn't represent the bias
  surface's structure.  A deterministic tree evaluator with a bias
  *function* `δ(V_x, V_y)` per cell would be more accurate for
  cascade accuracy forecasting; that's a future extension, not a
  fix to the noise model.

## Gotchas encountered (recording so the next pass doesn't relearn)

- **`alter <src> = PULSE(...)` is silently a no-op inside `.control`.**
  You can't retype a DC source to PULSE in a control block; the
  parser keeps the DC value.  `transient_step()` in `driver.py`
  rewrites the `Vx` line in the netlist text instead.
- **`.print` paginates at ~55 rows.**  Large DC sweeps get truncated
  in stdout.  `run_ngspice_wrdata` uses `wrdata` → `.dat` →
  `np.loadtxt` to avoid it.
- **`wrdata` column layout differs per analysis.**  DC / TRAN / NOISE
  → 2 cols per node (sweep, value).  AC → 3 cols (sweep, real, imag).
  Driver detects and handles both.
- **`diffamp` convention is `V_out = V_b − V_a`** (second pin minus
  first).  Standard 4-resistor topology but easy to flip.
- **`noise_sim.rmse` is `E|err|`, not σ.**  For Gaussian err they
  differ by `√(2/π) ≈ 0.798`.  To compare sigma-to-sigma against
  `ngspice .noise`'s RMS output, replicate the bias point many times
  so per-trial RMSE converges to σ (what `predict_depth2_noise_sigma`
  does).
- **Matched-reference-transdiode** is essential in the log arm.  A
  single transdiode drifts 4–5 mV/°C; the matched pair brings that
  to ≈ 0.3 mV/°C.  Temperature sweep residual drift is almost
  entirely the exp arm's passive divider.

## Known limitations

- **Op-amp model is behavioural**, not a full macromodel — no input
  bias current, offset voltage, CMRR, or slew-rate modelling.  Real
  op-amps (OPA209-class) will floor DC RMSE in the tens of mV range.
- **Monte Carlo covers only resistor tolerance.**  Extend by
  randomising `.model` parameters (BJT β, `I_s`) per trial if
  needed.
- **Transient harness runs one PULSE.**  Characterises settle time,
  not large-signal slew over multiple edges.
- **Temperature sweep is discrete**, not continuous — no `∂/∂T`
  fit.  Use the discrete points as anchors.
- **Cascade validated only at depth 2.**  Depth-3 would need a
  different netlist (three cells) and a re-check of the envelope
  math; at depth 3 the `noise_sim` prediction itself would need
  verification as internal ranges grow.

## Provenance

- Paper: Odrzywolek, A.  "All elementary functions from a single
  operator" (arXiv:2603.21852, April 2026) — §4.2 (analog) and
  §4.3 (depth-N recovery) are the relevant sections.
- Gummel-Poon parameters chosen to match 2N3904 datasheet curves at
  25 °C; XTB / XTI / EG enable the T sweep.
- Issue #35: original scope — depth-2 SPICE validation of
  `analog/noise_sim.py`.  PR #44 (merged) implemented the
  infrastructure.  This PR replaces that PR's cascade validation
  and writeup with the more careful two-track analysis.
