# DCA vs Adam on structural recovery (issue #58)

`eml(x, y) = exp(x) − ln(y)` is jointly convex on y > 0 (Hessian
`diag(e^x, 1/y²)`, PD). Issue #58 asked whether exploiting that
difference-of-convex structure — DCA on a constructive DC split of the
SR loss — improves *structural* recovery on multiplicative targets over
the Adam paths, with the canonical failure case x0·x1 as the success
gate. This resurrects the measurement intent of #57 with a treatment
arm.

## Setup

- **Targets**: the 15 STATUS_COLLAPSED polynomial rows of the
  llm-as-computer catalog (`dev/symbolic_collapse_report.md`,
  "Collapsed (branchless, polynomial-closed)" table); `native_multiply`
  is x0·x1.
- **Arms**: `adam_softmax` = the current default `discover()` path
  (Option A); `adam_linear` = Option-B ladder (`_train_one_linear` +
  `iterative_snap`); `dca_linear` = identical ladder and snap pipeline
  with phase-1 search replaced by block-cyclic DCA (`eml_dca.dca_train`,
  40 outer × 50 inner ≈ the Adam search budget). The search optimizer is
  the only treatment difference between the two linear arms.
- **Protocol**: 256 train samples U(0.5, 2.5)^n; structural recovery =
  max abs error < 1e-6 (relative) on 512 held-out samples over the
  extrapolation band U(0.25, 4.0)^n; depths 0–4, 6 seeds each.
- Full per-target JSONs: `results/dca_recovery/`; merged table:
  `results/dca_recovery/summary_all.md` (regenerate with
  `python3 -m benchmarks.dca_recovery_report`).

## Results

| target | expr | adam_softmax | adam_linear | dca_linear |
|---|---|---|---|---|
| push_halt | `x0` | ✓ | ✓ | ✓ |
| push_pop | `x0` | ✓ | ✓ | ✓ |
| stack_depth | `x0` | ✓ | ✓ | ✓ |
| overwrite | `x2` | ✓ | ✓ | ✓ |
| dup_add | `2*x0` | ✗ | ✓ | ✓ |
| dup_add_chain_x4 | `16*x0` | ✗ | ✓ | ✓ |
| basic_add | `x0 + x1` | ✗ | ✓ | ✓ |
| add_dup_add | `2*x0 + 2*x1` | ✗ | ✓ | ✓ |
| multi_add | `x0 + x1 + x2` | ✗ | ✓ | ✓ |
| complex | `2*x1 + 2*x2` | ✗ | ✓ | ✓ |
| alternating | `x0 + x1 + x3 + x5` | ✗ | ✓ | ✓ |
| many_pushes | `x0+…+x9` | ✗ | ✓ | ✓ |
| square_via_dupmul | `x0²` | ✗ | ✗ (mse 1.3e-3) | ✗ (mse 1.9e-3) |
| sum_of_squares | `x0² + x3²` | ✗ | ✗ (mse 0.216) | ✗ (mse 0.132) |
| native_multiply | `x0·x1` | ✗ | ✗ (mse 0.285) | ✗ (mse **0.048**) |
| **recovered** | | **4/15** | **12/15** | **12/15** |

## Success gate: not met — but the bottleneck moved

DCA does **not** flip x0·x1 to structurally correct. What it does do is
break the additive plateau that both Adam arms sit on:

- `adam_softmax` on x0·x1: best expression `(e − ln(e − ln(x2)))`,
  R² −0.43. The canonical complex-plane construction for x0·x1 needs
  depth 8 / size 35 (`eml_compiler`), beyond the depth-4 ladder, and
  Option A's simplex routing cannot reach the linear-gate compression.
- `adam_linear` on x0·x1: converges to the additive floor
  `−2 + x0 + 2·x1` at mse 0.285, R² 0.76 (best at depth 0 — every
  deeper seed NaNs out from double-exponential init blowup or fails to
  beat it). PR #56's distillation demo hit the same additive ceiling.
- `dca_linear` on x0·x1: certified-descent search reaches a depth-4
  structure at **mse 0.048, held-out R² 0.90** — 6× below the additive
  ceiling, with both variables composed multiplicatively through nested
  eml nodes. The snapped expression is still off-lattice in two
  coefficients (0.397, 0.232), so structural recovery fails at the
  snap stage, not the search stage.

The same pattern holds on `sum_of_squares` (DCA 0.132 vs Adam 0.216)
and inverts mildly on `square_via_dupmul` (1.9e-3 vs 1.3e-3, both
near-fits snapped to wrong depth-1 structures).

A depth-4 real linear-gate tree *can* express x0·x1 exactly with
lattice constants — `x0·x1 = exp(2 − D)`, `D = (2 − ln x1) − ln x0`
built from `eml(0, x1)` upward — so the failure is a search/snap gap,
not an expressibility bound. (Odrzywolek's completeness is over ℂ; on
the real positive branch this construction has to be found per target.)

## Costs and caveats

- **Wall-clock**: the DCA arm costs 3–4× the Adam-linear arm at full
  ladder (e.g. 72 min vs 14 min on x0²) — `dc_forward` builds per-node
  DC pairs in Python loops. On the three nonlinear targets both linear
  arms are dominated by `iterative_snap`'s one-coefficient-at-a-time
  retraining (O(n_coeffs) retrain loops per seed per depth).
- **Affine targets**: both linear arms solve all 12 exactly at depth 0
  (a leaf is already a linear regression), so they cannot separate the
  optimizers there; the 12/15-vs-4/15 gap over `adam_softmax` restates
  Option A's known vocabulary limits (`eml_sr_linear.py` §findings).
- The DC machinery required two corrections to the issue's sketch,
  documented in `eml_dca.py`: the depth-1 loss split `g = f² +
  2max(0,−y)f` is not convex for sign-changing residuals (replaced by
  the monotone split `r² = max(r,0)² + min(r,0)²`), and whole-θ DCA is
  blocked by the γ·child parameter product (hence block-cyclic, one
  gate level per block, with intervals/Lipschitz constants frozen at
  θ_k under a trust region).

## Conclusion

On this benchmark the convexity structure buys certified monotone
descent and measurably better basins on multiplicative targets, but
structural recovery is now gated by snap-to-lattice, which discards the
basin advantage. The natural follow-ups are (a) a snap stage that
exploits the DC structure (convex-constrained lattice projection per
block instead of per-coefficient rounding), and (b) vectorizing
`dc_forward` to close the 3–4× wall-clock gap.
