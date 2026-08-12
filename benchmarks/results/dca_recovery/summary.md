# DCA vs Adam on structural recovery

_Generated 2026-08-12 19:21 UTC_

Implements [eml-sr issue #58](https://github.com/oaustegard/eml-sr/issues/58).

## Protocol

- Targets: 1 STATUS_COLLAPSED polynomial rows from the llm-as-computer symbolic-collapse catalog (`dev/symbolic_collapse_report.md`, "Collapsed (branchless, polynomial-closed)" table)
- Arms: adam_softmax, adam_linear, dca_linear
- Train: 256 samples, X ~ U(0.5, 2.5)^n_vars
- Held-out (extrapolation): 512 samples, X ~ U(0.25, 4.0)^n_vars
- structural_recovered: max abs err of the snapped tree on held-out < 1e-06 * max(1, max|y_held|)
- Ladder success threshold (train snap MSE): 1e-10
- max_depth=4, n_tries(seeds)=6
- dca_linear budget: outer_iters=40 × inner_iters=50 (2000 total, ≈ Adam's default search_iters=2000)
- adam_linear / dca_linear ladders run **serial** per seed (no multiprocessing fan-out); only adam_softmax honors `--workers` (passed through to `eml_sr.discover`)


## Results

| target | adam_softmax (rec / R² / s) | adam_linear (rec / R² / s) | dca_linear (rec / R² / s) |
|---|---|---|---|
| native_multiply | ✗ / -0.434 / 177.8 | ✗ / nan / 1125.9 | ✗ / 0.903 / 4304.6 |
| **totals** | 0/1 / — / 177.8 | 0/1 / — / 1125.9 | 0/1 / — / 4304.6 |

## Per-target expressions

| target | arm | depth | expr |
|---|---|---|---|
| native_multiply | adam_softmax | 4 | `(e - ln((e - ln(x2))))` |
| native_multiply | adam_linear | None | `<error: cannot convert float NaN to integer>` |
| native_multiply | dca_linear | 4 | `eml(1*x2 + 0.397*(eml(0, 1 + -1*x2 + 0.232*(eml(1*x2 + 1*(eml(1*(1), 1*x1 + 1*(1 + 1*x1 + -1*x2))), 1 + -1*x1 + e*x2)))), e)` |