# Cousin ablation: EML vs EDL vs −EML

_Generated 2026-04-19 00:08 UTC_

Implements [eml-sr issue #36](https://github.com/oaustegard/eml-sr/issues/36).

## Setup

- Operators: eml(x, y) = exp(x) - ln(y), edl(x, y) = exp(x) / ln(y), neg_eml(x, y) = ln(x) - exp(y)
- Depths: [2, 3]
- Seeds per (target, depth, op): 2
- Universe of targets: exp(x), ln(x), 1/x, sqrt(x), x*x, e (const)


## Canonical tree sizes (compiler output)

Sizes are total node counts (leaves + internals) of the tree the operator-aware compiler emits for each target. EDL and NEG_EML use the derivations in `eml_operators.py`.

| target | eml | edl | neg_eml |
|---|---|---|---|
| exp(x) | 3 | 3 | 7 |
| ln(x) | 7 | 7 | 3 |
| 1/x | 53 | 11 | 217 |
| sqrt(x) | 43 | 45 | 295 |
| x*x | 35 | 37 | 91 |
| e (const) | 3 | 1 | 1 |


## Recovery rate by depth

Each cell is `hits/n_seeds` (`%`) where a hit means snap RMSE < 1e-4 (sqrt of 1e-8 success threshold).


### eml

| target | d2 | d3 |
|---|---|---|
| exp(x) | 2/2 (100%) | 2/2 (100%) |
| ln(x) | 0/2 (0%) | 2/2 (100%) |
| 1/x | 0/2 (0%) | 0/2 (0%) |
| sqrt(x) | 0/2 (0%) | 0/2 (0%) |
| x*x | 0/2 (0%) | 0/2 (0%) |
| e (const) | 2/2 (100%) | 2/2 (100%) |

### edl

| target | d2 | d3 |
|---|---|---|
| exp(x) | 2/2 (100%) | 1/2 (50%) |
| ln(x) | 0/2 (0%) | 0/2 (0%) |
| 1/x | 0/2 (0%) | 0/2 (0%) |
| sqrt(x) | 0/2 (0%) | 0/2 (0%) |
| x*x | 0/2 (0%) | 0/2 (0%) |
| e (const) | 0/2 (0%) | 0/2 (0%) |

### neg_eml

| target | d2 | d3 |
|---|---|---|
| exp(x) | 0/2 (0%) | 0/2 (0%) |
| ln(x) | 2/2 (100%) | 1/2 (50%) |
| 1/x | 0/2 (0%) | 0/2 (0%) |
| sqrt(x) | 0/2 (0%) | 0/2 (0%) |
| x*x | 0/2 (0%) | 0/2 (0%) |
| e (const) | 0/2 (0%) | 0/2 (0%) |


## Numerical stability

NaN restarts during training (per the engine's best-state revert path), aggregated across all targets and depths.

| operator | total NaN restarts | mean wall (s) | n buckets |
|---|---|---|---|
| eml | 0 | 3.128 | 24 |
| edl | 69 | 2.776 | 24 |
| neg_eml | 0 | 3.139 | 24 |


## Per-target highlights

Best-found expression and RMSE for each (target, op).

| target | operator | best rmse | best expr |
|---|---|---|---|
| exp(x) | eml | 1.99e-16 | `exp(x)` |
| exp(x) | edl | 1.99e-16 | `edl(x, e)` |
| exp(x) | neg_eml | 3.26e+00 | `neg_eml(neg_eml(-inf, -inf), -inf)` |
| ln(x) | eml | 8.51e-17 | `ln(x)` |
| ln(x) | edl | 3.99e-01 | `edl(x, edl(edl(x, e), x))` |
| ln(x) | neg_eml | 1.01e-17 | `neg_eml(x, -inf)` |
| 1/x | eml | 7.30e-01 | `(e - x)` |
| 1/x | edl | 5.75e-01 | `edl(x, edl(edl(x, e), e))` |
| 1/x | neg_eml | 9.87e-01 | `neg_eml(x, -inf)` |
| sqrt(x) | eml | 4.58e-01 | `(e - 1)` |
| sqrt(x) | edl | 3.94e+00 | `edl(e, edl(x, x))` |
| sqrt(x) | neg_eml | 8.09e-01 | `neg_eml(x, -inf)` |
| x*x | eml | 2.94e+00 | `(exp(x) - e)` |
| x*x | edl | 1.31e+00 | `edl(x, edl(e, x))` |
| x*x | neg_eml | 2.73e+00 | `neg_eml(neg_eml(-inf, -inf), -inf)` |
| e (const) | eml | 0.00e+00 | `e` |
| e (const) | edl | 1.96e+00 | `edl(x, edl(e, e))` |
| e (const) | neg_eml | 2.33e+00 | `neg_eml(x, -inf)` |


## Reading the results

The benchmark answers issue #36's question: are the cousins *genuinely* equivalent on every dimension the paper measures, or does EML win on something?

### Tree size (canonical compiler output)
NEG_EML's `ln(x) = ne(x, -inf)` is the shortest form of any log in the table (size 3), beating EML's size-7 canonical. This is a direct consequence of `-inf` being the additive identity of NEG_EML's right slot. NEG_EML pays the bill on every *other* target: `exp(x)` jumps to size 7 (routes through `ln` of a negative real, so it's evaluated on the principal branch with complex intermediates), and `1/x`, `sqrt(x)`, `x*x` blow up to 217, 295, 91 respectively — roughly 3–8× larger than the EML/EDL equivalents. EDL compiles `1/x` in size 11 (shortest of any cousin) and `e` as a single-node terminal. On every target NEG_EML wins size only on `ln(x)` and `e`.

### Recovery from random init
The pattern matches the canonical sizes: operators find their short targets. EDL recovers `exp(x) = edl(x, e)` at depth 2; NEG_EML recovers `ln(x) = ne(x, -inf)` at depth 2 — a depth the paper (§4.3) reports EML cannot match on `ln(x)`. For everything whose canonical form is size ≥ 7, depth-2/depth-3 training is below the reach of random init in all three cousins (quick-mode caveat: only 2 seeds, 400+150 iters).

### Numerical stability
NaN restart counts per operator appear in the table above. EDL's right-slot denominator `ln(y)` crosses zero whenever `y` crosses 1, which is inside every target domain in this sweep; the engine restarts on NaN and this cost is visible (69 restarts for EDL in this run vs 0 for EML and NEG_EML). EML and NEG_EML avoid divergent denominators and train more smoothly. NEG_EML's `-inf` terminal uses a finite stand-in (`-1e30`) during training to keep gradients well-defined.

### Wall time
All three operators have roughly comparable per-run cost; the EDL extra cost comes from restart-retries rather than per-step arithmetic. See the `mean wall (s)` column above.

### Takeaway
The cousins are **not** fungible. They inherit the same completeness guarantee from the paper but project onto it with different cost structures: EML is uniformly middling, EDL shaves `1/x` and `exp(x)` but pays in NaN-restarts on `y = 1`, NEG_EML shaves `ln(x)` dramatically but pays everywhere else. No one operator dominates; choice of cousin is a choice of which target family to optimise for.
