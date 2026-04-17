# eml-sr

**EML Symbolic Regression** — discover elementary formulas from data using a single operator.

Based on [Odrzywolek (2026)](https://arxiv.org/abs/2603.21852), "All elementary functions from a single operator":
the EML operator `eml(x, y) = exp(x) - ln(y)` (Exponent Minus Log), together with the constant `1`, generates all standard
elementary functions. This makes `S → 1 | x | eml(S, S)` a complete, regular search space for
symbolic regression.

## Quick start

```python
import numpy as np
from eml_sr import discover

x = np.linspace(0.5, 5.0, 50)
y = np.log(x)  # secret formula

result = discover(x, y, max_depth=4, n_tries=8)
print(result["expr"])  # → ln(x)
```

## How it works

A full binary tree of depth *n* has 2ⁿ leaves and 2ⁿ−1 internal nodes. Each leaf soft-routes
between the constant `1` and the variable `x`. Each internal node computes `eml(left, right)`,
and its gate 3-way-routes each child input between `1`, `x`, and the child's own output — so
`x` can flow straight into any gate, not just through leaves.

Training (Adam + tau-annealing) pushes the soft weights toward a single hard choice per gate,
recovering an exact symbolic expression. When the generating law is elementary, the snapped
weights yield machine-epsilon RMSE. The final expression is then simplified by a recursive
AST rewriter that applies EML identities (e.g. `eml(1, eml(eml(1, x), 1)) → ln(x)`).

### Current recovery rates

| Target | Depth | RMSE |
|--------|-------|------|
| `x` (identity) | 0 | 0 |
| `1` (constant) | 0 | 0 |
| `exp(x)` | 1 | 4.6e-16 |
| `e` (constant) | 1 | 0 |
| `ln(x)` | 3 | 7.3e-17 |
| `exp(x) - ln(x)` | 1 | 3.2e-16 |

Depth 0 is a single leaf with no gates — the "passthrough" vocabulary
`{1, x}`. Without it, the identity `y = x` is unreachable until depth ≥ 4,
because every gated output is wrapped in an outer `eml(·,·)`. The depth
ladder in `discover()` now starts at 0 to close that gap (issue #11).

## Curriculum learning (growing trees)

At deep depths, random initialization becomes a needle-in-a-haystack search —
the paper reports 0% recovery at depth 6 from random init. `discover_curriculum`
grows the tree one leaf at a time as a warm start:

1. Start at depth 1 (1 internal node, 2 leaves)
2. Train to convergence
3. If the fit isn't good enough, pick the leaf with the largest gradient
   magnitude (the one most "wanting" to change) and split it — replacing it
   with an `eml(x, 1) = exp(x)` subtree, and biasing the parent gate to
   route through the new subtree
4. Continue training and repeat until `max_depth` or exact recovery

```python
from eml_sr import discover_curriculum
result = discover_curriculum(x, y, max_depth=6, n_tries=8)
```

This is analogous to Net2Net / progressive growing in neural architecture
search. It solves nested compositions like `exp(exp(exp(x)))` several times
faster than random init, and lets shallow solutions seed deeper ones.

## CSV ingestion

Run eml-sr against any two-column CSV from the command line:

```bash
python eml_sr.py csv data.csv --x-col time --y-col temperature \
    --max-depth 5 --tries 16 --workers 8
```

Options:
- `--method {discover,curriculum}` — fixed-depth ladder vs growing tree
- `--normalize {minmax,standard,none}` — affine pre-scaling (EML overflows
  on un-scaled data; `minmax` is the safest default)
- `--workers N` — parallel seed processes for `discover` (~linear speedup)

The discovered formula is reported in normalized coordinates along with
the affine transformation needed to convert it back. Use `--normalize none`
when your data is already in a numerically friendly range — that's the
only setting that yields a directly-readable formula in the original units.

## sklearn interface (for SRBench)

`EMLRegressor` exposes the standard `fit`/`predict` API plus the `model_`
attribute SRBench (La Cava et al. 2021) reads to extract the symbolic
expression:

```python
from eml_sr_sklearn import EMLRegressor

est = EMLRegressor(max_depth=4, n_tries=16, n_workers=8)
est.fit(X, y)
yhat = est.predict(X_test)
print(est.model_)            # symbolic form in normalized coords
print(est.original_model_)   # with the affine substitution stitched in
```

When `X` has multiple columns, `EMLRegressor` projects onto the column with
the highest absolute Spearman correlation with `y`. This puts eml-sr on the
SRBench leaderboard for univariate / near-univariate problems, while
flagging cleanly that the engine is single-variable by construction.

## Feynman benchmark

A curated set of univariate slices from the AI Feynman dataset
([Udrescu & Tegmark 2020](https://github.com/SJ001/AI-Feynman)) lives in
`benchmarks/feynman.py`:

```bash
python -m benchmarks.feynman --workers 8                # quick (8 problems)
python -m benchmarks.feynman --all --method curriculum  # full suite
```

Each problem reports recovery success, RMSE, depth used, and time to solution.

## Performance

- **`n_workers > 1`**: `discover()` accepts an `n_workers` argument that
  fans out per-depth seeds across worker processes. Each child process is
  pinned to a single torch thread to avoid oversubscription. Empirical
  speedup is near-linear up to `os.cpu_count()`.
- **Normalization**: pre-scaling with `Normalizer.fit(x, y, mode="minmax")`
  prevents the `exp(x)` chain inside the EML tree from saturating to the
  1e300 clamp, which would otherwise kill gradients on real-world data.
- **Curriculum**: for deep formulas, `discover_curriculum` warm-starts
  by growing the tree leaf-by-leaf, dramatically beating random init
  past depth 4.

## Files

| File | Description |
|------|-------------|
| `eml_sr.py` | Symbolic regression engine + CLI (the product) |
| `eml_sr_sklearn.py` | sklearn-style `EMLRegressor` for SRBench |
| `benchmarks/feynman.py` | Univariate Feynman-equation benchmark |
| `legacy/eml_executor.mojo` | Original parabolic-attention stack machine (archived) |
| `legacy/test_eml.mojo` | 109-test bootstrap chain verification (archived) |

## References

- Paper: [arXiv:2603.21852](https://arxiv.org/abs/2603.21852)
- Blog: [Two buttons and a constant](https://muninn.austegard.com/blog/two-buttons-and-a-constant.html)
- Blog: [Two buttons, back row](https://muninn.austegard.com/blog/two-buttons-back-row.html)
- Demo: [EML Calculator](https://austegard.com/fun-and-games/eml-calc.html)
