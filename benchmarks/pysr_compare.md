# eml-sr vs PySR — head-to-head

> **This file is a stub.** Running the benchmark requires the PySR
> package plus a Julia toolchain, which we don't ship in CI. Regenerate
> with real numbers by running:
>
> ```bash
> pip install pysr
> python -c "import pysr; pysr.install()"   # one-time Julia deps
> PYSR_ENABLED=1 python -m benchmarks.pysr_compare \
>     --output benchmarks/pysr_compare.md
> ```
>
> A first-run baseline without PySR (eml-sr columns only) is also useful —
> omit `PYSR_ENABLED` and the PySR columns are skipped. See
> `benchmarks/pysr_compare.py` for methodology.

## What this benchmark measures

Per-target recovery comparison across two engines on the same data:

- **eml-sr** (this repo): single-primitive grammar `S → 1 | x | eml(S, S)`.
  Two configurations compared — `discover` (fixed-depth ladder) and
  `discover_curriculum` (growing tree).
- **PySR** (Cranmer, MIT): heterogeneous AST over `{+, -, *, /, exp, log, sqrt}`,
  C-compiled evolutionary search. Two configurations — defaults and a
  more-permissive budget.

Target set: the univariate Feynman slice from `benchmarks/feynman.py` plus
a small multivariate subset mirroring `tests/test_e2e_multivariate.py`.

## Metrics

Per target formula:

- **Wall time** — total, not time-to-first-hit. Both engines run their
  full budget; recovery flag comes from final RMSE.
- **RMSE** — in original coordinate space (not normalized).
- **Exact flag** — `RMSE < 1e-6` (machine-epsilon for reasonable y magnitudes).
- **Size** — node count. For eml-sr, `n_internal + n_leaves` of the snapped
  tree. For PySR, AST complexity from `PySRRegressor.get_best()`.
- **Ref EML size** — `tree_size(compile_expr(formula))`: the compiler's
  canonical EML tree for the target, independent of search. Sets the
  "apples-to-apples" baseline for the uniformity cost.

## Positioning

Three axes the benchmark makes concrete, even before real numbers land:

1. **Grammar uniformity.** Every eml-sr node is `eml`. PySR expressions are
   heterogeneous. The "ref EML size" column shows what uniformity costs.

2. **Exact recovery vs Pareto front.** eml-sr targets machine-epsilon RMSE on
   a specific symbolic form. PySR trades complexity for RMSE and often lands
   *near* the target but not *on* it.

3. **Expression length.** EML trees for standard operations are long by
   design — multiplication is depth 8 in EML, depth 1 in a conventional AST.
   That's the cost of grammar uniformity, not a bug.

PySR will usually be faster and find more formulas on harder targets. eml-sr
recovers exact symbolic forms cleanly when the target is inside its reachable
vocabulary, and the tree is always a pure `eml` circuit.

## Out of scope for this benchmark

- Not publication-grade. A positioning tool, not a paper.
- Not tuning PySR to lose. Uses PySR defaults plus one reasonable upgrade.
- Not proving eml-sr is "better". The uniformity claim is valuable on its
  own terms; this benchmark documents the cost.

## Stretch

- **gplearn** (scikit-learn-style, Python-only): closer substrate to eml-sr
  than PySR. Worth a column if someone adds it.
- **SymbolicRegression.jl / DSR**: full comparison out of scope.
- **`VerifyBaseSet`** (paper author's Rust): if a Python callable surfaces,
  worth a row.
