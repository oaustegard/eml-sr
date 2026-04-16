# CLAUDE.md

Guidance for AI coding agents (Claude Code, etc.) working on this repository.

## What this repo is

`eml-sr` is a symbolic-regression engine built on a single binary operator:

```
eml(x, y) = exp(x) − ln(y)
```

Together with the constant `1`, this operator generates all standard elementary
functions. The repo trains full binary trees of EML nodes with gradient descent
(Adam) + hardening, then snaps soft weights to exact symbolic form and
simplifies via a recursive AST rewriter.

## Source of truth

**`docs/odrzywolek-2026-eml.pdf`** — Odrzywolek, A. "All elementary functions
from a single operator" (arXiv:2603.21852, April 2026). This is the paper the
whole repo is an implementation of. **Read it before making non-trivial
changes.** In particular, before changing:

- the definition of `eml` or its branch conventions
- the set of terminal symbols (currently `{1, x}` for univariate; the paper's
  pure two-button form is `{1}`)
- any of the identity rewrites in the simplifier
- training dynamics (leaf gating, logit-softmax parameterization, hardening)

The paper's Figure 1 ("phylogenetic tree") and Table 4 (EML complexity of each
standard function) are the canonical reference for which identities should hold.

## Non-negotiable invariants

These are fixed by the paper and must not drift during refactoring:

1. **Operator definition.** `eml(x, y) = exp(x) − ln(y)`. Non-commutative.
   First argument goes through `exp`, second through `ln`, result is the
   difference. Do not "symmetrize" or swap.

2. **Complex internals, real surface.** Computations run over `C` using the
   principal branch of `ln`. Real-valued targets are the common case, but the
   internal dtype must support complex intermediates (see §4.1 of the paper:
   generating `i`, `π`, and trig functions requires `ln` of negative reals).
   PyTorch: use `torch.complex128`. NumPy: `complex128`.

3. **Grammar.** `S → 1 | x | eml(S, S)` for univariate. Every non-leaf is an
   EML node. Leaves are either the constant `1` or the input variable.

4. **Constant `1` is load-bearing.** It neutralises the `ln` branch via
   `ln(1) = 0`. Do not replace it with `0` or another constant without
   re-deriving the bootstrap chain from scratch.

5. **Canonical identities** (used by the simplifier; verified in the paper):
   - `exp(x)       = eml(x, 1)`
   - `ln(x)        = eml(1, eml(eml(1, x), 1))`
   - `e            = eml(1, 1)`
   - `-x`, `1/x`, `x·y`, `x+y`, etc. — see Figure 1 and Table 4 of the paper
     for the canonical trees. Do not invent new rewrites without verifying
     numerically first.

6. **IEEE-754 edge cases are intentional.** Some EML expressions rely on
   `ln(0) = -∞` and `exp(-∞) = 0` (signed infinities + signed zeros). This
   works in NumPy/PyTorch/C `<math.h>` but fails in pure Python (raises) and
   in Lean 4 (junk value). Do not "fix" code that looks like it's relying on
   infinities — it probably is, deliberately.

## Architecture quick-reference

| File | Role |
|------|------|
| `eml_sr.py` | Main engine. `EMLTree` (trainable binary tree), `discover()`, `discover_curriculum()`, CLI, simplifier. |
| `eml_sr_sklearn.py` | `EMLRegressor` with `fit`/`predict` + `model_` attribute for SRBench compatibility. |
| `eml_sr_hybrid.py` | Hybrid variant — see docstring; not the primary entry point. |
| `eml_sr_linear.py` | Linear-combination variant — see docstring. |
| `benchmarks/feynman.py` | Curated univariate slices of AI-Feynman. |
| `legacy/*.mojo` | Archived parabolic-attention stack-machine prototype. Historical; do not modify in place. |
| `docs/odrzywolek-2026-eml.pdf` | The paper. |

## How to stay faithful to the paper

When asked to add a capability, first check whether the paper already describes
how to do it:

- **New elementary target** (e.g. `erf`, `gamma`): the paper's scope is the
  scientific-calculator basis in Table 1. Anything outside that list is a
  research extension, not a paper-faithful implementation. Say so explicitly.
- **"Can we do multivariate?"** Yes, mentioned in §4.3 (master formula
  generalises to arbitrary input variables). Preserve the `αᵢ + βᵢx + γᵢf`
  soft-routing pattern; add more `βᵢⱼ xⱼ` slots per additional variable.
- **"Can we use a different operator?"** §5 lists two cousins: `edl(x, y) =
  exp(x)/ln(y)` with constant `e`, and `−eml(y, x) = ln(x) − exp(y)` with
  constant `−∞`. These have the same completeness property. Do not invent a
  new operator and claim it works without running `VerifyBaseSet`-style
  bootstrapping (the paper's Rust re-implementation is at the author's repo,
  `SymbolicRegressionPackage`).
- **"Recovery rate at depth N":** §4.3 reports 100% at depth 2, ~25% at 3–4,
  <1% at 5, 0% at 6 from random init. Curriculum / tree-growing (this repo's
  `discover_curriculum`) beats this substantially but don't claim exact
  numbers you haven't measured.

## Common pitfalls (observed in practice)

- **`exp` overflow.** Nested `exp(exp(x))` saturates float64 at ~709. The
  training loop clamps `exp` arguments; do not remove the clamp without a
  replacement. `Normalizer` (minmax/standard) is the first line of defence
  for real-world data — don't skip it.
- **Complex NaN propagation.** `exp(inf + 0j) * sin(0)` is `NaN`, not `0`.
  Guard with explicit IEEE-754 checks before trusting intermediates (the
  legacy Mojo harness caught 109/109 cases this way).
- **Branch of `ln`.** Principal branch. `ln(-1) = iπ`, not `-iπ`. If the
  derived `i` comes out with the wrong sign, the fix is in the branch
  convention, not in patching downstream constants. See §4.1.
- **"Simplifying" identity chains.** The depth-8 tree for multiplication in
  Table 4 is genuinely shorter in EML than the shortest-known alternative.
  Do not "simplify" it to an algebraic form and then re-emit — the point is
  that multiplication is not a primitive here.

## Before committing

For anything that changes math behavior (training loop, simplifier, identity
table, operator definition):

1. Re-read the relevant section of the paper.
2. Run the benchmark suite: `python -m benchmarks.feynman --workers 8`.
3. Verify the canonical recovery rates from the README table still hold
   (`exp` depth 1, `ln` depth 3, RMSE at machine-epsilon).
4. If you touch the simplifier, round-trip every identity in §3 of the paper
   numerically (plug in a transcendental like `γ ≈ 0.577216` and compare).

For pure code hygiene (refactors, typing, docstrings, tests that don't change
math): the above is overkill. Just keep the invariants above intact.

## References

- Paper: `docs/odrzywolek-2026-eml.pdf` / arXiv:2603.21852
- Author's reference implementation: github.com/VA00/SymbolicRegressionPackage
- Companion blog posts:
  - muninn.austegard.com/blog/two-buttons-and-a-constant.html
  - muninn.austegard.com/blog/two-buttons-back-row.html
- Interactive demo: austegard.com/fun-and-games/eml-calc.html
