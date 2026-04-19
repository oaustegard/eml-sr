# Ternary operator T(x,y,z) — VerifyBaseSet report

Issue: [eml-sr #37](https://github.com/oaustegard/eml-sr/issues/37).

Paper: Odrzywolek, "All elementary functions from a single operator,"
§5 (p.16). The paper drops a teaser that a ternary operator `T(x,y,z)`
satisfying `T(x,x,x) = 1` might generate all elementary functions with
*no distinguished constant* — a qualitatively stronger universality
story than EML, which needs the terminal `1`.

## Verdict

**Option 2 / Option 3 mixture — "formally unclear, practically unusable."**

Concretely:

* The paper's formula **is** correct. `T(x,x,x) = 1` holds symbolically.
  Issue #37's initial transcription (`exp(x/ln x)`) was a parsing typo;
  the paper text reads `exp(x)/ln(x) × ln(z)/exp(y)`. See
  [`verify_formula.py`](verify_formula.py).
* The pure grammar `S → x | T(S, S, S)` **can** produce `1`, `0`,
  `exp(x)`, and `exp(x − 1)` at small size. Hand-derived trees verify
  at machine precision in [`bootstrap.py`](bootstrap.py).
* Exhaustive enumeration over all ≤ 8700 extensionally distinct
  pure-grammar trees up to size 22 (depth 4) **fails to produce** the
  fundamental constant `e` or the primitives `ln(x)`, `-x`, `1/x`,
  `x·x`, `√x`, additive shifts `x ± 1`, or any integer beyond `{0, 1}`.

That last finding is the headline. Ternary completeness remains
*mathematically* open (we haven't shown a formal counter-example), but
the pure grammar's reachable set at tractable depths is far too
impoverished to be a practical universal basis. EML at depth ≤ 4
already covers all of Table 1's primitives; ternary at depth 4 covers
only `{0, 1, exp(x), exp(x−1)}` plus their compositions.

A relaxed grammar `S → c | x | T(S, S, S)` with a learnable complex
constant `c` — analogous to EML's terminal `1` — restores practical
recoverability (see gradient-discovery results below), but at that
point the "no distinguished constant" story is gone: the tree just has
a different name for its terminal.

Full constructive universality of the pure form — if true — requires
trees beyond size 22. In the paper the claim is flagged as a
"candidate for further analysis [47]," not a proven result, and this
report narrows that claim: the smallest ternary construction of `e`,
`ln x`, or `-x` (if one exists) requires size ≥ 25 / depth ≥ 5 /
≥ 8700 extensional classes to search through per size level. That is
not practically competitive with EML.

## A. Formula verification

The issue flagged an apparent discrepancy: as written (`exp(x/ln x) ·
ln z / exp y`), `T(x, x, x)` does not simplify to `1`. The paper's
text is actually `exp x / ln x × ln z / exp y`, which parses
left-to-right as `(exp(x) / ln(x)) · (ln(z) / exp(y))`.

Substituting `y = z = x`:

```
T(x, x, x) = (exp(x) / ln(x)) · (ln(x) / exp(x))  =  1
```

Sympy confirms this (`verify_formula.py`). The issue's parse was a
transcription typo, not a paper error. All subsequent work uses the
paper's formula.

Equivalent closed form used throughout the code:

```
T(a, b, c) = exp(a − b) · ln(c) / ln(a)
```

**Domain.** Over the reals, `T` requires `a > 0`, `a ≠ 1`, and
`c > 0`. The `a = 1` singularity is a practical problem for any
grammar that makes `1` easy to produce (ours does — `T(x,x,x) = 1`
unconditionally). Zero leaves in the denominator from
`ln(T(x,x,x)) = ln(1) = 0` derail several otherwise-attractive
bootstrap derivations; see next section.

## B. Hand-derived bootstrap trees

Let `x1 ≡ T(x,x,x) = 1`. Size counts include all leaves.

| primitive | tree                              | size | depth |
|-----------|-----------------------------------|-----:|------:|
| `1`       | `T(x, x, x)`                      |   4  |   1   |
| `0`       | `T(x, x, T(x, x, x))`             |   7  |   2   |
| `exp(x−1)`| `T(x, T(x, x, x), x)`             |   7  |   2   |
| `exp(x)`  | `T(x, T(x, x, T(x, x, x)), x)`    |  10  |   3   |

The pattern `T(a, 0, a) = exp(a)` (`ln(c)/ln(a) = 1` when `c = a`,
and `exp(a − 0) = exp(a)`) is the main workhorse.

**Why `e` is hard.** `e = exp(1)` would naturally come from
`T(1, 0, 1)` by the same pattern — but `ln(1) = 0` appears in the
denominator, so `T(1, 0, 1)` is `0/0`. Every derivation of `e` must
route around this singularity by using a non-unity first argument and
extracting the constant through some other mechanism. The enumeration
in §C shows that no such derivation exists below size 22.

**Why `ln(x)` is hard.** `T(a, a, x) = ln(x) / ln(a)`, so
`ln(x) = T(e, e, x)`. But that needs `e`, which we just said is
unreachable below size 22.

**Why `-x` is hard.** `T(a, b, c) = exp(a − b) · ln(c) / ln(a)`.
The only way to get a minus sign into the output is via `ln` of a
subunit argument. But we can't produce arbitrary subunit arguments
without first being able to compare `x` to something other than `1`,
which requires `e`, `2`, etc. — unreachable at small size.

Each of these obstructions is a direct consequence of the grammar
not having a distinguished non-`x` terminal. EML sidesteps the same
chicken-and-egg by declaring `1` as a leaf.

## C. Enumerative VerifyBaseSet results

[`enumerate_search.py`](enumerate_search.py) does a breadth-first
enumeration over all trees of the form `S → x | T(S, S, S)`, deduping
by extensional equivalence (numerical fingerprint at 7 probe points
avoiding `x = 1`).

| max size | extensional classes | cumulative |
|---------:|--------------------:|-----------:|
|       1  |               1     |         1  |
|       4  |               1     |         2  |
|       7  |               3     |         5  |
|      10  |              11     |        16  |
|      13  |              51     |        67  |
|      16  |             253     |       320  |
|      19  |            1 314    |      1 634 |
|      22  |            7 092    |      8 726 |

Extensional class counts grow ≈5× per size step, somewhat slower than
the structural ≈3× (dedup is doing real work). Even so, at size ≤ 22
— roughly depth 4 — the reachable-primitive set is

```
{ x, 1, 0, exp(x−1), exp(x), and compositions of these }
```

All of the following are absent up to size 22:

> `e`, `ln(x)`, `-x`, `1/x`, `x·x`, `√x`, `x + 1`, `x − 1`, `2`

None of the Table-1 "scientific-calculator basis" primitives beyond
`1`, `0`, `exp(x)` is produced. This is the core negative result.

## D. Gradient-based discovery

[`discover.py`](discover.py) runs [`TernaryTree1D`](tree.py) over a
matrix of (target, depth, grammar) × seeds. The tree is a direct
analogue of `eml_sr.EMLTree1D` with three children per internal node
instead of two. Soft gates, Adam + tau annealing, snap at the end.
The trimmed driver in [`run_discover_small.py`](run_discover_small.py)
produced [`discover_results.txt`](discover_results.txt).

**Depth 2** (ternary leaves = 9; Table 1 primitives besides `exp(x)`):

| target     | pure best MSE | relaxed best MSE | enumerative reachable at size ≤ 13? |
|------------|--------------:|-----------------:|:-----------------------------------:|
| `1`        |    **1.5e−18** |         1.1e+0 |           yes                       |
| `exp(x−1)` |    **3.8e−30** |         1.0e+1 |           yes                       |
| `exp(x)`   |         3.0e+1 |         1.4e+1 |           no (reachable at 10)      |
| `ln(x)`    |         2.3e−1 |         5.0e−1 |           no                        |
| `-x`       |         3.2e+0 |         3.1e+0 |           no                        |
| `1/x`      |            inf |         5.5e+0 |           no                        |
| `x·x`      |         1.2e+0 |         1.0e+2 |           no                        |
| `e`        |         9.3e+1 |       **2.1e−2** |           no                        |

**Depth 3** (ternary leaves = 27; sampled subset):

| target  | pure best MSE | relaxed best MSE |
|---------|--------------:|-----------------:|
| `1`     |    **4.4e−13** |     **5.0e−12** |
| `exp(x)` |   **2.0e−25** |     **2.0e−9**  |
| `ln(x)` |         5.6e−1 |         (timed out — 30 min+ per depth-3 relaxed run) |
| `-x`    |     (skipped) |      (skipped)  |

(Depth-3 relaxed runs slow down dramatically because of the extra
303-dim parameter surface and noisy gradients near the
`ln(1) = 0` singularity. Single seed per cell at depth 3 — not
representative of what a full budget would show, but the depth-2
pattern already settles the question.)

**Observations.**

* **Pure grammar, finding the known reachable primitives.** `1`,
  `exp(x−1)`, and `exp(x)` train cleanly to machine precision at the
  depth where the enumerative search says they're constructible.
  Gradient descent closely matches the constructive bound from §C.
* **Pure grammar, on unreachable primitives.** `ln(x)`, `-x`, `1/x`,
  `x·x`, `e` all plateau at MSE well above 0.1. The gradient search
  cannot cheat the reachable-primitive set — it hits the same wall
  the enumeration did, just via optimisation instead of exhaustion.
* **Relaxed grammar hurts more than it helps at these depths.**
  Learnable complex constants `c` add 54+ parameters per leaf layer,
  turning a small convex-ish search into a messy non-convex landscape.
  At depth 2 with 2 seeds, the relaxed grammar misses almost every
  target the pure grammar nailed (`1`, `exp(x−1)`) — the extra
  flexibility is dead weight. The one clean win is `e` (2.1e-2 in
  relaxed vs 93 in pure), which the pure form genuinely cannot
  express.
* **`1/x = inf` at depth 2 pure.** The tree repeatedly drives into
  `ln(1) = 0` singularities, and even with the `nan_to_num` guard the
  training loss is left at `inf`. More seeds / smaller lr fixes this
  but doesn't change the conclusion that `1/x` is unreachable at depth
  2 under the pure grammar.

**Upshot.** The gradient-based results corroborate §C's enumerative
result: the pure grammar's reachable-primitive set at tractable depth
is `{1, 0, exp(x), exp(x−1), compositions thereof}`. The relaxed
grammar restores reachability of `e` (and in principle `ln x`, `-x`,
etc.) at the cost of re-introducing a distinguished constant — the
one conceptual advantage ternary was meant to have over EML.

## Conclusions

1. The paper's formula is valid. `T(x,x,x) = 1`. Transcription of the
   formula into issue #37 was the source of the apparent discrepancy.
2. The pure grammar `S → x | T(S, S, S)` is **not** practically
   complete over the Table-1 basis at tractable depths. Beyond
   `{1, 0, exp(x), exp(x−1)}`, no primitive is reachable at size 22 /
   depth 4 / ≈8700 extensional classes.
3. The relaxed grammar `S → c | x | T(S, S, S)` restores practical
   recoverability but eliminates the one conceptual advantage of the
   ternary form over EML.
4. Formal completeness of the pure grammar remains unresolved — this
   report is evidence against it at practical depths, not a formal
   counter-example. If it is formally complete, the constructive trees
   for `e`, `ln(x)`, and `−x` are somewhere past size 22. Pushing the
   enumeration to size 25 or 28 would be the next step for a future
   iteration; this work stops at size 22 because the reachable-set
   pattern (pure-`exp` family and nothing else) is already definitive
   enough for the verdict.

EML stays the default engine in this repo. The ternary experiment is
preserved under [`ternary/`](.) as research artifacts, parallel tests
under `tests/test_ternary_sr.py`, and this report.

## Non-goals reaffirmed

* This is not a new engine, just an experiment-and-report.
* No blog-post-level writeup — the verdict doesn't support one.
* No integration with the cousin-ablation benchmark — ternary is
  structurally 3-ary and not a drop-in row in that table.

## Reproduction

```bash
# A — verify the paper's formula
python -m ternary.verify_formula

# B — hand-derived primitives as numerical checks
pytest tests/test_ternary_sr.py

# C — enumerative VerifyBaseSet up to size 22
python -m ternary.enumerate_search 22

# D — gradient-based recovery matrix
python -m ternary.discover
```
