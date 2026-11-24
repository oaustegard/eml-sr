# Permanent vs determinant as pure EML trees

How many `eml` nodes does the smallest tree in `S -> 1 | x_i | eml(S, S)` need to compute the 2x2 and 3x3 permanent and determinant? Prompted by the claimed VP != VNP proof (hexagon:2611.00954v1), whose bound is for `+`/`x` circuits.

Matrix convention: 2x2 is `[[x0, x1], [x2, x3]]`, so `det = x0*x3 - x1*x2` and `perm = x0*x3 + x1*x2`. 3x3 is row-major `x0..x8`. Size counts `eml` nodes only; leaves are free. No constants other than `1`, no affine leaves, and no tree that relies on `ln 0 = -inf`.

Scripts: `benchmarks/perm_det_eml.py` (enumeration and joins), `benchmarks/perm_det_gadgets.py` (constructions). Results: `benchmarks/results/perm_det_eml/`. Tests: `tests/test_perm_det_eml.py`.

## Results

| target | lower bound (search) | upper bound (verified tree) | depth of that tree |
|---|---|---|---|
| 2x2 det | > 9 real, > 8 complex | **17** | 10 |
| 2x2 perm | > 9 real, > 8 complex | **25** | 16 |
| 3x3 det | trivial only (>= 8) | **77** | 18 |
| 3x3 perm | trivial only (>= 8) | **101** | 36 |

The 17- and 25-node trees are exact on the complex principal branch for every nonzero real input (sympy reduces `tree - target` to 0, 4000 random points agree to 5e-15, 30 points agree at 40 digits). So are the 3x3 trees (sympy identity, 4000 points to 2.4e-14). On the real branch, where every `ln` argument must be a positive real, the domains shrink; see Domains.

Building blocks, minimal on the real branch (frontier or root join, sampling box `[0.2, 5]^4`):

| function | size | tree |
|---|---|---|
| `x - y` | 5 | `e(e(1,e(e(1,x),1)),e(y,1))` = `e(ln x, exp y)` |
| `ln x + ln y` | 7 | `e(1,e(e(e(1,e(e(1,e(1,y)),1)),x),1))` |
| `x * y` | 8 | `e(e(1,e(e(e(1,e(e(1,e(1,y)),1)),x),1)),1)` |
| `x / y` | 8 | `e(e(e(1,e(e(1,e(1,y)),1)),e(e(1,x),1)),1)` |
| `exp(x*y)` | 9 | `e(e(e(1,e(e(e(1,e(e(1,e(1,y)),1)),x),1)),1),1)` |
| `x + y` | 9 | `e(1,e(e(e(1,e(e(1,e(1,e(y,1))),1)),e(x,1)),1))` (`LNPROD(exp x, exp y)`), exact for `y < e` on the real branch and everywhere on the complex branch; `e(x,e(e(e(1,e(e(1,e(x,e(x,1))),1)),e(y,1)),1))` is also 9 and exact on all 1000 fresh points of the box |
| `x*y*z`, `-x*y`, `exp(-x*y)` | > 9 | |

## The trees

Gadgets (sizes add; `|w|` is the size of a subtree):

- `EXP(x) = e(x,1)`, `LN(x) = e(1,e(e(1,x),1))`, `SUB(x,y) = e(LN x, EXP y)`.
- Accumulator `F(w1..wk) = e - ln w1 - ... - ln wk`: `F1 = e(1, w1)`, `Fj = e(LN F(j-1), wj)`.
- `LNPROD(w1..wk) = e(1, EXP F) = sum ln wi`, size `4k - 1 + sum|wi|`. `PROD = EXP(LNPROD)`, size `4k + sum|wi|`.
- `SUM(v1..vk) = LNPROD(EXP v1, ..., EXP vk)`, size `5k - 1 + sum|vi|`.
- Peephole: `LN(PROD(...)) = LNPROD(...)`.

The enumerator's 7-node `ln x + ln y` and 8-node `x*y` are `LNPROD(x, y)` and `PROD(x, y)`.

- **det2 = 17**: `SUB(x0*x3, x1*x2) = e(LNPROD(x0,x3), EXP(PROD(x1,x2)))` = 1 + 7 + 1 + 8.
  `e(e(1,e(e(e(1,e(e(1,e(1,x0)),1)),x3),1)),e(e(e(1,e(e(e(1,e(e(1,e(1,x1)),1)),x2),1)),1),1))`
  The real-branch shape join (below) found the same tree independently.
- **perm2 = 25**: `SUM(x0*x3, x1*x2) = LNPROD(exp(x0*x3), exp(x1*x2))` = 7 + 9 + 9.
  `e(1,e(e(e(1,e(e(1,e(1,e(e(e(1,e(e(e(1,e(e(1,e(1,x0)),1)),x3),1)),1),1))),1)),e(e(e(1,e(e(e(1,e(e(1,e(1,x1)),1)),x2),1)),1),1)),1))`
- **det3 = 77**: cofactor expansion arranged so only subtraction is used, `a*m1 - (b*m2 - c*m3)` with each minor a det2 (17): `SUB(PROD(a,m1), SUB(PROD(b,m2), PROD(c,m3)))`. The flat `sum(P) - sum(Q)` form costs 105.
- **perm3 = 101**: the flat sum of six triple products, `SUM` over `PROD(a,e,i)` etc. (5*6 - 1 + 6*12). The cofactor form `SUM(a*p1, b*p2, c*p3)` costs 113.

Full 3x3 witnesses are in `results/perm_det_eml/gadgets.json`.

## Search

Bottom-up enumeration by size over leaves `{1, x0, x1, x2, x3}` with observational dedupe: each tree is evaluated at K = 6 random points and trees agreeing to 11 significant digits are one function. A tree is kept only if every `ln` argument is valid at every sample point (positive real on the real branch, nonzero on the complex one), with the exactness filter from `eml_complexity.py` (no term swallowed by the other) and no infinities.

- The frontier is stored with values to size N-1 and streamed at size N (keys and parents only).
- Root join `eml(L, R) = T <=> R = exp(exp(L) - T)` covers every tree of size <= N+1, so a miss proves size > N+1 for every tree that is exact at the sample points. Any tree exact on the whole sampling region is exact at the sample points, so the bound holds for those.
- Shape joins for upper bounds take one grandchild of the root from the leaves and look the other up: `eml(L, eml(Y, z))`, `eml(L, eml(z, Y))`, `eml(eml(Y, z), R)`, `eml(eml(z, Y), R)`. They reach size 18 for these shapes. A miss there proves nothing in general.

| branch | inputs | distinct functions per size 1..N | N | root join exact to | wall clock |
|---|---|---|---|---|---|
| real | log-uniform on `[0.2, 5]` | 25, 200, 1,443, 10,195, 72,616, 538,389, 4,068,475, 31,385,020 | 8 | 9 | ~5 min |
| complex | uniform on `[-3, 3]`, `|x| >= 0.1` | 25, 250, 2,573, 29,693, 361,080, 4,614,740, 60,960,736 | 7 | 8 | ~25 min |

Growth is about 7.6x per size on the real branch and 13x on the complex branch, so the next size up (real 9: about 2.4e8 functions; complex 8: about 7.5e8) does not fit the 4-core / 15 GB container these ran in. A 3x3 search over 9 leaves grows faster still; the 3x3 numbers above are constructions only.

On the complex branch with inputs of both signs, `x - y` is again 5 and det, perm, `x + y`, `exp(x*y)`, `-x*y` and `x*y*z` are all > 8. (`ln x + ln y` there is a different function from the tree's `ln(xy)` once both inputs are negative, so its complex row is not comparable.)

Shape joins on the real branch found det at 17 (the gadget tree) and nothing smaller in those shapes, and found no perm tree at all up to 18 in those shapes.

## Domains

- Complex principal branch: all trees in the table are exact for every nonzero real input. Each gadget uses `ln` only as `exp(ln w) = w` (true for every `w != 0`) or `ln(exp v) = v` with real `v`; the `2*pi*i` ambiguity of a sum of logs is removed by the final `exp`. Zero entries hit `ln 0` and are excluded. Large inputs overflow `exp(x*y)` in floating point; that is numerics, not the identity.
- Real branch (positive entries): `LNPROD(x, y)` needs `x < e^e ~ 15.15` (the first accumulator slot takes `ln(e - ln x)`); `y` is free. So det2 is exact for positive inputs with `x0, x1 < e^e`. perm2 additionally needs `x0*x3 < e` (the slot holds `exp(x0*x3)`), which is why it is exact on 3045 of 4000 points of `[0.2, 5]^4` and on all of `[0.2, 1.5]^4`. det3 on the real branch also needs the minors that sit in `ln` slots to be positive, which positive matrices do not guarantee (exact on 644 of 4000 points of `[0.2, 5]^9`).
- The enumeration's smallest hits are sometimes exact only on part of the box (`x + y` at 9 is exact for `y < e`); `results/perm_det_eml/2x2_real.json` lists, per target, both the smallest hit and the smallest one exact on 1000 fresh points.

## Interpretation

det2 is cheaper than perm2 because `eml` contains a subtraction: `x - y` is 5 nodes and `x + y` is 9. det2 is one subtraction of two products with the outer `ln` absorbed into the left product; perm2 has to route both products through `exp` and back. At 3x3 the gap is 77 vs 101 by construction. These are upper bounds; the proven lower bounds (> 9 for 2x2) are far below them, so the measured ratio is a ratio of constructions.

Nothing asymptotic follows from sizes at n = 2, 3. Two theory points bound what could:

- `det = exp(tr log M)` does not give a small exact EML tree. `tr log M` is `log det M`; computing it without first computing det needs a matrix logarithm, which is an infinite series or goes through the eigenvalues, so it is no finite shortcut. The constructions here are Laplace expansions, which grow factorially for both det and perm (EML trees are formulas: no sharing).
- EML size and `+`/`x` circuit size are polynomially related in one direction. A tree of size s that equals a degree-d polynomial on an open set can be turned into a `+`/`x` circuit of size poly(s, d) by expanding every `exp` and `ln` as a power series around a generic point and truncating at degree d (Strassen's division-elimination argument, which works for any analytic gate; constants become `e^c`, `ln b`, `1/k`). So if the claimed superpolynomial `+`/`x` bound for the permanent holds, it carries over to EML trees for the permanent up to a polynomial factor. It says nothing at n = 2 or 3, where its constants are unspecified. This transfer is a standard argument stated here, not a theorem from a paper.
