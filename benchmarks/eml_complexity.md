# EML complexity of constants

Minimal number of `eml` nodes in a closed tree `S -> 1 | eml(S,S)` evaluating exactly to the constant. Enumeration is bottom-up with observational dedupe (11 significant digits), an exactness filter (no term may be numerically swallowed by the other; `exp` underflow rejected), and every reported witness checked symbolically with sympy. Real branch: `ln` requires `y>0`. Complex branch: principal branch, near-real values snapped to the axis so `ln(-x) = ln x + i*pi` deterministically.

Two runs. PR #69 (2026-09-02, claude.ai, 1 core / 3 GB) enumerated real to size 18 and complex to 16. Issue #70 (2026-09-03/04, Claude Code on the Web, 4 cores / 15 GB / 25 GB disk) reached real 20 (328M distinct values) and complex 18 (94M) with `benchmarks/eml_complexity.py`, and extended the table past the frontier with the inverse joins in `benchmarks/eml_complexity_join.py`. JSON with every witness: `benchmarks/results/eml_complexity/`.

## Table conventions

- A plain number is the minimal size: the constant appeared on the enumerated frontier, or a join found it at size N+1 where N is the frontier, which is exact because both root children of any tree of that size were enumerated.
- `<= k` is an upper bound: a verified tree of size k found by the root join with both children enumerated. The join keeps one representative per 11-digit class, so a smaller tree whose partner shares a class with an earlier value is invisible to it. A bound is a witness; "not found" is not a lower bound beyond the frontier.
- `>N` means neither the frontier to N nor the joins over it produced a verified tree.

| constant | real | complex | witness (real; complex where real is n/a) |
|---|---|---|---|
| `1` | 0 | 0 | `1` |
| `e` | 1 | 1 | `e(1,1)` |
| `e-1` | 2 | 2 | `e(1,e(1,1))` |
| `e^e` | 2 | 2 | `e(e(1,1),1)` |
| `0` | 3 | 3 | `e(1,e(e(1,1),1))` |
| `ln(e-1)` | 5 | 5 | `e(1,e(e(1,e(1,e(1,1))),1))` |
| `e-2` | 7 | 7 | `e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1))` |
| `-1` | 8 | 8 | `e(e(1,e(e(1,e(1,e(1,1))),1)),e(e(1,1),1))` |
| `2` | 9 | 9 | `e(1,e(e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1)),1))` |
| `1/e` | 9 | 9 | `e(e(e(1,e(e(1,e(1,e(1,1))),1)),e(e(1,1),1)),1)` |
| `e^2` | 10 | 10 | `e(e(1,e(e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1)),1)),1)` |
| `-e` | 11 | 11 | `e(e(1,e(e(1,e(e(1,1),e(e(1,1),1))),1)),e(e(e(1,1),1),1))` |
| `-i*pi` | n/a | 11 | `e(1,e(e(1,e(e(1,e(1,e(e(e(1,1),1),1))),1)),e(e(1,1),1)))` |
| `e-3` | 12 | 12 | `e(e(1,e(e(1,e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1))),1)),e(1,1))` |
| `ln2` | 12 | 12 | `e(1,e(e(1,e(1,e(e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1)),1))),1))` |
| `-2` | 13 | 13 | `e(e(1,e(e(1,e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1))),1)),e(e(1,1),1))` |
| `2e` | 13 | 13 | `e(1,e(e(e(1,e(e(1,e(e(1,1),e(e(1,1),1))),1)),e(e(e(1,1),1),1)),1))` |
| `3` | 14 | 14 | `e(1,e(e(e(1,e(e(1,e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1))),1)),e(1,1)),1))` |
| `e/2` | 14 | 14 | `e(e(e(1,e(e(1,1),1)),e(1,e(e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1)),1))),1)` |
| `e^3` | 15 | 15 | `e(e(1,e(e(e(1,e(e(1,e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1))),1)),e(1,1)),1)),1)` |
| `1/2` | 17 | 15 | `e(e(e(1,e(e(1,e(1,e(1,e(e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1)),1)))),1)),e(e(1,1),1)),1)` |
| `sqrt(e)` | 18 | 16 | `e(e(e(e(1,e(e(1,e(1,e(1,e(e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1)),1)))),1)),e(e(1,1),1)),1),1)` |
| `2/3` | 19 | <= 37 | `e(e(1,e(e(1,1),e(e(e(1,e(1,e(e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1)),1))),e(e(e(1,1),1),1)),1))),1)` |
| `-3` | 20 | 17 | `e(e(1,e(e(1,e(e(1,e(e(1,e(e(1,e(e(1,e(e(1,1),e(1,1))),1)),e(1,1))),1)),e(1,1))),1)),e(e(e(1,1),1),1))` |
| `e-4` | 21 | 17 | `e(e(1,e(e(1,e(1,e(1,1))),1)),e(e(1,e(e(e(1,e(e(1,e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1))),1)),e(1,1)),1)),1))` |
| `4` | 21 | 19 | `e(e(1,1),e(e(e(1,e(e(1,e(e(1,e(e(1,e(e(1,e(e(1,e(e(1,1),e(1,1))),1)),e(1,1))),1)),e(1,1))),1)),e(1,1)),1))` |
| `3/2` | <= 23 | <= 20 | `e(e(1,e(e(1,1),1)),e(e(e(e(1,e(e(1,e(1,e(1,e(e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1)),1)))),1)),e(e(1,1),1)),e(1,1)),1))` |
| `1/3` | <= 24 | <= 36 | `e(e(1,e(e(1,1),1)),e(e(e(1,e(e(1,1),e(e(e(1,e(1,e(e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1)),1))),e(e(e(1,1),1),1)),1))),1),1))` |
| `-4` | <= 27 | <= 26 | `e(e(1,e(e(1,e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1))),1)),e(e(1,e(e(e(1,e(e(1,e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1))),1)),e(e(1,1),1)),1)),1))` |
| `5` | <= 32 | <= 31 | `e(e(1,e(e(1,e(1,e(e(e(1,e(e(1,e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1))),1)),e(1,1)),1))),1)),e(e(e(1,e(e(1,e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1))),1)),e(e(1,1),1)),1))` |
| `-5` | <= 34 | <= 31 | `e(e(1,e(e(1,e(e(1,e(e(1,e(e(1,e(e(1,e(e(1,1),e(1,1))),1)),e(1,1))),1)),e(1,1))),1)),e(e(1,e(e(e(1,e(e(1,e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1))),1)),e(e(e(1,1),1),1)),1)),1))` |
| `6` | <= 39 | <= 36 | `e(e(1,e(e(1,e(e(1,1),e(e(e(1,e(e(1,e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1))),1)),e(1,1)),1))),1)),e(e(e(1,e(e(1,e(e(1,e(e(1,e(e(1,e(e(1,e(e(1,1),e(1,1))),1)),e(1,1))),1)),e(1,1))),1)),e(e(1,1),1)),1))` |
| `-6` | >20 (joins: none) | <= 37 | complex: `e(e(1,e(e(1,e(e(1,e(e(1,e(e(e(1,e(e(1,1),1)),e(e(1,1),e(e(1,1),1))),e(1,1))),1)),e(1,1))),1)),e(e(e(1,e(e(1,1),1)),e(e(e(1,e(e(1,e(e(1,e(1,e(e(e(1,1),1),1))),e(1,1))),1)),e(1,1)),1)),1))` |
| `1/4` | >20 (joins: none) | >18 (joins: none) | |
| `pi` | >20 (joins: none) | >18 (joins: none) | |
| `sqrt2` | >20 (joins: none) | >18 (joins: none) | |
| `phi` | >20 (joins: none) | >18 (joins: none) | |
| `ln pi` | >20 (joins: none) | >18 (joins: none) | |
| `i*pi` | n/a | <= 23 | `e(e(1,e(1,e(1,e(e(e(1,1),1),1)))),e(e(1,e(e(1,e(1,e(e(e(e(1,e(1,e(1,e(e(e(1,1),1),1)))),1),1),1))),1)),e(e(1,1),1)))` |
| `i` | n/a | >18 (joins: none) | |
| `1+i` | n/a | >18 (joins: none) | |
| `e^i` | n/a | >18 (joins: none) | |

Every witness in the table passed the sympy check (`|x - t| < 1e-20` at 30 digits). One witness is multiplication-shaped: the complex `6` at size 36 computes `3 * 2` from a size-14 subtree for `3` and a size-9 subtree for `2`, 23 nodes of factors and 13 of `exp(ln x + ln y)` scaffolding. It is a bound, and it is 3 below the best subtraction-built real `6` (<= 39); with `5` at 31 and 32 on the two branches, size 6 is where a product first undercuts the ladder in these results. No other witness has a node equal to the product of two of its non-trivial descendants.

## Distinct exact values per size (vs Catalan)

| size | real | complex | Catalan |
|---|---|---|---|
| 0 | 1 | 1 | 1 |
| 1 | 1 | 1 | 1 |
| 2 | 2 | 2 | 2 |
| 3 | 5 | 5 | 5 |
| 4 | 10 | 10 | 14 |
| 5 | 27 | 28 | 42 |
| 6 | 73 | 79 | 132 |
| 7 | 197 | 228 | 429 |
| 8 | 545 | 676 | 1430 |
| 9 | 1518 | 2034 | 4862 |
| 10 | 4326 | 6242 | 16796 |
| 11 | 12455 | 19389 | 58786 |
| 12 | 36210 | 60776 | 208012 |
| 13 | 106002 | 192014 | 742900 |
| 14 | 311948 | 610280 | 2674440 |
| 15 | 921965 | 1949625 | 9694845 |
| 16 | 2731278 | 6254027 | 35357670 |
| 17 | 8117499 | 20144271 | 129644790 |
| 18 | 24239997 | 65186623 | 477638700 |
| 19 | 72697426 | 211882756 (unverified) | 1767263190 |
| 20 | 218660435 |  | 6564120420 |

Distinct/Catalan keeps decaying geometrically: real 0.0626, 0.0507, 0.0411, 0.0333 for sizes 17 to 20 (0.81 per size), complex 0.177, 0.155, 0.137 for 16 to 18 (0.88 per size). At size 20 one real tree in thirty computes a new constant.

### Count drift from PR #69

The sequences above differ from PR #69 by 1 at real 13, 2 at 14, 1 at 16, 50 at 17 and 155 at 18, and by 1, 1, 1, 5, 14, 60 at complex 11 to 16. Both runs use the same `qkey` and `eml_vec`. The CCotw engine stores each level sorted by key, so pairs are visited in a different order and a different float can be kept as the representative of an 11-digit class; a child within float noise of a class boundary then lands one class over. Float noise near 1e-16 against 1e-11 class spacing over 8M values predicts about 80 such events at real 17, against 50 observed. The known minimal sizes all reproduced.

## Reading

- **Real `4` is 21, exactly.** The real `e-k` ladder dies at `e-3 < 0`, so `4` cannot be `eml(1, exp(e-4))` with a real `e-4`. The size-21 tree is `eml(e, y)` with `y` at size 20: `4 = e - ln y`, `y = exp(e-4)` built as `exp(exp(1) - ln(...))` from a size-18 core. Real `e-4` is 21 as well, from a different tree. On the complex branch `e-4` is 17 as the ladder predicts, and `-3` is 17 there against 20 real.
- **Negative integers keep the 5-per-unit slope.** `-3` = 20, `-4` <= 27, `-5` <= 34: each join witness is the previous one wrapped in the same 7-node step `-(k+1) = ln(exp(-k) ... )`, so the true slope is at most 7 and the ladder argument says at least 5. `-6` has no verified witness from the size-20 cache.
- **Positive integers past 4 are expensive.** Real `5` <= 32 and `6` <= 39, complex `5` <= 31 and `6` <= 36, are the only verified trees; 7 and 8 have none on either branch. Mahler-Popken integer complexity (1s with + and x) reaches 6 with 5 symbols. The complex `6` at <= 36 is the first witness built by multiplication (`3 * 2`); every other integer in the table is built by subtraction and exponentials.
- **Fractions.** `2/3` is 19 on the real frontier; `3/2` <= 23 and `1/3` <= 24 by real join, `3/2` <= 20 by complex join; `1/4` has no verified witness from either cache. The complex detour that made `1/2` cheaper (15 vs 17) does not show up for `2/3` (<= 37 complex against 19 real).
- **Transcendentals and `i`.** `pi`, `sqrt2`, `phi` and `ln pi` have no verified tree at any size the joins reach, on either branch. `i*pi` is <= 23 on the complex branch (`ln(-1)` costs 3 on top of `-1`'s 8, and the join finds a 23-node route); `i`, `1+i` and `e^i` have none, so dividing `i*pi` by `pi` is not on the table either.

## Inverse joins and the exact check

The root join solves `b* = exp(exp(a) - t)` for every cached `a` (328M on real 20) and looks `b*` up; a hit with `size(a) + size(b) + 1 <= N+1` is the exact minimum, anything larger is a bound. The first full run confirmed hits by recomputing the tree from the cached partner and comparing 11-digit keys, and reported 55 rows for real 19 of which 28 failed the sympy pass, every join row above size 23. At 1e8 candidates against 1e8 cached classes the key match admits thousands of coincidences per target, and since a hit larger than the current best is never examined, the smallest false one displaces any true one. Every hit now has to pass a 40-digit mpmath evaluation before it can become the best (0.6 ms per check). On the real size-20 cache the check rejected 244 key hits for `-5`, 35 for `5`, 5,579 for `6`, and 24k to 99k for each transcendental target; on complex 18, 409 for `1/3`, 561 for `6`, and 1,000 to 2,200 for each of `pi`, `sqrt2`, `phi`, `ln pi` and `1/4`. The survivors are the bounds in the table.

The two-level join (`t = eml(a, eml(c, d))` and `t = eml(eml(c, d), b)`, inner operand at size <= 4) costs 1.2e10 pair evaluations per target on the size-20 cache and rejected 850k and 1.2M coincidences on its first two targets over about ten hours without a verified find. It is left in the script with `--join2 K`; the table used `K=0` (inner operand the leaf `1`), which costs 1.9e8 pairs per target on complex 18 and found nothing there either.

## The compiler against the minima

`benchmarks/compiler_vs_minimal.py` compiles a few natural elementary expressions for each constant with `eml_compiler` in strict mode (leaves are the constant `1` only), verifies each tree numerically, and keeps the smallest. "inf steps" counts subtrees whose float64 value is infinite: the compiler's strict negation is `sub(ln(1), x)`, which goes through `ln(0) = -inf` and `exp(-inf) = 0`, and its addition is `a - (-b)`, so every `+` and every unary minus takes that route. The enumeration rejects non-finite intermediates, so the two columns measure two grammars: finite-valued closed trees for the minima, IEEE-754 closed trees for the compiler.

| constant | minimal real | minimal complex | compiler (strict) | best expression | ratio | inf steps |
|---|---|---|---|---|---|---|
| `0` | 3 | 3 | 3 | `ln(1)` | 1.0 | 0 |
| `e` | 1 | 1 | 1 | `e` | 1.0 | 0 |
| `e-1` | 2 | 2 | 6 | `e-1` | 3.0 | 0 |
| `e^e` | 2 | 2 | 2 | `exp(e)` | 1.0 | 0 |
| `ln(e-1)` | 5 | 5 | 9 | `ln(e-1)` | 1.8 | 0 |
| `e-2` | 7 | 7 | 11 | `e-1-1` | 1.6 | 0 |
| `-1` | 8 | 8 | 8 | `-1` | 1.0 | 3 |
| `2` | 9 | 9 | 13 | `1+1` | 1.4 | 3 |
| `1/e` | 9 | 9 | 9 | `exp(-1)` | 1.0 | 3 |
| `e^2` | 10 | 10 | 14 | `exp(1+1)` | 1.4 | 3 |
| `-e` | 11 | 11 | 9 | `-e` | 0.8 | 3 |
| `e-3` | 12 | 12 | 16 | `e-1-1-1` | 1.3 | 0 |
| `ln2` | 12 | 12 | 16 | `ln(1+1)` | 1.3 | 3 |
| `-2` | 13 | 13 | 13 | `-1-1` | 1.0 | 3 |
| `2e` | 13 | 13 | 15 | `e+e` | 1.2 | 3 |
| `3` | 14 | 14 | 26 | `1+1+1` | 1.9 | 6 |
| `e/2` | 14 | 14 | 22 | `exp(1-ln(1+1))` | 1.6 | 3 |
| `e^3` | 15 | 15 | 27 | `exp(1+1+1)` | 1.8 | 6 |
| `1/2` | 17 | 15 | 25 | `exp(-ln(1+1))` | 1.5 | 6 |
| `sqrt(e)` | 18 | 16 | 46 | `exp(1/(1+1))` | 2.6 | 12 |
| `2/3` | 19 | <=37 | 71 | `(1+1)/(1+1+1)` | 3.7 | 15 |
| `-3` | 20 | 17 | 18 | `-1-1-1` | 0.9 | 3 |
| `e-4` | 21 | 17 | 21 | `e-1-1-1-1` | 1.0 | 0 |
| `4` | 21 | 19 | 39 | `1+1+1+1` | 1.9 | 9 |
| `3/2` | <=23 | <=20 | 58 | `1+1/(1+1)` |  | 15 |
| `1/3` | <=24 | <=36 | 58 | `1/(1+1+1)` |  | 15 |
| `-4` | <=27 | <=26 | 47 | `-(1+1+1+1)` |  | 12 |
| `5` | <=32 | <=31 | 52 | `1+1+1+1+1` |  | 12 |
| `-5` | <=34 | <=31 | 60 | `-(1+1+1+1+1)` |  | 15 |
| `6` | <=39 | <=36 | 59 | `(1+1+1)*(1+1)` |  | 12 |
| `i*pi` | n/a | <=23 | 11 | `ln(-1)` |  | 3 |
| `-i*pi` | n/a | 11 | 19 | `-ln(-1)` | 1.7 | 6 |

- **The compiler overpays by 1.3x to 3.7x on everything it builds by arithmetic.** `e-1` costs 6 against 2 because generic subtraction is `eml(ln a, exp b)` and never notices that `e - ln(e)` is already the answer. `2` costs 13 against 9, `3` 26 against 14, `4` 39 against 21, `2/3` 71 against 19, `sqrt(e)` 46 against 18. The ratio grows with the number of arithmetic operators in the expression, because each `+`, `*`, `/` and `^` is expanded to its Table 4 identity without any sharing or simplification.
- **Exact hits: `0`, `e`, `e^e`, `1/e`, `-1`, `-2`, `e-4`.** Where the expression maps onto one identity (`ln(1)`, `exp(e)`, `exp(-1)`) the compiler is minimal.
- **With infinities allowed, the negative constants are cheaper than the finite minima.** `-e` compiles to 9 nodes against the finite minimum 11, `-3` to 18 against 20. Both pass a 40-digit mpmath check (mpmath handles `log(0) = -inf`) and fail sympy (`zoo`). The finite minima for negatives are therefore minima of the finite grammar only; the paper's grammar, which relies on the IEEE route (see `CLAUDE.md`, invariant 6), has smaller trees for them. Every `+` in the compiler also passes through `-inf`, so under the finite grammar the compiler could not express addition at all without a literal `0`.
- **`ln(-1)` has a roundoff-determined sign.** The 11-node compiled tree for `ln(-1)` evaluates to `+i*pi` in float64 and in mpmath at 40 digits, because `exp(e - i*pi)` lands a rounding error below the negative real axis. Under the enumeration's canonicalisation (near-real values snapped to the axis, `ln(-x) = ln x + i*pi`) the same tree evaluates to `-i*pi`, which is the exact principal-branch value on the cut. The enumeration's `-i*pi` at 11 is a different tree. So `i*pi` is 11 or 23 depending on the convention, and any consumer of the compiler's `ln(-1)` should not rely on its sign.

## Resources

| run | frontier | level time | peak RSS | wall | notes |
|---|---|---|---|---|---|
| real 20 | 218,660,435 new, 327,841,920 total | size 19: 325 s; size 20: 1024 s | 12.0 GB | 4.1 h | 16 root joins at ~15 min each over 328M candidates dominate; levels spill to 8.7 GB on disk |
| complex 18 | 65,186,623 new, 94,426,311 total | size 17: 70 s; size 18: 234 s | 5.5 GB (frontier), 4.3 GB (joins) | 13 min frontier + 1.9 h joins | root joins over 94M complex candidates at ~4 min per target; levels spill to 3.8 GB |
| complex 19 | 211,882,756 new (from the memmap headers; not self-checked) | | >15 GB | | the level was enumerated and deduped, then the kernel OOM killer took the process while it was writing the level to disk, 15 min in; the guard's estimate (1.6x the level's arrays) was under the true peak of the finalisation |

Real 21 would need about 640M new values (18 GB of level arrays); with the current single-pass finalisation it is out of reach at 15 GB. Spilling pending chunks during the level and merging on disk would bring real 21 and complex 19 into range.

Generated 2026-09-04 by `benchmarks/eml_complexity.py` and `benchmarks/eml_complexity_join.py` on Claude Code on the Web (issue #70).
