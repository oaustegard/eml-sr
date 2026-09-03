# EML complexity of constants

Minimal number of `eml` nodes in a closed tree `S -> 1 | eml(S,S)` evaluating exactly to the constant. Enumeration is bottom-up with observational dedupe (11 significant digits), an exactness filter (no term may be numerically swallowed by the other; `exp` underflow rejected), and every reported witness re-verified symbolically with sympy. Real branch: `ln` requires `y>0`. Complex branch: principal branch, near-real values snapped to the axis so `ln(-x) = ln x + i*pi` deterministically.

Generated 2026-09-02 by `benchmarks/eml_complexity.py`. Real enumerated to size 18 (36.5M distinct values), complex to size 16 (9.1M).

| constant | real | complex | real witness |
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
| `e-3` | 12 | 12 | `e(e(1,e(e(1,e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1))),1)),e(1,1))` |
| `ln2` | 12 | 12 | `e(1,e(e(1,e(1,e(e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1)),1))),1))` |
| `-2` | 13 | 13 | `e(e(1,e(e(1,e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1))),1)),e(e(1,1),1))` |
| `2e` | 13 | 13 | `e(1,e(e(e(1,e(e(1,e(e(1,1),e(e(1,1),1))),1)),e(e(e(1,1),1),1)),1))` |
| `3` | 14 | 14 | `e(1,e(e(e(1,e(e(1,e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1))),1)),e(1,1)),1))` |
| `e/2` | 14 | 14 | `e(e(e(1,e(e(1,1),1)),e(1,e(e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1)),1))),1)` |
| `e^3` | 15 | 15 | `e(e(1,e(e(e(1,e(e(1,e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1))),1)),e(1,1)),1)),1)` |
| `1/2` | 17 | 15 | `e(e(e(1,e(e(1,e(1,e(1,e(e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1)),1)))),1)),e(e(1,1),1)),1)` |
| `sqrt(e)` | 18 | 16 | `e(e(e(e(1,e(e(1,e(1,e(1,e(e(e(1,e(e(1,e(1,e(1,1))),1)),e(1,1)),1)))),1)),e(e(1,1),1)),1),1)` |
| `-i*pi` | n/a | 11 | `e(1,e(e(1,e(e(1,e(1,e(e(e(1,1),1),1))),1)),e(e(1,1),1)))` |
| `i` | n/a | &gt;16 | `` |
| `pi` | &gt;18 | &gt;16 | `` |
| `sqrt2` | &gt;18 | &gt;16 | `` |
| `4` | &gt;18 | &gt;16 | `` |
| `-3` | &gt;18 | &gt;16 | `` |
| `1/3` | &gt;18 | &gt;16 | `` |
| `phi` | &gt;18 | &gt;16 | `` |

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
| 11 | 12455 | 19388 | 58786 |
| 12 | 36210 | 60775 | 208012 |
| 13 | 106003 | 192013 | 742900 |
| 14 | 311950 | 610275 | 2674440 |
| 15 | 921965 | 1949611 | 9694845 |
| 16 | 2731279 | 6253967 | 35357670 |
| 17 | 8117549 |  | 129644790 |
| 18 | 24240152 |  | 477638700 |

## Reading

- Integers grow linearly, not logarithmically: 0:3, 1:0, 2:9, 3:14, and 4 is not reachable at size 18 on the real branch. Mahler-Popken integer complexity (1s with + and x) grows like 3 log3 n. In EML the cheap ladder is `e-k`: `e-1` (2), `e-2` (7), `e-3` (12), five nodes per step via `e-(k+1) = eml(ln(e-k), e)`, and `k = eml(1, exp(e-k))` adds two. The real ladder dies at `e-3 < 0` because the next step needs `ln(e-3)`; the complex branch continues (predicts `e-4` at 17, `4` at 19).
- Every non-trivial constant routes through the 3-node `ln x = eml(1, eml(eml(1,x),1))` and subtraction `a-b = eml(ln a, exp b)` (5 nodes overhead). Multiplication is never used at these sizes; the crossover where `x*y = exp(ln x + ln y)` beats repeated subtraction has not been reached.
- Complex detours pay: `1/2` costs 15 in C vs 17 in R, `sqrt(e)` 16 vs 18, `-i*pi` is 11 (`ln(-1)` after `-1` at 8). `i` itself is beyond 16.
- Redundancy of the tree encoding: distinct/Catalan falls as ~0.78^size (real) and ~0.83^size (complex); by size 18 only 5% of real trees compute a new constant.
- Two artifacts the exactness filter caught: `exp(-huge) ~ 0` produced a fake `2e` at size 11, and `exp(e^e - i*pi)` landed exactly on the log branch cut where roundoff chose the sign of `i*pi`. Both are the same failure the SR work sees as 'near-identities'; a constant table needs them excluded.
- Not found: `pi`, `sqrt2`, `phi`, `1/3`, `1/4`, `i` within the enumerated sizes.