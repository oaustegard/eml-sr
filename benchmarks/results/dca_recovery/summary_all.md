# DCA vs Adam on structural recovery (issue #58)

Targets: the 15 STATUS_COLLAPSED polynomial rows of the
llm-as-computer catalog (`dev/symbolic_collapse_report.md`,
'Collapsed (branchless, polynomial-closed)' table).
`native_multiply` is the canonical x0*x1 multiplicative failure
case from PR #56 / issue #57.

| target | expr | adam_softmax rec / R2 / s | adam_linear rec / R2 / s | dca_linear rec / R2 / s |
|---|---|---|---|---|
| push_halt | `x0` | ✓ / 1.0000 / 9 | ✓ / 1.0000 / 6 | ✓ / 1.0000 / 3 |
| push_pop | `x0` | ✓ / 1.0000 / 1 | ✓ / 1.0000 / 3 | ✓ / 1.0000 / 2 |
| stack_depth | `x0` | ✓ / 1.0000 / 1 | ✓ / 1.0000 / 3 | ✓ / 1.0000 / 2 |
| overwrite | `x2` | ✓ / 1.0000 / 1 | ✓ / 1.0000 / 4 | ✓ / 1.0000 / 2 |
| dup_add | `2*x0` | ✗ / -0.5068 / 88 | ✓ / 1.0000 / 3 | ✓ / 1.0000 / 2 |
| dup_add_chain_x4 | `16*x0` | ✗ / -1.1509 / 81 | ✓ / 1.0000 / 3 | ✓ / 1.0000 / 2 |
| basic_add | `x0 + x1` | ✗ / -1.1238 / 182 | ✓ / 1.0000 / 4 | ✓ / 1.0000 / 3 |
| add_dup_add | `2*x0 + 2*x1` | ✗ / 0.4812 / 181 | ✓ / 1.0000 / 4 | ✓ / 1.0000 / 3 |
| multi_add | `x0 + x1 + x2` | ✗ / -4.1517 / 118 | ✓ / 1.0000 / 3 | ✓ / 1.0000 / 2 |
| complex | `2*x1 + 2*x2` | ✗ / 0.4379 / 117 | ✓ / 1.0000 / 3 | ✓ / 1.0000 / 3 |
| alternating | `x0 + x1 + x3 + x5` | ✗ / -49.7904 / 167 | ✓ / 1.0000 / 4 | ✓ / 1.0000 / 3 |
| many_pushes | `x0+x1+...+x9` | ✗ / -3.2455 / 211 | ✓ / 1.0000 / 7 | ✓ / 1.0000 / 5 |
| square_via_dupmul | `x0**2` | ✗ / -4.9732 / 153 | ✗ / 0.8859 / 861 | ✗ / 0.7307 / 3202 |
| sum_of_squares | `x0**2 + x3**2` | ✗ / -1.6782 / 257 | ✗ / 0.7164 / 2504 | ✗ / 0.6923 / 7460 |
| native_multiply | `x0*x1` | ✗ / -0.4335 / 178 | ✗ / 0.7612 / 1191 | ✗ / 0.9032 / 4305 |
|  **recovered** |  | **4/15** | **12/15** | **12/15** |

## Success gate (issue #58)

- x0*x1 via adam_softmax: NOT recovered (R2 -0.4335, expr `(e - ln((e - ln(x2))))`)
- x0*x1 via adam_linear: NOT recovered (R2 0.7612, expr `-2 + 1*x1 + 2*x2`)
- x0*x1 via dca_linear: NOT recovered (R2 0.9032, expr `eml(1*x2 + 0.397*(eml(0, 1 + -1*x2 + 0.232*(eml(1*x2 + 1*(eml(1*(1), 1*x1 + 1*(1 + 1*x1 + -1*x2))), 1 + -1*x1 + e*x2))))`)
