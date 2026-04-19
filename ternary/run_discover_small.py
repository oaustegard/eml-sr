"""Small gradient-discovery run for the report.

Trims :mod:`ternary.discover` to depths ≤ 2 + a single depth-3 attempt
for the two primitives we expect to succeed (``exp(x)``, ``1``). Full
matrix at depth 3 with 8 targets × 2 grammars × 3 seeds takes >30min
because each ternary tree at depth 3 has 27 leaves and the forward
pass through ``nan_to_num`` and complex tensors is slow.

Run: ``python -m ternary.run_discover_small``
"""

from __future__ import annotations

import json
import sys
import time

from .discover import TARGETS, run_one


def main():
    # Depth-2 over all targets, both grammars, 2 seeds. This is the main
    # "does gradient recovery match the enumerative result?" check.
    plan = [(t, 2, allow) for t in TARGETS for allow in (False, True)]
    # Depth-3: pure only on the sampled set. Relaxed at depth 3 is skipped
    # by default — 303 params, the loss surface is brutal, each run takes
    # tens of minutes, and the pattern from depth 2 already settles the
    # verdict. Re-enable by adding ``allow=True`` to the comprehension.
    depth3_targets = [t for t in TARGETS if t.name in ("exp(x)", "1", "ln(x)")]
    plan += [(t, 3, False) for t in depth3_targets]

    results = []
    t0 = time.time()
    for i, (target, depth, allow) in enumerate(plan, 1):
        best = float("inf")
        for seed in range(2):
            r = run_one(target, depth, allow, seed)
            best = min(best, r["final_mse"])
            results.append(r)
        tag = "relaxed" if allow else "pure   "
        print(f"[{i:>2}/{len(plan)}] {target.name:<10} d{depth} {tag}  "
              f"best final MSE = {best:.3e}   "
              f"elapsed {time.time() - t0:.0f}s", flush=True)

    print(json.dumps({"results": results}, default=str), flush=True)


if __name__ == "__main__":
    main()
