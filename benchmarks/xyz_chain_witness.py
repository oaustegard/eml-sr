"""x0*x1*x2 is a depth-6 chain in skeleton_exact's own family (issue #62, PR #67 follow-up).

PR #67 reported 0 forms for triple_product in 2.53e9 joins of the
[chain <= 4] x join x [chain <= 4] family and read the zero as a family
boundary ("each exp crossing converts the accumulated log-carrier into one
product factor; the third exits wrapped in exp contamination"). This script
evaluates an explicit chain through `skeleton_exact.eval_chain`:

    N1 = eml(0, x1)            = 1 - ln x1
    E  = eml(0, 1 + N1)        = 1 - ln(2 - ln x1)
    N2 = eml(1 - E, x0)        = (2 - ln x1) - ln x0        <- PR #64's x0*x1 interior
    F  = eml(0, c + N2)        = 1 - ln(c + N2)
    N3 = eml(1 - F, x2)        = c + 2 - (ln x0 + ln x1 + ln x2)
    y  = eml((c + 2) - N3, 1)  = x0*x1*x2

Six eml nodes, every link on the engine's (alpha, gamma) lattice. The join
engine's longest path is a side chain of 4 plus the root, five nodes, so the
zero is the depth cap, not the operator. The log-carrier c + N2 must stay
positive wherever ln is taken: c = 0 works on the train domain (0.5, 2.5)
and fails on the held-out band (0.25, 4) where x0*x1 > e^2; c = 1 works on
both but needs the root offset 3, which is outside ALPHAS = {0, 1, 2, -1}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.dca_recovery import Target, make_data  # noqa: E402
from benchmarks.skeleton_exact import (  # noqa: E402
    ALPHAS,
    GAMMAS,
    STRUCT_TOL,
    eval_chain,
    spec_to_expr,
)

TARGET = Target("triple_product", 3, lambda X: X[:, 0] * X[:, 1] * X[:, 2],
                "x0*x1*x2")


def xyz_spec_and_assign(c: float):
    """Top-down links: root, N3, F, N2, E; deepest node is N1 = eml(0, x1)."""
    links = (("u", "const"),   # root: exp((c+2) - N3) - ln(1)
             ("u", "var:2"),   # N3:   exp(1 - F) - ln(x2)
             ("v", "const"),   # F:    exp(0) - ln(c + N2)
             ("u", "var:0"),   # N2:   exp(1 - E) - ln(x0)
             ("v", "const"))   # E:    exp(0) - ln(1 + N1)
    deepest = ("const", "var:1")  # N1: exp(0) - ln(x1)
    assign = [(c + 2.0, -1.0), (1.0, None),     # root link, root other (v = 1)
              (1.0, -1.0), (0.0, 1.0),          # N3 link, x2
              (c, 1.0), (0.0, None),            # F link, u = 0
              (1.0, -1.0), (0.0, 1.0),          # N2 link, x0
              (1.0, 1.0), (0.0, None),          # E link, u = 0
              (0.0, None), (0.0, 1.0)]          # deepest: u = 0, v = x1
    return (links, deepest), assign


def main() -> int:
    X_tr, y_tr, X_held, y_held = make_data(TARGET)
    out = {"target": "triple_product", "chain_nodes": 6,
           "engine_max_path_nodes": 5, "variants": []}
    for c in (0.0, 1.0):
        spec, assign = xyz_spec_and_assign(c)
        on_lattice = all(a in ALPHAS and g in GAMMAS
                         for (a, g) in assign[0::2][:5])
        rec = {"c": c, "expr": spec_to_expr(spec, assign),
               "links_on_lattice": on_lattice,
               "off_lattice_alphas": sorted({a for (a, g) in assign[0::2][:5]
                                             if a not in ALPHAS})}
        for name, X, y in (("train", X_tr, y_tr), ("held", X_held, y_held)):
            pred = eval_chain(spec, assign, X)
            finite = np.isfinite(pred)
            err = float(np.max(np.abs(pred[finite] - y[finite]))) if finite.any() else float("inf")
            rec[name] = {"max_abs_err": err, "n_nonfinite": int((~finite).sum()),
                         "structural": bool(finite.all() and err < STRUCT_TOL * max(1.0, float(np.max(np.abs(y)))))}
        out["variants"].append(rec)
        print(f"c={c:g}: {rec['expr']}")
        print(f"   links on lattice: {on_lattice} (off-lattice alphas {rec['off_lattice_alphas']})")
        for name in ("train", "held"):
            r = rec[name]
            print(f"   {name:5s}: max|err|={r['max_abs_err']:.2e} nonfinite={r['n_nonfinite']} structural={r['structural']}")
    path = ROOT / "benchmarks" / "results" / "skeleton_enum" / "triple_product_chain_witness.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
