"""Compare eml_compiler's constant trees with the minimal sizes from eml_complexity.

For each constant in benchmarks/eml_complexity.md, compile a few natural
elementary expressions in strict mode (leaves are the constant 1 only, the
paper's grammar), verify each tree numerically, and keep the smallest. The
node count is the number of eml nodes, the same measure the enumeration uses.

    python3 benchmarks/compiler_vs_minimal.py            # prints the table
    python3 benchmarks/compiler_vs_minimal.py --json OUT  # also writes JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import sympy as sp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eml_compiler import GrammarError, Leaf, Node, compile_expr, eval_eml, to_string  # noqa: E402

E, PI, I = sp.E, sp.pi, sp.I

#: constant -> (sympy value, minimal real size or None, minimal complex size or None, candidate expressions)
#: Sizes are the exact minima from eml_complexity.md; bounds are given as strings "<=k".
TABLE = {
    "0": (sp.Integer(0), 3, 3, ["ln(1)", "1-1"]),
    "e": (E, 1, 1, ["e", "exp(1)"]),
    "e-1": (E - 1, 2, 2, ["e-1", "exp(1)-1"]),
    "e^e": (E ** E, 2, 2, ["exp(e)", "e^e"]),
    "ln(e-1)": (sp.log(E - 1), 5, 5, ["ln(e-1)"]),
    "e-2": (E - 2, 7, 7, ["e-1-1", "e-(1+1)"]),
    "-1": (sp.Integer(-1), 8, 8, ["-1", "ln(1)-1"]),
    "2": (sp.Integer(2), 9, 9, ["1+1", "exp(ln(1+1))", "1-(-1)"]),
    "1/e": (1 / E, 9, 9, ["1/e", "exp(-1)", "exp(ln(1)-1)"]),
    "e^2": (E ** 2, 10, 10, ["exp(1+1)", "e*e", "e^(1+1)"]),
    "-e": (-E, 11, 11, ["-e", "ln(1)-e"]),
    "e-3": (E - 3, 12, 12, ["e-1-1-1", "e-(1+1+1)"]),
    "ln2": (sp.log(2), 12, 12, ["ln(1+1)"]),
    "-2": (sp.Integer(-2), 13, 13, ["-(1+1)", "-1-1"]),
    "2e": (2 * E, 13, 13, ["e+e", "(1+1)*e", "exp(1+ln(1+1))"]),
    "3": (sp.Integer(3), 14, 14, ["1+1+1"]),
    "e/2": (E / 2, 14, 14, ["e/(1+1)", "exp(1-ln(1+1))"]),
    "e^3": (E ** 3, 15, 15, ["exp(1+1+1)", "e*e*e"]),
    "1/2": (sp.Rational(1, 2), 17, 15, ["1/(1+1)", "exp(-ln(1+1))"]),
    "sqrt(e)": (sp.sqrt(E), 18, 16, ["sqrt(e)", "exp(1/(1+1))"]),
    "2/3": (sp.Rational(2, 3), 19, "<=37", ["(1+1)/(1+1+1)"]),
    "-3": (sp.Integer(-3), 20, 17, ["-(1+1+1)", "-1-1-1"]),
    "e-4": (E - 4, 21, 17, ["e-1-1-1-1", "e-(1+1)*(1+1)"]),
    "4": (sp.Integer(4), 21, 19, ["1+1+1+1", "(1+1)*(1+1)", "(1+1)^(1+1)", "exp(ln(1+1)+ln(1+1))"]),
    "3/2": (sp.Rational(3, 2), "<=23", "<=20", ["(1+1+1)/(1+1)", "1+1/(1+1)"]),
    "1/3": (sp.Rational(1, 3), "<=24", "<=36", ["1/(1+1+1)"]),
    "-4": (sp.Integer(-4), "<=27", "<=26", ["-(1+1+1+1)", "-((1+1)*(1+1))"]),
    "5": (sp.Integer(5), "<=32", "<=31", ["1+1+1+1+1", "(1+1)*(1+1)+1"]),
    "-5": (sp.Integer(-5), "<=34", "<=31", ["-(1+1+1+1+1)"]),
    "6": (sp.Integer(6), "<=39", "<=36", ["1+1+1+1+1+1", "(1+1+1)*(1+1)", "(1+1)*(1+1+1)"]),
    "i*pi": (I * PI, None, "<=23", ["ln(-1)"]),
    "-i*pi": (-I * PI, None, 11, ["-ln(-1)", "ln(1)-ln(-1)"]),
}


def eml_nodes(tree) -> int:
    """Number of eml nodes; every leaf must be the constant 1."""
    if isinstance(tree, Leaf):
        if tree.value is None or complex(tree.value) != 1:
            raise ValueError(f"non-1 leaf {tree.label!r}")
        return 0
    return 1 + eml_nodes(tree.left) + eml_nodes(tree.right)


def nonfinite_intermediates(tree) -> int:
    """Count subtrees whose float64 value is inf or nan (the ln(0) = -inf route)."""
    count = [0]

    def ev(t):
        if isinstance(t, Leaf):
            return np.complex128(1)
        l, r = ev(t.left), ev(t.right)
        with np.errstate(all="ignore"):
            v = np.exp(l) - np.log(r)
        if not (np.isfinite(v.real) and np.isfinite(v.imag)):
            count[0] += 1
        return v

    ev(tree)
    return count[0]


def best_compilation(candidates, target):
    """Smallest strict compilation among the candidates that evaluates to target.

    Returns (eml nodes, expression, tree string, nonfinite intermediates) or None.
    """
    tv = complex(sp.N(target, 20))
    best = None
    for expr in candidates:
        try:
            tree = compile_expr(expr, strict=True)
            n = eml_nodes(tree)
            val = eval_eml(tree)
        except (GrammarError, ValueError):
            continue
        if abs(val - tv) > 1e-9 * max(1.0, abs(tv)):
            continue
        if best is None or n < best[0]:
            best = (n, expr, to_string(tree), nonfinite_intermediates(tree))
    return best


def run():
    rows = []
    for name, (target, real_min, cplx_min, cands) in TABLE.items():
        b = best_compilation(cands, target)
        rows.append({"constant": name, "minimal_real": real_min, "minimal_complex": cplx_min,
                     "compiler": None if b is None else b[0],
                     "expression": None if b is None else b[1],
                     "tree": None if b is None else b[2],
                     "nonfinite_intermediates": None if b is None else b[3]})
    return rows


def as_markdown(rows) -> str:
    out = ["| constant | minimal real | minimal complex | compiler (strict) | best expression | ratio | inf steps |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        ref = r["minimal_real"] if isinstance(r["minimal_real"], int) else r["minimal_complex"]
        ratio = f"{r['compiler'] / ref:.1f}" if isinstance(ref, int) and r["compiler"] else ""
        out.append(f"| `{r['constant']}` | {r['minimal_real'] if r['minimal_real'] is not None else 'n/a'} | "
                   f"{r['minimal_complex'] if r['minimal_complex'] is not None else 'n/a'} | "
                   f"{r['compiler'] if r['compiler'] is not None else 'no strict form'} | "
                   f"`{r['expression']}` | {ratio} | {r['nonfinite_intermediates'] if r['nonfinite_intermediates'] is not None else ''} |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="write rows to this JSON path")
    args = ap.parse_args(argv)
    rows = run()
    print(as_markdown(rows))
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
