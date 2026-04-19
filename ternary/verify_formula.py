"""Deliverable A: verify the §5 ternary formula.

Resolves the parsing ambiguity flagged in issue #37 by:

1. Parsing the paper text two ways — the issue's initial reading
   (``exp(x/ln x)``) and the left-to-right reading (``exp(x)/ln(x)``) —
   and checking ``T(x, x, x) = 1`` symbolically with sympy for each.
2. Spot-checking numerically at several real x values.

Expected outcome (the paper is correct, the issue's transcription had a
typo): ``(exp(x)/ln(x)) * (ln(z)/exp(y))`` satisfies ``T(x,x,x) = 1``;
``exp(x/ln(x)) * (ln(z)/exp(y))`` does not.

Run: ``python -m ternary.verify_formula``.
"""

from __future__ import annotations

import math

import numpy as np
import sympy as sp


def symbolic_check():
    """Symbolic verification with sympy. Returns a dict of parse → identity."""
    x, y, z = sp.symbols("x y z", positive=True)

    # Reading A: issue #37 initial parse — exp in numerator has (x/ln x) inside
    t_issue = sp.exp(x / sp.log(x)) * sp.log(z) / sp.exp(y)
    # Reading B: left-to-right math reading of the paper text
    t_paper = sp.exp(x) / sp.log(x) * sp.log(z) / sp.exp(y)

    # T(x, x, x)
    sub = {y: x, z: x}
    identity_issue = sp.simplify(t_issue.subs(sub))
    identity_paper = sp.simplify(t_paper.subs(sub))

    return {
        "issue_parse": {
            "expr": t_issue,
            "T(x,x,x)": identity_issue,
            "is_one": identity_issue == 1,
        },
        "paper_parse": {
            "expr": t_paper,
            "T(x,x,x)": identity_paper,
            "is_one": identity_paper == 1,
        },
    }


def numeric_check():
    """Numeric spot-check at several real x > 0, x != 1."""
    xs = [0.3, 0.577, 1.414, math.pi / 4, 2.71828, 5.0, 10.0]

    def t_issue(x, y, z):
        return np.exp(x / np.log(x)) * np.log(z) / np.exp(y)

    def t_paper(x, y, z):
        return np.exp(x) / np.log(x) * np.log(z) / np.exp(y)

    rows = []
    for x in xs:
        rows.append({
            "x": x,
            "issue_T(x,x,x)": t_issue(x, x, x),
            "paper_T(x,x,x)": t_paper(x, x, x),
        })
    return rows


def main():
    print("=" * 72)
    print("Ternary formula verification — eml-sr issue #37, deliverable A")
    print("=" * 72)

    print("\n--- Symbolic check (sympy) ---")
    sym = symbolic_check()
    for name, entry in sym.items():
        print(f"\n  {name}:")
        print(f"    expr     = {entry['expr']}")
        print(f"    T(x,x,x) = {entry['T(x,x,x)']}")
        print(f"    == 1 ?    {entry['is_one']}")

    print("\n--- Numeric spot-check ---")
    print(f"  {'x':>10}  {'issue parse':>24}  {'paper parse':>24}")
    for row in numeric_check():
        print(f"  {row['x']:>10.6f}  "
              f"{str(row['issue_T(x,x,x)']):>24}  "
              f"{str(row['paper_T(x,x,x)']):>24}")

    print("\n--- Verdict ---")
    paper_ok = sym["paper_parse"]["is_one"]
    issue_ok = sym["issue_parse"]["is_one"]
    if paper_ok and not issue_ok:
        print("  The paper's formula reads T = (exp(x)/ln(x)) * (ln(z)/exp(y)).")
        print("  T(x,x,x) = 1 symbolically. The issue's parse (exp(x/ln(x))) is")
        print("  a transcription typo; the paper text is correct.")
    elif paper_ok and issue_ok:
        print("  Both parses satisfy T(x,x,x) = 1. Unexpected — investigate.")
    elif issue_ok:
        print("  Only the issue's parse satisfies the identity. Paper text needs")
        print("  re-reading (maybe a different bracket convention).")
    else:
        print("  Neither parse satisfies T(x,x,x) = 1. Check PDF directly.")


if __name__ == "__main__":
    main()
