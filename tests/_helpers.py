"""Shared test helpers.

Compiler-backed evaluation utilities: turn a formula string into a
real-valued target array by compiling to an EML tree and evaluating it
point-by-point. Using these in place of hand-crafted ``np.exp`` /
``np.log`` targets makes the formula string the spec and catches drift
between the compiler's eval semantics and the searcher's simplifier.
"""
from __future__ import annotations

import numpy as np

from eml_compiler import compile_expr, eval_eml


def eval_formula_univariate(formula: str, x: np.ndarray) -> np.ndarray:
    """Compile ``formula`` and evaluate it at each ``x[i]`` — returns real array."""
    tree = compile_expr(formula)
    ys = np.empty_like(x, dtype=float)
    for i, xi in enumerate(x):
        v = eval_eml(tree, x=float(xi))
        ys[i] = v.real
    return ys


def eval_formula_multivariate(formula: str, X: np.ndarray,
                              var_names: tuple) -> np.ndarray:
    """Compile ``formula`` and evaluate at each row of ``X`` — returns real array."""
    tree = compile_expr(formula, variables=var_names)
    ys = np.empty(X.shape[0], dtype=float)
    for i in range(X.shape[0]):
        bindings = {name: float(X[i, j]) for j, name in enumerate(var_names)}
        v = eval_eml(tree, bindings)
        ys[i] = v.real
    return ys
