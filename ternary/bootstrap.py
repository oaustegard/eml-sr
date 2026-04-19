"""Hand-derived ternary-tree constructions of basic primitives.

Grammar: ``S → x | T(S, S, S)`` with no distinguished constants.

Each helper builds a nested ``T``-expression tree for a target primitive
and exposes it both as a symbolic string (for reporting) and as a NumPy
callable (for numerical verification).

Derivations (see ``docs/tree_size.md`` in this directory for the figure
proofs):

.. code::

    1        = T(x, x, x)                                  size 4   depth 1
    0        = T(x, x, 1)                                  size 7   depth 2
    exp(x-1) = T(x, 1, x)                                  size 7   depth 2
    exp(x)   = T(x, 0, x)                                  size 10  depth 3
    exp(x-y) = T(x, y, x)         (trivial; needs only x)  size 10  depth 3

Beyond this the derivations become difficult *without* the terminal
``1`` as a literal leaf, because constructing ``e`` requires producing
``exp(1)`` and the only easy path to ``exp(u)`` is ``T(u, 0, u)`` —
but ``T(1, 0, 1)`` evaluates to ``0 / 0`` (``ln(1) = 0`` in the
denominator). See :mod:`ternary.enumerate_search` for an automated
enumeration over small trees that confirms ``e``, ``ln x``, and ``-x``
are *not* reachable at depth ≤ 4 with the pure ``S → x | T(S,S,S)``
grammar.

Each derivation below is given as a :class:`Tree` AST that can be
pretty-printed, numerically evaluated, and converted to a sympy
expression for cross-checking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
import sympy as sp

from .operator import t_np


# ─── AST ─────────────────────────────────────────────────────────

Tree = Union["X", "T"]


@dataclass(frozen=True)
class X:
    """The input variable leaf."""

    def __str__(self) -> str:
        return "x"

    def eval(self, x_val: complex) -> complex:
        return complex(x_val)

    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 0

    def to_sympy(self, x_sym: sp.Symbol) -> sp.Expr:
        return x_sym


@dataclass(frozen=True)
class T:
    """A ternary-operator internal node: ``T(a, b, c)``."""
    a: Tree
    b: Tree
    c: Tree

    def __str__(self) -> str:
        return f"T({self.a}, {self.b}, {self.c})"

    def eval(self, x_val: complex) -> complex:
        a = self.a.eval(x_val)
        b = self.b.eval(x_val)
        c = self.c.eval(x_val)
        return complex(t_np(a, b, c))

    def size(self) -> int:
        return 1 + self.a.size() + self.b.size() + self.c.size()

    def depth(self) -> int:
        return 1 + max(self.a.depth(), self.b.depth(), self.c.depth())

    def to_sympy(self, x_sym: sp.Symbol) -> sp.Expr:
        a = self.a.to_sympy(x_sym)
        b = self.b.to_sympy(x_sym)
        c = self.c.to_sympy(x_sym)
        return sp.exp(a) * sp.log(c) / (sp.log(a) * sp.exp(b))


# ─── Hand-derived trees ──────────────────────────────────────────

def one() -> Tree:
    """``1 = T(x, x, x)`` (depth 1, size 4)."""
    return T(X(), X(), X())


def zero() -> Tree:
    """``0 = T(x, x, 1)`` = ``T(x, x, T(x,x,x))`` (depth 2, size 7)."""
    return T(X(), X(), one())


def exp_x_minus_1() -> Tree:
    """``exp(x-1) = T(x, 1, x)`` = ``T(x, T(x,x,x), x)`` (depth 2, size 7)."""
    return T(X(), one(), X())


def exp_x() -> Tree:
    """``exp(x) = T(x, 0, x)``, using ``0 = T(x, x, 1)`` (depth 3, size 10)."""
    return T(X(), zero(), X())


def exp_x_minus_y(var_y: Tree) -> Tree:
    """``exp(x - y) = T(x, y, x)`` parameterised by the y-subtree.

    This is the most general "difference in exponent" pattern — given any
    subtree representing some value ``y``, ``T(x, y, x)`` evaluates to
    ``exp(x - y)`` in the univariate case (where ``y`` is itself a
    function of ``x``, of course).
    """
    return T(X(), var_y, X())


# ─── Registry for tests and reporting ────────────────────────────

@dataclass(frozen=True)
class Primitive:
    name: str
    builder: Callable[[], Tree]
    target: Callable[[complex], complex]

    @property
    def tree(self) -> Tree:
        return self.builder()


PRIMITIVES = [
    Primitive("1",          one,             lambda x: 1.0 + 0j),
    Primitive("0",          zero,            lambda x: 0.0 + 0j),
    Primitive("exp(x-1)",   exp_x_minus_1,   lambda x: np.exp(x - 1)),
    Primitive("exp(x)",     exp_x,           np.exp),
]


def verify(primitive: Primitive, xs, atol: float = 1e-10):
    """Return (max_abs_err, rows) where rows = [(x, tree_val, target_val), ...]."""
    tree = primitive.tree
    rows = []
    max_err = 0.0
    for x in xs:
        t_val = tree.eval(complex(x))
        y_val = primitive.target(complex(x))
        err = abs(t_val - y_val)
        max_err = max(max_err, err)
        rows.append((complex(x), t_val, y_val, err))
    return max_err, rows
