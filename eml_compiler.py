"""
EML compiler: convert elementary-function expressions into pure EML trees.

The paper (Odrzywolek 2026, §4.1) describes a compiler that takes an
arbitrary AST over the elementary basis {+, -, *, /, ^, exp, ln, sqrt,
neg, constants, variables} and emits an EML tree over the single
operator ``eml(x, y) = exp(x) - ln(y)`` plus the constant ``1``.

This module is a straight port of the JS reference compiler that ships
with ``oaustegard.github.io/fun-and-games/eml-calc.html``. It exposes:

- :func:`parse` — tokenizer + recursive-descent parser for a small
  math-expression language (binary ``+ - * / ^``, unary ``-``, function
  calls ``exp ln log sqrt``, and the 2-ary function ``eml(a, b)``).
- :func:`compile` — AST → EML tree, using the paper's bootstrap chain.
- :func:`eval_eml` — numerical evaluation of an EML tree over ``C``.
- :func:`tree_size`, :func:`tree_depth` — structural metrics.
- :func:`to_string` — EML tree pretty-printer (matches the
  ``eml(a, b)`` / leaf syntax used by :meth:`eml_sr.EMLTree1D.to_expr`).
- :func:`compile_expr` — one-shot ``str → EMLTree``.

Grammar relaxation
------------------
The paper's pure grammar is ``S -> 1 | x | eml(S, S)``. The JS compiler
and this port both default to the pragmatic relaxation
``S -> c | x | eml(S, S)`` where ``c`` is any complex constant
(``π = -i·ln(-1)`` would otherwise produce an enormous tree, and so
would arbitrary floats from measured data). ``e`` is derived as
``eml(1, 1)`` either way.

Pass ``strict=True`` to :func:`compile` (or the CLI ``--strict``) to
enforce the paper's pure form: arbitrary numeric literals other than
``0`` and ``1`` are rejected, and ``π`` / other named constants other
than ``e`` are rejected. ``0`` is tolerated because the compiler's
negation primitive shortcuts through it; see :func:`_eml_neg` for a
paper-faithful alternative that uses ``eml_ln(1) = 0`` instead.

Trig (sin/cos/tan) is **out of scope** per §4.1. Use ``exp`` / ``ln``
with complex arguments if you need a transcendental identity path.

CLI
---
``python -m eml_compiler "ln(x) + exp(y)"`` prints the compiled tree
plus size and depth. ``--strict`` enables paper-faithful mode.
"""

from __future__ import annotations

import cmath
import math
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional, Union

import numpy as np

# ─── Tree representation ───────────────────────────────────────────

@dataclass
class Leaf:
    """Terminal node of an EML tree.

    A leaf is either *numeric* (``value`` is a concrete complex number;
    ``label`` is its string form) or *symbolic* (``value is None``;
    ``label`` is a variable name resolved at :func:`eval_eml` time via
    a bindings dict).
    """
    value: Optional[complex]
    label: str

    @property
    def is_symbolic(self) -> bool:
        return self.value is None


@dataclass
class Node:
    """Internal ``eml(left, right)`` node of an EML tree."""
    left: "EMLTree"
    right: "EMLTree"
    tag: str = ""  # decorative, identifies which primitive produced this node


EMLTree = Union[Leaf, Node]


# Canonical leaf constants — re-used everywhere so identity comparisons
# remain cheap. Each construction call builds a fresh leaf so callers
# that mutate ``tag`` or embed them in larger trees get their own copy.
def _L(v: complex, lbl: Optional[str] = None) -> Leaf:
    return Leaf(value=complex(v), label=lbl if lbl is not None else _fmt_num(v))


def _E(left: EMLTree, right: EMLTree, tag: str = "") -> Node:
    return Node(left=left, right=right, tag=tag)


def _fmt_num(v: complex) -> str:
    v = complex(v)
    if v.imag == 0:
        r = v.real
        if r == int(r):
            return str(int(r))
        return repr(r)
    return repr(v)


# ─── Tokenizer + parser ────────────────────────────────────────────
#
# Matches the JS implementation's surface syntax. Token shapes:
#   ('punct', ch)   — one of ( ) + - * / ^ ,
#   ('num',   v)    — float literal
#   ('ident', s)    — identifier (variable or function name)

_IDENT_START = re.compile(r"[A-Za-z_π]")
_IDENT_CONT = re.compile(r"[A-Za-z0-9_π]")
_NUM = re.compile(r"[0-9.]")


def _tokenize(s: str) -> list[tuple]:
    tokens: list[tuple] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c in " \t\n\r":
            i += 1
            continue
        if c in "()+-*/^,":
            tokens.append(("punct", c))
            i += 1
            continue
        if _NUM.match(c):
            j = i
            while j < n and _NUM.match(s[j]):
                j += 1
            frag = s[i:j]
            try:
                v = float(frag)
            except ValueError as ex:
                raise ValueError(f"Bad numeric literal: {frag!r}") from ex
            tokens.append(("num", v))
            i = j
            continue
        if _IDENT_START.match(c):
            j = i
            while j < n and _IDENT_CONT.match(s[j]):
                j += 1
            tokens.append(("ident", s[i:j]))
            i = j
            continue
        raise ValueError(f"Bad char at position {i}: {c!r}")
    return tokens


# AST node forms (mirrors the JS shapes, with Python dicts):
#   {"k": "num", "v": float}
#   {"k": "cst", "name": str}        — named constant (e, pi, π) OR a variable
#   {"k": "un",  "op": str, "a": AST}
#   {"k": "bin", "op": str, "l": AST, "r": AST}
#   {"k": "eml", "l": AST, "r": AST} — raw eml(a, b) form (extension)


_KNOWN_FUNCS = {"exp", "ln", "log", "sqrt", "eml"}


def parse(input_str: str) -> dict:
    """Parse an elementary expression string into an AST dict.

    The grammar is (in Pratt-ish precedence order)::

        expr   := term (('+' | '-') term)*
        term   := factor (('*' | '/') factor)*
        factor := unary ('^' factor)?             # right-assoc
        unary  := '-' unary | atom
        atom   := NUM | '(' expr ')'
                | IDENT                            # variable or named const
                | IDENT '(' expr ')'               # unary fn
                | 'eml' '(' expr ',' expr ')'      # 2-ary fn (extension)

    Anything not derivable from this grammar raises :class:`ValueError`.
    """
    tokens = _tokenize(input_str)
    pos = [0]

    def peek():
        if pos[0] < len(tokens):
            return tokens[pos[0]]
        return None

    def eat(expected_kind: Optional[str] = None, expected_val=None):
        tk = peek()
        if tk is None:
            raise ValueError(f"Unexpected end of input (wanted {expected_kind}={expected_val})")
        if expected_kind is not None and tk[0] != expected_kind:
            raise ValueError(f"Expected {expected_kind} but got {tk}")
        if expected_val is not None and tk[1] != expected_val:
            raise ValueError(f"Expected {expected_val!r} but got {tk[1]!r}")
        pos[0] += 1
        return tk

    def expr():
        left = term()
        while peek() in (("punct", "+"), ("punct", "-")):
            op = eat("punct")[1]
            left = {"k": "bin", "op": op, "l": left, "r": term()}
        return left

    def term():
        left = factor()
        while peek() in (("punct", "*"), ("punct", "/")):
            op = eat("punct")[1]
            left = {"k": "bin", "op": op, "l": left, "r": factor()}
        return left

    def factor():
        base = unary()
        if peek() == ("punct", "^"):
            eat("punct", "^")
            base = {"k": "bin", "op": "^", "l": base, "r": factor()}
        return base

    def unary():
        if peek() == ("punct", "-"):
            eat("punct", "-")
            return {"k": "un", "op": "neg", "a": unary()}
        return atom()

    def atom():
        tk = peek()
        if tk is None:
            raise ValueError("Unexpected end of input")
        kind, val = tk
        if kind == "num":
            eat("num")
            return {"k": "num", "v": val}
        if kind == "ident":
            eat("ident")
            if peek() == ("punct", "("):
                eat("punct", "(")
                if val == "eml":
                    a = expr()
                    eat("punct", ",")
                    b = expr()
                    eat("punct", ")")
                    return {"k": "eml", "l": a, "r": b}
                a = expr()
                eat("punct", ")")
                return {"k": "un", "op": val.lower(), "a": a}
            # bare identifier: named constant or variable — compiler decides
            return {"k": "cst", "name": val}
        if kind == "punct" and val == "(":
            eat("punct", "(")
            e = expr()
            eat("punct", ")")
            return e
        raise ValueError(f"Unexpected token: {tk}")

    result = expr()
    if pos[0] < len(tokens):
        raise ValueError(f"Extra tokens after expression: {tokens[pos[0]:]}")
    return result


# ─── Primitive chain (mirrors the JS reference exactly) ────────────
#
# Each primitive wraps its inputs in EML nodes following the identities
# of §3 / Table 4 of the paper. Tags are for display only; they don't
# affect evaluation.

def _one() -> Leaf:
    return _L(1, "1")


def _zero() -> Leaf:
    return _L(0, "0")


def _eml_exp(x: EMLTree) -> Node:
    # exp(x) = eml(x, 1)
    return _E(x, _one(), "eˣ")


def _eml_ln(x: EMLTree) -> Node:
    # ln(x) = eml(1, eml(eml(1, x), 1))
    return _E(_one(), _E(_E(_one(), x), _one()), "ln")


def _eml_sub(a: EMLTree, b: EMLTree) -> Node:
    # a - b = eml(ln(a), exp(b))
    return _E(_eml_ln(a), _eml_exp(b), "−")


def _eml_neg(x: EMLTree, *, strict: bool = False) -> Node:
    # Default: -x = sub(0, x). Needs a literal 0 leaf.
    # Strict:  -x = sub(ln(1), x). ln(1) evaluates to 0 but is built
    # from {1} and eml nodes alone — paper-faithful.
    zero = _eml_ln(_one()) if strict else _zero()
    return _eml_sub(zero, x)


def _eml_add(a: EMLTree, b: EMLTree, *, strict: bool = False) -> Node:
    # a + b = a - (-b)
    return _eml_sub(a, _eml_neg(b, strict=strict))


def _eml_inv(x: EMLTree, *, strict: bool = False) -> Node:
    # 1/x = exp(-ln(x))
    return _eml_exp(_eml_neg(_eml_ln(x), strict=strict))


def _eml_mul(a: EMLTree, b: EMLTree, *, strict: bool = False) -> Node:
    # a * b = exp(ln(a) + ln(b))
    return _eml_exp(_eml_add(_eml_ln(a), _eml_ln(b), strict=strict))


def _eml_div(a: EMLTree, b: EMLTree, *, strict: bool = False) -> Node:
    # a / b = a * (1/b)
    return _eml_mul(a, _eml_inv(b, strict=strict), strict=strict)


def _eml_pow(a: EMLTree, b: EMLTree, *, strict: bool = False) -> Node:
    # a^b = exp(b * ln(a))
    return _eml_exp(_eml_mul(b, _eml_ln(a), strict=strict))


# ─── Compile: AST → EMLTree ────────────────────────────────────────

def compile(ast: dict, *, strict: bool = False,
            variables: Optional[Iterable[str]] = None) -> EMLTree:
    """Compile a parsed AST into an EML tree.

    Parameters
    ----------
    ast : dict
        Output of :func:`parse`.
    strict : bool, default ``False``
        If ``True``, refuse inputs that aren't derivable from ``{1, x}``
        alone — i.e. reject numeric literals other than ``0`` / ``1`` and
        named constants other than ``e``. Also routes negation through
        ``eml_ln(1)`` instead of a literal ``0`` leaf, so the compiled
        tree's leaves are all either ``1`` or a variable name.
    variables : iterable of str, optional
        Names to treat as symbolic variables. If given, any other bare
        identifier (other than ``e``, ``pi``, ``π``) raises. If ``None``,
        any bare identifier is accepted as a variable.

    Returns
    -------
    EMLTree
        A :class:`Leaf` or :class:`Node` representing the compiled tree.

    Raises
    ------
    ValueError
        On unsupported operators (trig) or strict-mode violations.
    """
    allowed_vars = set(variables) if variables is not None else None

    def recur(node: dict) -> EMLTree:
        kind = node["k"]

        if kind == "num":
            v = node["v"]
            if strict and v not in (0, 1):
                raise ValueError(
                    f"strict mode: numeric literal {v!r} is not derivable from "
                    "{{1, x}}; rewrite it algebraically (e.g. 2 -> 1+1) or use strict=False"
                )
            return _L(v)

        if kind == "cst":
            name = node["name"]
            if name in ("e", "E"):
                # paper-faithful: e = eml(1, 1)
                return _E(_one(), _one(), "e")
            if name in ("pi", "π", "PI", "Pi"):
                if strict:
                    raise ValueError(
                        "strict mode: π requires a huge bootstrap tree "
                        "(π = -i·ln(-1)); use strict=False"
                    )
                return _L(math.pi, "π")
            # treat as a variable
            if allowed_vars is not None and name not in allowed_vars:
                raise ValueError(
                    f"unknown identifier {name!r}; allowed variables: {sorted(allowed_vars)}"
                )
            return Leaf(value=None, label=name)

        if kind == "un":
            a = recur(node["a"])
            op = node["op"]
            if op == "neg":
                return _eml_neg(a, strict=strict)
            if op == "exp":
                return _eml_exp(a)
            if op in ("ln", "log"):
                return _eml_ln(a)
            if op == "sqrt":
                # sqrt(x) = x^(1/2). Needs 0.5; in strict mode rewrite
                # as x^(1/(1+1)).
                if strict:
                    half = _eml_div(_one(), _eml_add(_one(), _one(), strict=True), strict=True)
                else:
                    half = _L(0.5, "½")
                return _eml_pow(a, half, strict=strict)
            if op in ("sin", "cos", "tan"):
                raise ValueError(
                    f"{op}: trig EML trees are enormous — out of scope per §4.1. "
                    "Use exp/ln with complex arguments if you need a transcendental path."
                )
            raise ValueError(f"unsupported unary operator: {op}")

        if kind == "eml":
            return _E(recur(node["l"]), recur(node["r"]))

        if kind == "bin":
            l = recur(node["l"])
            r = recur(node["r"])
            op = node["op"]
            if op == "+":
                return _eml_add(l, r, strict=strict)
            if op == "-":
                return _eml_sub(l, r)
            if op == "*":
                return _eml_mul(l, r, strict=strict)
            if op == "/":
                return _eml_div(l, r, strict=strict)
            if op == "^":
                return _eml_pow(l, r, strict=strict)
            raise ValueError(f"unsupported binary operator: {op}")

        raise ValueError(f"unknown AST node kind: {kind}")

    return recur(ast)


def compile_expr(expr: str, *, strict: bool = False,
                 variables: Optional[Iterable[str]] = None) -> EMLTree:
    """One-shot: parse a string then compile."""
    return compile(parse(expr), strict=strict, variables=variables)


# ─── Evaluation + tree utilities ───────────────────────────────────

def eval_eml(tree: EMLTree, bindings: Optional[dict] = None, **kwargs) -> complex:
    """Evaluate an EML tree numerically over ``C``.

    Symbolic leaves (bare variables) are resolved via ``bindings``
    and/or ``**kwargs``; kwargs take precedence. Raises ``ValueError``
    if a variable has no binding.

    Uses :func:`numpy.exp` and :func:`numpy.log` (principal branch) on
    ``complex128`` for the ``eml`` operator: ``eml(x, y) = exp(x) - log(y)``.
    This honors the IEEE-754 edge cases the bootstrap chain depends on
    — ``log(0) = -inf + 0j`` and ``exp(-inf) = 0`` — per §6 of ``CLAUDE.md``.
    :mod:`cmath` cannot be used here because it raises on ``log(0)``.
    """
    env = dict(bindings) if bindings else {}
    env.update(kwargs)

    def recur(n: EMLTree) -> complex:
        if isinstance(n, Leaf):
            if n.value is not None:
                return np.complex128(n.value)
            if n.label in env:
                return np.complex128(env[n.label])
            raise ValueError(f"no binding for variable {n.label!r}")
        left = recur(n.left)
        right = recur(n.right)
        return np.exp(left) - np.log(right)

    # suppress "divide by zero encountered in log" — that's the
    # intended IEEE semantics here, not a bug.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = recur(tree)
    return complex(result)


def tree_size(tree: EMLTree) -> int:
    """Total node count (leaves + internals)."""
    if isinstance(tree, Leaf):
        return 1
    return 1 + tree_size(tree.left) + tree_size(tree.right)


def tree_depth(tree: EMLTree) -> int:
    """Depth of the tree (a leaf has depth 0)."""
    if isinstance(tree, Leaf):
        return 0
    return 1 + max(tree_depth(tree.left), tree_depth(tree.right))


def to_string(tree: EMLTree) -> str:
    """Pretty-print an EML tree using the same ``eml(a, b)`` / leaf
    syntax produced by :meth:`eml_sr.EMLTree1D.to_expr`.

    Numeric leaves are emitted as parseable decimal (``"1"``, ``"0.5"``,
    ``"3.141592653589793"``) so that :func:`to_string` round-trips
    through :func:`parse` / :func:`compile`. Symbolic leaves (variables)
    are emitted as their ``label``. For display purposes (``½``, ``π``),
    use :func:`to_string_pretty`.
    """
    if isinstance(tree, Leaf):
        if tree.is_symbolic:
            return tree.label
        # numeric leaf: emit machine-parseable form
        v = tree.value
        if v.imag == 0:
            r = v.real
            if r == int(r):
                return str(int(r))
            return repr(r)
        return f"({v.real}+{v.imag}j)"  # rare; not in standard inputs
    return f"eml({to_string(tree.left)}, {to_string(tree.right)})"


def to_string_pretty(tree: EMLTree) -> str:
    """Like :func:`to_string` but uses decorative labels (``½``, ``π``).

    Matches the JS ``emlStr`` function. Not parseable by :func:`parse`;
    use :func:`to_string` for round-trip.
    """
    if isinstance(tree, Leaf):
        return tree.label
    return f"eml({to_string_pretty(tree.left)}, {to_string_pretty(tree.right)})"


def free_variables(tree: EMLTree) -> set[str]:
    """Collect the set of symbolic variable labels in a tree."""
    out: set[str] = set()

    def recur(n: EMLTree) -> None:
        if isinstance(n, Leaf):
            if n.is_symbolic:
                out.add(n.label)
            return
        recur(n.left)
        recur(n.right)

    recur(tree)
    return out


# ─── CLI ───────────────────────────────────────────────────────────

def _cli(argv: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="eml_compiler",
        description="Compile an elementary expression into an EML tree.",
    )
    p.add_argument("expr", help="Expression to compile, e.g. 'ln(x) + exp(y)'")
    p.add_argument("--strict", action="store_true",
                   help="Paper-faithful mode: reject non-{0,1} numerics and "
                        "non-e named constants.")
    p.add_argument("--eval", nargs="*", metavar="VAR=VALUE",
                   help="Evaluate the compiled tree at the given bindings, "
                        "e.g. --eval x=2 y=1.5. Real values only.")
    p.add_argument("--vars", nargs="*", metavar="NAME",
                   help="Explicit list of allowed variable names.")
    args = p.parse_args(argv)

    try:
        tree = compile_expr(args.expr, strict=args.strict, variables=args.vars)
    except ValueError as ex:
        print(f"compile error: {ex}", file=sys.stderr)
        return 2

    print(to_string(tree))
    print(f"size:  {tree_size(tree)}")
    print(f"depth: {tree_depth(tree)}")

    if args.eval:
        bindings: dict = {}
        for kv in args.eval:
            if "=" not in kv:
                print(f"bad --eval entry {kv!r}; expected NAME=VALUE", file=sys.stderr)
                return 2
            k, v = kv.split("=", 1)
            try:
                bindings[k] = float(v)
            except ValueError:
                print(f"bad numeric value in {kv!r}", file=sys.stderr)
                return 2
        try:
            val = eval_eml(tree, bindings)
        except ValueError as ex:
            print(f"eval error: {ex}", file=sys.stderr)
            return 2
        print(f"value: {val}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
