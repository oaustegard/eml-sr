"""Operator configuration for EML and its cousins.

The paper's §3 (p.9) and §5 (p.15) name two cousins of EML that claim the
same universal-basis property over the elementary functions:

    eml(x, y)     = exp(x) - ln(y)     terminal: 1        (id of -, ln(1)=0)
    edl(x, y)     = exp(x) / ln(y)     terminal: e        (id of /, ln(e)=1)
    neg_eml(x, y) = ln(x) - exp(y)     terminal: -inf     (id of -, exp(-inf)=0)

This module packages each choice as an :class:`OperatorConfig` so the
search engine (:mod:`eml_sr`), the compiler (:mod:`eml_compiler`), and
benchmarks can be parameterized on operator choice at a single point.

Identity derivations
--------------------
For EML the identities are in the paper directly. For EDL and NEG_EML
they were derived here and verified numerically by enumeration over
depth-3 trees (see ``tests/test_cousin_identities.py``).

EDL identities (terminal e, operator exp(x)/ln(y))::

    exp(x) = edl(x, e)                              size 3, depth 1
    ln(x)  = edl(e, edl(edl(e, x), e))              size 7, depth 3
    1      = edl(e, edl(edl(e, e), e))              size 7, depth 3

NEG_EML identities (terminal -inf, operator ln(x)-exp(y))::

    ln(x)  = ne(x, -inf)                            size 3, depth 1
    exp(x) = ne(x, ne(ne(x, x), -inf))              size 7, depth 3
    0      = ne(x, ne(ne(x, -inf), -inf))           size 7, depth 3 (uses x)
    ln(ln(x)) = ne(ne(x, -inf), -inf)               size 5, depth 2

The NEG_EML exp(x) derivation flows through complex intermediates
(``ln(ln(x)-exp(x))`` has negative argument for x>0), so callers must
use ``complex128`` throughout. Division, multiplication, and negation
in the cousins are delegated to ``exp``/``ln`` + add/sub/neg composition
via the usual identities (``a*b = exp(ln(a)+ln(b))``, etc.) and inherit
the resulting tree sizes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch


DTYPE = torch.complex128

# Large finite stand-in for -inf during training. IEEE ``float('-inf')``
# propagates through ``exp`` cleanly (→ 0) but poisons gradients at
# ``ln(-inf)``; the clamp in the training loop already bounds values to
# ``1e300``, so using a large finite value keeps numerical behaviour
# consistent and lets autograd produce meaningful gradients on the rare
# paths that pass through a ``-inf`` leaf in left slot.
_NEG_INF_APPROX = -1e30


# ─── Torch operators ───────────────────────────────────────────────

def _eml_op(x, y):
    """EML: exp(x) - ln(y). Identity of ``-``, terminal 1."""
    return torch.exp(x) - torch.log(y)


def _edl_op(x, y):
    """EDL: exp(x) / ln(y). Identity of ``/``, terminal e."""
    return torch.exp(x) / torch.log(y)


def _neg_eml_op(x, y):
    """−EML: ln(x) - exp(y). Identity of ``-`` on right, terminal -inf."""
    return torch.log(x) - torch.exp(y)


# ─── NumPy operators (for compiler eval) ───────────────────────────

def _eml_np(x, y):
    return np.exp(x) - np.log(y)


def _edl_np(x, y):
    return np.exp(x) / np.log(y)


def _neg_eml_np(x, y):
    return np.log(x) - np.exp(y)


# ─── Config ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OperatorConfig:
    """Bundle of ``(op, terminal, identities)`` for a binary operator.

    Used to parameterize the search engine, simplifier, and compiler on
    operator choice. Do not instantiate directly unless adding a new
    operator variant — use the module-level :data:`EML`, :data:`EDL`,
    :data:`NEG_EML` constants.

    Attributes
    ----------
    name:
        Short identifier (``"eml" | "edl" | "neg_eml"``). Used in
        benchmark output and logging.
    op:
        Torch-differentiable ``(x, y) -> value`` operator. Must accept
        and return ``torch.complex128`` tensors.
    op_numpy:
        NumPy equivalent for static evaluation (compiler, tests).
    terminal:
        The distinguished constant that plays the identity role for the
        operator's "subtraction" side. Either a finite complex number
        (``1`` for EML, ``e`` for EDL) or a sentinel for ``-inf``
        (NEG_EML).
    terminal_label:
        String used when printing trees (``"1"``, ``"e"``, ``"-inf"``).
    is_neg_inf_terminal:
        Set when ``terminal`` is the symbolic ``-inf``. The training
        loop uses :data:`_NEG_INF_APPROX` numerically and preserves the
        symbol when emitting expressions.
    simplifier_enabled:
        Whether the EML-specific simplifier in :mod:`eml_sr._simplify`
        applies. False for the cousins — their identity tables don't
        match the EML rewrites and naive reuse produces wrong output.
    """
    name: str
    op: Callable
    op_numpy: Callable
    terminal: complex
    terminal_label: str
    is_neg_inf_terminal: bool = False
    simplifier_enabled: bool = True

    @property
    def terminal_numeric(self) -> complex:
        """Finite numeric stand-in for the terminal during training."""
        if self.is_neg_inf_terminal:
            return complex(_NEG_INF_APPROX, 0.0)
        return complex(self.terminal)

    def op_str(self) -> str:
        """Human-readable operator definition."""
        if self.name == "eml":
            return "eml(x, y) = exp(x) - ln(y)"
        if self.name == "edl":
            return "edl(x, y) = exp(x) / ln(y)"
        if self.name == "neg_eml":
            return "neg_eml(x, y) = ln(x) - exp(y)"
        return f"{self.name}(x, y)"


# ─── Module-level configs ──────────────────────────────────────────

EML = OperatorConfig(
    name="eml",
    op=_eml_op,
    op_numpy=_eml_np,
    terminal=complex(1.0, 0.0),
    terminal_label="1",
    simplifier_enabled=True,
)

EDL = OperatorConfig(
    name="edl",
    op=_edl_op,
    op_numpy=_edl_np,
    terminal=complex(math.e, 0.0),
    terminal_label="e",
    simplifier_enabled=False,
)

NEG_EML = OperatorConfig(
    name="neg_eml",
    op=_neg_eml_op,
    op_numpy=_neg_eml_np,
    terminal=complex(_NEG_INF_APPROX, 0.0),
    terminal_label="-inf",
    is_neg_inf_terminal=True,
    simplifier_enabled=False,
)


ALL_OPERATORS: dict[str, OperatorConfig] = {
    "eml": EML,
    "edl": EDL,
    "neg_eml": NEG_EML,
}


def get(name: str) -> OperatorConfig:
    """Look up a registered operator by name (``eml``, ``edl``, ``neg_eml``)."""
    try:
        return ALL_OPERATORS[name]
    except KeyError as ex:
        raise KeyError(
            f"unknown operator {name!r}; "
            f"known: {sorted(ALL_OPERATORS)}"
        ) from ex
