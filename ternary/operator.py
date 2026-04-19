"""Ternary operator T from §5 of Odrzywolek (2026).

The paper's §5 (p.16) states:

    A ternary operator, T(x, y, z) = e^x / ln x × ln z / e^y,
    for which T(x, x, x) = 1, is next candidate for further analysis [47].

With left-to-right math precedence this parses as

    T(x, y, z) = (exp(x) / ln(x)) * (ln(z) / exp(y))
               = exp(x - y) * ln(z) / ln(x)

Issue #37's initial transcription used ``exp(x / ln x)`` (division under the
exponent); that is a parsing typo — the correct reading is ``exp(x) / ln(x)``.
The verification script in :mod:`ternary.verify_formula` confirms the correct
parsing symbolically and numerically.

Domain restrictions (real branch):
    * ``x > 0``            (``ln(x)`` defined)
    * ``x != 1``           (``ln(x) != 0``, avoids divide-by-zero)
    * ``z > 0``            (``ln(z)`` defined)

The complex branch of ``ln`` extends the domain to ``x, z != 0`` but the
branch cut still excludes ``x = 1``.
"""

from __future__ import annotations

import numpy as np
import torch


DTYPE = torch.complex128
_CLAMP = 1e300


def t_np(x, y, z):
    """NumPy ternary operator over complex128.

    ``T(x, y, z) = exp(x) * ln(z) / (ln(x) * exp(y))``.
    """
    x = np.asarray(x, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128)
    z = np.asarray(z, dtype=np.complex128)
    return np.exp(x) * np.log(z) / (np.log(x) * np.exp(y))


def t_torch(x, y, z):
    """Torch ternary operator over complex128, differentiable."""
    return torch.exp(x) * torch.log(z) / (torch.log(x) * torch.exp(y))


def t_clamped(x, y, z):
    """Torch ternary operator with NaN/inf scrubbing.

    Used inside the training forward pass — matches the defensive clamping
    in :class:`eml_sr.EMLTree1D` so gradients flow through numerically
    problematic intermediates without poisoning the rest of the tree.
    """
    val = t_torch(x, y, z)
    return torch.complex(
        torch.nan_to_num(val.real, nan=0.0, posinf=_CLAMP, neginf=-_CLAMP)
            .clamp(-_CLAMP, _CLAMP),
        torch.nan_to_num(val.imag, nan=0.0, posinf=_CLAMP, neginf=-_CLAMP)
            .clamp(-_CLAMP, _CLAMP),
    )
