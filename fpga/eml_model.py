"""Bit-exact model of the fixed-point EML cell in rtl/eml_cell.v.

eml(x, y) = exp(x) - ln(y), inputs and output in signed Q16.16 (32 bits,
value = int / 65536).  The RTL is checked against this model bit for bit
(tb/); accuracy is measured by comparing this model with float64.

Design (one choice among several; see RESULTS.md for why):

  exp(x) = 2**(x*log2 e).  t = x*K with K = round(log2(e) * 2**25); integer
  part n shifts, the 24-bit fraction f indexes a 256-entry table of
  2**(i/256) with linear interpolation on the low 16 bits.
  ln(y)  = (p - 16)*ln 2 + ln(m), p the leading-one position of y and
  m = y / 2**p in [1, 2); ln(m) from a 256-entry table with linear
  interpolation.
  Each table entry is shifted by half the segment's chord error, so the
  interpolation error is centred (minimax) instead of one-sided.

Out-of-range behaviour: exp saturates for x*log2 e >= 15 and flushes to 0
below 2**-17; y <= 0 raises the `invalid` flag and the output saturates to
the largest positive value (ln y -> -inf).  The difference saturates to the
int32 range.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

FRAC = 16
ONE = 1 << FRAC
INT_MAX = (1 << 31) - 1
INT_MIN = -(1 << 31)

TBL_BITS = 8
TBL_N = 1 << TBL_BITS
C_FRAC = 30  # table base fraction bits (Q2.30)
S_FRAC = 24  # table slope fraction bits: keeps each slope <= 17 bits, so the
             # interpolation multiply fits one 18x18 DSP

K_FRAC = 25
K_LOG2E = round(math.log2(math.e) * (1 << K_FRAC))  # 26-bit constant
LN2_Q30 = round(math.log(2) * (1 << C_FRAC))

# x is clamped to [-16, 16) before the constant multiply; anything outside
# is already saturated or flushed.
X_CLAMP_HI = (16 << FRAC) - 1
X_CLAMP_LO = -(16 << FRAC)


def _segment_tables(fn, lo: float, hi: float, convex: bool):
    """Base and slope tables for linear interpolation of fn on [lo, hi)."""
    base, slope = [], []
    h = (hi - lo) / TBL_N
    for i in range(TBL_N):
        a = lo + i * h
        b = a + h
        fa, fb = fn(a), fn(b)
        # max chord error on the segment, sampled densely
        ts = np.linspace(0.0, 1.0, 257)
        err = (fa + (fb - fa) * ts) - fn(a + h * ts)
        e_max = float(err.max() if convex else -err.min())
        shift = -e_max / 2 if convex else e_max / 2
        base.append(round((fa + shift) * (1 << C_FRAC)))
        slope.append(round((fb - fa) * (1 << S_FRAC)))
    return base, slope


def _exp2_fn(t):
    return np.exp2(t)


def _ln1p_fn(t):
    return np.log(t)


E0, ES = _segment_tables(_exp2_fn, 0.0, 1.0, convex=True)
L0, LS = _segment_tables(_ln1p_fn, 1.0, 2.0, convex=False)

E0_W, ES_W = 31, 17
L0_W, LS_W = 30, 16
assert max(E0) < (1 << E0_W) and min(E0) >= 0
assert max(ES) < (1 << ES_W) and min(ES) >= 0
assert max(L0) < (1 << L0_W) and min(L0) >= 0
assert max(LS) < (1 << LS_W) and min(LS) >= 0


def write_hex(dirpath: Path) -> None:
    """Write the four ROM images for $readmemh."""
    dirpath.mkdir(parents=True, exist_ok=True)
    for name, vals, w in (("exp_base", E0, E0_W), ("exp_slope", ES, ES_W),
                          ("ln_base", L0, L0_W), ("ln_slope", LS, LS_W)):
        digits = (w + 3) // 4
        (dirpath / f"{name}.hex").write_text(
            "".join(f"{v:0{digits}x}\n" for v in vals))


# ---------------------------------------------------------------- model

def exp_q(x):
    """exp of Q16.16 int64 array -> Q16.16 int64 (saturated)."""
    x = np.asarray(x, dtype=np.int64)
    xc = np.clip(x, X_CLAMP_LO, X_CLAMP_HI)
    t = xc * K_LOG2E                      # Q.41
    t24 = t >> (FRAC + K_FRAC - 24)       # Q.24, floor
    n = t24 >> 24                         # floor
    f = t24 & ((1 << 24) - 1)
    idx = f >> 16
    r = f & 0xFFFF
    e0 = np.array(E0, dtype=np.int64)[idx]
    es = np.array(ES, dtype=np.int64)[idx]
    m = e0 + ((es * r) >> (16 + S_FRAC - C_FRAC))   # Q.30 in [1, 2)
    sh = 14 - n                           # >= 0 when not saturated
    sat = n >= 15
    sh_c = np.clip(sh, 0, 63)
    rnd = np.where(sh_c > 0, np.left_shift(np.int64(1), np.maximum(sh_c - 1, 0)), 0)
    out = np.where(sh_c >= 32, 0, (m + rnd) >> sh_c)
    out = np.where(sat, INT_MAX, out)
    return np.minimum(out, INT_MAX)


def ln_q(y):
    """ln of Q16.16 int64 array -> (Q16.16 int64, invalid bool)."""
    y = np.asarray(y, dtype=np.int64)
    invalid = y <= 0
    yp = np.where(invalid, 1, y)
    p = np.zeros_like(yp)
    for k in range(1, 31):
        p += (yp >= (1 << k)).astype(np.int64)
    norm = yp << (30 - p)                 # Q1.30 in [2**30, 2**31)
    idx = (norm >> 22) & 0xFF
    r = (norm >> 6) & 0xFFFF
    l0 = np.array(L0, dtype=np.int64)[idx]
    ls = np.array(LS, dtype=np.int64)[idx]
    lm = l0 + ((ls * r) >> (16 + S_FRAC - C_FRAC))   # Q.30 ln(m)
    ln30 = (p - FRAC) * LN2_Q30 + lm
    out = (ln30 + (1 << 13)) >> 14
    return out, invalid


def eml_q(x, y):
    """Q16.16 eml(x, y) -> (Q16.16 int64, invalid bool)."""
    e = exp_q(x)
    lg, inv = ln_q(y)
    d = np.clip(e - lg, INT_MIN, INT_MAX)
    d = np.where(inv, INT_MAX, d)
    return d, inv


def to_q(v):
    return np.round(np.asarray(v, dtype=np.float64) * ONE).astype(np.int64)


def from_q(q):
    return np.asarray(q, dtype=np.float64) / ONE


if __name__ == "__main__":
    write_hex(Path(__file__).resolve().parent / "rtl")
    print("tables written; K_LOG2E", K_LOG2E, "LN2_Q30", LN2_Q30)
