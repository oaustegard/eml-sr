"""Write tb/vectors.txt for tb_eml_cell.v from the bit-exact model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import eml_model as M


def vectors(n_random: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    # in-domain: x where exp fits, y spanning all positive magnitudes
    x1 = M.to_q(rng.uniform(-12.0, 10.5, n_random))
    y1 = M.to_q(np.exp(rng.uniform(np.log(2.0**-16), np.log(32767.0), n_random)))
    # raw 32-bit patterns (saturation, negatives, zero)
    x2 = rng.integers(M.INT_MIN, M.INT_MAX, n_random // 4, endpoint=True)
    y2 = rng.integers(M.INT_MIN, M.INT_MAX, n_random // 4, endpoint=True)
    edges = np.array([M.INT_MIN, M.INT_MIN + 1, -1, 0, 1, 2, M.ONE - 1, M.ONE,
                      M.ONE + 1, M.INT_MAX - 1, M.INT_MAX], dtype=np.int64)
    ex, ey = np.meshgrid(edges, edges)
    # every exp table segment boundary and every power of two for y
    x3 = M.to_q(np.arange(-24 * 256, 24 * 256) / 256.0 * np.log(2))
    y3 = np.concatenate([np.left_shift(np.int64(1), np.arange(31)),
                         np.left_shift(np.int64(1), np.arange(31)) - 1])
    y3 = np.resize(y3[y3 > 0], x3.size)
    x = np.concatenate([x1, x2, ex.ravel(), x3])
    y = np.concatenate([y1, y2, ey.ravel(), y3])
    out, inv = M.eml_q(x, y)
    return x, y, out, inv


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    x, y, out, inv = vectors(n)
    m = 0xFFFFFFFF
    lines = [f"{int(a) & m:08x} {int(b) & m:08x} {int(c) & m:08x} {int(d)}\n"
             for a, b, c, d in zip(x, y, out, inv)]
    (HERE / "vectors.txt").write_text("".join(lines))
    print(f"wrote {len(lines)} vectors")


if __name__ == "__main__":
    main()
