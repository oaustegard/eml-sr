"""Write tb/vectors.txt for tb_eml_tree.v (bit-exact tree model)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import eml_model as M
from gen_tree import TREES, eval_tree_q

from benchmarks.perm_det_eml import parse


def main() -> None:
    name = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
    rng = np.random.default_rng(2)
    a = M.to_q(np.exp(rng.uniform(np.log(1e-3), np.log(40.0), n)))
    b = M.to_q(np.exp(rng.uniform(np.log(1e-3), np.log(40.0), n)))
    c = rng.integers(M.INT_MIN, M.INT_MAX, n // 4, endpoint=True)
    d = rng.integers(M.INT_MIN, M.INT_MAX, n // 4, endpoint=True)
    x = np.concatenate([a, c, M.to_q(rng.uniform(-3, 3, n // 4))])
    y = np.concatenate([b, d, M.to_q(rng.uniform(-3, 3, n // 4))])
    out, _ = eval_tree_q(parse(TREES[name]), {"x0": x, "x1": y})
    # root flag only (the RTL exposes the root cell's invalid)
    t = parse(TREES[name])
    a_, _ = eval_tree_q(t[0], {"x0": x, "x1": y})
    b_, _ = eval_tree_q(t[1], {"x0": x, "x1": y})
    _, inv = M.eml_q(a_, b_)
    m = 0xFFFFFFFF
    (HERE / "vectors.txt").write_text("".join(
        f"{int(p) & m:08x} {int(q) & m:08x} {int(r) & m:08x} {int(s)}\n"
        for p, q, r, s in zip(x, y, out, inv)))
    print(f"{name}: wrote {x.size} vectors")


if __name__ == "__main__":
    main()
