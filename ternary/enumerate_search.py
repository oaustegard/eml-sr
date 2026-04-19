"""Exhaustive enumeration over small ternary trees.

Generates all pure-grammar trees (``S → x | T(S, S, S)``) up to a
given size bound, evaluates each at a set of test points, and reports
which target primitives match (within ``atol``).

This is the VerifyBaseSet procedure from §2 of the paper, adapted to
the ternary grammar. Because the grammar has branching factor 3 (not 2
as in EML), the tree count explodes fast:

    size 1  (x)              : 1 tree
    size 4  (T(x,x,x))       : 1 tree
    size 7                   : 3 trees  (T-node with one size-4 child)
    size 10                  : 12 trees
    size 13                  : 66 trees
    size 16                  : ~432 trees
    size 19                  : ~2916 trees

(Exact counts come out of the enumerator itself.) Deduplication by
numerical fingerprint at the test points is essential — many trees
are extensionally equivalent even when structurally distinct.

Intended use: find small closed-form ternary constructions for
``e``, ``ln x``, ``-x``, ``1/x``, ``x*x``, ``sqrt(x)`` — the paper's
Table-1 primitives. If none exist below some tractable size bound, that
is evidence that the pure ``S → x | T(S,S,S)`` grammar is not
practically usable even if formally complete.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .bootstrap import T, X, Tree
from .operator import t_np


# Test points — chosen to avoid the grammar's singularities:
#   * x > 0 (real-branch ``ln(x)``)
#   * x != 1 (avoid ln(x)=0 in the denominator)
#   * spread over a few orders of magnitude
DEFAULT_PROBES = (0.3, 0.577, 1.414, math.pi / 4.0, 2.5, 5.0, 10.0)


def _fingerprint(tree: Tree, probes: tuple[float, ...]) -> tuple[complex, ...]:
    """Numerical fingerprint of a tree — used for extensional dedup."""
    with np.errstate(all="ignore"):
        return tuple(tree.eval(complex(x)) for x in probes)


def _round_fp(fp: tuple[complex, ...], digits: int = 10) -> tuple[complex, ...]:
    """Round a fingerprint so near-equal trees collapse to one class."""
    out = []
    for v in fp:
        if not np.isfinite(v.real) or not np.isfinite(v.imag):
            out.append(complex("nan"))
        else:
            out.append(complex(round(v.real, digits), round(v.imag, digits)))
    return tuple(out)


def enumerate_trees(max_size: int, probes=DEFAULT_PROBES,
                    dedup: bool = True) -> dict[int, list[Tree]]:
    """Enumerate pure-grammar trees up to ``max_size`` internal+leaf nodes.

    Returns a dict mapping ``size → [tree, ...]`` of *extensionally
    distinct* trees (one representative per numerical equivalence class
    at the probe points).

    Size increments by 3 for each T-node: size 1 (x) → 4 → 7 → 10 → …
    """
    # Pool[s] = list of trees of exactly size s, deduped.
    pool: dict[int, list[Tree]] = {1: [X()]}
    seen_by_size: dict[int, set[tuple]] = {1: {_round_fp(_fingerprint(X(), probes))}}

    sizes_ascending = [1]
    target_sizes = []
    s = 4
    while s <= max_size:
        target_sizes.append(s)
        s += 3

    for s in target_sizes:
        pool[s] = []
        seen_by_size[s] = set()
        # Compose T(a, b, c) with sizes summing to s-1: |a|+|b|+|c| = s-1
        want = s - 1
        for sa in sizes_ascending:
            for sb in sizes_ascending:
                sc = want - sa - sb
                if sc < 1 or sc not in pool:
                    continue
                for a in pool[sa]:
                    for b in pool[sb]:
                        for c in pool[sc]:
                            node = T(a, b, c)
                            fp = _round_fp(_fingerprint(node, probes))
                            if dedup:
                                # Check against *all* previously seen
                                # (possibly smaller) trees too, so we
                                # always keep the smallest representative.
                                already = any(
                                    fp in seen_by_size.get(ss, set())
                                    for ss in sizes_ascending + [s]
                                )
                                if already:
                                    continue
                            seen_by_size[s].add(fp)
                            pool[s].append(node)
        sizes_ascending.append(s)
    return pool


def search_targets(pool: dict[int, list[Tree]],
                   targets: dict[str, Callable[[complex], complex]],
                   probes=DEFAULT_PROBES,
                   atol: float = 1e-8) -> dict[str, tuple[int, Tree] | None]:
    """For each named target, find the smallest tree matching at probes."""
    out: dict[str, tuple[int, Tree] | None] = {name: None for name in targets}
    for size in sorted(pool.keys()):
        for tree in pool[size]:
            fp = _fingerprint(tree, probes)
            for name, target_fn in targets.items():
                if out[name] is not None:
                    continue
                expected = tuple(complex(target_fn(complex(x))) for x in probes)
                if all(
                    np.isfinite(a.real) and np.isfinite(a.imag)
                    and abs(a - e) < atol * (1 + abs(e))
                    for a, e in zip(fp, expected)
                ):
                    out[name] = (size, tree)
    return out


def default_targets() -> dict[str, Callable[[complex], complex]]:
    """Paper's Table-1-style primitives on the scientific-calculator basis."""
    return {
        "1":        lambda x: 1.0 + 0j,
        "0":        lambda x: 0.0 + 0j,
        "e":        lambda x: complex(math.e),
        "exp(x)":   np.exp,
        "exp(x-1)": lambda x: np.exp(x - 1),
        "ln(x)":    np.log,
        "-x":       lambda x: -x,
        "1/x":      lambda x: 1.0 / x,
        "x*x":      lambda x: x * x,
        "sqrt(x)":  lambda x: np.sqrt(x),
        "x+1":      lambda x: x + 1,
        "x-1":      lambda x: x - 1,
        "2":        lambda x: 2.0 + 0j,
    }


def main(max_size: int = 13):
    print(f"Enumerating pure-grammar ternary trees up to size {max_size}")
    print("(grammar: S → x | T(S, S, S))")
    pool = enumerate_trees(max_size)
    total = sum(len(v) for v in pool.values())
    print(f"  distinct trees: {total}")
    for size, trees in sorted(pool.items()):
        print(f"    size {size:>3}: {len(trees):>5} extensional classes")

    print("\nSearching for Table-1 primitives:")
    targets = default_targets()
    found = search_targets(pool, targets)
    for name, hit in found.items():
        if hit is None:
            print(f"  {name:<10}  NOT FOUND at size ≤ {max_size}")
        else:
            size, tree = hit
            print(f"  {name:<10}  size {size:<3}  {tree}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    main(n)
