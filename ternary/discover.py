"""Gradient-based recovery benchmark for the ternary operator.

Runs :func:`ternary.tree.train_one` across a matrix of
``(target, depth, grammar) × seeds`` and reports success/failure rates.

Two grammars:
    * **pure**     — ``S → x | T(S, S, S)``. No constants. Matches §5's
      "open question" form. Tree must synthesise all constants via
      ``T(x, x, x) = 1`` subtrees and their compositions.
    * **relaxed** — ``S → c | x | T(S, S, S)``. Learnable per-leaf and
      per-gate complex constants. Matches the practical form noted in
      issue #37's design-decisions section, analogous to how
      :class:`eml_sr.EMLTree1D` has a terminal-constant leaf option.

Success criterion: final snapped MSE < ``1e-6`` at the training points.
Ternary trees have worse conditioning than EML (``ln`` in both numerator
and denominator simultaneously), so machine-epsilon MSE is rare even for
the primitives that are formally reachable.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .tree import train_one


DEFAULT_XS = np.linspace(0.2, 3.0, 40)


@dataclass(frozen=True)
class Target:
    name: str
    fn: Callable[[np.ndarray], np.ndarray]


TARGETS = [
    Target("1",        lambda x: np.ones_like(x)),
    Target("exp(x)",   np.exp),
    Target("exp(x-1)", lambda x: np.exp(x - 1)),
    Target("ln(x)",    np.log),
    Target("-x",       lambda x: -x),
    Target("1/x",      lambda x: 1.0 / x),
    Target("x*x",      lambda x: x * x),
    Target("e",        lambda x: np.full_like(x, math.e)),
]


def run_one(target: Target, depth: int, allow_terminal: bool,
            seed: int, xs: np.ndarray = DEFAULT_XS) -> dict:
    xs_t = torch.tensor(xs, dtype=torch.float64)
    ys = target.fn(xs)
    ys_t = torch.tensor(ys, dtype=torch.complex128)

    # Effort scales with tree size — match the EML engine's budget scheme.
    n_leaves = 3 ** depth
    s_iters = 400 + 300 * depth
    h_iters = 150 + 150 * depth

    r = train_one(
        xs_t, ys_t, depth=depth, seed=seed,
        search_iters=s_iters, hard_iters=h_iters,
        allow_terminal=allow_terminal,
    )
    return {
        "target": target.name,
        "depth": depth,
        "allow_terminal": allow_terminal,
        "seed": seed,
        "best_mse": r["best_mse"],
        "final_mse": r["final_mse"],
        "n_leaves": n_leaves,
    }


def run_matrix(targets=None, depths=(2, 3), n_seeds: int = 3,
               xs: np.ndarray = DEFAULT_XS, verbose: bool = True) -> list[dict]:
    if targets is None:
        targets = TARGETS
    results = []
    t0 = time.time()
    for target in targets:
        for depth in depths:
            for allow_terminal in (False, True):
                best = float("inf")
                for seed in range(n_seeds):
                    r = run_one(target, depth, allow_terminal, seed, xs=xs)
                    best = min(best, r["final_mse"])
                    results.append(r)
                tag = "relaxed" if allow_terminal else "pure   "
                if verbose:
                    print(f"  {target.name:<10} d{depth} {tag}  "
                          f"best MSE over {n_seeds} seeds = {best:.3e}")
    if verbose:
        print(f"\ntotal wall time: {time.time() - t0:.1f}s")
    return results


def summarise(results: list[dict], success_threshold: float = 1e-6):
    """Aggregate into a (target, depth, grammar) → (hits, n, best_mse) table."""
    out: dict[tuple[str, int, bool], dict] = {}
    for r in results:
        key = (r["target"], r["depth"], r["allow_terminal"])
        entry = out.setdefault(key, {"hits": 0, "n": 0, "best": float("inf")})
        entry["n"] += 1
        if r["final_mse"] < success_threshold:
            entry["hits"] += 1
        entry["best"] = min(entry["best"], r["final_mse"])
    return out


def print_markdown_table(summary: dict, output_file=None):
    """Emit a GitHub-flavoured markdown table."""
    # One table per (grammar).
    import io
    buf = io.StringIO()
    for grammar_label, grammar_flag in [("pure (S → x | T(S,S,S))", False),
                                        ("relaxed (S → c | x | T(S,S,S))", True)]:
        print(f"\n### Recovery: {grammar_label}\n", file=buf)
        print("| target | depth | hits / n | best final MSE |", file=buf)
        print("|---|---|---|---|", file=buf)
        for (target, depth, flag), entry in sorted(summary.items()):
            if flag != grammar_flag:
                continue
            print(f"| {target} | {depth} | "
                  f"{entry['hits']}/{entry['n']} | "
                  f"{entry['best']:.2e} |", file=buf)
    s = buf.getvalue()
    print(s)
    if output_file is not None:
        with open(output_file, "w") as f:
            f.write(s)
    return s


def main():
    results = run_matrix(depths=(2, 3), n_seeds=3)
    summary = summarise(results)
    print_markdown_table(summary)


if __name__ == "__main__":
    main()
