"""Head-to-head benchmark: eml-sr vs PySR on a common target set (issue #33).

PySR (Cranmer, MIT) is the standard symbolic-regression baseline: a
C-compiled evolutionary search over a rich heterogeneous operator set
(``+ - * / exp log sqrt sin cos ...``). eml-sr uses a single primitive
(``eml(x, y) = exp(x) - ln(y)``) plus terminals — every non-leaf is the
same operator. That grammar uniformity is the paper's whole point; it is
also the thing that makes a head-to-head interesting.

The benchmark does *not* try to prove eml-sr is faster — PySR almost
certainly will be. It measures three dimensions that actually matter
for positioning:

1. **Grammar uniformity.** Every eml-sr node is ``eml``; PySR trees are
   heterogeneous ASTs over the operator set above.
2. **Exact recovery vs Pareto front.** When the generating law is an
   elementary function, eml-sr targets machine-epsilon RMSE and a
   specific symbolic form. PySR produces a Pareto front trading
   complexity for RMSE and often lands near, but not on, the target.
3. **Expression length.** EML trees for standard operations are *long*
   by design — multiplication is depth 8 in EML, depth 1 in conventional
   SR. That's the cost of uniformity, not a bug.

Run::

    PYSR_ENABLED=1 python -m benchmarks.pysr_compare
    PYSR_ENABLED=1 python -m benchmarks.pysr_compare --quick
    PYSR_ENABLED=1 python -m benchmarks.pysr_compare --output benchmarks/pysr_compare.md

When ``PYSR_ENABLED`` is unset (or PySR is not installed), the PySR
columns are skipped and only eml-sr is benchmarked — still useful as a
standalone baseline table.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# Allow running as a script from the repo root or as `python -m benchmarks.pysr_compare`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from eml_sr import REAL, Normalizer, discover, discover_curriculum  # noqa: E402
from eml_compiler import compile_expr, tree_size  # noqa: E402
from benchmarks.feynman import PROBLEMS as FEYNMAN_PROBLEMS, FeynmanProblem  # noqa: E402


# ─── PySR availability ────────────────────────────────────────────

def _pysr_available() -> tuple[bool, str]:
    """Return (available, reason). PySR is gated on env var + import."""
    if os.environ.get("PYSR_ENABLED", "") != "1":
        return False, "PYSR_ENABLED != '1' (set to enable)"
    try:
        import pysr  # noqa: F401
        return True, ""
    except ImportError as e:
        return False, f"pysr import failed: {e}"


# ─── Target set ───────────────────────────────────────────────────

# Quick slice: the eight univariate targets the feynman runner defaults to,
# plus the small multivariate subset that `test_e2e_multivariate.py` exercises
# (the depth-1 bivariate atom, the exp-of-sum that needs addition, and the
# coulomb-like division target).
#
# We reuse feynman.PROBLEMS directly rather than restating the formulas so
# the two benchmarks stay in lockstep. The mapping below selects by feynman_id.

QUICK_IDS = [
    # Univariate: the catalogue's first 8
    "eml.exp", "eml.eml", "eml.const-e", "eml.e-ln",
    "eml.lnx", "eml.expexp", "eml.expexpexp", "eml.exp-1",
    # Multivariate: the ones test_e2e_multivariate.py covers directly or by family
    "mv.eml", "mv.expsum", "II.6.15a",
]

FULL_IDS = QUICK_IDS + [
    "I.34.8", "I.12.1", "I.14.3",            # univariate linear targets
    "mv.ln-ratio", "mv.sum-exp",              # more multivariate stress
]


def _select_problems(ids: list[str]) -> list[FeynmanProblem]:
    by_id = {p.feynman_id: p for p in FEYNMAN_PROBLEMS}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise RuntimeError(
            f"feynman catalogue missing expected IDs: {missing}. "
            "Either feynman.py has been refactored or QUICK_IDS needs updating."
        )
    return [by_id[i] for i in ids]


# ─── Reference EML size via compiler ───────────────────────────────

def _reference_eml_size(prob: FeynmanProblem) -> Optional[int]:
    """Compile the target formula into its canonical EML tree and return
    node count. Returns None if the compiler can't handle the formula
    (e.g. a lambda-only ``projection_const`` has no symbolic form, or the
    formula uses operators outside the compiler's grammar).

    We translate a few feynman notations to compiler syntax:
      ``exp(x1 + x2) = exp(x1)*exp(x2)`` → pass the RHS equivalent
      ``x1 / x2^2`` → ``x1 / (x2 ^ 2)``
    """
    formula = prob.formula
    # The feynman catalogue sometimes stores two forms separated by ` = `;
    # keep whichever parses (prefer the equivalent that stays within the
    # compiler's grammar — additions are representable, so either works).
    candidates = [s.strip() for s in formula.split("=")]
    for expr in candidates:
        # Skip obviously-non-formula entries (e.g. "e" constant)
        if not expr or expr in ("e",):
            # 'e' is a primitive — build it directly
            if expr == "e":
                try:
                    return tree_size(compile_expr("eml(1, 1)"))
                except Exception:
                    return None
            continue
        # Feynman uses ^ for power; compiler accepts ^
        expr_norm = expr.replace(" ", "")
        # Detect variables — compiler needs them declared for tree_size
        # but the parser auto-detects; just try.
        try:
            tree = compile_expr(expr)
            return tree_size(tree)
        except Exception:
            continue
    return None


# ─── Data sampling ─────────────────────────────────────────────────

def _sample(prob: FeynmanProblem, n_samples: int = 100,
            seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sample a fresh (X, y) pair with fixed seeding per feynman_id.

    Univariate: linspace over the declared range (matches the feynman runner).
    Multivariate: uniform per-column, seeded on (feynman_id, seed).
    """
    if prob.n_vars == 1:
        lo, hi = prob.x_ranges[0]
        X = np.linspace(lo, hi, n_samples)
    else:
        h = abs(hash((prob.feynman_id, seed))) % (2**32)
        rng = np.random.default_rng(h)
        cols = [rng.uniform(lo, hi, n_samples) for (lo, hi) in prob.x_ranges]
        X = np.stack(cols, axis=1)
    y = prob.fn(X)
    return X, y


# ─── eml-sr runner ─────────────────────────────────────────────────

@dataclass
class EngineResult:
    """Per-target, per-engine-config result row."""
    engine: str                     # "eml-sr/discover", "pysr/default", ...
    wall_time: float
    rmse: float
    exact: bool                     # rmse < threshold
    expr: str
    size: Optional[int]             # node count (None if not reported)
    extra: dict = field(default_factory=dict)


def _run_eml_sr(prob: FeynmanProblem, *, method: str, max_depth: int,
                n_tries: int, threshold: float) -> EngineResult:
    X, y = _sample(prob)
    # Paper-faithful: no normalization (normalizing destroys symbolic
    # recoverability for nonlinear targets — see feynman.py §issue-11 note).
    # We still wrap in Normalizer(mode='none') for API uniformity.
    norm = Normalizer.fit(X, y, mode="none")
    X_n = norm.transform_x(X)
    y_n = norm.transform_y(y)

    t0 = time.time()
    if method == "curriculum":
        result = discover_curriculum(
            X_n, y_n, max_depth=max_depth, n_tries=n_tries,
            verbose=False, success_threshold=threshold,
        )
    else:
        result = discover(
            X_n, y_n, max_depth=max_depth, n_tries=n_tries,
            verbose=False, success_threshold=threshold, n_workers=1,
        )
    elapsed = time.time() - t0

    if result is None:
        return EngineResult(
            engine=f"eml-sr/{method}",
            wall_time=elapsed, rmse=float("inf"), exact=False,
            expr="<no formula>", size=None,
        )

    # Original-space RMSE (mode='none' makes this trivial, but keep the
    # inverse-transform for robustness if someone flips the mode).
    snapped = result.get("snapped_tree")
    rmse_orig = float("inf")
    node_count = None
    if snapped is not None:
        with torch.no_grad():
            pred_n, _, _ = snapped(torch.tensor(X_n, dtype=REAL), tau=0.01)
            pred_np = norm.inverse_y(pred_n.real.detach().numpy())
            # A snapped tree can overflow float64 on unrecovered targets
            # (e.g. 9.81*x forced through an exp chain). The rmse we compute
            # is correctly inf in that case; the RuntimeWarning is noise.
            with np.errstate(over="ignore", invalid="ignore"):
                rmse_orig = float(np.sqrt(np.mean((pred_np - y) ** 2)))
        # Tree size = internal eml nodes + leaves. Matches the compiler's
        # tree_size() convention so the "ref EML size" column is
        # apples-to-apples. `discover` returns an ``EMLTree1D`` (with
        # ``n_internal`` / ``n_leaves``); ``discover_curriculum`` returns
        # a ``GrowingEMLTree`` whose reachable subset is ``active_nodes()``.
        if hasattr(snapped, "active_nodes"):
            try:
                node_count = len(snapped.active_nodes())
            except Exception:
                node_count = None
        else:
            try:
                node_count = int(snapped.n_internal + snapped.n_leaves)
            except AttributeError:
                node_count = None

    return EngineResult(
        engine=f"eml-sr/{method}",
        wall_time=elapsed,
        rmse=rmse_orig,
        exact=math.isfinite(rmse_orig) and rmse_orig < threshold,
        expr=str(result.get("expr", "<no expr>"))[:80],
        size=node_count,
        extra={"depth": result.get("depth")},
    )


# ─── PySR runner ───────────────────────────────────────────────────

# PySR operator sets kept conservative: the same primitives the EML
# vocabulary can *represent*, so "PySR found a smaller tree" is a
# fair comparison rather than "PySR used sin, which EML can't".
# (Trig is out of scope per Odrzywolek §4.1.)
_PYSR_BINARY = ["+", "-", "*", "/"]
_PYSR_UNARY = ["exp", "log", "sqrt"]


def _run_pysr(prob: FeynmanProblem, *, config: str,
              threshold: float, deterministic: bool = False) -> EngineResult:
    """Run PySR with one of two configs: 'default' or 'permissive'.

    ``deterministic=True`` forces ``parallelism='serial'`` and a fixed
    ``random_state``; PySR's bare ``random_state`` without this pairing
    is a no-op and emits a warning per fit. Serial mode loses PySR's
    multi-population parallel speedup — roughly ~10× slower — so default
    is off. See issue #50.
    """
    from pysr import PySRRegressor

    if config == "default":
        kwargs = dict(niterations=40, populations=15, population_size=33)
    elif config == "permissive":
        kwargs = dict(niterations=100, populations=30, population_size=50)
    else:
        raise ValueError(f"unknown pysr config: {config}")

    X, y = _sample(prob)
    # PySR expects 2D X even for univariate.
    X_2d = X.reshape(-1, 1) if X.ndim == 1 else X

    parallelism = "serial" if deterministic else "multiprocessing"
    model = PySRRegressor(
        binary_operators=_PYSR_BINARY,
        unary_operators=_PYSR_UNARY,
        model_selection="best",
        progress=False,
        verbosity=0,
        temp_equation_file=True,
        deterministic=deterministic,
        parallelism=parallelism,
        random_state=0 if deterministic else None,
        **kwargs,
    )

    t0 = time.time()
    try:
        model.fit(X_2d, y)
    except Exception as e:
        return EngineResult(
            engine=f"pysr/{config}",
            wall_time=time.time() - t0, rmse=float("inf"), exact=False,
            expr=f"<pysr error: {type(e).__name__}>", size=None,
        )
    elapsed = time.time() - t0

    # Pull the model-selected best equation.
    try:
        best = model.get_best()
    except Exception as e:
        return EngineResult(
            engine=f"pysr/{config}",
            wall_time=elapsed, rmse=float("inf"), exact=False,
            expr=f"<pysr get_best failed: {type(e).__name__}>", size=None,
        )

    # PySR's best is a pandas Series with 'loss', 'complexity', 'equation',
    # 'sympy_format', etc. 'loss' is MSE by default.
    loss = float(best.get("loss", float("inf")))
    rmse = math.sqrt(loss) if math.isfinite(loss) and loss >= 0 else float("inf")
    complexity = int(best["complexity"]) if "complexity" in best else None
    equation_str = str(best.get("equation", "<no equation>"))[:80]

    return EngineResult(
        engine=f"pysr/{config}",
        wall_time=elapsed,
        rmse=rmse,
        exact=math.isfinite(rmse) and rmse < threshold,
        expr=equation_str,
        size=complexity,
    )


# ─── Driver ────────────────────────────────────────────────────────

def _mark(exact: bool) -> str:
    return "✓" if exact else " "


def _fmt_num(x: Optional[float], width: int = 8) -> str:
    if x is None:
        return "—".rjust(width)
    if not math.isfinite(x):
        return "—".rjust(width)
    return f"{x:.2e}".rjust(width)


def _fmt_int(x: Optional[int], width: int = 3) -> str:
    return "—".rjust(width) if x is None else str(x).rjust(width)


def run_benchmark(problems: list[FeynmanProblem], *,
                  max_depth: int, n_tries: int, threshold: float,
                  pysr_enabled: bool, deterministic: bool = False) -> dict:
    """Run all engines on all problems. Returns a nested result dict."""
    rows = []
    for prob in problems:
        print(f"\n▸ {prob.feynman_id}  ({prob.formula})", flush=True)

        ref_size = _reference_eml_size(prob)

        # eml-sr: both discover and curriculum, so readers see where each shines.
        r_disc = _run_eml_sr(prob, method="discover",
                             max_depth=max_depth, n_tries=n_tries,
                             threshold=threshold)
        print(f"    eml-sr/discover    : {r_disc.wall_time:5.1f}s  "
              f"rmse={r_disc.rmse:.2e}  size={r_disc.size}  "
              f"exact={r_disc.exact}", flush=True)

        r_curr = _run_eml_sr(prob, method="curriculum",
                             max_depth=max_depth, n_tries=n_tries,
                             threshold=threshold)
        print(f"    eml-sr/curriculum  : {r_curr.wall_time:5.1f}s  "
              f"rmse={r_curr.rmse:.2e}  size={r_curr.size}  "
              f"exact={r_curr.exact}", flush=True)

        pysr_results: list[EngineResult] = []
        if pysr_enabled:
            for cfg in ("default", "permissive"):
                r = _run_pysr(prob, config=cfg, threshold=threshold,
                              deterministic=deterministic)
                pysr_results.append(r)
                print(f"    pysr/{cfg:10s}   : {r.wall_time:5.1f}s  "
                      f"rmse={r.rmse:.2e}  size={r.size}  "
                      f"exact={r.exact}", flush=True)

        rows.append({
            "prob": prob,
            "ref_size": ref_size,
            "eml_discover": r_disc,
            "eml_curriculum": r_curr,
            "pysr": pysr_results,      # empty list when disabled
        })
    return {"rows": rows, "pysr_enabled": pysr_enabled, "threshold": threshold}


# ─── Markdown emission ─────────────────────────────────────────────

_SUMMARY_HEADER = """# eml-sr vs PySR — head-to-head

Per-target recovery comparison. See `pysr_compare.py` for methodology.
Regenerate with::

    PYSR_ENABLED=1 python -m benchmarks.pysr_compare --output benchmarks/pysr_compare.md

**Exact recovery** ≡ `RMSE < {threshold:.0e}` in the original coordinate space.
**Size** ≡ node count (both engines). **Ref EML size** is the compiler's canonical EML
tree for the target formula.
"""


def _positioning_statement() -> str:
    return """
## Positioning

Three axes the benchmark makes concrete:

1. **Grammar uniformity.** Every eml-sr node is the same `eml` operator. PySR
   expressions are heterogeneous ASTs over `{+, -, *, /, exp, log, sqrt}`.
   The "ref EML size" column shows what uniformity costs: the compiler's
   canonical EML tree for a given formula, independent of search.

2. **Exact recovery vs Pareto front.** eml-sr aims for machine-epsilon RMSE on
   a specific symbolic form. PySR trades complexity against RMSE and often
   lands near but not on the target — visible in the table as a PySR row with
   low RMSE but no ✓ and a small AST that doesn't algebraically match.

3. **Expression length.** EML trees for standard operations are long by
   design — multiplication is depth 8 in EML, depth 1 in a conventional AST.
   That's the cost of grammar uniformity, not a bug. The "size" vs
   "ref EML size" columns tell this story directly.

PySR is usually faster and finds more formulas on harder targets. eml-sr
recovers exact symbolic forms cleanly when the target is inside its reachable
vocabulary (the elementary basis), and the tree is always a pure `eml` circuit.
"""


def to_markdown(results: dict) -> str:
    rows = results["rows"]
    threshold = results["threshold"]
    pysr_on = results["pysr_enabled"]

    out = [_SUMMARY_HEADER.format(threshold=threshold)]

    # Table header
    if pysr_on:
        out.append(
            "| formula | eml-sr time | eml-sr rmse | eml-sr exact | eml-sr size | "
            "ref EML size | PySR time | PySR rmse | PySR exact | PySR size |\n"
            "|---|---:|---:|:---:|---:|---:|---:|---:|:---:|---:|"
        )
    else:
        out.append(
            "| formula | eml-sr time | eml-sr rmse | eml-sr exact | eml-sr size | "
            "ref EML size | PySR |\n"
            "|---|---:|---:|:---:|---:|---:|:---:|"
        )

    for row in rows:
        p = row["prob"]
        # Take the better of the two eml-sr runs per target (lower rmse wins).
        emls = min(
            (row["eml_discover"], row["eml_curriculum"]),
            key=lambda r: (not r.exact, r.rmse),
        )
        ref_size = _fmt_int(row["ref_size"])

        if pysr_on and row["pysr"]:
            pysr_best = min(row["pysr"], key=lambda r: (not r.exact, r.rmse))
            out.append(
                f"| `{p.formula}` | {emls.wall_time:.1f}s | {emls.rmse:.2e} | "
                f"{_mark(emls.exact)} | {_fmt_int(emls.size)} | {ref_size} | "
                f"{pysr_best.wall_time:.1f}s | {pysr_best.rmse:.2e} | "
                f"{_mark(pysr_best.exact)} | {_fmt_int(pysr_best.size)} |"
            )
        else:
            out.append(
                f"| `{p.formula}` | {emls.wall_time:.1f}s | {emls.rmse:.2e} | "
                f"{_mark(emls.exact)} | {_fmt_int(emls.size)} | {ref_size} | "
                "not run |"
            )

    # Aggregate
    n_total = len(rows)
    n_eml_exact = sum(
        1 for r in rows
        if min((r["eml_discover"], r["eml_curriculum"]),
               key=lambda x: (not x.exact, x.rmse)).exact
    )
    out.append("")
    out.append(f"**eml-sr exact-recovery:** {n_eml_exact}/{n_total}")
    if pysr_on:
        n_pysr_exact = sum(
            1 for r in rows
            if r["pysr"] and min(r["pysr"],
                                 key=lambda x: (not x.exact, x.rmse)).exact
        )
        out.append(f"**PySR exact-recovery:**   {n_pysr_exact}/{n_total}")
    else:
        out.append(
            "**PySR:** skipped (set `PYSR_ENABLED=1` and install `pysr` to run)."
        )

    out.append(_positioning_statement())
    return "\n".join(out) + "\n"


# ─── CLI ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Head-to-head benchmark: eml-sr vs PySR (issue #33)."
    )
    p.add_argument("--quick", action="store_true",
                   help="Run the small slice (default: full set).")
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--tries", type=int, default=16)
    p.add_argument("--threshold", type=float, default=1e-6,
                   help="Original-space RMSE threshold for exact recovery.")
    p.add_argument("--output", type=str,
                   default="benchmarks/pysr_compare.md",
                   help="Markdown output path (set to '-' for stdout).")
    p.add_argument("--deterministic", action="store_true",
                   help="Reproducible PySR runs (parallelism='serial', "
                        "random_state=0). ~10x slower; use when chasing a "
                        "specific regression. Default is fast, non-deterministic.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    pysr_on, reason = _pysr_available()
    if pysr_on:
        print("▸ PySR enabled.", flush=True)
    else:
        print(f"▸ PySR disabled ({reason}). eml-sr-only run.", flush=True)

    ids = QUICK_IDS if args.quick else FULL_IDS
    problems = _select_problems(ids)

    print(f"▸ Running {len(problems)} targets  "
          f"(max_depth={args.max_depth}, n_tries={args.tries}, "
          f"threshold={args.threshold:.0e})", flush=True)

    if args.deterministic and pysr_on:
        print("▸ PySR: deterministic mode (serial, ~10x slower).", flush=True)

    results = run_benchmark(
        problems,
        max_depth=args.max_depth,
        n_tries=args.tries,
        threshold=args.threshold,
        pysr_enabled=pysr_on,
        deterministic=args.deterministic,
    )

    md = to_markdown(results)
    if args.output == "-":
        sys.stdout.write(md)
    else:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(md)
        print(f"\n▸ Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
