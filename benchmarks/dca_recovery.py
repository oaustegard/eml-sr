"""DCA vs Adam on structural recovery — eml-sr issue #58.

Compares three search strategies for fitting an ``EMLTree1DLinear``
(Option B, unconstrained affine leaves + affine gates, `eml_sr_linear.py`)
against the 15 ``STATUS_COLLAPSED`` polynomial rows of the llm-as-computer
symbolic-collapse catalog:

  source: oaustegard/llm-as-computer, dev/symbolic_collapse_report.md,
  the "Collapsed (branchless, polynomial-closed)" table.

Arms
----

* ``adam_softmax`` — the repo's current default path, `eml_sr.discover()`
  (Option A: softmax-gated trees).
* ``adam_linear`` — Option B baseline. A depth ladder of
  `eml_sr_linear._train_one_linear` (free-real-coefficient Adam search,
  including its own built-in discreteness-ramp phase 2) followed by
  `eml_sr_linear.iterative_snap` on the best tree per depth.
* ``dca_linear`` — treatment. Same ladder/seeds/snap pipeline as
  ``adam_linear``, but phase-1 search is `eml_dca.dca_train` (block-cyclic
  DCA/majorization) instead of free-fall Adam. Because `dca_train` has no
  discreteness-ramp phase of its own, this arm replays the *same* ramp
  code (`_discreteness_ramp` below, mirroring `_train_one_linear`'s phase
  2 byte-for-byte) before `iterative_snap`, so the search optimizer is the
  only treatment difference between ``adam_linear`` and ``dca_linear``.

Protocol
--------

Per target: 256 training samples ``X ~ U(0.5, 2.5)^n_vars``, 512 held-out
samples ``X ~ U(0.25, 4.0)^n_vars`` (an extrapolation band relative to
training). ``structural_recovered`` is true iff the max abs error of the
*snapped* tree's forward pass on the held-out set is below
``1e-6 * max(1, max|y_held|)``. Data draws are seeded per-target
(sha256 hash of the target name); per-seed training uses seeds
``0..n_tries-1`` as in `discover()` / `discover_linear()`.

Budget fairness for the dca arm: ``outer_iters * inner_iters`` is chosen
to roughly match Adam's default ``search_iters`` (2000) — default
``--dca-outer-iters 40`` × ``--dca-inner-iters 50`` = 2000.

Usage::

    python -m benchmarks.dca_recovery                  # full run
    python -m benchmarks.dca_recovery --quick           # ~few minutes

Output: one JSON per target in ``--out-dir`` (default
``benchmarks/results/dca_recovery``), plus an aggregate ``summary.md``
with a markdown table (target × arm) and a totals row. The table is also
printed to stdout.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

# Make sibling modules importable both as `python -m benchmarks.dca_recovery`
# and as `python benchmarks/dca_recovery.py`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eml_dca import dca_train  # noqa: E402
from eml_sr import DTYPE, REAL, discover
from eml_sr_linear import (
    EMLTree1DLinear,
    _discreteness_penalty,
    _train_one_linear,
    iterative_snap,
)

# ─── Protocol constants ─────────────────────────────────────────────

TRAIN_N = 256
TRAIN_DOMAIN = (0.5, 2.5)
HELD_N = 512
HELD_DOMAIN = (0.25, 4.0)
SUCCESS_THRESHOLD = 1e-10          # snap-MSE ladder stop criterion
STRUCTURAL_REL_TOL = 1e-6          # held-out max-abs-err tolerance factor

DEFAULT_MAX_DEPTH = 4
DEFAULT_N_TRIES = 6
QUICK_MAX_DEPTH = 2
QUICK_N_TRIES = 2
QUICK_TARGETS = ["push_halt", "dup_add", "basic_add", "native_multiply"]

# eml_sr_linear defaults (mirrored, not imported, since they're function
# default args rather than module constants).
RAMP_SNAP_ITERS = 1500
RAMP_LR = 0.01
RAMP_LAM_DISC_MAX = 0.5
RETRAIN_ITERS = 300
RETRAIN_LR = 0.005

DCA_OUTER_ITERS_DEFAULT = 40
DCA_INNER_ITERS_DEFAULT = 50   # 40*50 = 2000 ≈ Adam's default search_iters


# ─── Targets: the 15 STATUS_COLLAPSED polynomial rows ───────────────
#
# source: oaustegard/llm-as-computer, dev/symbolic_collapse_report.md,
# "Collapsed (branchless, polynomial-closed)" table.

@dataclass
class Target:
    name: str
    n_vars: int
    fn: Callable[[np.ndarray], np.ndarray]   # X (n, n_vars) -> y (n,)
    expr: str
    canonical: bool = False   # PR-#56 / issue-#57 canonical mul failure case


def _sum_cols(idxs):
    return lambda X: sum(X[:, i] for i in idxs)


TARGETS: list[Target] = [
    Target("basic_add", 2, lambda X: X[:, 0] + X[:, 1], "x0 + x1"),
    Target("push_halt", 1, lambda X: X[:, 0], "x0"),
    Target("push_pop", 1, lambda X: X[:, 0], "x0"),
    Target("dup_add", 1, lambda X: 2.0 * X[:, 0], "2*x0"),
    Target("multi_add", 3, _sum_cols([0, 1, 2]), "x0 + x1 + x2"),
    Target("stack_depth", 1, lambda X: X[:, 0], "x0"),
    Target("overwrite", 3, lambda X: X[:, 2], "x2"),
    Target("complex", 3, lambda X: 2.0 * X[:, 1] + 2.0 * X[:, 2], "2*x1 + 2*x2"),
    Target("many_pushes", 10, _sum_cols(range(10)),
           "x0+x1+...+x9"),
    Target("alternating", 6, _sum_cols([0, 1, 3, 5]), "x0 + x1 + x3 + x5"),
    Target("native_multiply", 2, lambda X: X[:, 0] * X[:, 1], "x0*x1",
           canonical=True),
    Target("square_via_dupmul", 1, lambda X: X[:, 0] ** 2, "x0**2"),
    Target("sum_of_squares", 4, lambda X: X[:, 0] ** 2 + X[:, 3] ** 2,
           "x0**2 + x3**2"),
    Target("dup_add_chain_x4", 1, lambda X: 16.0 * X[:, 0], "16*x0"),
    Target("add_dup_add", 2, lambda X: 2.0 * X[:, 0] + 2.0 * X[:, 1],
           "2*x0 + 2*x1"),
]
TARGETS_BY_NAME = {t.name: t for t in TARGETS}

ARM_NAMES = ["adam_softmax", "adam_linear", "dca_linear"]


# ─── Data ────────────────────────────────────────────────────────────

def _target_seed(name: str) -> int:
    """Stable seed derived from the target name (str hash is randomized
    per-process by default; sha256 is not)."""
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % (2 ** 31 - 1)


def make_data(target: Target):
    """(X_train, y_train, X_held, y_held), deterministic per target name."""
    rng = np.random.default_rng(_target_seed(target.name))
    lo, hi = TRAIN_DOMAIN
    X_train = rng.uniform(lo, hi, size=(TRAIN_N, target.n_vars))
    y_train = target.fn(X_train).astype(np.float64)
    lo_h, hi_h = HELD_DOMAIN
    X_held = rng.uniform(lo_h, hi_h, size=(HELD_N, target.n_vars))
    y_held = target.fn(X_held).astype(np.float64)
    return X_train, y_train, X_held, y_held


# ─── Evaluation helpers ─────────────────────────────────────────────

def _eval_forward(tree, X: np.ndarray) -> np.ndarray:
    """Real-valued forward pass, complex-safe (.real of predictions),
    NaN-guarded (non-finite -> NaN, never raises)."""
    x_t = torch.tensor(X, dtype=REAL)
    with torch.no_grad():
        pred, _, _ = tree(x_t)
    pred_np = pred.numpy()
    real = np.real(pred_np)
    real = np.where(np.isfinite(real), real, np.nan)
    return real


def _structural_check(pred_held: np.ndarray, y_held: np.ndarray) -> tuple:
    """(recovered: bool, max_abs_err: float)."""
    if np.any(np.isnan(pred_held)):
        return False, float("nan")
    err = float(np.max(np.abs(pred_held - y_held)))
    thresh = STRUCTURAL_REL_TOL * max(1.0, float(np.max(np.abs(y_held))))
    return err < thresh, err


def _r2(pred: np.ndarray, y: np.ndarray) -> float:
    if np.any(np.isnan(pred)):
        return float("nan")
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot < 1e-30:
        return 1.0 if ss_res < 1e-20 else float("nan")
    return 1.0 - ss_res / ss_tot


# ─── Result record ───────────────────────────────────────────────────

@dataclass
class ArmResult:
    target: str
    arm: str
    structural_recovered: bool
    train_snap_mse: float
    held_r2: float
    wall_clock_s: float
    expr: str
    depth_found: int | None
    n_seeds_run: int
    total_outer_iters: int = 0     # meaningful only for dca_linear
    max_abs_err_held: float = float("nan")
    error: str | None = None


def _to_json_safe(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def arm_result_to_dict(r: ArmResult) -> dict:
    d = asdict(r)
    d["expr"] = d["expr"][:200]
    for k in ("train_snap_mse", "held_r2", "wall_clock_s", "max_abs_err_held"):
        d[k] = _to_json_safe(d[k])
    return d


# ─── Shared post-search snap pipeline (adam_linear / dca_linear) ────
#
# _train_one_linear already runs a discreteness-ramp phase 2 internally
# (search_iters MSE-only, then snap_iters of MSE + ramped discreteness
# penalty) before its own (non-iterative) tree.snap(). For adam_linear we
# reuse that tree — the ramp already happened — and apply iterative_snap
# ourselves instead of the naive one-shot snap.
#
# dca_train has no such phase (pure DC/MM outer loop, MSE objective only),
# so for dca_linear we replay the identical ramp here before iterative_snap
# — same code, same default hyperparameters as _train_one_linear's phase 2
# — so the *only* difference between the two arms is the phase-1 search.

def _discreteness_ramp(
    tree: EMLTree1DLinear,
    x_data: torch.Tensor,
    targets: torch.Tensor,
    snap_iters: int = RAMP_SNAP_ITERS,
    lr: float = RAMP_LR,
    lam_disc_max: float = RAMP_LAM_DISC_MAX,
) -> EMLTree1DLinear:
    """Replay of `_train_one_linear`'s phase-2 discreteness ramp, as a
    standalone post-search step. Operates on (and returns) a deep copy;
    keeps the best-MSE state seen during the ramp, mirroring the parent
    function's own best-state tracking."""
    tree = copy.deepcopy(tree)
    opt = torch.optim.Adam(tree.parameters(), lr=lr)
    best_loss = float("inf")
    best_state = None

    for it in range(1, snap_iters + 1):
        opt.zero_grad()
        pred, _, _ = tree(x_data)
        mse = torch.mean((pred - targets).abs() ** 2).real

        t = it / max(1, snap_iters)
        lam = lam_disc_max * (t ** 2)
        disc = (_discreteness_penalty(tree.leaf_logits)
                + _discreteness_penalty(tree.gate_logits))
        loss = mse + lam * disc

        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tree.parameters(), 1.0)
        opt.step()

        val = float(mse.item())
        if math.isfinite(val) and val < best_loss:
            best_loss = val
            best_state = {k: v.clone() for k, v in tree.state_dict().items()}

    if best_state is not None:
        tree.load_state_dict(best_state)
    return tree


def _post_search_snap(
    tree: EMLTree1DLinear,
    x_t: torch.Tensor,
    y_t: torch.Tensor,
    ramp: bool,
) -> EMLTree1DLinear:
    """Shared post-search snap pipeline. `ramp=True` replays the
    discreteness-ramp phase first (dca_linear, whose phase-1 search has
    no such phase built in); `ramp=False` skips it (adam_linear, where
    `_train_one_linear` already ran it as part of the search call)."""
    t = tree
    if ramp:
        t = _discreteness_ramp(t, x_t, y_t)
    return iterative_snap(t, x_t, y_t, retrain_iters=RETRAIN_ITERS, lr=RETRAIN_LR)


# ─── Arm: adam_softmax (current default, eml_sr.discover) ──────────

def run_adam_softmax(target: Target, X_train, y_train, X_held, y_held,
                     max_depth: int, n_tries: int, workers: int) -> ArmResult:
    t0 = time.time()
    try:
        result = discover(X_train, y_train, max_depth=max_depth,
                          n_tries=n_tries, verbose=False,
                          success_threshold=SUCCESS_THRESHOLD,
                          n_workers=workers)
        wall = time.time() - t0
        tree = result["snapped_tree"]
        pred_held = _eval_forward(tree, X_held)
        recovered, max_err = _structural_check(pred_held, y_held)
        r2 = _r2(pred_held, y_held)
        train_mse = float(result["snap_rmse"]) ** 2
        return ArmResult(
            target=target.name, arm="adam_softmax",
            structural_recovered=recovered, train_snap_mse=train_mse,
            held_r2=r2, wall_clock_s=wall, expr=str(result["expr"])[:200],
            depth_found=int(result["depth"]), n_seeds_run=n_tries,
            max_abs_err_held=max_err,
        )
    except Exception as ex:  # noqa: BLE001 - keep the benchmark alive
        return ArmResult(
            target=target.name, arm="adam_softmax",
            structural_recovered=False, train_snap_mse=float("nan"),
            held_r2=float("nan"), wall_clock_s=time.time() - t0,
            expr=f"<error: {ex}>", depth_found=None, n_seeds_run=n_tries,
            error=str(ex),
        )


# ─── Arm: adam_linear / dca_linear (shared ladder driver) ──────────

def run_linear_ladder(target: Target, X_train, y_train, X_held, y_held,
                      max_depth: int, n_tries: int, arm: str,
                      dca_outer_iters: int, dca_inner_iters: int) -> ArmResult:
    assert arm in ("adam_linear", "dca_linear")
    x_t = torch.tensor(X_train, dtype=REAL)
    y_t = torch.tensor(y_train, dtype=DTYPE)   # matches discover_linear's convention

    t0 = time.time()
    n_seeds_run = 0
    total_outer = 0
    best_overall = None   # dict: depth, tree(snapped), expr, snap_mse

    try:
        for depth in range(max_depth + 1):
            best_at_depth = None
            for seed in range(n_tries):
                n_seeds_run += 1
                if arm == "adam_linear":
                    # _train_one_linear runs a naive tree.snap() internally;
                    # a NaN-parameter tree makes that raise (round(nan)) --
                    # treat the seed as failed rather than aborting the arm.
                    try:
                        raw = _train_one_linear(x_t, y_t, depth, seed,
                                                n_vars=target.n_vars)
                    except (ValueError, OverflowError):
                        continue
                    pre_snap_tree = raw["tree"]
                else:
                    dca_res = dca_train(x_t, y_t, depth, seed,
                                        n_vars=target.n_vars,
                                        outer_iters=dca_outer_iters,
                                        inner_iters=dca_inner_iters)
                    pre_snap_tree = dca_res["tree"]
                    total_outer += int(dca_res["outer_iters_used"])

                # A NaN-parameter tree (deep-init blowup) poisons
                # iterative_snap and to_expr (round(nan) raises); skip
                # the seed instead of aborting the arm.
                with torch.no_grad():
                    pred_raw, _, _ = pre_snap_tree(x_t)
                    raw_mse = float(torch.mean(
                        (pred_raw - y_t).abs() ** 2).real.item())
                if not math.isfinite(raw_mse):
                    continue
                try:
                    snapped = _post_search_snap(pre_snap_tree, x_t, y_t,
                                                ramp=(arm == "dca_linear"))
                    with torch.no_grad():
                        pred_t, _, _ = snapped(x_t)
                        snap_mse = float(torch.mean(
                            (pred_t - y_t).abs() ** 2).real.item())
                    if not math.isfinite(snap_mse):
                        continue
                    cand = {"depth": depth, "tree": snapped,
                            "expr": snapped.to_expr(), "snap_mse": snap_mse}
                except (ValueError, OverflowError):
                    continue
                if best_at_depth is None or snap_mse < best_at_depth["snap_mse"]:
                    best_at_depth = cand
                if snap_mse < SUCCESS_THRESHOLD:
                    break   # this depth is good enough, stop seeding it

            if best_at_depth is not None and (
                best_overall is None
                or best_at_depth["snap_mse"] < best_overall["snap_mse"]
            ):
                best_overall = best_at_depth

            if best_at_depth is not None and best_at_depth["snap_mse"] < SUCCESS_THRESHOLD:
                break   # ladder stops at the first depth that succeeds

        wall = time.time() - t0

        if best_overall is None:
            return ArmResult(
                target=target.name, arm=arm, structural_recovered=False,
                train_snap_mse=float("nan"), held_r2=float("nan"),
                wall_clock_s=wall, expr="<no finite candidate>",
                depth_found=None, n_seeds_run=n_seeds_run,
                total_outer_iters=total_outer,
            )

        pred_held = _eval_forward(best_overall["tree"], X_held)
        recovered, max_err = _structural_check(pred_held, y_held)
        r2 = _r2(pred_held, y_held)
        return ArmResult(
            target=target.name, arm=arm,
            structural_recovered=recovered,
            train_snap_mse=best_overall["snap_mse"], held_r2=r2,
            wall_clock_s=wall, expr=best_overall["expr"][:200],
            depth_found=best_overall["depth"], n_seeds_run=n_seeds_run,
            total_outer_iters=total_outer, max_abs_err_held=max_err,
        )
    except Exception as ex:  # noqa: BLE001 - keep the benchmark alive
        return ArmResult(
            target=target.name, arm=arm, structural_recovered=False,
            train_snap_mse=float("nan"), held_r2=float("nan"),
            wall_clock_s=time.time() - t0, expr=f"<error: {ex}>",
            depth_found=None, n_seeds_run=n_seeds_run,
            total_outer_iters=total_outer, error=str(ex),
        )


# ─── Orchestration ───────────────────────────────────────────────────

def run_target(target: Target, arms: list, max_depth: int, n_tries: int,
               workers: int, dca_outer_iters: int, dca_inner_iters: int,
               verbose: bool = True) -> dict:
    X_train, y_train, X_held, y_held = make_data(target)
    results = {}
    for arm in arms:
        if verbose:
            print(f"  [{target.name}] running {arm} ...", flush=True)
        if arm == "adam_softmax":
            r = run_adam_softmax(target, X_train, y_train, X_held, y_held,
                                 max_depth, n_tries, workers)
        elif arm in ("adam_linear", "dca_linear"):
            r = run_linear_ladder(target, X_train, y_train, X_held, y_held,
                                  max_depth, n_tries, arm,
                                  dca_outer_iters, dca_inner_iters)
        else:
            raise ValueError(f"unknown arm {arm!r}")
        results[arm] = r
        if verbose:
            tag = "recovered" if r.structural_recovered else "not recovered"
            print(f"    -> {tag}  train_mse={r.train_snap_mse:.3e}  "
                  f"R2={r.held_r2:.4f}  {r.wall_clock_s:.1f}s  "
                  f"expr={r.expr[:60]}", flush=True)
    return results


def write_target_json(out_dir: Path, target: Target, results: dict,
                      protocol: dict) -> Path:
    out = {
        "target": {
            "name": target.name, "n_vars": target.n_vars,
            "expr": target.expr, "canonical": target.canonical,
        },
        "protocol": protocol,
        "arms": {arm: arm_result_to_dict(r) for arm, r in results.items()},
    }
    path = out_dir / f"{target.name}.json"
    path.write_text(json.dumps(out, indent=2))
    return path


def render_summary_md(all_results: dict, targets: list, arms: list,
                      protocol: dict, timestamp: str) -> str:
    """all_results: {target_name: {arm_name: ArmResult}}."""
    parts = []
    parts.append("# DCA vs Adam on structural recovery\n")
    parts.append(f"_Generated {timestamp}_\n")
    parts.append("Implements [eml-sr issue #58](https://github.com/oaustegard/eml-sr/issues/58).\n")
    parts.append("## Protocol\n")
    parts.append(
        f"- Targets: {len(targets)} STATUS_COLLAPSED polynomial rows from the "
        f"llm-as-computer symbolic-collapse catalog "
        f"(`dev/symbolic_collapse_report.md`, "
        f"\"Collapsed (branchless, polynomial-closed)\" table)\n"
        f"- Arms: {', '.join(arms)}\n"
        f"- Train: {TRAIN_N} samples, X ~ U{TRAIN_DOMAIN}^n_vars\n"
        f"- Held-out (extrapolation): {HELD_N} samples, X ~ U{HELD_DOMAIN}^n_vars\n"
        f"- structural_recovered: max abs err of the snapped tree on "
        f"held-out < {STRUCTURAL_REL_TOL} * max(1, max|y_held|)\n"
        f"- Ladder success threshold (train snap MSE): {SUCCESS_THRESHOLD}\n"
        f"- max_depth={protocol['max_depth']}, n_tries(seeds)={protocol['n_tries']}\n"
        f"- dca_linear budget: outer_iters={protocol['dca_outer_iters']} × "
        f"inner_iters={protocol['dca_inner_iters']} "
        f"({protocol['dca_outer_iters'] * protocol['dca_inner_iters']} total, "
        f"≈ Adam's default search_iters=2000)\n"
        f"- adam_linear / dca_linear ladders run **serial** per seed "
        f"(no multiprocessing fan-out); only adam_softmax honors "
        f"`--workers` (passed through to `eml_sr.discover`)\n"
    )

    parts.append("\n## Results\n")
    header = "| target | " + " | ".join(
        f"{a} (rec / R² / s)" for a in arms) + " |"
    sep = "|---" * (len(arms) + 1) + "|"
    parts.append(header)
    parts.append(sep)

    totals = {a: {"recovered": 0, "n": 0, "wall": 0.0} for a in arms}
    for t in targets:
        row = [t.name]
        res = all_results.get(t.name, {})
        for a in arms:
            r = res.get(a)
            if r is None:
                row.append("—")
                continue
            totals[a]["n"] += 1
            totals[a]["wall"] += r.wall_clock_s
            mark = "✓" if r.structural_recovered else "✗"
            if r.structural_recovered:
                totals[a]["recovered"] += 1
            r2s = "nan" if not math.isfinite(r.held_r2) else f"{r.held_r2:.3f}"
            row.append(f"{mark} / {r2s} / {r.wall_clock_s:.1f}")
        parts.append("| " + " | ".join(row) + " |")

    totals_row = ["**totals**"]
    for a in arms:
        n = totals[a]["n"]
        rec = totals[a]["recovered"]
        wall = totals[a]["wall"]
        totals_row.append(f"{rec}/{n} / — / {wall:.1f}")
    parts.append("| " + " | ".join(totals_row) + " |")

    parts.append("\n## Per-target expressions\n")
    parts.append("| target | arm | depth | expr |")
    parts.append("|---|---|---|---|")
    for t in targets:
        res = all_results.get(t.name, {})
        for a in arms:
            r = res.get(a)
            if r is None:
                continue
            expr = r.expr.replace("|", "\\|")
            parts.append(f"| {t.name} | {a} | {r.depth_found} | `{expr}` |")

    return "\n".join(parts)


def print_summary_table(all_results: dict, targets: list, arms: list) -> None:
    print("\n" + "=" * 78)
    header = f"{'target':22s} " + " ".join(f"{a:26s}" for a in arms)
    print(header)
    print("-" * len(header))
    for t in targets:
        res = all_results.get(t.name, {})
        cells = []
        for a in arms:
            r = res.get(a)
            if r is None:
                cells.append(f"{'—':26s}")
                continue
            mark = "REC" if r.structural_recovered else "no "
            r2s = "nan" if not math.isfinite(r.held_r2) else f"{r.held_r2:6.3f}"
            cells.append(f"{mark} R2={r2s} {r.wall_clock_s:5.1f}s".ljust(26))
        print(f"{t.name:22s} " + " ".join(cells))
    print("=" * 78)


# ─── CLI ───────────────────────────────────────────────────────────

def main(argv: list | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", default="all",
                   help="Comma-separated target names, or 'all' (default)")
    p.add_argument("--arms", default=",".join(ARM_NAMES),
                   help=f"Comma-separated arms (default: {','.join(ARM_NAMES)})")
    p.add_argument("--seeds", type=int, default=DEFAULT_N_TRIES,
                   help=f"Seeds per depth per arm (default {DEFAULT_N_TRIES})")
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
                   help=f"Max tree depth to ladder through (default {DEFAULT_MAX_DEPTH})")
    p.add_argument("--workers", type=int, default=1,
                   help="n_workers passed to eml_sr.discover (adam_softmax only)")
    p.add_argument("--out-dir", default="benchmarks/results/dca_recovery",
                   help="Output directory (default benchmarks/results/dca_recovery)")
    p.add_argument("--dca-outer-iters", type=int, default=DCA_OUTER_ITERS_DEFAULT,
                   help=f"dca_train outer_iters (default {DCA_OUTER_ITERS_DEFAULT})")
    p.add_argument("--dca-inner-iters", type=int, default=DCA_INNER_ITERS_DEFAULT,
                   help=f"dca_train inner_iters (default {DCA_INNER_ITERS_DEFAULT})")
    p.add_argument("--quick", action="store_true",
                   help=f"Quick mode: max_depth={QUICK_MAX_DEPTH}, "
                        f"seeds={QUICK_N_TRIES}, targets={QUICK_TARGETS}")
    args = p.parse_args(argv)

    if args.quick:
        max_depth = QUICK_MAX_DEPTH
        n_tries = QUICK_N_TRIES
        target_names = QUICK_TARGETS
    else:
        max_depth = args.max_depth
        n_tries = args.seeds
        target_names = (list(TARGETS_BY_NAME) if args.targets == "all"
                        else [s.strip() for s in args.targets.split(",")])

    unknown = [n for n in target_names if n not in TARGETS_BY_NAME]
    if unknown:
        print(f"unknown target(s): {unknown}. Choices: {list(TARGETS_BY_NAME)}")
        return 2
    targets = [TARGETS_BY_NAME[n] for n in target_names]

    arms = [a.strip() for a in args.arms.split(",")]
    unknown_arms = [a for a in arms if a not in ARM_NAMES]
    if unknown_arms:
        print(f"unknown arm(s): {unknown_arms}. Choices: {ARM_NAMES}")
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"# dca_recovery: {len(targets)} targets × {len(arms)} arms  "
          f"(max_depth={max_depth}, seeds={n_tries}, "
          f"dca={args.dca_outer_iters}x{args.dca_inner_iters})")

    protocol = {
        "train_n": TRAIN_N, "train_domain": list(TRAIN_DOMAIN),
        "held_n": HELD_N, "held_domain": list(HELD_DOMAIN),
        "structural_rel_tol": STRUCTURAL_REL_TOL,
        "success_threshold": SUCCESS_THRESHOLD,
        "max_depth": max_depth, "n_tries": n_tries,
        "dca_outer_iters": args.dca_outer_iters,
        "dca_inner_iters": args.dca_inner_iters,
        "workers": args.workers,
    }

    all_results: dict = {}
    for target in targets:
        print(f"\n─── {target.name} (n_vars={target.n_vars}, "
              f"expr={target.expr}) ───")
        res = run_target(target, arms, max_depth, n_tries, args.workers,
                         args.dca_outer_iters, args.dca_inner_iters)
        all_results[target.name] = res
        path = write_target_json(out_dir, target, res, protocol)
        print(f"  wrote {path}")

    timestamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    summary_text = render_summary_md(all_results, targets, arms, protocol,
                                     timestamp)
    summary_path = out_dir / "summary.md"
    summary_path.write_text(summary_text)
    print(f"\nwrote {summary_path}")

    print_summary_table(all_results, targets, arms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
