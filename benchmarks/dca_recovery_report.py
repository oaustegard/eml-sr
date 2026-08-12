"""Aggregate benchmarks/results/dca_recovery/*.json into one report.

The runner writes one JSON per target and regenerates summary.md only
for the targets of its own invocation; sharded runs therefore need this
merge step. Emits a full markdown table (all targets x arms) plus the
headline numbers for issue #58's success gate.

Usage: python3 -m benchmarks.dca_recovery_report [--dir DIR] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = ["adam_softmax", "adam_linear", "dca_linear"]

# Presentation order: additive/affine targets first, nonlinear last.
TARGET_ORDER = [
    "push_halt", "push_pop", "stack_depth", "overwrite", "dup_add",
    "dup_add_chain_x4", "basic_add", "add_dup_add", "multi_add",
    "complex", "alternating", "many_pushes",
    "square_via_dupmul", "sum_of_squares", "native_multiply",
]


def _fmt_r2(v) -> str:
    if v is None or v != v:
        return "nan"
    return f"{v:.4f}"


def load(results_dir: Path) -> dict:
    out = {}
    for f in sorted(results_dir.glob("*.json")):
        d = json.loads(f.read_text())
        out[d["target"]["name"]] = d
    return out


def render(results: dict) -> str:
    lines = [
        "# DCA vs Adam on structural recovery (issue #58)",
        "",
        "Targets: the 15 STATUS_COLLAPSED polynomial rows of the",
        "llm-as-computer catalog (`dev/symbolic_collapse_report.md`,",
        "'Collapsed (branchless, polynomial-closed)' table).",
        "`native_multiply` is the canonical x0*x1 multiplicative failure",
        "case from PR #56 / issue #57.",
        "",
        "| target | expr | " + " | ".join(
            f"{a} rec / R2 / s" for a in ARMS) + " |",
        "|---|---|" + "---|" * len(ARMS),
    ]
    counts = {a: 0 for a in ARMS}
    n = 0
    for name in TARGET_ORDER:
        if name not in results:
            continue
        n += 1
        d = results[name]
        cells = []
        for a in ARMS:
            arm = d["arms"].get(a)
            if arm is None:
                cells.append("—")
                continue
            rec = "✓" if arm["structural_recovered"] else "✗"
            if arm["structural_recovered"]:
                counts[a] += 1
            cells.append(
                f"{rec} / {_fmt_r2(arm['held_r2'])} / "
                f"{arm['wall_clock_s']:.0f}")
        lines.append(
            f"| {name} | `{d['target']['expr']}` | " +
            " | ".join(cells) + " |")
    lines += [
        "|  **recovered** |  | " + " | ".join(
            f"**{counts[a]}/{n}**" for a in ARMS) + " |",
        "",
    ]
    nm = results.get("native_multiply")
    if nm:
        lines.append("## Success gate (issue #58)")
        lines.append("")
        for a in ARMS:
            arm = nm["arms"].get(a)
            if arm:
                lines.append(
                    f"- x0*x1 via {a}: "
                    f"{'recovered' if arm['structural_recovered'] else 'NOT recovered'}"
                    f" (R2 {_fmt_r2(arm['held_r2'])}, "
                    f"expr `{(arm.get('expr') or '')[:120]}`)")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="benchmarks/results/dca_recovery")
    ap.add_argument("--out", default="benchmarks/results/dca_recovery/summary_all.md")
    args = ap.parse_args()
    results = load(Path(args.dir))
    md = render(results)
    Path(args.out).write_text(md)
    print(md)
    print(f"\n[{len(results)} targets aggregated -> {args.out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
