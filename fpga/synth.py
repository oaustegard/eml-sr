"""Synthesize and place-and-route each design; write results/synth.json.

Toolchain: yowasp-yosys + yowasp-nextpnr-{ecp5,ice40} (pip, WebAssembly
builds of Yosys and nextpnr).  Targets:
  ecp5  : Lattice LFE5U-85F, CABGA381 (18x18 DSPs, 18 kbit EBR)
  ice40 : Lattice iCE40-HX8K, CT256  (no DSPs: multipliers become LUTs)

    pip install yowasp-yosys yowasp-nextpnr-ecp5 yowasp-nextpnr-ice40
    python fpga/synth.py [design ...] [--target ecp5|ice40]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
RTL = HERE / "rtl"
OUT = HERE / "results"
BUILD = HERE / "build"

DESIGNS = {
    "q16_add": ["baselines.v"],
    "q16_mul": ["baselines.v"],
    "q16_mac": ["baselines.v"],
    "eml_cell": ["eml_cell.v"],
    "eml_tree_sub": ["eml_cell.v", "eml_tree_sub.v"],
    "eml_tree_mul": ["eml_cell.v", "eml_tree_mul.v"],
    "eml_tree_add": ["eml_cell.v", "eml_tree_add.v"],
}

TARGETS = {
    "ecp5": {
        "synth": "synth_ecp5",
        "pnr": ["yowasp-nextpnr-ecp5", "--85k", "--package", "CABGA381"],
    },
    "ice40": {
        "synth": "synth_ice40",
        "pnr": ["yowasp-nextpnr-ice40", "--hx8k", "--package", "ct256"],
    },
}


def run(cmd: list[str], log: Path, cwd: Path) -> int:
    env = dict(os.environ, TMPDIR=str(BUILD / "tmp"))
    with log.open("w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=cwd, env=env, check=False).returncode


def parse_stat(text: str) -> dict:
    cells = {}
    block = text.split("Printing statistics")[-1]
    for m in re.finditer(r"^\s+(\d+)\s+([A-Za-z_][\w$]*)\s*$", block, re.MULTILINE):
        n, name = int(m.group(1)), m.group(2)
        if name not in ("wires", "cells", "ports"):
            cells[name] = cells.get(name, 0) + n
    return cells


def one(design: str, target: str, seed: int = 1) -> dict:
    t = TARGETS[target]
    BUILD.mkdir(exist_ok=True)
    (BUILD / "tmp").mkdir(exist_ok=True)
    stem = f"{design}_{target}"
    js = BUILD / f"{stem}.json"
    stat = BUILD / f"{stem}_stat.txt"
    files = " ".join(DESIGNS[design])
    ys = (f"read_verilog {files}; {t['synth']} -top {design} -json {js}; "
          f"tee -q -o {stat} stat")
    rc = run(["yowasp-yosys", "-p", ys], BUILD / f"{stem}_yosys.log", RTL)
    res = {"design": design, "target": target, "yosys_rc": rc}
    if rc != 0:
        return res
    res["cells"] = parse_stat(stat.read_text())
    rep = BUILD / f"{stem}_report.json"
    rc = run(t["pnr"] + ["--json", str(js), "--report", str(rep),
                         "--freq", "100", "--timing-allow-fail",
                         "--seed", str(seed)],
             BUILD / f"{stem}_pnr.log", RTL)
    res["pnr_rc"] = rc
    if rc == 0 and rep.exists():
        r = json.loads(rep.read_text())
        fm = r.get("fmax", {})
        res["fmax_mhz"] = min((v["achieved"] for v in fm.values()), default=None)
        res["utilization"] = {k: v["used"] for k, v in r.get("utilization", {}).items()
                              if v.get("used")}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("designs", nargs="*", default=list(DESIGNS))
    ap.add_argument("--target", default="ecp5", choices=list(TARGETS))
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    path = OUT / "synth.json"
    allres = json.loads(path.read_text()) if path.exists() else {}
    for d in a.designs:
        res = one(d, a.target)
        allres[f"{d}/{a.target}"] = res
        path.write_text(json.dumps(allres, indent=1, sort_keys=True))
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
