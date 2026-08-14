"""A/B test: remex-quantized join keys vs exact float64 keys (issue #62).

The meet-in-the-middle join's memory wall is the side cache (40M+
entries at depth-4 sides). A float32 key drive-by failed correctness
(last-write-wins dedupe evicted true entries: 37 -> 0 on
sum_of_squares), leaving a design requirement for lossy keys:
multi-occupancy buckets, with the exact confirmation stage absorbing
the extra collisions.

remex (the house multi-bit quantizer: unit-norm + Haar rotation +
Lloyd-Max scalar codebook; data-oblivious, deterministic for fixed
(d, bits, seed)) supplies the codes. Key = packed Lloyd-Max indices
+ the norm rounded to 6 decimals. Identical float64 vectors encode
identically, so every exact-key discovery must survive; coarser cells
only add collisions. Measured per key mode: discoveries (ground truth
37 on sum_of_squares at side-depth 3), hash hits, bucket stats, and
bytes spent on keys.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.skeleton_branch import (
    TARGETS,
    _confirm,
    build_side_cache,
)
from benchmarks.skeleton_exact import (
    ALPHAS,
    GAMMAS,
    HELD_DOMAIN,
    HELD_N,
    SCREEN_N,
    TRAIN_DOMAIN,
    TRAIN_N,
)


class ScalarKeyer:
    """Uniform scalar quantization of arcsinh(V) at fixed resolution --
    no unit-norm factorization, so constant-direction families do not
    collapse. int16 codes: 32 bytes/key."""

    def __init__(self, resolution: float = 0.01):
        self.res = resolution

    def keys(self, V: np.ndarray) -> list:
        Q = np.clip(np.rint(np.arcsinh(V) / self.res), -32000, 32000
                    ).astype(np.int16)
        return [Q[i].tobytes() for i in range(V.shape[0])]


def keys_exact(V: np.ndarray) -> list:
    Vr = np.round(V, 8)
    return [Vr[i].tobytes() for i in range(V.shape[0])]


class RemexKeyer:
    def __init__(self, d: int, bits: int):
        from remex import Quantizer, pack
        self.q = Quantizer(d=d, bits=bits, seed=42)
        self.pack = pack
        self.bits = bits

    def keys(self, V: np.ndarray) -> list:
        comp = self.q.encode(np.ascontiguousarray(V))
        packed = self.pack(comp.indices, self.bits).reshape(V.shape[0], -1)
        norms = np.round(np.asarray(comp.norms, dtype=np.float64), 6)
        return [packed[i].tobytes() + norms[i].tobytes()
                for i in range(V.shape[0])]


def run_join(entries, values, keyer, y_scr, X_train, y_train,
             X_held, y_held, label: str):
    t0 = time.time()
    buckets = defaultdict(list)
    for i, k in enumerate(keyer(values)):
        buckets[k].append(i)
    key_bytes = sum(len(k) for k in buckets)
    bsizes = np.array([len(v) for v in buckets.values()])

    hits = tested = 0
    discoveries = []
    CHUNK = 65536
    for start in range(0, values.shape[0], CHUNK):
        U = values[start:start + CHUNK]
        for a_u in ALPHAS:
            for g_u in GAMMAS:
                with np.errstate(all="ignore"):
                    P = np.exp(a_u + g_u * U)
                    v_in_req = np.exp(P - y_scr[None, :])
                for a_v in ALPHAS:
                    for g_v in GAMMAS:
                        tested += U.shape[0]
                        with np.errstate(all="ignore"):
                            V_req = (v_in_req - a_v) / g_v
                        ok = np.isfinite(V_req).all(axis=1)
                        idx = np.nonzero(ok)[0]
                        if idx.size == 0:
                            continue
                        for j, k in zip(idx, keyer(V_req[idx])):
                            for vi in buckets.get(k, ()):
                                hits += 1
                                # cheap 16-sample pre-screen: both value
                                # vectors are in RAM; kill false hits
                                # before any full chain rebuild
                                with np.errstate(all="ignore"):
                                    y_s = (np.exp(a_u + g_u * values[start + j])
                                           - np.log(a_v + g_v * values[vi]))
                                    d_s = y_s - y_scr
                                if not (np.all(np.isfinite(d_s))
                                        and float(np.mean(d_s ** 2)) < 1e-9):
                                    continue
                                disc = _confirm(
                                    entries[start + j], entries[vi],
                                    a_u, g_u, a_v, g_v,
                                    X_train, y_train, X_held, y_held)
                                if disc is not None:
                                    discoveries.append(disc)
    return dict(label=label, discoveries=len(discoveries),
                hash_hits=hits, joins_tested=tested,
                key_bytes=int(key_bytes), n_buckets=len(buckets),
                max_bucket=int(bsizes.max()),
                wall_s=round(time.time() - t0, 1),
                sample_forms=sorted({d["expr"] for d in discoveries})[:3])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="sum_of_squares",
                    choices=sorted(TARGETS))
    ap.add_argument("--max-side-depth", type=int, default=3)
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 2])
    ap.add_argument("--out",
                    default="benchmarks/results/skeleton_enum/remex_join_ab.json")
    args = ap.parse_args()

    cfg = TARGETS[args.target]
    n_vars = cfg["n_vars"]
    import zlib
    rng = np.random.default_rng(zlib.crc32(args.target.encode()))
    X = rng.uniform(*TRAIN_DOMAIN, size=(TRAIN_N, n_vars))
    y = cfg["fn"](X)
    X_scr, y_scr = X[:SCREEN_N], y[:SCREEN_N]
    X_held = rng.uniform(*HELD_DOMAIN, size=(HELD_N, n_vars))
    y_held = cfg["fn"](X_held)

    t0 = time.time()
    cache, n_skel, n_assign = build_side_cache(n_vars,
                                               args.max_side_depth, X_scr)
    entries = list(cache.values())
    values = np.stack([e[3] for e in entries])
    print(f"# cache: {len(entries)} entries "
          f"({values.nbytes / 1e6:.0f} MB values) in "
          f"{time.time() - t0:.0f}s", flush=True)

    results = [run_join(entries, values, keys_exact, y_scr, X, y,
                        X_held, y_held, "exact-f64")]
    print(json.dumps(results[-1]), flush=True)
    results.append(run_join(entries, values, ScalarKeyer().keys, y_scr,
                            X, y, X_held, y_held, "arcsinh-int16"))
    print(json.dumps(results[-1]), flush=True)
    for b in args.bits:
        keyer = RemexKeyer(values.shape[1], b)
        results.append(run_join(entries, values, keyer.keys, y_scr,
                                X, y, X_held, y_held, f"remex-{b}bit"))
        print(json.dumps(results[-1]), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(target=args.target,
                   max_side_depth=args.max_side_depth,
                   cache_entries=len(entries), results=results),
              open(args.out, "w"), indent=1)
    print(f"# DONE -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
