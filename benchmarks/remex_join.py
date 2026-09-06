"""A/B test: quantized join keys vs exact float64 keys (issue #62), and
OR-amplified multi-table joins (issue #75).

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

ISSUE #75 -- the second tuning axis. A key built by concatenating A
sub-hashes is an AND: a pair collides only if all A codes agree.
Running O such tables under independent seeds and taking the union of
their candidate lists is an OR. A pair whose per-sub-hash agreement
probability is s collides somewhere with probability

    1 - (1 - s**A)**O

so A and O move recall and bucket size on separate axes, where a single
table can only trade one against the other through key resolution. The
measured single-table A/B is in results/skeleton_enum/remex_join_ab.json:
coarsening from exact-f64 to arcsinh-int16 bought 14x recall (32 -> 473
discoveries) and cost 67,000x candidate load (244 -> 16.4M hash hits),
with the largest bucket going from 1 to 5,568.

`run_join` takes a list of keyers, probes every table, and unions the
candidate indices before the 16-sample screen, so each pair reaches
`_confirm` once regardless of how many tables admitted it. The screen
and `_confirm` are untouched: the union changes which pairs are offered,
never which are accepted.

`--mode size` sizes a configuration before running it. Per (band,
resolution) cell it measures, on a sample of the side cache: the
random-pair collision rate, the bucket profile, the share of colliding
pairs coming from cells over a given size, the union-vs-sum candidate
ratio across seeds, and -- by running real probes through tables built
on the sample -- the probe-side candidate load. Two capped arms (52 and
66 minutes) were spent learning that shape empirically.

Two things the sizer will not tell you, and one it will get wrong if
you skip it:

  * `s` in the formula is not observable from the cache alone. `run_join`
    records `key_mismatch` -- max_j |arcsinh(V_req_j) - arcsinh(V_j)| --
    for every confirmed discovery, which is what fixes a resolution.
  * the 16 key coordinates are screen samples of one smooth function,
    so a pair's per-coordinate mismatches are strongly correlated. The
    AND is therefore less selective than `s**A` says, and the recall
    column should be read as a bound, not a prediction.
  * `BandedScalarKeyer` codes are int32, not int16. The clip sets where
    arcsinh(v)/res saturates, and at int16 a resolution fine enough to
    break up the big buckets (res=0.001) collapses everything past
    |v| ~ 4e13 into one cell -- while the probe side is
    exp(exp(a + g*U) - y), which is routinely that large. A finer key
    would be coarser in exactly the tail that builds the big buckets.
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

from benchmarks.skeleton_branch import (  # noqa: E402
    TARGETS,
    _confirm,
    build_side_cache,
)
from benchmarks.skeleton_exact import (  # noqa: E402
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
        Q = np.clip(np.rint(np.arcsinh(V) / self.res),
                    -INT16_LIMIT, INT16_LIMIT).astype(np.int16)
        return [Q[i].tobytes() for i in range(V.shape[0])]


class BandedScalarKeyer:
    """One LSH table: `band` of the d screen coordinates, each dithered
    by a per-table offset, uniformly quantized on arcsinh(V).

    AND factor: the `band` codes are concatenated into one key, so a pair
    lands in the same cell only if all `band` of them agree.
    OR factor: build several of these with distinct seeds and hand the
    list to `run_join`, which unions their candidates.

    The dither is what makes the tables independent. On a shared grid
    every seed cuts arcsinh(V) at the same boundaries, so a pair straddling
    a boundary straddles it in every table and the union is no wider than
    one table. With an offset drawn uniformly per coordinate, two values
    at distance |d| in arcsinh space land in one cell with probability
    max(0, 1 - |d|/res), independently across tables.

    Codes are int32 (4 bytes/coordinate), not int16. The clip is what
    sets the saturation point, and at int16 a resolution fine enough to
    break up the big buckets saturates the tail into a single cell --
    see `clip_fraction`. `band=d, seed=None` reproduces ScalarKeyer's
    cell assignment (all coordinates, zero dither); the key bytes differ
    by the code width.
    """

    def __init__(self, d: int, resolution: float = 0.01,
                 band: int | None = None, seed: int | None = 0):
        self.d = int(d)
        self.res = float(resolution)
        self.band = self.d if band is None else min(int(band), self.d)
        self.seed = seed
        if seed is None:
            self.cols = np.arange(self.d)
            self.offset = np.zeros(self.d)
        else:
            rng = np.random.default_rng(seed)
            self.cols = np.sort(rng.choice(self.d, size=self.band,
                                           replace=False))
            self.offset = rng.uniform(0.0, 1.0, size=self.band)
        self.label = f"band{self.band}@{self.res:g}/s{seed}"

    def keys(self, V: np.ndarray) -> list:
        with np.errstate(all="ignore"):
            Q = np.clip(np.rint(np.arcsinh(V[:, self.cols]) / self.res
                                + self.offset),
                        -INT32_LIMIT, INT32_LIMIT).astype(np.int32)
        return [Q[i].tobytes() for i in range(V.shape[0])]


def or_table_set(d: int, resolution: float, band: int, n_tables: int,
                 seed0: int = 0) -> list:
    """`n_tables` independently seeded BandedScalarKeyers -- the OR."""
    return [BandedScalarKeyer(d, resolution, band, seed0 + t)
            for t in range(n_tables)]


def collide_prob(s: float, band: int, n_tables: int) -> float:
    """1 - (1 - s**A)**O: probability a pair at per-sub-hash agreement
    `s` collides in at least one of `n_tables` tables.

    `s**A` treats the A coordinates as independent draws. Here they are
    16 screen samples of the same smooth function, so a pair's
    coordinate mismatches move together -- all tiny or all large. Read
    this as the independent-coordinate bound; the real per-table
    collision probability on correlated coordinates is higher at fixed
    mean mismatch, and `simulate_probe_load` measures the load side
    without the assumption.
    """
    return 1.0 - (1.0 - s ** band) ** n_tables


INT16_LIMIT = 32000
INT32_LIMIT = 2_000_000_000


def clip_fraction(V: np.ndarray, resolution: float,
                  limit: int = INT32_LIMIT) -> float:
    """Fraction of coordinates whose code saturates the integer range.

    `rint(arcsinh(v)/res)` is clipped, so every coordinate above
    `limit * res` in arcsinh space takes the same code. The saturation
    point moves with the resolution, which inverts the whole premise of
    the resolution sweep: at int16 and res=0.001 anything past
    |v| ~ 4e13 collapses into one cell, and the probe side is
    exp(exp(a + g*U) - y), which is routinely that large. A finer key
    would then be COARSER in exactly the tail that builds the big
    buckets. BandedScalarKeyer uses int32 for that reason; the
    int16 ScalarKeyer stays as-is because it is the measured
    arcsinh-int16 arm and has to stay byte-reproducible.
    """
    with np.errstate(all="ignore"):
        A = np.abs(np.arcsinh(V)) / float(resolution)
    A = A[np.isfinite(A)]
    if A.size == 0:
        return 0.0
    return float(np.mean(A > limit))


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


def _build_tables(values: np.ndarray, keyers: list, bucket_cap: int = 0):
    """One bucket dict per keyer. Returns (tables, stats)."""
    tables = []
    per_table = []
    key_bytes = 0
    capped_cells = capped_entries = 0
    for kf in keyers:
        buckets = defaultdict(list)
        for i, k in enumerate(kf(values)):
            buckets[k].append(i)
        if bucket_cap:
            over = [k for k, v in buckets.items() if len(v) > bucket_cap]
            capped_cells += len(over)
            for k in over:
                capped_entries += len(buckets[k])
                del buckets[k]
        bs = np.array([len(v) for v in buckets.values()])
        key_bytes += sum(len(k) for k in buckets)
        per_table.append(dict(n_buckets=len(buckets),
                              max_bucket=int(bs.max()) if bs.size else 0,
                              mean_bucket=round(float(bs.mean()), 3)
                              if bs.size else 0.0))
        tables.append(dict(buckets))
    stats = dict(n_tables=len(tables), key_bytes=int(key_bytes),
                 n_buckets=sum(t["n_buckets"] for t in per_table),
                 max_bucket=max((t["max_bucket"] for t in per_table),
                                default=0),
                 per_table=per_table)
    if bucket_cap:
        stats["bucket_cap"] = bucket_cap
        stats["capped_cells"] = capped_cells
        stats["capped_entries"] = capped_entries
    return tables, stats


def probe_tables(tables: list, key_cols: list, r: int) -> tuple:
    """Candidate indices for query row `r` -- the union over all tables.

    Returns (candidates, raw_hits) where `raw_hits` is the pre-dedupe sum
    over tables. A single table returns its bucket list as-is; with more
    than one the union is deduped so `_confirm` runs once per pair no
    matter how many tables admitted it.
    """
    if len(tables) == 1:
        c = tables[0].get(key_cols[0][r], ())
        return c, len(c)
    cand = set()
    raw = 0
    for tb, kl in zip(tables, key_cols):
        c = tb.get(kl[r])
        if c:
            raw += len(c)
            cand.update(c)
    return cand, raw


def run_join(entries, values, keyers, y_scr, X_train, y_train,
             X_held, y_held, label: str, bucket_cap: int = 0):
    """Meet-in-the-middle join over one or more hash tables.

    `keyers` is a single keyer callable or a list of them. With a list,
    a candidate is any index found in ANY table (the OR); the union is
    deduped per query row so `_confirm` runs once per pair no matter how
    many tables admitted it. The 16-sample screen and `_confirm` are
    identical in either case -- more tables change which pairs are
    offered, never which are accepted.
    """
    if callable(keyers):
        keyers = [keyers]
    t0 = time.time()
    tables, stats = _build_tables(values, keyers, bucket_cap)

    hits = raw_hits = tested = 0
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
                        Q = V_req[idx]
                        key_cols = [kf(Q) for kf in keyers]
                        for r in range(idx.size):
                            j = int(idx[r])
                            cand, raw = probe_tables(tables, key_cols, r)
                            raw_hits += raw
                            for vi in cand:
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
                                    # how far apart the probe and the cache
                                    # row actually are in the space the key
                                    # quantizes -- the number that fixes a
                                    # resolution, and the `s` the sizer's
                                    # recall table is parametric over
                                    with np.errstate(all="ignore"):
                                        disc["key_mismatch"] = float(np.max(
                                            np.abs(np.arcsinh(Q[r])
                                                   - np.arcsinh(values[vi]))))
                                    discoveries.append(disc)
    out = dict(label=label, discoveries=len(discoveries),
               hash_hits=hits, raw_hits=raw_hits, joins_tested=tested,
               wall_s=round(time.time() - t0, 1),
               sample_forms=sorted({d["expr"] for d in discoveries})[:3])
    if discoveries:
        mm = np.array([d["key_mismatch"] for d in discoveries])
        mm = mm[np.isfinite(mm)]
        if mm.size:
            out["key_mismatch"] = dict(
                n=int(mm.size), min=float(mm.min()),
                median=float(np.median(mm)), p90=float(np.quantile(mm, 0.9)),
                max=float(mm.max()))
    out.update(stats)
    return out


# ─── Sizing: predict before spending an hour ────────────────────────

def _pair_collision_rate(keyer, sample: np.ndarray) -> tuple:
    """(random-pair collision rate, max bucket, n_buckets) on `sample`.

    p = sum_cells C(b,2) / C(m,2): the probability that a uniformly drawn
    pair of sample rows shares a cell.
    """
    m = sample.shape[0]
    buckets = defaultdict(int)
    for k in keyer.keys(sample):
        buckets[k] += 1
    b = np.array(list(buckets.values()), dtype=np.float64)
    pairs = float(np.sum(b * (b - 1.0) / 2.0))
    total = m * (m - 1.0) / 2.0
    return pairs / total, int(b.max()), len(buckets)


def _bucket_sizes(keyer, sample: np.ndarray) -> list:
    counts = defaultdict(int)
    for k in keyer.keys(sample):
        counts[k] += 1
    return list(counts.values())


def pair_mass_by_bucket_size(bucket_sizes, thresholds=(1, 10, 100, 1000)) -> dict:
    """Share of colliding pairs contributed by cells above each size.

    A single global collision rate blends two regimes: a diffuse mass of
    small cells, which finer resolution and more tables actually move,
    and a handful of degenerate cells (constant-direction families,
    saturated codes) whose pairs are the same in every table and are
    insensitive to both knobs. sum_{b>B} C(b,2) / sum_b C(b,2) says how
    the load splits, which max_bucket alone does not.
    """
    b = np.asarray(bucket_sizes, dtype=np.float64)
    pairs = b * (b - 1.0) / 2.0
    total = float(pairs.sum())
    if total <= 0:
        return {f">{t}": 0.0 for t in thresholds}
    return {f">{t}": round(float(pairs[b > t].sum()) / total, 6)
            for t in thresholds}


def _union_ratio(keyers: list, sample: np.ndarray, n_query: int,
                 rng: np.random.Generator) -> dict:
    """Union-vs-sum candidate load across tables, measured directly.

    1 - (1-p)**O assumes the tables collide independently. On this cache
    they do not: a constant-direction family that degenerates into one
    huge cell degenerates in every table, so all O tables return the same
    rows and the union is no bigger than one table's. That shows up as
    union/sum -> 1/O and union/first -> 1. The opposite extreme --
    tables finding disjoint rows -- is union/sum -> 1 and union/first
    -> O. Both the load and the recall the OR buys live in the gap.
    """
    tables = []
    for kf in keyers:
        buckets = defaultdict(list)
        for i, k in enumerate(kf.keys(sample)):
            buckets[k].append(i)
        tables.append(dict(buckets))
    qi = rng.choice(sample.shape[0], size=min(n_query, sample.shape[0]),
                    replace=False)
    Q = sample[qi]
    key_cols = [kf.keys(Q) for kf in keyers]
    tot_sum = tot_union = 0
    first = 0
    for r in range(Q.shape[0]):
        self_i = int(qi[r])
        u = set()
        for t, (tb, kl) in enumerate(zip(tables, key_cols)):
            # the query row is itself in the sample, so it lands in its own
            # cell in every table; counting that self-hit would drive
            # union/sum toward 1/O no matter how the tables behave
            c = [i for i in tb.get(kl[r], ()) if i != self_i]
            tot_sum += len(c)
            if t == 0:
                first += len(c)
            u.update(c)
        tot_union += len(u)
    n = float(Q.shape[0])
    return dict(mean_first_table=round(first / n, 2),
                mean_sum=round(tot_sum / n, 2),
                mean_union=round(tot_union / n, 2),
                union_over_sum=round(tot_union / max(tot_sum, 1), 4),
                union_over_first=round(tot_union / max(first, 1), 4))


def simulate_probe_load(values: np.ndarray, y_scr: np.ndarray, keyers: list,
                        cache_sample: int = 0, u_sample: int = 4096,
                        seed: int = 0) -> dict:
    """Probe-side candidate load, measured rather than modelled.

    Calibrating from a cache-vs-cache collision rate transfers only if
    the probe keys are distributed like the cache rows, and they are not:
    a probe is (exp(exp(a_u + g_u*U) - y_scr) - a_v)/g_v over the 256
    affine combinations, which is far heavier-tailed than the cache. So
    run real probes through tables built on a cache sample and scale by
    linearity of expectation over (probe, row) pairs:

        hits_full ~= hits_sample * (N / m_cache) * (P / m_probe)

    That is exact in expectation and assumes nothing about independence
    across tables. It is not low-variance, and the shortfall is not
    small: 95% of the colliding pairs sit in cells over 1000 entries, and
    at u_sample=4096 against the whole cache this estimator saw 30 hits
    where the measured single-table run implies ~68,000. The candidate
    load is carried by a rare set of U rows whose required V lands in a
    degenerate cell, not spread across probes, so a uniform probe sample
    misses it almost entirely. Budget U rows in the 1e5 range, or pass
    `u_sample=0` for every row (exact, and about as expensive as running
    the join without the confirm stage).

    Subsampling the CACHE multiplies the same variance by (N/m) on top,
    so `cache_sample=0` (the default) indexes the whole cache and takes
    the variance only on the probe side, where more U rows buy it down
    directly.
    """
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    m = n if cache_sample <= 0 else min(cache_sample, n)
    S = (values if m == n else
         np.ascontiguousarray(values[rng.choice(n, size=m, replace=False)]))
    t_build = time.time()
    tables, stats = _build_tables(S, [k.keys for k in keyers])
    t_build = round(time.time() - t_build, 1)

    nu = n if u_sample <= 0 else min(u_sample, n)
    U = (values if nu == n else
         np.ascontiguousarray(values[rng.choice(n, size=nu, replace=False)]))
    probes = union = raw = 0
    finite_probes = 0
    CHUNK = 65536
    for start in range(0, nu, CHUNK):
        Uc = U[start:start + CHUNK]
        for a_u in ALPHAS:
            for g_u in GAMMAS:
                with np.errstate(all="ignore"):
                    P = np.exp(a_u + g_u * Uc)
                    v_in_req = np.exp(P - y_scr[None, :])
                for a_v in ALPHAS:
                    for g_v in GAMMAS:
                        probes += Uc.shape[0]
                        with np.errstate(all="ignore"):
                            V_req = (v_in_req - a_v) / g_v
                        idx = np.nonzero(np.isfinite(V_req).all(axis=1))[0]
                        if idx.size == 0:
                            continue
                        finite_probes += int(idx.size)
                        key_cols = [kf.keys(V_req[idx]) for kf in keyers]
                        for r in range(idx.size):
                            cand, rh = probe_tables(tables, key_cols, r)
                            union += len(cand)
                            raw += rh
    # the full run probes every cache row against every cache row, so the
    # sample covers m/n of the table and nu/n of the probes
    scale = (n / m) * (n / max(nu, 1))
    return dict(n_tables=len(keyers), cache_sample=m, cache_full=(m == n),
                u_sample=nu, scale=round(scale, 2), build_s=t_build,
                probes=probes, finite_probes=finite_probes,
                sample_union_hits=union, sample_raw_hits=raw,
                hits_per_finite_probe=round(union / max(finite_probes, 1), 4),
                union_over_raw=round(union / max(raw, 1), 4),
                pred_hits=int(round(union * scale)),
                pred_raw_hits=int(round(raw * scale)),
                key_bytes=stats["key_bytes"],
                max_bucket=stats["max_bucket"],
                pair_mass=pair_mass_by_bucket_size(
                    [b for t in tables for b in map(len, t.values())]))


def plan_or_amplification(values: np.ndarray, bands, resolutions, table_counts,
                          sample: int = 50000, n_query: int = 2000,
                          seed: int = 0, baseline_hits: int | None = None,
                          baseline_p: float | None = None) -> dict:
    """Measure the collision load of each (band, resolution) cell on a
    sample of the side cache, then project it to O tables.

    Two numbers per cell come from the data:
      p                random-pair collision rate on the sample
      union_over_sum   how much the O tables actually overlap
      pair_mass        share of colliding pairs from cells over a size
    One comes from the model:
      recall(s)  = 1 - (1 - s**A)**O, tabulated over a grid of s

    The recall table is parametric on purpose: s is the per-sub-hash
    agreement probability of a TRUE pair, which this function cannot see
    (it has no confirmed pairs). `run_join` records the actual
    coordinate mismatch of every discovery as `key_mismatch`; divide
    that by a candidate resolution to get s and read the row.

    The reported load is calibrated: it scales a measured single-table
    baseline by the union rate, which assumes the probe keys are
    distributed like the cache rows. They are not -- a probe is
    exp(exp(a + g*U) - y), far heavier-tailed than the cache -- so read
    the grid to SHORTLIST cells, then measure the shortlist with
    `--mode probe`, which runs real probes and assumes neither.
    """
    rng = np.random.default_rng(seed)
    n, d = values.shape
    m = min(sample, n)
    idx = rng.choice(n, size=m, replace=False)
    S = np.ascontiguousarray(values[idx])

    s_grid = [1.0, 0.99, 0.95, 0.9, 0.8, 0.7, 0.5]
    cells = []
    for band in bands:
        band = min(int(band), d)
        for res in resolutions:
            kf0 = BandedScalarKeyer(d, res, band, seed=seed)
            p, maxb, nb = _pair_collision_rate(kf0, S)
            bsizes = _bucket_sizes(kf0, S)
            row = dict(band=band, resolution=res,
                       key_bytes_per_item=4 * band,
                       sample_p=p, sample_max_bucket=maxb,
                       sample_n_buckets=nb,
                       scaled_max_bucket=int(round(maxb * n / m)),
                       clip_fraction=round(clip_fraction(S, res), 6),
                       pair_mass=pair_mass_by_bucket_size(bsizes),
                       tables={})
            for O in table_counts:
                O = int(O)
                keyers = or_table_set(d, res, band, O, seed0=seed)
                ur = _union_ratio(keyers, S, n_query, rng)
                p_union_model = 1.0 - (1.0 - p) ** O
                # measured overlap correction: how much of the O-fold
                # candidate sum the union actually keeps
                p_union_meas = p * O * ur["union_over_sum"]
                ent = dict(n_tables=O,
                           key_bytes_per_item=4 * band * O,
                           p_union_model=p_union_model,
                           p_union_measured=p_union_meas,
                           recall={f"s={s:g}": round(collide_prob(s, band, O), 6)
                                   for s in s_grid},
                           **ur)
                if baseline_hits and baseline_p:
                    ent["pred_hits_model"] = int(
                        baseline_hits * p_union_model / baseline_p)
                    ent["pred_hits_calibrated"] = int(
                        baseline_hits * p_union_meas / baseline_p)
                row["tables"][str(O)] = ent
            cells.append(row)
    return dict(n_cache=int(n), d=int(d), sample=int(m),
                n_query=int(n_query), s_grid=s_grid, cells=cells,
                baseline_hits=baseline_hits, baseline_p=baseline_p)


def _baseline_probe(values: np.ndarray, sample: int, seed: int) -> float:
    """Random-pair collision rate of the measured single-table
    arcsinh-int16 arm, on the same sample -- the calibration denominator."""
    rng = np.random.default_rng(seed)
    m = min(sample, values.shape[0])
    S = np.ascontiguousarray(values[rng.choice(values.shape[0], size=m,
                                               replace=False)])
    kf = BandedScalarKeyer(values.shape[1], 0.01, values.shape[1], seed=None)
    p, _, _ = _pair_collision_rate(kf, S)
    return p


def _parse_or_config(spec: str) -> tuple:
    """'band,resolution,n_tables' -> (int, float, int)."""
    parts = spec.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected band,resolution,n_tables -- got {spec!r}")
    return int(parts[0]), float(parts[1]), int(parts[2])


def _load_cache(args):
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
    return entries, values, y_scr, X, y, X_held, y_held


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="sum_of_squares",
                    choices=sorted(TARGETS))
    ap.add_argument("--max-side-depth", type=int, default=3)
    ap.add_argument("--mode", default="ab",
                    choices=("ab", "or", "size", "probe"),
                    help="ab: the issue-#62 single-table key A/B. "
                         "or: single-table baseline + OR-amplified arms. "
                         "size: sweep the (band, resolution) grid on a "
                         "cache sample, run no join. "
                         "probe: measure the real probe-side load of each "
                         "--or-config against the whole cache.")
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 2])
    ap.add_argument("--or-config", type=_parse_or_config, nargs="+",
                    default=[(16, 0.002, 4)], metavar="BAND,RES,TABLES",
                    help="OR arms to run in --mode or.")
    ap.add_argument("--bucket-cap", type=int, default=0,
                    help="drop cells larger than this before probing. "
                         "0 = off. A cap loses every pair in a dropped "
                         "cell, in every table that drops it.")
    ap.add_argument("--size-bands", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--size-res", type=float, nargs="+",
                    default=[0.01, 0.005, 0.002, 0.001])
    ap.add_argument("--size-tables", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--size-sample", type=int, default=50000)
    ap.add_argument("--size-queries", type=int, default=2000)
    ap.add_argument("--probe-u", type=int, default=131072,
                    help="cache rows run as real probes in --mode probe; "
                         "0 runs every row (exact). Each contributes 256 "
                         "affine probes. The load is carried by a rare set "
                         "of U rows landing in degenerate cells, so a small "
                         "sample under-reports by orders of magnitude -- "
                         "4096 rows saw 30 hits where the measured run "
                         "implies 68,000.")
    ap.add_argument("--baseline-hits", type=int, default=16402595,
                    help="measured single-table arcsinh-int16 hash hits, "
                         "used to calibrate predicted load.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    default_out = {
        "ab": "benchmarks/results/skeleton_enum/remex_join_ab.json",
        "or": "benchmarks/results/skeleton_enum/remex_join_or.json",
        "size": "benchmarks/results/skeleton_enum/remex_join_or_sizing.json",
        "probe": "benchmarks/results/skeleton_enum/remex_join_or_probe.json",
    }
    out = Path(args.out or default_out[args.mode])

    entries, values, y_scr, X, y, X_held, y_held = _load_cache(args)

    if args.mode == "size":
        t0 = time.time()
        p_base = _baseline_probe(values, args.size_sample, args.seed)
        plan = plan_or_amplification(
            values, args.size_bands, args.size_res, args.size_tables,
            sample=args.size_sample, n_query=args.size_queries,
            seed=args.seed, baseline_hits=args.baseline_hits,
            baseline_p=p_base)
        plan.update(target=args.target, max_side_depth=args.max_side_depth,
                    wall_s=round(time.time() - t0, 1))
        for cell in plan["cells"]:
            for O, ent in sorted(cell["tables"].items(), key=lambda kv: int(kv[0])):
                print(f"# band={cell['band']:2d} res={cell['resolution']:<6g} "
                      f"O={O:<2s} p={cell['sample_p']:.3e} "
                      f"maxbucket~{cell['scaled_max_bucket']:>8d} "
                      f"union/sum={ent['union_over_sum']:.3f} "
                      f"clip={cell['clip_fraction']:.3f} "
                      f"mass>100={cell['pair_mass']['>100']:.3f} "
                      f"calib_hits={ent.get('pred_hits_calibrated', '-')}",
                      flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(plan, open(out, "w"), indent=1)
        print(f"# DONE -> {out}")
        return 0

    if args.mode == "probe":
        d = values.shape[1]
        rows = []
        base = simulate_probe_load(values, y_scr,
                                   [BandedScalarKeyer(d, 0.01, d, seed=None)],
                                   u_sample=args.probe_u, seed=args.seed)
        base["label"] = "arcsinh-int16-equivalent"
        rows.append(base)
        print(json.dumps(base), flush=True)
        for band, res, n_tables in args.or_config:
            out_row = simulate_probe_load(
                values, y_scr, or_table_set(d, res, band, n_tables,
                                            seed0=args.seed),
                u_sample=args.probe_u, seed=args.seed)
            out_row["label"] = f"or-band{min(band, d)}@{res:g}x{n_tables}"
            rows.append(out_row)
            print(json.dumps(out_row), flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(target=args.target,
                       max_side_depth=args.max_side_depth,
                       cache_entries=len(entries),
                       baseline_hits=args.baseline_hits, results=rows),
                  open(out, "w"), indent=1)
        print(f"# DONE -> {out}")
        return 0

    results = []
    if args.mode == "ab":
        results.append(run_join(entries, values, keys_exact, y_scr, X, y,
                                X_held, y_held, "exact-f64"))
        print(json.dumps(results[-1]), flush=True)
        results.append(run_join(entries, values, ScalarKeyer().keys, y_scr,
                                X, y, X_held, y_held, "arcsinh-int16"))
        print(json.dumps(results[-1]), flush=True)
        for b in args.bits:
            keyer = RemexKeyer(values.shape[1], b)
            results.append(run_join(entries, values, keyer.keys, y_scr,
                                    X, y, X_held, y_held, f"remex-{b}bit"))
            print(json.dumps(results[-1]), flush=True)
    else:
        results.append(run_join(entries, values, ScalarKeyer().keys, y_scr,
                                X, y, X_held, y_held, "arcsinh-int16",
                                bucket_cap=args.bucket_cap))
        print(json.dumps(results[-1]), flush=True)
        d = values.shape[1]
        for band, res, n_tables in args.or_config:
            keyers = or_table_set(d, res, band, n_tables, seed0=args.seed)
            label = f"or-band{min(band, d)}@{res:g}x{n_tables}"
            results.append(run_join(entries, values,
                                    [k.keys for k in keyers], y_scr, X, y,
                                    X_held, y_held, label,
                                    bucket_cap=args.bucket_cap))
            print(json.dumps(results[-1]), flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(target=args.target,
                   max_side_depth=args.max_side_depth,
                   cache_entries=len(entries), results=results),
              open(out, "w"), indent=1)
    print(f"# DONE -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
