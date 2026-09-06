"""OR-amplification in the meet-in-the-middle join (issue #75).

Covers the three things the multi-table path can get wrong: keys that
are not actually independent across tables, a union that loses or
duplicates candidates, and a sizing formula that disagrees with its own
closed form.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.remex_join import (
    INT16_LIMIT,
    INT32_LIMIT,
    BandedScalarKeyer,
    ScalarKeyer,
    _build_tables,
    clip_fraction,
    _pair_collision_rate,
    _union_ratio,
    collide_prob,
    or_table_set,
    pair_mass_by_bucket_size,
    plan_or_amplification,
    probe_tables,
    run_join,
    simulate_probe_load,
)
from benchmarks.skeleton_exact import (
    HELD_DOMAIN,
    HELD_N,
    SCREEN_N,
    TRAIN_DOMAIN,
    TRAIN_N,
)

D = 16


def _values(n=400, seed=7):
    rng = np.random.default_rng(seed)
    V = rng.normal(scale=3.0, size=(n, D))
    # near-duplicate block: pairs that a coarse key should merge and an
    # exact key should not
    V[1::2] = V[0::2] + rng.normal(scale=1e-3, size=V[0::2].shape)
    return V


# ─── keyers ─────────────────────────────────────────────────────────

def _partition(keys):
    cells = {}
    for i, k in enumerate(keys):
        cells.setdefault(k, []).append(i)
    return sorted(map(tuple, cells.values()))


def test_undithered_band_d_reproduces_the_scalarkeyer_partition():
    """BandedScalarKeyer widens the codes to int32, so its bytes differ
    from the measured arcsinh-int16 arm -- the cells must not."""
    V = _values()
    assert _partition(BandedScalarKeyer(D, 0.01, D, seed=None).keys(V)) == \
        _partition(ScalarKeyer(0.01).keys(V))


def test_seeds_produce_different_cell_boundaries():
    V = _values()
    a = BandedScalarKeyer(D, 0.01, D, seed=0).keys(V)
    b = BandedScalarKeyer(D, 0.01, D, seed=1).keys(V)
    assert a != b
    # but each is deterministic for its seed
    assert a == BandedScalarKeyer(D, 0.01, D, seed=0).keys(V)


def test_identical_rows_always_share_a_cell():
    """A pair the exact-f64 arm finds must survive every table: identical
    float64 rows encode identically under any seed."""
    V = _values()
    W = np.vstack([V, V[:50]])
    for kf in or_table_set(D, 0.01, D, 4):
        keys = kf.keys(W)
        for i in range(50):
            assert keys[i] == keys[V.shape[0] + i]


def test_band_selects_that_many_coordinates():
    kf = BandedScalarKeyer(D, 0.01, 5, seed=3)
    assert kf.cols.size == 5
    assert len(set(kf.cols.tolist())) == 5
    assert len(kf.keys(_values(10))[0]) == 5 * 4  # int32


def test_band_larger_than_d_clamps():
    assert BandedScalarKeyer(D, 0.01, 999, seed=0).band == D


def test_nonfinite_rows_do_not_raise():
    V = _values(20)
    V[0, 0] = np.inf
    V[1, 1] = np.nan
    assert len(BandedScalarKeyer(D, 0.01, D, seed=0).keys(V)) == 20


# ─── tables and the union ───────────────────────────────────────────

def test_every_index_lands_in_exactly_one_cell_per_table():
    V = _values()
    tables, stats = _build_tables(V, [k.keys for k in or_table_set(D, 0.05, D, 3)])
    assert stats["n_tables"] == 3
    for tb in tables:
        seen = [i for v in tb.values() for i in v]
        assert sorted(seen) == list(range(V.shape[0]))


def test_union_is_a_superset_of_each_table_and_is_deduped():
    V = _values()
    keyers = or_table_set(D, 0.05, D, 4)
    tables, _ = _build_tables(V, [k.keys for k in keyers])
    key_cols = [k.keys(V) for k in keyers]
    for r in range(0, V.shape[0], 17):
        cand, raw = probe_tables(tables, key_cols, r)
        cand = set(cand)
        assert len(cand) == len(set(cand))
        assert raw >= len(cand)
        for t in range(len(tables)):
            single, _ = probe_tables([tables[t]], [key_cols[t]], r)
            assert set(single) <= cand


def test_more_tables_never_lose_candidates():
    """Recall is monotone in O: the union over O+1 tables contains the
    union over the first O."""
    V = _values()
    keyers = or_table_set(D, 0.05, D, 6)
    tables, _ = _build_tables(V, [k.keys for k in keyers])
    key_cols = [k.keys(V) for k in keyers]
    for r in range(0, V.shape[0], 29):
        prev = set()
        for o in range(1, 7):
            cur, _ = probe_tables(tables[:o], key_cols[:o], r)
            cur = set(cur)
            assert prev <= cur
            prev = cur


def test_single_table_probe_returns_the_bucket_itself():
    V = _values()
    keyers = or_table_set(D, 0.05, D, 1)
    tables, _ = _build_tables(V, [k.keys for k in keyers])
    key_cols = [k.keys(V) for k in keyers]
    cand, raw = probe_tables(tables, key_cols, 0)
    assert raw == len(cand)
    assert 0 in set(cand)


def test_bucket_cap_drops_whole_cells_and_reports_the_loss():
    V = np.zeros((100, D))                  # one cell, 90 entries
    V[:10] = np.arange(1, 11)[:, None]      # ...plus 10 singletons
    keyers = [BandedScalarKeyer(D, 0.5, D, seed=0)]
    tables, stats = _build_tables(V, [k.keys for k in keyers], bucket_cap=20)
    assert stats["capped_cells"] == 1
    assert stats["capped_entries"] == 90
    assert stats["max_bucket"] <= 20
    assert sum(len(v) for v in tables[0].values()) == 10


def test_or_arm_runs_and_reports_per_table_stats():
    V = _values()
    keyers = or_table_set(D, 0.05, D, 3)
    _, stats = _build_tables(V, [k.keys for k in keyers])
    assert len(stats["per_table"]) == 3
    assert stats["key_bytes"] == sum(
        4 * D * t["n_buckets"] for t in stats["per_table"])


# ─── sizing ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("band,tables", [(1, 1), (4, 1), (4, 4), (16, 8)])
def test_collide_prob_matches_closed_form(band, tables):
    for s in (0.0, 0.3, 0.7, 1.0):
        assert collide_prob(s, band, tables) == \
            pytest.approx(1.0 - (1.0 - s ** band) ** tables)


def test_collide_prob_is_monotone_in_both_axes():
    s = 0.8
    by_o = [collide_prob(s, 8, o) for o in range(1, 9)]
    assert by_o == sorted(by_o)
    by_a = [collide_prob(s, a, 4) for a in range(1, 17)]
    assert by_a == sorted(by_a, reverse=True)


def test_exact_pair_always_collides():
    assert collide_prob(1.0, 16, 1) == 1.0


def test_pair_collision_rate_counts_pairs_not_rows():
    """Two cells of 3 out of 6 rows: 2*C(3,2)/C(6,2) = 6/15."""
    V = np.zeros((6, D))
    V[3:] = 100.0

    class _Fixed:
        def keys(self, X):
            return [b"a" if X[i, 0] == 0 else b"b" for i in range(X.shape[0])]

    p, maxb, nb = _pair_collision_rate(_Fixed(), V)
    assert p == pytest.approx(6 / 15)
    assert (maxb, nb) == (3, 2)


def test_union_ratio_excludes_the_query_row_itself():
    """Every sample row is in its own cell in every table; counting that
    would peg union/sum at 1/O regardless of the data."""
    V = _values(300)
    rng = np.random.default_rng(0)
    # fine enough that every cell is a singleton, coarse enough that the
    # codes do not saturate int16 (see clip_fraction)
    keyers = or_table_set(D, 1e-4, D, 4)
    ur = _union_ratio(keyers, V, 100, rng)
    assert ur["mean_sum"] == 0.0
    assert ur["mean_union"] == 0.0


def test_union_ratio_is_bounded_by_first_table_and_sum():
    V = _values(600)
    rng = np.random.default_rng(1)
    keyers = or_table_set(D, 0.2, D, 4)
    ur = _union_ratio(keyers, V, 200, rng)
    assert ur["mean_first_table"] <= ur["mean_union"] + 1e-9
    assert ur["mean_union"] <= ur["mean_sum"] + 1e-9
    assert 0.0 <= ur["union_over_sum"] <= 1.0


def test_plan_covers_the_grid_and_carries_the_recall_curve():
    V = _values(1200)
    plan = plan_or_amplification(V, bands=[4, 16], resolutions=[0.01, 0.001],
                                 table_counts=[1, 4], sample=800, n_query=100,
                                 baseline_hits=16402595, baseline_p=1e-4)
    assert len(plan["cells"]) == 4
    for cell in plan["cells"]:
        assert set(cell["tables"]) == {"1", "4"}
        for o, ent in cell["tables"].items():
            assert ent["key_bytes_per_item"] == 4 * cell["band"] * int(o)
            assert ent["recall"]["s=1"] == 1.0
            assert "pred_hits_calibrated" in ent
        # a finer grid cannot collide more often than a coarser one
    by_res = {(c["band"], c["resolution"]): c["sample_p"] for c in plan["cells"]}
    for band in (4, 16):
        assert by_res[(band, 0.001)] <= by_res[(band, 0.01)]


def test_plan_clamps_band_to_d():
    plan = plan_or_amplification(_values(200), bands=[999], resolutions=[0.01],
                                 table_counts=[1], sample=200, n_query=20)
    assert plan["cells"][0]["band"] == D


def test_clip_fraction_flags_a_saturating_resolution():
    """The int16 arm saturates four orders of magnitude sooner than the
    int32 one, which is why the banded keyer widened the codes."""
    V = _values(200)
    assert clip_fraction(V, 0.01) == 0.0
    assert clip_fraction(V, 1e-7, limit=INT16_LIMIT) > 0.99
    assert clip_fraction(V, 1e-7, limit=INT32_LIMIT) == 0.0
    assert clip_fraction(V, 1e-11, limit=INT32_LIMIT) > 0.99
    codes = np.frombuffer(b"".join(BandedScalarKeyer(D, 1e-11, D, seed=0)
                                   .keys(V)), dtype=np.int32)
    # all but the handful of near-zero coordinates pin to the rail
    assert np.mean(np.abs(codes) == INT32_LIMIT) > 0.99


def test_clip_fraction_ignores_nonfinite_coordinates():
    V = _values(50)
    V[0, 0] = np.inf
    assert clip_fraction(V, 0.01) == 0.0


# ─── probe-side load ────────────────────────────────────────────────

def test_simulate_probe_load_scales_and_bounds_the_union():
    V = _values(500)
    y_scr = np.linspace(-1.0, 1.0, D)
    for o in (1, 3):
        out = simulate_probe_load(V, y_scr, or_table_set(D, 0.05, D, o),
                                  cache_sample=200, u_sample=8, seed=0)
        assert out["n_tables"] == o
        assert out["sample_union_hits"] <= out["sample_raw_hits"]
        assert out["pred_hits"] >= out["sample_union_hits"]
        assert 0.0 <= out["union_over_raw"] <= 1.0
        assert set(out["pair_mass"]) == {">1", ">10", ">100", ">1000"}


def test_pair_mass_by_bucket_size_splits_the_load():
    # 1000 singletons contribute nothing; one cell of 100 carries it all
    assert pair_mass_by_bucket_size([1] * 1000 + [100])[">10"] == 1.0
    assert pair_mass_by_bucket_size([1] * 1000)[">1"] == 0.0
    # two equal cells of 10: half the pair mass each, all of it above 1
    assert pair_mass_by_bucket_size([10, 10])[">1"] == 1.0
    assert pair_mass_by_bucket_size([10, 10])[">10"] == 0.0


# ─── end-to-end join ────────────────────────────────────────────────

def _planted_join(seed=11):
    """A two-entry cache and a target that is exactly one root join of
    them: y = exp(x0) - ln(x1), with (a_u, g_u, a_v, g_v) = (0, 1, 0, 1)
    on the ALPHAS x GAMMAS lattice. The join must find it."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(*TRAIN_DOMAIN, size=(TRAIN_N, 2))
    X_held = rng.uniform(*HELD_DOMAIN, size=(HELD_N, 2))
    fn = lambda A: np.exp(A[:, 0]) - np.log(A[:, 1])  # noqa: E731
    X_scr = X[:SCREEN_N]
    entries = [("var", None, 0, X_scr[:, 0].astype(np.float64)),
               ("var", None, 1, X_scr[:, 1].astype(np.float64))]
    values = np.stack([e[3] for e in entries])
    return (entries, values, fn(X_scr), X, fn(X), X_held, fn(X_held))


def _run(args, keyers, label):
    entries, values, y_scr, X, y, X_held, y_held = args
    return run_join(entries, values, keyers, y_scr, X, y, X_held, y_held,
                    label)


def test_run_join_finds_the_planted_pair_under_one_table():
    args = _planted_join()
    out = _run(args, ScalarKeyer().keys, "single")
    assert out["discoveries"] >= 1
    assert "eml(1*(x0), 1*(x1))" in out["sample_forms"]


def test_run_join_records_the_key_mismatch_of_each_discovery():
    """The mismatch is what turns the sizer's parametric recall table
    into a resolution: s = 1 - mismatch/res."""
    args = _planted_join()
    out = _run(args, ScalarKeyer().keys, "single")
    mm = out["key_mismatch"]
    assert mm["n"] == out["discoveries"]
    assert 0.0 <= mm["min"] <= mm["median"] <= mm["max"]
    # the planted pair is exact, so its probe lands on the cache row
    assert mm["min"] < 1e-9


def test_or_finds_at_least_what_one_table_finds():
    """Recall is monotone in O end to end, not just per bucket."""
    args = _planted_join()
    one = _run(args, [BandedScalarKeyer(D, 0.01, D, seed=0).keys], "or-1")
    four = _run(args, [k.keys for k in or_table_set(D, 0.01, D, 4)], "or-4")
    assert four["discoveries"] >= one["discoveries"] >= 1
    assert four["hash_hits"] >= one["hash_hits"]
    assert four["raw_hits"] >= four["hash_hits"]  # dedupe never adds


def test_or_and_single_table_agree_on_the_planted_pair():
    """More tables change which pairs are OFFERED, never which are
    accepted: the screen and _confirm are untouched."""
    args = _planted_join()
    single = _run(args, ScalarKeyer().keys, "single")
    many = _run(args, [k.keys for k in or_table_set(D, 0.01, D, 3)], "or-3")
    assert set(single["sample_forms"]) <= set(many["sample_forms"])
