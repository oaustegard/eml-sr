import numpy as np

import eml_complexity as ec


def _confirm(levels, t_key, outer_inner, n_part, pos_part, form):
    """Recompute the candidate tree from the CACHED partner value and compare keys.

    The solve step produces b*, d* from the target, so recomputing with those
    values is circular: it returns the target by construction. Only the value
    actually stored in the cache proves the tree; anything that merely rounds
    to it at 11 digits is the near-identity class the frontier filter rejects.
    """
    outer, inner = outer_inner
    part = np.asarray(levels.vals[n_part][pos_part:pos_part + 1], dtype=ec.DT)
    outer = np.asarray([outer], dtype=ec.DT)
    if form == "root":
        v, ok = ec.eml_vec(outer, part)
    else:
        inner = np.asarray([inner], dtype=ec.DT)
        y, ok1 = ec.eml_vec(inner, part)
        if form == "form1":
            v, ok2 = ec.eml_vec(outer, y)
        else:
            v, ok2 = ec.eml_vec(y, outer)
        ok = ok1 & ok2
    return bool(ok[0]) and int(ec.qkey(v)[0]) == int(t_key)


def _sorted_lookup_hits(keys, levels, log, err_prefix):
    """Sort keys, look them up, and yield lookup hits.

    Yields (unsorted_index, size, position) for each hit, keeping the
    sort permutation so callers can map back to unsorted survivors.
    """
    order = np.argsort(keys)
    k_sorted = keys[order]
    try:
        sizes, positions = levels.lookup(k_sorted)
    except Exception as exc:
        log(f"{err_prefix}: {exc}")
        return
    for j in np.nonzero(sizes >= 0)[0]:
        yield int(order[j]), int(sizes[j]), int(positions[j])


def root_join(
    levels: ec.Levels,
    target_value: complex,
    target_key: int,
    *,
    chunk: int = 2_000_000,
    log=print,
) -> tuple[int, str] | None:
    """Search t as eml(a, b) with a enumerated and b looked up.

    Minimal size na + nb + 1 over all hits is returned with its witness.
    """
    max_n = int(levels.N)
    t_key = int(target_key)
    if ec.CPLX:
        t_target = complex(target_value)
    else:
        t_target = float(target_value.real)
    level_dtype = ec.DT
    chunk_int = int(chunk)
    if chunk_int <= 0:
        log(f"root_join: non-positive chunk {chunk_int}")
        return None
    best_size = None
    best_witness = None
    total_tested = 0
    if max_n < 0:
        log(f"root_join: tested {total_tested} candidates, best size None")
        return None
    for na in range(max_n + 1):
        try:
            level_vals = levels.vals[na]
        except Exception as exc:
            log(f"root_join: missing level {na}: {exc}")
            continue
        try:
            m_total = len(level_vals)
        except Exception as exc:
            log(f"root_join: cannot size level {na}: {exc}")
            continue
        if m_total == 0:
            continue
        for start in range(0, m_total, chunk_int):
            end = start + chunk_int
            if end > m_total:
                end = m_total
            try:
                a_chunk = np.asarray(level_vals[start:end], dtype=level_dtype)
            except Exception as exc:
                log(f"root_join: cannot read level {na} slice {start}:{end}: {exc}")
                continue
            idx_chunk = np.arange(start, end, dtype=np.int64)
            total_tested = total_tested + int(end - start)
            if ec.CPLX:
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    raw = np.exp(np.exp(a_chunk.astype(np.complex128, copy=False)) - t_target)
                b_star = ec.snap(raw)
                valid = np.isfinite(b_star)
            else:
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    b_star = np.exp(np.exp(a_chunk.astype(np.float64, copy=False)) - t_target)
                valid = np.isfinite(b_star) & (b_star > 0)
            if not bool(np.any(valid)):
                continue
            a_valid = a_chunk[valid]
            b_valid = b_star[valid]
            idx_valid = idx_chunk[valid]
            v_pred, ok = ec.eml_vec(a_valid, b_valid)
            q_pred = ec.qkey(v_pred)
            match = ok & (q_pred == t_key)
            if not bool(np.any(match)):
                continue
            b_surv = b_valid[match]
            a_surv = a_valid[match]
            ia_surv = idx_valid[match]
            k_b = ec.qkey(b_surv)
            err_prefix = f"root_join: lookup failed at level {na}"
            for unsorted, size_b, pos_b in _sorted_lookup_hits(k_b, levels, log, err_prefix):
                if size_b < 0:
                    continue
                if pos_b < 0:
                    continue
                ia = int(ia_surv[unsorted])
                cand_size = int(na) + size_b + 1
                if best_size is not None and cand_size >= best_size:
                    continue
                if not _confirm(levels, t_key, (a_surv[unsorted], None), size_b, pos_b, "root"):
                    continue
                try:
                    left_w = levels.witness(int(na), ia)
                except Exception as exc:
                    log(f"root_join: witness left failed: {exc}")
                    continue
                try:
                    right_w = levels.witness(size_b, pos_b)
                except Exception as exc:
                    log(f"root_join: witness right failed: {exc}")
                    continue
                best_size = cand_size
                best_witness = f"e({left_w},{right_w})"
    log(f"root_join: tested {total_tested} candidates, best size {best_size}")
    if best_size is None:
        return None
    return (best_size, best_witness)


def two_level_join(
    levels: ec.Levels,
    target_value: complex,
    target_key: int,
    *,
    K: int,
    budget: float,
    chunk: int = 500_000,
    log=print,
) -> tuple[int, str] | None | str:
    """Search t as eml(a, eml(c, d)) and eml(eml(c, d), b).

    Inner level c is restricted to 0..K, outer over 0..N, d looked up.
    Returns minimal (size, witness), None, or "skipped" over budget.
    """
    max_n = int(levels.N)
    k_lim = int(K)
    t_key = int(target_key)
    if ec.CPLX:
        t_target = complex(target_value)
    else:
        t_target = float(target_value.real)
    level_dtype = ec.DT
    chunk_int = int(chunk)
    if chunk_int <= 0:
        log(f"two_level_join: non-positive chunk {chunk_int}")
        return None
    budget_val = float(budget)
    if max_n < 0:
        log("two_level_join: empty levels, nothing to do")
        return None
    if k_lim < 0:
        log("two_level_join: negative K, nothing to do")
        return None
    k_cap = k_lim
    if k_cap > max_n:
        k_cap = max_n
    try:
        total_n = 0
        for n in range(max_n + 1):
            total_n = total_n + len(levels.vals[n])
        total_k = 0
        for n in range(k_cap + 1):
            total_k = total_k + len(levels.vals[n])
    except Exception as exc:
        log(f"two_level_join: cannot sum level sizes: {exc}")
        return None
    planned = 2 * int(total_n) * int(total_k)
    log(f"two_level_join: planned {planned} pairs, budget {budget_val}")
    if float(planned) > budget_val:
        log(f"two_level_join: skipped, planned {planned} exceeds budget {budget_val}")
        return "skipped"
    best_size = None
    best_witness = None
    pair_limit = chunk_int
    if pair_limit <= 0:
        log("two_level_join: non-positive pair limit")
        return None
    for na in range(max_n + 1):
        try:
            m_a = len(levels.vals[na])
        except Exception as exc:
            log(f"two_level_join form1: cannot size outer {na}: {exc}")
            continue
        if m_a == 0:
            continue
        for a_start in range(0, m_a, chunk_int):
            a_end = a_start + chunk_int
            if a_end > m_a:
                a_end = m_a
            try:
                a_block = np.asarray(levels.vals[na][a_start:a_end], dtype=level_dtype)
            except Exception as exc:
                log(f"two_level_join form1: cannot read outer {na}: {exc}")
                continue
            a_idx_block = np.arange(a_start, a_end, dtype=np.int64)
            len_a = len(a_block)
            if len_a == 0:
                continue
            denom = len_a
            if denom <= 0:
                denom = 1
            inner_step_outer = pair_limit // denom
            if inner_step_outer <= 0:
                inner_step_outer = 1
            for nc in range(k_cap + 1):
                try:
                    m_c = len(levels.vals[nc])
                except Exception as exc:
                    log(f"two_level_join form1: cannot size inner {nc}: {exc}")
                    continue
                if m_c == 0:
                    continue
                inner_step = inner_step_outer
                if inner_step > m_c:
                    inner_step = m_c
                for c_start in range(0, m_c, inner_step):
                    c_end = c_start + inner_step
                    if c_end > m_c:
                        c_end = m_c
                    try:
                        c_block = np.asarray(levels.vals[nc][c_start:c_end], dtype=level_dtype)
                    except Exception as exc:
                        log(f"two_level_join form1: cannot read inner {nc}: {exc}")
                        continue
                    c_idx_block = np.arange(c_start, c_end, dtype=np.int64)
                    len_c = len(c_block)
                    if len_c == 0:
                        continue
                    a_rep = np.repeat(a_block, len_c)
                    c_tile = np.tile(c_block, len_a)
                    a_idx_rep = np.repeat(a_idx_block, len_c)
                    c_idx_tile = np.tile(c_idx_block, len_a)
                    if ec.CPLX:
                        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                            y_raw = np.exp(np.exp(a_rep.astype(np.complex128, copy=False)) - t_target)
                        y_star = ec.snap(y_raw)
                        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                            d_raw = np.exp(np.exp(c_tile.astype(np.complex128, copy=False)) - y_star)
                        d_star = ec.snap(d_raw)
                        valid = np.isfinite(y_star) & np.isfinite(d_star)
                    else:
                        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                            y_star = np.exp(np.exp(a_rep.astype(np.float64, copy=False)) - t_target)
                            d_star = np.exp(np.exp(c_tile.astype(np.float64, copy=False)) - y_star)
                        valid = np.isfinite(y_star) & np.isfinite(d_star)
                        valid = valid & (y_star > 0) & (d_star > 0)
                    if not bool(np.any(valid)):
                        continue
                    a_f = a_rep[valid]
                    c_f = c_tile[valid]
                    d_f = d_star[valid]
                    a_idx_f = a_idx_rep[valid]
                    c_idx_f = c_idx_tile[valid]
                    y_mid, ok1 = ec.eml_vec(c_f, d_f)
                    t_pred, ok2 = ec.eml_vec(a_f, y_mid)
                    q_pred = ec.qkey(t_pred)
                    match = ok1 & ok2 & (q_pred == t_key)
                    if not bool(np.any(match)):
                        continue
                    d_surv = d_f[match]
                    a_surv = a_f[match]
                    c_surv = c_f[match]
                    a_idx_surv = a_idx_f[match]
                    c_idx_surv = c_idx_f[match]
                    k_d = ec.qkey(d_surv)
                    for unsorted, nd, pos_d in _sorted_lookup_hits(
                        k_d, levels, log, "two_level_join form1: lookup failed"
                    ):
                        if nd < 0:
                            continue
                        if pos_d < 0:
                            continue
                        ia = int(a_idx_surv[unsorted])
                        ic = int(c_idx_surv[unsorted])
                        cand_size = int(na) + int(nc) + nd + 2
                        if best_size is not None and cand_size >= best_size:
                            continue
                        if not _confirm(levels, t_key, (a_surv[unsorted], c_surv[unsorted]), nd, pos_d, "form1"):
                            continue
                        try:
                            w_a = levels.witness(int(na), ia)
                        except Exception as exc:
                            log(f"two_level_join form1: witness a failed: {exc}")
                            continue
                        try:
                            w_c = levels.witness(int(nc), ic)
                        except Exception as exc:
                            log(f"two_level_join form1: witness c failed: {exc}")
                            continue
                        try:
                            w_d = levels.witness(nd, pos_d)
                        except Exception as exc:
                            log(f"two_level_join form1: witness d failed: {exc}")
                            continue
                        best_size = cand_size
                        best_witness = f"e({w_a},e({w_c},{w_d}))"
    for nb in range(max_n + 1):
        try:
            m_b = len(levels.vals[nb])
        except Exception as exc:
            log(f"two_level_join form2: cannot size outer {nb}: {exc}")
            continue
        if m_b == 0:
            continue
        for b_start in range(0, m_b, chunk_int):
            b_end = b_start + chunk_int
            if b_end > m_b:
                b_end = m_b
            try:
                b_block = np.asarray(levels.vals[nb][b_start:b_end], dtype=level_dtype)
            except Exception as exc:
                log(f"two_level_join form2: cannot read outer {nb}: {exc}")
                continue
            b_idx_block = np.arange(b_start, b_end, dtype=np.int64)
            len_b = len(b_block)
            if len_b == 0:
                continue
            denom2 = len_b
            if denom2 <= 0:
                denom2 = 1
            inner_step_outer2 = pair_limit // denom2
            if inner_step_outer2 <= 0:
                inner_step_outer2 = 1
            for nc in range(k_cap + 1):
                try:
                    m_c = len(levels.vals[nc])
                except Exception as exc:
                    log(f"two_level_join form2: cannot size inner {nc}: {exc}")
                    continue
                if m_c == 0:
                    continue
                inner_step2 = inner_step_outer2
                if inner_step2 > m_c:
                    inner_step2 = m_c
                for c_start in range(0, m_c, inner_step2):
                    c_end = c_start + inner_step2
                    if c_end > m_c:
                        c_end = m_c
                    try:
                        c_block = np.asarray(levels.vals[nc][c_start:c_end], dtype=level_dtype)
                    except Exception as exc:
                        log(f"two_level_join form2: cannot read inner {nc}: {exc}")
                        continue
                    c_idx_block = np.arange(c_start, c_end, dtype=np.int64)
                    len_c = len(c_block)
                    if len_c == 0:
                        continue
                    b_rep = np.repeat(b_block, len_c)
                    c_tile = np.tile(c_block, len_b)
                    b_idx_rep = np.repeat(b_idx_block, len_c)
                    c_idx_tile = np.tile(c_idx_block, len_b)
                    if ec.CPLX:
                        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                            ln_b = np.log(b_rep.astype(np.complex128, copy=False))
                            w_val = t_target + ln_b
                            y_raw = np.log(w_val)
                        y_star = ec.snap(y_raw)
                        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                            d_raw = np.exp(np.exp(c_tile.astype(np.complex128, copy=False)) - y_star)
                        d_star = ec.snap(d_raw)
                        valid = np.isfinite(y_star) & np.isfinite(d_star)
                    else:
                        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                            ln_b = np.log(b_rep.astype(np.float64, copy=False))
                            w_val = t_target + ln_b
                            y_star = np.log(w_val)
                            d_star = np.exp(np.exp(c_tile.astype(np.float64, copy=False)) - y_star)
                        valid = (b_rep > 0) & (w_val > 0)
                        valid = valid & np.isfinite(y_star) & np.isfinite(d_star) & (d_star > 0)
                    if not bool(np.any(valid)):
                        continue
                    b_f = b_rep[valid]
                    c_f = c_tile[valid]
                    d_f = d_star[valid]
                    b_idx_f = b_idx_rep[valid]
                    c_idx_f = c_idx_tile[valid]
                    y_mid, ok1 = ec.eml_vec(c_f, d_f)
                    t_pred, ok2 = ec.eml_vec(y_mid, b_f)
                    q_pred = ec.qkey(t_pred)
                    match = ok1 & ok2 & (q_pred == t_key)
                    if not bool(np.any(match)):
                        continue
                    d_surv = d_f[match]
                    b_surv = b_f[match]
                    c_surv = c_f[match]
                    b_idx_surv = b_idx_f[match]
                    c_idx_surv = c_idx_f[match]
                    k_d = ec.qkey(d_surv)
                    for unsorted, nd, pos_d in _sorted_lookup_hits(
                        k_d, levels, log, "two_level_join form2: lookup failed"
                    ):
                        if nd < 0:
                            continue
                        if pos_d < 0:
                            continue
                        ib = int(b_idx_surv[unsorted])
                        ic = int(c_idx_surv[unsorted])
                        cand_size = int(nb) + int(nc) + nd + 2
                        if best_size is not None and cand_size >= best_size:
                            continue
                        if not _confirm(levels, t_key, (b_surv[unsorted], c_surv[unsorted]), nd, pos_d, "form2"):
                            continue
                        try:
                            w_b = levels.witness(int(nb), ib)
                        except Exception as exc:
                            log(f"two_level_join form2: witness b failed: {exc}")
                            continue
                        try:
                            w_c = levels.witness(int(nc), ic)
                        except Exception as exc:
                            log(f"two_level_join form2: witness c failed: {exc}")
                            continue
                        try:
                            w_d = levels.witness(nd, pos_d)
                        except Exception as exc:
                            log(f"two_level_join form2: witness d failed: {exc}")
                            continue
                        best_size = cand_size
                        best_witness = f"e(e({w_c},{w_d}),{w_b})"
    log(f"two_level_join: planned {planned} pairs, best size {best_size}")
    if best_size is None:
        return None
    return (best_size, best_witness)

