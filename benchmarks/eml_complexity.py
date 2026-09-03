"""Vectorized EML constant complexity with per-level storage and spill."""
import argparse
import json
import math
import os
import resource
import shutil
import sys
import time
import numpy as np
import sympy as sp


SIG = 11
MAG = 1e300
EXACT = 1e-9
BRANCH = "real"
CPLX = False
DT = np.float64


def set_branch(branch: str) -> None:
    """Set the global branch state before any other call."""
    global BRANCH
    global CPLX
    global DT
    BRANCH = branch
    CPLX = branch == "complex"
    DT = np.complex128 if CPLX else np.float64


def qkey(v: np.ndarray) -> np.ndarray:
    """Quantize to SIG significant digits per component and hash to int64."""
    def q(t):
        """Quantize a float64 array to SIG significant digits."""
        t = np.asarray(t, dtype=np.float64)
        out = np.zeros_like(t)
        nz = np.abs(t) >= 1e-12
        e = np.floor(np.log10(np.abs(t[nz])))
        scale = 10.0 ** (SIG - 1 - e)
        out[nz] = np.round(t[nz] * scale) / scale
        return out

    def mix(h):
        """Splitmix style finalizer for int64 hashes."""
        h = h ^ (h >> np.int64(30))
        h = h * np.int64(-4658895280553007687)
        h = h ^ (h >> np.int64(27))
        h = h * np.int64(-7723592293110705685)
        return h ^ (h >> np.int64(31))

    if CPLX:
        r = q(v.real)
        i = q(v.imag)
        combined = mix(r.view(np.int64)) + np.int64(7150367640395946793) * i.view(np.int64)
        return mix(combined)
    else:
        return mix(q(v).view(np.int64))


def eml_vec(a, b) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate exp(a) - log(b) vectorized with an exactness guard."""
    with np.errstate(all="ignore"):
        if CPLX:
            # canonicalize: snap near-real values to the real axis so ln on the
            # negative axis takes the principal value +i*pi deterministically
            b = np.where(np.abs(b.imag) < 1e-11 * np.abs(b), b.real + 0j, b)
            a = np.where(np.abs(a.imag) < 1e-11 * np.abs(a), a.real + 0j, a)
            v = np.exp(a) - np.log(b)
            v = np.where(np.abs(v.imag) < 1e-11 * np.abs(v), v.real + 0j, v)
            ok = np.isfinite(v.real) & np.isfinite(v.imag) & (np.abs(v) < MAG) & (b != 0)
        else:
            v = np.exp(a) - np.log(b)
            ok = np.isfinite(v) & (b > 0) & (np.abs(v) < MAG)
        # exactness: neither term may be numerically swallowed by the other
        ea = np.abs(np.exp(a))
        lb = np.abs(np.log(b))
        mag = np.maximum(ea, lb)
        # exp never truly 0
        ok &= ~(ea < EXACT * mag) & ~((lb > 0) & (lb < EXACT * mag))
    return v, ok


def snap(x: np.ndarray) -> np.ndarray:
    """Snap near-real complex values to the real axis like eml_vec does."""
    if not CPLX:
        return x
    arr = np.asarray(x, dtype=np.complex128)
    snapped = np.where(np.abs(arr.imag) < 1e-11 * np.abs(arr), arr.real + 0j, arr)
    return snapped


class Levels:
    """Everything enumerated so far. Index = tree size."""

    def __init__(self, branch: str):
        """Create an empty frontier for the given branch."""
        self.branch = branch
        self.keys: list[np.ndarray] = []
        self.vals: list = []
        self.par: list = []

    @property
    def N(self) -> int:
        """Largest size present."""
        return len(self.keys) - 1

    def lookup(self, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Find each sorted key in the levels, first hit wins."""
        if k.size > 1:
            step = max(1, int(k.size // 100))
            sample = k[::step]
            assert np.all(sample[1:] >= sample[:-1])
        sizes = np.full(k.shape, -1, dtype=np.int32)
        positions = np.full(k.shape, -1, dtype=np.int64)
        for n in range(len(self.keys)):
            kn = self.keys[n]
            if kn is None:
                continue
            if len(kn) == 0:
                continue
            pos = np.searchsorted(kn, k)
            valid = pos < len(kn)
            hit = np.zeros(k.shape, dtype=bool)
            idx = np.nonzero(valid)[0]
            if len(idx) > 0:
                eq = kn[pos[idx]] == k[idx]
                hit[idx[eq]] = True
            need = hit & (sizes == -1)
            sizes[need] = np.int32(n)
            positions[need] = pos[need]
        return sizes, positions

    def witness(self, n: int, i: int) -> str:
        """Reconstruct the closed-tree string for value i of size n."""
        if n == 0:
            return "1"
        row = self.par[n][int(i)]
        a = int(row[0])
        ia = int(row[1])
        ib = int(row[2])
        left = self.witness(a, ia)
        right = self.witness(n - 1 - a, ib)
        return f"e({left},{right})"

    def value_key_of(self, n: int, i: int) -> int:
        """Return the int64 key of value i of size n."""
        return int(self.keys[n][int(i)])


def _finalize_level(pending_keys, pending_vals, pending_pars, vals_path, par_path, log=print):
    """Sort a finished level by key once and materialise it, field by field.

    The pending lists are consumed (cleared) as each field is concatenated, so
    the peak is one field's copy plus the permutation, not a copy of the level.
    Values and parents are gathered in slices straight into the memmap when a
    path is given.
    """
    if len(pending_keys) == 0:
        keys = np.empty(0, dtype=np.int64)
    else:
        keys = np.concatenate(pending_keys)
    pending_keys.clear()
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    if len(keys) > 1:
        uniq = np.empty(len(keys), dtype=bool)
        uniq[0] = True
        uniq[1:] = keys[1:] != keys[:-1]
        dups = int(len(keys) - np.count_nonzero(uniq))
        if dups:
            log(f"  finalize: dropped {dups} duplicate keys")
            keys = keys[uniq]
            order = order[uniq]
    total = len(order)

    def gather(parts, out):
        if len(parts) == 0:
            src = np.empty((0,) + out.shape[1:], dtype=out.dtype)
        else:
            src = np.concatenate(parts, axis=0)
        parts.clear()
        for s in range(0, total, 5_000_000):
            idx = order[s:s + 5_000_000]
            out[s:s + len(idx)] = src[idx]
        return out

    if vals_path is not None:
        out_v = np.lib.format.open_memmap(vals_path, mode="w+", dtype=np.dtype(DT), shape=(total,))
        out_p = np.lib.format.open_memmap(par_path, mode="w+", dtype=np.dtype(np.int32), shape=(total, 3))
    else:
        out_v = np.empty(total, dtype=DT)
        out_p = np.empty((total, 3), dtype=np.int32)
    vals = gather(pending_vals, out_v)
    par = gather(pending_pars, out_p)
    if vals_path is not None:
        vals.flush()
        par.flush()
        del vals, par
        vals = np.load(vals_path, mmap_mode="r")
        par = np.load(par_path, mmap_mode="r")
    return keys, vals, par


def expand(branch: str, nmax: int, *, spill_dir: str | None, chunk: int, ram_gb: float, resume: bool, log=print) -> tuple[Levels, dict]:
    """Build the frontier up to nmax with chunking, spill, and guards."""
    set_branch(branch)
    t0 = time.time()
    levels = Levels(branch)
    counts: list[int] = []
    catalan: list[int] = []
    timing_s: list[float] = []
    guard = None
    if spill_dir is not None:
        try:
            os.makedirs(spill_dir, exist_ok=True)
        except Exception as ex:
            log(f"could not create spill dir {spill_dir}: {ex}")
            spill_dir = None
    if resume and spill_dir is not None:
        # SHIM: spec says reopen file pairs from size 0, but par[0] is None
        # so no par file is written for size 0; resume treats that as expected.
        try:
            n_try = 0
            while True:
                vals_path = os.path.join(spill_dir, f"{branch}_vals_{n_try}.npy")
                par_path = os.path.join(spill_dir, f"{branch}_par_{n_try}.npy")
                if not os.path.exists(vals_path):
                    break
                if n_try > 0 and not os.path.exists(par_path):
                    break
                vals_loaded = np.load(vals_path, mmap_mode="r")
                if n_try == 0:
                    par_loaded = None
                else:
                    par_loaded = np.load(par_path, mmap_mode="r")
                keys_re = qkey(np.asarray(vals_loaded))
                if len(keys_re) > 1:
                    assert np.all(keys_re[1:] >= keys_re[:-1])
                levels.keys.append(np.asarray(keys_re))
                levels.vals.append(vals_loaded)
                levels.par.append(par_loaded)
                n_try = n_try + 1
        except Exception as ex:
            log(f"resume failed, continuing with {len(levels.keys)} levels: {ex}")
        if len(levels.keys) > 0:
            log(f"resumed {len(levels.keys)} levels from {spill_dir}")
    if len(levels.keys) == 0:
        v0 = np.array([1.0], dtype=DT)
        k0 = qkey(v0)
        p0 = None
        if spill_dir is not None:
            try:
                vals_path = os.path.join(spill_dir, f"{branch}_vals_0.npy")
                fp = np.lib.format.open_memmap(vals_path, mode="w+", dtype=np.dtype(DT), shape=v0.shape)
                fp[:] = v0[:]
                del fp
                v0 = np.load(vals_path, mmap_mode="r")
            except Exception as ex:
                log(f"spill write for level 0 failed: {ex}")
        levels.keys.append(np.asarray(k0))
        levels.vals.append(v0)
        levels.par.append(p0)
    for n in range(len(levels.keys)):
        counts.append(int(len(levels.keys[n])))
        catalan.append(int(math.comb(2 * n, n) // (n + 1)))
        timing_s.append(0.0)
    start = len(levels.keys)
    for n in range(start, nmax + 1):
        prev_count = int(len(levels.keys[n - 1]))
        resident = 0
        for kk in levels.keys:
            resident = resident + int(kk.nbytes)
        working = int(chunk) * 60
        itemsize = int(np.dtype(DT).itemsize)
        level_bytes = int(3 * prev_count) * (8 + itemsize + 12)
        resident_total = resident + int(1.6 * level_bytes) + working
        ram_limit = float(ram_gb) * float(1024 ** 3)
        if float(resident_total) > ram_limit:
            guard = f"ram guard at size {n}: need {resident_total} bytes > {ram_limit:.0f}"
            log(guard)
            break
        if spill_dir is not None:
            try:
                itemsize = int(np.dtype(DT).itemsize)
                need_disk = int(3 * prev_count * (itemsize + 12))
                free_disk = int(shutil.disk_usage(spill_dir).free)
                if need_disk > free_disk:
                    guard = f"disk guard at size {n}: need {need_disk} bytes, free {free_disk}"
                    log(guard)
                    break
            except Exception as ex:
                log(f"disk check failed at size {n}: {ex}")
        t_start = time.time()
        pending_keys: list[np.ndarray] = []
        pending_vals: list[np.ndarray] = []
        pending_pars: list[np.ndarray] = []
        pending_total = 0
        try:
            for a in range(n):
                b = n - 1 - a
                va = levels.vals[a]
                vb = levels.vals[b]
                len_a = int(len(va))
                len_b = int(len(vb))
                if len_a == 0:
                    continue
                if len_b == 0:
                    continue
                if len_b >= int(chunk):
                    step_a = 1
                else:
                    step_a = max(1, int(int(chunk) // max(1, len_b)))
                    if step_a > len_a:
                        step_a = len_a
                for a_start in range(0, len_a, step_a):
                    a_end = a_start + step_a
                    if a_end > len_a:
                        a_end = len_a
                    a_slice = va[a_start:a_end]
                    cur_a = a_end - a_start
                    if cur_a == 0:
                        continue
                    step_b = max(1, int(int(chunk) // max(1, cur_a)))
                    if step_b > len_b:
                        step_b = len_b
                    for b_start in range(0, len_b, step_b):
                        b_end = b_start + step_b
                        if b_end > len_b:
                            b_end = len_b
                        b_slice = vb[b_start:b_end]
                        cur_b = b_end - b_start
                        if cur_b == 0:
                            continue
                        a_rep = np.repeat(np.asarray(a_slice), cur_b)
                        b_tile = np.tile(np.asarray(b_slice), cur_a)
                        ia_base = np.arange(a_start, a_end, dtype=np.int64)
                        ib_base = np.arange(b_start, b_end, dtype=np.int64)
                        ia = np.repeat(ia_base, cur_b)
                        ib = np.tile(ib_base, cur_a)
                        v, ok = eml_vec(a_rep, b_tile)
                        del a_rep
                        del b_tile
                        if int(np.count_nonzero(ok)) == 0:
                            continue
                        v_ok = v[ok]
                        ia_ok = ia[ok]
                        ib_ok = ib[ok]
                        del v
                        del ia
                        del ib
                        k = qkey(v_ok)
                        k_u, first = np.unique(k, return_index=True)
                        v_u = v_ok[first]
                        ia_u = ia_ok[first]
                        ib_u = ib_ok[first]
                        del k
                        del v_ok
                        del ia_ok
                        del ib_ok
                        sizes, _pos = levels.lookup(k_u)
                        keep_global = sizes == -1
                        if int(np.count_nonzero(keep_global)) == 0:
                            continue
                        k_f = k_u[keep_global]
                        v_f = v_u[keep_global]
                        ia_f = ia_u[keep_global]
                        ib_f = ib_u[keep_global]
                        del k_u
                        del v_u
                        del ia_u
                        del ib_u
                        par_full = np.empty((len(k_f), 3), dtype=np.int32)
                        par_full[:, 0] = np.int32(a)
                        par_full[:, 1] = ia_f.astype(np.int32)
                        par_full[:, 2] = ib_f.astype(np.int32)
                        for acc in pending_keys:
                            if len(k_f) == 0:
                                break
                            if len(acc) == 0:
                                continue
                            pos = np.searchsorted(acc, k_f)
                            valid = pos < len(acc)
                            dup = np.zeros(len(k_f), dtype=bool)
                            vidx = np.nonzero(valid)[0]
                            if len(vidx) > 0:
                                eq = acc[pos[vidx]] == k_f[vidx]
                                dup[vidx[eq]] = True
                            keep = ~dup
                            k_f = k_f[keep]
                            v_f = v_f[keep]
                            par_full = par_full[keep]
                        if len(k_f) == 0:
                            continue
                        pending_keys.append(k_f)
                        pending_vals.append(v_f)
                        pending_pars.append(par_full)
                        pending_total = pending_total + len(k_f)
            if spill_dir is not None:
                vals_path = os.path.join(spill_dir, f"{branch}_vals_{n}.npy")
                par_path = os.path.join(spill_dir, f"{branch}_par_{n}.npy")
            else:
                vals_path = None
                par_path = None
            k_final, v_stored, p_stored = _finalize_level(pending_keys, pending_vals, pending_pars, vals_path, par_path, log)
        except Exception as ex:
            log(f"expansion failed at size {n}: {ex}")
            guard = f"expansion failed at size {n}: {ex}"
            break
        levels.keys.append(np.asarray(k_final))
        levels.vals.append(v_stored)
        levels.par.append(p_stored)
        counts.append(int(len(k_final)))
        catalan.append(int(math.comb(2 * n, n) // (n + 1)))
        elapsed_level = float(time.time() - t_start)
        timing_s.append(elapsed_level)
        total_elapsed = float(time.time() - t0)
        cum = 0
        for kk in levels.keys:
            cum = cum + int(len(kk))
        ratio = float(len(k_final)) / float(catalan[-1]) if catalan[-1] != 0 else 0.0
        log(f"size {n:2d}: new {len(k_final):>10d}  catalan {catalan[-1]:>12d}  ratio {ratio:.4f}  cum {cum:>10d}  {total_elapsed:6.1f}s")
    nmax_reached = int(levels.N)
    try:
        peak_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak_rss_mb = float(peak_kb / 1024.0)
    except Exception as ex:
        log(f"peak rss check failed: {ex}")
        peak_rss_mb = None
    info = {}
    info["counts"] = counts
    info["catalan"] = catalan
    info["timing_s"] = timing_s
    info["peak_rss_mb"] = peak_rss_mb
    info["nmax_reached"] = nmax_reached
    info["guard"] = guard
    return levels, info


def selfcheck(levels: Levels, chunk: int = 2000000, log=print) -> int:
    """Recompute stored values from parents and count mismatches."""
    bad = 0
    try:
        for n in range(1, int(levels.N) + 1):
            par_n = levels.par[n]
            if par_n is None:
                continue
            if len(par_n) == 0:
                continue
            vals_n = levels.vals[n]
            total = int(len(par_n))
            for start in range(0, total, int(chunk)):
                end = start + int(chunk)
                if end > total:
                    end = total
                pc = np.asarray(par_n[start:end])
                vc = np.asarray(vals_n[start:end])
                a_col = pc[:, 0].astype(np.int64)
                for aa in np.unique(a_col):
                    aa_int = int(aa)
                    mask = a_col == aa
                    bb_int = int(n - 1 - aa_int)
                    ia_m = pc[mask, 1].astype(np.int64)
                    ib_m = pc[mask, 2].astype(np.int64)
                    va = levels.vals[aa_int]
                    vb = levels.vals[bb_int]
                    a_vals = np.asarray(va)[ia_m]
                    b_vals = np.asarray(vb)[ib_m]
                    v, ok = eml_vec(a_vals, b_vals)
                    v_exp = vc[mask]
                    denom = np.maximum(1.0, np.abs(v_exp).astype(np.float64))
                    diff = np.abs(v - v_exp) / denom
                    bad = bad + int(np.count_nonzero(~ok)) + int(np.count_nonzero(diff > 1e-9))
    except Exception as ex:
        log(f"selfcheck failed: {ex}")
    return int(bad)


def targets(branch: str) -> dict[str, sp.Expr]:
    """Return the target dictionary for the branch."""
    e_val = sp.E
    pi_val = sp.pi
    i_val = sp.I
    out: dict[str, sp.Expr] = {}
    for k in range(-6, 25):
        out[str(k)] = sp.Integer(k)
    out["1/2"] = sp.Rational(1, 2)
    out["1/3"] = sp.Rational(1, 3)
    out["2/3"] = sp.Rational(2, 3)
    out["3/2"] = sp.Rational(3, 2)
    out["e"] = e_val
    out["pi"] = pi_val
    out["sqrt2"] = sp.sqrt(2)
    out["ln2"] = sp.log(2)
    out["phi"] = (1 + sp.sqrt(5)) / 2
    out["e^2"] = e_val ** 2
    out["1/e"] = 1 / e_val
    out["e-1"] = e_val - 1
    out["e-2"] = e_val - 2
    out["e-3"] = e_val - 3
    out["e-4"] = e_val - 4
    out["sqrt(e)"] = sp.sqrt(e_val)
    out["ln pi"] = sp.log(pi_val)
    out["2e"] = 2 * e_val
    out["e/2"] = e_val / 2
    out["e^e"] = e_val ** e_val
    out["ln(e-1)"] = sp.log(e_val - 1)
    out["-e"] = -e_val
    out["1/4"] = sp.Rational(1, 4)
    out["e^3"] = e_val ** 3
    if branch == "complex":
        out["i"] = i_val
        out["-i"] = -i_val
        out["i*pi"] = i_val * pi_val
        out["-i*pi"] = -i_val * pi_val
        out["1+i"] = 1 + i_val
        out["e^i"] = sp.exp(i_val)
        out["2i"] = 2 * i_val
        out["e^(ipi/2)?"] = sp.exp(i_val * pi_val / 2)
    return out


def frontier_lookup(levels: Levels, target_complex) -> tuple[int, str] | None:
    """Quantise the target and return the first level hit with witness."""
    try:
        tc = complex(target_complex)
        if levels.branch == "real":
            if abs(tc.imag) > 0:
                return None
            arr = np.array([float(tc.real)], dtype=np.float64)
        else:
            arr = np.array([tc], dtype=np.complex128)
        key = int(qkey(arr)[0])
        for n in range(int(levels.N) + 1):
            kn = levels.keys[n]
            if len(kn) == 0:
                continue
            pos = int(np.searchsorted(kn, np.int64(key)))
            if pos < len(kn) and int(kn[pos]) == key:
                wit = levels.witness(n, pos)
                return (int(n), wit)
        return None
    except Exception:
        return None


def mult_shaped(witness: str) -> bool | None:
    """Heuristic: does some node of the witness compute a product of two of its descendants?

    Multiplication in EML is x*y = exp(ln x + ln y), so a product node has both
    factors below it. Requiring the factors to be descendants (not just any two
    subtrees) rules out the tautology a = x * (a/x) that fires whenever a
    quotient node exists elsewhere in the tree. Factors within 1e-9 of 0, 1,
    -1 or e are ignored. A True here is a signal for the linear-to-logarithmic
    integer-cost crossover, not a proof of it. None when the string does not parse.
    """
    try:
        s = str(witness)
        nodes: list[tuple[int, int]] = []

        def parse_at(pos):
            if pos < len(s) and s[pos] == "1":
                nodes.append((-1, -1))
                return len(nodes) - 1, pos + 1
            if s[pos:pos + 2] != "e(":
                raise ValueError("unexpected token")
            left_id, p1 = parse_at(pos + 2)
            if s[p1] != ",":
                raise ValueError("expected comma")
            right_id, p2 = parse_at(p1 + 1)
            if s[p2] != ")":
                raise ValueError("expected close")
            nodes.append((left_id, right_id))
            return len(nodes) - 1, p2 + 1

        root_id, next_pos = parse_at(0)
        if next_pos != len(s):
            return None
        values: list[complex] = []
        below: list[set] = []
        with np.errstate(all="ignore"):
            for left_id, right_id in nodes:
                if left_id < 0:
                    values.append(1.0 + 0j)
                    below.append(set())
                    continue
                val = np.exp(np.complex128(values[left_id])) - np.log(np.complex128(values[right_id]))
                values.append(complex(val))
                below.append({left_id, right_id} | below[left_id] | below[right_id])
        base = (0.0, 1.0, -1.0, float(np.e))

        def trivial(x):
            return any(abs(x - b) <= 1e-9 * max(1.0, abs(x)) for b in base)

        for pid, desc in enumerate(below):
            pv = values[pid]
            if not desc or not np.isfinite(pv.real) or not np.isfinite(pv.imag) or trivial(pv):
                continue
            cands = [d for d in desc if np.isfinite(values[d].real) and np.isfinite(values[d].imag) and not trivial(values[d])]
            for i, xid in enumerate(cands):
                for yid in cands[i + 1:]:
                    prod = values[xid] * values[yid]
                    if abs(pv - prod) <= 1e-9 * max(1.0, abs(pv), abs(prod)):
                        return True
        return False
    except Exception:
        return None


def parse_witness(witness: str):
    """Parse "e(<l>,<r>)" / "1" into nested tuples; None if the string is malformed."""
    s = str(witness)

    def parse_at(pos):
        if pos < len(s) and s[pos] == "1":
            return None, pos + 1
        if s[pos:pos + 2] != "e(":
            raise ValueError("unexpected token")
        left, p1 = parse_at(pos + 2)
        if s[p1] != ",":
            raise ValueError("expected comma")
        right, p2 = parse_at(p1 + 1)
        if s[p2] != ")":
            raise ValueError("expected close")
        return (left, right), p2 + 1

    try:
        tree, end = parse_at(0)
    except (ValueError, IndexError):
        return None
    if end != len(s):
        return None
    return tree


def exact_verifier(target_sym, dps: int = 40, tol: float = 1e-25):
    """Return a callable witness -> bool that evaluates the tree at `dps` digits with mpmath.

    The 11-digit keys that make enumeration tractable also make coincidences
    common once a join tests 1e8 candidates against 1e8 classes: on the real
    size-19 cache, every root-join hit above size 23 was such a coincidence.
    A tree is accepted only if it matches the target to `tol` at `dps` digits.
    """
    import mpmath as mp
    mp.mp.dps = dps
    tv = sp.N(target_sym, dps)
    target = mp.mpc(str(sp.re(tv)), str(sp.im(tv)))
    scale = max(1.0, float(abs(target)))

    def ev(node):
        if node is None:
            return mp.mpc(1)
        left, right = node
        return mp.exp(ev(left)) - mp.log(ev(right))

    def check(witness: str) -> bool:
        tree = parse_witness(witness)
        if tree is None and witness != "1":
            return False
        try:
            val = ev(tree)
        except (ValueError, ZeroDivisionError, OverflowError):
            return False
        return bool(abs(val - target) < tol * scale)

    return check


def verify(witness: str, target_sym) -> bool:
    """Build the witness with sympy and check it against the target."""
    try:
        expr_str = str(witness).replace("e(", "eml(")
        env = {"__builtins__": {}, "eml": lambda a, b: sp.exp(a) - sp.log(b)}
        expr = eval(expr_str, env)
        diff = complex(sp.N(expr - target_sym, 30))
        return bool(abs(diff) < 1e-20)
    except Exception:
        return False


JOIN2_TARGETS = "4,5,-3,e-4,1/3,1/4,3/2,2/3,pi,sqrt2,phi,ln pi,i,1+i,e^i"


def _parse_count(s: str) -> int:
    """Parse an int that may use float scientific notation."""
    return int(float(s))


def main(argv=None) -> None:
    """CLI entry point for frontier expansion and target reporting."""
    wall_t0 = time.time()
    parser = argparse.ArgumentParser(description="EML complexity frontier")
    parser.add_argument("branch", nargs="?", default="real", choices=["real", "complex"])
    parser.add_argument("NMAX", nargs="?", type=int, default=18)
    parser.add_argument("--spill", default=None)
    parser.add_argument("--out", default="benchmarks/results/eml_complexity/")
    parser.add_argument("--chunk", type=_parse_count, default=20000000)
    parser.add_argument("--ram-gb", type=float, default=12.0)
    parser.add_argument("--no-join", action="store_true")
    parser.add_argument("--join2", type=int, default=8)
    parser.add_argument("--join2-budget", type=float, default=5e10)
    parser.add_argument("--join2-targets", default=JOIN2_TARGETS,
                        help="comma-separated target names for the two-level join (default: the issue #70 list); 'all' for every target")
    parser.add_argument("--no-selfcheck", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    branch = str(args.branch)
    nmax_req = int(args.NMAX)
    out_dir = str(args.out)
    chunk = int(args.chunk)
    ram_gb = float(args.ram_gb)
    if args.spill is None:
        spill_dir = os.path.join(out_dir, "_eml_spill")
    elif str(args.spill) == "none":
        spill_dir = None
    else:
        spill_dir = str(args.spill)
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as ex:
        print(f"could not create out dir {out_dir}: {ex}")
    set_branch(branch)
    levels, info = expand(branch, nmax_req, spill_dir=spill_dir, chunk=chunk, ram_gb=ram_gb, resume=bool(args.resume), log=print)
    nmax_reached = int(info["nmax_reached"])
    if bool(args.no_selfcheck):
        mismatches = None
    else:
        mismatches = int(selfcheck(levels, chunk=2000000, log=print))
        print(f"parent self-check mismatches: {mismatches}")
    sym_targets = targets(branch)
    target_complex: dict[str, complex] = {}
    for name, expr in sym_targets.items():
        try:
            target_complex[name] = complex(sp.N(expr, 20))
        except Exception as ex:
            print(f"target {name} numeric conversion failed: {ex}")
            target_complex[name] = None
    results: dict[str, dict | None] = {}
    for name in sym_targets.keys():
        tc = target_complex.get(name)
        if tc is None:
            results[name] = None
            continue
        try:
            hit = frontier_lookup(levels, tc)
        except Exception as ex:
            print(f"frontier lookup failed for {name}: {ex}")
            hit = None
        if hit is None:
            results[name] = None
        else:
            results[name] = {"size": int(hit[0]), "witness": str(hit[1]), "source": "frontier", "exact": True, "verified": False, "mult_shaped": None}
    if not bool(args.no_join):
        try:
            here = os.path.dirname(os.path.abspath(__file__))
        except Exception as ex:
            print(f"could not locate script dir: {ex}")
            here = None
        if here is not None and here not in sys.path:
            sys.path.insert(0, here)
        try:
            from eml_complexity_join import root_join
            from eml_complexity_join import two_level_join
            join_ok = True
        except Exception as ex:
            print(f"join import failed: {ex}")
            join_ok = False
        if join_ok:
            if str(args.join2_targets) == "all":
                join2_names = None
            else:
                join2_names = {t.strip() for t in str(args.join2_targets).split(",") if t.strip()}
            for name in list(sym_targets.keys()):
                if results[name] is not None:
                    continue
                tc = target_complex.get(name)
                if tc is None:
                    continue
                if branch == "real" and abs(complex(tc).imag) > 0:
                    continue
                try:
                    if branch == "complex":
                        arr = np.array([complex(tc)], dtype=np.complex128)
                    else:
                        arr = np.array([float(complex(tc).real)], dtype=np.float64)
                    tkey = int(qkey(arr)[0])
                except Exception as ex:
                    print(f"key computation failed for {name}: {ex}")
                    continue
                check = exact_verifier(sym_targets[name])
                try:
                    rj = root_join(levels, complex(tc), int(tkey), log=print, verify=check)
                except Exception as ex:
                    print(f"root join failed for {name}: {ex}")
                    rj = None
                if rj is not None:
                    exact = bool(int(rj[0]) <= int(nmax_reached) + 1)
                    results[name] = {"size": int(rj[0]), "witness": str(rj[1]), "source": "join", "exact": exact, "verified": False, "mult_shaped": None}
                    continue
                if join2_names is not None and name not in join2_names:
                    continue
                try:
                    r2 = two_level_join(levels, complex(tc), int(tkey), K=int(args.join2), budget=float(args.join2_budget), log=print, verify=check)
                except Exception as ex:
                    print(f"two level join failed for {name}: {ex}")
                    r2 = None
                if isinstance(r2, str):
                    print(f"two level join {name}: {r2}")
                    continue
                if r2 is not None:
                    exact2 = bool(int(r2[0]) <= int(nmax_reached) + 1)
                    results[name] = {"size": int(r2[0]), "witness": str(r2[1]), "source": "join2", "exact": exact2, "verified": False, "mult_shaped": None}
    for name, rec in list(results.items()):
        if rec is None:
            continue
        try:
            rec["verified"] = bool(verify(str(rec["witness"]), sym_targets[name]))
        except Exception as ex:
            print(f"verify failed for {name}: {ex}")
            rec["verified"] = False
        try:
            rec["mult_shaped"] = mult_shaped(str(rec["witness"]))
        except Exception as ex:
            print(f"mult_shaped failed for {name}: {ex}")
            rec["mult_shaped"] = None
    regression: list[dict] = []
    try:
        if branch == "real" and nmax_reached >= 18:
            known_real = [1, 1, 2, 5, 10, 27, 73, 197, 545, 1518, 4326, 12455, 36210, 106003, 311950, 921965, 2731279, 8117549, 24240152]
            for n, want in enumerate(known_real):
                got = int(info["counts"][n]) if n < len(info["counts"]) else None
                passed = bool(got == want)
                near = bool(got is not None and abs(got - want) <= max(2, 1e-4 * want))
                regression.append({"item": f"count_{n}", "pass": passed, "near": near, "got": got, "want": want})
                print(f"regression count_{n}: {'PASS' if passed else ('NEAR' if near else 'FAIL')} got {got} want {want}")
            known_sizes = {"0": 3, "-1": 8, "2": 9, "1/e": 9, "e^2": 10, "-e": 11, "e-3": 12, "ln2": 12, "-2": 13, "2e": 13, "3": 14, "e/2": 14, "e^3": 15, "1/2": 17, "sqrt(e)": 18}
            for tname, want in known_sizes.items():
                rec = results.get(tname)
                got = int(rec["size"]) if rec is not None else None
                passed = bool(got == want)
                regression.append({"item": f"size_{tname}", "pass": passed, "got": got, "want": want})
                print(f"regression size_{tname}: {'PASS' if passed else 'FAIL'} got {got} want {want}")
        if branch == "complex" and nmax_reached >= 16:
            known_cplx = [1, 1, 2, 5, 10, 28, 79, 228, 676, 2034, 6242, 19388, 60775, 192013, 610275, 1949611, 6253967]
            for n, want in enumerate(known_cplx):
                got = int(info["counts"][n]) if n < len(info["counts"]) else None
                passed = bool(got == want)
                near = bool(got is not None and abs(got - want) <= max(2, 1e-4 * want))
                regression.append({"item": f"count_{n}", "pass": passed, "near": near, "got": got, "want": want})
                print(f"regression count_{n}: {'PASS' if passed else ('NEAR' if near else 'FAIL')} got {got} want {want}")
            known_cplx_sizes = {"1/2": 15, "sqrt(e)": 16, "-i*pi": 11}
            for tname, want in known_cplx_sizes.items():
                rec = results.get(tname)
                got = int(rec["size"]) if rec is not None else None
                passed = bool(got == want)
                regression.append({"item": f"size_{tname}", "pass": passed, "got": got, "want": want})
                print(f"regression size_{tname}: {'PASS' if passed else 'FAIL'} got {got} want {want}")
    except Exception as ex:
        print(f"regression failed: {ex}")
    wall_s = float(time.time() - wall_t0)
    payload = {}
    payload["branch"] = branch
    payload["nmax_requested"] = int(nmax_req)
    payload["nmax_reached"] = int(nmax_reached)
    payload["counts"] = [int(x) for x in info["counts"]]
    payload["catalan"] = [int(x) for x in info["catalan"]]
    payload["targets"] = results
    payload["timing_s"] = [float(x) for x in info["timing_s"]]
    payload["peak_rss_mb"] = info["peak_rss_mb"]
    payload["wall_s"] = float(wall_s)
    payload["guard"] = info["guard"]
    payload["selfcheck_mismatches"] = mismatches
    payload["regression"] = regression
    json_path = os.path.join(out_dir, f"{branch}_{nmax_req}.json")
    try:
        with open(json_path, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"wrote {json_path}")
    except Exception as ex:
        print(f"output write failed for {json_path}: {ex}")
    found_rows = []
    missing_rows = []
    for name, rec in results.items():
        if rec is None:
            missing_rows.append(name)
        else:
            found_rows.append((name, rec))
    found_rows.sort(key=lambda kv: (int(kv[1]["size"]), str(kv[0])))
    missing_rows.sort()
    print("| constant | size | exact/bound | source | mult | witness |")
    print("|---|---|---|---|---|---|")
    for name, rec in found_rows:
        size = int(rec["size"])
        exact_str = "exact" if bool(rec["exact"]) else "bound"
        source = str(rec["source"])
        mult = rec["mult_shaped"]
        wit = str(rec["witness"])
        print(f"| {name} | {size} | {exact_str} | {source} | {mult} | {wit} |")
    for name in missing_rows:
        print(f"| {name} | not found <= {nmax_reached} | - | - | - | - |")


if __name__ == "__main__":
    main()

