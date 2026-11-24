"""MAD-HYPE and T-SHELL pair calling on the merged well-frequency matrices.

Both reproduce the original backend scripts (numpy_backend_script.py) except that T-SHELL
correlations are exact float64 (validated plan): candidates from float32 products on any backend,
then exact re-scoring with a proven error bound (rows that cannot be certified are recomputed
fully in float64).
"""
from __future__ import annotations

import numpy as np

from ._native import core
from ._timing import stage

TSHELL_TOP = 3
TSHELL_CANDIDATES = 12  # float32 candidates per alpha (>= TOP + 1)


# --------------------------------------------------------------------------- helpers (fallbacks)
def _overlaps_np(a_bin, b_bin, ai, bi):
    return np.einsum("ij,ij->i", a_bin[ai].astype(np.int32), b_bin[bi].astype(np.int32)).astype(np.uint32)


def _normalize_np(m):
    # sequential sums (np.cumsum) = same rounding as the Rust kernel
    w = m.shape[1]
    mean = np.cumsum(m, axis=1)[:, -1:] / w if w else np.zeros((m.shape[0], 1))
    c = m - mean
    with np.errstate(invalid="ignore", divide="ignore"):
        return c / np.sqrt(np.cumsum(c * c, axis=1)[:, -1:])


def _dot_fixed(av, bm):
    """float64 dot products with the same fixed summation order as the Rust kernel."""
    w = av.shape[-1]
    k4 = (w // 4) * 4
    prod = bm * av
    s = np.zeros((bm.shape[0], 4))
    for k in range(0, k4, 4):
        s += prod[:, k:k + 4]
    for k in range(k4, w):
        s[:, 0] += prod[:, k]
    return (s[:, 0] + s[:, 1]) + (s[:, 2] + s[:, 3])


def _rank_top(r, idx, top):
    """indices of the `top` best (r desc, NaN last, idx asc)"""
    key_nan = np.isnan(r)
    order = np.lexsort((idx, -np.where(key_nan, -np.inf, r), key_nan))
    return order[:top]


def _refine_np(anorm, bnorm, rows, cand_idx, cand_val, bound, top):
    nb = bnorm.shape[0]
    oa, ob, orr, nfb = [], [], [], 0
    for t, ar in enumerate(rows):
        av = anorm[ar]
        ci = cand_idx[t]
        r = _dot_fixed(av, bnorm[ci])
        sel = _rank_top(r, ci, top)
        best_r, best_j = r[sel], ci[sel]
        ok = len(ci) >= nb
        if not ok and len(sel) == top:
            thr = np.nanmin(cand_val[t].astype(np.float64)) + bound
            ok = (not np.isnan(best_r[-1])) and np.isfinite(thr) and best_r[-1] > thr
        if not ok:
            nfb += 1
            allj = np.arange(nb)
            r = _dot_fixed(av, bnorm)
            sel = _rank_top(r, allj, top)
            best_r, best_j = r[sel], allj[sel]
        o = np.argsort(best_j, kind="stable")
        oa.append(np.full(len(o), ar, dtype=np.uint32))
        ob.append(best_j[o].astype(np.uint32))
        orr.append(best_r[o])
    cat = lambda x, d: np.concatenate(x).astype(d) if x else np.zeros(0, d)  # noqa: E731
    return cat(oa, np.uint32), cat(ob, np.uint32), cat(orr, np.float64), nfb


def f32_dot_bound(w):
    u = 2.0 ** -24
    n = w + 4.0
    return 2.0 * (n * u / (1.0 - n * u)) + 1e-7


# --------------------------------------------------------------------------- MAD-HYPE
class Occupancy:
    """Well occupancy of a frequency matrix (value > 0 as float32, like the original backend)."""

    def __init__(self, mat):
        c = core()
        self.n = mat.shape[0]
        if c is not None:
            self.bits = c.Bitsets(np.ascontiguousarray(mat, dtype=np.float64))
            self.total = self.bits.totals()
            self.binary = None
        else:
            self.bits = None
            self.binary = mat.astype(np.float32) > 0
            self.total = self.binary.sum(axis=1).astype(np.int64)

    def get_binary(self, rows=None):
        if self.binary is None:
            raise RuntimeError("binary matrix not kept with the Rust backend")
        return self.binary if rows is None else self.binary[rows]


def overlaps(occ_a, occ_b, ai, bi):
    c = core()
    if c is not None and occ_a.bits is not None:
        return c.overlaps(occ_a.bits, occ_b.bits, np.ascontiguousarray(ai, np.uint32), np.ascontiguousarray(bi, np.uint32))
    return _overlaps_np(occ_a.binary, occ_b.binary, ai, bi)


def madhype(occ_a, occ_b, mdh, backend):
    """pairs with wj < mdh[wij, wi] -> (alpha rows, beta rows, wij) sorted by (alpha, beta)"""
    c = core()
    mdh_i = np.ascontiguousarray(np.asarray(mdh).astype(np.int64))
    if c is not None and occ_a.bits is not None:
        return c.madhype_pairs(occ_a.bits, occ_b.bits, mdh_i)
    # dense fallback on the array backend (float32 products are exact counts)
    xb = backend.to_dev(occ_b.binary.T.astype(np.float32))
    b_tot = occ_b.total
    out_a, out_b, out_w = [], [], []
    step = backend.chunk_rows
    for ch in range(0, occ_a.n, step):
        sl = slice(ch, min(ch + step, occ_a.n))
        ov = backend.to_host(backend.matmul(backend.to_dev(occ_a.binary[sl].astype(np.float32)), xb)).astype(np.int64)
        wi = occ_a.total[sl][:, None] - ov
        wj = b_tot[None, :] - ov
        ia, ib = np.nonzero(wj < mdh_i[ov, wi])
        out_a.append(ia + ch)
        out_b.append(ib)
        out_w.append(ov[ia, ib])
    cat = lambda x: np.concatenate(x) if x else np.zeros(0, np.int64)  # noqa: E731
    return cat(out_a).astype(np.uint32), cat(out_b).astype(np.uint32), cat(out_w).astype(np.uint32)


# --------------------------------------------------------------------------- T-SHELL
def tshell(bigmas, bigmbs, occ_a, occ_b, backend, min_wells=2, top=TSHELL_TOP, n_cand=TSHELL_CANDIDATES):
    """Top-`top` Pearson correlations per alpha (exact float64).

    Like the original backend, only rows present in more than `min_wells` (=2, hard-coded in the
    original script) wells take part. Returns (alpha rows, beta rows, r) in global row indices.
    """
    c = core()
    rows_a = np.nonzero(occ_a.total > min_wells)[0]
    rows_b = np.nonzero(occ_b.total > min_wells)[0]
    w = bigmas.shape[1]
    if len(rows_a) == 0 or len(rows_b) == 0:
        z = np.zeros(0, np.uint32)
        return z, z, np.zeros(0), 0
    with stage("tshell_normalize"):
        if c is not None:
            an = c.normalize_rows(np.ascontiguousarray(bigmas[rows_a]))
            bn = c.normalize_rows(np.ascontiguousarray(bigmbs[rows_b]))
        else:
            an, bn = _normalize_np(bigmas[rows_a]), _normalize_np(bigmbs[rows_b])
        a32 = an.astype(np.float32)
        b32_t = backend.to_dev(np.ascontiguousarray(bn.astype(np.float32).T))
    bound = f32_dot_bound(w)
    k = min(n_cand, len(rows_b))
    oa, ob, orr, nfb = [], [], [], 0
    step = backend.chunk_rows
    for ch in range(0, len(rows_a), step):
        sl = np.arange(ch, min(ch + step, len(rows_a)), dtype=np.uint32)
        with stage("tshell_matmul_topk"):
            s = backend.matmul(backend.to_dev(a32[ch:ch + len(sl)]), b32_t)
            ci, cv = backend.topk(s, k)
            del s
        with stage("tshell_refine"):
            if c is not None:
                a_, b_, r_, f_ = c.tshell_refine(an, bn, sl, np.ascontiguousarray(ci, np.int64),
                                                 np.ascontiguousarray(cv, np.float32), bound, top)
            else:
                a_, b_, r_, f_ = _refine_np(an, bn, sl, ci, cv, bound, top)
        oa.append(rows_a[a_])
        ob.append(rows_b[b_])
        orr.append(r_)
        nfb += f_
    backend.free()
    return (np.concatenate(oa).astype(np.uint32), np.concatenate(ob).astype(np.uint32),
            np.concatenate(orr), nfb)
