"""Single time-point TIRTL-seq pairing pipeline (run_single_point_analysis_sub_gpu).

Same steps, parameters and output columns as the original R pipeline (TIRTL_functions.R) and the
previous Python port, with all heavy steps in-process (no subprocess, no text intermediates).
"""
from __future__ import annotations

import math
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import _timing
from ._native import core, have_rust
from ._timing import stage
from .backends import get_backend
from .clones import load_chain
from .collate import r_order
from .madhyper import estimate_pair_prob_batch as _default_score_fn
from .madhyper import madhyper_surface as _default_surface_fn
from .pairing import Occupancy, madhype, overlaps, tshell

OUTPUT_COLUMNS = ["alpha_nuc", "beta_nuc", "wij", "wa", "wb", "alpha_nuc_seq", "beta_nuc_seq", "alpha_beta",
                  "method", "r", "ts", "pval", "pval_adj", "loss_a_frac", "loss_b_frac", "wi", "wj", "score",
                  "cdr3a", "va", "ja", "cdr3b", "vb", "jb"]


def get_well_subset(row_range=range(1, 17), col_range=range(1, 25)):
    rows = [chr(ord("A") + int(r) - 1) for r in row_range]
    return [f"{r}{int(c)}" for r in rows for c in col_range]


def list_files_r(folder):
    """list.files(folder) of R: visible entries, R (ICU) collation order."""
    names = [n for n in os.listdir(folder) if not n.startswith(".")]
    return [names[i] for i in r_order(names)]


def _well_of(name, pos):
    parts = str(name).split("_")
    i = int(pos) - 1
    return parts[i] if 0 <= i < len(parts) else ""


def _log(*a):
    print(*a, flush=True)


def _betacf(a, b, x, maxit=300, eps=1e-17):
    """continued fraction of the regularised incomplete beta function (vectorised, modified Lentz)"""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = np.ones_like(x)
    d = 1.0 - qab * x / qap
    d = np.where(np.abs(d) < tiny, tiny, d)
    d = 1.0 / d
    h = d.copy()
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < tiny, tiny, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < tiny, tiny, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1.0 / d
        de = d * c
        h *= de
        if np.all(np.abs(de - 1.0) < eps):
            break
    return h


def _pt_two_sided(ts, df):
    """2 * pt(-|t|, df) as in R.

    scipy's Student-t CDF flushes to 0 around 1e-300 whereas R's pt() continues into the subnormal
    range (e.g. 5.4e-319 for t=150, df=349). Since R then uses pval / sort(pval)[3], the far tail is
    computed like R's pt(): I_{1/(1+t^2/df)}(df/2, 1/2), in log scale.
    """
    from scipy.special import stdtr
    ts = np.asarray(ts, dtype=np.float64)
    out = 2.0 * stdtr(df, -np.abs(ts))
    tail = np.isfinite(ts) & (out < 1e-280)
    if tail.any():
        t = np.abs(ts[tail])
        a, b = df / 2.0, 0.5
        x = 1.0 / (1.0 + (t / df) * t)
        lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        with np.errstate(divide="ignore", under="ignore", over="ignore", invalid="ignore"):
            logv = a * np.log(x) + b * np.log1p(-x) - lbeta + np.log(_betacf(a, b, x)) - math.log(a)
            # R: pt() returns val/2 and the caller doubles it; in the subnormal range the halving
            # rounds (the smallest subnormal becomes 0)
            out[tail] = 2.0 * (np.exp(logv) / 2.0)
    out[np.isinf(ts)] = 0.0
    return out


class Plate:
    """Clone tables of one folder split by chain (R: mlista / mlistb)."""

    def __init__(self, folder):
        self.folder = Path(folder)
        files = list_files_r(self.folder)
        names = [f.replace(".", "_") for f in files]
        self.a_idx = [i for i, n in enumerate(names) if "TRA" in n]
        self.b_idx = [i for i, n in enumerate(names) if "TRB" in n]
        self.a_names = [names[i] for i in self.a_idx]
        self.b_names = [names[i] for i in self.b_idx]
        with stage("load"):
            self.alpha = load_chain([self.folder / files[i] for i in self.a_idx])
            self.beta = load_chain([self.folder / files[i] for i in self.b_idx])
        self.a_nrow = np.asarray(self.alpha.nrows())
        self.b_nrow = np.asarray(self.beta.nrows())


def good_wells(plate, thres, pos, wellset):
    """get_good_wells_sub()"""
    wa = [_well_of(n, pos) for n in plate.a_names]
    wb = [_well_of(n, pos) for n in plate.b_names]
    ga = [w for w, n in zip(wa, plate.a_nrow) if n > thres]
    gb = set(w for w, n in zip(wb, plate.b_nrow) if n > thres)
    good = [w for w in dict.fromkeys(ga) if w in gb]
    ws = set(wellset)
    keep = [w for w in good if w in ws]
    ks = set(keep)
    return {"a": np.array([w in ks for w in wa]), "b": np.array([w in ks for w in wb]), "well_ids": keep}


def _pseudobulk(chain, file_idx, max_wells, path):
    c = core()
    if c is not None and not str(path).endswith(".gz"):
        chain.write_pseudobulk(list(file_idx), int(max_wells), str(path))
        return
    pb = chain.pseudobulk(list(file_idx))
    seq = chain.seq_strings(np.asarray(pb["seq"], dtype=np.uint32))
    genes = chain.gene_strings()
    aas = list(pb["aaSeqCDR3"])
    n = len(seq)
    cols = [("targetSequences", ("s", np.arange(n, dtype=np.int64), list(seq))),
            ("readCount", ("f", pb["readCount"])), ("v", ("s", pb["v"], genes)), ("j", ("s", pb["j"], genes)),
            ("aaSeqCDR3", ("s", np.arange(n, dtype=np.int64), aas)), ("n_wells", ("i", pb["n_wells"].astype(np.int64))),
            ("readCount_max", ("f", pb["readCount_max"])), ("readCount_median", ("f", pb["readCount_median"])),
            ("avg", ("f", pb["avg"])), ("sem", ("f", pb["sem"])), ("readFraction", ("f", pb["readFraction"])),
            ("max_wells", ("i", np.full(n, int(max_wells), dtype=np.int64)))]
    _frame_from_cols(cols).to_csv(path, sep="\t", index=False)


def _arrow_strings_enabled():
    """pandas >= 3 (or future.infer_string) stores strings in Arrow: build them directly"""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    if int(pd.__version__.split(".")[0]) >= 3:
        return True
    try:
        return bool(pd.get_option("future.infer_string"))
    except Exception:  # noqa: BLE001
        return False


def _arrow_col(c, spec):
    import pyarrow as pa
    if spec[0] == "s":
        off, data, valid = c.arrow_strings(np.ascontiguousarray(spec[1], np.int64), list(spec[2]))
    else:
        off, data, valid = c.arrow_strings(np.ascontiguousarray(spec[1], np.int64), list(spec[2]),
                                           np.ascontiguousarray(spec[3], np.int64), list(spec[4]), spec[5])
    n = len(valid)
    nulls = n - int(valid.sum())
    bitmap = pa.py_buffer(np.packbits(valid, bitorder="little")) if nulls else None
    arr = pa.LargeStringArray.from_buffers(n, pa.py_buffer(off), pa.py_buffer(data), bitmap, nulls)
    try:
        return pd.array(arr, dtype=pd.StringDtype("pyarrow", na_value=np.nan))
    except TypeError:  # pandas < 2.1
        return pd.array(arr, dtype="string[pyarrow]")


def _frame_from_cols(cols):
    data = {}
    c = core()
    use_arrow = c is not None and _arrow_strings_enabled()
    cache = {}
    with warnings_suppressed():
        for name, spec in cols:
            kind = spec[0]
            if kind in ("f", "i"):
                data[name] = spec[1]
                continue
            key = (kind, id(spec[1]), id(spec[2]))  # same codes + dict -> reuse the column
            if key in cache:
                data[name] = cache[key].copy() if not use_arrow else cache[key]
                continue
            if use_arrow:
                col = _arrow_col(c, spec)
            elif kind == "s":
                d = np.array(list(spec[2]) + [np.nan], dtype=object)
                col = d[np.where(spec[1] < 0, len(spec[2]), spec[1])]
            elif c is not None:
                col = np.array(c.concat_strings(spec[1], list(spec[2]), spec[3], list(spec[4]), spec[5]), dtype=object)
            else:
                d1 = np.array(list(spec[2]), dtype=object)
                d2 = np.array(list(spec[4]), dtype=object)
                col = d1[spec[1]] + spec[5] + d2[spec[3]]
            cache[key] = col
            data[name] = col
    return pd.DataFrame(data)


def warnings_suppressed():
    import warnings
    cm = warnings.catch_warnings()
    cm.__enter__()
    warnings.simplefilter("ignore")

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            cm.__exit__(*a)
    return _Ctx()


def _read_backend_results(prefix):
    """compute=False: results of a separately-run backend script (original file formats)."""
    md = pd.read_csv(f"{prefix}_madhyperesults.csv")
    co = pd.read_csv(f"{prefix}_corresults.csv")
    return md, co


def run_single_point_analysis_sub_gpu(
        folder_path, prefix="tmp", well_filter_thres=0.5, min_reads=0, min_wells=2, well_pos=3,
        wellset1=None, compute=True, backend="numpy", pval_thres_tshell=1e-10, wij_thres_tshell=2,
        pseudobulk=False, cpw=None, write_intermediates=False, return_frame=True,
        surface_fn=None, score_fn=None, engine="auto"):
    """Pair TCR alpha and beta chains of one plate with MAD-HYPE and T-SHELL.

    Parameters are those of the original R function. Additional (optional) ones:
      cpw: cells per well used by MAD-HYPE (default: the QC clone threshold, as in R)
      write_intermediates: also write {prefix}_bigmas.tsv ... in the original text formats
      surface_fn / score_fn: override madhyper_surface / estimate_pair_prob_batch
      engine: "sparse" (fused Rust kernel over shared wells, default when Rust is available) or
              "dense" (float32 products on `backend` = numpy/cupy/mlx, then exact float64 re-scoring)
    Writes {prefix}_TIRTLoutput.tsv and returns it as a pandas DataFrame.
    """
    _timing.reset()
    surface_fn = surface_fn or _default_surface_fn
    score_fn = score_fn or _default_score_fn
    if wellset1 is None:
        wellset1 = get_well_subset()
    with stage("total"):
        _log("start")
        _log(datetime.now().isoformat(timespec="seconds"), "| rust kernels:", have_rust(), "| backend:", backend)
        plate = Plate(folder_path)
        _log("clonesets loaded")
        _log(plate.a_names)
        wellsub = np.array([_well_of(n, well_pos) in set(wellset1) for n in plate.a_names])
        clone_thres = float(round(well_filter_thres * float(np.mean(plate.a_nrow[wellsub]))))
        qc = good_wells(plate, clone_thres, well_pos, wellset1)
        _log("clone_threshold for QC:", int(clone_thres))
        _log("alpha wells passing QC:", dict(pd.Series(qc["a"]).value_counts()))
        _log("beta wells passing QC:", dict(pd.Series(qc["b"]).value_counts()))
        qa = [int(i) for i in np.nonzero(qc["a"])[0]]
        qb = [int(i) for i in np.nonzero(qc["b"])[0]]

        if pseudobulk:
            with stage("pseudobulk"):
                _log("Merging pseudobulk alpha...")
                _pseudobulk(plate.alpha, qa, int(np.sum(qc["a"])), f"{prefix}_pseudobulk_TRA.tsv")
                _log("Merging pseudobulk beta...")
                _pseudobulk(plate.beta, qb, int(np.sum(qc["b"])), f"{prefix}_pseudobulk_TRB.tsv")

        with stage("merge"):
            _log("Merging alpha clonesets...")
            a_ids, bigmas, na_tot, a_cols = plate.alpha.merge(qa, float(min_reads), int(min_wells))
            _log("Done! Unique alpha clones and wells after filtering:", (na_tot, len(a_cols)),
                 f"| in more than {min_wells} wells:", bigmas.shape)
            _log("Merging beta clonesets...")
            b_ids, bigmbs, nb_tot, b_cols = plate.beta.merge(qb, float(min_reads), int(min_wells))
            _log("Done! Unique beta clones and wells after filtering:", (nb_tot, len(b_cols)),
                 f"| in more than {min_wells} wells:", bigmbs.shape)

        n_wells = bigmas.shape[1]
        cells = clone_thres if cpw is None else cpw
        if bigmas.shape[0] == 0 or bigmbs.shape[0] == 0:
            raise ValueError(f"no clone found in more than {min_wells} wells (alpha: {bigmas.shape[0]}, "
                             f"beta: {bigmbs.shape[0]}); check the input files / well_filter_thres")
        prior = 1 / (float(bigmas.shape[0]) * float(bigmbs.shape[0])) ** 0.5
        with stage("lookup_table"):
            _log("Pre-computing look-up table:")
            mdh = np.asarray(surface_fn(n_wells=n_wells, cells=cells, alpha=2, prior=prior))

        if write_intermediates:
            with stage("write_intermediates"):
                _write_intermediates(prefix, plate, a_ids, b_ids, bigmas, bigmbs, mdh)

        occ_a, occ_b = Occupancy(bigmas), Occupancy(bigmbs)
        eng = engine if engine != "auto" else ("sparse" if have_rust() else "dense")
        if compute and eng == "sparse":
            if not have_rust():
                raise RuntimeError("engine='sparse' needs the Rust extension (tirtl._core)")
            with stage("pairing_sparse"):
                ma, mb, mw, ta, tb, tr, tw = core().pair_sparse(
                    np.ascontiguousarray(bigmas), np.ascontiguousarray(bigmbs),
                    np.ascontiguousarray(mdh.astype(np.int64)), True, 2, 3)
            _log(f"Number of pairs from MAD-HYPE: {len(ma)}; T-SHELL candidate pairs: {len(ta)} (sparse engine)")
        elif compute:
            be = get_backend(backend)
            with stage("madhype"):
                ma, mb, mw = madhype(occ_a, occ_b, mdh, be)
                _log(f"Number of pairs from MAD-HYPE: {len(ma)}")
            with stage("tshell"):
                ta, tb, tr, nfb = tshell(bigmas, bigmbs, occ_a, occ_b, be)
                tw = overlaps(occ_a, occ_b, ta, tb)
                _log(f"T-SHELL: {len(ta)} candidate pairs ({nfb} alpha rows recomputed exactly)")
        else:
            md, co = _read_backend_results(prefix)
            ma, mb, mw = (md["alpha_nuc"].to_numpy() - 1).astype(np.uint32), (md["beta_nuc"].to_numpy() - 1).astype(np.uint32), md["wij"].to_numpy()
            ta, tb = (co["alpha_nuc"].to_numpy() - 1).astype(np.uint32), (co["beta_nuc"].to_numpy() - 1).astype(np.uint32)
            tr, tw = co["r"].to_numpy(dtype=float), co["wij"].to_numpy()

        _log("Loading and filtering results, adding amino acid and V segment information")
        with stage("postprocess"):
            res = _assemble(ma, mb, mw, ta, tb, tr, tw, occ_a, occ_b, n_wells, cells, prior, score_fn,
                            wij_thres_tshell, pval_thres_tshell)
        with stage("annotate"):
            cols = _annotate(res, plate, qa, qb, a_ids, b_ids)
        with stage("write_output"):
            out_path = f"{prefix}_TIRTLoutput.tsv"
            c = core()
            if c is not None:
                c.write_tsv(out_path, cols)
            frame = _frame_from_cols(cols) if (return_frame or c is None) else None
            if c is None:
                frame.to_csv(out_path, sep="\t", index=False)
        _log("All is done! Number of paired clones:")
        _log(dict(pd.Series(res["method"]).map({0: "madhype", 1: "tshell"}).value_counts()))
    return frame


def _assemble(ma, mb, mw, ta, tb, tr, tw, occ_a, occ_b, n_wells, cells, prior, score_fn, wij_thr, pval_thr):
    a_tot, b_tot = np.asarray(occ_a.total, np.int64), np.asarray(occ_b.total, np.int64)
    # ---- T-SHELL rows: read_gpu_corr() statistics, then the R filter
    ta64, tb64 = np.asarray(ta, np.int64), np.asarray(tb, np.int64)
    nt = len(ta64)
    t_wij = np.asarray(tw, np.float64)
    t_wa, t_wb = a_tot[ta64], b_tot[tb64]
    with np.errstate(divide="ignore", invalid="ignore"):
        rt = np.asarray(tr, np.float64)
        ts_t = rt * np.sqrt((n_wells - 2) / (1 - rt ** 2))
        pv_t = _pt_two_sided(ts_t, n_wells - 2)
        padj_t = np.full(nt, np.nan)
        if nt:
            # R: pval_adj := pval / sort(pval)[3] by alpha (NA when < 3 non-NA p-values)
            gid = np.r_[0, np.cumsum(np.diff(ta64) != 0)]  # rows are grouped by alpha
            nanp = np.isnan(pv_t)
            o = np.lexsort((np.where(nanp, np.inf, pv_t), nanp, gid))
            g_sorted = gid[o]
            first = np.r_[0, np.nonzero(np.diff(g_sorted))[0] + 1]
            pos = np.arange(nt) - np.repeat(first, np.diff(np.r_[first, nt]))
            third = np.full(gid[-1] + 1, np.nan)
            sel3 = (pos == 2) & ~nanp[o]
            third[g_sorted[sel3]] = pv_t[o][sel3]
            padj_t = pv_t / third[gid]
        t_den = t_wij + (t_wb - t_wij) + (t_wa - t_wij)
        t_la = (t_wb - t_wij) / t_den
        t_lb = (t_wa - t_wij) / t_den
    keep = (t_wij > wij_thr) & (padj_t < pval_thr) & ((t_la + t_lb) < 0.5)
    # ---- MAD-HYPE rows are all kept
    m_wij = np.asarray(mw, np.float64)
    ma64, mb64 = np.asarray(ma, np.int64), np.asarray(mb, np.int64)
    m_wa, m_wb = a_tot[ma64], b_tot[mb64]
    nm = len(ma64)
    with np.errstate(divide="ignore", invalid="ignore"):
        m_den = m_wij + (m_wb - m_wij) + (m_wa - m_wij)
        m_la = (m_wb - m_wij) / m_den
        m_lb = (m_wa - m_wij) / m_den
    cat = lambda m, t: np.concatenate([m, t[keep]])  # noqa: E731
    nan_m = np.full(nm, np.nan)
    res = dict(ai=cat(ma64, ta64), bi=cat(mb64, tb64), wij=cat(m_wij, t_wij), wa=cat(m_wa, t_wa), wb=cat(m_wb, t_wb),
               method=np.concatenate([np.zeros(nm, np.int8), np.ones(int(keep.sum()), np.int8)]),
               r=cat(nan_m, rt), ts=cat(nan_m, ts_t), pval=cat(nan_m, pv_t), pval_adj=cat(nan_m, padj_t),
               loss_a_frac=cat(m_la, t_la), loss_b_frac=cat(m_lb, t_lb))
    res["wi"] = res["wa"] - res["wij"]
    res["wj"] = res["wb"] - res["wij"]
    # ---- MAD-HYPE score for each distinct (wi, wj, wij); integers in [0, n_wells] -> one int64 key
    k_i, k_j, k_ij = (res[x].astype(np.int64) for x in ("wi", "wj", "wij"))
    key = (k_i << 42) | (k_j << 21) | k_ij
    ukey, inv = np.unique(key, return_inverse=True)
    u_i, u_j, u_ij = (ukey >> 42) & 0x1FFFFF, (ukey >> 21) & 0x1FFFFF, ukey & 0x1FFFFF
    probs = np.asarray(score_fn(wi=u_i.astype(np.float64), wj=u_j.astype(np.float64), w_ij=u_ij.astype(np.float64),
                                w_tot=n_wells, cpw=cells, alpha=2, prior=prior), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        score_u = np.log10(probs)
    res["score"] = score_u[np.asarray(inv).reshape(-1)]
    return res


def _annotate(res, plate, qa, qb, a_ids, b_ids):
    """output columns (TSV writer specs); strings are kept as dictionary codes"""
    cols = {}
    for side, chain, files, ids, key in (("a", plate.alpha, qa, a_ids, "ai"), ("b", plate.beta, qb, b_ids, "bi")):
        rows = res[key]
        uniq_rows, codes = np.unique(rows, return_inverse=True)
        seq_ids = np.asarray(ids, dtype=np.uint32)[uniq_rows]
        seqs = list(chain.seq_strings(seq_ids))
        aa, v, j = chain.annotate(list(files), seq_ids)
        aa_c = np.array([-1 if x is None else k for k, x in enumerate(aa)], dtype=np.int64)
        aa_dict = ["" if x is None else x for x in aa]
        genes = chain.gene_strings()
        codes = np.asarray(codes, np.int64).reshape(-1)
        cols[side] = dict(codes=codes, seqs=seqs, aa=(aa_c[codes], aa_dict), v=(np.asarray(v)[codes], genes),
                          j=(np.asarray(j)[codes], genes))
    A, B = cols["a"], cols["b"]
    f = lambda x: ("f", np.ascontiguousarray(x, dtype=np.float64))  # noqa: E731
    i64 = lambda x: ("i", np.ascontiguousarray(x, dtype=np.int64))  # noqa: E731
    s = lambda c, d: ("s", np.ascontiguousarray(c, dtype=np.int64), d)  # noqa: E731
    return [
        ("alpha_nuc", s(A["codes"], A["seqs"])), ("beta_nuc", s(B["codes"], B["seqs"])),
        ("wij", f(res["wij"])), ("wa", i64(res["wa"])), ("wb", i64(res["wb"])),
        ("alpha_nuc_seq", s(A["codes"], A["seqs"])), ("beta_nuc_seq", s(B["codes"], B["seqs"])),
        ("alpha_beta", ("c", A["codes"], A["seqs"], B["codes"], B["seqs"], "_")),
        ("method", s(res["method"].astype(np.int64), ["madhype", "tshell"])),
        ("r", f(res["r"])), ("ts", f(res["ts"])), ("pval", f(res["pval"])), ("pval_adj", f(res["pval_adj"])),
        ("loss_a_frac", f(res["loss_a_frac"])), ("loss_b_frac", f(res["loss_b_frac"])),
        ("wi", f(res["wi"])), ("wj", f(res["wj"])), ("score", f(res["score"])),
        ("cdr3a", s(*A["aa"])), ("va", s(*A["v"])), ("ja", s(*A["j"])),
        ("cdr3b", s(*B["aa"])), ("vb", s(*B["v"])), ("jb", s(*B["j"])),
    ]


def _write_intermediates(prefix, plate, a_ids, b_ids, bigmas, bigmbs, mdh):
    """original text intermediates (for the standalone backend scripts / R)"""
    Path(f"{prefix}_bigmas_names.tsv").write_text("\n".join(plate.alpha.seq_strings(a_ids)) + "\n")
    Path(f"{prefix}_bigmbs_names.tsv").write_text("\n".join(plate.beta.seq_strings(b_ids)) + "\n")
    np.savetxt(f"{prefix}_bigmas.tsv", bigmas, delimiter="\t", fmt="%.15g")
    np.savetxt(f"{prefix}_bigmbs.tsv", bigmbs, delimiter="\t", fmt="%.15g")
    np.savetxt(f"{prefix}_mdh.tsv", mdh, delimiter="\t", fmt="%.15g")
