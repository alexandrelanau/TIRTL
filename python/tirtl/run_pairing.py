"""TIRTLtools::run_pairing(): pairing of one sample over one or several plates.

Faithful to TIRTLtools 0.2.9 (R/run_pairing.R, R/utils_run_pairing.R, backend "cpp"):
well listing and order, multi-plate well names ("<well>_<plate>"), QC thresholds, pseudobulk,
read-fraction matrices (rows by total reads), MAD-HYPE / T-SHELL (double precision, same
tie-breaking), scoring, T-SHELL filter, annotation, optional best-pair selection, output columns
and row order. Heavy steps run in Rust (tirtl._core) with a numpy fallback.
"""
from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import _timing
from ._native import core, have_rust
from ._timing import stage
from .clones import load_chain
from .collate import r_order
from .madhyper import estimate_pair_prob_batch, madhyper_surface
from .pipeline import _annotate, _arrow_strings_enabled, _frame_from_cols, _pseudobulk, _pt_two_sided, get_well_subset

TSHELL_AUTO_MIN_WELLS = 150

OUTPUT_COLUMNS = ["wi", "wj", "wij", "alpha_nuc", "beta_nuc", "wa", "wb", "alpha_nuc_seq", "beta_nuc_seq",
                  "alpha_beta", "method", "r", "ts", "pval", "pval_adj", "loss_a_frac", "loss_b_frac", "score",
                  "cdr3a", "va", "ja", "cdr3b", "vb", "jb"]


def get_tshell_settings(format="auto"):
    """T-SHELL thresholds: "384_well" (p < 1e-10, wij > 2), "96_well" (p < 1e-3, wij > 3), "auto" (None)."""
    if format == "384_well":
        return {"pval_thres_tshell": 1e-10, "wij_thres_tshell": 2}
    if format == "96_well":
        return {"pval_thres_tshell": 1e-3, "wij_thres_tshell": 3}
    if format == "auto":
        return None
    raise ValueError("format must be 'auto', '384_well' or '96_well'")


def _log(verbose, *a):
    if verbose:
        print(*a, flush=True)


def _well_from_filename(name, well_pos):
    parts = os.path.basename(name).replace(".", "_").split("_")
    return parts[well_pos - 1] if len(parts) >= well_pos else None


def _well_sort_key(w):
    """arrange(substr(well, 1, 1), as.integer(substr(well, 2, 3))): C-locale letter, then the
    integer (NA last); stable."""
    try:
        num = int(w[1:3])
        return (w[:1].encode(), 0, num)
    except ValueError:
        return (w[:1].encode(), 1, 0)


def _list_well_files(folder, wellset, well_pos, verbose):
    names = [n for n in os.listdir(folder) if not n.startswith(".")]
    names = [names[i] for i in r_order(names)]
    out = {}
    for chain, pat in (("alpha", re.compile(r".*_TRA\.tsv")), ("beta", re.compile(r".*_TRB\.tsv"))):
        files = [n for n in names if pat.search(n)]  # list.files(pattern=) = regex search
        by_well = {}
        for f in files:
            w = _well_from_filename(f, well_pos)
            if w in set(wellset):
                by_well.setdefault(w, []).append(f)
        missing = [w for w in wellset if w not in by_well]
        if missing:
            _log(verbose, f"{len(missing)} wells missing .tsv file (TCR{chain}): {', '.join(missing)}")
        out[chain] = by_well
    rows = []
    for w in dict.fromkeys(wellset):  # unique(wellset), then left joins (one row per matching file pair)
        fa = out["alpha"].get(w, [None])
        fb = out["beta"].get(w, [None])
        for a in fa:
            for b in fb:
                rows.append((w, a, b))
    rows.sort(key=lambda r: _well_sort_key(r[0]))  # Python sort is stable
    return rows


def _plates(folder_path, wellset):
    folders = [folder_path] if isinstance(folder_path, (str, os.PathLike)) else list(folder_path)
    if isinstance(wellset, (list, tuple)) and wellset and isinstance(wellset[0], (list, tuple)):
        if len(wellset) != len(folders):
            raise ValueError("If 'wellset' is a list, it needs to be the same length as 'folder_path'")
        sets = [list(w) for w in wellset]
    else:
        sets = [list(wellset)] * len(folders)
    # plate name = basename(normalizePath(folder)) (symbolic links resolved, as in R)
    plates = [dict(path=str(f), wells=s, name=os.path.basename(os.path.realpath(f))) for f, s in zip(folders, sets)]
    for p in plates:
        if not os.path.isdir(p["path"]):
            raise FileNotFoundError(p["path"])
    if len(plates) > 1 and len({p["name"] for p in plates}) < len(plates):
        raise ValueError("Folder names in 'folder_path' need to be unique to tell plates apart")
    return plates


# --------------------------------------------------------------------------- numpy fallback kernel
def _pair_tt_numpy(A, B, mdh, do_tshell=True, filter_before_top3=False, chunk=1024):
    w = A.shape[1]
    Ab, Bb = (A > 0).astype(np.float64), (B > 0).astype(np.float64)
    wa, wb = Ab.sum(1).astype(np.int64), Bb.sum(1).astype(np.int64)

    def stats(M, cnt):
        s = np.where(M > 0, M, 0).sum(1)
        mean = s / w
        ss = np.where(M > 0, (M - mean[:, None]) ** 2, 0).sum(1)
        return mean, np.sqrt(ss + (w - cnt) * mean ** 2)
    ma, sa = stats(A, wa)
    mb, sb = stats(B, wb)
    tb = np.nonzero(wb > 2)[0]
    out = {k: [] for k in ("ma", "mb", "mo", "ta", "tb", "tr", "to")}
    for c0 in range(0, A.shape[0], chunk):
        sl = slice(c0, min(c0 + chunk, A.shape[0]))
        ov = (Ab[sl] @ Bb.T).astype(np.int64)
        if mdh is not None:
            wi = wa[sl][:, None] - ov
            ia, ib = np.nonzero((wb[None, :] - ov) < mdh[ov, wi])
            out["ma"].append(ia + c0), out["mb"].append(ib), out["mo"].append(ov[ia, ib])
        if do_tshell and len(tb):
            prod = np.where(A[sl] > 0, A[sl], 0) @ np.where(B[tb] > 0, B[tb], 0).T
            r = (prod - w * ma[sl][:, None] * mb[tb][None, :]) / (sa[sl][:, None] * sb[tb][None, :])
            if filter_before_top3:
                o = ov[:, tb].astype(np.float32)
                fa, fb = wa[sl][:, None].astype(np.float32), wb[tb][None, :].astype(np.float32)
                den = o + (fb - o) + (fa - o)
                bad = (ov[:, tb] <= 2) | (((fb - o) / den + (fa - o) / den) >= np.float32(0.5))
                r = np.where(bad, -1.0, r)
            n_top = min(3, len(tb))
            for k, i in enumerate(range(sl.start, sl.stop)):
                if wa[i] <= 2:
                    continue
                order = np.lexsort((-np.arange(len(tb)), -r[k]))[:n_top]  # r desc, higher index first
                for q in order[::-1]:
                    out["ta"].append(np.array([i])), out["tb"].append(np.array([tb[q]]))
                    out["tr"].append(np.array([r[k, q]])), out["to"].append(np.array([ov[k, tb[q]]]))
    cat = lambda x, d: np.concatenate(x).astype(d) if x else np.zeros(0, d)  # noqa: E731
    return (cat(out["ma"], np.uint32), cat(out["mb"], np.uint32), cat(out["mo"], np.uint32), cat(out["ta"], np.uint32),
            cat(out["tb"], np.uint32), cat(out["tr"], np.float64), cat(out["to"], np.uint32))


# --------------------------------------------------------------------------- main function
def run_pairing(folder_path, folder_out, prefix, well_filter_thres=0.5, min_reads=0, min_wells=2, well_pos=3,
                tshell_settings=None, wellset=None, verbose=True, write_extra_files=False,
                filter_before_top3=False, exclude_nonfunctional=False, select_best_madhype=False,
                select_best_tshell=False, gzip_output=False, pseudobulk_only=False, run_qc=True,
                clone_threshold_chain="alpha", algorithm="both", return_frame=True):
    """Find TCR alpha/beta pairs (MAD-HYPE + T-SHELL), as TIRTLtools::run_pairing(backend="cpp").

    folder_path: folder, or list of folders (one per plate; well names become "<well>_<plate>").
    tshell_settings: None ("auto": 384-well settings if >= 150 wells pass QC, else 96-well) or a dict
    with pval_thres_tshell and wij_thres_tshell (see get_tshell_settings()).
    Writes <prefix>_pseudobulk_TRA.tsv, <prefix>_pseudobulk_TRB.tsv and <prefix>_TIRTLoutput.tsv in
    folder_out and returns the pairs as a pandas DataFrame.
    """
    _timing.reset()
    if algorithm not in ("both", "madhype", "tshell"):
        raise ValueError("algorithm must be 'both', 'madhype' or 'tshell'")
    if clone_threshold_chain not in ("alpha", "beta", "independent"):
        raise ValueError("clone_threshold_chain must be 'alpha', 'beta' or 'independent'")
    if wellset is None:
        wellset = get_well_subset()
    folder_out = Path(folder_out)
    folder_out.mkdir(parents=True, exist_ok=True)
    gz = ".gz" if gzip_output else ""
    with stage("total"):
        # 1. load well data ------------------------------------------------------------
        with stage("load"):
            plates = _plates(folder_path, wellset)
            multi = len(plates) > 1
            wells, fa, fb = [], [], []
            for pl in plates:
                if multi:
                    _log(verbose, f"Plate {pl['name']}:")
                for w, a, b in _list_well_files(pl["path"], pl["wells"], well_pos, verbose):
                    wells.append(f"{w}_{pl['name']}" if multi else w)
                    fa.append(os.path.join(pl["path"], a) if a else None)
                    fb.append(os.path.join(pl["path"], b) if b else None)
            alpha = load_chain(fa, functional_only=exclude_nonfunctional)
            beta = load_chain(fb, functional_only=exclude_nonfunctional)
            na_rows, nb_rows = np.asarray(alpha.nrows()), np.asarray(beta.nrows())
            _log(verbose, f"Loaded {int((na_rows > 0).sum())} TCRalpha and {int((nb_rows > 0).sum())} TCRbeta well files")
        # 2. well QC ---------------------------------------------------------------------
        thr = {}
        for ch, n in (("alpha", na_rows), ("beta", nb_rows)):
            nz = n[n != 0]
            thr[ch] = float(round(well_filter_thres * float(nz.mean()))) if len(nz) else float("nan")
        if clone_threshold_chain == "alpha":
            thr["beta"] = thr["alpha"]
        elif clone_threshold_chain == "beta":
            thr["alpha"] = thr["beta"]
        idx = list(range(len(wells)))
        if run_qc:
            good_a = [wells[i] for i in idx if na_rows[i] > thr["alpha"]]
            gb = {wells[i] for i in idx if nb_rows[i] > thr["beta"]}
            all_ws = set(f"{w}_{pl['name']}" if multi else w for pl in plates for w in pl["wells"])
            keep = set(w for w in dict.fromkeys(good_a) if w in gb and w in all_ws)
            _log(verbose, f"Alpha clone threshold for QC: {thr['alpha']:g}", f"| beta: {thr['beta']:g}",
                 f"| wells passing alpha and beta QC: {len(keep)} of {len(wells)}")
            idx = [i for i in idx if wells[i] in keep]
        # 3. pseudobulk ------------------------------------------------------------------
        with stage("pseudobulk"):
            _log(verbose, "Tabulating pseudobulk counts")
            _pseudobulk(alpha, idx, len(idx), folder_out / f"{prefix}_pseudobulk_TRA.tsv{gz}")
            _pseudobulk(beta, idx, len(idx), folder_out / f"{prefix}_pseudobulk_TRB.tsv{gz}")
        if pseudobulk_only:
            return None
        # 4. read-fraction matrices ------------------------------------------------------
        with stage("merge"):
            # wells with >= 1 chain above min_reads for both TCRalpha and TCRbeta
            mx_a, mx_b = np.asarray(alpha.max_counts()), np.asarray(beta.max_counts())
            pw = [i for i in idx if mx_a[i] > min_reads and mx_b[i] > min_reads]
            a_ids, A, na_tot = alpha.merge_tt(pw, float(min_reads), int(min_wells))
            b_ids, B, nb_tot = beta.merge_tt(pw, float(min_reads), int(min_wells))
            _log(verbose, f"Unique alpha clones: {na_tot}, in more than {min_wells} wells: {A.shape[0]}; "
                          f"beta: {nb_tot} / {B.shape[0]}; wells used for pairing: {len(pw)}")
        # 5. pairing ---------------------------------------------------------------------
        if tshell_settings is None:
            fmt = "384_well" if len(idx) >= TSHELL_AUTO_MIN_WELLS else "96_well"
            _log(verbose, f"{len(idx)} wells passing: using '{fmt}' T-SHELL settings")
            tshell_settings = get_tshell_settings(fmt)
        n_wells = A.shape[1]
        if A.shape[0] == 0 or B.shape[0] == 0:
            raise ValueError("no alpha or beta clone found in more than min_wells wells")
        prior = 1 / (float(A.shape[0]) * float(B.shape[0])) ** 0.5
        with stage("lookup_table"):
            mdh = np.asarray(madhyper_surface(n_wells=n_wells, cells=thr["alpha"], alpha=2, prior=prior))
        do_m, do_t = algorithm in ("both", "madhype"), algorithm in ("both", "tshell")
        with stage("pairing"):
            c = core()
            mdh_i = np.ascontiguousarray(mdh.astype(np.int64)) if do_m else None
            if c is not None:
                ma, mb, mw, ta, tb, tr, tw = c.pair_sparse_tt(np.ascontiguousarray(A), np.ascontiguousarray(B),
                                                              mdh_i, do_t, bool(filter_before_top3))
            else:
                ma, mb, mw, ta, tb, tr, tw = _pair_tt_numpy(A, B, mdh_i, do_t, filter_before_top3)
            _log(verbose, f"MAD-HYPE pairs: {len(ma)}; T-SHELL candidate pairs: {len(ta)}")
        if write_extra_files:
            np.savetxt(folder_out / f"{prefix}_mdh.tsv", mdh, delimiter="\t", fmt="%.15g")
        # 6. post-processing -------------------------------------------------------------
        with stage("postprocess"):
            res = _assemble_tt(ma, mb, mw, ta, tb, tr, tw, A, B, n_wells, thr["alpha"], prior, tshell_settings)
        with stage("annotate"):

            class _P:  # what _annotate needs
                pass
            p = _P()
            p.alpha, p.beta = alpha, beta
            cols = _annotate(res, p, idx, idx, a_ids, b_ids)
            cols = _finalize_tt(res, cols, select_best_madhype, select_best_tshell)
        with stage("write_output"):
            out_path = folder_out / f"{prefix}_TIRTLoutput.tsv{gz}"
            frame = None
            if c is not None and not gzip_output:
                c.write_tsv(str(out_path), cols)
            if return_frame or c is None or gzip_output:
                fcols = [(k, v) for k, v in cols if k != "is_functional"]
                frame = _frame_from_cols(fcols)
                flag = dict(cols).get("is_functional")
                if flag is not None:
                    fl = flag[1]
                    frame["is_functional"] = pd.array(np.where(fl < 0, None, fl == 1), dtype="boolean") \
                        if (fl < 0).any() else (fl == 1)
                if c is None or gzip_output:
                    frame.to_csv(out_path, sep="\t", index=False)
        if verbose:
            m = cols_method_counts(cols)
            _log(verbose, f"Number of clones paired by MAD-HYPE: {m[0]}, by T-SHELL: {m[1]}")
    return frame


def cols_method_counts(cols):
    m = dict(cols)["method"][1]
    return int((m == 0).sum()), int((m == 1).sum())


def _assemble_tt(ma, mb, mw, ta, tb, tr, tw, A, B, n_wells, cells, prior, ts):
    wa = (A > 0).sum(1).astype(np.int64)
    wb = (B > 0).sum(1).astype(np.int64)
    ma, mb, ta, tb = (np.asarray(x, np.int64) for x in (ma, mb, ta, tb))
    nm, nt = len(ma), len(ta)
    ai, bi = np.concatenate([ma, ta]), np.concatenate([mb, tb])
    wij = np.concatenate([np.asarray(mw, np.float64), np.asarray(tw, np.float64)])
    r = np.concatenate([np.full(nm, np.nan), np.asarray(tr, np.float64)])
    with np.errstate(divide="ignore", invalid="ignore"):
        ts_ = r * np.sqrt((n_wells - 2) / (1 - r ** 2))
        pv = np.full(nm + nt, np.nan)
        pv[nm:] = _pt_two_sided(ts_[nm:], n_wells - 2)
        padj = np.full(nm + nt, np.nan)
        if nt:
            p_t = pv[nm:]
            gid = np.r_[0, np.cumsum(np.diff(ta) != 0)]
            nanp = np.isnan(p_t)
            o = np.lexsort((np.where(nanp, np.inf, p_t), nanp, gid))
            g_sorted = gid[o]
            first = np.r_[0, np.nonzero(np.diff(g_sorted))[0] + 1]
            pos = np.arange(nt) - np.repeat(first, np.diff(np.r_[first, nt]))
            third = np.full(gid[-1] + 1, np.nan)
            sel3 = (pos == 2) & ~nanp[o]
            third[g_sorted[sel3]] = p_t[o][sel3]
            padj[nm:] = p_t / third[gid]
        wa_, wb_ = wa[ai], wb[bi]
        den = wij + (wb_ - wij) + (wa_ - wij)
        la, lb = (wb_ - wij) / den, (wa_ - wij) / den
    method = np.r_[np.zeros(nm, np.int8), np.ones(nt, np.int8)]
    ts_ = np.where(method == 1, ts_, np.nan)
    wi, wj = wa_ - wij, wb_ - wij
    # score of each distinct (wi, wj, wij) over ALL pairs (before the T-SHELL filter, as in R)
    key = (wi.astype(np.int64) << 42) | (wj.astype(np.int64) << 21) | wij.astype(np.int64)
    ukey, inv = np.unique(key, return_inverse=True)
    probs = np.asarray(estimate_pair_prob_batch(wi=((ukey >> 42) & 0x1FFFFF).astype(float),
                                                wj=((ukey >> 21) & 0x1FFFFF).astype(float),
                                                w_ij=(ukey & 0x1FFFFF).astype(float), w_tot=n_wells, cpw=cells,
                                                alpha=2, prior=prior))
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.log10(probs)[np.asarray(inv).reshape(-1)]
    # merge(result, well_counts, by = c("wi","wj","wij")): rows sorted by the keys (stable)
    order = np.lexsort((np.arange(nm + nt), wij, wj, wi))
    with np.errstate(invalid="ignore"):
        keep = (method == 0) | ((method == 1) & (wij > ts["wij_thres_tshell"]) & (padj < ts["pval_thres_tshell"])
                                & ((la + lb) < 0.5))
    order = order[keep[order]]
    s = lambda x: x[order]  # noqa: E731
    return dict(ai=s(ai), bi=s(bi), wij=s(wij), wa=s(wa_), wb=s(wb_), method=s(method), r=s(r), ts=s(ts_),
                pval=s(pv), pval_adj=s(padj), loss_a_frac=s(la), loss_b_frac=s(lb), wi=s(wi), wj=s(wj),
                score=s(score))


def _finalize_tt(res, cols, best_m, best_t):
    """.filter_pairs(): sort MAD-HYPE by (is_functional desc, score desc) and T-SHELL by
    (is_functional desc, pval asc), optional greedy best-pair selection, TIRTLtools column order."""
    d = dict(cols)

    def strings(name):
        codes, dct = d[name][1], d[name][2]
        return codes, dct
    func = np.ones(len(res["method"]), dtype=bool)
    for nm in ("cdr3a", "cdr3b"):
        codes, dct = strings(nm)
        bad = np.array([("*" in x) or ("_" in x) for x in dct] + [False], dtype=bool)
        func &= ~bad[np.where(codes < 0, len(dct), codes)]
    n = len(func)
    m_idx = np.nonzero(res["method"] == 0)[0]
    t_idx = np.nonzero(res["method"] == 1)[0]
    m_idx = m_idx[np.lexsort((np.arange(len(m_idx)), -np.nan_to_num(res["score"][m_idx], nan=-np.inf), ~func[m_idx]))]
    t_idx = t_idx[np.lexsort((np.arange(len(t_idx)), np.nan_to_num(res["pval"][t_idx], nan=np.inf), ~func[t_idx]))]
    with_flag = not (best_m and best_t)
    c = core()

    def select(ix):
        if len(ix) == 0:
            return ix
        a = np.ascontiguousarray(d["alpha_nuc"][1][ix], np.int64)
        b = np.ascontiguousarray(d["beta_nuc"][1][ix], np.int64)
        if c is not None:
            k = np.asarray(c.greedy_select(a, b, 2, 1))
        else:
            from collections import Counter
            ca, cb, k = Counter(), Counter(), np.zeros(len(ix), bool)
            for q in range(len(ix)):
                if cb[b[q]] < 2 and ca[a[q]] < 1:
                    k[q] = True
                    cb[b[q]] += 1
                    ca[a[q]] += 1
        return ix[k]
    if best_m:
        m_idx = select(m_idx)
    if best_t:
        t_idx = select(t_idx)
    # bind_rows(): rows that went through .build_pairs_df() lose is_functional (NA in the output)
    flag = np.where(func, 1, 0).astype(np.int64)
    if best_m:
        flag[m_idx] = -1
    if best_t:
        flag[t_idx] = -1
    order = np.r_[m_idx, t_idx].astype(np.int64)
    sel = {k: v[order] for k, v in res.items()}
    new = []
    reord = {}  # same code array -> same reordered array (lets the frame builder share columns)

    def ro(x):
        k = id(x)
        if k not in reord:
            reord[k] = np.ascontiguousarray(x[order])
        return reord[k]
    for name, spec in cols:
        kind = spec[0]
        if kind in ("f", "i"):
            new.append((name, (kind, ro(spec[1]))))
        elif kind == "s":
            new.append((name, ("s", ro(spec[1]), spec[2])))
        else:
            new.append((name, ("c", ro(spec[1]), spec[2], ro(spec[3]), spec[4], spec[5])))
    dd = dict(new)
    out = [(k, dd[k]) for k in OUTPUT_COLUMNS]
    if with_flag:
        out.append(("is_functional", ("s", np.ascontiguousarray(flag[order]), ["FALSE", "TRUE"])))
    del sel
    return out
