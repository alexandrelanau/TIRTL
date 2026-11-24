"""Multi-run analyses of the original TIRTL R code: longitudinal comparison of two time points
(run_longitudinal_analysis_sub, merge_TIRTL, add_sign), concordance of two pairings, and 10x
single-cell clonotypes (process_10x, get_clonotypes_10x). Faithful to reference/R/TIRTL_functions.R,
vectorised (no per-clone / per-barcode loops).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .clones import load_chain
from .collate import r_order
from .madhyper import r_sum
from ._native import core
from .pipeline import get_well_subset


def _gsum(x, starts):
    """per-group sums (groups = consecutive runs starting at `starts`), R long-double accumulation"""
    c = core()
    if c is not None:
        return np.asarray(c.grouped_rsum(np.ascontiguousarray(x, dtype=np.float64),
                                         np.ascontiguousarray(starts, dtype=np.int64)))
    ends = np.r_[starts[1:], len(x)]
    return np.array([r_sum(x[a:b]) for a, b in zip(starts, ends)])


def _list_files(folder):
    names = [n for n in os.listdir(folder) if not n.startswith(".")]
    return [names[i] for i in r_order(names)]


def _well(name, pos):
    parts = str(name).split("_")
    return parts[pos - 1] if len(parts) >= pos else None


def _good_well_ids(a_names, b_names, a_rows, b_rows, thres, pos, wellset):
    """get_good_wells_sub(...)$well_ids"""
    wa = [_well(n, pos) for n in a_names]
    wb = [_well(n, pos) for n in b_names]
    ga = [w for w, n in zip(wa, a_rows) if n > thres]
    gb = set(w for w, n in zip(wb, b_rows) if n > thres)
    good = [w for w in dict.fromkeys(ga) if w in gb]
    ws = set(wellset)
    return [w for w in good if w in ws]


class _TimePoints:
    """clone tables of two folders keyed by nSeqCDR3, loaded together so that both time points share
    the same sequence ids (R: rbindlist(geta/getb(mlist), idcol = "filename"))"""

    def __init__(self, folder1, folder2, pos1, pos2):
        f1, f2 = _list_files(folder1), _list_files(folder2)
        self.names = {}
        self.chains = {}
        self.nrows = {}
        for ch, tag in (("alpha", "TRA"), ("beta", "TRB")):
            n1 = [f for f in f1 if tag in f]
            n2 = [f for f in f2 if tag in f]
            self.names[ch] = (n1, n2)
            data = load_chain([os.path.join(folder1, f) for f in n1] + [os.path.join(folder2, f) for f in n2],
                              seq_column="nSeqCDR3")
            self.chains[ch] = data
            r = np.asarray(data.nrows())
            self.nrows[ch] = (r[:len(n1)], r[len(n1):])
        self.pos = (pos1, pos2)

    def rows_mean(self, t):
        return float(np.mean(np.r_[self.nrows["alpha"][t], self.nrows["beta"][t]]))

    def chain_rows(self, ch, t):
        n1, n2 = self.names[ch]
        idx = list(range(len(n1))) if t == 0 else list(range(len(n1), len(n1) + len(n2)))
        names = n1 if t == 0 else n2
        f, q, x = self.chains[ch].rows(idx)
        wells = [_well(n, self.pos[t]) for n in names]
        return np.asarray(f), wells, np.asarray(q), np.asarray(x)


def _summary(file_of_row, file_wells, seq, frac, wells):
    """tp <- mlist[well %in% wells, .(avg, sem, wells, avg_well), nSeqCDR3] -> arrays by sequence id.
    Wells are per file: the filter and the well codes are computed per file, then indexed per row."""
    wells = list(wells)
    nw = len(wells)
    wset = set(wells)
    wcode_of = {}
    fcode = np.array([wcode_of.setdefault(w, len(wcode_of)) for w in file_wells], dtype=np.int64)
    fkeep = np.array([w in wset for w in file_wells], dtype=bool)
    keep = fkeep[file_of_row] if len(file_of_row) else np.zeros(0, bool)
    q, x, wc = seq[keep], frac[keep], fcode[file_of_row[keep]]
    order = np.argsort(q, kind="stable")
    qs, xs, wcs = q[order], x[order], wc[order]
    if len(qs) == 0:
        return dict(id=np.zeros(0, np.uint32), avg=np.zeros(0), sem=np.zeros(0), wells=np.zeros(0, np.int64),
                    avg_well=np.zeros(0))
    starts = np.r_[0, np.nonzero(np.diff(qs))[0] + 1]
    ends = np.r_[starts[1:], len(qs)]
    ids = qs[starts]
    s = _gsum(xs, starts)
    g = np.repeat(np.arange(len(starts)), ends - starts)
    nwc = max(len(wcode_of), 1)
    pair = np.unique(g.astype(np.int64) * nwc + wcs)  # integer keys: (clone, well) pairs
    nuw = np.bincount(pair // nwc, minlength=len(starts)).astype(float)
    # sd(c(readFraction, rep(0, nw - n_unique_wells))) / sqrt(nw): R two-pass var
    n_rows = (ends - starts).astype(float)
    nz = nw - nuw
    N = n_rows + nz
    mean = s / N
    d = xs - mean[g]
    mean = mean + (_gsum(d, starts) - nz * mean) / N
    d = xs - mean[g]
    ss = _gsum(d * d, starts) + nz * mean * mean
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt(ss / (N - 1))
    return dict(id=ids.astype(np.uint32), avg=s / nw, sem=sd / np.sqrt(nw), wells=nuw.astype(np.int64),
                avg_well=s / nuw)


def _merge_ids(chain, t1, t2, thres1=4, thres2=4, pseudo1=1e-6, pseudo2=1e-6):
    """merge_TIRTL() on summaries keyed by shared sequence ids: outer join, NA -> 0, filter, then rows
    sorted by nSeqCDR3 (only the rows kept are turned into strings and sorted)"""
    ids = np.union1d(t1["id"], t2["id"])
    out = {}
    for t, suf in ((t1, ".x"), (t2, ".y")):
        pos = np.searchsorted(t["id"], ids)
        hit = (pos < len(t["id"])) & (t["id"][np.minimum(pos, max(len(t["id"]) - 1, 0))] == ids) if len(t["id"]) \
            else np.zeros(len(ids), bool)
        for c in ("avg", "sem", "wells", "avg_well"):
            v = t[c][np.minimum(pos, max(len(t[c]) - 1, 0))] if len(t[c]) else np.zeros(len(ids))
            out[c + suf] = np.where(hit, v, 0)
    keep = (out["wells.x"] > thres1) | (out["wells.y"] > thres2)
    ids = ids[keep]
    seqs = np.asarray(chain.seq_strings(ids.astype(np.uint32)), dtype=object)
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        o = np.asarray(pc.sort_indices(pa.array(seqs, type=pa.string())), dtype=np.int64)
    except Exception:  # noqa: BLE001
        o = np.argsort(np.asarray([q.encode() for q in seqs], dtype=object), kind="stable")
    m = pd.DataFrame({"nSeqCDR3": seqs[o]})
    for c in ("avg.x", "sem.x", "wells.x", "avg_well.x", "avg.y", "sem.y", "wells.y", "avg_well.y"):
        v = out[c][keep][o]
        m[c] = v.astype(np.int64) if c.startswith("wells") else v.astype(float)
    for s in ("x", "y"):
        w = m[f"wells.{s}"].to_numpy()
        ref = m.loc[w == 3, f"sem.{s}"].to_numpy(float)
        if (w < 3).any():
            m.loc[w < 3, f"sem.{s}"] = (ref.mean() if len(ref) else np.nan) * 2
    m["log2FC"] = np.log2((m["avg.y"] + pseudo1) / (m["avg.x"] + pseudo2))
    return m


def _na_to0(df):
    df = df.copy()
    for c in df.columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(0) if pd.api.types.is_numeric_dtype(df[c]) else df[c].fillna("0")
    return df


def _sorted_codes(keys):
    """factorize string keys; return (codes, number of uniques, rank of each unique in byte order)"""
    codes, uniq = pd.factorize(pd.Series(keys), use_na_sentinel=True)
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        order = np.asarray(pc.sort_indices(pa.array(np.asarray(uniq, dtype=object))), dtype=np.int64)
    except Exception:  # noqa: BLE001
        order = np.argsort(np.asarray([u.encode() for u in uniq], dtype=object), kind="stable")
    return np.asarray(codes), uniq, order


def merge_TIRTL(tp1, tp2, thres1=4, thres2=4, pseudo1=1e-6, pseudo2=1e-6):
    """merge two time-point summaries (avg, sem, wells, avg_well by nSeqCDR3) and add log2FC.
    Full outer join on the (unique) nSeqCDR3 keys, rows sorted by key (data.table merge), NA -> 0."""
    n1 = len(tp1)
    codes, uniq, order = _sorted_codes(pd.concat([tp1["nSeqCDR3"], tp2["nSeqCDR3"]], ignore_index=True))
    nu = len(uniq)
    row1 = np.full(nu, -1, np.int64)
    row2 = np.full(nu, -1, np.int64)
    row1[codes[:n1]] = np.arange(n1)
    row2[codes[n1:]] = np.arange(len(tp2))
    row1, row2 = row1[order], row2[order]
    m = pd.DataFrame({"nSeqCDR3": np.asarray(uniq, dtype=object)[order]})
    for src, rows, suf in ((tp1, row1, ".x"), (tp2, row2, ".y")):
        for c in ("avg", "sem", "wells", "avg_well"):
            v = src[c].to_numpy()
            col = np.where(rows >= 0, v[np.maximum(rows, 0)] if len(v) else 0, 0)  # na_to0
            m[c + suf] = col.astype(np.int64) if c == "wells" else col.astype(float)
    m = m[["nSeqCDR3", "avg.x", "sem.x", "wells.x", "avg_well.x", "avg.y", "sem.y", "wells.y", "avg_well.y"]]
    m = m[(m["wells.x"] > thres1) | (m["wells.y"] > thres2)].reset_index(drop=True)
    for s in ("x", "y"):
        w = m[f"wells.{s}"].to_numpy()
        ref = m.loc[w == 3, f"sem.{s}"].to_numpy(float)
        if (w < 3).any():
            m.loc[w < 3, f"sem.{s}"] = (ref.mean() if len(ref) else np.nan) * 2
    m["log2FC"] = np.log2((m["avg.y"] + pseudo1) / (m["avg.x"] + pseudo2))
    return m


def add_sign(tirtl_m, sem_threshold=2.5, log2FC_threshold=3, pseudo1=1e-6, pseudo2=1e-6):
    out = tirtl_m.copy()
    ax, ay, sx, sy, lf = (out[c].to_numpy(float) for c in ("avg.x", "avg.y", "sem.x", "sem.y", "log2FC"))
    sign = np.full(len(out), "stable", dtype=object)
    sign[(lf < -log2FC_threshold) & ((ay + pseudo2 + sem_threshold * sy) < (ax + pseudo1 - sem_threshold * sx))] = "down"
    sign[(lf > log2FC_threshold) & ((ax + pseudo1 + sem_threshold * sx) < (ay + pseudo2 - sem_threshold * sy))] = "up"
    out["sign"] = sign
    return out


def run_longitudinal_analysis_sub(folder1, folder2, well_pos1=3, well_pos2=3, well_filter_thres=0.75, wellset1=None,
                                  wellset2=None):
    """Compare clone frequencies (by nSeqCDR3) between two time points -> {"alpha": df, "beta": df}"""
    wellset1 = wellset1 or get_well_subset()
    wellset2 = wellset2 or get_well_subset()
    tp = _TimePoints(folder1, folder2, well_pos1, well_pos2)
    w = []
    for t, (pos, ws) in enumerate(((well_pos1, wellset1), (well_pos2, wellset2))):
        an, bn = tp.names["alpha"][t], tp.names["beta"][t]
        w.append(_good_well_ids(an, bn, tp.nrows["alpha"][t], tp.nrows["beta"][t],
                                round(well_filter_thres * tp.rows_mean(t)), pos, ws))
    out = {}
    for ch in ("alpha", "beta"):
        s1 = _summary(*tp.chain_rows(ch, 0), w[0])
        s2 = _summary(*tp.chain_rows(ch, 1), w[1])
        out[ch] = add_sign(_merge_ids(tp.chains[ch], s1, s2, 4, 4))
    return out


def concordance(tirtlseq1, tirtlseq2):
    """table(tirtlseq1[beta_nuc %in% tirtlseq2$beta_nuc, alpha_beta] %in% tirtlseq2$alpha_beta)"""
    sub = tirtlseq1.loc[tirtlseq1["beta_nuc"].isin(set(tirtlseq2["beta_nuc"])), "alpha_beta"]
    v = sub.isin(set(tirtlseq2["alpha_beta"]))
    counts = v.value_counts()
    return pd.Series({k: int(counts[b]) for k, b in (("FALSE", False), ("TRUE", True)) if b in counts.index},
                     name="N", dtype=np.int64)


def get_clonotypes_10x(TCRs):
    """one row per barcode: top-UMI TRB, top two TRA (ties: file order), plus n_cells per (cdr3b_nt, cdr3a_nt)"""
    t = TCRs.reset_index(drop=True)
    bc_codes, bcs = pd.factorize(t["barcode"], sort=False)
    t = t.assign(_bc=bc_codes, _i=np.arange(len(t)))
    t = t.iloc[np.lexsort((t["_i"].to_numpy(), -pd.to_numeric(t["umis"]).to_numpy(float), t["_bc"].to_numpy()))]
    ctg = pd.DataFrame({"V1": bcs})
    nb = len(bcs)
    for chain, ranks in (("TRB", [("", 0)]), ("TRA", [("", 0), ("2", 1)])):
        c = t[t["chain"] == chain]
        k = c.groupby("_bc", sort=False).cumcount().to_numpy()
        for suffix, r in ranks:
            sel = c[k == r]
            cols = {"cdr3": "cdr3", "cdr3_nt": "cdr3_nt", "v_gene": "v", "j_gene": "j"}
            for src, dst in cols.items():
                letter = "b" if chain == "TRB" else "a"
                name = {"cdr3": f"cdr3{letter}{suffix}", "cdr3_nt": f"cdr3{letter}{suffix}_nt",
                        "v": f"v{letter}{suffix}", "j": f"j{letter}{suffix}"}[dst]
                col = np.full(nb, None, dtype=object)
                col[sel["_bc"].to_numpy()] = sel[src].astype(object).to_numpy()
                ctg[name] = col
    ctg = ctg[["V1", "cdr3b", "cdr3b_nt", "vb", "jb", "cdr3a", "cdr3a_nt", "va", "ja", "cdr3a2", "cdr3a2_nt", "va2",
               "ja2"]]
    key = ctg["cdr3b_nt"].astype(object).where(ctg["cdr3b_nt"].notna(), "\x00NA") + "\x01" + \
        ctg["cdr3a_nt"].astype(object).where(ctg["cdr3a_nt"].notna(), "\x00NA")
    ctg["n_cells"] = key.map(key.value_counts()).astype(np.int64)
    return ctg


def process_10x(path):
    """read filtered_contig_annotations.csv -> {"complete", "clean", "raw"}"""
    raw = pd.read_csv(path, keep_default_na=False, na_values=["", "NA"], low_memory=False)
    clean = get_clonotypes_10x(raw)
    o = np.argsort(-clean["n_cells"].to_numpy(), kind="stable")
    c = clean.iloc[o]
    paste = lambda a, b: a.astype(object).where(a.notna(), "NA").astype(str) + "_" + \
        b.astype(object).where(b.notna(), "NA").astype(str)  # noqa: E731
    c = c[~paste(c["cdr3b_nt"], c["cdr3a_nt"]).duplicated().to_numpy()]
    c = c[(c["cdr3b_nt"].notna() & c["cdr3a_nt"].notna()).to_numpy()].reset_index(drop=True)
    c["alpha_beta"] = c["cdr3a_nt"].astype(str) + "_" + c["cdr3b_nt"].astype(str)
    c["beta_nuc"] = c["cdr3b_nt"]
    c["alpha_nuc"] = c["cdr3a_nt"]
    return {"complete": c, "clean": clean, "raw": raw}
