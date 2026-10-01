"""
Python port of TIRTL/TIRTL_functions.R.

This module keeps the original TIRTL function names where practical.
Data-table objects are represented by pandas.DataFrame objects and R lists of
data.tables are represented by list[DataFrame] or dict[str, DataFrame].
"""

from __future__ import annotations

import math
import os
import re
import subprocess
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

from mad_hyper import estimate_pair_prob
from mad_hyper import estimate_pair_prob_batch


def _as_list(dtlist):
    if isinstance(dtlist, dict):
        return list(dtlist.items())
    if isinstance(dtlist, (list, tuple)):
        return [(str(i), x) for i, x in enumerate(dtlist)]
    raise TypeError("Expected a dict/list/tuple of pandas DataFrames")


def _subset_named(dtlist, pattern):
    items = _as_list(dtlist)
    return {name: df for name, df in items if pattern in str(name)}


def geta(dtlist):
    return _subset_named(dtlist, "TRA")


def getb(dtlist):
    return _subset_named(dtlist, "TRB")


def geta_sm(dtlist):
    return {
        name: df[["readCount", "readFraction", "targetSequences"]].copy()
        for name, df in _as_list(dtlist)
        if "TRA" in str(name)
    }


def getb_sm(dtlist):
    return {
        name: df[["readCount", "readFraction", "targetSequences"]].copy()
        for name, df in _as_list(dtlist)
        if "TRB" in str(name)
    }


def get_well_subset(row_range=range(1, 17), col_range=range(1, 25)):
    # R: LETTERS[row_range], where row_range is 1-based.
    rows = [chr(ord("A") + int(r) - 1) for r in row_range]
    return [f"{r}{int(c)}" for r in rows for c in col_range]


def _filename_well(name, pos):
    parts = str(name).split("_")
    # R indexing is 1-based.
    idx = int(pos) - 1
    return parts[idx] if 0 <= idx < len(parts) else ""


def get_good_wells_sub(alpha_list, beta_list, thres, pos=4,
                       wellset=None):
    if wellset is None:
        wellset = get_well_subset()

    a_items = _as_list(alpha_list)
    b_items = _as_list(beta_list)

    wellinds_a = np.array([_filename_well(n, pos) for n, _ in a_items], dtype=object)
    wellinds_b = np.array([_filename_well(n, pos) for n, _ in b_items], dtype=object)

    good_a = wellinds_a[np.array([len(df) > thres for _, df in a_items])]
    good_b = wellinds_b[np.array([len(df) > thres for _, df in b_items])]
    good_wells = set(good_a).intersection(set(good_b))
    good_wells.intersection_update(set(wellset))

    a_mask = np.array([w in good_wells for w in wellinds_a])
    b_mask = np.array([w in good_wells for w in wellinds_b])

    return {
        "a": a_mask,
        "b": b_mask,
        "well_ids": [w for w in wellset if w in good_wells],
    }


def big_merge_freqs2(dtlist, min_reads=0):
    """
    R equivalent of big_merge_freqs2().

    Returns a dense pandas DataFrame with targetSequences as index and file
    names as columns. Values are readFraction.
    """
    records = []
    for source, df in _as_list(dtlist):
        if df is None or len(df) == 0:
            continue
        x = df.loc[df["readCount"] > min_reads,
                   ["targetSequences", "readFraction"]].copy()
        if len(x):
            x["source"] = str(source)
            records.append(x)

    if not records:
        return pd.DataFrame()

    all_data = pd.concat(records, ignore_index=True)
    all_data["targetSequences"] = all_data["targetSequences"].astype(str)
    # R sparseMatrix sums duplicate (row,column) entries.
    mat = all_data.pivot_table(
        index="targetSequences",
        columns="source",
        values="readFraction",
        aggfunc="sum",
        fill_value=0.0,
    )
    return mat


def na_to0(x):
    if isinstance(x, pd.DataFrame):
        return x.fillna(0)
    return x.fillna(0)


def madhyper_surface(n_wells, cells=1000, alpha=2, prior=1):
    """
    Python equivalent of madhyper_surface().

    The monotonic search strategy mirrors the R implementation to avoid
    evaluating estimate_pair_prob() for every cell of the cube.
    """
    n_wells = int(n_wells)
    cells = int(cells)
    new_cube = np.zeros((n_wells + 1, n_wells + 1), dtype=float)
    i = 1

    def lp(wi, wj, wij):
        p = estimate_pair_prob(
            wi=wi, wj=wj, w_ij=wij, w_tot=n_wells,
            cpw=cells, alpha=alpha, prior=prior
        )
        if p <= 0:
            return -np.inf
        return math.log10(p)

    for wij in range(n_wells, -1, -1):
        current = lp(i - 1, 0, wij)

        if current > 0.1:
            while lp(i - 1, 0, wij) > 0.1 and i <= n_wells + 1:
                i += 1
        else:
            while i > 1 and lp(i - 1, 0, wij) < 0.1:
                i -= 1

        ans = np.zeros(n_wells + 1, dtype=float)
        z = 1
        for j in range(i, 0, -1):
            while lp(j - 1, z - 1, wij) > 0.1 and z <= n_wells + 1:
                z += 1
            ans[j - 1] = z - 1

        # R: new_cube[wij+1,] <- ans
        new_cube[wij, :] = ans

    return new_cube


def write_dat(x, fname, rows=False):
    """Equivalent of write.table(..., sep='\\t', quote=F)."""
    path = Path(fname)
    if isinstance(x, pd.DataFrame):
        x.to_csv(path, sep="\t", header=False, index=rows, na_rep="")
    else:
        np.savetxt(path, np.asarray(x), delimiter="\t", fmt="%.18g")


def _read_backend_names(path):
    return pd.read_csv(path, sep="\t", header=None, names=["sequence"])


def read_gpu(prefix):
    res_gpu = pd.read_csv(f"{prefix}_madhyperesults.csv")
    b = _read_backend_names(f"{prefix}_bigmbs_names.tsv")["sequence"].to_numpy()
    a = _read_backend_names(f"{prefix}_bigmas_names.tsv")["sequence"].to_numpy()

    res_gpu["alpha_nuc_seq"] = res_gpu["alpha_nuc"].astype(int).map(
        lambda i: a[i - 1] if 1 <= i <= len(a) else np.nan
    )
    res_gpu["beta_nuc_seq"] = res_gpu["beta_nuc"].astype(int).map(
        lambda i: b[i - 1] if 1 <= i <= len(b) else np.nan
    )
    res_gpu["alpha_nuc"] = res_gpu["alpha_nuc_seq"]
    res_gpu["beta_nuc"] = res_gpu["beta_nuc_seq"]
    res_gpu["alpha_beta"] = (
        res_gpu["alpha_nuc_seq"].astype(str) + "_" +
        res_gpu["beta_nuc_seq"].astype(str)
    )
    res_gpu["method"] = "madhype"
    return res_gpu


def read_gpu_corr(prefix):
    res = pd.read_csv(f"{prefix}_corresults.csv")
    b = _read_backend_names(f"{prefix}_bigmbs_names.tsv")["sequence"].to_numpy()
    a = _read_backend_names(f"{prefix}_bigmas_names.tsv")["sequence"].to_numpy()

    n_wells = pd.read_csv(
        f"{prefix}_bigmas.tsv", sep="\t", header=None, nrows=1
    ).shape[1]

    res["alpha_nuc_seq"] = res["alpha_nuc"].astype(int).map(
        lambda i: a[i - 1] if 1 <= i <= len(a) else np.nan
    )
    res["beta_nuc_seq"] = res["beta_nuc"].astype(int).map(
        lambda i: b[i - 1] if 1 <= i <= len(b) else np.nan
    )
    res["alpha_nuc"] = res["alpha_nuc_seq"]
    res["beta_nuc"] = res["beta_nuc_seq"]
    res["alpha_beta"] = (
        res["alpha_nuc_seq"].astype(str) + "_" +
        res["beta_nuc_seq"].astype(str)
    )

    r = pd.to_numeric(res["r"], errors="coerce").clip(-1 + 1e-15, 1 - 1e-15)
    res["ts"] = r * np.sqrt((n_wells - 2) / (1 - r**2))

    # Equivalent to 2*pt(-abs(ts), df=n_wells-2).
    try:
        from scipy.stats import t as student_t
        res["pval"] = 2.0 * student_t.sf(np.abs(res["ts"]), df=n_wells - 2)
    except ImportError as exc:
        raise ImportError(
            "read_gpu_corr() requires scipy for the Student-t CDF. "
            "Install with: pip install scipy"
        ) from exc

    # The original source computes pval_adj inside alpha_nuc groups.
    # We use the same literal normalization, with a safe fallback for groups
    # containing fewer than three observations.
    def normalize(group):
        vals = np.sort(group["pval"].to_numpy(dtype=float))
        denom = vals[2] if len(vals) >= 3 else (vals[-1] if len(vals) else np.nan)
        group = group.copy()
        group["pval_adj"] = group["pval"] / denom if denom > 0 else np.inf
        return group

    res = res.groupby("alpha_nuc", group_keys=False, dropna=False).apply(normalize)
    res["method"] = "tshell"
    return res.reset_index(drop=True)


def get_most_popularV(values):
    if values is None:
        return ""
    values = [x for x in values if pd.notna(x) and str(x) != ""]
    if not values:
        return ""
    first = [str(x).split("*", 1)[0] for x in values]
    return Counter(first).most_common(1)[0][0]


def add_VJ_aa(nSeqCDR3s, source_data):
    """
    Equivalent of add_VJ_aa().

    Returns columns cdr3aa, v and j in the same order as nSeqCDR3s.
    """
    source = source_data.copy()
    needed = {"targetSequences", "aaSeqCDR3", "allVHitsWithScore", "allJHitsWithScore"}
    missing = needed.difference(source.columns)
    if missing:
        raise ValueError(f"Missing MiXCR columns: {sorted(missing)}")

    rows = []
    for seq, g in source.groupby("targetSequences", sort=False):
        rows.append({
            "targetSequences": seq,
            "cdr3aa": g["aaSeqCDR3"].dropna().iloc[0] if g["aaSeqCDR3"].notna().any() else np.nan,
            "v": get_most_popularV(g["allVHitsWithScore"].dropna().tolist()),
            "j": get_most_popularV(g["allJHitsWithScore"].dropna().tolist()),
        })
    lookup = pd.DataFrame(rows).set_index("targetSequences") if rows else pd.DataFrame(
        columns=["cdr3aa", "v", "j"]
    )

    out = lookup.reindex(pd.Index(nSeqCDR3s, name="targetSequences")).reset_index()
    return out


def add_sign(tirtl_m, sem_threshold=2.5, log2FC_threshold=3,
             pseudo1=1e-6, pseudo2=1e-6):
    out = tirtl_m.copy()
    out["sign"] = "stable"

    down = (
        (out["log2FC"] < -log2FC_threshold) &
        ((out["avg.y"] + pseudo2 + sem_threshold * out["sem.y"]) <
         (out["avg.x"] + pseudo1 - sem_threshold * out["sem.x"]))
    )
    up = (
        (out["log2FC"] > log2FC_threshold) &
        ((out["avg.x"] + pseudo1 + sem_threshold * out["sem.x"]) <
         (out["avg.y"] + pseudo2 - sem_threshold * out["sem.y"]))
    )
    out.loc[down, "sign"] = "down"
    out.loc[up, "sign"] = "up"
    return out


def _longitudinal_summary(dt, wells):
    wells = list(wells)
    if not wells:
        return pd.DataFrame(columns=["nSeqCDR3", "avg", "sem", "wells", "avg_well"])

    d = dt[dt["well"].isin(wells)].copy()
    if d.empty:
        return pd.DataFrame(columns=["nSeqCDR3", "avg", "sem", "wells", "avg_well"])

    rows = []
    n_wells = len(wells)
    for seq, g in d.groupby("nSeqCDR3", sort=False):
        values = g["readFraction"].astype(float).to_numpy()
        unique_wells = g["well"].nunique()
        padded = np.concatenate([values, np.zeros(max(0, n_wells - unique_wells))])
        rows.append({
            "nSeqCDR3": seq,
            "avg": values.sum() / n_wells,
            "sem": np.std(padded, ddof=1) / math.sqrt(n_wells) if n_wells > 1 else np.nan,
            "wells": unique_wells,
            "avg_well": values.sum() / unique_wells if unique_wells else 0.0,
        })
    return pd.DataFrame(rows)


def merge_TIRTL(mlist1, mlist2, wells1, wells2, thres1=4, thres2=4,
                pseudo1=1e-6, pseudo2=1e-6):
    tp1 = _longitudinal_summary(mlist1, wells1)
    tp2 = _longitudinal_summary(mlist2, wells2)

    tmp = pd.merge(tp1, tp2, on="nSeqCDR3", how="outer").fillna(0)
    tmp = tmp[(tmp["wells.x"] > thres1) | (tmp["wells.y"] > thres2)].copy()

    for side in ("x", "y"):
        mask = tmp[f"wells.{side}"] < 3
        ref = tmp.loc[tmp[f"wells.{side}"] == 3, f"sem.{side}"]
        if mask.any() and len(ref):
            tmp.loc[mask, f"sem.{side}"] = ref.mean() * 2

    tmp["log2FC"] = np.log2(
        (tmp["avg.y"] + pseudo1) / (tmp["avg.x"] + pseudo2)
    )
    return tmp


def _load_folder(folder):
    folder = Path(folder)
    result = {}
    for path in sorted(folder.iterdir()):
        if path.is_file():
            try:
                result[path.name] = pd.read_csv(path, sep="\t")
            except Exception:
                # Match fread-style permissiveness as much as possible.
                result[path.name] = pd.read_csv(path, sep="\t", engine="python")
    return result


def run_longitudinal_analysis_sub(folder1, folder2, well_pos1=3, well_pos2=3,
                                  well_filter_thres=0.75, wellset1=None,
                                  wellset2=None):
    if wellset1 is None:
        wellset1 = get_well_subset()
    if wellset2 is None:
        wellset2 = get_well_subset()

    print("start double timepoint analysis")
    print(datetime.now().isoformat(timespec="seconds"))

    mlist1 = _load_folder(folder1)
    mlist2 = _load_folder(folder2)

    mlist1_a = geta(mlist1)
    mlist1_b = getb(mlist1)
    mlist2_a = geta(mlist2)
    mlist2_b = getb(mlist2)

    def concat_with_well(d, pos):
        frames = []
        for filename, df in _as_list(d):
            x = df.copy()
            x["well"] = _filename_well(filename, pos)
            frames.append(x)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    mlist1_a = concat_with_well(mlist1_a, well_pos1)
    mlist1_b = concat_with_well(mlist1_b, well_pos1)
    mlist2_a = concat_with_well(mlist2_a, well_pos2)
    mlist2_b = concat_with_well(mlist2_b, well_pos2)

    threshold1 = round(well_filter_thres * np.mean([len(x) for x in mlist1.values()]))
    threshold2 = round(well_filter_thres * np.mean([len(x) for x in mlist2.values()]))

    qc1 = get_good_wells_sub(geta(_load_folder(folder1)), getb(_load_folder(folder1)),
                             threshold1, well_pos1, wellset1)
    qc2 = get_good_wells_sub(geta(_load_folder(folder2)), getb(_load_folder(folder2)),
                             threshold2, well_pos2, wellset2)

    return {
        "alpha": add_sign(merge_TIRTL(
            mlist1_a, mlist2_a, qc1["well_ids"], qc2["well_ids"], 4, 4
        )),
        "beta": add_sign(merge_TIRTL(
            mlist1_b, mlist2_b, qc1["well_ids"], qc2["well_ids"], 4, 4
        )),
    }


def combineTCR(dt):
    """
    Equivalent of combineTCR().

    Aggregates MiXCR rows by targetSequences and returns the same summary
    columns used by the R pipeline.
    """
    x = dt.copy()
    x["v"] = x["allVHitsWithScore"].fillna("").astype(str).str.split("*").str[0]
    x["j"] = x["allJHitsWithScore"].fillna("").astype(str).str.split("*").str[0]

    nmax = x["file"].nunique() if "file" in x.columns else 1

    def mode_or_nan(s):
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        return s.mode().iloc[0]

    rows = []
    for seq, g in x.groupby("targetSequences", sort=False):
        vals = g["readFraction"].astype(float)
        padded = np.concatenate([
            vals.to_numpy(),
            np.zeros(max(0, nmax - g.shape[0]))
        ])
        rows.append({
            "targetSequences": seq,
            "readCount": g["readCount"].sum(),
            "v": mode_or_nan(g["v"]),
            "j": mode_or_nan(g["j"]),
            "aaSeqCDR3": mode_or_nan(g["aaSeqCDR3"]),
            "n_wells": len(g),
            "readCount_max": g["readCount"].max(),
            "readCount_median": g["readCount"].median(),
            "avg": vals.sum() / nmax,
            "sem": np.std(padded, ddof=1) / math.sqrt(nmax) if nmax > 1 else np.nan,
        })

    out = pd.DataFrame(rows)
    if len(out):
        out["readFraction"] = out["readCount"] / out["readCount"].sum()
    return out


def _select_qc_lists(mlista, mlistb, qc):
    a_items = _as_list(mlista)
    b_items = _as_list(mlistb)
    return (
        {name: df for (name, df), keep in zip(a_items, qc["a"]) if keep},
        {name: df for (name, df), keep in zip(b_items, qc["b"]) if keep},
    )


def _bh_adjust(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(np.nan_to_num(p, nan=np.inf))
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1.0)
    out[np.isnan(p)] = np.nan
    return out


def run_single_point_analysis_sub_gpu(
    folder_path, prefix="tmp", well_filter_thres=0.5, min_reads=0,
    min_wells=2, well_pos=3, wellset1=None, compute=True, backend="numpy",
    pval_thres_tshell=1e-10, wij_thres_tshell=2, pseudobulk=False,
):
    """
    Main Python replacement for run_single_point_analysis_sub_gpu().

    It writes exactly the intermediate TSV/CSV naming convention expected by
    the existing backend scripts in the TIRTL repository.
    """
    if wellset1 is None:
        wellset1 = get_well_subset()

    print("start")
    print(datetime.now().isoformat(timespec="seconds"))

    raw = _load_folder(folder_path)
    # R replaces '.' in filenames with '_'.
    raw = {
        re.sub(r"\.", "_", name): df for name, df in raw.items()
    }
    mlista = geta(raw)
    mlistb = getb(raw)
    print("clonesets loaded")
    print([str(x) for x in mlista.keys()])

    wellsub = np.array([
        _filename_well(name, well_pos) in set(wellset1)
        for name, _ in _as_list(mlista)
    ])
    sizes = np.array([len(df) for _, df in _as_list(mlista)])
    clone_thres = round(well_filter_thres * np.mean(sizes[wellsub]))

    qc = get_good_wells_sub(
        mlista, mlistb, clone_thres, pos=well_pos, wellset=wellset1
    )
    print("clone_threshold for QC:")
    print(clone_thres)
    print("alpha wells passing QC:")
    print(pd.Series(qc["a"]).value_counts())
    print("beta wells passing QC:")
    print(pd.Series(qc["b"]).value_counts())

    mlista, mlistb = _select_qc_lists(mlista, mlistb, qc)

    if pseudobulk:
        print("Merging pseudobulk alpha...")
        a = pd.concat(
            [df.assign(file=name) for name, df in _as_list(mlista)],
            ignore_index=True
        )
        combd_a = combineTCR(a)
        combd_a["max_wells"] = int(np.sum(qc["a"]))
        combd_a.sort_values("readCount", ascending=False).to_csv(
            f"{prefix}_pseudobulk_TRA.tsv", sep="\t", index=False
        )

        print("Merging pseudobulk beta...")
        b = pd.concat(
            [df.assign(file=name) for name, df in _as_list(mlistb)],
            ignore_index=True
        )
        combd_b = combineTCR(b)
        combd_b["max_wells"] = int(np.sum(qc["b"]))
        combd_b.sort_values("readCount", ascending=False).to_csv(
            f"{prefix}_pseudobulk_TRB.tsv", sep="\t", index=False
        )

    print("Merging alpha clonesets...")
    bigma = big_merge_freqs2(mlista, min_reads=min_reads)
    print("Done! Unique alpha clones and wells after filtering:", bigma.shape)
    bigmas = bigma.loc[(bigma > 0).sum(axis=1) > min_wells]
    print("Merging beta clonesets...")
    bigmb = big_merge_freqs2(mlistb, min_reads=min_reads)
    print("Done! Unique beta clones and wells after filtering:", bigmb.shape)
    bigmbs = bigmb.loc[(bigmb > 0).sum(axis=1) > min_wells]

    print("Writing files for back-end pairing script...")
    Path(f"{prefix}_bigmas_names.tsv").write_text(
        "\n".join(map(str, bigmas.index)) + "\n"
    )
    Path(f"{prefix}_bigmbs_names.tsv").write_text(
        "\n".join(map(str, bigmbs.index)) + "\n"
    )
    write_dat(bigmas.to_numpy(), f"{prefix}_bigmas.tsv")
    write_dat(bigmbs.to_numpy(), f"{prefix}_bigmbs.tsv")

    n_wells = bigmas.shape[1]
    prior = 1.0 / math.sqrt(max(1, bigmas.shape[0] * bigmbs.shape[0]))
    print("Pre-computing look-up table:")
    mdh = madhyper_surface(
        n_wells=n_wells,
        cells=clone_thres,
        alpha=2,
        prior=prior,
    )
    write_dat(mdh, f"{prefix}_mdh.tsv")

    if compute:
        backend_script = {
            "numpy": "numpy_backend_script.py",
            "cupy": "cupy_backend_script.py",
            "mlx": "mlx_backend_script.py",
        }.get(backend)
        if backend_script is None:
            raise ValueError("backend must be 'numpy', 'cupy', or 'mlx'")

        subprocess.run(
            ["python3", backend_script, prefix],
            check=True,
        )

    print("Loading and filtering results, adding amino acid and V segment information")
    gpu_res = read_gpu(prefix)
    gpu_res_corr = read_gpu_corr(prefix)
    result = pd.concat([gpu_res, gpu_res_corr], ignore_index=True, sort=False)

    denom = result["wij"] + (result["wb"] - result["wij"]) + (result["wa"] - result["wij"])
    result["loss_a_frac"] = (result["wb"] - result["wij"]) / denom.replace(0, np.nan)
    result["loss_b_frac"] = (result["wa"] - result["wij"]) / denom.replace(0, np.nan)
    result["wi"] = result["wa"] - result["wij"]
    result["wj"] = result["wb"] - result["wij"]

    # combos = result[["wi", "wj", "wij"]].drop_duplicates().copy()
    # combos["score"] = [
    #     math.log10(estimate_pair_prob(
    #         wi=int(row.wi), wj=int(row.wj), w_ij=int(row.wij),
    #         w_tot=n_wells, cpw=clone_thres, alpha=2, prior=prior
    #     )) if estimate_pair_prob(
    #         wi=int(row.wi), wj=int(row.wj), w_ij=int(row.wij),
    #         w_tot=n_wells, cpw=clone_thres, alpha=2, prior=prior
    #     ) > 0 else -np.inf
    #     for row in combos.itertuples()
    # ]
    combos = (
        result[["wi", "wj", "wij"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    scores = estimate_pair_prob_batch(
        wi=combos["wi"].to_numpy(dtype=np.float64),
        wj=combos["wj"].to_numpy(dtype=np.float64),
        w_ij=combos["wij"].to_numpy(dtype=np.float64),
        w_tot=n_wells,
        cpw=clone_thres,
        alpha=2,
        prior=prior,
    )

    combos["score"] = np.where(
        scores > 0,
        np.log10(scores),
        -np.inf,
    )


    result = result.merge(combos, on=["wi", "wj", "wij"], how="left")
    keep = (
        (result["method"] == "madhype") |
        (
            (result["method"] == "tshell") &
            (result["wij"] > wij_thres_tshell) &
            (result["pval_adj"] < pval_thres_tshell) &
            ((result["loss_a_frac"] + result["loss_b_frac"]) < 0.5)
        ) |
        (result["score"] > 0.1)
    )
    result = result.loc[keep].copy()

    source_a = pd.concat(
        [df for _, df in _as_list(mlista)], ignore_index=True
    ) if mlista else pd.DataFrame()
    source_b = pd.concat(
        [df for _, df in _as_list(mlistb)], ignore_index=True
    ) if mlistb else pd.DataFrame()

    tp_a = add_VJ_aa(result["alpha_nuc"].tolist(), source_a)
    tp_b = add_VJ_aa(result["beta_nuc"].tolist(), source_b)

    result["cdr3a"] = tp_a["cdr3aa"].to_numpy()
    result["va"] = tp_a["v"].to_numpy()
    result["ja"] = tp_a["j"].to_numpy()
    result["cdr3b"] = tp_b["cdr3aa"].to_numpy()
    result["vb"] = tp_b["v"].to_numpy()
    result["jb"] = tp_b["j"].to_numpy()

    print("All is done! Number of paired clones:")
    print(result["method"].value_counts())
    result.to_csv(f"{prefix}_TIRTLoutput.tsv", sep="\t", index=False)
    return result


def concordance(tirtlseq1, tirtlseq2):
    """
    Equivalent of concordance().

    Returns counts of whether alpha-beta pairs in tirtlseq1 also occur in
    tirtlseq2 among shared beta sequences.
    """
    shared = tirtlseq1[tirtlseq1["beta_nuc"].isin(tirtlseq2["beta_nuc"])]
    values = shared["alpha_beta"].isin(set(tirtlseq2["alpha_beta"]))
    return values.value_counts(dropna=False)


def get_clonotypes_10x(TCRs):
    """
    Port of get_clonotypes_10x().

    For every barcode, select the highest-UMI TRB and TRA chains, plus the
    second-highest-UMI TRA chain.
    """
    required = {
        "barcode", "umis", "chain", "cdr3", "cdr3_nt",
        "v_gene", "j_gene"
    }
    missing = required.difference(TCRs.columns)
    if missing:
        raise ValueError(f"Missing 10x columns: {sorted(missing)}")

    rows = []
    for barcode, g in TCRs.groupby("barcode", sort=False):
        g = g.sort_values("umis", ascending=False)
        a = g[g["chain"] == "TRA"].head(2).reset_index(drop=True)
        b = g[g["chain"] == "TRB"].head(1).reset_index(drop=True)

        row = {
            "V1": barcode,
            "cdr3b": b["cdr3"].iloc[0] if len(b) else np.nan,
            "cdr3b_nt": b["cdr3_nt"].iloc[0] if len(b) else np.nan,
            "vb": b["v_gene"].iloc[0] if len(b) else np.nan,
            "jb": b["j_gene"].iloc[0] if len(b) else np.nan,
            "cdr3a": a["cdr3"].iloc[0] if len(a) else np.nan,
            "cdr3a_nt": a["cdr3_nt"].iloc[0] if len(a) else np.nan,
            "va": a["v_gene"].iloc[0] if len(a) else np.nan,
            "ja": a["j_gene"].iloc[0] if len(a) else np.nan,
            "cdr3a2": a["cdr3"].iloc[1] if len(a) > 1 else np.nan,
            "cdr3a2_nt": a["cdr3_nt"].iloc[1] if len(a) > 1 else np.nan,
            "va2": a["v_gene"].iloc[1] if len(a) > 1 else np.nan,
            "ja2": a["j_gene"].iloc[1] if len(a) > 1 else np.nan,
        }
        rows.append(row)

    ctg = pd.DataFrame(rows)
    if len(ctg):
        ctg["n_cells"] = ctg.groupby(
            ["cdr3b_nt", "cdr3a_nt"], dropna=False
        )["V1"].transform("size")
    return ctg


def process_10x(path):
    dt_10x = pd.read_csv(path, sep="\t")
    dt_10x_clean = get_clonotypes_10x(dt_10x)

    complete = (
        dt_10x_clean
        .sort_values("n_cells", ascending=False)
        .assign(_pair=lambda x: x["cdr3a_nt"].astype(str) + "_" + x["cdr3b_nt"].astype(str))
        .drop_duplicates("_pair")
    )
    complete = complete[
        complete["cdr3b_nt"].notna() & complete["cdr3a_nt"].notna()
    ].drop(columns="_pair")

    complete["alpha_beta"] = (
        complete["cdr3a_nt"].astype(str) + "_" +
        complete["cdr3b_nt"].astype(str)
    )
    complete["beta_nuc"] = complete["cdr3b_nt"]
    complete["alpha_nuc"] = complete["cdr3a_nt"]

    return {
        "complete": complete,
        "clean": dt_10x_clean,
        "raw": dt_10x,
    }
