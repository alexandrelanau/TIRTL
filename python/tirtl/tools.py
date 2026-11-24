"""TIRTLtools data processing in Python (faithful to TIRTLtools 0.2.9).

A TIRTLseqDataSet holds `data` (sample id -> dict of DataFrames "alpha", "beta", "paired",
"paired_alt"), `meta` (one row per sample) and `info`. Functions keep the R names and arguments:

    ds = load_tirtlseq("results/", meta_columns=["run", "rep"])
    ds = clean_pairs(ds)
    get_pair_stats(ds); summarize_data(ds); get_all_tcrs(ds, "paired")
    get_expanded_clones(ds.data["tp1"], ds.data["tp2"])

Orderings follow dplyr / data.table (strings in C locale, NA last, stable), R's sort() for
sample names (ICU collation), and R's numeric conventions, so that tables are identical to the
R package (checked by tests/parity/compare_tools.py).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .collate import r_sorted

# ----------------------------------------------------------------------------- column sets
PAIRED_COLS = ["method", "va", "ja", "cdr3a", "cdr3b", "vb", "jb", "alpha_rank", "beta_rank",
               "alpha_readFraction", "beta_readFraction", "alpha_readCount", "beta_readCount", "is_functional",
               "alpha_nuc", "beta_nuc", "alpha_beta", "wi", "wj", "wij", "wa", "wb", "score", "r", "ts", "pval",
               "pval_adj", "loss_a_frac", "loss_b_frac", "alpha_readCount_max", "alpha_readCount_median",
               "alpha_sem", "alpha_max_wells", "beta_readCount_max", "beta_readCount_median", "beta_sem",
               "beta_max_wells", "alpha_has_stop_codon", "alpha_has_frameshift", "beta_has_stop_codon",
               "beta_has_frameshift", "alpha_is_functional", "beta_is_functional"]
PAIRED_ALT_COLS = ["paired_status"] + [c for c in PAIRED_COLS if c != "method"] + ["is_paired_madhype",
                                                                                     "is_paired_tshell"]
SINGLE_COLS = ["id", "rank", "aaSeqCDR3", "v", "j", "readFraction", "readCount", "n_wells", "paired_status",
               "n_paired", "targetSequences", "readCount_median", "readCount_max", "sem", "max_wells",
               "n_paired_madhype", "n_paired_tshell", "has_stop_codon", "has_frameshift", "is_functional",
               "is_paired", "is_paired_tshell", "is_paired_madhype"]

_PAIRED_TYPES = {"wi": "int", "wj": "int", "wij": "int", "alpha_nuc": "str", "beta_nuc": "str", "wa": "int",
                 "wb": "int", "alpha_nuc_seq": "str", "beta_nuc_seq": "str", "alpha_beta": "str", "method": "str",
                 "r": "float", "ts": "float", "pval": "float", "pval_adj": "float", "loss_a_frac": "float",
                 "loss_b_frac": "float", "score": "float", "cdr3a": "str", "va": "str", "ja": "str",
                 "cdr3b": "str", "vb": "str", "jb": "str"}
_PB_TYPES = {"targetSequences": "str", "readCount": "int", "v": "str", "j": "str", "aaSeqCDR3": "str",
             "n_wells": "int", "readCount_max": "int", "readCount_median": "float", "sem": "float", "avg": "float",
             "readFraction": "float", "max_wells": "int"}


def _read_cols(chain, cols):
    if isinstance(cols, (list, tuple)):
        return list(cols)
    if chain == "paired":
        allc, drop = list(_PAIRED_TYPES), ["alpha_nuc_seq", "beta_nuc_seq"]
        minimal = ["alpha_nuc", "beta_nuc", "method", "cdr3a", "va", "ja", "cdr3b", "vb", "jb"]
    else:
        allc, drop = list(_PB_TYPES), ["avg"]
        minimal = ["targetSequences", "readCount", "v", "j", "aaSeqCDR3", "sem", "readFraction"]
    if cols == "all":
        return allc
    if cols == "auto":
        return [c for c in allc if c not in drop]
    if cols == "minimal":
        return minimal
    return None


# ----------------------------------------------------------------------------- data set object
@dataclass
class TIRTLseqDataSet:
    data: Dict[str, Dict[str, pd.DataFrame]]
    meta: pd.DataFrame
    info: dict = field(default_factory=dict)

    def __repr__(self):
        return f"TIRTLseqDataSet with {len(self.data)} samples: {', '.join(list(self.data)[:6])}" + \
            (" ..." if len(self.data) > 6 else "")


# ----------------------------------------------------------------------------- helpers
def _order_columns(df, cols, keep_extra=True):
    found = [c for c in cols if c in df.columns]
    extra = [c for c in df.columns if c not in found]
    return df[found + extra] if keep_extra else df[found]


def _has(s, ch):
    """grepl(ch, x, fixed = TRUE) with NA -> FALSE"""
    s = pd.Series(s)
    if pd.api.types.is_string_dtype(s) and s.dtype != object:
        return s.str.contains(ch, regex=False).fillna(False).to_numpy(dtype=bool)
    return np.array([isinstance(v, str) and ch in v for v in s], dtype=bool)


def _arrow(s):
    """pyarrow array of a string Series (zero-copy for Arrow-backed strings), or None"""
    try:
        import pyarrow as pa
        s = pd.Series(s)
        if s.dtype == object:
            return None
        return pa.chunked_array(pa.array(s).chunks if hasattr(pa.array(s), "chunks") else [pa.array(s)])
    except Exception:  # noqa: BLE001
        return None


def _str_key(s):
    """sort key for strings in C locale (byte order) with NA last (dplyr / data.table)"""
    s = pd.Series(s)
    if s.dtype != object:
        # hash factorisation, then sort only the distinct values (byte order = C locale)
        codes, uniq = pd.factorize(s, use_na_sentinel=True)
        codes = np.asarray(codes, dtype=np.int64)
        try:
            import pyarrow as pa
            import pyarrow.compute as pc
            order = np.asarray(pc.sort_indices(pa.array(uniq)), dtype=np.int64)  # byte order
        except Exception:  # noqa: BLE001
            order = np.argsort(np.asarray([u.encode() for u in uniq], dtype=object), kind="stable")
        rank = np.empty(len(uniq), dtype=np.int64)
        rank[order] = np.arange(len(uniq))
        return np.where(codes >= 0, rank[np.maximum(codes, 0)] if len(uniq) else 0, len(uniq))
    codes, uniq = pd.factorize(pd.Series(s).to_numpy(dtype=object, na_value=None), sort=True,
                               use_na_sentinel=True)
    codes = codes.astype(np.int64)
    codes[codes < 0] = len(uniq)
    return codes


def _isin(x, values):
    """x %in% values for string columns (NA never matches)"""
    a, v = _arrow(x), _arrow(pd.Series(values))
    if a is not None and v is not None:
        import pyarrow.compute as pc
        res = pc.is_in(a, value_set=v.combine_chunks().unique(), skip_nulls=True)
        return np.asarray(res.to_numpy(zero_copy_only=False), dtype=bool)
    return pd.Series(x).isin(set(pd.Series(values).dropna())).to_numpy()


def _count_in(target, source):
    """for each target value, number of occurrences in source (0 when absent / NA)"""
    t, s_ = pd.Series(target), pd.Series(source)
    allv = pd.concat([t, s_], ignore_index=True)
    codes, uniq = pd.factorize(allv, use_na_sentinel=True)
    codes = np.asarray(codes)
    src = codes[len(t):]
    cnt = np.bincount(src[src >= 0], minlength=len(uniq)) if len(uniq) else np.zeros(0, int)
    tc = codes[:len(t)]
    return np.where(tc >= 0, cnt[np.maximum(tc, 0)] if len(cnt) else 0, 0).astype(float)


def _num_key(s, desc=False):
    """sort key for numbers/booleans, NA last"""
    x = pd.to_numeric(pd.Series(s), errors="coerce").to_numpy(dtype=float)
    if desc:
        x = -x
    return np.where(np.isnan(x), np.inf, x)


def _arrange(df, keys):
    """stable sort; keys = list of (column, 'asc'|'desc')"""
    if len(df) == 0:
        return df
    arrs = []
    for col, d in reversed(keys):
        s = df[col]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            k = _str_key(s)
            arrs.append(-k if d == "desc" else k)  # NA stays last only for asc; desc string sort is rare
        else:
            arrs.append(_num_key(s, desc=(d == "desc")))
    order = np.lexsort(arrs) if arrs else np.arange(len(df))
    return df.iloc[order].reset_index(drop=True)


def _identify_non_functional(df, remove=False):
    df = df.copy()
    ua, ub = "cdr3a" in df.columns, "cdr3b" in df.columns
    if ua:
        df["alpha_has_stop_codon"] = _has(df["cdr3a"], "*")
        df["alpha_has_frameshift"] = _has(df["cdr3a"], "_")
        df["alpha_is_functional"] = ~(df["alpha_has_stop_codon"] | df["alpha_has_frameshift"])
    if ub:
        df["beta_has_stop_codon"] = _has(df["cdr3b"], "*")
        df["beta_has_frameshift"] = _has(df["cdr3b"], "_")
        df["beta_is_functional"] = ~(df["beta_has_stop_codon"] | df["beta_has_frameshift"])
    if ua and ub:
        df["is_functional"] = df["alpha_is_functional"] & df["beta_is_functional"]
    elif ua:
        df["is_functional"] = df["alpha_is_functional"]
    elif ub:
        df["is_functional"] = df["beta_is_functional"]
    elif "aaSeqCDR3" in df.columns:
        df["has_stop_codon"] = _has(df["aaSeqCDR3"], "*")
        df["has_frameshift"] = _has(df["aaSeqCDR3"], "_")
        df["is_functional"] = ~(df["has_stop_codon"] | df["has_frameshift"])
    if remove:
        df = df[df["is_functional"].to_numpy(bool)].reset_index(drop=True)
    return df


def remove_duplicates(data):
    """keep the first row of each alpha_beta (data frame, or list/dict of data frames)"""
    def one(x):
        if "alpha_beta" not in x.columns:
            x = x.assign(alpha_beta=x["alpha_nuc"].astype(str) + "_" + x["beta_nuc"].astype(str))
        return x[~x["alpha_beta"].duplicated().to_numpy()].reset_index(drop=True)
    if isinstance(data, pd.DataFrame):
        return one(data)
    if isinstance(data, dict):
        return {k: one(v) for k, v in data.items()}
    return [one(v) for v in data]


def filter_duplicate_tcrs(df, verbose=True):
    if "alpha_beta" in df.columns:
        dup = df["alpha_beta"].duplicated().to_numpy()
        if verbose:
            print(f"Removed {int(dup.sum()):,} duplicate TCRs ({100 * dup.sum() / max(len(df), 1):.2g}%) "
                  f"from a total of {len(df):,} TCRs.")
        df = df[~dup].reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------- loading
def get_samples(data_dirs, pattern=None):
    """one row per sample with its _TIRTLoutput / _pseudobulk_TRA / _pseudobulk_TRB files"""
    if isinstance(data_dirs, (str, os.PathLike)):
        data_dirs = [data_dirs]
    rows = []
    for d in data_dirs:
        d = str(d).rstrip("/")
        files = r_sorted([f for f in os.listdir(d) if not f.startswith(".")])

        def names(suffix):
            rx = re.compile(re.escape(suffix))
            return sorted({re.sub(re.escape(suffix) + ".*", "", f) for f in files if rx.search(f)},
                          key=lambda s: r_sorted([s])[0])
        paired, tra, trb = names("_TIRTLoutput.tsv"), names("_pseudobulk_TRA.tsv"), names("_pseudobulk_TRB.tsv")
        all_s = r_sorted(set(paired) | set(tra) | set(trb))
        if pattern is not None:
            all_s = [s for s in all_s if re.search(pattern, s)]

        def find(s, suffix):
            hits = [f for f in files if f.startswith(s + suffix)]
            return os.path.join(d, hits[0]) if hits else None
        for s in all_s:
            rows.append(dict(sample=s, has_paired=s in paired, has_tra=s in tra, has_trb=s in trb,
                             paired_file=find(s, "_TIRTLoutput.tsv"), tra_file=find(s, "_pseudobulk_TRA.tsv"),
                             trb_file=find(s, "_pseudobulk_TRB.tsv")))
    df = pd.DataFrame(rows, columns=["sample", "has_paired", "has_tra", "has_trb", "paired_file", "tra_file",
                                     "trb_file"])
    dup = df["sample"].duplicated()
    if dup.any():
        import warnings
        warnings.warn("Sample name collision: " + ", ".join(df.loc[dup, "sample"]))
    return df[~dup].reset_index(drop=True)


def _to_pandas_strings(table):
    """pyarrow Table -> DataFrame; strings stay Arrow-backed with pandas >= 3 (no Python objects)"""
    if int(pd.__version__.split(".")[0]) >= 3:
        dt = pd.StringDtype("pyarrow", na_value=np.nan)
        import pyarrow as pa
        return table.to_pandas(types_mapper=lambda t: dt if pa.types.is_string(t) or pa.types.is_large_string(t)
                               else None)
    return table.to_pandas()


def _read_table(path, chain, cols):
    """fread(select = ...): character fields keep "" (only "NA" is missing); numbers: "" / "NA" missing"""
    types = _PAIRED_TYPES if chain == "paired" else _PB_TYPES
    use = _read_cols(chain, cols)
    head = pd.read_csv(path, sep="\t", nrows=0).columns
    use = [c for c in (use or list(head)) if c in head]
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.csv as pv
    except ImportError:
        pa = None
    if pa is not None:
        ctypes = {c: (pa.string() if types.get(c, "str") == "str" else pa.float64()) for c in use}
        t = pv.read_csv(path, parse_options=pv.ParseOptions(delimiter="\t", quote_char=False),
                        convert_options=pv.ConvertOptions(include_columns=use, column_types=ctypes,
                                                          null_values=["", "NA"], strings_can_be_null=False,
                                                          quoted_strings_can_be_null=False))
        arrays = []
        for c in use:
            a = t.column(c)
            if types.get(c, "str") == "str":
                a = pc.if_else(pc.equal(a, "NA"), pa.scalar(None, pa.string()), a)
            elif types.get(c) == "int" and a.null_count == 0:
                a = a.cast(pa.int64())
            arrays.append(a)
        return _to_pandas_strings(pa.table(arrays, names=use))
    str_cols = [c for c in use if types.get(c, "str") == "str"]
    df = pd.read_csv(path, sep="\t", usecols=use, dtype={c: str for c in str_cols}, keep_default_na=False,
                     na_values={c: ["", "NA"] for c in use if c not in str_cols}, low_memory=False)
    for c in str_cols:
        v = df[c].astype(object)
        df[c] = v.where(v != "NA", None)
    for c in use:
        if types.get(c) == "int" and not df[c].isna().any():
            df[c] = df[c].astype(np.int64)
    return df[use]


def _empty_table(chain, cols):
    use = _read_cols(chain, cols) or list(_PAIRED_TYPES if chain == "paired" else _PB_TYPES)
    return pd.DataFrame({c: pd.Series([], dtype=object) for c in use})


def load_tirtlseq(directory, chain="all", sep="_", meta_columns=None, samples=None, pattern=None, clean=False,
                  remove_nonfunctional=False, process=True, pseudobulk_columns="auto", paired_columns="auto",
                  verbose=True, n_max=np.inf):
    """Load TIRTL-seq outputs (pseudobulk + paired) of the samples found in `directory`."""
    if chain not in ("all", "alpha", "beta", "paired"):
        raise ValueError("'chain' must be 'all', 'alpha', 'beta', or 'paired'")
    if pattern is not None and samples is not None:
        raise ValueError("Only one of 'pattern' and 'samples' may be non-null")
    sdf = get_samples(directory, pattern=pattern)
    if samples is not None:
        sdf = sdf.set_index("sample").loc[[s for s in samples if s in set(sdf["sample"])]].reset_index()
    col = {"alpha": "tra_file", "beta": "trb_file", "paired": "paired_file"}
    if chain != "all":
        sdf = sdf[sdf[col[chain]].notna()]
    if n_max < len(sdf):
        sdf = sdf.iloc[: int(n_max)]
    load = ["alpha", "beta", "paired"] if chain == "all" else [chain]
    data = {}
    for _, row in sdf.iterrows():
        if verbose:
            print(f"-- Loading files for sample: {row['sample']}...")
        obj = {}
        for ch in ("alpha", "beta", "paired"):
            cols = paired_columns if ch == "paired" else pseudobulk_columns
            f = row[col[ch]]
            obj[ch] = _read_table(f, ch, cols) if (ch in load and isinstance(f, str)) else _empty_table(ch, cols)
        if process and len(obj["alpha"]) and len(obj["beta"]):
            obj = _process_single(obj, clean=clean, remove_nonfunctional=remove_nonfunctional)
        data[row["sample"]] = obj
    meta = _make_sample_meta(list(data), meta_columns, sep)
    return TIRTLseqDataSet(data=data, meta=meta, info={"package": "tirtl (TIRTLtools 0.2.9 semantics)"})


def _make_sample_meta(ids, meta_columns=None, sep="_"):
    if not meta_columns:
        return pd.DataFrame({"sample_id": ids, "label": ids})
    rows = []
    for i in ids:
        spl = i.split(sep)
        n = min(len(spl), len(meta_columns))
        rows.append({"sample_id": i, **{meta_columns[k]: spl[k] for k in range(n)}})
    meta = pd.DataFrame(rows)
    other = [c for c in meta.columns if c != "sample_id"]
    meta["label"] = [" | ".join(f"{c}: {'NA' if pd.isna(r[c]) else r[c]}" for c in other) for _, r in meta.iterrows()]
    return meta


def _process_single(data, clean=False, remove_nonfunctional=False):
    data = {k: _identify_non_functional(v, remove=remove_nonfunctional) for k, v in data.items()}
    for ch in ("alpha", "beta"):
        d = _arrange(data[ch], [("readFraction", "desc"), ("is_functional", "desc")])
        d["rank"] = np.arange(1, len(d) + 1)
        d["id"] = [f"{ch}_{r}" for r in d["rank"]]
        data[ch] = d
    paired = _add_single_chain_data(data)
    paired = _arrange(paired, [("beta_rank", "asc"), ("alpha_rank", "asc")])
    paired["receptor_rank"] = paired["beta_rank"]
    paired["receptor_id"] = ["receptor_" + ("NA" if pd.isna(x) else str(int(x))) for x in paired["receptor_rank"]]
    data["paired"] = paired
    data = _annotate_paired_single(data, with_alt=False)
    if clean:
        data["paired"] = _clean_pairs_single(data["paired"])
    data["paired"] = _arrange(_order_columns(data["paired"], PAIRED_COLS), [("receptor_rank", "asc")])
    data["paired_alt"] = _paired_df_to_paired_alt(data["paired"])
    for ch in ("alpha", "beta"):
        data[ch] = _arrange(_order_columns(data[ch], SINGLE_COLS), [("rank", "asc")])
    return data


def _add_single_chain_data(data):
    sc = ["targetSequences", "readCount", "readCount_max", "readCount_median", "sem", "readFraction", "max_wells",
          "rank", "id"]
    out = data["paired"]
    for ch in ("alpha", "beta"):
        d = data[ch][[c for c in sc if c in data[ch].columns]]
        d = d.rename(columns={c: f"{ch}_{c}" for c in d.columns}).rename(
            columns={f"{ch}_targetSequences": f"{ch}_nuc"})
        out = out.merge(d, on=f"{ch}_nuc", how="left", sort=False)
    return out


def _paired_status(m, t):
    return np.where(m & t, "both", np.where(m, "MAD-HYPE only", np.where(t, "T-SHELL only", "neither")))


def _count_by(df, col):
    return df.groupby(col, sort=False).size()


def _annotate_paired_single(data, with_alt=True):
    p = data["paired"]
    nodup = remove_duplicates(p)
    pm, pt = p[p["method"] == "madhype"], p[p["method"] == "tshell"]
    for ch, key in (("alpha", "alpha_nuc"), ("beta", "beta_nuc")):
        d = data[ch].copy()
        ts = d["targetSequences"]
        for name, src in (("n_paired_madhype", pm), ("n_paired_tshell", pt), ("n_paired", nodup)):
            d[name] = _count_in(ts, src[key])
        d["is_paired"] = d["n_paired"].to_numpy() != 0
        d["is_paired_tshell"] = d["n_paired_tshell"].to_numpy() != 0
        d["is_paired_madhype"] = d["n_paired_madhype"].to_numpy() != 0
        d["paired_status"] = _paired_status(d["is_paired_madhype"].to_numpy(), d["is_paired_tshell"].to_numpy())
        data[ch] = d
    if with_alt:  # in load_tirtlseq this table is rebuilt from the final paired table anyway
        data["paired_alt"] = _paired_df_to_paired_alt(p)
    return data


def _paired_df_to_paired_alt(p):
    has_m = "method" in p.columns
    ab_m = p.loc[(p["method"] == "madhype").to_numpy(bool), "alpha_beta"] if has_m else pd.Series([], dtype=object)
    ab_t = p.loc[(p["method"] == "tshell").to_numpy(bool), "alpha_beta"] if has_m else pd.Series([], dtype=object)
    alt = remove_duplicates(p).copy()
    alt["paired_status"] = "paired"
    if has_m:
        alt = alt.drop(columns="method")
    alt["is_paired_madhype"] = _isin(alt["alpha_beta"], ab_m)
    alt["is_paired_tshell"] = _isin(alt["alpha_beta"], ab_t)
    alt["paired_status"] = _paired_status(alt["is_paired_madhype"].to_numpy(), alt["is_paired_tshell"].to_numpy())
    alt = _order_columns(alt, PAIRED_ALT_COLS)
    return _arrange(alt, [("receptor_rank", "asc")])


# ----------------------------------------------------------------------------- cleaning / filters
def _head_by(df, order_keys, group_cols, n):
    """DT[order(...), head(.SD, n), by = group_cols]: groups in order of first appearance"""
    d = _arrange(df, order_keys)
    if len(d) == 0:
        return d
    g = d.groupby(group_cols, sort=False, dropna=False).ngroup().to_numpy()
    rank_in_group = pd.Series(np.zeros(len(d))).groupby(g).cumcount().to_numpy()
    keep = rank_in_group < n
    d = d[keep]
    g = g[keep]
    # data.table returns the groups one after the other, in order of first appearance
    o = np.argsort(g, kind="stable")
    d = d.iloc[o]
    cols = group_cols + [c for c in d.columns if c not in group_cols]
    return d[cols].reset_index(drop=True)


def _clean_pairs_single(df, n_max_alpha=2, n_max_beta=1, prefer_functional=True):
    """data.table chain of TIRTLtools .clean_pairs_single(), on integer keys only: each step orders by
    (group columns, -is_functional, -readFraction) and keeps the first n rows of each group; groups are
    contiguous after the sort, so data.table's group order is the sorted order."""
    cols = list(df.columns)
    d = _identify_non_functional(remove_duplicates(df))
    if len(d) == 0:
        return d[cols]
    key = {c: _str_key(d[c]) for c in ("alpha_nuc", "beta_nuc", "va", "ja", "vb", "jb")}
    key["-is_functional"] = _num_key(d["is_functional"], desc=True)
    key["-alpha_readFraction"] = _num_key(d["alpha_readFraction"], desc=True)
    key["-beta_readFraction"] = _num_key(d["beta_readFraction"], desc=True)
    idx = np.arange(len(d))
    f = ["-is_functional"] if prefer_functional else []
    steps = [(["beta_nuc", "vb", "jb", "va", "ja"], f + ["-alpha_readFraction"], 1),
             (["beta_nuc", "vb", "jb"], f + ["-alpha_readFraction"], n_max_alpha),
             (["alpha_nuc", "va", "ja", "vb", "jb"], f + ["-beta_readFraction"], 1),
             (["alpha_nuc", "va", "ja"], f + ["-beta_readFraction"], n_max_beta)]
    for groups, extra, n in steps:
        ks = groups + extra
        o = np.lexsort([key[k][idx] for k in reversed(ks)])
        idx = idx[o]
        g = np.column_stack([key[k][idx] for k in groups])
        new_grp = np.r_[True, np.any(g[1:] != g[:-1], axis=1)]
        start = np.maximum.accumulate(np.where(new_grp, np.arange(len(idx)), 0))
        idx = idx[(np.arange(len(idx)) - start) < n]
    return d.iloc[idx][cols].reset_index(drop=True)


def clean_pairs(data, n_max_alpha=2, n_max_beta=1, prefer_functional=True, verbose=True):
    """keep at most `n_max_alpha` alphas per beta and `n_max_beta` betas per alpha (most frequent partners)"""
    out = {}
    for i, (name, x) in enumerate(data.data.items()):
        if verbose:
            print(f"Cleaning paired dataframe for sample {i + 1}")
        y = dict(x)
        y["paired"] = _clean_pairs_single(x["paired"], n_max_alpha, n_max_beta, prefer_functional)
        if x.get("paired_alt") is not None:
            y["paired_alt"] = _clean_pairs_single(x["paired_alt"], n_max_alpha, n_max_beta, prefer_functional)
        out[name] = y
    return TIRTLseqDataSet(data=out, meta=data.meta.copy(), info={**data.info, "pairs_cleaned": True})


def filter_mait(df, verbose=True):
    """remove MAIT TCRs (TRAV1-2 with TRAJ33/12/20)"""
    va = ["TRAV1-2", "TRAV1-2*01"]
    ja = ["TRAJ33", "TRAJ12", "TRAJ20", "TRAJ33*01", "TRAJ12*01", "TRAJ20*01"]
    if "va" in df.columns and "ja" in df.columns:
        m = _isin(df["va"], va) & _isin(df["ja"], ja)
        if verbose:
            print(f"Removed {int(m.sum()):,} MAIT TCRs ({100 * m.sum() / max(len(df), 1):.2g}%) "
                  f"from a total of {len(df):,} TCRs.")
        df = df[~m].reset_index(drop=True)
    return df


def _filter_nonfunctional_single(df, verbose=True):
    d = _identify_non_functional(df)
    keep = d["is_functional"].to_numpy(bool)
    if verbose:
        n = int((~keep).sum())
        print(f"Removed {n:,} TCRs with non-functional CDR3 amino acid sequences ({100 * n / max(len(d), 1):.2g}%) "
              f"from a total of {len(d):,} TCRs.")
    return d[keep].reset_index(drop=True)


def filter_nonfunctional_TCRs(data, verbose=True):
    if isinstance(data, TIRTLseqDataSet):
        out = {}
        for k, x in data.data.items():
            y = dict(x)
            y["paired"] = _filter_nonfunctional_single(x["paired"], verbose)
            out[k] = y
        return TIRTLseqDataSet(data=out, meta=data.meta.copy(), info=dict(data.info))
    return _filter_nonfunctional_single(data, verbose)


def filter_short_cdr3s(df, verbose=True, min_aa=6):
    """remove TCRs with a short CDR3. Note: TIRTLtools tests `is_short_cdr3a & is_short_cdr3a`
    (the beta flag is computed but not used); this is reproduced as is."""
    df = df.copy()
    na_rows = np.zeros(len(df), bool)
    ua, ub = "cdr3a" in df.columns, "cdr3b" in df.columns

    def short(col):
        s = df[col].astype(object)
        na = pd.isna(s).to_numpy()
        ln = np.array([len(x) if isinstance(x, str) else 0 for x in s])
        return np.where(na, np.nan, (ln < min_aa).astype(float)), na
    if ua:
        a, na_a = short("cdr3a")
        df["is_short_cdr3a"] = pd.array(np.where(na_a, None, a == 1), dtype="boolean")
    if ub:
        b, na_b = short("cdr3b")
        df["is_short_cdr3b"] = pd.array(np.where(na_b, None, b == 1), dtype="boolean")
    if ua:
        flag, na_rows = a == 1, na_a
    else:
        flag, na_rows = b == 1, na_b
    n = int(flag.sum())
    if verbose:
        print(f"Removed {n:,} TCRs with short CDR3 segments ({100 * n / max(len(df), 1):.2g}%) "
              f"from a total of {len(df):,} TCRs.")
    keep = ~flag & ~na_rows  # dplyr::filter drops NA
    return df[keep].reset_index(drop=True)


def filter_dataset(data, samples):
    """subset samples by name, index, or conditions like "cell_type==cd4" (intersected)"""
    meta = data.meta
    if isinstance(samples, str):
        samples = [samples]
    samples = list(samples)
    if samples and isinstance(samples[0], str):
        if any(s not in set(meta["sample_id"]) for s in samples):
            idx = None
            for cond in samples:
                k, _, v = cond.partition("==")
                if not _ or k not in meta.columns:
                    raise ValueError(f"Error for condition: {cond}")
                hit = [i for i in range(len(meta)) if str(meta[k].iloc[i]) == v]
                if not hit:
                    raise ValueError("Cannot select zero samples")
                idx = hit if idx is None else [i for i in idx if i in set(hit)]
            samples = idx
        else:
            pos = {s: i for i, s in enumerate(meta["sample_id"])}
            samples = [pos[s] for s in samples]
    names = list(data.data)
    return TIRTLseqDataSet(data={names[i]: data.data[names[i]] for i in samples},
                           meta=meta.iloc[samples].reset_index(drop=True), info=dict(data.info))


def reorder_samples(data, samples):
    if len(samples) != len(data.data):
        raise ValueError("'samples' must be the same length as the number of samples in 'data'.")
    return filter_dataset(data, samples)


def add_metadata(obj, meta_columns=None, sep="_"):
    meta = _make_sample_meta(list(obj.data), meta_columns or [], sep)
    if not meta_columns:
        meta["label"] = "sample_id: " + meta["sample_id"] if False else meta["label"]
    return TIRTLseqDataSet(data=obj.data, meta=meta, info=dict(obj.info))


def combine_TIRTLseqData(*objs):
    meta = pd.concat([o.meta for o in objs], ignore_index=True)
    if meta["sample_id"].duplicated().any():
        raise ValueError("Sample names (<data>$meta$sample_id) must not overlap between two data objects")
    data = {}
    for o in objs:
        data.update(o.data)
    return TIRTLseqDataSet(data={s: data[s] for s in meta["sample_id"]}, meta=meta, info={})


def get_tshell_settings(format="auto"):
    from .run_pairing import get_tshell_settings as g
    return g(format)


# ----------------------------------------------------------------------------- summaries
def _table(values):
    """R table() of a character vector: present values in R (ICU) collation order"""
    s = pd.Series(values).dropna()
    counts = s.value_counts()
    lv = r_sorted(counts.index.astype(str))
    return pd.DataFrame({"category": lv, "Freq": [int(counts[v]) for v in lv]})


def get_pair_stats(data, verbose=True, by_method=True):
    out = []
    for i, (name, x) in enumerate(data.data.items()):
        if verbose:
            print(f"Calculating pairing stats for sample {i + 1}")
        if "is_paired" not in x["beta"].columns:
            x = _annotate_paired_single(dict(x))
        parts = []
        for ch, df in (("paired", x["paired_alt"]), ("alpha", x["alpha"]), ("beta", x["beta"])):
            t = _table(df["paired_status"])
            t["chain"] = ch
            parts.append(t)
        d = pd.concat(parts, ignore_index=True)
        d["sample_id"] = name
        out.append(d)
    res = pd.concat(out, ignore_index=True).merge(data.meta, on="sample_id", how="left", sort=False)
    res["sample_id"] = pd.Categorical(res["sample_id"], categories=list(data.meta["sample_id"]))
    return res


def _n_unique_pairs(p):
    """nrow(remove_duplicates(p)) without building the table"""
    ab = p["alpha_beta"] if "alpha_beta" in p.columns else p["alpha_nuc"].astype(str) + "_" + p["beta_nuc"].astype(str)
    return len(pd.unique(ab))


def summarize_data(data):
    rows = []
    for name, x in data.data.items():
        rows.append(dict(sample_id=name, n_alpha_clones=len(x["alpha"]), n_beta_clones=len(x["beta"]),
                         n_clone_pairs=int(_n_unique_pairs(x["paired"])),
                         n_reads_alpha=int(x["alpha"]["readCount"].to_numpy().sum()),
                         n_reads_beta=int(x["beta"]["readCount"].to_numpy().sum())))
    return pd.DataFrame(rows)


def get_all_tcrs(data, chain="paired", remove_duplicates_=True, **kw):
    if "remove_duplicates" in kw:
        remove_duplicates_ = kw["remove_duplicates"]
    parts = []
    for i, (name, x) in enumerate(data.data.items()):
        d = x[chain]
        if chain == "paired" and remove_duplicates_:
            d = remove_duplicates(d)
        m = data.meta.iloc[[i] * len(d)].reset_index(drop=True)
        parts.append(pd.concat([d.reset_index(drop=True), m], axis=1))
    return pd.concat(parts, ignore_index=True)


def _r_cut_label(x):
    # formatC(breaks, digits = 3, width = 1) as used by cut()
    s = f"{x:.3g}"
    if "e" in s:
        m, e = s.split("e")
        s = f"{m}e{e[0]}{int(e[1:]):02d}"
    return s


def get_paired_by_read_fraction_range(data, chain="beta", cutoffs=None, by_method=True):
    if cutoffs is None:
        cutoffs = [10.0 ** k for k in range(-6, 0)]
    breaks = [0.0] + list(cutoffs) + [1.0]
    labels = [f"[{_r_cut_label(breaks[k])},{_r_cut_label(breaks[k + 1])})" for k in range(len(breaks) - 1)]
    labels[-1] = labels[-1][:-1] + "]"
    out = []
    for name, x in data.data.items():
        if "is_paired" not in x["beta"].columns:
            x = _annotate_paired_single(dict(x))
        df = x[chain]
        rf = pd.to_numeric(df["readFraction"]).to_numpy(float)
        # cut(..., right = FALSE, include.lowest = TRUE): [a, b) and the last bin closed
        b = np.searchsorted(np.asarray(breaks), rf, side="right") - 1
        b = np.where(rf == breaks[-1], len(breaks) - 2, b)
        valid = (rf >= breaks[0]) & (rf <= breaks[-1])
        rows = []
        for k in sorted(set(b[valid])):
            sel = valid & (b == k)
            r = dict(range=labels[k], n_paired_sum=int(df["is_paired"].to_numpy(bool)[sel].sum()))
            if "is_paired_tshell" in df.columns:
                r["n_paired_sum_tshell"] = int(df["is_paired_tshell"].to_numpy(bool)[sel].sum())
                r["n_paired_sum_madhype"] = int(df["is_paired_madhype"].to_numpy(bool)[sel].sum())
            r["n_total"] = int(sel.sum())
            rows.append(r)
        d = pd.DataFrame(rows)
        d["fraction_paired"] = d["n_paired_sum"] / d["n_total"]
        if "n_paired_sum_tshell" in d:
            d["fraction_paired_tshell"] = d["n_paired_sum_tshell"] / d["n_total"]
            d["fraction_paired_madhype"] = d["n_paired_sum_madhype"] / d["n_total"]
        d["sample_id"] = name
        out.append(d)
    return pd.concat(out, ignore_index=True).merge(data.meta, on="sample_id", how="left", sort=False)


# ----------------------------------------------------------------------------- expanded clones
def _r_mean_windows(x, starts, ends):
    """R mean() of x[start:end] for each window (long-double mean + refinement), vectorised"""
    c = np.concatenate([[0.0], np.cumsum(x)])
    n = (ends - starts + 1).astype(float)
    m = (c[ends + 1] - c[starts]) / n
    # one refinement pass, as R's mean(): m + sum(x - m) / n  (exact enough with float64 windows)
    out = np.empty(len(starts))
    for k in range(len(starts)):
        w = x[starts[k]:ends[k] + 1]
        mm = w.sum() / n[k]
        out[k] = mm + (w - mm).sum() / n[k]
    return out


def _moving_avg_closest(x, window_size=30, end_window_size=5):
    from ._native import core
    x = np.asarray(x, dtype=float)
    n = len(x)
    i = np.arange(1, n + 1)
    hw, he = window_size // 2, end_window_size // 2
    end = i >= n - 10
    start = np.where(end, np.maximum(1, i - he), np.maximum(1, i - hw)) - 1
    stop = np.where(end, np.minimum(n, i + he), np.minimum(n, i + hw)) - 1
    c = core()
    if c is not None and hasattr(c, "window_means"):
        return np.asarray(c.window_means(x, start.astype(np.int64), stop.astype(np.int64)))
    return _r_mean_windows(x, start, stop)


def _na_to0(df):
    """data.table x[is.na(x)] <- 0: numbers -> 0, logicals -> FALSE, characters -> "0" """
    df = df.copy()
    for c in df.columns:
        s = df[c]
        na = s.isna().to_numpy()
        if not na.any():
            continue
        if pd.api.types.is_bool_dtype(s) or str(s.dtype) == "boolean":
            df[c] = s.fillna(False).astype(bool)
        elif pd.api.types.is_numeric_dtype(s):
            df[c] = s.fillna(0)
        else:
            if s.dtype != object:  # Arrow-backed strings
                df[c] = s.fillna("0")
                continue
            first = s[~na].iloc[0] if (~na).any() else None
            if isinstance(first, (bool, np.bool_)):
                df[c] = s.astype(object).where(~na, False).astype(bool)
            else:
                df[c] = s.astype(object).where(~na, "0")
    return df


def _make_sample_vs_sample_df(d1, d2, chain="beta", smooth_sem="window", pseudo1=1e-6, pseudo2=1e-6,
                              sem_cutoff=2.5, log2fc_cutoff=3, window_size=30, end_window_size=5):
    df1, df2 = d1[chain].copy(), d2[chain].copy()
    if smooth_sem == "window":
        for d in (df1, df2):
            d["sem_orig"] = d["sem"]
            d["sem"] = _moving_avg_closest(d["sem"].to_numpy(float), window_size, end_window_size)
    for d in (df1, df2):  # tp[readCount_median > 0, avg := readFraction]
        d["avg"] = np.where(pd.to_numeric(d["readCount_median"]).to_numpy(float) > 0,
                            pd.to_numeric(d["readFraction"]).to_numpy(float), np.nan)
    keys = ["targetSequences", "aaSeqCDR3", "v", "j"]
    m = df1.merge(df2, on=keys, how="outer", suffixes=(".x", ".y"), sort=False)
    m = _arrange(m, [(k, "asc") for k in keys])  # data.table merge(all = TRUE) sorts by the keys
    m = _na_to0(m)
    m = m[(m["n_wells.x"].to_numpy(float) > 4) | (m["n_wells.y"].to_numpy(float) > 4)].reset_index(drop=True)
    for s in ("x", "y"):
        nw = m[f"n_wells.{s}"].to_numpy(float)
        ref = m.loc[nw == 3, f"sem.{s}"].to_numpy(float)
        if (nw < 3).any():
            m.loc[nw < 3, f"sem.{s}"] = (ref.mean() if len(ref) else np.nan) * 2
    ax, ay = m["avg.x"].to_numpy(float), m["avg.y"].to_numpy(float)
    m["log2FC"] = np.log2((ay + pseudo1) / (ax + pseudo2))
    sx, sy = m["sem.x"].to_numpy(float), m["sem.y"].to_numpy(float)
    lf = m["log2FC"].to_numpy(float)
    sign = np.full(len(m), "stable", dtype=object)
    sign[(lf < -log2fc_cutoff) & ((ay + pseudo2 + sem_cutoff * sy) < (ax + pseudo1 - sem_cutoff * sx))] = "down"
    sign[(lf > log2fc_cutoff) & ((ax + pseudo1 + sem_cutoff * sx) < (ay + pseudo2 - sem_cutoff * sy))] = "up"
    m["sign"] = sign
    return m


def _merge_with_pairs(sc, direction, p1, p2, chain, filter_pairs, remove_nonfunctional):
    nt, aa, v, j = (("alpha_nuc", "cdr3a", "va", "ja") if chain == "alpha" else ("beta_nuc", "cdr3b", "vb", "jb"))
    sel = ["sign", "log2FC", "targetSequences", "aaSeqCDR3", "v", "j", "readFraction.x", "readFraction.y", "sem.x",
           "sem.y", "n_wells.x", "n_wells.y", "max_wells.x", "max_wells.y", "rank.x", "rank.y", "readCount.x",
           "readCount.y", "readCount_median.x", "readCount_median.y", "readCount_max.x", "readCount_max.y"]
    ds = sc[sel].rename(columns={"targetSequences": nt, "aaSeqCDR3": aa, "v": v, "j": j})
    if filter_pairs:
        p1, p2 = _clean_pairs_single(p1), _clean_pairs_single(p2)
    pcols = ["va", "ja", "cdr3a", "cdr3b", "vb", "jb", "alpha_nuc", "beta_nuc", "alpha_beta", "is_functional",
             "alpha_is_functional", "beta_is_functional", "pairs_from"]
    jc = [nt, aa, v, j]
    s1 = ds.merge(p1[pcols], on=jc, how="inner", sort=False)
    s2 = ds.merge(p2[pcols], on=jc, how="inner", sort=False)
    allp = pd.concat([s2, s1] if direction == "up" else [s1, s2], ignore_index=True)
    allp["comparison_chain"] = chain
    allp = allp[~allp["alpha_beta"].duplicated().to_numpy()]
    out_cols = ["sign", "log2FC", "va", "ja", "cdr3a", "cdr3b", "vb", "jb", "alpha_nuc", "beta_nuc",
                "comparison_chain", "readFraction.x", "readFraction.y", "sem.x", "sem.y", "n_wells.x", "n_wells.y",
                "max_wells.x", "max_wells.y", "rank.x", "rank.y", "readCount.x", "readCount.y",
                "readCount_median.x", "readCount_median.y", "readCount_max.x", "readCount_max.y", "alpha_beta",
                "is_functional", "alpha_is_functional", "beta_is_functional", "pairs_from"]
    allp = allp[out_cols]
    intc = [c for c in allp.columns if c.startswith(("readCount", "max_wells", "n_wells", "rank"))]
    for c in intc:
        allp[c] = pd.to_numeric(allp[c]).astype(np.int64)
    allp = _arrange_abs_desc(allp)
    single = ds.copy()
    for c in [c for c in single.columns if c.startswith(("readCount", "max_wells", "n_wells", "rank"))]:
        single[c] = pd.to_numeric(single[c]).astype(np.int64)
    single["is_paired"] = _isin(single[nt], allp[nt])
    single = single[["sign", "log2FC", "is_paired"] + [c for c in single.columns if c not in ("sign", "log2FC", "is_paired")]]
    single = _arrange_abs_desc(single)
    if remove_nonfunctional:
        allp = allp[allp["is_functional"].to_numpy(bool)].reset_index(drop=True)
        single["has_stop_codon"] = _has(single[aa], "*")
        single["has_frameshift"] = _has(single[aa], "_")
        single["is_functional"] = ~(single["has_stop_codon"] | single["has_frameshift"])
        single = single[single["is_functional"].to_numpy(bool)].reset_index(drop=True)
    return {"paired": allp.reset_index(drop=True), "single_chain": single.reset_index(drop=True)}


def _arrange_abs_desc(df):
    if len(df) == 0:
        return df.reset_index(drop=True)
    k = -np.abs(df["log2FC"].to_numpy(float))
    o = np.argsort(np.where(np.isnan(k), np.inf, k), kind="stable")
    return df.iloc[o].reset_index(drop=True)


def get_expanded_clones(data1, data2, chain="beta", filter_pairs=False, remove_nonfunctional=True, log2fc_cutoff=3,
                        sem_cutoff=2.5, pseudo1=1e-6, pseudo2=1e-6, smooth_sem="window", window_size=30,
                        end_window_size=5):
    """clones expanded / contracted between two samples (single-chain) and their paired TCRs"""
    if chain not in ("alpha", "beta"):
        raise ValueError("chain must be 'alpha' or 'beta'")
    df = _make_sample_vs_sample_df(data1, data2, chain, smooth_sem, pseudo1, pseudo2, sem_cutoff, log2fc_cutoff,
                                   window_size, end_window_size)
    p1 = data1["paired_alt"].assign(pairs_from="x")
    p2 = data2["paired_alt"].assign(pairs_from="y")
    first = ["targetSequences", "aaSeqCDR3", "v", "j", "sign", "log2FC", "avg.x", "avg.y", "sem.x", "sem.y"]
    res = {}
    for key, sg in (("expanded", "up"), ("contracted", "down")):
        d = df[df["sign"] == sg]
        d = _arrange_abs_desc(d)
        d = d[first + [c for c in d.columns if c not in first]]
        res[key] = _merge_with_pairs(d, sg, p1, p2, chain, filter_pairs, remove_nonfunctional)
    return res


# ----------------------------------------------------------------------------- export / external data
THIMBLE_COLS = ["TCR_name", "TRAV", "TRAJ", "TRA_CDR3", "TRBV", "TRBJ", "TRB_CDR3", "TRAC", "TRBC", "TRA_leader",
                "TRB_leader", "Linker", "Link_order", "TRA_5_prime_seq", "TRA_3_prime_seq", "TRB_5_prime_seq",
                "TRB_3_prime_seq"]


def _make_unique(names, sep="_"):
    """R make.unique(names, sep)"""
    seen, out = {}, []
    used = set(names)
    for n in names:
        if n not in seen:
            seen[n] = 0
            out.append(n)
            continue
        k = seen[n]
        while True:
            k += 1
            cand = f"{n}{sep}{k}"
            if cand not in used:
                break
        seen[n] = k
        used.add(cand)
        out.append(cand)
    return out


def create_thimble_df(df, preset="default", TCR_names="TCR", exclude_non_functional=True, remove_duplicates=True,
                      verbose=True, **overrides):
    """paired TCRs -> stitchr/Thimble input table (columns THIMBLE_COLS)"""
    df = df.reset_index(drop=True)
    if remove_duplicates:
        key = df[["va", "vb", "ja", "jb", "cdr3a", "cdr3b"]].astype(object).fillna("NA").astype(str).agg(" ".join, axis=1)
        dup = key.duplicated().to_numpy()
        df = df[~dup].reset_index(drop=True)
        if verbose:
            print(f"Note: Removed {int(dup.sum())} duplicate TCRs.")
    if exclude_non_functional:
        df = _identify_non_functional(df)
        if verbose:
            print(f"Note: Removed {int((~df['is_functional'].to_numpy(bool)).sum())} TCRs with non-functional CDR3 sequences.")
        df = df[df["is_functional"].to_numpy(bool)].reset_index(drop=True)
    out = pd.DataFrame({c: [""] * len(df) for c in THIMBLE_COLS})
    for dst, src in (("TRAV", "va"), ("TRAJ", "ja"), ("TRA_CDR3", "cdr3a"), ("TRBV", "vb"), ("TRBJ", "jb"),
                     ("TRB_CDR3", "cdr3b")):
        out[dst] = df[src].to_numpy()
    if preset == "default":
        out["Linker"], out["Link_order"], out["TRAC"], out["TRBC"] = "T2A", "BA", "mTRAC*01", "mTRBC2*01"
    for k, v in overrides.items():
        if k in THIMBLE_COLS and v is not None:
            out[k] = v
    names = [f"{TCR_names}_{i}" for i in range(1, len(out) + 1)] if isinstance(TCR_names, str) else list(TCR_names)
    out["TCR_name"] = _make_unique(names)
    if (out["TRAC"].astype(str).str.startswith("m") | out["TRBC"].astype(str).str.startswith("m")).any():
        print("Note: You may need to run stitchr with the '-xg' option for 'additional/custom genes' since your "
              "TRAC or TRBC uses a mouse gene.")
    return out


def write_stitchr_tsv(df, output_name, output_folder=""):
    missing = [c for c in THIMBLE_COLS if c not in df.columns]
    if missing:
        raise ValueError("Incorrectly formatted data frame (use create_thimble_df()); missing: " + ", ".join(missing))
    path = os.path.join(output_folder, f"{output_name}.tsv")
    df.to_csv(path, sep="\t", index=False, quoting=1)  # write.table quotes strings and the header
    return None


def read_external_bulk(path, format="MiXCR"):
    """MiXCR clone table -> {"df": with v / j columns first, "df_raw": as read}"""
    raw = pd.read_csv(path, sep="\t", keep_default_na=False, na_values=["NA"], low_memory=False)
    need = ["targetSequences", "readCount", "allVHitsWithScore", "allJHitsWithScore", "aaSeqCDR3", "readFraction"]
    miss = [c for c in need if c not in raw.columns]
    if miss:
        raise ValueError("missing columns: " + ", ".join(miss))
    d = raw.copy()
    d["v"] = d["allVHitsWithScore"].astype(str).str.replace(r"\*.*", "", regex=True)
    d["j"] = d["allJHitsWithScore"].astype(str).str.replace(r"\*.*", "", regex=True)
    first = ["targetSequences", "readCount", "v", "j", "aaSeqCDR3", "readFraction"]
    return {"df": d[first + [c for c in d.columns if c not in first]], "df_raw": raw}
