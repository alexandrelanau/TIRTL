"""MiXCR clone tables of one chain: loading, big_merge_freqs2, add_VJ_aa and combineTCR.

`load_chain(paths)` returns the Rust `ChainData` when the extension is available, otherwise the
pure-Python `PyChainData` below, which has the same interface and semantics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._native import core
from .collate import r_sort_key
from .madhyper import r_sum

_COLS = ["readCount", "readFraction", "targetSequences", "aaSeqCDR3", "allVHitsWithScore", "allJHitsWithScore",
         "nSeqCDR3"]


def _gene_token(s):
    s = "" if s is None else str(s)
    p = s.find("*")
    return s if p < 0 else s[:p]


class _Interner:
    def __init__(self):
        self.map = {}
        self.strings = []

    def ids(self, values):
        out = np.empty(len(values), dtype=np.uint32)
        m, st = self.map, self.strings
        for k, v in enumerate(values):
            i = m.get(v)
            if i is None:
                i = len(st)
                m[v] = i
                st.append(v)
            out[k] = i
        return out


class PyChainData:
    """Pure-Python implementation of tirtl._core.ChainData (same methods, same results)."""

    def __init__(self, files, seqs, aas, genes):
        self.files, self._seqs, self._aas, self._genes = files, seqs, aas, genes

    @staticmethod
    def load(paths, functional_only=False, seq_column="targetSequences"):
        seqs, aas, genes = _Interner(), _Interner(), _Interner()
        files = []
        for p in paths:
            if not p:  # well without a file
                d = pd.DataFrame({c: pd.Series([], dtype=str) for c in _COLS})
            else:
                head = pd.read_csv(p, sep="\t", nrows=0).columns
                use = [c for c in _COLS if c in head]
                d = pd.read_csv(p, sep="\t", usecols=use, dtype=str, keep_default_na=False, na_filter=False)
            if functional_only and "aaSeqCDR3" in d:
                d = d[~d["aaSeqCDR3"].str.contains(r"[*_]", regex=True)]
            col = lambda c: d[c].tolist() if c in d else [""] * len(d)  # noqa: E731
            # Python float() parsing is correctly rounded (like Rust's parser)
            num = lambda c: d[c].replace({"": "nan", "NA": "nan"}).astype(float).to_numpy()  # noqa: E731
            files.append(dict(
                seq=seqs.ids(col(seq_column)), count=num("readCount"), frac=num("readFraction"),
                aa=aas.ids(col("aaSeqCDR3")),
                v=genes.ids([_gene_token(x) for x in col("allVHitsWithScore")]),
                j=genes.ids([_gene_token(x) for x in col("allJHitsWithScore")])))
        return PyChainData(files, seqs, aas, genes)

    def nrows(self):
        return [len(f["seq"]) for f in self.files]

    def max_counts(self):
        return [float(np.nanmax(f["count"])) if len(f["count"]) and not np.all(np.isnan(f["count"])) else -np.inf
                for f in self.files]

    def n_seqs(self):
        return len(self._seqs.strings)

    def merge(self, file_idx, min_reads, min_wells):
        cols = [f for f in file_idx if np.any(self.files[f]["count"] > min_reads)]
        ncol = len(cols)
        parts = []
        for ci, f in enumerate(cols):
            fd = self.files[f]
            keep = fd["count"] > min_reads
            parts.append(pd.DataFrame({"seq": fd["seq"][keep], "col": ci, "frac": fd["frac"][keep]}))
        if not parts:
            return np.zeros(0, np.uint32), np.zeros((0, 0)), 0, []
        d = pd.concat(parts, ignore_index=True)
        codes, uniq = pd.factorize(d["seq"], sort=False)  # first-appearance order
        d["row"] = codes
        s = d.groupby(["row", "col"], sort=False)["frac"].sum()  # duplicates summed
        npos = (s > 0).groupby(level=0).sum().reindex(range(len(uniq)), fill_value=0).to_numpy()
        keep_rows = np.nonzero(npos > min_wells)[0]
        newrow = np.full(len(uniq), -1)
        newrow[keep_rows] = np.arange(len(keep_rows))
        mat = np.zeros((len(keep_rows), ncol))
        r = newrow[d["row"].to_numpy()]
        ok = r >= 0
        np.add.at(mat, (r[ok], d["col"].to_numpy()[ok]), d["frac"].to_numpy()[ok])
        return np.asarray(uniq, dtype=np.uint32)[keep_rows], mat, len(uniq), cols

    def rows(self, file_idx):
        f = np.concatenate([np.full(len(self.files[i]["seq"]), k) for k, i in enumerate(file_idx)] or [np.zeros(0)])
        q = np.concatenate([self.files[i]["seq"] for i in file_idx] or [np.zeros(0, np.uint32)])
        x = np.concatenate([self.files[i]["frac"] for i in file_idx] or [np.zeros(0)])
        return f.astype(np.int64), q.astype(np.uint32), x

    def merge_tt(self, file_idx, min_reads, min_wells):
        parts = []
        for ci, f in enumerate(file_idx):
            fd = self.files[f]
            keep = fd["count"] > min_reads
            parts.append(pd.DataFrame({"seq": fd["seq"][keep], "col": ci, "frac": fd["frac"][keep],
                                       "count": fd["count"][keep]}))
        d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["seq", "col", "frac", "count"])
        st = self._seqs.strings
        tot = d.groupby("seq")["count"].agg(lambda x: r_sum(x.to_numpy()))
        order = sorted(tot.index, key=lambda q: (-tot[q], st[q].encode()))
        row_of = {q: r for r, q in enumerate(order)}
        full = np.zeros((len(order), len(file_idx)))
        np.add.at(full, (np.array([row_of[q] for q in d["seq"]], dtype=np.int64), d["col"].to_numpy(np.int64)),
                  d["frac"].to_numpy(float))
        keep = (full > 0).sum(axis=1) > min_wells
        return np.asarray(order, dtype=np.uint32)[keep], full[keep], len(order)

    def seq_strings(self, ids):
        st = self._seqs.strings
        return [st[i] for i in np.asarray(ids)]

    def gene_strings(self):
        return list(self._genes.strings)

    def annotate(self, file_idx, ids):
        ids = np.asarray(ids)
        slot = {int(i): k for k, i in enumerate(ids)}
        aa = [None] * len(ids)
        vc = [dict() for _ in ids]
        jc = [dict() for _ in ids]
        for f in file_idx:
            fd = self.files[f]
            for k, s in enumerate(fd["seq"]):
                sl = slot.get(int(s))
                if sl is None:
                    continue
                if aa[sl] is None:
                    aa[sl] = self._aas.strings[fd["aa"][k]]
                vc[sl][int(fd["v"][k])] = vc[sl].get(int(fd["v"][k]), 0) + 1
                jc[sl][int(fd["j"][k])] = jc[sl].get(int(fd["j"][k]), 0) + 1
        g = self._genes.strings

        def pick(m):
            if not m:
                return -1
            return min(m.items(), key=lambda kv: (-kv[1], r_sort_key(g[kv[0]])))[0]
        return aa, np.array([pick(m) for m in vc], dtype=np.int64), np.array([pick(m) for m in jc], dtype=np.int64)

    def pseudobulk(self, file_idx):
        parts = []
        for fi, f in enumerate(file_idx):
            fd = self.files[f]
            parts.append(pd.DataFrame({k: fd[k] for k in ["seq", "count", "frac", "aa", "v", "j"]}))
        nmax = float(sum(1 for f in file_idx if len(self.files[f]["seq"]) > 0))
        d = pd.concat(parts, ignore_index=True)
        st = self._seqs.strings
        out = {k: [] for k in ["seq", "readCount", "v", "j", "aaSeqCDR3", "n_wells", "readCount_max",
                               "readCount_median", "avg", "sem"]}

        def mode(x):
            vals, cnt = list(dict.fromkeys(x)), {}
            for v in x:
                cnt[v] = cnt.get(v, 0) + 1
            best = vals[0]
            for v in vals:
                if cnt[v] > cnt[best]:
                    best = v
            return best
        groups = d.groupby("seq", sort=False)
        keys = sorted(groups.groups.keys(), key=lambda s: st[s].encode())
        for key in keys:
            g = d.loc[groups.groups[key]]
            n = len(g)
            fr = g["frac"].to_numpy()
            allv = np.concatenate([fr, np.zeros(max(0, int(nmax - n)))])
            nn = len(allv)
            mean = r_sum(allv) / nn
            if np.isfinite(mean):
                mean += r_sum(allv - mean) / nn
            sd = np.sqrt(r_sum((allv - mean) ** 2) / (nn - 1)) if nn > 1 else np.nan
            out["seq"].append(key)
            out["readCount"].append(r_sum(g["count"].to_numpy()))
            out["v"].append(mode(g["v"].tolist()))
            out["j"].append(mode(g["j"].tolist()))
            out["aaSeqCDR3"].append(mode(g["aa"].tolist()))
            out["n_wells"].append(float(n))
            out["readCount_max"].append(g["count"].max())
            out["readCount_median"].append(float(np.median(g["count"].to_numpy())))
            out["avg"].append(r_sum(fr) / nmax)
            out["sem"].append(sd / np.sqrt(nmax))
        res = {k: np.asarray(v) for k, v in out.items()}
        order = np.argsort(-res["readCount"], kind="stable")
        res = {k: v[order] for k, v in res.items()}
        res["readFraction"] = res["readCount"] / r_sum(res["readCount"])
        for k in ["seq", "v", "j"]:
            res[k] = res[k].astype(np.int64)
        res["aaSeqCDR3"] = [self._aas.strings[i] for i in res["aaSeqCDR3"]]
        return res


def load_chain(paths, functional_only=False, seq_column="targetSequences"):
    """paths: one per well; None / "" for a well without a file (empty table)"""
    paths = ["" if p is None else str(p) for p in paths]
    c = core()
    if c is not None:
        return c.ChainData.load(paths, functional_only, seq_column)
    return PyChainData.load(paths, functional_only, seq_column)
