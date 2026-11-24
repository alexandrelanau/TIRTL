"""Diversity indices of TIRTLtools (calculate_diversity, get_all_div_metrics), faithful to R:
proportions summed by targetSequences (groups in C-locale order), R's long-double sum()/cumsum(),
R_pow for p^q."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._native import core
from .madhyper import r_sum as _r_sum_py

ALL_METRICS = ["simpson", "gini", "gini.simpson", "inv.simpson", "shannon", "berger.parker", "richness", "d50",
               "dXX", "renyi", "hill", "top10fraction", "top100fraction", "topNfraction"]


def get_all_div_metrics():
    return list(ALL_METRICS)


def _rsum(x):
    c = core()
    x = np.ascontiguousarray(x, dtype=np.float64)
    return c.r_sum_arr(x) if c is not None else _r_sum_py(x)


def _rcumsum(x):
    c = core()
    x = np.ascontiguousarray(x, dtype=np.float64)
    if c is not None:
        return np.asarray(c.r_cumsum(x))
    out = np.empty(len(x))
    s = cc = 0.0
    for k, v in enumerate(x):
        t = s + v
        cc += (s - t) + v if abs(s) >= abs(v) else (v - t) + s
        s = t
        out[k] = s + cc
    return out


def _pow(p, q):
    """R_pow(p, q) elementwise (y == 2 -> x*x; x == 1 or y == 0 -> 1)"""
    if q == 2:
        return p * p
    if q == 0:
        return np.ones_like(p)
    return np.power(p, q)


def _dxx(p_desc_cum, xx):
    if xx < 0 or xx > 100:
        raise ValueError("'xx' needs to be a percentage between 0 and 100")
    hit = np.nonzero(p_desc_cum >= xx / 100)[0]
    if len(hit) == 0:  # R: min(which(...)) of an empty set -> Inf with a warning; run_sum[Inf] -> NA
        return {"n_clones": np.inf, "percent_required": xx, "percent_observed": np.nan}
    i = int(hit[0])
    return {"n_clones": i + 1, "percent_required": xx, "percent_observed": 100 * p_desc_cum[i]}


def _calc_all_diversity(proportions, q=range(0, 7), percent=range(10, 100, 10), tol=1e-14, n=10, metrics=None):
    metrics = metrics or ALL_METRICS
    p = np.asarray(proportions, dtype=float)
    p = p[~np.isnan(p)]
    p = p[p != 0]
    s = _rsum(p)
    if not (abs(s - 1) < tol) or not (p.min() > 0 and p.max() <= 1):
        raise ValueError(f"Data should be non-zero probabilities that sum to one. Sum of data is: {s}")
    desc = np.sort(p)[::-1]
    cum = _rcumsum(desc)
    asc = desc[::-1]
    nn = len(p)
    simpson = _rsum(p * p)
    shannon = -_rsum(p * np.log(p))

    def renyi(x):
        if x == 1:
            return shannon
        if x == math.inf:
            return -math.log(p.max())
        return (1 / (1 - x)) * math.log(_rsum(_pow(p, x)))

    def hill(x):
        if x == 0:
            return float(nn)
        if x == 1:
            return math.exp(-_rsum(p * np.log(p)))
        if x == math.inf:
            return 1 / p.max()
        return _rsum(_pow(p, x)) ** (1 / (1 - x))

    def topn(k):
        m = min(k, nn)
        return {"value": _rsum(desc[:m]), "n": k}
    res = {
        "simpson": simpson,
        "gini": (1 / nn) * (nn + 1 - 2 * (_rsum(np.arange(nn, 0, -1, dtype=float) * asc) / _rsum(asc))),
        "gini.simpson": 1 - simpson,
        "inv.simpson": 1 / simpson,
        "shannon": shannon,
        "berger.parker": float(p.max()),
        "richness": nn,
        "d50": pd.DataFrame([_dxx(cum, 50)]),
        "dXX": pd.DataFrame([_dxx(cum, x) for x in percent]),
        "renyi": pd.DataFrame([{"value": renyi(x), "q": x} for x in q]),
        "hill": pd.DataFrame([{"value": hill(x), "q": x} for x in q]),
        "top10fraction": topn(10)["value"],
        "top100fraction": topn(100)["value"],
        "topNfraction": topn(n),
    }
    return {m: res[m] for m in metrics}


def _calculate_proportions(df, type_column="targetSequences", proportion_column="readFraction"):
    cols = type_column.split("+")
    if proportion_column not in df.columns:
        g = df.groupby(cols, sort=True, dropna=False).size().rename("n").reset_index()
        g["prop"] = g["n"] / _rsum(g["n"].to_numpy(float))
        return g
    from .tools import _str_key
    # group_by(...) %>% summarize(sum(x, na.rm = TRUE)): groups sorted (C locale), NA last
    keys = [_str_key(df[c]) if not pd.api.types.is_numeric_dtype(df[c]) else df[c].to_numpy() for c in cols]
    o = np.lexsort(keys[::-1])
    d = df.iloc[o]
    k = np.column_stack([kk[o] for kk in keys])
    starts = np.r_[0, np.nonzero(np.any(k[1:] != k[:-1], axis=1))[0] + 1] if len(d) else np.zeros(0, int)
    x = pd.to_numeric(d[proportion_column]).to_numpy(float)
    x = np.where(np.isnan(x), 0.0, x)
    c = core()
    sums = np.asarray(c.grouped_rsum(np.ascontiguousarray(x), np.ascontiguousarray(starts, np.int64))) \
        if c is not None else np.add.reduceat(x, starts)
    out = d.iloc[starts][cols].reset_index(drop=True)
    out[proportion_column] = sums
    out["prop"] = sums / _rsum(sums)
    return out


def calculate_diversity(data, chain="beta", metrics=None, samples=None, q=range(0, 7), percent=range(10, 100, 10),
                        n=10, verbose=True):
    """diversity indices of each sample's single-chain repertoire -> {"result": {sample: {...}}, "meta": df}"""
    from .tools import filter_dataset
    metrics = metrics or get_all_div_metrics()
    bad = [m for m in metrics if m not in ALL_METRICS]
    if bad:
        raise ValueError("Requested diversity metrics are not available: " + " ".join(bad))
    if samples is not None:
        data = filter_dataset(data, samples)
    res = {}
    for i, (name, x) in enumerate(data.data.items()):
        if verbose:
            print(f"-- Calculating diversity indices for sample {i + 1} of {len(data.data)}.")
        prop = _calculate_proportions(x[chain])
        res[name] = {"diversity": _calc_all_diversity(prop["prop"].to_numpy(), q=list(q), percent=list(percent),
                                                      tol=1e-10, n=n, metrics=metrics),
                     "prop_df": prop}
    return {"result": res, "meta": data.meta}
