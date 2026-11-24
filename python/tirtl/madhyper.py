"""MAD-HYPE pair probabilities, faithful to the original mad_hyper.R.

R semantics reproduced: `^` (R_pow), uniroot() (R_zeroin2 / Brent, tol=1e-9), dmultinom(),
long-double sum(). Cases where R would stop with an error give NaN.
The Rust kernels (tirtl._core) are used when available; the Python code below is the fallback
and the readable specification.
"""
from __future__ import annotations

import math

import numpy as np

from ._native import core

_DBL_MAX = np.finfo(float).max
_EPS = np.finfo(float).eps


# --------------------------------------------------------------------------- R primitives
def r_pow(x, y):
    if y == 2.0:
        return x * x
    if x == 1.0 or y == 0.0:
        return 1.0
    if x == 0.0:
        return 0.0 if y > 0 else (math.inf if y < 0 else y)
    if math.isfinite(x) and math.isfinite(y):
        try:
            return x ** y
        except (OverflowError, ValueError):
            return math.nan if x < 0 else math.inf
    return float(np.power(x, y))


def r_sum(xs):
    """sum() with R's long-double accumulator emulated by an error-free transformation."""
    s = 0.0
    c = 0.0
    for x in xs:
        t = s + x
        if abs(s) >= abs(x):
            c += (s - t) + x
        else:
            c += (x - t) + s
        s = t
    return s + c


def well_freq(f, c):
    return 1.0 - np.power(1.0 - np.asarray(f, dtype=float), np.asarray(c, dtype=float))


def multinom_coeff(vec):
    vec = np.asarray(vec, dtype=float)
    return math.exp(math.lgamma(np.sum(vec) + 1.0) - sum(math.lgamma(x + 1.0) for x in vec))


def derivative_prob_function(f, alpha, W, w, c):
    f, alpha, W, w, c = float(f), float(alpha), float(W), float(w), float(c)
    if f == 0:
        return -(alpha * (1 - f)) + (f * c * (W - w) - w)
    if f == 1:
        return f * c * (W - w)
    omf = 1.0 - f
    cm = c if math.isnan(c) else max(c, 0.001)
    inner = (W - w) + w * r_pow(omf, c) / (r_pow(omf, cm) - 1.0)
    return -(alpha * (1 - f)) + f * (c * inner)


def _zeroin2(ax, bx, fa, fb, f, tol, maxit):
    a, b = ax, bx
    c, fc = a, fa
    maxit += 1
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    while maxit > 0:
        maxit -= 1
        prev_step = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol_act = 2 * _EPS * abs(b) + tol / 2
        new_step = (c - b) / 2
        if abs(new_step) <= tol_act or fb == 0.0:
            return b
        if abs(prev_step) >= tol_act and abs(fa) > abs(fb):
            cb = c - b
            if a == c:
                t1 = fb / fa
                p = cb * t1
                q = 1.0 - t1
            else:
                q = fa / fc
                t1 = fb / fc
                t2 = fb / fa
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0))
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)
            if p > 0:
                q = -q
            else:
                p = -p
            if p < (0.75 * cb * q - abs(tol_act * q) / 2) and p < abs(prev_step * q / 2):
                new_step = p / q
        if abs(new_step) < tol_act:
            new_step = tol_act if new_step > 0 else -tol_act
        a, fa = b, fb
        b += new_step
        fb = f(b)
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c, fc = a, fa
    return b


def _fcn2(v):
    if not math.isfinite(v):
        return -_DBL_MAX if v == -math.inf else _DBL_MAX
    return v


def _sign(x):
    return 1.0 if x > 0 else (-1.0 if x < 0 else x)


def _find_freq_py(w, w_tot, cpw, a):
    def f(x):
        return derivative_prob_function(x, a, w_tot, w, cpw)
    fl, fu = f(0.0), f(1.0)
    if math.isnan(fl) or math.isnan(fu) or not (_sign(fl) * _sign(fu) <= 0):
        return math.nan
    tr = lambda x: max(min(x, _DBL_MAX), -_DBL_MAX)  # noqa: E731
    return _zeroin2(0.0, 1.0, tr(fl), tr(fu), lambda x: _fcn2(f(x)), 1e-9, 1000)


def find_freq(w, w_tot=96, cpw=1000, a=1):
    c = core()
    if c is not None:
        return c.find_freq(float(w), float(w_tot), float(cpw), float(a))
    return _find_freq_py(float(w), float(w_tot), float(cpw), float(a))


def _dmultinom(x, prob):
    if any((not math.isfinite(p)) or p < 0 for p in prob):
        return math.nan
    s = r_sum(prob)
    if s == 0:
        return math.nan
    p = [q / s for q in prob]
    xi = []
    for v in x:
        v = v + 0.5
        if math.isnan(v):
            return math.nan
        k = int(math.trunc(v))
        if k < 0:
            return math.nan
        xi.append(k)
    size = sum(xi)
    terms = []
    n0 = 0
    for k in range(len(p)):
        if p[k] == 0:
            n0 += 1
            if xi[k] != 0:
                return 0.0
            continue
        terms.append(xi[k] * math.log(p[k]) - math.lgamma(xi[k] + 1.0))
    if n0 == len(p):
        return 1.0
    return math.exp(math.lgamma(size + 1.0) + r_sum(terms))


def _estimate_probs(wi, wj, wij, w_tot, cpw, fr):
    wo = w_tot - (wi + wj + wij)
    fs = [1.0 - r_pow(1.0 - f, cpw) for f in fr]
    p_o = (1 - fs[0]) * (1 - fs[1]) * (1 - fs[2])
    p_i = fs[0] * (1 - fs[1]) * (1 - fs[2])
    p_j = fs[1] * (1 - fs[0]) * (1 - fs[2])
    p_ij = 1 - p_o - p_i - p_j
    return 1.0 * _dmultinom([wi, wj, wij, wo], [p_i, p_j, p_ij, p_o])


def estimate_probs_vec(wi, wj, w_ij, w_tot, cpw, freqvec):
    """Product over wells sets (R vector recycling) of the multinomial probabilities."""
    wi, wj, w_ij, w_tot, cpw = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (wi, wj, w_ij, w_tot, cpw))
    n = len(wi)
    p = 1.0
    for i in range(n):
        p *= _estimate_probs(wi[i], wj[i % len(wj)], w_ij[i % len(w_ij)], w_tot[i % len(w_tot)],
                             cpw[i % len(cpw)], list(np.asarray(freqvec, dtype=float)))
    return p


def wij_adjustment(wij, w_t, c, freq_i, freq_j):
    est = w_t * (1 - r_pow(1 - freq_i, c)) * (1 - r_pow(1 - freq_j, c))
    est = wij - est
    return max(est, 0.0) if not math.isnan(est) else est


def estimate_match_frequencies(wi, wj, w_ij, w_tot, cpw, alpha):
    w_o = w_tot - (wi + wj + w_ij)
    fi = find_freq(wi, wi + w_o, cpw, alpha)
    fj = find_freq(wj, wj + w_o, cpw, alpha)
    clonal = wij_adjustment(w_ij, w_tot, cpw, fi, fj)
    fij = find_freq(clonal, w_tot - w_ij + clonal, cpw, alpha)
    return [fi, fj, fij]


def estimate_nonmatch_frequencies(wi, wj, w_ij, w_tot, cpw, alpha):
    return [find_freq(wi + w_ij, w_tot, cpw, alpha), find_freq(wj + w_ij, w_tot, cpw, alpha), 0.0]


def _estimate_pair_prob_py(wi, wj, w_ij, w_tot, cpw, alpha, prior=1.0):
    m = _estimate_probs(wi, wj, w_ij, w_tot, cpw, estimate_match_frequencies(wi, wj, w_ij, w_tot, cpw, alpha))
    n = _estimate_probs(wi, wj, w_ij, w_tot, cpw, estimate_nonmatch_frequencies(wi, wj, w_ij, w_tot, cpw, alpha))
    with np.errstate(all="ignore"):
        return float(np.float64(prior) * np.float64(m) / np.float64(n))


def estimate_pair_prob(wi, wj, w_ij, w_tot, cpw, alpha, prior=1):
    """Likelihood ratio (linear scale) that wells (wi, wj, w_ij) come from a true pair (R semantics)."""
    c = core()
    args = (float(wi), float(wj), float(w_ij), float(w_tot), float(cpw), float(alpha), float(prior))
    if c is not None:
        return c.estimate_pair_prob(*args)
    return _estimate_pair_prob_py(*args)


def estimate_pair_prob_batch(wi, wj, w_ij, w_tot, cpw, alpha, prior=1, **_ignored):
    """Vectorised estimate_pair_prob (linear scale), computed in parallel by Rust when available."""
    wi = np.ascontiguousarray(wi, dtype=np.float64)
    wj = np.ascontiguousarray(wj, dtype=np.float64)
    w_ij = np.ascontiguousarray(w_ij, dtype=np.float64)
    c = core()
    if c is not None:
        return c.estimate_pair_prob_many(wi, wj, w_ij, float(w_tot), float(cpw), float(alpha), float(prior))
    return np.array([_estimate_pair_prob_py(a, b, d, float(w_tot), float(cpw), float(alpha), float(prior))
                     for a, b, d in zip(wi, wj, w_ij)], dtype=float)


def _log10(p):
    with np.errstate(divide="ignore", invalid="ignore"):
        return math.log10(p) if p > 0 else (-math.inf if p == 0 else math.nan)


def _surface_py(n_wells, cells, alpha, prior):
    n = int(n_wells)
    np1 = n + 1
    cap = 10 * np1 + 10

    def lp(wi, wj, wij):
        return _log10(_estimate_pair_prob_py(wi, wj, wij, n, cells, alpha, prior))

    cube = np.zeros((np1, np1))
    i = 1
    for wij in range(n, -1, -1):
        if lp(i - 1, 0, wij) > 0.1:
            while i <= cap and lp(i - 1, 0, wij) > 0.1:
                i += 1
        else:
            while i > 1 and lp(i - 1, 0, wij) < 0.1:
                i -= 1
        ans = np.zeros(max(np1, i))
        z = 1
        for j in range(i, 0, -1):
            while z <= cap and lp(j - 1, z - 1, wij) > 0.1:
                z += 1
            ans[j - 1] = z - 1
        cube[wij, :] = ans[:np1]
    return cube


def madhyper_surface(n_wells, cells=1000, alpha=2, prior=1):
    """MAD-HYPE decision table: mdh[wij, wi] = smallest wj that is *not* called a pair."""
    c = core()
    if c is not None:
        return c.madhyper_surface(int(n_wells), float(cells), float(alpha), float(prior))
    return _surface_py(int(n_wells), float(cells), float(alpha), float(prior))


def get_input(vec, vec2):
    vec = np.asarray(vec, dtype=bool)
    vec2 = np.asarray(vec2, dtype=bool)
    return np.array([np.sum(vec & ~vec2), np.sum(vec2 & ~vec), np.sum(vec & vec2)])


def get_input_mat(mat, mat2):
    mat = np.asarray(mat, dtype=bool)
    mat2 = np.asarray(mat2, dtype=bool)
    return np.column_stack([np.sum(mat & ~mat2, axis=1), np.sum(mat2 & ~mat, axis=1), np.sum(mat & mat2, axis=1)])
