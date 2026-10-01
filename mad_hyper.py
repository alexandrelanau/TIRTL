"""
Python port of TIRTL/mad_hyper.R.

The functions intentionally keep the R function names/signatures as closely as
practical so the algorithm can be compared line-by-line with the original.
"""

import math
from scipy.special import gammaln
import numpy as np


def well_freq(f, c):
    return 1.0 - np.power(1.0 - np.asarray(f, dtype=float), np.asarray(c, dtype=float))


def multinom_coeff(vec):
    # Kept for API parity with R.  Use log-gamma to avoid integer overflow.
    vec = np.asarray(vec, dtype=float)
    return math.exp(math.lgamma(np.sum(vec) + 1.0) -
                    np.sum([math.lgamma(x + 1.0) for x in vec]))


def _dmultinom_log(x, prob):
    x = np.asarray(x, dtype=float)
    prob = np.asarray(prob, dtype=float)
    if np.any(prob < 0):
        return -np.inf
    if np.any((prob == 0) & (x > 0)):
        return -np.inf
    n = np.sum(x)
    logcoef = math.lgamma(n + 1.0) - np.sum([math.lgamma(v + 1.0) for v in x])
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.sum(np.where(x > 0, x * np.log(prob), 0.0))
    return logcoef + logp


def estimate_probs_vec(wi, wj, w_ij, w_tot, cpw, freqvec):
    """
    R-compatible multinomial probability calculation.

    The original R implementation naturally accepts both scalars and vectors.
    NumPy needs explicit broadcasting so that scalar inputs such as
    wi=10, wj=5, w_ij=2 work correctly.
    """
    wi = np.atleast_1d(np.asarray(wi, dtype=float))
    wj = np.atleast_1d(np.asarray(wj, dtype=float))
    w_ij = np.atleast_1d(np.asarray(w_ij, dtype=float))
    w_tot = np.atleast_1d(np.asarray(w_tot, dtype=float))
    cpw = np.atleast_1d(np.asarray(cpw, dtype=float))
    freqvec = np.atleast_1d(np.asarray(freqvec, dtype=float))

    # Make scalar inputs behave like R's vector recycling for the
    # single-well case.
    n = max(
        wi.size,
        wj.size,
        w_ij.size,
        w_tot.size,
        cpw.size,
    )

    def recycle(x):
        if x.size == n:
            return x
        if x.size == 1:
            return np.full(n, x.item(), dtype=float)
        if n % x.size == 0:
            return np.resize(x, n)
        raise ValueError(
            f"Incompatible vector lengths: {x.size} cannot be "
            f"broadcast to {n}"
        )

    wi = recycle(wi)
    wj = recycle(wj)
    w_ij = recycle(w_ij)
    w_tot = recycle(w_tot)
    cpw = recycle(cpw)

    # freqvec is the frequency vector [i, j, ij].
    # It should normally contain 3 elements.
    if freqvec.size != 3:
        raise ValueError(
            f"freqvec must contain 3 frequencies, got {freqvec.size}"
        )

    log_total = 0.0

    for i in range(n):
        wo = w_tot[i] - (wi[i] + wj[i] + w_ij[i])

        fs = np.asarray(
            well_freq(freqvec, cpw[i]),
            dtype=float,
        )

        p_o = (
            (1.0 - fs[0]) *
            (1.0 - fs[1]) *
            (1.0 - fs[2])
        )

        p_i = (
            fs[0] *
            (1.0 - fs[1]) *
            (1.0 - fs[2])
        )

        p_j = (
            fs[1] *
            (1.0 - fs[0]) *
            (1.0 - fs[2])
        )

        p_ij = 1.0 - p_o - p_i - p_j

        log_p = _dmultinom_log(
            [wi[i], wj[i], w_ij[i], wo],
            [p_i, p_j, p_ij, p_o],
        )

        log_total += log_p

    if not np.isfinite(log_total):
        return 0.0

    if log_total < math.log(np.finfo(float).tiny):
        return 0.0

    return math.exp(log_total)


def derivative_prob_function(f, alpha, W, w, c):
    """
    Direct translation of derivative_prob_function().

    W, w and c may be scalars or vectors.  R's sum() behavior is retained.
    """
    f = float(f)
    alpha = float(alpha)
    W = np.asarray(W, dtype=float)
    w = np.asarray(w, dtype=float)
    c = np.asarray(c, dtype=float)

    if f == 0:
        return -(alpha * (1 - f)) + np.sum(f * c * (W - w) - w)
    if f == 1:
        return np.sum(f * c * (W - w))

    one_minus = 1.0 - f
    denom = np.power(one_minus, np.maximum(c, 0.001)) - 1.0
    term = c * ((W - w) + w * np.power(one_minus, c) / denom)
    return -(alpha * (1 - f)) + f * np.sum(term)


def _bisect_root(fn, lower=0.0, upper=1.0, tol=1e-9, max_iter=200):
    """Small dependency-free equivalent of R's uniroot() for this use case."""
    fl = float(fn(lower))
    fu = float(fn(upper))

    if not np.isfinite(fl) or not np.isfinite(fu):
        # Match the intended [0,1] bounded search while being robust to
        # endpoint numerical singularities.
        lower = max(lower, 1e-12)
        upper = min(upper, 1.0 - 1e-12)
        fl = float(fn(lower))
        fu = float(fn(upper))

    if abs(fl) < tol:
        return lower
    if abs(fu) < tol:
        return upper

    if fl * fu > 0:
        # Numerical edge cases can leave the exact R bracket without a sign
        # change. Search a dense bounded grid before failing.
        grid = np.linspace(lower, upper, 1001)
        vals = np.array([fn(x) for x in grid], dtype=float)
        good = np.isfinite(vals)
        for k in range(len(grid) - 1):
            if good[k] and good[k + 1] and vals[k] * vals[k + 1] <= 0:
                lower, upper = grid[k], grid[k + 1]
                fl, fu = vals[k], vals[k + 1]
                break
        else:
            return float(np.clip(grid[np.nanargmin(np.abs(vals))], 0, 1))

    for _ in range(max_iter):
        mid = (lower + upper) / 2.0
        fm = float(fn(mid))
        if not np.isfinite(fm):
            mid = np.nextafter(mid, upper)
            fm = float(fn(mid))
        if abs(fm) < tol or (upper - lower) < tol:
            return mid
        if fl * fm <= 0:
            upper, fu = mid, fm
        else:
            lower, fl = mid, fm
    return (lower + upper) / 2.0


def find_freq(w, w_tot=96, cpw=1000, a=1):
    """R equivalent of find_freq()."""
    return _bisect_root(
        lambda x: derivative_prob_function(x, a, w_tot, w, cpw),
        lower=0.0,
        upper=1.0,
        tol=1e-9,
    )

def _derivative_prob_batch(f, alpha, W, w, c):
    """
    Vectorized version of derivative_prob_function().

    f, W and w are arrays with the same shape.
    c is normally scalar in TIRTL.
    """
    f = np.asarray(f, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    c = float(c)
    alpha = float(alpha)

    out = np.empty_like(f)

    # Same behavior as derivative_prob_function()
    mask0 = f <= 0.0
    mask1 = f >= 1.0
    mask = ~(mask0 | mask1)

    # R code:
    # if (f == 0) {
    #   -(alpha * (1 - f)) + sum(f * c * (W - w) - w)
    # }
    out[mask0] = -alpha - w[mask0]

    # R code:
    # if (f == 1) {
    #   sum(f * c * (W - w))
    # }
    out[mask1] = c * (W[mask1] - w[mask1])

    if np.any(mask):
        fm = f[mask]
        Wm = W[mask]
        wm = w[mask]

        one_minus = 1.0 - fm

        denominator = (
            np.power(one_minus, max(c, 0.001)) - 1.0
        )

        term = c * (
            (Wm - wm)
            + wm * np.power(one_minus, c) / denominator
        )

        out[mask] = (
            -alpha * (1.0 - fm)
            + fm * term
        )

    return out

def find_freq_batch(
    w,
    W,
    cpw=1000,
    alpha=1,
    tol=1e-9,
    max_iter=100,
):
    """
    Vectorized equivalent of find_freq().

    Solves all roots simultaneously using NumPy bisection.

    Parameters
    ----------
    w : array-like
        Observed counts.

    W : array-like
        Total counts.

    cpw : float
        Cells per well / clone threshold.

    alpha : float
        MAD-HYPE alpha.

    Returns
    -------
    np.ndarray
        One frequency estimate per input pair.
    """

    w = np.asarray(w, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    w, W = np.broadcast_arrays(w, W)

    shape = w.shape

    w = w.ravel()
    W = W.ravel()

    lower = np.zeros_like(w)
    upper = np.ones_like(w)

    # ---------------------------------------------------------
    # Evaluate endpoints
    # ---------------------------------------------------------

    f_lower = _derivative_prob_batch(
        lower,
        alpha,
        W,
        w,
        cpw,
    )

    f_upper = _derivative_prob_batch(
        upper,
        alpha,
        W,
        w,
        cpw,
    )

    # Roots exactly at boundaries
    result = np.empty_like(w)

    done_lower = np.abs(f_lower) < tol
    done_upper = np.abs(f_upper) < tol

    result[done_lower] = lower[done_lower]
    result[done_upper] = upper[done_upper]

    active = ~(done_lower | done_upper)

    # ---------------------------------------------------------
    # Vectorized bisection
    # ---------------------------------------------------------

    for _ in range(max_iter):

        if not np.any(active):
            break

        mid = (
            lower[active]
            + upper[active]
        ) * 0.5

        fm = _derivative_prob_batch(
            mid,
            alpha,
            W[active],
            w[active],
            cpw,
        )

        lo = lower[active]
        hi = upper[active]

        fl = _derivative_prob_batch(
            lo,
            alpha,
            W[active],
            w[active],
            cpw,
        )

        # Root reached
        exact = np.abs(fm) < tol

        # Converged interval
        narrow = (hi - lo) < tol

        finished = exact | narrow

        # Write finished roots
        if np.any(finished):
            active_indices = np.flatnonzero(active)

            result[
                active_indices[finished]
            ] = mid[finished]

        # Remaining roots
        keep = ~finished

        if not np.any(keep):
            active[:] = False
            break

        active_indices = np.flatnonzero(active)

        keep_indices = active_indices[keep]

        fm_keep = fm[keep]
        fl_keep = fl[keep]
        mid_keep = mid[keep]

        # Standard bisection:
        #
        # if fl * fm <= 0:
        #     root is [lo, mid]
        #
        # otherwise:
        #     root is [mid, hi]

        left = fl_keep * fm_keep <= 0.0

        lower[keep_indices[left]] = (
            lower[keep_indices[left]]
        )

        upper[keep_indices[left]] = (
            mid_keep[left]
        )

        lower[keep_indices[~left]] = (
            mid_keep[~left]
        )

        # upper unchanged for right side

        active[:] = False
        active[keep_indices] = True

    # Anything still active after max_iter
    if np.any(active):
        result[active] = (
            lower[active] + upper[active]
        ) * 0.5

    return result.reshape(shape)

def _estimate_probs_batch(
    wi,
    wj,
    w_ij,
    w_tot,
    cpw,
    freqvec,
):
    """
    Vectorized multinomial probability calculation.

    Parameters
    ----------
    wi, wj, w_ij : arrays of length N

    w_tot : scalar or array

    cpw : scalar or array

    freqvec : shape (3, N)

    Returns
    -------
    np.ndarray
        Probability for each of N combinations.
    """

    wi = np.asarray(wi, dtype=np.float64)
    wj = np.asarray(wj, dtype=np.float64)
    w_ij = np.asarray(w_ij, dtype=np.float64)

    w_tot = np.asarray(w_tot, dtype=np.float64)
    cpw = np.asarray(cpw, dtype=np.float64)

    freqvec = np.asarray(
        freqvec,
        dtype=np.float64,
    )

    if freqvec.ndim != 2 or freqvec.shape[0] != 3:
        raise ValueError(
            "freqvec must have shape (3, N)"
        )

    fi = freqvec[0]
    fj = freqvec[1]
    fij = freqvec[2]

    # ---------------------------------------------------------
    # Probability of seeing each clone in a well
    # ---------------------------------------------------------

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):

        oi = np.power(
            1.0 - fi,
            cpw,
        )

        oj = np.power(
            1.0 - fj,
            cpw,
        )

        oij = np.power(
            1.0 - fij,
            cpw,
        )

    # Probability of:
    #
    # none
    p_o = oi * oj * oij

    # i only
    p_i = (
        (1.0 - oi)
        * oj
        * oij
    )

    # j only
    p_j = (
        (1.0 - oj)
        * oi
        * oij
    )

    # i and j
    p_ij = (
        1.0
        - p_o
        - p_i
        - p_j
    )

    # ---------------------------------------------------------
    # Observed counts
    # ---------------------------------------------------------

    wo = (
        w_tot
        - wi
        - wj
        - w_ij
    )

    # ---------------------------------------------------------
    # Multinomial coefficient
    #
    # log(n!) - log(wi!) - log(wj!)
    # - log(wij!) - log(wo!)
    # ---------------------------------------------------------

    logp = (
        gammaln(w_tot + 1.0)
        - gammaln(wi + 1.0)
        - gammaln(wj + 1.0)
        - gammaln(w_ij + 1.0)
        - gammaln(wo + 1.0)
    )

    # ---------------------------------------------------------
    # Probability terms
    # ---------------------------------------------------------

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        logp += np.where(
            wi > 0,
            wi * np.log(p_i),
            0.0,
        )

        logp += np.where(
            wj > 0,
            wj * np.log(p_j),
            0.0,
        )

        logp += np.where(
            w_ij > 0,
            w_ij * np.log(p_ij),
            0.0,
        )

        logp += np.where(
            wo > 0,
            wo * np.log(p_o),
            0.0,
        )

    with np.errstate(
        under="ignore",
        over="ignore",
        invalid="ignore",
    ):
        probability = np.exp(logp)

    probability[~np.isfinite(probability)] = 0.0

    return probability


def estimate_nonmatch_frequencies(wi, wj, w_ij, w_tot, cpw, alpha):
    wi = np.asarray(wi, dtype=float)
    wj = np.asarray(wj, dtype=float)
    w_ij = np.asarray(w_ij, dtype=float)
    w_tot = np.asarray(w_tot, dtype=float)
    freqs_i = find_freq(wi + w_ij, w_tot, cpw, alpha)
    freqs_j = find_freq(wj + w_ij, w_tot, cpw, alpha)
    return np.asarray([freqs_i, freqs_j, 0.0], dtype=float)


def wij_adjustment(wij, w_t, c, freq_i, freq_j):
    w_ij_est = w_t * (
        1.0 - np.power(1.0 - np.asarray(freq_i), c)
    ) * (
        1.0 - np.power(1.0 - np.asarray(freq_j), c)
    )
    w_ij_est = np.asarray(wij, dtype=float) - w_ij_est
    return np.maximum(w_ij_est, 0.0)


def estimate_match_frequencies(wi, wj, w_ij, w_tot, cpw, alpha):
    wi = np.asarray(wi, dtype=float)
    wj = np.asarray(wj, dtype=float)
    w_ij = np.asarray(w_ij, dtype=float)
    w_tot = np.asarray(w_tot, dtype=float)

    w_o = w_tot - (wi + wj + w_ij)
    freqs_i = find_freq(wi, wi + w_o, cpw, alpha)
    freqs_j = find_freq(wj, wj + w_o, cpw, alpha)
    wij_clonal = wij_adjustment(w_ij, w_tot, cpw, freqs_i, freqs_j)
    freqs_ij = find_freq(
        wij_clonal,
        w_tot - w_ij + wij_clonal,
        cpw,
        alpha,
    )
    return np.asarray([freqs_i, freqs_j, freqs_ij], dtype=float)


def estimate_pair_prob(wi, wj, w_ij, w_tot, cpw, alpha, prior=1):
    """R equivalent of estimate_pair_prob()."""
    fr_match = estimate_match_frequencies(wi, wj, w_ij, w_tot, cpw, alpha)
    fr_nonmatch = estimate_nonmatch_frequencies(
        wi, wj, w_ij, w_tot, cpw, alpha
    )
    match_probs = estimate_probs_vec(
        wi, wj, w_ij, w_tot, cpw, fr_match
    )
    nonmatch_probs = estimate_probs_vec(
        wi, wj, w_ij, w_tot, cpw, fr_nonmatch
    )
    if nonmatch_probs == 0:
        return np.inf if match_probs > 0 else 0.0
    return float(prior) * match_probs / nonmatch_probs

def estimate_pair_prob_batch(
    wi,
    wj,
    w_ij,
    w_tot,
    cpw,
    alpha,
    prior=1.0,
):
    """
    Vectorized equivalent of estimate_pair_prob().

    Calculates all unique TIRTL pair probabilities in one batch.
    """

    wi = np.asarray(wi, dtype=np.float64)
    wj = np.asarray(wj, dtype=np.float64)
    w_ij = np.asarray(w_ij, dtype=np.float64)

    w_tot = float(w_tot)
    cpw = float(cpw)
    alpha = float(alpha)

    # ---------------------------------------------------------
    # NON-MATCH
    # ---------------------------------------------------------

    nonmatch_i = find_freq_batch(
        wi + w_ij,
        w_tot,
        cpw,
        alpha,
    )

    nonmatch_j = find_freq_batch(
        wj + w_ij,
        w_tot,
        cpw,
        alpha,
    )

    nonmatch_freq = np.vstack((
        nonmatch_i,
        nonmatch_j,
        np.zeros_like(nonmatch_i),
    ))

    # ---------------------------------------------------------
    # MATCH
    # ---------------------------------------------------------

    w_o = (
        w_tot
        - wi
        - wj
        - w_ij
    )

    match_i = find_freq_batch(
        wi,
        wi + w_o,
        cpw,
        alpha,
    )

    match_j = find_freq_batch(
        wj,
        wj + w_o,
        cpw,
        alpha,
    )

    # ---------------------------------------------------------
    # Estimate clonal wij
    # ---------------------------------------------------------

    w_ij_est = (
        w_tot
        * (
            1.0
            - np.power(
                1.0 - match_i,
                cpw,
            )
        )
        * (
            1.0
            - np.power(
                1.0 - match_j,
                cpw,
            )
        )
    )

    wij_clonal = (
        w_ij
        - w_ij_est
    )

    wij_clonal = np.maximum(
        wij_clonal,
        0.0,
    )

    # ---------------------------------------------------------
    # Match frequency for ij
    # ---------------------------------------------------------

    match_ij = find_freq_batch(
        wij_clonal,
        w_tot - w_ij + wij_clonal,
        cpw,
        alpha,
    )

    match_freq = np.vstack((
        match_i,
        match_j,
        match_ij,
    ))

    # ---------------------------------------------------------
    # Multinomial probabilities
    # ---------------------------------------------------------

    match_probs = _estimate_probs_batch(
        wi,
        wj,
        w_ij,
        w_tot,
        cpw,
        match_freq,
    )

    nonmatch_probs = _estimate_probs_batch(
        wi,
        wj,
        w_ij,
        w_tot,
        cpw,
        nonmatch_freq,
    )

    # ---------------------------------------------------------
    # Match / non-match ratio
    # ---------------------------------------------------------

    result = np.zeros_like(match_probs)

    valid = nonmatch_probs > 0

    result[valid] = (
        float(prior)
        * match_probs[valid]
        / nonmatch_probs[valid]
    )

    inf_mask = (
        ~valid
        & (match_probs > 0)
    )

    result[inf_mask] = np.inf

    return result


def get_input(vec, vec2):
    """R equivalent of get_input()."""
    a = np.asarray(vec, dtype=bool)
    b = np.asarray(vec2, dtype=bool)
    wij = np.sum(a & b)
    wj = np.sum(~a & b)
    wi = np.sum(a & ~b)
    return np.asarray([wi, wj, wij], dtype=int)


def get_input_mat(mat, mat2):
    """R equivalent of get_input_mat()."""
    a = np.asarray(mat, dtype=bool)
    b = np.asarray(mat2, dtype=bool)
    wij = np.sum(a & b, axis=1)
    wj = np.sum(b & ~a, axis=1)
    wi = np.sum(a & ~b, axis=1)
    return np.column_stack((wi, wj, wij))
