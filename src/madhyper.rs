//! MAD-HYPE pair probability, reproducing the original R code (mad_hyper.R) operation by operation:
//! R's `^` (R_pow), `uniroot()` (R_zeroin2, Brent), `dmultinom()` and `sum()` (long-double accumulation).
//! Anything that would make R stop with an error yields NaN here.

use rayon::prelude::*;

/// R_pow (arithmetic.c): x ^ y
#[inline]
pub fn r_pow(x: f64, y: f64) -> f64 {
    if y == 2.0 {
        return x * x;
    }
    if x == 1.0 || y == 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        return if y > 0.0 {
            0.0
        } else if y < 0.0 {
            f64::INFINITY
        } else {
            y
        };
    }
    if x.is_finite() && y.is_finite() {
        return x.powf(y);
    }
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    if !x.is_finite() {
        if x > 0.0 {
            return if y < 0.0 { 0.0 } else { f64::INFINITY };
        } else if y.is_finite() && y == y.floor() {
            return if y < 0.0 {
                0.0
            } else if (y % 2.0) != 0.0 {
                x
            } else {
                -x
            };
        }
    }
    if !y.is_finite() && x >= 0.0 {
        return if y > 0.0 {
            if x >= 1.0 { f64::INFINITY } else { 0.0 }
        } else if x < 1.0 {
            f64::INFINITY
        } else {
            0.0
        };
    }
    f64::NAN
}

/// Sum emulating R's long-double accumulator (error-free transformation, then one rounding).
#[inline]
pub fn r_sum(xs: &[f64]) -> f64 {
    let mut s = 0.0f64;
    let mut c = 0.0f64;
    for &x in xs {
        let t = s + x;
        if s.abs() >= x.abs() {
            c += (s - t) + x;
        } else {
            c += (x - t) + s;
        }
        s = t;
    }
    s + c
}

/// derivative_prob_function(f, alpha, W, w, c) for scalar W, w, c
#[inline]
pub fn derivative(f: f64, alpha: f64, big_w: f64, w: f64, c: f64) -> f64 {
    if f == 0.0 {
        -(alpha * (1.0 - f)) + (f * c * (big_w - w) - w)
    } else if f == 1.0 {
        f * c * (big_w - w)
    } else {
        let omf = 1.0 - f;
        let cm = if c.is_nan() { c } else { c.max(0.001) };
        let inner = (big_w - w) + w * r_pow(omf, c) / (r_pow(omf, cm) - 1.0);
        -(alpha * (1.0 - f)) + f * (c * inner)
    }
}

/// R_zeroin2 (src/library/stats/src/zeroin.c)
pub fn zeroin2<F: Fn(f64) -> f64>(ax: f64, bx: f64, fa0: f64, fb0: f64, f: F, tol: f64, maxit_in: i32) -> f64 {
    let (mut a, mut b) = (ax, bx);
    let (mut fa, mut fb) = (fa0, fb0);
    let mut c = a;
    let mut fc = fa;
    let mut maxit = maxit_in + 1;
    if fa == 0.0 {
        return a;
    }
    if fb == 0.0 {
        return b;
    }
    while maxit > 0 {
        maxit -= 1;
        let prev_step = b - a;
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol_act = 2.0 * f64::EPSILON * b.abs() + tol / 2.0;
        let mut new_step = (c - b) / 2.0;
        if new_step.abs() <= tol_act || fb == 0.0 {
            return b;
        }
        if prev_step.abs() >= tol_act && fa.abs() > fb.abs() {
            let cb = c - b;
            let (mut p, mut q);
            if a == c {
                let t1 = fb / fa;
                p = cb * t1;
                q = 1.0 - t1;
            } else {
                q = fa / fc;
                let t1 = fb / fc;
                let t2 = fb / fa;
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0));
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0);
            }
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }
            if p < (0.75 * cb * q - (tol_act * q).abs() / 2.0) && p < (prev_step * q / 2.0).abs() {
                new_step = p / q;
            }
        }
        if new_step.abs() < tol_act {
            new_step = if new_step > 0.0 { tol_act } else { -tol_act };
        }
        a = b;
        fa = fb;
        b += new_step;
        fb = f(b);
        if (fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0) {
            c = a;
            fc = fa;
        }
    }
    b
}

#[inline]
fn fcn2(v: f64) -> f64 {
    if !v.is_finite() {
        if v == f64::NEG_INFINITY {
            -f64::MAX
        } else {
            f64::MAX
        }
    } else {
        v
    }
}

#[inline]
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x // 0 or NaN
    }
}

/// find_freq(w, w_tot, cpw, a) = uniroot(derivative, c(0,1), tol=1e-9)$root
pub fn find_freq(w: f64, w_tot: f64, cpw: f64, a: f64) -> f64 {
    let f = |x: f64| derivative(x, a, w_tot, w, cpw);
    let fl = f(0.0);
    let fu = f(1.0);
    if fl.is_nan() || fu.is_nan() {
        return f64::NAN;
    }
    if !(sign(fl) * sign(fu) <= 0.0) {
        return f64::NAN;
    }
    let tr = |x: f64| x.min(f64::MAX).max(-f64::MAX);
    zeroin2(0.0, 1.0, tr(fl), tr(fu), |x| fcn2(f(x)), 1e-9, 1000)
}

#[inline]
fn lgamma(x: f64) -> f64 {
    libm::lgamma(x)
}

/// dmultinom(x, prob) (log = FALSE)
pub fn dmultinom(x: [f64; 4], prob: [f64; 4]) -> f64 {
    if prob.iter().any(|p| !p.is_finite() || *p < 0.0) {
        return f64::NAN;
    }
    let s = r_sum(&prob);
    if s == 0.0 {
        return f64::NAN;
    }
    let mut p = [0.0; 4];
    let mut xi = [0i64; 4];
    for k in 0..4 {
        p[k] = prob[k] / s;
        let v = x[k] + 0.5;
        if v.is_nan() {
            return f64::NAN;
        }
        xi[k] = v.trunc() as i64;
        if xi[k] < 0 {
            return f64::NAN;
        }
    }
    let size: i64 = xi.iter().sum();
    let mut terms = [0.0f64; 4];
    let mut nt = 0;
    let mut n0 = 0;
    for k in 0..4 {
        if p[k] == 0.0 {
            n0 += 1;
            if xi[k] != 0 {
                return 0.0;
            }
            continue;
        }
        let xk = xi[k] as f64;
        terms[nt] = xk * p[k].ln() - lgamma(xk + 1.0);
        nt += 1;
    }
    if n0 == 4 {
        return 1.0;
    }
    let r = lgamma(size as f64 + 1.0) + r_sum(&terms[..nt]);
    r.exp()
}

#[inline]
fn well_freq(f: f64, c: f64) -> f64 {
    1.0 - r_pow(1.0 - f, c)
}

fn estimate_probs(wi: f64, wj: f64, wij: f64, w_tot: f64, cpw: f64, fr: [f64; 3]) -> f64 {
    let wo = w_tot - (wi + wj + wij);
    let fs = [well_freq(fr[0], cpw), well_freq(fr[1], cpw), well_freq(fr[2], cpw)];
    let p_o = (1.0 - fs[0]) * (1.0 - fs[1]) * (1.0 - fs[2]);
    let p_i = fs[0] * (1.0 - fs[1]) * (1.0 - fs[2]);
    let p_j = fs[1] * (1.0 - fs[0]) * (1.0 - fs[2]);
    let p_ij = 1.0 - p_o - p_i - p_j;
    1.0 * dmultinom([wi, wj, wij, wo], [p_i, p_j, p_ij, p_o])
}

/// estimate_pair_prob(wi, wj, w_ij, w_tot, cpw, alpha, prior)  (linear scale, as in R)
pub fn estimate_pair_prob(wi: f64, wj: f64, wij: f64, w_tot: f64, cpw: f64, alpha: f64, prior: f64) -> f64 {
    // estimate_match_frequencies
    let w_o = w_tot - (wi + wj + wij);
    let fi = find_freq(wi, wi + w_o, cpw, alpha);
    let fj = find_freq(wj, wj + w_o, cpw, alpha);
    let mut est = w_tot * (1.0 - r_pow(1.0 - fi, cpw)) * (1.0 - r_pow(1.0 - fj, cpw));
    est = wij - est;
    if est < 0.0 {
        est = 0.0;
    }
    let fij = find_freq(est, w_tot - wij + est, cpw, alpha);
    // estimate_nonmatch_frequencies
    let ni = find_freq(wi + wij, w_tot, cpw, alpha);
    let nj = find_freq(wj + wij, w_tot, cpw, alpha);
    let m = estimate_probs(wi, wj, wij, w_tot, cpw, [fi, fj, fij]);
    let n = estimate_probs(wi, wj, wij, w_tot, cpw, [ni, nj, 0.0]);
    prior * m / n
}

#[inline]
fn lp(wi: usize, wj: usize, wij: usize, n: f64, cells: f64, alpha: f64, prior: f64) -> f64 {
    estimate_pair_prob(wi as f64, wj as f64, wij as f64, n, cells, alpha, prior).log10()
}

/// madhyper_surface(n_wells, cells, alpha, prior): (n+1) x (n+1) row-major, row = wij, col = wi.
/// The `i` boundary is searched sequentially exactly like R; the per-row `z` sweeps (independent
/// once `i` is known) run in parallel.
pub fn surface(n_wells: usize, cells: f64, alpha: f64, prior: f64) -> Vec<f64> {
    let n = n_wells as f64;
    let np1 = n_wells + 1;
    let mut bounds = vec![0usize; np1]; // i for each wij
    let mut i: usize = 1;
    // R has no bound here; the cap only prevents an endless loop where R would have errored
    let cap = 10 * np1 + 10;
    for wij in (0..=n_wells).rev() {
        if lp(i - 1, 0, wij, n, cells, alpha, prior) > 0.1 {
            while i <= cap && lp(i - 1, 0, wij, n, cells, alpha, prior) > 0.1 {
                i += 1;
            }
        } else {
            while i > 1 && lp(i - 1, 0, wij, n, cells, alpha, prior) < 0.1 {
                i -= 1;
            }
        }
        bounds[wij] = i;
    }
    let rows: Vec<Vec<f64>> = (0..=n_wells)
        .into_par_iter()
        .map(|wij| {
            let i = bounds[wij];
            let mut ans = vec![0.0f64; np1.max(i)];
            let mut z: usize = 1;
            for j in (1..=i).rev() {
                while z <= 10 * np1 + 10 && lp(j - 1, z - 1, wij, n, cells, alpha, prior) > 0.1 {
                    z += 1;
                }
                ans[j - 1] = (z - 1) as f64;
            }
            ans.truncate(np1);
            ans
        })
        .collect();
    let mut out = vec![0.0f64; np1 * np1];
    for (wij, r) in rows.into_iter().enumerate() {
        out[wij * np1..(wij + 1) * np1].copy_from_slice(&r);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zeroin_simple() {
        let r = find_freq(10.0, 96.0, 1000.0, 2.0);
        assert!(r > 0.0 && r < 1.0);
    }
    #[test]
    fn rsum_exact() {
        assert_eq!(r_sum(&[1e16, 1.0, -1e16]), 1.0);
    }
}
