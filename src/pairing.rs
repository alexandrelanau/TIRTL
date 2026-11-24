//! Pairing kernels.
//!
//! MAD-HYPE: well-occupancy bitsets, fused popcount-overlap + look-up-table test (no dense
//! intermediate). Integer arithmetic -> exactly the pairs of the original backend.
//!
//! T-SHELL: top-3 Pearson correlations per alpha, exact in float64. Candidates come from a
//! float32 matrix product done by any backend (BLAS / cupy / mlx); they are re-scored in float64
//! here, and a rigorous float32 error bound proves that no non-candidate can enter the top-3
//! (otherwise the row is recomputed fully in float64). Ties: lower beta index first.

use rayon::prelude::*;
use std::cmp::Ordering;

pub struct Bits {
    pub words: usize,
    pub data: Vec<u64>,
    pub total: Vec<u32>,
}

/// occupancy bitsets of a row-major (n x w) matrix: bit set when value > 0 (as float32, like the
/// original backend which loads the matrix as float32)
pub fn pack_bits(mat: &[f64], n: usize, w: usize) -> Bits {
    let words = (w + 63) / 64;
    let mut data = vec![0u64; n * words];
    data.par_chunks_mut(words.max(1)).enumerate().for_each(|(i, row)| {
        if words == 0 {
            return;
        }
        let src = &mat[i * w..(i + 1) * w];
        for (k, &x) in src.iter().enumerate() {
            if (x as f32) > 0.0 {
                row[k / 64] |= 1u64 << (k % 64);
            }
        }
    });
    let total = (0..n)
        .into_par_iter()
        .map(|i| data[i * words..(i + 1) * words].iter().map(|x| x.count_ones()).sum())
        .collect();
    Bits { words, data, total }
}

#[inline(always)]
fn overlap(a: &[u64], b: &[u64]) -> u32 {
    let mut s = 0u32;
    for k in 0..a.len() {
        s += (a[k] & b[k]).count_ones();
    }
    s
}

/// pairs with (wb - wij) < mdh[wij, wa - wij]; output sorted by (alpha, beta)
pub fn madhype(a: &Bits, b: &Bits, mdh: &[i64], n_wells: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let np1 = n_wells + 1;
    let na = a.total.len();
    let nb = b.total.len();
    let words = a.words;
    assert_eq!(words, b.words);
    // a pair needs mdh[wij, wi] > wj >= 0: skip alpha whose row of possible tables is all <= 0
    let res: Vec<Vec<(u32, u32, u32)>> = (0..na)
        .into_par_iter()
        .with_min_len(16)
        .map(|i| {
            let ai = &a.data[i * words..(i + 1) * words];
            let wa = a.total[i] as usize;
            let mut out = Vec::new();
            // max over wij of mdh[wij, wa-wij] + wij bounds wb for any pair of this alpha
            let mut wb_max: i64 = -1;
            for wij in 0..=wa.min(n_wells) {
                let wi = wa - wij;
                if wi < np1 {
                    wb_max = wb_max.max(mdh[wij * np1 + wi] - 1 + wij as i64);
                }
            }
            if wb_max < 0 {
                return out;
            }
            for j in 0..nb {
                let wb = b.total[j] as i64;
                if wb > wb_max {
                    continue;
                }
                let ov = overlap(ai, &b.data[j * words..(j + 1) * words]) as usize;
                let wi = wa - ov;
                let wj = wb - ov as i64;
                if wj < mdh[ov * np1 + wi] {
                    out.push((i as u32, j as u32, ov as u32));
                }
            }
            out
        })
        .collect();
    let n: usize = res.iter().map(|v| v.len()).sum();
    let (mut ai, mut bi, mut ov) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for v in res {
        for (x, y, z) in v {
            ai.push(x);
            bi.push(y);
            ov.push(z);
        }
    }
    (ai, bi, ov)
}

pub fn overlaps_of(a: &Bits, b: &Bits, ai: &[u32], bi: &[u32]) -> Vec<u32> {
    let w = a.words;
    ai.par_iter()
        .zip(bi.par_iter())
        .map(|(&i, &j)| {
            overlap(&a.data[i as usize * w..(i as usize + 1) * w], &b.data[j as usize * w..(j as usize + 1) * w])
        })
        .collect()
}

/// rows centred and scaled to unit L2 norm (float64); NaN rows when the norm is 0
pub fn normalize_rows(mat: &[f64], n: usize, w: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * w];
    out.par_chunks_mut(w.max(1)).enumerate().for_each(|(i, row)| {
        if w == 0 {
            return;
        }
        let src = &mat[i * w..(i + 1) * w];
        let mean = src.iter().sum::<f64>() / w as f64;
        let mut ss = 0.0;
        for k in 0..w {
            let d = src[k] - mean;
            row[k] = d;
            ss += d * d;
        }
        let nrm = ss.sqrt();
        for x in row.iter_mut() {
            *x /= nrm;
        }
    });
    out
}

#[inline(always)]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    // fixed summation order (4 interleaved partial sums) -> identical result on every machine
    let n = a.len();
    let mut s = [0.0f64; 4];
    let mut k = 0;
    while k + 4 <= n {
        s[0] += a[k] * b[k];
        s[1] += a[k + 1] * b[k + 1];
        s[2] += a[k + 2] * b[k + 2];
        s[3] += a[k + 3] * b[k + 3];
        k += 4;
    }
    while k < n {
        s[0] += a[k] * b[k];
        k += 1;
    }
    (s[0] + s[1]) + (s[2] + s[3])
}

/// ranking key: higher r first, NaN last, then lower index
#[inline]
fn better(r1: f64, j1: u32, r2: f64, j2: u32) -> bool {
    match (r1.is_nan(), r2.is_nan()) {
        (true, true) => j1 < j2,
        (true, false) => false,
        (false, true) => true,
        _ => r1 > r2 || (r1 == r2 && j1 < j2),
    }
}

fn top_insert(top: &mut Vec<(f64, u32)>, k: usize, r: f64, j: u32) {
    if top.len() == k {
        let (lr, lj) = top[k - 1];
        if !better(r, j, lr, lj) {
            return;
        }
        top.pop();
    }
    let pos = top.iter().position(|&(tr, tj)| better(r, j, tr, tj)).unwrap_or(top.len());
    top.insert(pos, (r, j));
}

/// top-k column indices per row of a float32 row-major matrix (n x m); descending, ties -> lower
/// index, NaN last. Returns (idx n*k, values n*k).
pub fn topk_rows_f32(mat: &[f32], n: usize, m: usize, k: usize) -> (Vec<i64>, Vec<f32>) {
    let k = k.min(m);
    let mut idx = vec![0i64; n * k];
    let mut val = vec![0f32; n * k];
    idx.par_chunks_mut(k.max(1))
        .zip(val.par_chunks_mut(k.max(1)))
        .enumerate()
        .for_each(|(i, (ri, rv))| {
            if k == 0 {
                return;
            }
            let row = &mat[i * m..(i + 1) * m];
            let mut top: Vec<(f64, u32)> = Vec::with_capacity(k + 1);
            for (j, &x) in row.iter().enumerate() {
                let x = x as f64;
                if top.len() == k {
                    let (lr, lj) = top[k - 1];
                    if !better(x, j as u32, lr, lj) {
                        continue;
                    }
                }
                top_insert(&mut top, k, x, j as u32);
            }
            for (t, &(r, j)) in top.iter().enumerate() {
                ri[t] = j as i64;
                rv[t] = r as f32;
            }
        });
    (idx, val)
}

pub struct TshellOut {
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    pub r: Vec<f64>,
    pub n_fallback: usize,
}

/// Exact float64 top-3 for alpha rows `rows` (global indices into anorm), given float32 candidates
/// (cand_idx/cand_val: rows.len() x kc, sorted descending). `bound` = max |r32 - r64|.
pub fn tshell_refine(anorm: &[f64], bnorm: &[f64], w: usize, nb: usize, rows: &[u32], cand_idx: &[i64],
                     cand_val: &[f32], kc: usize, bound: f64, top: usize) -> TshellOut {
    let res: Vec<(Vec<(f64, u32)>, bool)> = rows
        .par_iter()
        .enumerate()
        .map(|(t, &ar)| {
            let av = &anorm[ar as usize * w..(ar as usize + 1) * w];
            let r64 = |j: usize| dot(av, &bnorm[j * w..(j + 1) * w]);
            let mut best: Vec<(f64, u32)> = Vec::with_capacity(top + 1);
            let ci = &cand_idx[t * kc..(t + 1) * kc];
            let cv = &cand_val[t * kc..(t + 1) * kc];
            for &j in ci {
                top_insert(&mut best, top, r64(j as usize), j as u32);
            }
            let complete = kc >= nb; // every beta was a candidate
            let mut ok = complete;
            if !complete && best.len() == top {
                // any non-candidate has r32 <= min candidate r32 (NaN-free rows), so r64 <= that + bound
                let thr = cv.iter().map(|&x| x as f64).fold(f64::INFINITY, f64::min) + bound;
                let last = best[top - 1].0;
                ok = !last.is_nan() && thr.is_finite() && last > thr;
            }
            if ok {
                (best, false)
            } else {
                let mut full: Vec<(f64, u32)> = Vec::with_capacity(top + 1);
                for j in 0..nb {
                    top_insert(&mut full, top, r64(j), j as u32);
                }
                (full, true)
            }
        })
        .collect();
    let mut out = TshellOut { a: vec![], b: vec![], r: vec![], n_fallback: 0 };
    for (t, (best, fb)) in res.into_iter().enumerate() {
        if fb {
            out.n_fallback += 1;
        }
        // output ordered by beta index within an alpha (np.argwhere order of the original backend)
        let mut v = best;
        v.sort_by(|x, y| x.1.cmp(&y.1));
        for (r, j) in v {
            out.a.push(rows[t]);
            out.b.push(j);
            out.r.push(r);
        }
    }
    out
}

/// float32 error bound for <x,y> of unit vectors of length w computed with any summation order
pub fn f32_dot_bound(w: usize) -> f64 {
    let u = (2.0f64).powi(-24);
    let n = w as f64 + 4.0;
    2.0 * (n * u / (1.0 - n * u)) + 1e-7
}

#[allow(dead_code)]
pub fn cmp_desc(a: f64, b: f64) -> Ordering {
    b.partial_cmp(&a).unwrap_or(Ordering::Equal)
}
