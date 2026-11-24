//! Fused sparse MAD-HYPE + T-SHELL kernel.
//!
//! Clones are present in a handful of wells, so for one alpha only the betas sharing at least one
//! well need an explicit dot product (inverted well -> betas index). Everything stays exact:
//!  * MAD-HYPE: integer overlaps; betas without a shared well (overlap 0) are tested through a
//!    wb-sorted list (pair iff wb < mdh[0, wa]).
//!  * T-SHELL: r = (sum_shared x*y - xbar * sum(y)) / (||x - xbar|| * ||y - ybar||) in float64;
//!    betas without a shared well have r = -xbar * sum(y) / (||xc|| ||yc||), ranked through the
//!    precomputed key sum(y)/||yc|| (ascending), with a safety margin against rounding.
//! Rows / betas used by T-SHELL are those present in more than `tshell_min_wells` wells.

use rayon::prelude::*;
use std::cell::RefCell;

pub struct SparseOut {
    pub m_a: Vec<u32>,
    pub m_b: Vec<u32>,
    pub m_ov: Vec<u32>,
    pub t_a: Vec<u32>,
    pub t_b: Vec<u32>,
    pub t_r: Vec<f64>,
    pub t_ov: Vec<u32>,
}

struct Csc {
    ptr: Vec<usize>,
    row: Vec<u32>,
    val: Vec<f64>,
    pres: Vec<bool>,
}

#[inline]
fn present(x: f64) -> bool {
    (x as f32) > 0.0
}

fn csc(mat: &[f64], n: usize, w: usize) -> Csc {
    let mut cnt = vec![0usize; w + 1];
    for i in 0..n {
        for k in 0..w {
            if mat[i * w + k] != 0.0 {
                cnt[k + 1] += 1;
            }
        }
    }
    for k in 0..w {
        cnt[k + 1] += cnt[k];
    }
    let nnz = cnt[w];
    let mut pos = cnt.clone();
    let mut row = vec![0u32; nnz];
    let mut val = vec![0f64; nnz];
    let mut pres = vec![false; nnz];
    for i in 0..n {
        for k in 0..w {
            let x = mat[i * w + k];
            if x != 0.0 {
                let p = pos[k];
                row[p] = i as u32;
                val[p] = x;
                pres[p] = present(x);
                pos[k] += 1;
            }
        }
    }
    Csc { ptr: cnt, row, val, pres }
}

/// per-row (sum, centred L2 norm, presence count), sequential sums in well order
fn row_stats(mat: &[f64], n: usize, w: usize) -> (Vec<f64>, Vec<f64>, Vec<u32>) {
    let v: Vec<(f64, f64, u32)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let r = &mat[i * w..(i + 1) * w];
            let mut s = 0.0;
            let mut c = 0u32;
            for &x in r {
                s += x;
                if present(x) {
                    c += 1;
                }
            }
            let mean = s / w as f64;
            let mut ss = 0.0;
            for &x in r {
                let d = x - mean;
                ss += d * d;
            }
            (s, ss.sqrt(), c)
        })
        .collect();
    (v.iter().map(|x| x.0).collect(), v.iter().map(|x| x.1).collect(), v.iter().map(|x| x.2).collect())
}

#[inline]
fn better(r1: f64, j1: u32, r2: f64, j2: u32) -> bool {
    match (r1.is_nan(), r2.is_nan()) {
        (true, true) => j1 < j2,
        (true, false) => false,
        (false, true) => true,
        _ => r1 > r2 || (r1 == r2 && j1 < j2),
    }
}

#[inline]
fn top_insert(top: &mut Vec<(f64, u32, u32)>, k: usize, r: f64, j: u32, ov: u32) {
    if top.len() == k {
        let (lr, lj, _) = top[k - 1];
        if !better(r, j, lr, lj) {
            return;
        }
        top.pop();
    }
    let pos = top.iter().position(|&(tr, tj, _)| better(r, j, tr, tj)).unwrap_or(top.len());
    top.insert(pos, (r, j, ov));
}

struct Scratch {
    dot: Vec<f64>,
    cnt: Vec<u32>,
    seen: Vec<bool>,
    touched: Vec<u32>,
}

#[allow(clippy::too_many_arguments)]
pub fn pair_sparse(a: &[f64], na: usize, b: &[f64], nb: usize, w: usize, mdh: Option<&[i64]>,
                   do_tshell: bool, tshell_min_wells: u32, top: usize) -> SparseOut {
    let bc = csc(b, nb, w);
    let (a_sum, a_norm, a_tot) = row_stats(a, na, w);
    let (b_sum, b_norm, b_tot) = row_stats(b, nb, w);
    let np1 = w + 1;
    let wf = w as f64;
    // betas sorted by wb (MAD-HYPE, overlap 0)
    let mut by_wb: Vec<u32> = (0..nb as u32).collect();
    by_wb.sort_by_key(|&j| (b_tot[j as usize], j));
    // T-SHELL-eligible betas sorted by h = sum(y)/||yc|| ascending (NaN last), then index
    let h: Vec<f64> = (0..nb).map(|j| b_sum[j] / b_norm[j]).collect();
    let mut by_h: Vec<u32> = (0..nb as u32).filter(|&j| b_tot[j as usize] > tshell_min_wells).collect();
    by_h.sort_by(|&x, &y| {
        let (hx, hy) = (h[x as usize], h[y as usize]);
        match (hx.is_nan(), hy.is_nan()) {
            (true, true) => x.cmp(&y),
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            _ => hx.partial_cmp(&hy).unwrap().then(x.cmp(&y)),
        }
    });

    thread_local! {
        static SCR: RefCell<Option<Scratch>> = const { RefCell::new(None) };
    }
    type Row = (Vec<(u32, u32)>, Vec<(f64, u32, u32)>);
    let rows: Vec<Row> = (0..na)
        .into_par_iter()
        .with_min_len(8)
        .map(|i| {
            SCR.with(|cell| {
                let mut opt = cell.borrow_mut();
                if opt.as_ref().map(|s| s.dot.len() != nb).unwrap_or(true) {
                    *opt = Some(Scratch { dot: vec![0.0; nb], cnt: vec![0; nb], seen: vec![false; nb], touched: Vec::new() });
                }
                let s = opt.as_mut().unwrap();
                let xr = &a[i * w..(i + 1) * w];
                for k in 0..w {
                    let x = xr[k];
                    if x == 0.0 {
                        continue;
                    }
                    let xp = present(x);
                    for p in bc.ptr[k]..bc.ptr[k + 1] {
                        let j = bc.row[p] as usize;
                        if !s.seen[j] {
                            s.seen[j] = true;
                            s.touched.push(j as u32);
                        }
                        s.dot[j] += x * bc.val[p];
                        if xp && bc.pres[p] {
                            s.cnt[j] += 1;
                        }
                    }
                }
                let wa = a_tot[i] as usize;
                // ---- MAD-HYPE
                let mut mpairs: Vec<(u32, u32)> = Vec::new();
                if let Some(m) = mdh {
                    for &j in &s.touched {
                        let ov = s.cnt[j as usize] as usize;
                        let wi = wa - ov;
                        let wj = b_tot[j as usize] as i64 - ov as i64;
                        if wj < m[ov * np1 + wi] {
                            mpairs.push((j, ov as u32));
                        }
                    }
                    let lim = m[wa]; // mdh[0, wa]
                    for &j in &by_wb {
                        if (b_tot[j as usize] as i64) >= lim {
                            break;
                        }
                        if !s.seen[j as usize] {
                            mpairs.push((j, 0));
                        }
                    }
                    mpairs.sort_unstable();
                }
                // ---- T-SHELL
                let mut best: Vec<(f64, u32, u32)> = Vec::with_capacity(top + 1);
                if do_tshell && a_tot[i] > tshell_min_wells {
                    let xbar = a_sum[i] / wf;
                    let sx = a_norm[i];
                    for &j in &s.touched {
                        let ju = j as usize;
                        if b_tot[ju] <= tshell_min_wells {
                            continue;
                        }
                        let r = (s.dot[ju] - xbar * b_sum[ju]) / (sx * b_norm[ju]);
                        top_insert(&mut best, top, r, j, s.cnt[ju]);
                    }
                    // betas without a shared well: r = -xbar*sum(y)/(sx*||yc||), increasing h first
                    let mut taken = 0usize;
                    let mut h_stop = f64::INFINITY;
                    for &j in &by_h {
                        let ju = j as usize;
                        if s.seen[ju] {
                            continue;
                        }
                        let hj = h[ju];
                        if taken >= top && !(hj <= h_stop) {
                            break;
                        }
                        let r = (0.0 - xbar * b_sum[ju]) / (sx * b_norm[ju]);
                        top_insert(&mut best, top, r, j, 0);
                        taken += 1;
                        if taken == top {
                            // keep scanning while h is within rounding distance of the top-th one
                            h_stop = if hj.is_nan() { f64::NEG_INFINITY } else { hj + hj.abs() * 1e-12 };
                        }
                    }
                    best.sort_by(|x, y| x.1.cmp(&y.1));
                }
                for &j in &s.touched {
                    let ju = j as usize;
                    s.dot[ju] = 0.0;
                    s.cnt[ju] = 0;
                    s.seen[ju] = false;
                }
                s.touched.clear();
                (mpairs, best)
            })
        })
        .collect();
    let mut out = SparseOut { m_a: vec![], m_b: vec![], m_ov: vec![], t_a: vec![], t_b: vec![], t_r: vec![], t_ov: vec![] };
    for (i, (m, t)) in rows.into_iter().enumerate() {
        for (j, ov) in m {
            out.m_a.push(i as u32);
            out.m_b.push(j);
            out.m_ov.push(ov);
        }
        for (r, j, ov) in t {
            out.t_a.push(i as u32);
            out.t_b.push(j);
            out.t_r.push(r);
            out.t_ov.push(ov);
        }
    }
    out
}

/// TIRTLtools (backend "cpp") semantics of the fused kernel:
///  * presence = value > 0 (float64), wa/wb = number of positive wells;
///  * T-SHELL statistics of `.chain_stats()`: mean = sum/n_wells,
///    norm = sqrt(sum over nonzero wells (x-mean)^2 + n_zero * mean^2),
///    r = (prod - n_wells*mean_a*mean_b) / (norm_a*norm_b);
///  * ties: the beta with the higher index ranks first; pairs without a shared well come from
///    `beta_order` (mean/norm ascending, ties higher index first), exactly n_top of them;
///  * filter_before_top3: r = -1 for pairs with wij <= 2 or loss_a + loss_b >= 0.5 (float32).
/// T-SHELL alpha/beta rows are those with more than 2 wells; `t_*` indices are global rows.
#[allow(clippy::too_many_arguments)]
pub fn pair_sparse_tt(a: &[f64], na: usize, b: &[f64], nb: usize, w: usize, mdh: Option<&[i64]>,
                      do_tshell: bool, filter_before_top3: bool) -> SparseOut {
    // CSC of beta over positive entries
    let mut ptr = vec![0usize; w + 1];
    for j in 0..nb {
        for k in 0..w {
            if b[j * w + k] > 0.0 {
                ptr[k + 1] += 1;
            }
        }
    }
    for k in 0..w {
        ptr[k + 1] += ptr[k];
    }
    let mut pos = ptr.clone();
    let mut row = vec![0u32; ptr[w]];
    let mut val = vec![0f64; ptr[w]];
    for j in 0..nb {
        for k in 0..w {
            let x = b[j * w + k];
            if x > 0.0 {
                row[pos[k]] = j as u32;
                val[pos[k]] = x;
                pos[k] += 1;
            }
        }
    }
    let stats = |m: &[f64], n: usize| -> (Vec<f64>, Vec<f64>, Vec<u32>) {
        let v: Vec<(f64, f64, u32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let r = &m[i * w..(i + 1) * w];
                let mut s = 0.0;
                let mut c = 0u32;
                for &x in r {
                    if x > 0.0 {
                        s += x;
                        c += 1;
                    }
                }
                let mean = s / w as f64;
                let mut ss = 0.0;
                for &x in r {
                    if x > 0.0 {
                        let d = x - mean;
                        ss += d * d;
                    }
                }
                let nz = (w as u32 - c) as f64;
                (mean, (ss + nz * (mean * mean)).sqrt(), c)
            })
            .collect();
        (v.iter().map(|x| x.0).collect(), v.iter().map(|x| x.1).collect(), v.iter().map(|x| x.2).collect())
    };
    let (ma, sa, wa) = stats(a, na);
    let (mb, sb, wb) = stats(b, nb);
    let np1 = w + 1;
    let nwf = w as f64;
    let mut by_wb: Vec<u32> = (0..nb as u32).collect();
    by_wb.sort_by_key(|&j| (wb[j as usize], j)); // R: order(wb) (stable)
    // T-SHELL betas (wb > 2), local index = rank among them; beta_order over local indices
    let t_beta: Vec<u32> = (0..nb as u32).filter(|&j| wb[j as usize] > 2).collect();
    let mut local = vec![u32::MAX; nb];
    for (q, &j) in t_beta.iter().enumerate() {
        local[j as usize] = q as u32;
    }
    let key: Vec<f64> = t_beta.iter().map(|&j| mb[j as usize] / sb[j as usize]).collect();
    let mut beta_order: Vec<u32> = (0..t_beta.len() as u32).collect();
    // order(mean/norm, -index): NA (NaN) last
    beta_order.sort_by(|&x, &y| {
        let (kx, ky) = (key[x as usize], key[y as usize]);
        match (kx.is_nan(), ky.is_nan()) {
            (true, true) => y.cmp(&x),
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            _ => kx.partial_cmp(&ky).unwrap().then(y.cmp(&x)),
        }
    });
    let n_top = 3usize.min(t_beta.len());
    let above = |r1: f64, j1: u32, r2: f64, j2: u32| r1 > r2 || (r1 == r2 && j1 > j2);
    let insert = |top: &mut Vec<(f64, u32, u32)>, r: f64, j: u32, ov: u32| {
        if top.len() == n_top && !above(r, j, top[n_top - 1].0, top[n_top - 1].1) {
            return;
        }
        let p = top.iter().position(|&(tr, tj, _)| above(r, j, tr, tj)).unwrap_or(top.len());
        top.insert(p, (r, j, ov));
        if top.len() > n_top {
            top.pop();
        }
    };
    thread_local! {
        static SCR2: RefCell<Option<Scratch>> = const { RefCell::new(None) };
    }
    type Row = (Vec<(u32, u32)>, Vec<(f64, u32, u32)>);
    let rows: Vec<Row> = (0..na)
        .into_par_iter()
        .with_min_len(8)
        .map(|i| {
            SCR2.with(|cell| {
                let mut opt = cell.borrow_mut();
                if opt.as_ref().map(|s| s.dot.len() != nb).unwrap_or(true) {
                    *opt = Some(Scratch { dot: vec![0.0; nb], cnt: vec![0; nb], seen: vec![false; nb], touched: Vec::new() });
                }
                let s = opt.as_mut().unwrap();
                let xr = &a[i * w..(i + 1) * w];
                for k in 0..w {
                    let x = xr[k];
                    if x <= 0.0 {
                        continue;
                    }
                    for p in ptr[k]..ptr[k + 1] {
                        let j = row[p] as usize;
                        if s.cnt[j] == 0 {
                            s.touched.push(j as u32);
                        }
                        s.cnt[j] += 1;
                        s.dot[j] += x * val[p];
                    }
                }
                let wai = wa[i] as usize;
                let mut mp: Vec<(u32, u32)> = Vec::new();
                if let Some(m) = mdh {
                    for &j in &s.touched {
                        let ov = s.cnt[j as usize] as usize;
                        if (wb[j as usize] as i64 - ov as i64) < m[ov * np1 + (wai - ov)] {
                            mp.push((j, ov as u32));
                        }
                    }
                    let lim = m[wai];
                    for &j in &by_wb {
                        if (wb[j as usize] as i64) >= lim {
                            break;
                        }
                        if s.cnt[j as usize] == 0 {
                            mp.push((j, 0));
                        }
                    }
                    mp.sort_unstable();
                }
                let mut top: Vec<(f64, u32, u32)> = Vec::with_capacity(4);
                if do_tshell && wa[i] > 2 && n_top > 0 {
                    for &j in &s.touched {
                        let ju = j as usize;
                        let lj = local[ju];
                        if lj == u32::MAX {
                            continue;
                        }
                        let wij = s.cnt[ju];
                        let r = if filter_before_top3 && {
                            let (fw, fa, fb) = (wij as f32, wa[i] as f32, wb[ju] as f32);
                            let den = fw + (fb - fw) + (fa - fw);
                            wij <= 2 || ((fb - fw) / den + (fa - fw) / den) >= 0.5f32
                        } {
                            -1.0
                        } else {
                            (s.dot[ju] - nwf * ma[i] * mb[ju]) / (sa[i] * sb[ju])
                        };
                        insert(&mut top, r, lj, wij);
                    }
                    let mut added = 0;
                    if filter_before_top3 {
                        let mut q = t_beta.len();
                        while q > 0 && added < n_top {
                            q -= 1;
                            if s.cnt[t_beta[q] as usize] > 0 {
                                continue;
                            }
                            insert(&mut top, -1.0, q as u32, 0);
                            added += 1;
                        }
                    } else {
                        for &lq in &beta_order {
                            if added >= n_top {
                                break;
                            }
                            let ju = t_beta[lq as usize] as usize;
                            if s.cnt[ju] > 0 {
                                continue;
                            }
                            let r = (0.0 - nwf * ma[i] * mb[ju]) / (sa[i] * sb[ju]);
                            insert(&mut top, r, lq, 0);
                            added += 1;
                        }
                    }
                    top.reverse(); // lowest to highest correlation, as TIRTLtools
                    for t in top.iter_mut() {
                        t.1 = t_beta[t.1 as usize];
                    }
                }
                for &j in &s.touched {
                    let ju = j as usize;
                    s.dot[ju] = 0.0;
                    s.cnt[ju] = 0;
                }
                s.touched.clear();
                (mp, top)
            })
        })
        .collect();
    let mut out = SparseOut { m_a: vec![], m_b: vec![], m_ov: vec![], t_a: vec![], t_b: vec![], t_r: vec![], t_ov: vec![] };
    for (i, (m, t)) in rows.into_iter().enumerate() {
        for (j, ov) in m {
            out.m_a.push(i as u32);
            out.m_b.push(j);
            out.m_ov.push(ov);
        }
        for (r, j, ov) in t {
            out.t_a.push(i as u32);
            out.t_b.push(j);
            out.t_r.push(r);
            out.t_ov.push(ov);
        }
    }
    out
}

/// greedy selection of .build_pairs_df(): rows in priority order; keep a row when its beta is
/// kept fewer than n_max_alpha times and its alpha fewer than n_max_beta times
pub fn greedy_select(alpha: &[i64], beta: &[i64], n_max_alpha: u32, n_max_beta: u32) -> Vec<bool> {
    use hashbrown::HashMap;
    let mut na: HashMap<i64, u32> = HashMap::new();
    let mut nb: HashMap<i64, u32> = HashMap::new();
    let mut keep = vec![false; alpha.len()];
    for k in 0..alpha.len() {
        let cb = *nb.get(&beta[k]).unwrap_or(&0);
        let ca = *na.get(&alpha[k]).unwrap_or(&0);
        if cb < n_max_alpha && ca < n_max_beta {
            keep[k] = true;
            *nb.entry(beta[k]).or_insert(0) += 1;
            *na.entry(alpha[k]).or_insert(0) += 1;
        }
    }
    keep
}
