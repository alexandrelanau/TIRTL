//! tirtl._core: Rust kernels exposed to Python.
mod clones;
mod collate;
mod madhyper;
mod output;
mod pairing;
mod sparse;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

type A1<'py, T> = Bound<'py, PyArray1<T>>;

// ----------------------------------------------------------------------------- collation
#[pyfunction]
fn r_order(xs: Vec<String>) -> Vec<usize> {
    let v: Vec<&str> = xs.iter().map(|s| s.as_str()).collect();
    collate::r_order(&v)
}

// ----------------------------------------------------------------------------- MAD-HYPE maths
#[pyfunction]
#[pyo3(signature = (wi, wj, w_ij, w_tot, cpw, alpha, prior=1.0))]
fn estimate_pair_prob(wi: f64, wj: f64, w_ij: f64, w_tot: f64, cpw: f64, alpha: f64, prior: f64) -> f64 {
    madhyper::estimate_pair_prob(wi, wj, w_ij, w_tot, cpw, alpha, prior)
}

#[pyfunction]
fn find_freq(w: f64, w_tot: f64, cpw: f64, a: f64) -> f64 {
    madhyper::find_freq(w, w_tot, cpw, a)
}

/// vectorised estimate_pair_prob (parallel), linear scale
#[pyfunction]
#[pyo3(signature = (wi, wj, w_ij, w_tot, cpw, alpha, prior=1.0))]
fn estimate_pair_prob_many<'py>(py: Python<'py>, wi: PyReadonlyArray1<'py, f64>, wj: PyReadonlyArray1<'py, f64>,
                                w_ij: PyReadonlyArray1<'py, f64>, w_tot: f64, cpw: f64, alpha: f64, prior: f64)
                                -> PyResult<A1<'py, f64>> {
    use rayon::prelude::*;
    let (a, b, c) = (wi.as_slice()?, wj.as_slice()?, w_ij.as_slice()?);
    if a.len() != b.len() || a.len() != c.len() {
        return Err(PyValueError::new_err("wi, wj, w_ij must have the same length"));
    }
    let out: Vec<f64> = py.allow_threads(|| {
        (0..a.len()).into_par_iter()
            .map(|k| madhyper::estimate_pair_prob(a[k], b[k], c[k], w_tot, cpw, alpha, prior))
            .collect()
    });
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (n_wells, cells=1000.0, alpha=2.0, prior=1.0))]
fn madhyper_surface<'py>(py: Python<'py>, n_wells: usize, cells: f64, alpha: f64, prior: f64)
                         -> PyResult<Bound<'py, PyArray2<f64>>> {
    let v = py.allow_threads(|| madhyper::surface(n_wells, cells, alpha, prior));
    let arr = Array2::from_shape_vec((n_wells + 1, n_wells + 1), v).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

// ----------------------------------------------------------------------------- clone tables
#[pyclass(module = "tirtl._core")]
struct ChainData {
    inner: clones::ChainData,
}

#[pymethods]
impl ChainData {
    /// load MiXCR clone tables (in parallel); sequence ids are assigned by first appearance
    #[staticmethod]
    #[pyo3(signature = (paths, functional_only=false, seq_column="targetSequences"))]
    fn load(py: Python<'_>, paths: Vec<String>, functional_only: bool, seq_column: &str) -> PyResult<ChainData> {
        let inner = py.allow_threads(|| clones::ChainData::load(&paths, functional_only, seq_column)).map_err(PyIOError::new_err)?;
        Ok(ChainData { inner })
    }
    fn nrows(&self) -> Vec<usize> {
        self.inner.files.iter().map(|f| f.nrow()).collect()
    }
    /// largest readCount of each file (-inf for an empty file; NaN ignored)
    fn max_counts(&self) -> Vec<f64> {
        self.inner.files.iter().map(|f| f.count.iter().cloned().filter(|x| !x.is_nan()).fold(f64::NEG_INFINITY, f64::max)).collect()
    }
    fn n_seqs(&self) -> usize {
        self.inner.seqs.len()
    }
    /// big_merge_freqs2 + rowSums(>0) > min_wells filter ->
    /// (seq ids, matrix float64 (rows x cols), n rows before filter, column file indices)
    fn merge<'py>(&self, py: Python<'py>, file_idx: Vec<usize>, min_reads: f64, min_wells: usize)
                  -> PyResult<(A1<'py, u32>, Bound<'py, PyArray2<f64>>, usize, Vec<usize>)> {
        let (ids, mat, ntot, cols) = py.allow_threads(|| self.inner.merge(&file_idx, min_reads, min_wells));
        let n = ids.len();
        let arr = Array2::from_shape_vec((n, cols.len()), mat).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((ids.into_pyarray(py), arr.into_pyarray(py), ntot, cols))
    }
    /// TIRTLtools big_merge_freqs2(wells=...) + filter -> (seq ids, matrix, n rows before filter)
    fn merge_tt<'py>(&self, py: Python<'py>, file_idx: Vec<usize>, min_reads: f64, min_wells: usize)
                     -> PyResult<(A1<'py, u32>, Bound<'py, PyArray2<f64>>, usize)> {
        let n = file_idx.len();
        let (ids, mat, ntot) = py.allow_threads(|| self.inner.merge_tt(&file_idx, min_reads, min_wells));
        let arr = Array2::from_shape_vec((ids.len(), n), mat).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((ids.into_pyarray(py), arr.into_pyarray(py), ntot))
    }
    /// all rows of the given files: (file position, sequence id, readFraction)
    fn rows<'py>(&self, py: Python<'py>, file_idx: Vec<usize>) -> PyResult<(A1<'py, i64>, A1<'py, u32>, A1<'py, f64>)> {
        let (mut f, mut q, mut x) = (Vec::new(), Vec::new(), Vec::new());
        for (k, &fi) in file_idx.iter().enumerate() {
            let fd = &self.inner.files[fi];
            f.extend(std::iter::repeat(k as i64).take(fd.nrow()));
            q.extend_from_slice(&fd.seq);
            x.extend_from_slice(&fd.frac);
        }
        Ok((f.into_pyarray(py), q.into_pyarray(py), x.into_pyarray(py)))
    }
    /// strings of sequence ids
    fn seq_strings<'py>(&self, py: Python<'py>, ids: PyReadonlyArray1<'py, u32>) -> PyResult<Bound<'py, PyList>> {
        let s = &self.inner.seqs.strings;
        let v: Vec<&str> = ids.as_slice()?.iter().map(|&i| &*s[i as usize]).collect();
        PyList::new(py, v)
    }
    fn gene_strings(&self) -> Vec<String> {
        self.inner.genes.strings.iter().map(|s| s.to_string()).collect()
    }
    /// add_VJ_aa -> (first aaSeqCDR3 (None when absent), v gene ids, j gene ids (-1 when absent))
    fn annotate<'py>(&self, py: Python<'py>, file_idx: Vec<usize>, ids: PyReadonlyArray1<'py, u32>)
                     -> PyResult<(Vec<Option<String>>, A1<'py, i64>, A1<'py, i64>)> {
        let ids = ids.as_slice()?.to_vec();
        let (aa, v, j) = py.allow_threads(|| {
            let (aa, v, j) = self.inner.annotate(&file_idx, &ids);
            (aa.into_iter().map(|x| x.map(|s| s.to_string())).collect::<Vec<_>>(), v, j)
        });
        let cv = |x: Vec<u32>| x.into_iter().map(|y| if y == clones::NO_ID { -1 } else { y as i64 }).collect::<Vec<i64>>();
        Ok((aa, cv(v).into_pyarray(py), cv(j).into_pyarray(py)))
    }
    /// combineTCR ordered by -readCount, written directly as TSV (columns of the R pseudobulk file)
    fn write_pseudobulk(&self, py: Python<'_>, file_idx: Vec<usize>, max_wells: i64, path: String) -> PyResult<usize> {
        py.allow_threads(|| {
            let pb = self.inner.pseudobulk(&file_idx);
            let n = pb.seq.len();
            let seqs: Vec<String> = pb.seq.iter().map(|&q| self.inner.seqs.strings[q as usize].to_string()).collect();
            let genes: Vec<String> = self.inner.genes.strings.iter().map(|g| g.to_string()).collect();
            let idx: Vec<i64> = (0..n as i64).collect();
            let v: Vec<i64> = pb.v.iter().map(|&x| x as i64).collect();
            let j: Vec<i64> = pb.j.iter().map(|&x| x as i64).collect();
            let nw: Vec<i64> = pb.n_wells.iter().map(|&x| x as i64).collect();
            let mw = vec![max_wells; n];
            let names: Vec<String> = ["targetSequences", "readCount", "v", "j", "aaSeqCDR3", "n_wells", "readCount_max",
                "readCount_median", "avg", "sem", "readFraction", "max_wells"].iter().map(|s| s.to_string()).collect();
            let cols = vec![output::Col::Str(&idx, &seqs), output::Col::F64(&pb.read_count), output::Col::Str(&v, &genes),
                output::Col::Str(&j, &genes), output::Col::Str(&idx, &pb.aa), output::Col::I64(&nw),
                output::Col::F64(&pb.read_count_max), output::Col::F64(&pb.read_count_median), output::Col::F64(&pb.avg),
                output::Col::F64(&pb.sem), output::Col::F64(&pb.read_fraction), output::Col::I64(&mw)];
            output::write_tsv(&path, &names, &cols, n).map(|_| n)
        }).map_err(|e| PyIOError::new_err(e.to_string()))
    }
    /// combineTCR ordered by -readCount: dict of columns (seq/v/j/aa as int64 codes)
    fn pseudobulk<'py>(&self, py: Python<'py>, file_idx: Vec<usize>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let pb = py.allow_threads(|| self.inner.pseudobulk(&file_idx));
        let d = pyo3::types::PyDict::new(py);
        let ci = |x: Vec<u32>| x.into_iter().map(|y| y as i64).collect::<Vec<i64>>();
        d.set_item("seq", ci(pb.seq).into_pyarray(py))?;
        d.set_item("readCount", pb.read_count.into_pyarray(py))?;
        d.set_item("v", ci(pb.v).into_pyarray(py))?;
        d.set_item("j", ci(pb.j).into_pyarray(py))?;
        d.set_item("aaSeqCDR3", pb.aa)?;
        d.set_item("n_wells", pb.n_wells.into_pyarray(py))?;
        d.set_item("readCount_max", pb.read_count_max.into_pyarray(py))?;
        d.set_item("readCount_median", pb.read_count_median.into_pyarray(py))?;
        d.set_item("avg", pb.avg.into_pyarray(py))?;
        d.set_item("sem", pb.sem.into_pyarray(py))?;
        d.set_item("readFraction", pb.read_fraction.into_pyarray(py))?;
        Ok(d)
    }
}

// ----------------------------------------------------------------------------- pairing
#[pyclass(module = "tirtl._core")]
struct Bitsets {
    inner: pairing::Bits,
}

#[pymethods]
impl Bitsets {
    #[new]
    fn new(py: Python<'_>, mat: PyReadonlyArray2<'_, f64>) -> PyResult<Bitsets> {
        let (n, w) = (mat.shape()[0], mat.shape()[1]);
        let m = mat.as_slice()?;
        Ok(Bitsets { inner: py.allow_threads(|| pairing::pack_bits(m, n, w)) })
    }
    fn totals<'py>(&self, py: Python<'py>) -> A1<'py, i64> {
        self.inner.total.iter().map(|&x| x as i64).collect::<Vec<i64>>().into_pyarray(py)
    }
}

/// MAD-HYPE pairs: (alpha row, beta row, wij)
#[pyfunction]
fn madhype_pairs<'py>(py: Python<'py>, a: &Bitsets, b: &Bitsets, mdh: PyReadonlyArray2<'py, i64>)
                      -> PyResult<(A1<'py, u32>, A1<'py, u32>, A1<'py, u32>)> {
    let n_wells = mdh.shape()[0] - 1;
    let m = mdh.as_slice()?;
    if a.inner.words != b.inner.words {
        return Err(PyValueError::new_err("alpha and beta matrices have different numbers of wells"));
    }
    let (x, y, z) = py.allow_threads(|| pairing::madhype(&a.inner, &b.inner, m, n_wells));
    Ok((x.into_pyarray(py), y.into_pyarray(py), z.into_pyarray(py)))
}

#[pyfunction]
fn overlaps<'py>(py: Python<'py>, a: &Bitsets, b: &Bitsets, ai: PyReadonlyArray1<'py, u32>,
                 bi: PyReadonlyArray1<'py, u32>) -> PyResult<A1<'py, u32>> {
    let (x, y) = (ai.as_slice()?, bi.as_slice()?);
    Ok(py.allow_threads(|| pairing::overlaps_of(&a.inner, &b.inner, x, y)).into_pyarray(py))
}

#[pyfunction]
fn normalize_rows<'py>(py: Python<'py>, mat: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (n, w) = (mat.shape()[0], mat.shape()[1]);
    let m = mat.as_slice()?;
    let v = py.allow_threads(|| pairing::normalize_rows(m, n, w));
    Ok(Array2::from_shape_vec((n, w), v).unwrap().into_pyarray(py))
}

#[pyfunction]
fn topk_rows_f32<'py>(py: Python<'py>, mat: PyReadonlyArray2<'py, f32>, k: usize)
                      -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f32>>)> {
    let (n, m) = (mat.shape()[0], mat.shape()[1]);
    let s = mat.as_slice()?;
    let kk = k.min(m);
    let (i, v) = py.allow_threads(|| pairing::topk_rows_f32(s, n, m, k));
    Ok((Array2::from_shape_vec((n, kk), i).unwrap().into_pyarray(py),
        Array2::from_shape_vec((n, kk), v).unwrap().into_pyarray(py)))
}

/// exact float64 top-`top` per alpha row from float32 candidates -> (alpha rows, beta rows, r, n rows recomputed)
#[pyfunction]
#[pyo3(signature = (anorm, bnorm, rows, cand_idx, cand_val, bound, top=3))]
fn tshell_refine<'py>(py: Python<'py>, anorm: PyReadonlyArray2<'py, f64>, bnorm: PyReadonlyArray2<'py, f64>,
                      rows: PyReadonlyArray1<'py, u32>, cand_idx: PyReadonlyArray2<'py, i64>,
                      cand_val: PyReadonlyArray2<'py, f32>, bound: f64, top: usize)
                      -> PyResult<(A1<'py, u32>, A1<'py, u32>, A1<'py, f64>, usize)> {
    let w = anorm.shape()[1];
    if bnorm.shape()[1] != w {
        return Err(PyValueError::new_err("alpha/beta width mismatch"));
    }
    let nb = bnorm.shape()[0];
    let kc = cand_idx.shape()[1];
    let (a, b, r, ci, cv) = (anorm.as_slice()?, bnorm.as_slice()?, rows.as_slice()?, cand_idx.as_slice()?, cand_val.as_slice()?);
    let o = py.allow_threads(|| pairing::tshell_refine(a, b, w, nb, r, ci, cv, kc, bound, top));
    Ok((o.a.into_pyarray(py), o.b.into_pyarray(py), o.r.into_pyarray(py), o.n_fallback))
}

/// fused sparse MAD-HYPE + T-SHELL -> (m_a, m_b, m_ov, t_a, t_b, t_r, t_ov)
#[pyfunction]
#[pyo3(signature = (a, b, mdh=None, tshell=true, tshell_min_wells=2, top=3))]
#[allow(clippy::type_complexity)]
fn pair_sparse<'py>(py: Python<'py>, a: PyReadonlyArray2<'py, f64>, b: PyReadonlyArray2<'py, f64>,
                    mdh: Option<PyReadonlyArray2<'py, i64>>, tshell: bool, tshell_min_wells: u32, top: usize)
                    -> PyResult<(A1<'py, u32>, A1<'py, u32>, A1<'py, u32>, A1<'py, u32>, A1<'py, u32>, A1<'py, f64>, A1<'py, u32>)> {
    let (na, w) = (a.shape()[0], a.shape()[1]);
    let nb = b.shape()[0];
    if b.shape()[1] != w {
        return Err(PyValueError::new_err("alpha and beta matrices have different numbers of wells"));
    }
    if let Some(m) = &mdh {
        if m.shape()[0] != w + 1 || m.shape()[1] != w + 1 {
            return Err(PyValueError::new_err("mdh must be (n_wells+1) x (n_wells+1)"));
        }
    }
    let (sa, sb) = (a.as_slice()?, b.as_slice()?);
    let sm = match &mdh { Some(m) => Some(m.as_slice()?), None => None };
    let o = py.allow_threads(|| sparse::pair_sparse(sa, na, sb, nb, w, sm, tshell, tshell_min_wells, top));
    Ok((o.m_a.into_pyarray(py), o.m_b.into_pyarray(py), o.m_ov.into_pyarray(py), o.t_a.into_pyarray(py),
        o.t_b.into_pyarray(py), o.t_r.into_pyarray(py), o.t_ov.into_pyarray(py)))
}

/// TIRTLtools-semantics fused kernel -> (m_a, m_b, m_ov, t_a, t_b, t_r, t_ov)
#[pyfunction]
#[pyo3(signature = (a, b, mdh=None, tshell=true, filter_before_top3=false))]
#[allow(clippy::type_complexity)]
fn pair_sparse_tt<'py>(py: Python<'py>, a: PyReadonlyArray2<'py, f64>, b: PyReadonlyArray2<'py, f64>,
                       mdh: Option<PyReadonlyArray2<'py, i64>>, tshell: bool, filter_before_top3: bool)
                       -> PyResult<(A1<'py, u32>, A1<'py, u32>, A1<'py, u32>, A1<'py, u32>, A1<'py, u32>, A1<'py, f64>, A1<'py, u32>)> {
    let (na, w) = (a.shape()[0], a.shape()[1]);
    let nb = b.shape()[0];
    if b.shape()[1] != w {
        return Err(PyValueError::new_err("alpha and beta matrices have different numbers of wells"));
    }
    let (sa, sb) = (a.as_slice()?, b.as_slice()?);
    let sm = match &mdh { Some(m) => Some(m.as_slice()?), None => None };
    let o = py.allow_threads(|| sparse::pair_sparse_tt(sa, na, sb, nb, w, sm, tshell, filter_before_top3));
    Ok((o.m_a.into_pyarray(py), o.m_b.into_pyarray(py), o.m_ov.into_pyarray(py), o.t_a.into_pyarray(py),
        o.t_b.into_pyarray(py), o.t_r.into_pyarray(py), o.t_ov.into_pyarray(py)))
}

#[pyfunction]
fn greedy_select<'py>(py: Python<'py>, alpha: PyReadonlyArray1<'py, i64>, beta: PyReadonlyArray1<'py, i64>,
                      n_max_alpha: u32, n_max_beta: u32) -> PyResult<A1<'py, bool>> {
    Ok(sparse::greedy_select(alpha.as_slice()?, beta.as_slice()?, n_max_alpha, n_max_beta).into_pyarray(py))
}

/// R mean(x[start:end]) for each window (long-double sum emulation + R's refinement pass)
#[pyfunction]
fn window_means<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>, start: PyReadonlyArray1<'py, i64>,
                     end: PyReadonlyArray1<'py, i64>) -> PyResult<A1<'py, f64>> {
    use rayon::prelude::*;
    let (x, s, e) = (x.as_slice()?, start.as_slice()?, end.as_slice()?);
    let out: Vec<f64> = py.allow_threads(|| {
        (0..s.len()).into_par_iter().map(|k| {
            let w = &x[s[k] as usize..=e[k] as usize];
            let n = w.len() as f64;
            let mut m = madhyper::r_sum(w) / n;
            if m.is_finite() {
                let d: Vec<f64> = w.iter().map(|v| v - m).collect();
                m += madhyper::r_sum(&d) / n;
            }
            m
        }).collect()
    });
    Ok(out.into_pyarray(py))
}

/// sum of x over consecutive groups [starts[k], starts[k+1]) with R's long-double accumulator emulated
#[pyfunction]
fn grouped_rsum<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>, starts: PyReadonlyArray1<'py, i64>)
                     -> PyResult<A1<'py, f64>> {
    use rayon::prelude::*;
    let (x, st) = (x.as_slice()?, starts.as_slice()?);
    let n = x.len();
    let out: Vec<f64> = py.allow_threads(|| {
        (0..st.len()).into_par_iter().map(|k| {
            let a = st[k] as usize;
            let b = if k + 1 < st.len() { st[k + 1] as usize } else { n };
            madhyper::r_sum(&x[a..b])
        }).collect()
    });
    Ok(out.into_pyarray(py))
}

/// cumsum(x) with R's long-double accumulator emulated (compensated running sum)
#[pyfunction]
fn r_cumsum<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> PyResult<A1<'py, f64>> {
    let x = x.as_slice()?;
    let mut out = Vec::with_capacity(x.len());
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for &v in x {
        let t = s + v;
        c += if s.abs() >= v.abs() { (s - t) + v } else { (v - t) + s };
        s = t;
        out.push(s + c);
    }
    Ok(out.into_pyarray(py))
}

/// sum(x) with R's long-double accumulator emulated
#[pyfunction]
fn r_sum_arr(x: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(madhyper::r_sum(x.as_slice()?))
}

#[pyfunction]
fn f32_dot_bound(w: usize) -> f64 {
    pairing::f32_dot_bound(w)
}

// ----------------------------------------------------------------------------- output
/// write a TSV. columns: list of (name, spec) with spec one of
///   ("f", float64 array) | ("i", int64 array) | ("s", int64 codes, list[str]) |
///   ("c", codes1, dict1, codes2, dict2, sep)
#[pyfunction]
fn write_tsv(py: Python<'_>, path: String, columns: Vec<(String, Bound<'_, pyo3::types::PyTuple>)>) -> PyResult<()> {
    enum Own<'a> {
        F(PyReadonlyArray1<'a, f64>),
        I(PyReadonlyArray1<'a, i64>),
        S(PyReadonlyArray1<'a, i64>, Vec<String>),
        C(PyReadonlyArray1<'a, i64>, Vec<String>, PyReadonlyArray1<'a, i64>, Vec<String>, String),
    }
    let mut names = vec![];
    let mut owned: Vec<Own> = vec![];
    for (name, spec) in &columns {
        names.push(name.clone());
        let kind: String = spec.get_item(0)?.extract()?;
        owned.push(match kind.as_str() {
            "f" => Own::F(spec.get_item(1)?.extract()?),
            "i" => Own::I(spec.get_item(1)?.extract()?),
            "s" => Own::S(spec.get_item(1)?.extract()?, spec.get_item(2)?.extract()?),
            "c" => Own::C(spec.get_item(1)?.extract()?, spec.get_item(2)?.extract()?, spec.get_item(3)?.extract()?,
                          spec.get_item(4)?.extract()?, spec.get_item(5)?.extract()?),
            _ => return Err(PyValueError::new_err(format!("unknown column kind {kind}"))),
        });
    }
    let mut cols: Vec<output::Col> = vec![];
    let mut nrow: Option<usize> = None;
    for o in &owned {
        let (c, n) = match o {
            Own::F(a) => (output::Col::F64(a.as_slice()?), a.len()),
            Own::I(a) => (output::Col::I64(a.as_slice()?), a.len()),
            Own::S(a, d) => (output::Col::Str(a.as_slice()?, d), a.len()),
            Own::C(a, d, b, e, s) => (output::Col::Concat(a.as_slice()?, d, b.as_slice()?, e, s), a.len()),
        };
        if let Some(m) = nrow {
            if m != n {
                return Err(PyValueError::new_err("columns have different lengths"));
            }
        }
        nrow = Some(n);
        cols.push(c);
    }
    let nrow = nrow.unwrap_or(0);
    py.allow_threads(|| output::write_tsv(&path, &names, &cols, nrow)).map_err(|e| PyIOError::new_err(e.to_string()))
}

/// [d1[c1[i]] + sep + d2[c2[i]]] as a list of str
#[pyfunction]
fn concat_strings<'py>(py: Python<'py>, c1: PyReadonlyArray1<'py, i64>, d1: Vec<String>, c2: PyReadonlyArray1<'py, i64>,
                       d2: Vec<String>, sep: &str) -> PyResult<Bound<'py, PyList>> {
    let (a, b) = (c1.as_slice()?, c2.as_slice()?);
    let get = |d: &Vec<String>, c: i64| -> String { if c >= 0 { d[c as usize].clone() } else { String::new() } };
    let v: Vec<String> = a.iter().zip(b.iter()).map(|(&x, &y)| format!("{}{}{}", get(&d1, x), sep, get(&d2, y))).collect();
    PyList::new(py, v)
}

/// Arrow large_string buffers for a dictionary-coded string column (optionally a concatenation):
/// (offsets int64[n+1], data uint8, valid bool[n])
#[pyfunction]
#[pyo3(signature = (c1, d1, c2=None, d2=None, sep=""))]
#[allow(clippy::type_complexity)]
fn arrow_strings<'py>(py: Python<'py>, c1: PyReadonlyArray1<'py, i64>, d1: Vec<String>,
                      c2: Option<PyReadonlyArray1<'py, i64>>, d2: Option<Vec<String>>, sep: &str)
                      -> PyResult<(A1<'py, i64>, A1<'py, u8>, A1<'py, bool>)> {
    let a = c1.as_slice()?;
    let b = match &c2 { Some(x) => Some(x.as_slice()?), None => None };
    let (o, d, v) = py.allow_threads(|| match (b, &d2) {
        (Some(b), Some(d2)) => output::arrow_strings(a, &d1, Some((b, d2, sep))),
        _ => output::arrow_strings(a, &d1, None),
    });
    Ok((o.into_pyarray(py), d.into_pyarray(py), v.into_pyarray(py)))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(r_order, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_pair_prob, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_pair_prob_many, m)?)?;
    m.add_function(wrap_pyfunction!(find_freq, m)?)?;
    m.add_function(wrap_pyfunction!(madhyper_surface, m)?)?;
    m.add_function(wrap_pyfunction!(madhype_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_rows, m)?)?;
    m.add_function(wrap_pyfunction!(topk_rows_f32, m)?)?;
    m.add_function(wrap_pyfunction!(tshell_refine, m)?)?;
    m.add_function(wrap_pyfunction!(f32_dot_bound, m)?)?;
    m.add_function(wrap_pyfunction!(pair_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(pair_sparse_tt, m)?)?;
    m.add_function(wrap_pyfunction!(greedy_select, m)?)?;
    m.add_function(wrap_pyfunction!(window_means, m)?)?;
    m.add_function(wrap_pyfunction!(grouped_rsum, m)?)?;
    m.add_function(wrap_pyfunction!(r_cumsum, m)?)?;
    m.add_function(wrap_pyfunction!(r_sum_arr, m)?)?;
    m.add_function(wrap_pyfunction!(write_tsv, m)?)?;
    m.add_function(wrap_pyfunction!(concat_strings, m)?)?;
    m.add_function(wrap_pyfunction!(arrow_strings, m)?)?;
    m.add_class::<ChainData>()?;
    m.add_class::<Bitsets>()?;
    Ok(())
}
