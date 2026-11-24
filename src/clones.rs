//! Parallel MiXCR clone-table loader with sequence interning, and the R semantics of
//! big_merge_freqs2(), add_VJ_aa() and combineTCR() on top of it.

use crate::collate::r_cmp;
use crate::madhyper::r_sum;
use hashbrown::HashMap;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::io::Read;

pub const NO_ID: u32 = u32::MAX;

/// One clone table (one well, one chain). aaSeqCDR3 values are kept in a per-file pool.
#[derive(Default)]
pub struct FileData {
    pub seq: Vec<u32>,
    pub count: Vec<f64>,
    pub frac: Vec<f64>,
    pub aa: Vec<u32>,
    pub aa_pool: Vec<Box<str>>,
    pub v: Vec<u32>,
    pub j: Vec<u32>,
}

impl FileData {
    pub fn nrow(&self) -> usize {
        self.seq.len()
    }
    #[inline]
    pub fn aa_str(&self, k: usize) -> &str {
        &self.aa_pool[self.aa[k] as usize]
    }
}

/// Small interned string table (gene names), ids in order of first appearance
#[derive(Default)]
pub struct Interner {
    pub strings: Vec<Box<str>>,
    map: HashMap<Box<str>, u32, ahash::RandomState>,
}

impl Interner {
    pub fn get_or_insert(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = self.strings.len() as u32;
        let b: Box<str> = s.into();
        self.strings.push(b.clone());
        self.map.insert(b, id);
        id
    }
}

/// Global sequence table (ids are arbitrary but stable within one load)
#[derive(Default)]
pub struct SeqTable {
    pub strings: Vec<Box<str>>,
}

impl SeqTable {
    pub fn len(&self) -> usize {
        self.strings.len()
    }
}

pub struct ChainData {
    pub files: Vec<FileData>,
    pub seqs: SeqTable,
    pub genes: Interner,
}

const SHARDS: usize = 64;

/// Per-file parse result with file-local interning (first-appearance order)
struct LocalParse {
    seq_local: Vec<u32>,
    seq_uniq: Vec<String>,
    seq_shard: Vec<Vec<u32>>, // local unique indices per hash shard
    count: Vec<f64>,
    frac: Vec<f64>,
    aa_local: Vec<u32>,
    aa_uniq: Vec<String>,
    v_local: Vec<u32>,
    j_local: Vec<u32>,
    gene_uniq: Vec<String>,
}

fn read_file(path: &str) -> Result<String, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut s = String::new();
    f.read_to_string(&mut s).map_err(|e| format!("{path}: {e}"))?;
    Ok(s)
}

/// First token of a MiXCR hit list ("TRBV6-2*00(1234),TRBV6-3*00(..)" -> "TRBV6-2")
#[inline]
fn gene_token(s: &str) -> &str {
    match s.find('*') {
        Some(p) => &s[..p],
        None => s,
    }
}

/// R fread: numbers; empty / NA -> NaN
#[inline]
fn parse_num(s: &str) -> f64 {
    let t = s.trim();
    if t.is_empty() || t == "NA" {
        return f64::NAN;
    }
    t.parse::<f64>().unwrap_or(f64::NAN)
}

fn shard_hasher() -> ahash::RandomState {
    ahash::RandomState::with_seeds(0x5eed, 0x7172, 0x746c, 0x2024)
}

fn empty_parse() -> LocalParse {
    LocalParse {
        seq_local: vec![], seq_uniq: vec![], seq_shard: vec![Vec::new(); SHARDS], count: vec![], frac: vec![],
        aa_local: vec![], aa_uniq: vec![], v_local: vec![], j_local: vec![], gene_uniq: vec![],
    }
}

/// `functional_only`: drop rows whose aaSeqCDR3 contains '*' or '_' (TIRTLtools .get_functional)
fn parse_one(path: &str, functional_only: bool, seq_column: &str) -> Result<LocalParse, String> {
    if path.is_empty() {
        return Ok(empty_parse()); // well without a file (TIRTLtools: empty table)
    }
    let text = read_file(path)?;
    let mut lines = text.split('\n');
    let header = lines.next().ok_or_else(|| format!("{path}: empty file"))?;
    let header = header.strip_suffix('\r').unwrap_or(header);
    let cols: Vec<&str> = header.split('\t').collect();
    let find = |name: &str| cols.iter().position(|c| *c == name);
    let i_seq = find(seq_column).ok_or_else(|| format!("{path}: no {seq_column} column"))?;
    let i_cnt = find("readCount").ok_or_else(|| format!("{path}: no readCount column"))?;
    let i_frac = find("readFraction").ok_or_else(|| format!("{path}: no readFraction column"))?;
    let i_aa = find("aaSeqCDR3");
    let i_v = find("allVHitsWithScore");
    let i_j = find("allJHitsWithScore");
    let maxcol = [Some(i_seq), Some(i_cnt), Some(i_frac), i_aa, i_v, i_j].iter().flatten().copied().max().unwrap();
    let hs = shard_hasher();

    let mut seq_map: HashMap<&str, u32, ahash::RandomState> = HashMap::default();
    let mut aa_map: HashMap<&str, u32, ahash::RandomState> = HashMap::default();
    let mut gene_map: HashMap<&str, u32, ahash::RandomState> = HashMap::default();
    let mut out = LocalParse {
        seq_local: vec![], seq_uniq: vec![], seq_shard: vec![Vec::new(); SHARDS], count: vec![], frac: vec![],
        aa_local: vec![], aa_uniq: vec![], v_local: vec![], j_local: vec![], gene_uniq: vec![],
    };
    let mut fields: Vec<&str> = Vec::with_capacity(maxcol + 1);
    for line in lines {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() {
            continue;
        }
        fields.clear();
        for (k, f) in line.split('\t').enumerate() {
            fields.push(f);
            if k >= maxcol {
                break;
            }
        }
        while fields.len() <= maxcol {
            fields.push("");
        }
        if functional_only {
            let a = i_aa.map(|i| fields[i]).unwrap_or("");
            if a.contains('*') || a.contains('_') {
                continue;
            }
        }
        let s = fields[i_seq];
        let n = seq_map.len() as u32;
        let id = *seq_map.entry(s).or_insert_with(|| {
            out.seq_shard[(hs.hash_one(s) as usize) % SHARDS].push(n);
            out.seq_uniq.push(s.to_string());
            n
        });
        out.seq_local.push(id);
        out.count.push(parse_num(fields[i_cnt]));
        out.frac.push(parse_num(fields[i_frac]));
        let a = i_aa.map(|i| fields[i]).unwrap_or("");
        let n = aa_map.len() as u32;
        let id = *aa_map.entry(a).or_insert_with(|| {
            out.aa_uniq.push(a.to_string());
            n
        });
        out.aa_local.push(id);
        for (ic, dst) in [(i_v, 0usize), (i_j, 1usize)] {
            let g = gene_token(ic.map(|i| fields[i]).unwrap_or(""));
            let n = gene_map.len() as u32;
            let id = *gene_map.entry(g).or_insert_with(|| {
                out.gene_uniq.push(g.to_string());
                n
            });
            if dst == 0 {
                out.v_local.push(id)
            } else {
                out.j_local.push(id)
            }
        }
    }
    Ok(out)
}

impl ChainData {
    pub fn load(paths: &[String], functional_only: bool, seq_column: &str) -> Result<ChainData, String> {
        let parsed: Vec<Result<LocalParse, String>> =
            paths.par_iter().map(|p| parse_one(p, functional_only, seq_column)).collect();
        let mut locals = Vec::with_capacity(parsed.len());
        for r in parsed {
            locals.push(r?);
        }
        // sequences: interning sharded by hash, in parallel
        let shards: Vec<(Vec<Box<str>>, Vec<(u32, u32, u32)>)> = (0..SHARDS)
            .into_par_iter()
            .map(|sh| {
                let mut map: HashMap<&str, u32, ahash::RandomState> = HashMap::default();
                let mut strings: Vec<Box<str>> = Vec::new();
                let mut assign = Vec::new();
                for (fi, lp) in locals.iter().enumerate() {
                    for &li in &lp.seq_shard[sh] {
                        let s = lp.seq_uniq[li as usize].as_str();
                        let sid = *map.entry(s).or_insert_with(|| {
                            strings.push(s.into());
                            (strings.len() - 1) as u32
                        });
                        assign.push((fi as u32, li, sid));
                    }
                }
                (strings, assign)
            })
            .collect();
        let mut gmap: Vec<Vec<u32>> = locals.iter().map(|lp| vec![0u32; lp.seq_uniq.len()]).collect();
        let mut seqs = SeqTable::default();
        for (strings, assign) in shards {
            let off = seqs.strings.len() as u32;
            for (fi, li, sid) in assign {
                gmap[fi as usize][li as usize] = off + sid;
            }
            seqs.strings.extend(strings);
        }
        let mut genes = Interner::default();
        let gg: Vec<Vec<u32>> = locals.iter().map(|lp| lp.gene_uniq.iter().map(|s| genes.get_or_insert(s)).collect()).collect();
        let files: Vec<FileData> = locals
            .into_par_iter()
            .zip(gmap.into_par_iter())
            .zip(gg.into_par_iter())
            .map(|((lp, gs), gg)| FileData {
                seq: lp.seq_local.iter().map(|&i| gs[i as usize]).collect(),
                count: lp.count,
                frac: lp.frac,
                aa: lp.aa_local,
                aa_pool: lp.aa_uniq.into_iter().map(|s| s.into_boxed_str()).collect(),
                v: lp.v_local.iter().map(|&i| gg[i as usize]).collect(),
                j: lp.j_local.iter().map(|&i| gg[i as usize]).collect(),
            })
            .collect();
        Ok(ChainData { files, seqs, genes })
    }

    /// big_merge_freqs2(dtlist[file_idx], min_reads) followed by the `rowSums(m>0) > min_wells` filter.
    /// Returns (row seq ids kept, dense row-major matrix of kept rows, n rows before filter,
    /// file indices used as columns).
    pub fn merge(&self, file_idx: &[usize], min_reads: f64, min_wells: usize)
        -> (Vec<u32>, Vec<f64>, usize, Vec<usize>) {
        // columns: sources with >= 1 row passing the readCount filter, in order
        let cols: Vec<usize> = file_idx
            .iter()
            .copied()
            .filter(|&f| self.files[f].count.iter().any(|&c| c > min_reads))
            .collect();
        let ncol = cols.len();
        // rows: first appearance of targetSequences among filtered rows
        let mut row_of = vec![NO_ID; self.seqs.len()];
        let mut row_seq: Vec<u32> = Vec::new();
        for &f in &cols {
            let fd = &self.files[f];
            for k in 0..fd.nrow() {
                if fd.count[k] > min_reads {
                    let s = fd.seq[k] as usize;
                    if row_of[s] == NO_ID {
                        row_of[s] = row_seq.len() as u32;
                        row_seq.push(s as u32);
                    }
                }
            }
        }
        let nrow = row_seq.len();
        // per-row number of distinct columns with a positive (summed) value
        // entries are visited column by column, so duplicates of a (row,col) are summed in R order
        let mut last_col = vec![u32::MAX; nrow];
        let mut acc = vec![0.0f64; nrow];
        let mut npos = vec![0u32; nrow];
        let flush = |r: usize, acc: &mut Vec<f64>, npos: &mut Vec<u32>| {
            if acc[r] > 0.0 {
                npos[r] += 1;
            }
            acc[r] = 0.0;
        };
        for (ci, &f) in cols.iter().enumerate() {
            let fd = &self.files[f];
            for k in 0..fd.nrow() {
                if fd.count[k] > min_reads {
                    let r = row_of[fd.seq[k] as usize] as usize;
                    if last_col[r] != ci as u32 {
                        if last_col[r] != u32::MAX {
                            flush(r, &mut acc, &mut npos);
                        }
                        last_col[r] = ci as u32;
                    }
                    acc[r] += fd.frac[k];
                }
            }
        }
        for r in 0..nrow {
            if last_col[r] != u32::MAX {
                flush(r, &mut acc, &mut npos);
            }
        }
        let mut keep_row = vec![u32::MAX; nrow];
        let mut kept: Vec<u32> = Vec::new();
        for r in 0..nrow {
            if (npos[r] as usize) > min_wells {
                keep_row[r] = kept.len() as u32;
                kept.push(row_seq[r]);
            }
        }
        let mut mat = vec![0.0f64; kept.len() * ncol];
        for (ci, &f) in cols.iter().enumerate() {
            let fd = &self.files[f];
            for k in 0..fd.nrow() {
                if fd.count[k] > min_reads {
                    let r = row_of[fd.seq[k] as usize] as usize;
                    let kr = keep_row[r];
                    if kr != u32::MAX {
                        mat[kr as usize * ncol + ci] += fd.frac[k];
                    }
                }
            }
        }
        (kept, mat, nrow, cols)
    }

    /// TIRTLtools big_merge_freqs2(dtlist, min_reads, wells): columns are exactly `file_idx`;
    /// rows ordered by total readCount (descending; ties by sequence, C locale), then the
    /// `rowSums(m > 0) > min_wells` filter. Returns (seq ids, dense matrix, n rows before filter).
    pub fn merge_tt(&self, file_idx: &[usize], min_reads: f64, min_wells: usize) -> (Vec<u32>, Vec<f64>, usize) {
        let ncol = file_idx.len();
        let ns = self.seqs.len();
        // total reads per sequence (compensated sum in row order ~ R's long-double sum)
        let mut tot = vec![0.0f64; ns];
        let mut comp = vec![0.0f64; ns];
        let mut seen = vec![false; ns];
        let mut ids: Vec<u32> = Vec::new();
        for &f in file_idx {
            let fd = &self.files[f];
            for k in 0..fd.nrow() {
                let x = fd.count[k];
                if x > min_reads {
                    let sq = fd.seq[k] as usize;
                    if !seen[sq] {
                        seen[sq] = true;
                        ids.push(sq as u32);
                    }
                    let t = tot[sq] + x;
                    comp[sq] += if tot[sq].abs() >= x.abs() { (tot[sq] - t) + x } else { (x - t) + tot[sq] };
                    tot[sq] = t;
                }
            }
        }
        let strs = &self.seqs.strings;
        let total = |q: u32| tot[q as usize] + comp[q as usize];
        ids.par_sort_by(|&x, &y| {
            total(y).partial_cmp(&total(x)).unwrap_or(Ordering::Equal)
                .then_with(|| strs[x as usize].as_bytes().cmp(strs[y as usize].as_bytes()))
        });
        let nrow = ids.len();
        let mut row_of = vec![NO_ID; ns];
        for (r, &q) in ids.iter().enumerate() {
            row_of[q as usize] = r as u32;
        }
        // number of wells with a positive (summed) value per row
        let mut last_col = vec![u32::MAX; nrow];
        let mut acc = vec![0.0f64; nrow];
        let mut npos = vec![0u32; nrow];
        for (ci, &f) in file_idx.iter().enumerate() {
            let fd = &self.files[f];
            for k in 0..fd.nrow() {
                if fd.count[k] > min_reads {
                    let r = row_of[fd.seq[k] as usize] as usize;
                    if last_col[r] != ci as u32 {
                        if last_col[r] != u32::MAX && acc[r] > 0.0 {
                            npos[r] += 1;
                        }
                        acc[r] = 0.0;
                        last_col[r] = ci as u32;
                    }
                    acc[r] += fd.frac[k];
                }
            }
        }
        for r in 0..nrow {
            if last_col[r] != u32::MAX && acc[r] > 0.0 {
                npos[r] += 1;
            }
        }
        let mut keep_row = vec![u32::MAX; nrow];
        let mut kept: Vec<u32> = Vec::new();
        for r in 0..nrow {
            if (npos[r] as usize) > min_wells {
                keep_row[r] = kept.len() as u32;
                kept.push(ids[r]);
            }
        }
        let mut mat = vec![0.0f64; kept.len() * ncol];
        for (ci, &f) in file_idx.iter().enumerate() {
            let fd = &self.files[f];
            for k in 0..fd.nrow() {
                if fd.count[k] > min_reads {
                    let kr = keep_row[row_of[fd.seq[k] as usize] as usize];
                    if kr != u32::MAX {
                        mat[kr as usize * ncol + ci] += fd.frac[k];
                    }
                }
            }
        }
        (kept, mat, nrow)
    }

    /// add_VJ_aa(): for each requested sequence id, first aaSeqCDR3 and most frequent V / J token
    /// (ties -> first in R collation order, as names(sort(-table(x)))[1]) over the files in order.
    /// Returns (first aaSeqCDR3, v gene id, j gene id) per requested id (None / NO_ID when absent).
    pub fn annotate(&self, file_idx: &[usize], ids: &[u32]) -> (Vec<Option<&str>>, Vec<u32>, Vec<u32>) {
        let mut slot = vec![NO_ID; self.seqs.len()];
        for (k, &id) in ids.iter().enumerate() {
            slot[id as usize] = k as u32;
        }
        let n = ids.len();
        let mut aa: Vec<Option<&str>> = vec![None; n];
        let mut vc: Vec<HashMap<u32, u32>> = vec![HashMap::new(); n];
        let mut jc: Vec<HashMap<u32, u32>> = vec![HashMap::new(); n];
        for &f in file_idx {
            let fd = &self.files[f];
            for k in 0..fd.nrow() {
                let s = slot[fd.seq[k] as usize];
                if s == NO_ID {
                    continue;
                }
                let s = s as usize;
                if aa[s].is_none() {
                    aa[s] = Some(fd.aa_str(k));
                }
                *vc[s].entry(fd.v[k]).or_insert(0) += 1;
                *jc[s].entry(fd.j[k]).or_insert(0) += 1;
            }
        }
        let genes = &self.genes.strings;
        let pick = |m: &HashMap<u32, u32>| -> u32 {
            let mut best: Option<(u32, u32)> = None;
            for (&g, &c) in m {
                best = match best {
                    None => Some((g, c)),
                    Some((bg, bc)) => {
                        if c > bc || (c == bc && r_cmp(&genes[g as usize], &genes[bg as usize]) == Ordering::Less) {
                            Some((g, c))
                        } else {
                            Some((bg, bc))
                        }
                    }
                };
            }
            best.map(|b| b.0).unwrap_or(NO_ID)
        };
        let v: Vec<u32> = vc.par_iter().map(pick).collect();
        let j: Vec<u32> = jc.par_iter().map(pick).collect();
        (aa, v, j)
    }

    /// combineTCR(rbindlist(files, idcol="file")) ordered like fwrite(combd[order(-readCount),]).
    pub fn pseudobulk(&self, file_idx: &[usize]) -> Pseudobulk {
        // rows grouped by sequence (counting sort: groups keep the row order of the files);
        // data.table setkey sorts the groups by targetSequences (C locale)
        let nmax = file_idx.iter().filter(|&&f| self.files[f].nrow() > 0).count() as f64;
        let ns = self.seqs.len();
        let mut start = vec![0usize; ns + 1];
        for &f in file_idx {
            for &q in &self.files[f].seq {
                start[q as usize + 1] += 1;
            }
        }
        for q in 0..ns {
            start[q + 1] += start[q];
        }
        let mut fill = start.clone();
        let mut refs = vec![(0u32, 0u32); start[ns]];
        for (fi, &f) in file_idx.iter().enumerate() {
            for (k, &q) in self.files[f].seq.iter().enumerate() {
                refs[fill[q as usize]] = (fi as u32, k as u32);
                fill[q as usize] += 1;
            }
        }
        let mut keys: Vec<u32> = (0..ns as u32).filter(|&q| start[q as usize + 1] > start[q as usize]).collect();
        keys.par_sort_by(|a, b| self.seqs.strings[*a as usize].as_bytes().cmp(self.seqs.strings[*b as usize].as_bytes()));
        struct G<'a> { refs: &'a [(u32, u32)], start: &'a [usize] }
        impl<'a> std::ops::Index<&u32> for G<'a> {
            type Output = [(u32, u32)];
            fn index(&self, q: &u32) -> &[(u32, u32)] { &self.refs[self.start[*q as usize]..self.start[*q as usize + 1]] }
        }
        let groups = G { refs: &refs, start: &start };
        let rows: Vec<(u32, f64, u32, u32, &str, f64, f64, f64, f64, f64)> = keys
            .par_iter()
            .map(|&key| {
                let g = &groups[&key];
                let get = |fi: u32, _k: u32| &self.files[file_idx[fi as usize]];
                let counts: Vec<f64> = g.iter().map(|&(fi, k)| get(fi, k).count[k as usize]).collect();
                let fracs: Vec<f64> = g.iter().map(|&(fi, k)| get(fi, k).frac[k as usize]).collect();
                // x[which.max(tabulate(match(x, x)))]: highest count, ties -> first appearance.
                // Groups are small: linear scan over (value, count) in first-appearance order.
                fn mode_of<T: PartialEq + Copy>(xs: impl Iterator<Item = T>) -> T {
                    let mut tab: Vec<(T, u32)> = Vec::with_capacity(4);
                    for x in xs {
                        match tab.iter_mut().find(|e| e.0 == x) {
                            Some(e) => e.1 += 1,
                            None => tab.push((x, 1)),
                        }
                    }
                    let mut best = 0;
                    for t in 1..tab.len() {
                        if tab[t].1 > tab[best].1 {
                            best = t;
                        }
                    }
                    tab[best].0
                }
                let v = mode_of(g.iter().map(|&(fi, k)| get(fi, k).v[k as usize]));
                let j = mode_of(g.iter().map(|&(fi, k)| get(fi, k).j[k as usize]));
                let aa = mode_of(g.iter().map(|&(fi, k)| get(fi, k).aa_str(k as usize)));
                let n = g.len() as f64;
                let rc = r_sum(&counts);
                let rmax = counts.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut sorted = counts.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                let m = sorted.len();
                let med = if m % 2 == 1 { sorted[m / 2] } else { (sorted[m / 2 - 1] + sorted[m / 2]) / 2.0 };
                let avg = r_sum(&fracs) / nmax;
                // sd(c(readFraction, rep(0, nmax - .N))) / sqrt(nmax), R's two-pass var(); the
                // nz zeros contribute nz * (0 - mean) and nz * mean^2 (no padded vector)
                let nz = (nmax - n).max(0.0);
                let nn = n + nz;
                let mut mean = r_sum(&fracs) / nn;
                if mean.is_finite() {
                    let mut d: Vec<f64> = fracs.iter().map(|x| x - mean).collect();
                    d.push(nz * (0.0 - mean));
                    mean += r_sum(&d) / nn;
                }
                let mut dev: Vec<f64> = fracs.iter().map(|x| (x - mean) * (x - mean)).collect();
                dev.push(nz * (mean * mean));
                let sd = if nn > 1.0 { (r_sum(&dev) / (nn - 1.0)).sqrt() } else { f64::NAN };
                (key, rc, v, j, aa, n, rmax, med, avg, sd / nmax.sqrt())
            })
            .collect();
        let total = r_sum(&rows.iter().map(|r| r.1).collect::<Vec<_>>());
        let mut order: Vec<usize> = (0..rows.len()).collect();
        // order(-readCount): stable
        order.sort_by(|&a, &b| rows[b].1.partial_cmp(&rows[a].1).unwrap_or(Ordering::Equal));
        let mut pb = Pseudobulk::default();
        for i in order {
            let r = &rows[i];
            pb.seq.push(r.0);
            pb.read_count.push(r.1);
            pb.v.push(r.2);
            pb.j.push(r.3);
            pb.aa.push(r.4.to_string());
            pb.n_wells.push(r.5);
            pb.read_count_max.push(r.6);
            pb.read_count_median.push(r.7);
            pb.avg.push(r.8);
            pb.sem.push(r.9);
            pb.read_fraction.push(r.1 / total);
        }
        pb
    }
}

#[derive(Default)]
pub struct Pseudobulk {
    pub seq: Vec<u32>,
    pub read_count: Vec<f64>,
    pub v: Vec<u32>,
    pub j: Vec<u32>,
    pub aa: Vec<String>,
    pub n_wells: Vec<f64>,
    pub read_count_max: Vec<f64>,
    pub read_count_median: Vec<f64>,
    pub avg: Vec<f64>,
    pub sem: Vec<f64>,
    pub read_fraction: Vec<f64>,
}
