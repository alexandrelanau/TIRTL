//! Fast TSV writer producing the same text as pandas.DataFrame.to_csv(sep="\t", index=False)
//! for float64 (Python repr, NaN -> ""), int64 and string columns.

use rayon::prelude::*;
use std::io::Write;

pub enum Col<'a> {
    F64(&'a [f64]),
    I64(&'a [i64]),
    /// codes into a dictionary; negative / out of range -> empty field
    Str(&'a [i64], &'a [String]),
    /// dict1[codes1] + sep + dict2[codes2]
    Concat(&'a [i64], &'a [String], &'a [i64], &'a [String], &'a str),
}

/// Python repr() of a finite or non-finite float
pub fn py_repr(x: f64, buf: &mut Vec<u8>) {
    if x.is_nan() {
        buf.extend_from_slice(b"nan");
        return;
    }
    if x.is_infinite() {
        buf.extend_from_slice(if x > 0.0 { b"inf" } else { b"-inf" });
        return;
    }
    if x == 0.0 {
        buf.extend_from_slice(if x.is_sign_negative() { b"-0.0" } else { b"0.0" });
        return;
    }
    let mut rb = ryu::Buffer::new();
    let s = rb.format_finite(x);
    // decompose ryu output into sign, digits, decimal exponent
    let (neg, s) = if let Some(t) = s.strip_prefix('-') { (true, t) } else { (false, s) };
    let (mant, exp) = match s.find(['e', 'E']) {
        Some(p) => (&s[..p], s[p + 1..].parse::<i32>().unwrap()),
        None => (s, 0),
    };
    let (ip, fp) = match mant.find('.') {
        Some(p) => (&mant[..p], &mant[p + 1..]),
        None => (mant, ""),
    };
    let mut digits: String = format!("{ip}{fp}");
    let mut decpt = ip.len() as i32 + exp;
    // strip leading zeros
    let lz = digits.len() - digits.trim_start_matches('0').len();
    digits.drain(..lz);
    decpt -= lz as i32;
    let tz = digits.len() - digits.trim_end_matches('0').len();
    digits.truncate(digits.len() - tz);
    if digits.is_empty() {
        digits.push('0');
        decpt = 1;
    }
    if neg {
        buf.push(b'-');
    }
    let nd = digits.len() as i32;
    if decpt <= -4 || decpt > 16 {
        let d = digits.as_bytes();
        buf.push(d[0]);
        if nd > 1 {
            buf.push(b'.');
            buf.extend_from_slice(&d[1..]);
        }
        let e = decpt - 1;
        buf.push(b'e');
        buf.push(if e < 0 { b'-' } else { b'+' });
        let _ = write!(buf, "{:02}", e.abs());
    } else if decpt <= 0 {
        buf.extend_from_slice(b"0.");
        for _ in 0..(-decpt) {
            buf.push(b'0');
        }
        buf.extend_from_slice(digits.as_bytes());
    } else if decpt >= nd {
        buf.extend_from_slice(digits.as_bytes());
        for _ in 0..(decpt - nd) {
            buf.push(b'0');
        }
        buf.extend_from_slice(b".0");
    } else {
        let d = digits.as_bytes();
        buf.extend_from_slice(&d[..decpt as usize]);
        buf.push(b'.');
        buf.extend_from_slice(&d[decpt as usize..]);
    }
}

#[inline]
fn put_str(buf: &mut Vec<u8>, codes: &[i64], dict: &[String], i: usize) {
    let c = codes[i];
    if c >= 0 && (c as usize) < dict.len() {
        buf.extend_from_slice(dict[c as usize].as_bytes());
    }
}

pub fn write_tsv(path: &str, names: &[String], cols: &[Col], nrow: usize) -> std::io::Result<()> {
    let f = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::with_capacity(1 << 22, f);
    w.write_all(names.join("\t").as_bytes())?;
    w.write_all(b"\n")?;
    let block = 65536;
    let nblocks = (nrow + block - 1) / block;
    // format blocks in parallel, a bounded number at a time, write in order
    let par = rayon::current_num_threads().max(1) * 2;
    let mut start_block = 0;
    while start_block < nblocks {
        let end_block = (start_block + par).min(nblocks);
        let bufs: Vec<Vec<u8>> = (start_block..end_block)
            .into_par_iter()
            .map(|bk| {
                let lo = bk * block;
                let hi = ((bk + 1) * block).min(nrow);
                let mut buf = Vec::with_capacity((hi - lo) * 200);
                for i in lo..hi {
                    for (k, c) in cols.iter().enumerate() {
                        if k > 0 {
                            buf.push(b'\t');
                        }
                        match c {
                            Col::F64(v) => {
                                if !v[i].is_nan() {
                                    py_repr(v[i], &mut buf)
                                }
                            }
                            Col::I64(v) => {
                                let _ = write!(buf, "{}", v[i]);
                            }
                            Col::Str(codes, dict) => put_str(&mut buf, codes, dict, i),
                            Col::Concat(c1, d1, c2, d2, sep) => {
                                put_str(&mut buf, c1, d1, i);
                                buf.extend_from_slice(sep.as_bytes());
                                put_str(&mut buf, c2, d2, i);
                            }
                        }
                    }
                    buf.push(b'\n');
                }
                buf
            })
            .collect();
        for b in bufs {
            w.write_all(&b)?;
        }
        start_block = end_block;
    }
    w.flush()
}

/// Arrow large-string buffers (offsets i64, utf-8 data, validity) for dict[codes[i]]
/// (or d1[c1[i]] + sep + d2[c2[i]] when `second` is given); negative codes -> null.
pub fn arrow_strings(c1: &[i64], d1: &[String], second: Option<(&[i64], &[String], &str)>)
    -> (Vec<i64>, Vec<u8>, Vec<bool>) {
    let n = c1.len();
    let get = |d: &[String], c: i64| -> Option<usize> { if c >= 0 && (c as usize) < d.len() { Some(c as usize) } else { None } };
    let lens: Vec<(usize, bool)> = (0..n)
        .into_par_iter()
        .map(|i| match second {
            None => match get(d1, c1[i]) { Some(k) => (d1[k].len(), true), None => (0, false) },
            Some((c2, d2, sep)) => {
                let a = get(d1, c1[i]).map(|k| d1[k].len()).unwrap_or(0);
                let b = get(d2, c2[i]).map(|k| d2[k].len()).unwrap_or(0);
                (a + sep.len() + b, true)
            }
        })
        .collect();
    let mut offsets = vec![0i64; n + 1];
    for i in 0..n {
        offsets[i + 1] = offsets[i] + lens[i].0 as i64;
    }
    let mut data = vec![0u8; offsets[n] as usize];
    // fill in parallel: split data into disjoint row blocks
    let block = 16384;
    let mut slices: Vec<(usize, &mut [u8])> = Vec::new();
    let mut rest: &mut [u8] = &mut data;
    let mut start = 0;
    while start < n {
        let end = (start + block).min(n);
        let len = (offsets[end] - offsets[start]) as usize;
        let (head, tail) = rest.split_at_mut(len);
        slices.push((start, head));
        rest = tail;
        start = end;
    }
    slices.into_par_iter().for_each(|(st, buf)| {
        let mut pos = 0usize;
        let end = (st + block).min(n);
        for i in st..end {
            let mut put = |b: &[u8]| {
                buf[pos..pos + b.len()].copy_from_slice(b);
                pos += b.len();
            };
            match second {
                None => {
                    if let Some(k) = get(d1, c1[i]) {
                        put(d1[k].as_bytes())
                    }
                }
                Some((c2, d2, sep)) => {
                    if let Some(k) = get(d1, c1[i]) {
                        put(d1[k].as_bytes())
                    }
                    put(sep.as_bytes());
                    if let Some(k) = get(d2, c2[i]) {
                        put(d2[k].as_bytes())
                    }
                }
            }
        }
    });
    (offsets, data, lens.into_iter().map(|x| x.1).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn r(x: f64) -> String {
        let mut b = vec![];
        py_repr(x, &mut b);
        String::from_utf8(b).unwrap()
    }
    #[test]
    fn python_repr() {
        assert_eq!(r(6.0), "6.0");
        assert_eq!(r(0.2222222222222222), "0.2222222222222222");
        assert_eq!(r(1e-10), "1e-10");
        assert_eq!(r(1e-5), "1e-05");
        assert_eq!(r(0.0001), "0.0001");
        assert_eq!(r(1e16), "1e+16");
        assert_eq!(r(1e15), "1000000000000000.0");
        assert_eq!(r(123.456), "123.456");
        assert_eq!(r(-2.5e-7), "-2.5e-07");
        assert_eq!(r(4.649647203399094), "4.649647203399094");
        assert_eq!(r(1.5e300), "1.5e+300");
        assert_eq!(r(f64::INFINITY), "inf");
    }
}
