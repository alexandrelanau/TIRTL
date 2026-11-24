//! String ordering of R's sort()/list.files()/table() under ICU "root" collation (the R default when
//! built with ICU): primary = punctuation < symbols < digits < letters (case-insensitive), then
//! tertiary = lowercase before uppercase. Non-ASCII characters fall back to code-point order after
//! letters. Verified against R 4.4 (icuGetCollate() == "root").

use std::cmp::Ordering;

const PUNCT_ORDER: &[u8] = b" _-,;:!?.'\"()[]{}@*/\\&#%`^+<=>|~$";

#[inline]
fn primary(c: char) -> (u32, u32) {
    if c.is_ascii() {
        let b = c as u8;
        if let Some(p) = PUNCT_ORDER.iter().position(|&x| x == b) {
            return (0, p as u32);
        }
        if b.is_ascii_digit() {
            return (1, (b - b'0') as u32);
        }
        if b.is_ascii_alphabetic() {
            return (2, (b.to_ascii_lowercase() - b'a') as u32);
        }
        return (0, 100 + b as u32); // control chars etc.
    }
    (3, c as u32)
}

#[inline]
fn tertiary(c: char) -> u32 {
    if c.is_ascii_uppercase() {
        1
    } else {
        0
    }
}

pub fn r_cmp(a: &str, b: &str) -> Ordering {
    let pa = a.chars().map(primary);
    let pb = b.chars().map(primary);
    match pa.cmp(pb) {
        Ordering::Equal => {}
        o => return o,
    }
    let ta = a.chars().map(tertiary);
    let tb = b.chars().map(tertiary);
    match ta.cmp(tb) {
        Ordering::Equal => a.cmp(b),
        o => o,
    }
}

/// stable order (like R's order(x)) of the strings under R collation
pub fn r_order(xs: &[&str]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    idx.sort_by(|&i, &j| r_cmp(xs[i], xs[j]));
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_r() {
        // R 4.4.3, ICU root: sort(x)
        let x = ["a_b", "a-b", "ab", "a0", "aB", "Ab", "AB", "a b", "a.b", "a1", "a10", "a1_", "A1_S", "A10_S",
                 "TRBV6-2", "TRBV6-3", "TRBV6", "TRBV6-2*01", "x", "X", "x1", "X1"];
        let want = ["a b", "a_b", "a-b", "a.b", "a0", "a1", "a1_", "A1_S", "a10", "A10_S", "ab", "aB", "Ab", "AB",
                    "TRBV6", "TRBV6-2", "TRBV6-2*01", "TRBV6-3", "x", "X", "x1", "X1"];
        let o = r_order(&x);
        let got: Vec<&str> = o.iter().map(|&i| x[i]).collect();
        assert_eq!(got, want);
    }
}
