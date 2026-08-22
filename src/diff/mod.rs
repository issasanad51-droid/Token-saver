//! Direct Lossless File-Patch Diff Streams.
//!
//! Instead of feeding the AI raw text and expecting text back, this layer
//! forces compact Git-style unified diff blocks (e.g. `@@ -45,4 +45,5 @@`),
//! dropping the AI's output token usage dramatically. The diff is lossless:
//! [`DiffGenerator::apply_diff`] reconstructs the exact modified text.

use serde::{Deserialize, Serialize};

/// A single line-level diff operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Equal,
    Delete,
    Insert,
}

/// Errors produced while parsing/applying a unified diff.
#[derive(Debug, thiserror::Error)]
pub enum DiffError {
    #[error("malformed diff: {0}")]
    Malformed(String),
    #[error("context mismatch at line {line}: expected {expected}")]
    ContextMismatch { line: usize, expected: String },
}

/// A grouped hunk of diff operations.
#[derive(Debug, Clone)]
struct Hunk {
    /// 1-based starting line in the original file.
    old_start: usize,
    /// 1-based starting line in the modified file.
    new_start: usize,
    ops: Vec<(Op, usize, usize)>,
}

/// Compute and apply Git-style unified diffs.
pub struct DiffGenerator;

impl DiffGenerator {
    /// Produce a Git-style unified diff with `context` lines of context.
    pub fn unified_diff(original: &str, modified: &str, context: usize) -> String {
        let a: Vec<&str> = original.lines().collect();
        let b: Vec<&str> = modified.lines().collect();
        if a.is_empty() && b.is_empty() {
            return String::new();
        }
        let ops = lcs_ops(&a, &b);
        let hunks = group_hunks(&ops, context);
        if hunks.is_empty() {
            return String::new();
        }
        let mut out = String::new();
        out.push_str("--- a/original\n+++ b/modified\n");
        for hunk in hunks {
            out.push_str(&render_hunk(&a, &b, &hunk));
        }
        out
    }

    /// Apply a unified diff to `original`, returning the patched text.
    pub fn apply_diff(original: &str, diff: &str) -> Result<String, DiffError> {
        let original_lines: Vec<&str> = original.lines().collect();
        let diff_lines: Vec<&str> = diff.lines().collect();
        let mut result: Vec<String> = Vec::new();
        let mut old_pos = 0usize;
        let mut idx = 0usize;

        // Skip file headers.
        while idx < diff_lines.len()
            && (diff_lines[idx].starts_with("--- ")
                || diff_lines[idx].starts_with("+++ ")
                || diff_lines[idx].is_empty())
        {
            idx += 1;
        }

        while idx < diff_lines.len() {
            let line = diff_lines[idx];
            if !line.starts_with("@@") {
                return Err(DiffError::Malformed(format!("expected hunk, got: {line}")));
            }
            let (old_start, _old_count, _new_start, _new_count) = parse_hunk_header(line)?;
            // Copy unchanged lines up to the hunk start.
            let target = old_start.saturating_sub(1);
            while old_pos < target && old_pos < original_lines.len() {
                result.push(original_lines[old_pos].to_string());
                old_pos += 1;
            }
            idx += 1;
            while idx < diff_lines.len() {
                let l = diff_lines[idx];
                if l.starts_with("@@") {
                    break;
                }
                if let Some(content) = l.strip_prefix(' ') {
                    if old_pos >= original_lines.len() || original_lines[old_pos] != content {
                        return Err(DiffError::ContextMismatch {
                            line: old_pos + 1,
                            expected: content.to_string(),
                        });
                    }
                    result.push(content.to_string());
                    old_pos += 1;
                    idx += 1;
                } else if let Some(content) = l.strip_prefix('-') {
                    if old_pos >= original_lines.len() || original_lines[old_pos] != content {
                        return Err(DiffError::ContextMismatch {
                            line: old_pos + 1,
                            expected: content.to_string(),
                        });
                    }
                    old_pos += 1;
                    idx += 1;
                } else if let Some(content) = l.strip_prefix('+') {
                    result.push(content.to_string());
                    idx += 1;
                } else {
                    return Err(DiffError::Malformed(format!("unexpected line: {l}")));
                }
            }
        }

        // Append any remaining original lines.
        while old_pos < original_lines.len() {
            result.push(original_lines[old_pos].to_string());
            old_pos += 1;
        }
        Ok(result.join("\n"))
    }
}

/// Longest-common-subsequence line diff. Falls back to a whole-file replace
/// for pathologically large inputs to bound memory.
fn lcs_ops(a: &[&str], b: &[&str]) -> Vec<(Op, usize, usize)> {
    let n = a.len();
    let m = b.len();
    if n * m > 4_000_000 {
        let mut ops = Vec::new();
        for i in 0..n {
            ops.push((Op::Delete, i, 0));
        }
        for j in 0..m {
            ops.push((Op::Insert, 0, j));
        }
        return ops;
    }
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i][j] = if a[i] == b[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }
    let mut ops = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n && j < m {
        if a[i] == b[j] {
            ops.push((Op::Equal, i, j));
            i += 1;
            j += 1;
        } else if dp[i + 1][j] >= dp[i][j + 1] {
            ops.push((Op::Delete, i, j));
            i += 1;
        } else {
            ops.push((Op::Insert, i, j));
            j += 1;
        }
    }
    while i < n {
        ops.push((Op::Delete, i, m));
        i += 1;
    }
    while j < m {
        ops.push((Op::Insert, n, j));
        j += 1;
    }
    ops
}

fn group_hunks(ops: &[(Op, usize, usize)], context: usize) -> Vec<Hunk> {
    let n = ops.len();
    let mut hunks = Vec::new();
    let mut i = 0;
    while i < n {
        if ops[i].0 == Op::Equal {
            i += 1;
            continue;
        }
        let change_start = i;
        let mut j = i;
        while j < n && ops[j].0 != Op::Equal {
            j += 1;
        }
        let change_end = j;
        let start = change_start.saturating_sub(context);
        let end = (change_end + context).min(n);
        let (_, oi, bi) = ops[start];
        let (old_start, new_start) = (oi + 1, bi + 1);        hunks.push(Hunk {
            old_start,
            new_start,
            ops: ops[start..end].to_vec(),
        });
        i = end;
    }
    hunks
}

fn render_hunk(a: &[&str], b: &[&str], hunk: &Hunk) -> String {
    let old_count = hunk.ops.iter().filter(|(op, _, _)| *op != Op::Insert).count();
    let new_count = hunk.ops.iter().filter(|(op, _, _)| *op != Op::Delete).count();
    let mut out = String::new();
    out.push_str(&format!(
        "@@ -{},{} +{},{} @@\n",
        hunk.old_start, old_count, hunk.new_start, new_count
    ));
    for (op, oi, bi) in &hunk.ops {
        match op {
            Op::Equal => out.push_str(&format!(" {}\n", a[*oi])),
            Op::Delete => out.push_str(&format!("-{}\n", a[*oi])),
            Op::Insert => out.push_str(&format!("+{}\n", b[*bi])),
        }
    }
    out
}

fn parse_hunk_header(header: &str) -> Result<(usize, usize, usize, usize), DiffError> {
    let inner = header
        .trim_start_matches("@@")
        .trim_end_matches("@@")
        .trim();
    let parts: Vec<&str> = inner.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(DiffError::Malformed(format!("bad hunk header: {header}")));
    }
    let parse_range = |s: &str| -> Result<(usize, usize), DiffError> {
        let s = s.trim_start_matches(['-', '+']);
        if let Some((start, count)) = s.split_once(',') {
            Ok((
                start.parse().unwrap_or(1),
                count.parse().unwrap_or(1),
            ))
        } else {
            Ok((s.parse().unwrap_or(1), 1))
        }
    };
    let (old_start, old_count) = parse_range(parts[0])?;
    let (new_start, new_count) = parse_range(parts[1])?;
    Ok((old_start, old_count, new_start, new_count))
}

/// Serialized stats for a computed diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffStats {
    pub hunks: usize,
    pub added_lines: usize,
    pub removed_lines: usize,
    pub output_tokens: usize,
    pub output_bytes: usize,
}

impl DiffStats {
    pub fn from_diff(diff: &str) -> Self {
        let hunks = diff.matches("@@").count() / 2;
        let added_lines = diff
            .lines()
            .filter(|l| l.starts_with('+') && !l.starts_with("+++"))
            .count();
        let removed_lines = diff
            .lines()
            .filter(|l| l.starts_with('-') && !l.starts_with("---"))
            .count();
        Self {
            hunks,
            added_lines,
            removed_lines,
            output_tokens: crate::tracker::estimate_tokens(diff),
            output_bytes: diff.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_roundtrip() {
        let original = "fn a() {\n    let x = 1;\n}\nfn b() {\n    let y = 2;\n}\n";
        let modified = "fn a() {\n    let x = 2;\n}\nfn b() {\n    let y = 2;\n}\nfn c() {}\n";
        let diff = DiffGenerator::unified_diff(original, modified, 3);
        assert!(diff.contains("@@"));
        let applied = DiffGenerator::apply_diff(original, &diff).unwrap();
        assert_eq!(applied, modified);
    }

    #[test]
    fn empty_diff_when_identical() {
        let src = "fn a() {}\n";
        let diff = DiffGenerator::unified_diff(src, src, 3);
        assert!(diff.is_empty());
    }
}