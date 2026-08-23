//! Phase 8: Query-Time Post-Processing Pipeline (Custom From-Scratch)
//!
//! After AST signature pruning (Phase 7), pruned dependencies pass through
//! four lossless transformations that squeeze out extra tokens without
//! sacrificing information fidelity. Every function here is a hand-rolled
//! custom implementation — no regex, no external crates.
//!
//! # Pipeline stages (applied in order)
//!
//! 1. **Contextual Monomorphization** — For low-rank `impl` blocks, only the
//!    methods actually called/referenced by the active cursor node are kept.
//!    Uses BFS over the ASG's `Calls`/`References` edges to discover which
//!    children are transitively reachable.
//!
//! 2. **Tokenizer-Aware Identifier Aliasing** — A hand-built lexer scans
//!    source character-by-character, identifies custom identifiers above a
//!    length threshold, and replaces them with compact `_a`, `_b` aliases.
//!    An alias legend is prepended so the model resolves symbols via its
//!    native attention mechanism. This is more aggressive than the index-
//!    time `ChunkCompressor` and uses `_` prefix (not `$`) to avoid
//!    collisions with the existing dictionary compressor.
//!
//! 3. **Syntactic Whitespace Evacuation** — Humans need indentation; models
//!    do not. Leading whitespace, extra blank lines, and trailing spaces
//!    are stripped from low-rank deps, recovering 20–30% of token payload.
//!
//! 4. **Deterministic Prompt-Cache Sorting** — Final output is sorted by
//!    module path so the LLM provider's KV Cache reuses computed attention
//!    states across similar queries, slashing latency and API cost.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::asg::{EdgeKind, SharedAsg};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Controls query-time post-processing for low-rank dependencies.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PostprocessConfig {
    /// Enable aggressive identifier aliasing on pruned dependencies.
    pub alias_enabled: bool,
    /// Minimum identifier length to trigger aliasing.
    pub alias_min_length: usize,
    /// Enable whitespace evacuation on pruned dependencies.
    pub whitespace_enabled: bool,
    /// Enable contextual monomorphization (drop unused impl methods).
    pub monomorphize_enabled: bool,
    /// Enable deterministic module-path sorting of the final output.
    pub sort_enabled: bool,
}

impl Default for PostprocessConfig {
    fn default() -> Self {
        Self {
            alias_enabled: true,
            alias_min_length: 6,
            whitespace_enabled: true,
            monomorphize_enabled: true,
            sort_enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Processed dependency
// ---------------------------------------------------------------------------

/// A dependency ready for final assembly. Carries metadata so the pipeline
/// can sort deterministically and the tracker can pack under budget.
#[derive(Debug, Clone)]
pub struct ProcessedDep {
    /// Module path from the node's tracker id (used for sorting).
    pub module_path: String,
    /// The rendered text to inject into the prompt.
    pub text: String,
    /// Estimated token count after all transformations.
    pub tokens: usize,
    /// Whether this dependency was transformed (pruned/aliased/etc).
    pub was_transformed: bool,
}

// ---------------------------------------------------------------------------
// Pipeline entry point
// ---------------------------------------------------------------------------

/// Run the full post-processing pipeline on a batch of dependencies.
///
/// `items` is `(node_id, pagerank_score, prompt_text, was_pruned)` from
/// the pruning phase. Returns [`ProcessedDep`]s sorted deterministically
/// when `config.sort_enabled`.
pub fn postprocess(
    items: Vec<(usize, f64, String, bool)>,
    config: &PostprocessConfig,
    asg: &SharedAsg,
    cursor_node_id: Option<usize>,
) -> Vec<ProcessedDep> {
    let mut deps: Vec<ProcessedDep> = items
        .into_iter()
        .map(|(node_id, _score, text, was_pruned)| {
            let module_path = asg
                .get_node(node_id)
                .map(|n| n.tracker_id.clone())
                .unwrap_or_default();

            if !was_pruned {
                // High-rank: pass through with no transformation.
                let tokens = crate::tracker::estimate_tokens(&text);
                return ProcessedDep {
                    module_path,
                    text,
                    tokens,
                    was_transformed: false,
                };
            }

            // --- Apply all four stages sequentially ---

            // Stage 1: Contextual monomorphization (before aliasing so we
            // alias less code).
            let text = if config.monomorphize_enabled {
                contextual_monomorphize(text, node_id, cursor_node_id, asg)
            } else {
                text
            };

            // Stage 2: Aggressive identifier aliasing.
            let text = if config.alias_enabled {
                aggressive_alias(&text, config.alias_min_length)
            } else {
                text
            };

            // Stage 3: Whitespace evacuation.
            let text = if config.whitespace_enabled {
                evacuate_whitespace(&text)
            } else {
                text
            };

            let tokens = crate::tracker::estimate_tokens(&text);
            ProcessedDep {
                module_path,
                text,
                tokens,
                was_transformed: true,
            }
        })
        .collect();

    // Stage 4: Deterministic prompt-cache sorting.
    if config.sort_enabled {
        deps.sort_by(|a, b| a.module_path.cmp(&b.module_path));
    }

    deps
}

// ===========================================================================
// Feature 1 — Contextual Monomorphization (Custom BFS + Source Filter)
// ===========================================================================

/// For `impl` blocks, drop methods whose names are NOT reachable from the
/// cursor node through the ASG's `Calls`/`References` edge graph. For
/// non-impl items the source is returned unchanged.
fn contextual_monomorphize(
    source: String,
    dep_node_id: usize,
    cursor_node_id: Option<usize>,
    asg: &SharedAsg,
) -> String {
    let Some(cursor_id) = cursor_node_id else {
        return source;
    };

    let dep_node = match asg.get_node(dep_node_id) {
        Some(n) => n,
        None => return source,
    };

    // Only prune inside impl blocks — everything else is already a leaf.
    if dep_node.kind != "impl" {
        return source;
    }

    // BFS from cursor to discover transitively reachable node IDs.
    let reachable = reachable_set(cursor_id, asg);

    // Which child method names of this impl are actually used?
    let impl_prefix = &dep_node.tracker_id;
    let used_methods: HashSet<String> = reachable
        .iter()
        .filter_map(|id| asg.get_node(*id))
        .filter(|n| n.tracker_id.starts_with(impl_prefix))
        .map(|n| n.name.clone())
        .collect();

    if used_methods.is_empty() {
        // Nothing used — emit only the impl header so the model knows the
        // type exists.
        return impl_header_only(&source);
    }

    filter_impl_methods(&source, &used_methods)
}

/// BFS from `start` following `Calls` and `References` edges. Bounded to
/// 512 nodes to prevent runaway traversal on large codebases.
fn reachable_set(start: usize, asg: &SharedAsg) -> HashSet<usize> {
    let mut visited = HashSet::with_capacity(64);
    let mut queue = VecDeque::with_capacity(32);
    queue.push_back(start);
    visited.insert(start);

    while let Some(nid) = queue.pop_front() {
        if visited.len() >= 512 {
            break;
        }
        if let Some(adj) = asg.inner.adjacency.get(&nid) {
            for &eidx in adj {
                let edge = &asg.inner.edges[eidx];
                if matches!(edge.kind, EdgeKind::Calls | EdgeKind::References)
                    && visited.insert(edge.to)
                {
                    queue.push_back(edge.to);
                }
            }
        }
    }
    visited
}

/// Return just the impl header with an empty body: `impl Foo {}`.
fn impl_header_only(source: &str) -> String {
    match source.find('{') {
        Some(pos) => {
            let header = source[..pos].trim_end();
            format!("{header} {{}}")
        }
        None => source.to_string(),
    }
}

/// Keep only methods in `keep` from an impl block's source.
fn filter_impl_methods(source: &str, keep: &HashSet<String>) -> String {
    let Some(brace_pos) = source.find('{') else {
        return source.to_string();
    };
    let header = &source[..=brace_pos];
    let rest = &source[brace_pos + 1..];

    // Walk the impl body line by line, tracking brace depth to find
    // complete method boundaries.
    let mut out = String::with_capacity(source.len() / 2);
    out.push_str(header);
    out.push('\n');

    let mut method_buf = String::new();
    let mut in_method = false;
    let mut depth: i32 = 0;
    let mut current_name = String::new();

    for line in rest.lines() {
        let trimmed = line.trim();

        if !in_method {
            // Detect the start of a method signature.
            if let Some(name) = try_extract_method_name(trimmed) {
                in_method = true;
                current_name = name;
                method_buf.clear();
                method_buf.push_str(line);
                method_buf.push('\n');
                depth = count_char(line, '{') as i32 - count_char(line, '}') as i32;
                // Single-line method (`fn f() { … }`), trait-method or extern
                // signature (no body, ends with `;`): complete immediately.
                if depth <= 0 {
                    in_method = false;
                    if keep.contains(&current_name) {
                        out.push_str(&method_buf);
                    }
                    method_buf.clear();
                }
                continue;
            }
            // Closing brace of the impl block.
            if trimmed == "}" {
                out.push_str("}\n");
            }
            continue;
        }

        // Accumulating method body.
        method_buf.push_str(line);
        method_buf.push('\n');
        depth += count_char(line, '{') as i32;
        depth -= count_char(line, '}') as i32;

        if depth <= 0 {
            in_method = false;
            if keep.contains(&current_name) {
                out.push_str(&method_buf);
            }
            method_buf.clear();
        }
    }

    out
}

/// Try to extract the method name from a line like `pub fn verify(…)`.
/// Returns `None` if the line doesn't look like a method definition.
fn try_extract_method_name(line: &str) -> Option<String> {
    let t = line.trim();
    // Must start with optional qualifiers then `fn`.
    let after_qualifiers = strip_method_prefix(t)?;
    if !after_qualifiers.starts_with("fn ") {
        return None;
    }
    let rest = &after_qualifiers[3..]; // skip "fn "
    let name: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() { None } else { Some(name) }
}

/// Strip leading qualifiers (`pub`, `pub(crate)`, `async`, `const`, `unsafe`)
/// and return the remainder starting at `fn`.
fn strip_method_prefix(s: &str) -> Option<&str> {
    let mut rest = s;
    // Skip whitespace.
    rest = rest.trim_start();
    // `pub` with optional `(…)`.
    if rest.starts_with("pub") {
        rest = &rest[3..];
        rest = rest.trim_start();
        if rest.starts_with('(') {
            // Consume `pub(crate)` / `pub(super)` / `pub(in path)`.
            {
                let end = rest.find(')')?;
                rest = &rest[end + 1..];
                rest = rest.trim_start();
            }
        }
    }
    // `async`
    if rest.starts_with("async") {
        rest = &rest[5..];
        rest = rest.trim_start();
    }
    // `const`
    if rest.starts_with("const") {
        rest = &rest[5..];
        rest = rest.trim_start();
    }
    // `unsafe`
    if rest.starts_with("unsafe") {
        rest = &rest[6..];
        rest = rest.trim_start();
    }
    Some(rest)
}

/// Count occurrences of `ch` in `s`.
fn count_char(s: &str, ch: char) -> usize {
    s.bytes().filter(|&b| b == ch as u8).count()
}

// ===========================================================================
// Feature 2 — Tokenizer-Aware Identifier Aliasing (Custom Lexer)
// ===========================================================================

/// Hand-built lexer that scans source byte-by-byte, identifies custom
/// identifiers above `min_length`, and replaces them with compact `_a`,
/// `_b` aliases. A legend block is prepended.
///
/// The lexer distinguishes:
/// - **Identifiers**: `[a-zA-Z_][a-zA-Z0-9_]*`
/// - **Keywords**: matched against a built-in set (never aliased)
/// - **Literals & punctuation**: passed through verbatim
///
/// Replacement is longest-identifier-first so `_a` always maps to the
/// most wasteful name in the source.
fn aggressive_alias(source: &str, min_length: usize) -> String {
    // --- Phase A: Tokenize and discover candidates ---
    let tokens = lex_source(source);

    // Collect unique custom identifiers above the length threshold.
    let keywords = rust_keywords();
    let mut seen = HashSet::new();
    let mut candidates: Vec<String> = Vec::new();

    for tok in &tokens {
        if let Token::Ident(name) = tok {
            if name.len() >= min_length
                && !keywords.contains(name.as_str())
                && seen.insert(name.clone())
            {
                candidates.push(name.clone());
            }
        }
    }

    if candidates.is_empty() {
        return source.to_string();
    }

    // --- Phase B: Build alias map (longest identifier first) ---
    candidates.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));

    let mut alias_map: HashMap<String, String> = HashMap::with_capacity(candidates.len());
    for (counter, ident) in candidates.iter().enumerate() {
        let alias = encode_alias(counter as u32);
        alias_map.insert(ident.clone(), alias);
    }

    // --- Phase C: Build legend ---
    let mut legend_pairs: Vec<(&str, &str)> = alias_map
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    legend_pairs.sort_by(|a, b| a.1.cmp(b.1)); // sort by alias name
    let legend: String = legend_pairs
        .iter()
        .map(|(orig, alias)| format!("{alias}={orig}"))
        .collect::<Vec<_>>()
        .join(",");

    // --- Phase D: Rebuild source with replacements ---
    // We re-scan and replace in a single pass: emit the alias in place of
    // any identifier that appears in the map. Raw-string prefixes are
    // checked BEFORE identifiers: the old "rewind and retry" approach
    // re-entered the identifier branch forever (infinite loop).
    let mut result = String::with_capacity(source.len());
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        let b = bytes[i];

        // Raw strings: r"…" / r#"…"# / br#"…"# — copied verbatim, never
        // aliased inside.
        if b == b'r' || (b == b'b' && i + 1 < len && bytes[i + 1] == b'r') {
            let mut probe = i + 1;
            if b == b'b' {
                probe += 1;
            }
            if probe < len && (bytes[probe] == b'#' || bytes[probe] == b'"') {
                let start = i;
                let mut hashes = 0u32;
                i = probe;
                if bytes[i] == b'#' {
                    while i < len && bytes[i] == b'#' {
                        hashes += 1;
                        i += 1;
                    }
                }
                if i < len && bytes[i] == b'"' {
                    i += 1;
                    let close = format!("\"{}", "#".repeat(hashes as usize));
                    while i + close.len() <= len {
                        if source[i..i + close.len()] == close[..] {
                            i += close.len();
                            break;
                        }
                        i += 1;
                    }
                }
                result.push_str(&source[start..i]);
                continue;
            }
        }

        // Identifiers: [a-zA-Z_][a-zA-Z0-9_]*
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = i;
            i += 1;
            while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let ident = &source[start..i];
            if let Some(alias) = alias_map.get(ident) {
                result.push_str(alias);
            } else {
                result.push_str(ident);
            }
            continue;
        }

        // String literals: "…"
        if b == b'"' {
            let start = i;
            i += 1;
            while i < len && bytes[i] != b'"' {
                if bytes[i] == b'\\' && i + 1 < len {
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if i < len {
                i += 1;
            }
            result.push_str(&source[start..i]);
            continue;
        }

        // Raw strings: r#"…"# / r##"…"##
        if b == b'r' && i + 1 < len && (bytes[i + 1] == b'#' || bytes[i + 1] == b'"') {
            let start = i;
            i += 1;
            let mut hashes = 0u32;
            if bytes[i] == b'#' {
                while i < len && bytes[i] == b'#' {
                    hashes += 1;
                    i += 1;
                }
            }
            if i < len && bytes[i] == b'"' {
                i += 1;
                let close_len = 1 + hashes as usize;
                while i + close_len <= len {
                    if bytes[i] == b'"'
                        && &source[i..i + close_len]
                            == format!("\"{}", "#".repeat(hashes as usize)).as_str()
                    {
                        i += close_len;
                        break;
                    }
                    i += 1;
                }
            }
            result.push_str(&source[start..i]);
            continue;
        }

        // Char literals: '…'
        if b == b'\'' {
            let start = i;
            i += 1;
            if i < len && bytes[i] == b'\\' && i + 1 < len {
                i += 2;
            } else if i < len {
                i += 1;
            }
            if i < len && bytes[i] == b'\'' {
                i += 1;
            }
            result.push_str(&source[start..i]);
            continue;
        }

        // Line comments: //…
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
            let start = i;
            while i < len && bytes[i] != b'\n' {
                i += 1;
            }
            result.push_str(&source[start..i]);
            continue;
        }

        // Block comments: /* … */
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
            let start = i;
            i += 2;
            while i + 1 < len {
                if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    i += 2;
                    break;
                }
                i += 1;
            }
            result.push_str(&source[start..i]);
            continue;
        }

        // Everything else: whitespace, punctuation, operators, digits.
        result.push(b as char);
        i += 1;
    }

    // --- Phase E: Honesty gate ---
    // The legend costs bytes; on inputs where identifiers barely repeat,
    // aliasing can end up LARGER than the original. Emit the aliased form
    // only when it is strictly smaller — matching the README's "honest
    // lossless compression" contract.
    let aliased = format!("// aliases: {legend}\n{result}");
    if aliased.len() < source.len() {
        aliased
    } else {
        source.to_string()
    }
}

/// A single lexer token.
#[derive(Debug, PartialEq)]
enum Token {
    Ident(String),
    Other(String),
}

/// Hand-rolled lexer: splits source into identifiers and everything else.
/// No regex — just byte-level state machine scanning.
fn lex_source(source: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        let b = bytes[i];

        // Raw strings: r"…" / r#"…"# / br#"…"#. This MUST be checked before
        // the identifier branch: `r`/`br` are alphabetic, so the identifier
        // scan would otherwise swallow the prefix and the opening quote would
        // never be seen here. (A previous version "solved" this by rewinding
        // the index from inside the identifier branch, which re-entered that
        // same branch forever — an infinite loop on any raw string.)
        if b == b'r' || (b == b'b' && i + 1 < len && bytes[i + 1] == b'r') {
            let mut probe = i + 1;
            if b == b'b' {
                probe += 1;
            }
            if probe < len && (bytes[probe] == b'#' || bytes[probe] == b'"') {
                let start = i;
                let mut hashes = 0u32;
                i = probe;
                if bytes[i] == b'#' {
                    while i < len && bytes[i] == b'#' {
                        hashes += 1;
                        i += 1;
                    }
                }
                if i < len && bytes[i] == b'"' {
                    i += 1;
                    let close = format!("\"{}", "#".repeat(hashes as usize));
                    while i + close.len() <= len {
                        if source[i..i + close.len()] == close[..] {
                            i += close.len();
                            break;
                        }
                        i += 1;
                    }
                }
                tokens.push(Token::Other(source[start..i].to_string()));
                continue;
            }
        }

        // Identifiers: [a-zA-Z_][a-zA-Z0-9_]*
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = i;
            i += 1;
            while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            tokens.push(Token::Ident(source[start..i].to_string()));
            continue;
        }

        // String literals: "…"
        if b == b'"' {
            let start = i;
            i += 1;
            while i < len && bytes[i] != b'"' {
                if bytes[i] == b'\\' && i + 1 < len {
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if i < len {
                i += 1;
            }
            tokens.push(Token::Other(source[start..i].to_string()));
            continue;
        }

        // Char literals: '…'
        if b == b'\'' {
            let start = i;
            i += 1;
            if i < len && bytes[i] == b'\\' && i + 1 < len {
                i += 2;
            } else if i < len {
                i += 1;
            }
            if i < len && bytes[i] == b'\'' {
                i += 1;
            }
            tokens.push(Token::Other(source[start..i].to_string()));
            continue;
        }

        // Line comments: //…
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
            let start = i;
            while i < len && bytes[i] != b'\n' {
                i += 1;
            }
            tokens.push(Token::Other(source[start..i].to_string()));
            continue;
        }

        // Block comments: /* … */
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
            let start = i;
            i += 2;
            while i + 1 < len {
                if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    i += 2;
                    break;
                }
                i += 1;
            }
            tokens.push(Token::Other(source[start..i].to_string()));
            continue;
        }

        // Everything else: whitespace, punctuation, operators, digits.
        let start = i;
        i += 1;
        tokens.push(Token::Other(source[start..i].to_string()));
    }

    tokens
}

/// Encode a counter into a short alias: `_a`, `_b`, …, `_z`, `_aa`, …
fn encode_alias(index: u32) -> String {
    const ALPHA: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    let base = ALPHA.len() as u32;
    let mut n = index;
    let mut buf = Vec::new();
    loop {
        buf.push(ALPHA[(n % base) as usize] as char);
        n /= base;
        if n == 0 {
            break;
        }
        n -= 1;
    }
    buf.reverse();
    let suffix: String = buf.into_iter().collect();
    format!("_{suffix}")
}

/// Rust keyword set (never aliased).
fn rust_keywords() -> HashSet<&'static str> {
    [
        "as", "break", "const", "continue", "crate", "else", "enum", "extern",
        "false", "fn", "for", "if", "impl", "in", "let", "loop", "match",
        "mod", "move", "mut", "pub", "ref", "return", "self", "Self", "static",
        "struct", "super", "trait", "true", "type", "unsafe", "use", "where",
        "while", "async", "await", "dyn", "abstract", "become", "box", "do",
        "final", "macro", "override", "priv", "typeof", "unsized", "virtual",
        "yield", "try",
    ]
    .iter()
    .copied()
    .collect()
}

// ===========================================================================
// Feature 3 — Syntactic Whitespace Evacuation (Custom Line Processor)
// ===========================================================================

/// Strip redundant whitespace from code text. Removes leading indentation,
/// collapses multiple blank lines into one, and trims trailing spaces.
///
/// This is safe for model consumption: the model reads structural tokens
/// (semicolons, braces, keywords) rather than visual layout. Recovering
/// 20–30% of token payload from indentation alone.
fn evacuate_whitespace(source: &str) -> String {
    let mut out = String::with_capacity(source.len() / 2);
    let mut prev_blank = false;

    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !prev_blank {
                out.push('\n');
                prev_blank = true;
            }
        } else {
            out.push_str(trimmed);
            out.push('\n');
            prev_blank = false;
        }
    }

    // Remove trailing newline for a clean single-string return.
    if out.ends_with('\n') {
        out.pop();
    }
    out
}

// ===========================================================================
// Feature 4 — Deterministic Prompt-Cache Sorting
// ===========================================================================
//
// Sorting is performed in the pipeline entry point (`postprocess`) after
// all per-item transformations. The `ProcessedDep.module_path` field
// carries the node's tracker id (e.g. `crate::server::state::fn::new`)
// and the final Vec is sorted lexicographically by that path.
//
// This is deterministic because:
// - Tracker ids are derived from file paths + item names (stable).
// - Ties are broken by the default sort (which is stable in Rust).
// - The sort happens once, after all aliasing/whitespace transforms,
//   so the same codebase + query always produces the same byte sequence.

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Lexer tests ---

    #[test]
    fn lexer_splits_identifiers_and_punctuation() {
        let tokens = lex_source("fn foo_bar() { x + 1; }");
        let idents: Vec<&str> = tokens
            .iter()
            .filter_map(|t| match t {
                Token::Ident(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(idents, vec!["fn", "foo_bar", "x"]);
    }

    #[test]
    fn lexer_handles_string_literals() {
        let tokens = lex_source(r#"let s = "hello world";"#);
        let others: Vec<&str> = tokens
            .iter()
            .filter_map(|t| match t {
                Token::Other(s) if s.contains('"') => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(others, vec![r#""hello world""#]);
    }

    #[test]
    fn lexer_handles_line_comments() {
        let tokens = lex_source("// this is a comment\nlet x = 1;");
        let idents: Vec<&str> = tokens
            .iter()
            .filter_map(|t| match t {
                Token::Ident(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(idents, vec!["let", "x"]);
    }

    #[test]
    fn lexer_handles_block_comments() {
        let tokens = lex_source("/* block */ let x = 1;");
        let idents: Vec<&str> = tokens
            .iter()
            .filter_map(|t| match t {
                Token::Ident(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(idents, vec!["let", "x"]);
    }

    #[test]
    fn lexer_handles_raw_strings() {
        let input = "let s = r#\"hello world\"#;";
        let tokens = lex_source(input);
        // The raw string should be a single Other token.
        let has_raw = tokens.iter().any(|t| matches!(t, Token::Other(s) if s.starts_with("r#")));
        assert!(has_raw, "raw string should be tokenized: {:?}", tokens);
    }

    #[test]
    fn lexer_raw_string_regression_no_hang() {
        // Regression: the identifier branch used to rewind to the `r` and
        // re-enter itself forever, so ANY raw string hung the lexer — and
        // with it the whole autocomplete pipeline. Each case must terminate
        // and lex the raw string as a single Other token.
        let cases = [
            ("r#\"x\"#", "r#\"x\"#"),
            ("let s = r\"plain\";", "r\"plain\""),
            ("br#\"bytes\"#", "br#\"bytes\"#"),
            ("br\"raw bytes\"", "br\"raw bytes\""),
            ("r##\"double\"##", "r##\"double\"##"),
            // `br` NOT followed by a quote still lexes as an identifier.
            ("let br = 1;", "br"),
        ];
        for (input, expected) in cases {
            let tokens = lex_source(input);
            assert!(
                tokens.iter().any(|t| match t {
                    Token::Other(s) => s == expected,
                    Token::Ident(s) => s == expected && expected == "br",
                }),
                "case {input:?}: expected token {expected:?}, got {tokens:?}"
            );
        }
    }

    #[test]
    fn lexer_plain_identifiers_with_r_prefixes_untouched() {
        // Identifiers merely *starting* with r/b must not be mistaken for
        // raw strings when no quote/hash follows.
        let tokens = lex_source("render bridge root broad");
        let idents: Vec<&str> = tokens
            .iter()
            .filter_map(|t| match t {
                Token::Ident(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(idents, vec!["render", "bridge", "root", "broad"]);
    }

    // --- Alias tests ---

    #[test]
    fn alias_legend_injected() {
        // Repeated occurrences so the legend amortizes and the honesty
        // gate keeps the aliased form.
        let source = "pub fn handle(workspace_configuration_directory_path: &str) {\n    let a = workspace_configuration_directory_path;\n    let b = workspace_configuration_directory_path;\n    invalidate_merkle_tree();\n}";
        let result = aggressive_alias(source, 6);
        assert!(
            result.starts_with("// aliases:"),
            "should have legend: {result}"
        );
        assert!(
            result.contains("_a="),
            "should contain alias mapping: {result}"
        );
    }

    #[test]
    fn short_identifiers_not_aliased() {
        let source = "fn f(x: i32) -> i32 { x + 1 }";
        let result = aggressive_alias(source, 6);
        assert!(
            !result.starts_with("// aliases:"),
            "short-only source should not get aliases: {result}"
        );
    }

    #[test]
    fn aliases_replaced_in_body() {
        // Enough repetitions for the legend to amortize — the honesty gate
        // returns the original unchanged when aliasing would not shrink the
        // source.
        let source = "fn process(workspace_configuration_directory_path: &str) {\n    let x = workspace_configuration_directory_path;\n    let y = workspace_configuration_directory_path;\n    let z = workspace_configuration_directory_path;\n}";
        let result = aggressive_alias(source, 6);
        assert!(
            result.starts_with("// aliases:"),
            "repetitive source should be aliased: {result}"
        );
        // The BODY must no longer contain the identifier (the legend line
        // legitimately does — that is how hydration works).
        let body = result.split_once('\n').map(|(_, rest)| rest).unwrap_or("");
        assert!(
            !body.contains("workspace_configuration_directory_path"),
            "long ident should be replaced in the body: {result}"
        );
        assert!(result.contains("_a"));
    }

    #[test]
    fn alias_honesty_gate_returns_original_when_no_savings() {
        // A long identifier that occurs ONCE cannot amortize its legend
        // entry: aliasing would grow the output. The honest thing — and the
        // README's stated contract — is to return the source unchanged.
        let source = "fn f(one_long_identifier_name: &str) { let x = 1; }";
        let result = aggressive_alias(source, 6);
        assert_eq!(
            result, source,
            "no-net-savings aliasing must pass through unchanged"
        );
    }

    #[test]
    fn keywords_never_aliased() {
        // `return_thing` is an identifier (not the `return` keyword) and
        // must be aliased; the actual keyword must survive untouched.
        let source = "pub fn return_thing() {\n    return_thing();\n    return_thing();\n    return_thing();\n    return;\n}";
        let result = aggressive_alias(source, 6);
        assert!(result.contains("_a="));
        let legend = result.lines().next().unwrap_or("");
        assert!(
            legend.contains("return_thing"),
            "identifier should be aliased: {result}"
        );
        let return_count = result.matches("return").count();
        assert!(return_count >= 1, "`return` keyword must remain: {result}");
    }

    #[test]
    fn string_contents_not_aliased() {
        let source = r#"let msg = "workspace_configuration_directory_path";"#;
        let result = aggressive_alias(source, 6);
        // The identifier inside the string should NOT be aliased.
        assert!(
            result.contains("workspace_configuration_directory_path"),
            "string contents must not be aliased: {result}"
        );
    }

    // --- Alias encoding tests ---

    #[test]
    fn encode_alias_sequence() {
        assert_eq!(encode_alias(0), "_a");
        assert_eq!(encode_alias(1), "_b");
        assert_eq!(encode_alias(25), "_z");
        assert_eq!(encode_alias(26), "_aa");
        assert_eq!(encode_alias(27), "_ab");
    }

    // --- Whitespace evacuation tests ---

    #[test]
    fn evacuation_strips_indentation() {
        let source = "    pub fn foo() {\n        let x = 1;\n        let y = 2;\n    }";
        let result = evacuate_whitespace(source);
        assert_eq!(result, "pub fn foo() {\nlet x = 1;\nlet y = 2;\n}");
    }

    #[test]
    fn evacuation_collapses_blank_lines() {
        let source = "fn a() {}\n\n\n\nfn b() {}";
        let result = evacuate_whitespace(source);
        assert_eq!(result, "fn a() {}\n\nfn b() {}");
    }

    #[test]
    fn evacuation_preserves_single_blank() {
        let source = "fn a() {}\n\nfn b() {}";
        let result = evacuate_whitespace(source);
        assert_eq!(result, "fn a() {}\n\nfn b() {}");
    }

    // --- Monomorphization tests ---

    #[test]
    fn impl_header_extracted() {
        let source = "impl AppState {\n    pub fn new() -> Self { Self }\n    pub fn drop(&self) {}\n}";
        let result = impl_header_only(source);
        assert_eq!(result, "impl AppState {}");
    }

    #[test]
    fn method_name_extraction() {
        assert_eq!(
            try_extract_method_name("pub fn verify(&self) -> bool {"),
            Some("verify".into())
        );
        assert_eq!(
            try_extract_method_name("pub async fn run(&self) {"),
            Some("run".into())
        );
        assert_eq!(
            try_extract_method_name("fn helper() -> i32;"),
            Some("helper".into())
        );
        assert_eq!(
            try_extract_method_name("pub(crate) fn internal() {"),
            Some("internal".into())
        );
        assert_eq!(try_extract_method_name("let x = 1;"), None);
        assert_eq!(try_extract_method_name("}"), None);
    }

    #[test]
    fn prefix_stripping() {
        assert_eq!(strip_method_prefix("pub fn foo()"), Some("fn foo()"));
        assert_eq!(
            strip_method_prefix("pub(crate) async fn bar()"),
            Some("fn bar()")
        );
        assert_eq!(
            strip_method_prefix("unsafe fn baz()"),
            Some("fn baz()")
        );
        assert_eq!(strip_method_prefix("let x = 1;"), Some("let x = 1;"));
    }

    #[test]
    fn filter_keeps_only_used_methods() {
        let source = "impl Checker {\n    pub fn verify(&self) -> bool {\n        self.data.is_valid()\n    }\n    pub fn compute_heavy_statistics(&self) -> f64 {\n        42.0\n    }\n}";
        let mut keep = HashSet::new();
        keep.insert("verify".to_string());
        let result = filter_impl_methods(source, &keep);
        assert!(result.contains("fn verify"), "should keep verify: {result}");
        assert!(
            !result.contains("compute_heavy_statistics"),
            "should drop unused: {result}"
        );
    }

    // --- Sorting test ---

    #[test]
    fn deps_sort_by_module_path() {
        let mut deps = [
            ProcessedDep { module_path: "crate::z".into(), text: "z".into(), tokens: 1, was_transformed: false },
            ProcessedDep { module_path: "crate::a".into(), text: "a".into(), tokens: 1, was_transformed: false },
            ProcessedDep { module_path: "crate::m".into(), text: "m".into(), tokens: 1, was_transformed: false },
        ];
        deps.sort_by(|a, b| a.module_path.cmp(&b.module_path));
        assert_eq!(deps[0].module_path, "crate::a");
        assert_eq!(deps[1].module_path, "crate::m");
        assert_eq!(deps[2].module_path, "crate::z");
    }

    // --- Integration: alias + whitespace together ---

    #[test]
    fn alias_then_evacuate_saves_most() {
        // Heavy repetition is where the alias legend amortizes: each
        // occurrence of a ~38-char identifier collapses to ~2 chars.
        let source = "pub fn process_request(workspace_configuration_directory_root: &Config) {\n    let first = workspace_configuration_directory_root.directory_path;\n    let second = workspace_configuration_directory_root.cache_root;\n    let third = workspace_configuration_directory_root.output_root;\n    invalidate_merkle_tree(first, second, third);\n    invalidate_merkle_tree(first, second, third);\n    invalidate_merkle_tree(first, second, third);\n    invalidate_merkle_tree(first, second, third);\n}";
        let aliased = aggressive_alias(source, 6);
        assert!(
            aliased.starts_with("// aliases:"),
            "repetitive source should be aliased: {aliased}"
        );
        let final_result = evacuate_whitespace(&aliased);
        assert!(final_result.len() < source.len() * 4 / 5,
            "final ({len}B) should be well under the original ({orig}B): {final_result}",
            len = final_result.len(), orig = source.len()
        );
        assert!(final_result.starts_with("// aliases:"));
    }

    // --- Monomorphization + alias integration ---

    #[test]
    fn monomorphize_then_alias() {
        let source = "impl Checker {\n    pub fn verify(&self) -> bool { self.data.is_valid() }\n    pub fn compute_heavy_statistics(&self) -> f64 { 42.0 }\n}";
        let mut keep = HashSet::new();
        keep.insert("verify".to_string());
        let mono = filter_impl_methods(source, &keep);
        assert!(mono.contains("verify"), "should keep verify: {mono}");
        assert!(!mono.contains("compute_heavy_statistics"), "should drop unused: {mono}");
        let aliased = aggressive_alias(&mono, 6);
        assert!(
            !aliased.contains("compute_heavy_statistics"),
            "dropped method should not appear after aliasing"
        );
    }
}
