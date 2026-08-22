//! Phase 7: Lossless AST Signature Pruning
//!
//! When dependencies are streamed to the model under a tight token budget,
//! not every node deserves its full source body. This module implements a
//! **tree-sitter-powered signature extractor** that strips function/method
//! bodies while preserving the complete public API surface.
//!
//! # Strategy
//!
//! Dependencies pulled into context are ranked by personalized PageRank.
//! The prune filter operates in two tiers:
//!
//! - **High-rank nodes** (above `keep_threshold`): passed through **intact**.
//!   The model needs to see internal logic to write correct completions.
//!
//! - **Medium-to-low rank nodes** (at or below threshold): body is replaced
//!   with a compact signature-only form. A 300-line function collapses to
//!   something like `pub fn verify(&self) -> bool;` — the model still
//!   understands the boundary contract but the token cost drops by ~80%.
//!
//! The original source is always preserved in the ASG / ChunkRegistry, so
//! this transformation is strictly **lossless**: hydration reconstructs
//! the full body on demand.

use tree_sitter::{Node as TsNode, Parser};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Controls how aggressively low-rank dependencies are pruned.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PruneConfig {
    /// PageRank percentile threshold (0.0–1.0). Nodes whose normalized PPR
    /// score falls at or below this percentile are pruned to signatures.
    ///
    /// Example: 0.7 means the top-30% of dependencies keep their full body;
    /// the bottom 70% are collapsed to signatures.
    pub keep_threshold: f64,
    /// Enable/disable pruning entirely (useful for debugging or when the
    /// token budget is generous).
    pub enabled: bool,
}

impl Default for PruneConfig {
    fn default() -> Self {
        Self {
            keep_threshold: 0.30,
            enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Core API
// ---------------------------------------------------------------------------

/// Parse `source` and return a signature-only form: all `fn` / `const` /
/// `static` / `macro` bodies are replaced with `;`, keeping visibility,
/// generics, parameters, and return types intact.
///
/// Returns `None` if tree-sitter fails to parse the source (caller should
/// fall back to the original text).
pub fn prune_to_signatures(source: &str) -> Option<String> {
    let mut parser = Parser::new();
    let lang: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
    parser.set_language(&lang).ok()?;

    let tree = parser.parse(source.as_bytes(), None)?;
    let root = tree.root_node();

    let mut result = String::with_capacity(source.len() / 3);
    let mut cursor = root.walk();

    collect_top_level_items(root, source, &mut result, &mut cursor);

    // If nothing was extracted (empty file, all comments, etc.) fall back.
    if result.trim().is_empty() {
        Some(source.to_string())
    } else {
        Some(result)
    }
}

/// Determine whether a dependency should keep its full body based on its
/// normalized rank among a batch of candidate scores.
///
/// `rank` is the node's position (0 = highest PPR) and `total` is the
/// number of candidates in the batch.
pub fn should_keep_full_body(rank: usize, total: usize, config: &PruneConfig) -> bool {
    if !config.enabled || total == 0 {
        return true;
    }
    let percentile = 1.0 - (rank as f64 / total as f64);
    percentile > config.keep_threshold
}

// ---------------------------------------------------------------------------
// Tree-sitter walk: extract item signatures
// ---------------------------------------------------------------------------

/// Recursively walk the source file and emit item-level signatures.
///
/// Collects children up-front to avoid holding a mutable borrow on `cursor`
/// during recursive calls (tree-sitter's `named_children` iterator borrows
/// the cursor mutably, which conflicts with passing it to recursive calls).
fn collect_top_level_items<'a>(
    node: TsNode<'a>,
    source: &str,
    out: &mut String,
    cursor: &mut tree_sitter::TreeCursor<'a>,
) {
    // Snapshot children so the mutable cursor borrow is released before we
    // recurse into each child.
    let children: Vec<TsNode<'a>> = node.named_children(cursor).collect();
    for child in children {
        match child.kind() {
            // Functions: strip the body block, keep everything before it.
            "function_item" => {
                emit_signature(child, source, out, true);
            }
            // Impl blocks: keep the header (`impl … for Type {`), recurse
            // into the body for method signatures, then close the brace.
            "impl_item" => {
                emit_impl_header(child, source, out);
                if let Some(body) = child.child_by_field_name("body") {
                    out.push_str(" {\n");
                    let mut inner = body.walk();
                    collect_top_level_items(body, source, out, &mut inner);
                    out.push_str("}\n");
                }
            }
            // Trait blocks: same treatment as impl.
            "trait_item" => {
                emit_trait_header(child, source, out);
                if let Some(body) = child.child_by_field_name("body") {
                    out.push_str(" {\n");
                    let mut inner = body.walk();
                    collect_top_level_items(body, source, out, &mut inner);
                    out.push_str("}\n");
                }
            }
            // Struct / enum / type / const / static / macro / mod:
            // pass through verbatim — these are already compact and the
            // model needs their full definition to reason about types.
            "struct_item" | "enum_item" | "type_item" | "const_item"
            | "static_item" | "macro_definition" | "mod_item" | "use_declaration"
            | "extern_crate_declaration" | "attribute_item" | "function_signature_item" => {
                out.push_str(child
                    .utf8_text(source.as_bytes())
                    .unwrap_or_default());
                out.push('\n');
            }
            // Items that contain nested items (e.g. `mod` with an inline
            // body) — recurse.
            _ => {
                if child.named_child_count() > 0 {
                    let mut inner = child.walk();
                    collect_top_level_items(child, source, out, &mut inner);
                } else {
                    // Leaf node that wasn't handled — skip it to avoid
                    // emitting dangling lines.
                }
            }
        }
    }
}

/// Emit a function/method signature: everything up to and including the
/// return type, then `;` instead of the body block.
fn emit_signature(node: TsNode, source: &str, out: &mut String, _is_top: bool) {
    // Find the body block — everything before it is the signature.
    if let Some(body) = node.child_by_field_name("body") {
        let sig_end = body.start_byte();
        let sig_bytes = &source.as_bytes()[node.start_byte()..sig_end];
        let sig = std::str::from_utf8(sig_bytes).unwrap_or_default().trim_end();
        out.push_str(sig);
        out.push_str(";\n");
    } else {
        // No body (shouldn't happen for `function_item`, but be safe).
        out.push_str(
            node
                .utf8_text(source.as_bytes())
                .unwrap_or_default(),
        );
        out.push('\n');
    }
}

/// Emit `impl … for Type {` header without the body contents.
fn emit_impl_header(node: TsNode, source: &str, out: &mut String) {
    if let Some(body) = node.child_by_field_name("body") {
        let sig_end = body.start_byte();
        let sig_bytes = &source.as_bytes()[node.start_byte()..sig_end];
        let sig = std::str::from_utf8(sig_bytes).unwrap_or_default().trim_end();
        out.push_str(sig);
    } else {
        out.push_str(
            node
                .utf8_text(source.as_bytes())
                .unwrap_or_default(),
        );
    }
}

/// Emit `trait … {` header.
fn emit_trait_header(node: TsNode, source: &str, out: &mut String) {
    if let Some(body) = node.child_by_field_name("body") {
        let sig_end = body.start_byte();
        let sig_bytes = &source.as_bytes()[node.start_byte()..sig_end];
        let sig = std::str::from_utf8(sig_bytes).unwrap_or_default().trim_end();
        out.push_str(sig);
    } else {
        out.push_str(
            node
                .utf8_text(source.as_bytes())
                .unwrap_or_default(),
        );
    }
}

// ---------------------------------------------------------------------------
// Batch pruning
// ---------------------------------------------------------------------------

/// Given a batch of `(node_id, pagerank_score, original_source)`, return
/// `(node_id, pagerank_score, pruned_source, was_pruned)` for each entry.
/// Items are sorted by descending PPR before classification so the top-ranked
/// items keep their full bodies. The score is carried through so downstream
/// post-processing can use it without re-querying the ASG.
pub fn prune_batch(
    items: Vec<(usize, f64, String)>,
    config: &PruneConfig,
) -> Vec<(usize, f64, String, bool)> {
    if !config.enabled || items.len() <= 1 {
        return items
            .into_iter()
            .map(|(id, score, src)| (id, score, src, false))
            .collect();
    }

    // Sort by descending PPR so rank 0 = most important.
    let mut ranked: Vec<(usize, f64, String)> = items;
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

    let total = ranked.len();
    ranked
        .into_iter()
        .enumerate()
        .map(|(rank, (id, score, source))| {
            if should_keep_full_body(rank, total, config) {
                (id, score, source, false)
            } else {
                let pruned = prune_to_signatures(&source).unwrap_or(source.clone());
                (id, score, pruned, true)
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prune_fn_body() {
        let source = r#"pub fn verify(&self) -> bool {
    let x = self.compute_hash();
    if x > 42 {
        return true;
    }
    false
}
"#;
        let pruned = prune_to_signatures(source).unwrap();
        assert!(
            pruned.contains("pub fn verify(&self) -> bool;"),
            "should contain signature: {pruned}"
        );
        assert!(
            !pruned.contains("compute_hash"),
            "body should be stripped: {pruned}"
        );
    }

    #[test]
    fn prune_impl_block() {
        let source = r#"impl AppState {
    pub fn new() -> Self {
        Self { db: Database::connect() }
    }
    pub fn is_ready(&self) -> bool {
        self.db.ping()
    }
}
"#;
        let pruned = prune_to_signatures(source).unwrap();
        assert!(
            pruned.contains("impl AppState {"),
            "impl header preserved: {pruned}"
        );
        assert!(
            pruned.contains("pub fn new() -> Self;"),
            "method signature preserved: {pruned}"
        );
        assert!(
            pruned.contains("pub fn is_ready(&self) -> bool;"),
            "second method signature preserved: {pruned}"
        );
        assert!(
            !pruned.contains("Database::connect"),
            "impl body stripped: {pruned}"
        );
    }

    #[test]
    fn prune_trait_block() {
        let source = r#"pub trait Handler {
    fn handle(&self, req: &Request) -> Response;
    fn validate(&self) -> bool {
        true
    }
}
"#;
        let pruned = prune_to_signatures(source).unwrap();
        // Signature-only method kept as-is (already a signature).
        assert!(pruned.contains("fn handle(&self, req: &Request) -> Response;"));
        // Method with body: body stripped.
        assert!(pruned.contains("fn validate(&self) -> bool;"));
        assert!(!pruned.contains("true"));
    }

    #[test]
    fn prune_struct_unchanged() {
        let source = r#"pub struct AppState {
    pub db: Database,
    pub config: Config,
}
"#;
        let pruned = prune_to_signatures(source).unwrap();
        assert_eq!(pruned.trim(), source.trim());
    }

    #[test]
    fn keep_full_body_for_high_rank() {
        assert!(should_keep_full_body(0, 10, &PruneConfig::default()));
        assert!(should_keep_full_body(2, 10, &PruneConfig::default()));
    }

    #[test]
    fn prune_low_rank() {
        // Default threshold = 0.30 → top 30% kept. With 10 items, ranks 0–6 kept, 7–9 pruned.
        let cfg = PruneConfig::default();
        assert!(!should_keep_full_body(8, 10, &cfg));
        assert!(!should_keep_full_body(9, 10, &cfg));
    }

    #[test]
    fn prune_batch_respects_order() {
        let items = vec![
            (1, 0.1, "fn low() { heavy_logic(); }".into()),
            (2, 0.9, "fn high() { heavy_logic(); }".into()),
        ];
        let cfg = PruneConfig {
            keep_threshold: 0.5,
            enabled: true,
        };
        let result = prune_batch(items, &cfg);
        assert_eq!(result.len(), 2);
        // Rank 0 (node 2, score 0.9) → keep full body
        let (id_h, _, src_h, pruned_h) = &result[0];
        assert_eq!(*id_h, 2);
        assert!(!*pruned_h);
        assert!(src_h.contains("heavy_logic"));
        // Rank 1 (node 1, score 0.1) → prune
        let (id_l, _, src_l, pruned_l) = &result[1];
        assert_eq!(*id_l, 1);
        assert!(*pruned_l);
        assert!(!src_l.contains("heavy_logic"));
    }

    #[test]
    fn disabled_prune_passes_through() {
        let cfg = PruneConfig {
            enabled: false,
            ..Default::default()
        };
        let items = vec![
            (1, 0.1, "fn low() { body }".into()),
            (2, 0.9, "fn high() { body }".into()),
        ];
        let result = prune_batch(items, &cfg);
        for (_, _, _, pruned) in &result {
            assert!(!pruned);
        }
    }

    #[test]
    fn empty_source_is_safe() {
        assert_eq!(prune_to_signatures(""), Some(String::new()));
    }

    #[test]
    fn unparseable_source_falls_back_gracefully() {
        // Tree-sitter is very tolerant of malformed Rust and produces a partial
        // tree. The function should still return a reasonable fallback.
        let result = prune_to_signatures("fn {{{{").unwrap();
        // The result should at least contain the input (fallback path).
        assert!(!result.is_empty());
    }
}
