//! Real-time cursor-to-ASG mapping and token-budgeted context assembly.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::asg::SharedAsg;
use crate::ast::{PostprocessConfig, PruneConfig};
use crate::compressor::ChunkRegistry;
use crate::search::SearchEngine;

// ---------------------------------------------------------------------------
// Configuration and cursor payload
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextConfig {
    /// Raw source lines included on either side of the cursor.
    pub surrounding_lines: usize,
    /// Maximum number of ASG dependencies added to a prompt.
    pub max_dependencies: usize,
    /// Overall approximate prompt budget (one token ~= four UTF-8 bytes).
    pub max_context_tokens: usize,
    /// Independent cap for compressed dependency chunks.
    pub dependency_token_budget: usize,
    /// Controls AST-based signature pruning for low-rank dependencies.
    pub prune: PruneConfig,
    /// Controls query-time post-processing (aliasing, whitespace, mono, sort).
    pub postprocess: PostprocessConfig,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            surrounding_lines: 16,
            max_dependencies: 5,
            max_context_tokens: 4_000,
            dependency_token_budget: 2_400,
            prune: PruneConfig::default(),
            postprocess: PostprocessConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorPayload {
    pub file_path: String,
    pub line: usize,
    pub column: usize,
}

impl CursorPayload {
    pub fn new(file_path: impl Into<String>, line: usize, column: usize) -> Self {
        Self {
            file_path: file_path.into(),
            line,
            column,
        }
    }

    /// Convert a zero-indexed Unicode-scalar line/column to a UTF-8 byte
    /// offset. Out-of-range positions clamp to the selected line or file end.
    pub fn to_byte_offset(&self, source: &str) -> usize {
        let mut line_start = 0usize;
        let mut selected = None;
        for (line, segment) in source.split_inclusive('\n').enumerate() {
            if line == self.line {
                selected = Some((line_start, segment.trim_end_matches('\n')));
                break;
            }
            line_start += segment.len();
        }

        let Some((start, line_text)) = selected else {
            return source.len();
        };
        let column_offset = line_text
            .char_indices()
            .nth(self.column)
            .map(|(offset, _)| offset)
            .unwrap_or(line_text.len());
        start + column_offset
    }
}

// ---------------------------------------------------------------------------
// Context tracker
// ---------------------------------------------------------------------------

pub struct ContextTracker {
    asg: SharedAsg,
    registry: Arc<ChunkRegistry>,
    search_engine: Arc<SearchEngine>,
    workspace_root: PathBuf,
    config: ContextConfig,
}

impl ContextTracker {
    /// Backwards-compatible constructor rooted at the current workspace.
    pub fn new(asg: SharedAsg, registry: ChunkRegistry, search_engine: SearchEngine) -> Self {
        let workspace = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self::with_config(
            asg,
            registry,
            search_engine,
            workspace,
            ContextConfig::default(),
        )
    }

    pub fn with_config(
        asg: SharedAsg,
        registry: ChunkRegistry,
        search_engine: SearchEngine,
        workspace_root: PathBuf,
        config: ContextConfig,
    ) -> Self {
        let workspace_root = workspace_root
            .canonicalize()
            .unwrap_or(workspace_root);
        Self {
            asg,
            registry: Arc::new(registry),
            search_engine: Arc::new(search_engine),
            workspace_root,
            config,
        }
    }

    pub fn config(&self) -> &ContextConfig {
        &self.config
    }

    /// Find the smallest semantic ASG entity containing the cursor.
    pub fn find_node_at_cursor(&self, payload: &CursorPayload) -> Option<usize> {
        let file_path = self.resolve_path(&payload.file_path)?;
        let node_ids = self.asg.inner.file_index.get(&file_path)?;
        let source = std::fs::read_to_string(&file_path).ok()?;
        let byte_offset = payload.to_byte_offset(&source);

        node_ids
            .iter()
            .filter_map(|node_id| self.asg.get_node(*node_id))
            .filter(|node| byte_offset >= node.range.0 && byte_offset <= node.range.1)
            .min_by_key(|node| node.range.1.saturating_sub(node.range.0))
            .map(|node| node.id)
    }

    /// Return raw local context while ensuring request paths cannot escape the
    /// configured workspace.
    pub fn get_surrounding_lines(&self, payload: &CursorPayload) -> String {
        let Some(file_path) = self.resolve_path(&payload.file_path) else {
            return String::new();
        };
        let Ok(source) = std::fs::read_to_string(file_path) else {
            return String::new();
        };
        let lines: Vec<&str> = source.lines().collect();
        if lines.is_empty() {
            return String::new();
        }

        let cursor_line = payload.line.min(lines.len() - 1);
        let radius = self.config.surrounding_lines;
        let start = cursor_line.saturating_sub(radius);
        let end = (cursor_line + radius + 1).min(lines.len());
        lines[start..end].join("\n")
    }

    /// Retrieve dependencies with query/cursor-personalized PPR and pack them
    /// greedily under the configured token budget.
    ///
    /// The pipeline has four phases:
    /// 1. **Search** — retrieve candidate dependencies via PPR-ranked fusion.
    /// 2. **AST Prune** — low-rank nodes get collapsed to signatures only.
    /// 3. **Post-process** — alias long identifiers, evacuate whitespace,
    ///    monomorphize unused impl methods, and sort deterministically.
    /// 4. **Pack** — greedily fill the token budget.
    pub async fn get_compressed_dependencies(
        &self,
        payload: &CursorPayload,
        query: &str,
    ) -> Vec<String> {
        let cursor_node = self.find_node_at_cursor(payload);
        let candidate_count = self.config.max_dependencies.saturating_mul(4).max(10);
        let results = self
            .search_engine
            .search_with_context(query, candidate_count, cursor_node)
            .await;

        // Phase 1: Collect candidate chunks with their fused relevance
        // scores (RRF). The query-relative rank drives every downstream
        // decision — pruning tiers and the top-N selection — because global
        // PageRank says nothing about what THIS query needs.
        let mut candidates: Vec<(usize, f64, String)> = Vec::new();
        for result in &results {
            if Some(result.node_id) == cursor_node {
                continue;
            }
            let Some(chunk) = self.registry.get(result.node_id) else {
                continue;
            };
            let prompt_text = chunk.prompt_text();
            candidates.push((result.node_id, result.rrf_score, prompt_text));
        }

        // Phase 2: Relevance-based selection. The top-N cut MUST happen
        // here, BEFORE the alphabetical prompt-cache sort in Phase 4 —
        // after that sort, taking the first N would silently select the
        // alphabetically-first dependencies instead of the most relevant
        // ones (which is how `multiply` got dropped for `divide`).
        let mut selected = candidates;
        selected.sort_by(|a, b| b.1.total_cmp(&a.1));
        selected.truncate(self.config.max_dependencies);

        // Phase 3: AST signature pruning — within the DELIVERED set, only
        // the top-relevance dependencies keep full bodies; the rest are
        // collapsed to declarations only, saving ~80% of their tokens.
        let pruned = crate::ast::prune_batch(selected, &self.config.prune);

        // Phase 4: Post-processing pipeline — alias, whitespace-evacuate,
        // monomorphize, and deterministically sort (stable order for
        // prompt-cache reuse; applies to the already-selected set).
        let processed = crate::ast::postprocess(
            pruned,
            &self.config.postprocess,
            &self.asg,
            cursor_node,
        );

        // Phase 4: Deduplicate identical bodies, then pack greedily under
        // the token budget (see [`pack_dependencies`]).
        pack_dependencies(
            processed,
            self.config.dependency_token_budget,
            self.config.max_dependencies,
        )
    }

    /// Assemble a complete prompt and enforce the overall context cap.
    pub async fn assemble_prompt(&self, payload: &CursorPayload, query: &str) -> String {
        let surrounding = self.get_surrounding_lines(payload);
        let mut prompt = surrounding;
        let mut used_tokens = estimate_tokens(&prompt);

        let dependencies = self.get_compressed_dependencies(payload, query).await;
        for (index, dependency) in dependencies.into_iter().enumerate() {
            // Minimal delimiter: verbose section headers burn ~6 tokens per
            // dependency while carrying almost no signal for the model.
            let section = format!("\n\n// dep {}\n{}", index + 1, dependency);
            let section_tokens = estimate_tokens(&section);
            if used_tokens + section_tokens > self.config.max_context_tokens {
                break;
            }
            prompt.push_str(&section);
            used_tokens += section_tokens;
        }
        prompt
    }

    pub fn get_node_name_at_cursor(&self, payload: &CursorPayload) -> Option<String> {
        let node_id = self.find_node_at_cursor(payload)?;
        self.asg.get_node(node_id).map(|node| node.name.clone())
    }

    pub fn get_cursor_node_pagerank(&self, payload: &CursorPayload) -> Option<f64> {
        let node_id = self.find_node_at_cursor(payload)?;
        self.asg.get_node(node_id).map(|node| node.pagerank)
    }

    fn resolve_path(&self, requested: &str) -> Option<PathBuf> {
        let requested = Path::new(requested);
        let candidate = if requested.is_absolute() {
            requested.to_path_buf()
        } else {
            self.workspace_root.join(requested)
        };
        let canonical = candidate.canonicalize().ok()?;
        canonical.starts_with(&self.workspace_root).then_some(canonical)
    }
}

/// Estimate the number of LLM tokens in a text string.
///
/// Code tokenizes differently from natural language: punctuation-heavy syntax
/// (e.g. `let x: Vec<Arc<Mutex<T>>> = ...`) produces many more tokens per
/// byte than prose. We use a conservative 3.2 bytes/token ratio for code
/// (vs. ~4 for natural language) and count special characters that tend to
/// each consume a full token (brackets, colons, arrows, semicolons).
pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    let base = (text.len() as f64 / 3.2).ceil() as usize;
    // Count syntax-heavy characters that typically become standalone tokens.
    let standalone: usize = text
        .chars()
        .filter(|c| matches!(c, '{' | '}' | '(' | ')' | '[' | ']' | ':' | ';' | ',' | '=' | '<' | '>' | '&' | '|' | '#' | '@'))
        .count();
    // Each standalone char adds ~0.3 tokens on top of the base estimate
    // (they're partially covered by the byte ratio but undercounted).
    base.max(standalone / 3).max(1)
}

// ---------------------------------------------------------------------------
// Serializable context snapshot
// ---------------------------------------------------------------------------

/// Pack processed dependencies under the token budget, **skipping identical
/// bodies**.
///
/// Distinct ASG nodes can carry byte-identical source (boilerplate builders,
/// mirrored impls, generated code). Ranking treats them as separate
/// candidates, but emitting the same text twice only burns tokens — the
/// model gains nothing from the second copy. The first occurrence in the
/// deterministic packing order wins; later duplicates are dropped before
/// the budget arithmetic so the freed budget can carry a *new* dependency.
pub(crate) fn pack_dependencies(
    processed: Vec<crate::ast::postprocess::ProcessedDep>,
    token_budget: usize,
    max_dependencies: usize,
) -> Vec<String> {
    let mut seen_bodies: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut compressed = Vec::new();
    let mut used_tokens = 0usize;
    for dep in processed {
        if seen_bodies.contains(&dep.text) {
            continue;
        }
        seen_bodies.insert(dep.text.clone());
        if used_tokens + dep.tokens > token_budget {
            continue;
        }
        used_tokens += dep.tokens;
        compressed.push(dep.text);
        if compressed.len() >= max_dependencies {
            break;
        }
    }
    compressed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSnapshot {
    pub cursor: CursorPayload,
    pub node_id: Option<usize>,
    pub node_name: Option<String>,
    pub pagerank: Option<f64>,
    pub surrounding_lines: String,
    pub compressed_deps: Vec<String>,
}

impl ContextSnapshot {
    pub fn new(
        cursor: CursorPayload,
        node_id: Option<usize>,
        node_name: Option<String>,
        pagerank: Option<f64>,
        surrounding_lines: String,
        compressed_deps: Vec<String>,
    ) -> Self {
        Self {
            cursor,
            node_id,
            node_name,
            pagerank,
            surrounding_lines,
            compressed_deps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::postprocess::ProcessedDep;

    #[test]
    fn pack_dependencies_dedups_identical_bodies() {
        // Distinct nodes with byte-identical bodies (mirrored builders,
        // generated boilerplate) must be emitted once — the second copy
        // costs tokens and adds zero information for the model.
        let deps = vec![
            ProcessedDep {
                module_path: "crate::a::fn::with_crate_root".to_string(),
                text: "pub fn with_crate_root(mut self, root: impl Into<PathBuf>) -> Self {".to_string(),
                tokens: 12,
                was_transformed: false,
            },
            ProcessedDep {
                module_path: "crate::b::fn::with_crate_root".to_string(),
                text: "pub fn with_crate_root(mut self, root: impl Into<PathBuf>) -> Self {".to_string(),
                tokens: 12,
                was_transformed: false,
            },
            ProcessedDep {
                module_path: "crate::c::fn::unique".to_string(),
                text: "pub fn unique() -> u32 { 7 }".to_string(),
                tokens: 8,
                was_transformed: false,
            },
        ];
        let packed = pack_dependencies(deps, 100, 10);
        assert_eq!(packed.len(), 2, "identical body must be packed once");
        assert_eq!(packed[0], "pub fn with_crate_root(mut self, root: impl Into<PathBuf>) -> Self {");
        assert_eq!(packed[1], "pub fn unique() -> u32 { 7 }");
    }

    #[test]
    fn pack_dependencies_respects_budget_and_cap() {
        let dep = |path: &str, tokens: usize| ProcessedDep {
            module_path: path.to_string(),
            text: format!("// {path}"),
            tokens,
            was_transformed: false,
        };
        // Budget allows one 10-token dep, not two.
        let packed = pack_dependencies(vec![dep("a", 10), dep("b", 10)], 10, 10);
        assert_eq!(packed.len(), 1);
        // Max-dependency cap applies after dedup.
        let packed = pack_dependencies(
            (0..8).map(|i| dep(&format!("n{i}"), 1)).collect(),
            100,
            3,
        );
        assert_eq!(packed.len(), 3);
    }

    #[test]
    fn unicode_cursor_position_maps_to_utf8_offset() {
        let source = "one\nαβgamma\nthree";
        let cursor = CursorPayload::new("file.rs", 1, 2);
        assert_eq!(&source[cursor.to_byte_offset(source)..], "gamma\nthree");
    }

    #[test]
    fn token_estimate_rounds_up() {
        assert_eq!(estimate_tokens("12345"), 2);
    }
}
