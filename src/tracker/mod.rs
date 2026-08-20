//! Real-time cursor-to-ASG mapping and token-budgeted context assembly.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::asg::SharedAsg;
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
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            surrounding_lines: 16,
            max_dependencies: 5,
            max_context_tokens: 4_000,
            dependency_token_budget: 2_400,
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
    ///
    /// Both `\n` and `\r\n` line terminators are handled so the offset is
    /// correct on Windows-edited files (the previous implementation only
    /// stripped `\n`, leaving a stray `\r` that shifted the column).
    pub fn to_byte_offset(&self, source: &str) -> usize {
        let mut line_start = 0usize;
        let mut selected = None;
        for (line, segment) in source.split_inclusive('\n').enumerate() {
            if line == self.line {
                selected = Some((line_start, segment.trim_end_matches(['\r', '\n'])));
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
        let workspace_root = workspace_root.canonicalize().unwrap_or(workspace_root);
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
    ///
    /// The byte range on each [`Node`] is documented as `[start, end)` (a
    /// half-open range). We honour that contract here: a cursor exactly at
    /// `end` is considered *outside* the node. The previous implementation
    /// used `<=`, which mis-identified the byte immediately after a node as
    /// still being inside it and could mask a tighter child node sitting
    /// at the same boundary.
    pub fn find_node_at_cursor(&self, payload: &CursorPayload) -> Option<usize> {
        let file_path = self.resolve_path(&payload.file_path)?;
        let node_ids = self.asg.inner.file_index.get(&file_path)?;
        let source = std::fs::read_to_string(&file_path).ok()?;
        let byte_offset = payload.to_byte_offset(&source);

        node_ids
            .iter()
            .filter_map(|node_id| self.asg.get_node(*node_id))
            .filter(|node| byte_offset >= node.range.0 && byte_offset < node.range.1)
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

        let mut compressed = Vec::new();
        let mut used_tokens = 0usize;
        for result in results {
            if Some(result.node_id) == cursor_node {
                continue;
            }
            let Some(chunk) = self.registry.get(result.node_id) else {
                continue;
            };
            let prompt_text = chunk.prompt_text();
            let tokens = estimate_tokens(&prompt_text);
            if used_tokens + tokens > self.config.dependency_token_budget {
                continue;
            }
            used_tokens += tokens;
            compressed.push(prompt_text);
            if compressed.len() >= self.config.max_dependencies {
                break;
            }
        }
        compressed
    }

    /// Assemble a complete prompt and enforce the overall context cap.
    pub async fn assemble_prompt(&self, payload: &CursorPayload, query: &str) -> String {
        let surrounding = self.get_surrounding_lines(payload);
        let mut prompt = surrounding;
        let mut used_tokens = estimate_tokens(&prompt);

        let dependencies = self.get_compressed_dependencies(payload, query).await;
        for (index, dependency) in dependencies.into_iter().enumerate() {
            let section = format!(
                "\n\n--- COMPRESSED DEPENDENCY {} ---\n{}",
                index + 1,
                dependency
            );
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
        canonical
            .starts_with(&self.workspace_root)
            .then_some(canonical)
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
        .filter(|c| {
            matches!(
                c,
                '{' | '}'
                    | '('
                    | ')'
                    | '['
                    | ']'
                    | ':'
                    | ';'
                    | ','
                    | '='
                    | '<'
                    | '>'
                    | '&'
                    | '|'
                    | '#'
                    | '@'
            )
        })
        .count();
    // Each standalone char adds ~0.3 tokens on top of the base estimate
    // (they're partially covered by the byte ratio but undercounted).
    base.max(standalone / 3).max(1)
}

// ---------------------------------------------------------------------------
// Serializable context snapshot
// ---------------------------------------------------------------------------

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

    #[test]
    fn unicode_cursor_position_maps_to_utf8_offset() {
        let source = "one\nαβgamma\nthree";
        let cursor = CursorPayload::new("file.rs", 1, 2);
        assert_eq!(&source[cursor.to_byte_offset(source)..], "gamma\nthree");
    }

    /// Regression test: on Windows-edited files (CRLF line endings), the
    /// previous implementation only stripped `\n` from the line segment,
    /// leaving a stray `\r` that shifted the column one byte to the right.
    #[test]
    fn crlf_line_endings_do_not_shift_cursor_column() {
        let source = "alpha\r\nsecond_line\r\nthird";
        let cursor = CursorPayload::new("file.rs", 1, 4); // points at 'n' of 'second_line'
        let offset = cursor.to_byte_offset(source);
        // The byte at the offset should be the 'n' of "second", not the 'd'.
        assert_eq!(&source[offset..offset + 1], "n");
    }

    /// Smoke test for empty sources — must not panic.
    #[test]
    fn empty_source_returns_zero_offset() {
        let cursor = CursorPayload::new("file.rs", 0, 0);
        assert_eq!(cursor.to_byte_offset(""), 0);
    }

    #[test]
    fn token_estimate_rounds_up() {
        assert_eq!(estimate_tokens("12345"), 2);
    }
}
