//! Phase 4: Real-Time Copilot Ghost-Text Context Tracker
//!
//! Tracks editor cursor locations, maps them to ASG node IDs, and assembles
//! a hybrid prompt: 20 raw lines above/below the cursor + top 3 RRF/PageRank
//! adjacent dependency nodes in compressed token-saver notation.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::asg::{Asg, SharedAsg};
use crate::compressor::ChunkRegistry;
use crate::search::{MergedResult, SearchEngine};

// ---------------------------------------------------------------------------
// Cursor Payload
// ---------------------------------------------------------------------------

/// Represents the user's cursor location in the editor.
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

    /// Convert line/column to byte offset in the source.
    pub fn to_byte_offset(&self, source: &str) -> usize {
        let mut offset = 0;
        let mut current_line = 0;

        for (i, ch) in source.char_indices() {
            if current_line == self.line {
                return offset + self.column;
            }
            if ch == '\n' {
                current_line += 1;
            }
            offset = i + ch.len_utf8();
        }

        offset
    }
}

// ---------------------------------------------------------------------------
// Context Tracker
// ---------------------------------------------------------------------------

/// Tracks cursor position and assembles hybrid completion context.
pub struct ContextTracker {
    asg: SharedAsg,
    registry: Arc<ChunkRegistry>,
    search_engine: Arc<SearchEngine>,
}

impl ContextTracker {
    pub fn new(asg: SharedAsg, registry: ChunkRegistry, search_engine: SearchEngine) -> Self {
        Self {
            asg,
            registry: Arc::new(registry),
            search_engine: Arc::new(search_engine),
        }
    }

    /// Find the ASG node ID that contains the cursor position.
    pub fn find_node_at_cursor(&self, payload: &CursorPayload) -> Option<usize> {
        let file_path = PathBuf::from(&payload.file_path);

        // Get all nodes in this file.
        let node_ids = self.asg.inner.file_index.get(&file_path)?;

        // Find the smallest node that contains the cursor byte offset.
        let source = std::fs::read_to_string(&file_path).ok()?;
        let byte_offset = payload.to_byte_offset(&source);

        let mut best_node: Option<(usize, usize)> = None; // (node_id, range_size)

        for node_id in node_ids {
            if let Some(node) = self.asg.get_node(*node_id) {
                let (start, end) = node.range;
                if byte_offset >= start && byte_offset <= end {
                    let range_size = end - start;
                    if best_node.is_none() || range_size < best_node.unwrap().1 {
                        best_node = Some((*node_id, range_size));
                    }
                }
            }
        }

        best_node.map(|(id, _)| id)
    }

    /// Get the 20 lines above and below the cursor (raw, uncompressed).
    pub fn get_surrounding_lines(&self, payload: &CursorPayload) -> String {
        let source = match std::fs::read_to_string(&payload.file_path) {
            Ok(s) => s,
            Err(_) => return String::new(),
        };

        let lines: Vec<&str> = source.lines().collect();
        let cursor_line = payload.line;

        let start = cursor_line.saturating_sub(20);
        let end = (cursor_line + 20).min(lines.len());

        lines[start..end].join("\n")
    }

    /// Get the top 3 RRF/PageRank adjacent dependency nodes in compressed form.
    pub async fn get_compressed_dependencies(
        &self,
        payload: &CursorPayload,
        query: &str,
    ) -> Vec<String> {
        // Run search to get top results.
        let results = self.search_engine.search(query, 10).await;

        // Take top 3 and compress them.
        let mut compressed: Vec<String> = Vec::new();
        for result in results.iter().take(3) {
            if let Some(chunk) = self.registry.get(result.node_id) {
                compressed.push(chunk.compressed_source.clone());
            }
        }

        compressed
    }

    /// Assemble the full hybrid prompt.
    pub async fn assemble_prompt(&self, payload: &CursorPayload, query: &str) -> String {
        let mut prompt = String::new();

        // 1. Raw surrounding context (20 lines above/below).
        let surrounding = self.get_surrounding_lines(payload);
        prompt.push_str(&surrounding);
        prompt.push_str("\n\n--- COMPRESSED DEPENDENCIES ---\n");

        // 2. Top 3 compressed dependency nodes.
        let deps = self.get_compressed_dependencies(payload, query).await;
        for (i, dep) in deps.iter().enumerate() {
            prompt.push_str(&format!("[DEP {}]\n{}\n", i + 1, dep));
        }

        prompt
    }

    /// Get the node name at the cursor position.
    pub fn get_node_name_at_cursor(&self, payload: &CursorPayload) -> Option<String> {
        let node_id = self.find_node_at_cursor(payload)?;
        self.asg.get_node(node_id).map(|n| n.name.clone())
    }

    /// Get PageRank score of the node at cursor.
    pub fn get_cursor_node_pagerank(&self, payload: &CursorPayload) -> Option<f64> {
        let node_id = self.find_node_at_cursor(payload)?;
        self.asg.get_node(node_id).map(|n| n.pagerank)
    }
}

// ---------------------------------------------------------------------------
// Context Snapshot
// ---------------------------------------------------------------------------

/// A complete snapshot of the editing context at a point in time.
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
