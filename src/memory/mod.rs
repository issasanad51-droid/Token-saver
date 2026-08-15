//! Memory store for arbitrary facts and design decisions alongside code.
//!
//! Memories are stored as nodes in the ASG (via MemoryNode) and can be
//! searched alongside code results through the existing hybrid search pipeline.

use std::sync::Arc;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

/// A stored memory (fact, decision, pattern, preference).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    /// Unique id (auto-generated).
    pub id: String,
    /// The memory content.
    pub content: String,
    /// Optional namespace/tag for scoping (e.g. "project:auth", "user:pref").
    pub namespace: Option<String>,
    /// Timestamp when the memory was created.
    pub created_at: u64,
    /// Timestamp of last access (for decay).
    pub last_accessed: u64,
    /// Access count (for frequency weighting).
    pub access_count: u64,
    /// Importance score (0.0-1.0, default 0.5).
    pub importance: f64,
}

/// Thread-safe memory store.
#[derive(Debug, Clone, Default)]
pub struct MemoryStore {
    /// id -> Memory
    memories: Arc<DashMap<String, Memory>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a new memory. Returns the generated id.
    pub fn save(&self, content: &str, namespace: Option<String>) -> String {
        let id = format!(
            "mem_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let memory = Memory {
            id: id.clone(),
            content: content.to_string(),
            namespace,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            importance: 0.5,
        };
        self.memories.insert(id.clone(), memory);
        id
    }

    /// Recall memories matching a query (simple keyword matching for now).
    pub fn recall(&self, query: &str, top_k: usize) -> Vec<Memory> {
        let query_lower = query.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();
        let mut scored: Vec<(f64, Memory)> = self
            .memories
            .iter()
            .filter_map(|entry| {
                let mem = entry.value();
                let content_lower = mem.content.to_lowercase();
                let matches: usize = query_terms
                    .iter()
                    .filter(|term| content_lower.contains(*term))
                    .count();
                if matches > 0 {
                    // Boost by importance and access frequency.
                    let score = (matches as f64)
                        * mem.importance
                        * (1.0 + (mem.access_count as f64).ln_1p() * 0.1);
                    Some((score, mem.clone()))
                } else {
                    None
                }
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.truncate(top_k);
        scored.into_iter().map(|(_, m)| m).collect()
    }

    /// Delete a memory by id.
    pub fn forget(&self, id: &str) -> bool {
        self.memories.remove(id).is_some()
    }

    /// List all memories, optionally filtered by namespace.
    pub fn list(&self, namespace: Option<&str>) -> Vec<Memory> {
        self.memories
            .iter()
            .filter(|entry| {
                namespace.map_or(true, |ns| entry.value().namespace.as_deref() == Some(ns))
            })
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get a memory by id and increment access count.
    pub fn get(&self, id: &str) -> Option<Memory> {
        if let Some(mut mem) = self.memories.get_mut(id) {
            mem.access_count += 1;
            mem.last_accessed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            Some(mem.clone())
        } else {
            None
        }
    }

    /// Number of stored memories.
    pub fn len(&self) -> usize {
        self.memories.len()
    }

    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }
}
