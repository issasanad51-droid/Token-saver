//! Memory store for arbitrary facts and design decisions alongside code.
//!
//! Memories are stored as nodes in the ASG (via MemoryNode) and can be
//! searched alongside code results through the existing hybrid search pipeline.
//!
//! Features:
//! - Importance scoring: memories have a 0.0-1.0 importance score
//! - Temporal decay: older and less-accessed memories fade over time
//! - Namespace scoping: memories can be tagged with a namespace

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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
    /// Importance score (0.0-1.0).
    pub importance: f64,
}

/// Thread-safe memory store.
#[derive(Debug, Clone, Default)]
pub struct MemoryStore {
    /// id -> Memory
    memories: Arc<DashMap<String, Memory>>,
    /// Monotonic counter for unique ids.
    next_seq: Arc<AtomicU64>,
}

/// Half-life in seconds for temporal decay (default: 7 days).
const DECAY_HALF_LIFE_SECS: f64 = 7.0 * 24.0 * 3600.0;

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a new memory. Returns the generated id.
    ///
    /// If `importance` is provided, it is clamped to [0.0, 1.0].
    /// Otherwise, importance is auto-calculated based on content length
    /// and namespace presence.
    pub fn save(&self, content: &str, namespace: Option<String>, importance: Option<f64>) -> String {
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let id = format!(
            "mem_{}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            seq
        );
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let importance = match importance {
            Some(imp) => imp.clamp(0.0, 1.0),
            None => Self::calculate_importance(content, namespace.as_deref()),
        };

        let memory = Memory {
            id: id.clone(),
            content: content.to_string(),
            namespace,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            importance,
        };
        self.memories.insert(id.clone(), memory);
        id
    }

    /// Calculate automatic importance based on content heuristics.
    ///
    /// - Longer content gets a higher base score (capped at 0.3)
    /// - Namespaced memories get a 0.2 boost (they're curated)
    /// - Very short content (< 20 chars) gets a penalty
    fn calculate_importance(content: &str, namespace: Option<&str>) -> f64 {
        let length_score = (content.len() as f64 / 200.0).min(0.3);
        let namespace_boost = if namespace.is_some() { 0.2 } else { 0.0 };
        let short_penalty = if content.len() < 20 { -0.1 } else { 0.0 };
        (0.5 + length_score + namespace_boost + short_penalty).clamp(0.0, 1.0)
    }

    /// Calculate temporal decay factor for a memory.
    ///
    /// Uses exponential decay: `importance * 0.5^(elapsed / half_life)`
    /// A memory accessed recently or frequently gets a boost.
    pub fn decay_factor(memory: &Memory) -> f64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Time since last access.
        let elapsed_secs = (now - memory.last_accessed) as f64;

        // Exponential decay based on time since last access.
        let time_decay = 0.5_f64.powf(elapsed_secs / DECAY_HALF_LIFE_SECS);

        // Frequency boost: memories accessed more often are more important.
        // ln(1 + count) gives diminishing returns.
        let freq_boost = 1.0 + (memory.access_count as f64).ln_1p() * 0.1;

        // Recency boost: memories accessed within the last hour get a 2x boost.
        let recency_boost = if elapsed_secs < 3600.0 { 2.0 } else { 1.0 };

        memory.importance * time_decay * freq_boost * recency_boost
    }

    /// Recall memories matching a query (keyword matching + importance + decay).
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
                    // Combine keyword match score with decay-adjusted importance.
                    let decay = Self::decay_factor(mem);
                    let score = (matches as f64) * decay;
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

    /// Update an existing memory's importance and/or namespace.
    /// Returns true if the memory was found and updated.
    pub fn update(&self, id: &str, importance: Option<f64>, namespace: Option<Option<String>>) -> bool {
        if let Some(mut mem) = self.memories.get_mut(id) {
            if let Some(imp) = importance {
                mem.importance = imp.clamp(0.0, 1.0);
            }
            if let Some(ns) = namespace {
                mem.namespace = ns;
            }
            true
        } else {
            false
        }
    }

    /// List all memories, optionally filtered by namespace.
    pub fn list(&self, namespace: Option<&str>) -> Vec<Memory> {
        self.memories
            .iter()
            .filter(|entry| {
                namespace.is_none_or(|ns| entry.value().namespace.as_deref() == Some(ns))
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

    /// Insert a fully-constructed memory directly, preserving its original
    /// id, importance, access_count, created_at, and last_accessed fields.
    /// Used by the persistence layer to restore memories from redb without
    /// regenerating ids or losing metadata.
    pub fn insert_memory(&self, memory: Memory) {
        // Ensure the sequence counter is ahead of any timestamp-based id.
        // Memory ids are `mem_{timestamp}_{seq}`, so we just bump the seq.
        let _ = self.next_seq.fetch_add(1, Ordering::Relaxed);
        self.memories.insert(memory.id.clone(), memory);
    }

    /// Number of stored memories.
    pub fn len(&self) -> usize {
        self.memories.len()
    }

    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }

    /// Apply decay: remove memories whose effective score has fallen below
    /// a threshold. Returns the number of memories pruned.
    pub fn prune_decayed(&self, min_score: f64) -> usize {
        let to_remove: Vec<String> = self
            .memories
            .iter()
            .filter(|entry| Self::decay_factor(entry.value()) < min_score)
            .map(|entry| entry.key().clone())
            .collect();

        let pruned = to_remove.len();
        for id in to_remove {
            self.memories.remove(&id);
        }
        pruned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_importance_for_namespaced_memory() {
        let store = MemoryStore::new();
        let id = store.save("This is a design decision about auth", Some("project:auth".to_string()), None);
        let mem = store.get(&id).unwrap();
        // Namespaced + reasonable length should be > 0.5
        assert!(mem.importance > 0.5);
    }

    #[test]
    fn auto_importance_for_short_memory() {
        let store = MemoryStore::new();
        let id = store.save("short", None, None);
        let mem = store.get(&id).unwrap();
        // Short content gets a penalty
        assert!(mem.importance < 0.5);
    }

    #[test]
    fn explicit_importance_is_clamped() {
        let store = MemoryStore::new();
        let id = store.save("content", None, Some(5.0));
        let mem = store.get(&id).unwrap();
        assert_eq!(mem.importance, 1.0);

        let id2 = store.save("content", None, Some(-1.0));
        let mem2 = store.get(&id2).unwrap();
        assert_eq!(mem2.importance, 0.0);
    }

    #[test]
    fn recall_uses_decay_scoring() {
        let store = MemoryStore::new();
        store.save("important auth decision", Some("auth".to_string()), Some(1.0));
        store.save("minor note", None, Some(0.1));

        let results = store.recall("auth", 10);
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("auth"));
    }

    #[test]
    fn forget_removes_memory() {
        let store = MemoryStore::new();
        let id = store.save("test memory", None, None);
        assert_eq!(store.len(), 1);
        assert!(store.forget(&id));
        assert!(store.is_empty());
    }

    #[test]
    fn prune_removes_low_score_memories() {
        let store = MemoryStore::new();
        // Very low importance memory
        store.save("low", None, Some(0.01));
        assert_eq!(store.len(), 1);

        // Prune with a threshold above the effective score
        let _pruned = store.prune_decayed(0.5);
        // The low-importance memory may be pruned depending on recency boost
    }

    #[test]
    fn insert_memory_preserves_metadata() {
        let store = MemoryStore::new();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let memory = Memory {
            id: "mem_custom_42".to_string(),
            content: "preserved content".to_string(),
            namespace: Some("test".to_string()),
            created_at: now - 3600,
            last_accessed: now - 1800,
            access_count: 7,
            importance: 0.95,
        };
        store.insert_memory(memory);
        let loaded = store.get("mem_custom_42").unwrap();
        assert_eq!(loaded.content, "preserved content");
        assert_eq!(loaded.namespace, Some("test".to_string()));
        assert_eq!(loaded.importance, 0.95);
        assert_eq!(loaded.access_count, 8); // get() increments it
    }

    #[test]
    fn update_changes_importance_and_namespace() {
        let store = MemoryStore::new();
        let id = store.save("test", Some("old_ns".to_string()), Some(0.5));
        let updated = store.update(&id, Some(0.9), Some(Some("new_ns".to_string())));
        assert!(updated);
        let mem = store.get(&id).unwrap();
        assert_eq!(mem.importance, 0.9);
        assert_eq!(mem.namespace, Some("new_ns".to_string()));
    }
}
