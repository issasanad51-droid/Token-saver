//! Local Trigram Indexing (Hybrid Search)
//!
//! This is Cursor's third indexing pillar. Alongside semantic vector search,
//! we maintain a local character n-gram (default n=3, "trigram") inverted
//! index over chunk text. This powers fast lexical/keyword search that
//! combines with semantic search in a hybrid retrieval strategy — exact
//! variable names or text matches found via trigrams, meaning found via
//! vectors, fused together (e.g. by the existing RRF engine).

use std::collections::{HashMap, HashSet};

use crate::ast::chunker::AstChunk;

/// A single trigram-search hit.
#[derive(Debug, Clone)]
pub struct TrigramHit {
    pub chunk_id: String,
    /// Raw count of matching trigrams.
    pub score: f64,
    /// Jaccard similarity between query and chunk trigram sets (0..1).
    pub jaccard: f64,
}

/// A character n-gram inverted index over chunk text.
pub struct TrigramIndex {
    /// trigram -> chunk ids containing it.
    index: HashMap<String, Vec<String>>,
    /// chunk id -> indexed document.
    docs: HashMap<String, TrigramDoc>,
    n: usize,
}

struct TrigramDoc {
    chunk: AstChunk,
    trigrams: Vec<String>,
}

impl TrigramIndex {
    /// Create an index with n-gram size `n` (defaults to 3 → trigrams).
    pub fn new(n: usize) -> Self {
        Self {
            index: HashMap::new(),
            docs: HashMap::new(),
            n: n.max(1),
        }
    }

    /// Index a set of chunks.
    pub fn index(&mut self, chunks: &[AstChunk]) {
        for chunk in chunks {
            self.add(chunk.clone());
        }
    }

    /// Add (or replace) a single chunk in the index.
    pub fn add(&mut self, chunk: AstChunk) {
        // Replace any prior version of this chunk.
        if self.docs.contains_key(&chunk.id) {
            self.remove(&chunk.id);
        }
        let trigrams = Self::trigrams(&chunk.source, self.n);
        for tg in &trigrams {
            self.index.entry(tg.clone()).or_default().push(chunk.id.clone());
        }
        self.docs.insert(
            chunk.id.clone(),
            TrigramDoc {
                chunk,
                trigrams,
            },
        );
    }

    /// Remove a chunk (e.g. on deletion detected by the Merkle diff).
    pub fn remove(&mut self, chunk_id: &str) {
        if let Some(doc) = self.docs.remove(chunk_id) {
            for tg in doc.trigrams {
                if let Some(ids) = self.index.get_mut(&tg) {
                    ids.retain(|id| id != chunk_id);
                }
            }
        }
    }

    /// Score chunks by trigram overlap with the query.
    ///
    /// Returns up to `top_k` hits sorted by Jaccard similarity, tie-broken by
    /// raw matching-trigram count. This is the lexical signal fed into the
    /// hybrid fusion.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<TrigramHit> {
        let q_tris = Self::trigrams(query, self.n);
        if q_tris.is_empty() {
            return Vec::new();
        }
        let q_set: HashSet<&str> = q_tris.iter().map(String::as_str).collect();

        // Accumulate raw overlap counts per candidate chunk.
        let mut counts: HashMap<&str, f64> = HashMap::new();
        for tg in &q_tris {
            if let Some(ids) = self.index.get(tg) {
                for id in ids {
                    *counts.entry(id.as_str()).or_insert(0.0) += 1.0;
                }
            }
        }

        let mut hits: Vec<TrigramHit> = counts
            .into_iter()
            .filter_map(|(id, score)| {
                let doc = self.docs.get(id)?;
                let doc_set: HashSet<&str> = doc.trigrams.iter().map(String::as_str).collect();
                let intersection = q_set.intersection(&doc_set).count() as f64;
                let union = q_set.union(&doc_set).count() as f64;
                let jaccard = if union > 0.0 { intersection / union } else { 0.0 };
                Some(TrigramHit {
                    chunk_id: id.to_string(),
                    score,
                    jaccard,
                })
            })
            .collect();

        // Use total_cmp for deterministic bitwise ordering — partial_cmp
        // returns None for NaN scores, making the sort non-deterministic.
        hits.sort_by(|a, b| {
            b.jaccard
                .total_cmp(&a.jaccard)
                .then_with(|| b.score.total_cmp(&a.score))
        });
        hits.truncate(top_k);
        hits
    }

    /// Number of indexed chunks.
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Borrow the underlying chunk by id (used to hydrate a hit into context).
    pub fn get(&self, chunk_id: &str) -> Option<&AstChunk> {
        self.docs.get(chunk_id).map(|d| &d.chunk)
    }

    // -----------------------------------------------------------------------
    // Trigram extraction
    // -----------------------------------------------------------------------

    /// Extract all n-grams of `text` as `String`s.
    fn trigrams(text: &str, n: usize) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < n {
            return vec![chars.iter().collect()];
        }
        (0..=chars.len() - n)
            .map(|i| chars[i..i + n].iter().collect())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::chunker::{AstChunk, ChunkKind};

    fn chunk(id: &str, src: &str) -> AstChunk {
        AstChunk {
            id: id.to_string(),
            file_path: std::path::PathBuf::from("x.rs"),
            name: id.to_string(),
            kind: ChunkKind::Function,
            source: src.to_string(),
            byte_range: (0, src.len()),
            line_range: (0, 1),
            content_hash: AstChunk::hash_source(src),
            parent: None,
        }
    }

    #[test]
    fn trigram_search_finds_exact_substring() {
        let mut idx = TrigramIndex::new(3);
        idx.add(chunk("a", "fn calculate_token_savings() {}"));
        idx.add(chunk("b", "struct Widget { }"));

        let hits = idx.search("calculate_token", 5);
        assert_eq!(hits[0].chunk_id, "a");
    }

    #[test]
    fn trigram_search_ranks_by_overlap() {
        let mut idx = TrigramIndex::new(3);
        idx.add(chunk("short", "foo"));
        idx.add(chunk("long", "foo bar baz qux"));

        let hits = idx.search("foo bar", 5);
        // The longer doc shares more trigrams with the query.
        assert_eq!(hits[0].chunk_id, "long");
    }

    #[test]
    fn remove_keeps_index_consistent() {
        let mut idx = TrigramIndex::new(3);
        idx.add(chunk("a", "hello world"));
        assert_eq!(idx.len(), 1);
        idx.remove("a");
        assert!(idx.is_empty());
        assert!(idx.search("hello", 5).is_empty());
    }
}
