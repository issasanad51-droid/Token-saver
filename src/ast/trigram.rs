//! Local Trigram Indexing (Hybrid Search)
//!
//! This is Cursor's third indexing pillar. Alongside semantic vector search,
//! we maintain a local character n-gram (default n=3, "trigram") inverted
//! index over chunk text. This powers fast lexical/keyword search that
//! combines with semantic search in a hybrid retrieval strategy — exact
//! variable names or text matches found via trigrams, meaning found via
//! vectors, fused together (e.g. by the existing RRF engine).
//!
//! # Storage choice
//!
//! Both `index` (trigram → chunk-ids) and `trigrams` per doc use
//! [`HashSet`](std::collections::HashSet) rather than `Vec`. This is important
//! for correctness: a chunk whose source contains `foofoo` would otherwise
//! have the trigram `"foo"` recorded twice, doubling its score against any
//! query containing `"foo"`. Using a `HashSet` makes the per-trigram posting
//! list a set membership signal — each chunk contributes at most once per
//! trigram, which is what BM25/RRF downstream expect.

use std::collections::{HashMap, HashSet};

use crate::ast::chunker::AstChunk;

/// A single trigram-search hit.
#[derive(Debug, Clone)]
pub struct TrigramHit {
    pub chunk_id: String,
    /// Number of distinct query trigrams that appear in the chunk.
    pub score: f64,
    /// Jaccard similarity between query and chunk trigram sets (0..1).
    pub jaccard: f64,
}

/// A character n-gram inverted index over chunk text.
pub struct TrigramIndex {
    /// trigram -> set of chunk ids containing it.
    index: HashMap<String, HashSet<String>>,
    /// chunk id -> indexed document.
    docs: HashMap<String, TrigramDoc>,
    n: usize,
}

struct TrigramDoc {
    chunk: AstChunk,
    trigrams: HashSet<String>,
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
            self.index
                .entry(tg.clone())
                .or_default()
                .insert(chunk.id.clone());
        }
        self.docs
            .insert(chunk.id.clone(), TrigramDoc { chunk, trigrams });
    }

    /// Remove a chunk (e.g. on deletion detected by the Merkle diff).
    ///
    /// Posting lists that become empty after removal are dropped from the
    /// `index` map so that incremental reindex over many add/remove cycles
    /// does not leak dead trigram keys.
    pub fn remove(&mut self, chunk_id: &str) {
        if let Some(doc) = self.docs.remove(chunk_id) {
            for tg in doc.trigrams {
                let mut drop_key = false;
                if let Some(ids) = self.index.get_mut(&tg) {
                    ids.remove(chunk_id);
                    drop_key = ids.is_empty();
                }
                if drop_key {
                    self.index.remove(&tg);
                }
            }
        }
    }

    /// Remove all chunks belonging to a given file path.
    /// Returns the number of chunks removed.
    pub fn remove_by_file(&mut self, file_path: &std::path::Path) -> usize {
        let ids_to_remove: Vec<String> = self
            .docs
            .iter()
            .filter(|(_, doc)| doc.chunk.file_path == file_path)
            .map(|(id, _)| id.clone())
            .collect();
        let count = ids_to_remove.len();
        for id in &ids_to_remove {
            self.remove(id);
        }
        count
    }

    /// List all unique file paths currently indexed.
    pub fn indexed_files(&self) -> Vec<std::path::PathBuf> {
        let mut files: HashSet<std::path::PathBuf> = HashSet::new();
        for doc in self.docs.values() {
            files.insert(doc.chunk.file_path.clone());
        }
        files.into_iter().collect()
    }

    /// Score chunks by trigram overlap with the query.
    ///
    /// Returns up to `top_k` hits sorted by Jaccard similarity, tie-broken by
    /// raw matching-trigram count. This is the lexical signal fed into the
    /// hybrid fusion.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<TrigramHit> {
        if query.is_empty() {
            return Vec::new();
        }
        let q_set = Self::trigrams(query, self.n);
        if q_set.is_empty() {
            return Vec::new();
        }

        // Accumulate raw overlap counts per candidate chunk. Each chunk
        // contributes at most 1 per distinct query trigram because both the
        // posting list and `q_set` are sets.
        let mut counts: HashMap<&str, f64> = HashMap::new();
        for tg in &q_set {
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
                let intersection = q_set.intersection(&doc.trigrams).count() as f64;
                let union = q_set.union(&doc.trigrams).count() as f64;
                let jaccard = if union > 0.0 {
                    intersection / union
                } else {
                    0.0
                };
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

    /// Extract all *distinct* n-grams of `text` as a `HashSet<String>`.
    ///
    /// Returns an empty set when:
    ///   - `text` is empty, or
    ///   - `text` has fewer than `n` characters *and* those characters are
    ///     all whitespace.
    ///
    /// The "fewer than `n` chars" fallback previously returned a one-element
    /// `Vec` containing the whole string — even for the empty string — which
    /// silently defeated the empty-query early-return in [`search`](Self::search)
    /// and polluted the inverted index with a `""` key. We now drop empty
    /// trigrams entirely.
    fn trigrams(text: &str, n: usize) -> HashSet<String> {
        let chars: Vec<char> = text.chars().collect();
        if chars.is_empty() {
            return HashSet::new();
        }
        if chars.len() < n {
            // Only keep non-whitespace short-prefix trigrams; whitespace-only
            // inputs would otherwise produce a `""`-like degenerate key.
            let joined: String = chars.iter().collect();
            if joined.trim().is_empty() {
                return HashSet::new();
            }
            let mut set = HashSet::with_capacity(1);
            set.insert(joined);
            return set;
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

    #[test]
    fn remove_by_file_clears_all_chunks_for_path() {
        let mut idx = TrigramIndex::new(3);

        // Create chunks from two different files.
        let file_a = std::path::PathBuf::from("a.rs");
        let file_b = std::path::PathBuf::from("b.rs");
        let mut chunk_a = chunk("a", "function alpha");
        chunk_a.file_path = file_a.clone();
        let mut chunk_b1 = chunk("b1", "struct Beta");
        chunk_b1.file_path = file_b.clone();
        let mut chunk_b2 = chunk("b2", "enum Gamma");
        chunk_b2.file_path = file_b.clone();

        idx.add(chunk_a);
        idx.add(chunk_b1);
        idx.add(chunk_b2);
        assert_eq!(idx.len(), 3);

        // Remove all chunks from file b.rs
        let removed = idx.remove_by_file(&file_b);
        assert_eq!(removed, 2);
        assert_eq!(idx.len(), 1);

        // File a.rs chunk is still searchable
        let hits = idx.search("alpha", 5);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk_id, "a");

        // File b.rs chunks are gone
        assert!(idx.search("Beta", 5).is_empty());
    }

    /// Regression test: a chunk whose source contains the same trigram twice
    /// (e.g. `"foofoo"`) must contribute at most 1 to the score of any query
    /// that contains that trigram. With the old `Vec`-based posting list, the
    /// chunk would have its id pushed twice into `index["foo"]`, inflating the
    /// raw `score` field by 2x.
    #[test]
    fn duplicate_trigrams_in_chunk_do_not_inflate_score() {
        let mut idx = TrigramIndex::new(3);
        idx.add(chunk("dup", "foofoo")); // contains "foo" twice
        idx.add(chunk("once", "foobar")); // contains "foo" once

        let hits = idx.search("foo", 5);
        assert_eq!(hits.len(), 2);
        // Both chunks contain "foo" exactly once as a distinct trigram.
        assert_eq!(hits[0].score, 1.0);
        assert_eq!(hits[1].score, 1.0);
    }

    /// Regression test: an empty query must return zero hits immediately.
    /// The previous implementation called `trigrams("", 3)` which yielded
    /// `vec![""]`, so `q_tris.is_empty()` was `false` and the search
    /// proceeded with a degenerate `""` trigram key.
    #[test]
    fn empty_query_returns_no_hits() {
        let mut idx = TrigramIndex::new(3);
        idx.add(chunk("a", "fn alpha() {}"));
        assert!(idx.search("", 5).is_empty());
    }

    /// Regression test: removing the last chunk containing a given trigram
    /// must also remove the trigram key from `index`. Otherwise incremental
    /// reindex over many add/remove cycles grows the `index` map
    /// monotonically with dead keys.
    #[test]
    fn remove_drops_empty_posting_lists() {
        let mut idx = TrigramIndex::new(3);
        idx.add(chunk("a", "fn alpha() {}"));
        // The trigram "alp" should be in the index while chunk "a" exists.
        assert!(idx.index.contains_key("alp"));
        idx.remove("a");
        // After removal the posting list for "alp" should be gone entirely.
        assert!(
            !idx.index.contains_key("alp"),
            "trigram key leaked after removing the last chunk containing it"
        );
    }
}
