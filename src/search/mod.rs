//! Phase 3: Multi-Vector & Reciprocal Rank Fusion Router
//!
//! Implements a 3-way parallel retrieval pipeline using tokio::spawn:
//!   A) Vector similarity (dot-product / cosine over embedded arrays)
//!   B) BM25 keyword matching over identifier strings
//!   C) Structural lookup via PageRank-adjacent nodes
//!
//! Results are merged using the exact RRF formula:
//!   Score(d) = Sum[1.0 / (60.0 + rank_d)]

pub mod rrf;

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::asg::{Asg, EdgeKind, SharedAsg};
use crate::compressor::ChunkRegistry;

// ---------------------------------------------------------------------------
// Search Result Types
// ---------------------------------------------------------------------------

/// A single search result with a score and node ID.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub node_id: usize,
    pub score: f64,
    pub source: &'static str, // "vector", "bm25", "structural"
}

/// Merged result after RRF fusion.
#[derive(Debug, Clone)]
pub struct MergedResult {
    pub node_id: usize,
    pub rrf_score: f64,
    pub scores: HashMap<&'static str, f64>,
}

// ---------------------------------------------------------------------------
// Search Engine
// ---------------------------------------------------------------------------

/// The multi-vector RRF search engine.
#[derive(Clone)]
pub struct SearchEngine {
    asg: SharedAsg,
    registry: Arc<ChunkRegistry>,
    /// Pre-computed embeddings for each node (mock: hash-based vectors).
    embeddings: Arc<RwLock<HashMap<usize, Vec<f64>>>>,
}

impl SearchEngine {
    pub fn new(asg: SharedAsg, registry: ChunkRegistry) -> Self {
        let embeddings = Arc::new(RwLock::new(HashMap::new()));
        Self {
            asg,
            registry: Arc::new(registry),
            embeddings,
        }
    }

    /// Pre-compute embeddings for all nodes (mock: deterministic hash-based vectors).
    pub async fn precompute_embeddings(&self) {
        let mut emb = self.embeddings.write().await;
        for node in &self.asg.inner.nodes {
            let vec = self.hash_to_vector(&node.name, &node.kind);
            emb.insert(node.id, vec);
        }
    }

    /// Convert a string to a fixed-size embedding vector using a simple hash.
    fn hash_to_vector(&self, name: &str, kind: &str) -> Vec<f64> {
        let dim = 64;
        let mut vec = vec![0.0f64; dim];
        let combined = format!("{}{}", name, kind);
        for (i, ch) in combined.chars().enumerate() {
            let idx = (i * 7 + ch as usize) % dim;
            vec[idx] += (ch as u32 as f64) / 256.0;
        }
        // Normalize.
        let norm: f64 = vec.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }
        vec
    }

    /// Run the full 3-way parallel search pipeline.
    pub async fn search(&self, query: &str, top_k: usize) -> Vec<MergedResult> {
        // Ensure embeddings are computed.
        if self.embeddings.read().await.is_empty() {
            self.precompute_embeddings().await;
        }

        // Spawn three parallel search tasks.
        let query_vec = self.hash_to_vector(query, "");
        let asg_clone = self.asg.clone();
        let emb_clone = self.embeddings.clone();
        let registry_clone = self.registry.clone();

        let task_a = tokio::spawn(async move {
            Self::vector_search(asg_clone, emb_clone, &query_vec, top_k).await
        });

        let asg_clone2 = self.asg.clone();
        let query_b = query.to_owned();
        let task_b = tokio::spawn(async move {
            Self::bm25_search(asg_clone2, query_b, top_k).await
        });

        let asg_clone3 = self.asg.clone();
        let registry_clone2 = registry_clone.clone();
        let task_c = tokio::spawn(async move {
            Self::structural_search(asg_clone3, registry_clone2, top_k).await
        });

        let results_a = task_a.await.unwrap_or_default();
        let results_b = task_b.await.unwrap_or_default();
        let results_c = task_c.await.unwrap_or_default();

        // Merge using RRF.
        self.rrf_fusion(results_a, results_b, results_c)
    }

    // -----------------------------------------------------------------------
    // Search A: Vector Similarity
    // -----------------------------------------------------------------------

    async fn vector_search(
        asg: SharedAsg,
        embeddings: Arc<RwLock<HashMap<usize, Vec<f64>>>>,
        query_vec: &[f64],
        top_k: usize,
    ) -> Vec<SearchResult> {
        let emb = embeddings.read().await;
        let mut results: Vec<SearchResult> = Vec::new();

        for (node_id, node_vec) in emb.iter() {
            let similarity = Self::cosine_similarity(query_vec, node_vec);
            if similarity > 0.01 {
                results.push(SearchResult {
                    node_id: *node_id,
                    score: similarity,
                    source: "vector",
                });
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    // -----------------------------------------------------------------------
    // Search B: BM25 Keyword Matching
    // -----------------------------------------------------------------------

    async fn bm25_search(asg: SharedAsg, query: String, top_k: usize) -> Vec<SearchResult> {
        let query_terms: Vec<&str> = query.split_whitespace().collect();
        let mut results: Vec<SearchResult> = Vec::new();

        // BM25 parameters.
        let k1: f64 = 1.5;
        let b: f64 = 0.75;
        let avgdl = Self::average_doc_length(&asg);
        let n = asg.inner.nodes.len() as f64;

        for node in &asg.inner.nodes {
            let doc_len = node.name.len() + node.kind.len();
            let mut score: f64 = 0.0;

            for term in &query_terms {
                let tf = Self::term_frequency(&node.name, term)
                    + Self::term_frequency(&node.kind, term);
                if tf > 0.0 {
                    let idf = ((n - tf as f64 + 0.5) / (tf as f64 + 0.5)).ln();
                    let numerator = tf as f64 * (k1 + 1.0);
                    let denominator = tf as f64 + k1 * (1.0 - b + b * doc_len as f64 / avgdl);
                    score += idf * numerator / denominator;
                }
            }

            if score > 0.0 {
                results.push(SearchResult {
                    node_id: node.id,
                    score,
                    source: "bm25",
                });
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    fn term_frequency(text: &str, term: &str) -> f64 {
        text.matches(term).count() as f64
    }

    fn average_doc_length(asg: &SharedAsg) -> f64 {
        if asg.inner.nodes.is_empty() {
            return 1.0;
        }
        let total: usize = asg
            .inner
            .nodes
            .iter()
            .map(|n| n.name.len() + n.kind.len())
            .sum();
        total as f64 / asg.inner.nodes.len() as f64
    }

    // -----------------------------------------------------------------------
    // Search C: Structural Lookup via PageRank
    // -----------------------------------------------------------------------

    async fn structural_search(
        asg: SharedAsg,
        registry: Arc<ChunkRegistry>,
        top_k: usize,
    ) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = Vec::new();

        // Find the top PageRank nodes.
        let mut pr_nodes: Vec<(usize, f64)> = asg
            .inner
            .nodes
            .iter()
            .map(|n| (n.id, n.pagerank))
            .collect();
        pr_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // For each top PageRank node, find 1-hop and 2-hop neighbors.
        for (node_id, pr) in pr_nodes.iter().take(10) {
            let neighbors = Self::collect_neighbors(&asg, *node_id, 2);
            for neighbor_id in neighbors {
                if let Some(node) = asg.get_node(neighbor_id) {
                    results.push(SearchResult {
                        node_id: neighbor_id,
                        score: node.pagerank * pr,
                        source: "structural",
                    });
                }
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.dedup_by_key(|r| r.node_id);
        results.truncate(top_k);
        results
    }

    fn collect_neighbors(asg: &SharedAsg, start: usize, max_hops: usize) -> Vec<usize> {
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((start, 0));
        visited.insert(start);

        while let Some((current, hops)) = queue.pop_front() {
            if hops >= max_hops {
                continue;
            }

            // Outgoing edges.
            if let Some(edge_indices) = asg.inner.adjacency.get(&current) {
                for edge_idx in edge_indices {
                    if let Some(edge) = asg.inner.edges.get(*edge_idx) {
                        if !visited.contains(&edge.to) {
                            visited.insert(edge.to);
                            queue.push_back((edge.to, hops + 1));
                        }
                    }
                }
            }

            // Incoming edges.
            if let Some(edge_indices) = asg.inner.reverse_adjacency.get(&current) {
                for edge_idx in edge_indices {
                    if let Some(edge) = asg.inner.edges.get(*edge_idx) {
                        if !visited.contains(&edge.from) {
                            visited.insert(edge.from);
                            queue.push_back((edge.from, hops + 1));
                        }
                    }
                }
            }
        }

        visited.into_iter().collect()
    }

    // -----------------------------------------------------------------------
    // RRF Fusion
    // -----------------------------------------------------------------------

    /// Merge ranked lists using the exact RRF formula:
    /// Score(d) = Sum[1.0 / (60.0 + rank_d)]
    fn rrf_fusion(
        &self,
        list_a: Vec<SearchResult>,
        list_b: Vec<SearchResult>,
        list_c: Vec<SearchResult>,
    ) -> Vec<MergedResult> {
        let mut scores: HashMap<usize, MergedResult> = HashMap::new();

        for (rank, result) in list_a.iter().enumerate() {
            let rrf = 1.0 / (60.0 + rank as f64);
            let entry = scores
                .entry(result.node_id)
                .or_insert(MergedResult {
                    node_id: result.node_id,
                    rrf_score: 0.0,
                    scores: HashMap::new(),
                });
            entry.rrf_score += rrf;
            entry.scores.insert(result.source, result.score);
        }

        for (rank, result) in list_b.iter().enumerate() {
            let rrf = 1.0 / (60.0 + rank as f64);
            let entry = scores
                .entry(result.node_id)
                .or_insert(MergedResult {
                    node_id: result.node_id,
                    rrf_score: 0.0,
                    scores: HashMap::new(),
                });
            entry.rrf_score += rrf;
            entry.scores.insert(result.source, result.score);
        }

        for (rank, result) in list_c.iter().enumerate() {
            let rrf = 1.0 / (60.0 + rank as f64);
            let entry = scores
                .entry(result.node_id)
                .or_insert(MergedResult {
                    node_id: result.node_id,
                    rrf_score: 0.0,
                    scores: HashMap::new(),
                });
            entry.rrf_score += rrf;
            entry.scores.insert(result.source, result.score);
        }

        let mut merged: Vec<MergedResult> = scores.into_values().collect();
        merged.sort_by(|a, b| b.rrf_score.partial_cmp(&a.rrf_score).unwrap_or(std::cmp::Ordering::Equal));
        merged
    }
}
