//! Query-aware hybrid retrieval with weighted PPR and custom RRF.
//!
//! The live pipeline combines three deliberately different signals:
//!
//! 1. deterministic local semantic feature vectors,
//! 2. a real corpus-level BM25 lexical ranker, and
//! 3. edge-weighted Personalized PageRank seeded by the query candidates and
//!    the node under the editor cursor.
//!
//! The streams are fused by [`rrf`] with configurable per-stream weights and a
//! bounded score-aware multiplier. No source code leaves the process.

pub mod rrf;

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::asg::{
    EdgeKind, Node, PageRankEngine, PersonalizedPageRankConfig, SharedAsg,
};
use crate::compressor::ChunkRegistry;
use crate::recency::RecencyTracker;
use rrf::{RankedDoc, RrfConfig};

const STREAM_NAMES: [&str; 3] = ["semantic", "bm25", "structural"];
const EMBEDDING_DIMENSIONS: usize = 256;

// ---------------------------------------------------------------------------
// Configuration and result types
// ---------------------------------------------------------------------------

/// Tuning knobs for candidate generation, query-time PPR, and RRF fusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    /// Generate this many candidates per requested result before fusion.
    pub candidate_multiplier: usize,
    /// Ignore semantic candidates below this cosine similarity.
    pub semantic_min_score: f64,
    pub bm25_k1: f64,
    pub bm25_b: f64,
    /// Number of semantic/lexical candidates used as PPR teleport seeds.
    pub ppr_seed_candidates: usize,
    /// Relative weight of the editor's current ASG node in the teleport vector.
    pub context_seed_weight: f64,
    pub ppr: PersonalizedPageRankConfig,
    /// Stream order is `[semantic, bm25, structural]`.
    pub rrf: RrfConfig,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            candidate_multiplier: 4,
            semantic_min_score: 0.03,
            bm25_k1: 1.5,
            bm25_b: 0.75,
            ppr_seed_candidates: 8,
            context_seed_weight: 4.0,
            ppr: PersonalizedPageRankConfig::default(),
            rrf: RrfConfig {
                k: 60.0,
                weights: vec![1.15, 1.0, 1.1],
                score_alpha: 0.15,
            },
        }
    }
}

/// A result emitted by one retrieval stream.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub node_id: usize,
    pub score: f64,
    pub source: &'static str,
}

/// A fused result with both raw backend scores and RRF provenance.
#[derive(Debug, Clone, Serialize)]
pub struct MergedResult {
    pub node_id: usize,
    pub rrf_score: f64,
    pub scores: HashMap<&'static str, f64>,
    pub rrf_contributions: HashMap<&'static str, f64>,
}

// ---------------------------------------------------------------------------
// Search engine
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct SearchEngine {
    asg: SharedAsg,
    registry: Arc<ChunkRegistry>,
    config: SearchConfig,
    /// Precomputed local feature vectors keyed by dense ASG node ID.
    embeddings: Arc<RwLock<HashMap<usize, Vec<f64>>>>,
    /// Cached tokenized document terms for BM25 (node_id -> terms).
    document_terms: Arc<RwLock<HashMap<usize, Vec<String>>>>,
    /// Timestamp of last access for temporal decay.
    last_access: Arc<std::sync::Mutex<Instant>>,
    /// Temporal decay factor per second (default 0.995, half-life ~138s).
    decay_factor: f64,
    /// Sliding-window recency tracker that boosts recently touched nodes in
    /// the PPR teleport vector.
    recency: Arc<RecencyTracker>,
}

impl SearchEngine {
    pub fn new(asg: SharedAsg, registry: ChunkRegistry) -> Self {
        Self::with_config(asg, registry, SearchConfig::default())
    }

    pub fn with_config(asg: SharedAsg, registry: ChunkRegistry, config: SearchConfig) -> Self {
        Self {
            asg,
            registry: Arc::new(registry),
            config,
            embeddings: Arc::new(RwLock::new(HashMap::new())),
            document_terms: Arc::new(RwLock::new(HashMap::new())),
            last_access: Arc::new(std::sync::Mutex::new(Instant::now())),
            decay_factor: 0.995,
            recency: Arc::new(RecencyTracker::new()),
        }
    }

    /// Attach a shared recency tracker so recently edited/queried nodes get a
    /// temporary boost in the PPR teleport vector.
    pub fn set_recency(&mut self, recency: Arc<RecencyTracker>) {
        self.recency = recency;
    }

    pub fn recency(&self) -> &Arc<RecencyTracker> {
        &self.recency
    }

    pub fn config(&self) -> &SearchConfig {
        &self.config
    }

    /// Apply temporal decay to PPR scores so stale code fades over time.
    /// Each score is multiplied by `decay_factor^elapsed_secs`.
    pub fn apply_temporal_decay(&self, scores: &mut [f64]) {
        let last = *self.last_access.lock().unwrap();
        let elapsed = last.elapsed().as_secs_f64();
        if elapsed <= 0.0 || !self.decay_factor.is_finite() {
            return;
        }
        let decay = self.decay_factor.powf(elapsed);
        for score in scores.iter_mut() {
            *score *= decay;
        }
    }

    /// Update the last access timestamp to now.
    pub fn touch_access(&self) {
        *self.last_access.lock().unwrap() = Instant::now();
    }

    /// Precompute vectors and document terms once. The feature hasher is
    /// deterministic, local, dependency-free, and gives exact identifiers
    /// substantially more weight than fuzzy character trigrams.
    pub async fn precompute_embeddings(&self) {
        let mut embeddings = self.embeddings.write().await;
        let mut doc_terms = self.document_terms.write().await;
        embeddings.clear();
        doc_terms.clear();
        for node in &self.asg.inner.nodes {
            if self.is_searchable(node.id) {
                let ctx_text = contextual_text(node, &self.asg.inner);
                embeddings.insert(node.id, embed_text(&ctx_text));
                doc_terms.insert(node.id, tokenize(&ctx_text));
            }
        }
    }

    /// Search without editor context.
    pub async fn search(&self, query: &str, top_k: usize) -> Vec<MergedResult> {
        self.search_with_context(query, top_k, None).await
    }

    /// Run semantic and BM25 candidate generation in parallel, seed a
    /// query-specific weighted PPR pass from those candidates and the optional
    /// cursor node, then fuse all three rankings.
    pub async fn search_with_context(
        &self,
        query: &str,
        top_k: usize,
        context_node: Option<usize>,
    ) -> Vec<MergedResult> {
        if top_k == 0 || query.trim().is_empty() || self.asg.inner.nodes.is_empty() {
            return Vec::new();
        }
        if self.embeddings.read().await.is_empty() {
            self.precompute_embeddings().await;
        }

        self.touch_access();

        let pool_size = top_k
            .saturating_mul(self.config.candidate_multiplier.max(1))
            .max(top_k)
            .min(self.asg.inner.nodes.len());

        let query_vector = embed_text(query);
        let vector_asg = self.asg.clone();
        let vector_embeddings = self.embeddings.clone();
        let vector_registry = self.registry.clone();
        let semantic_floor = if self.config.semantic_min_score.is_finite() {
            self.config.semantic_min_score
        } else {
            0.0
        };
        let semantic_task = tokio::spawn(async move {
            Self::semantic_search(
                vector_asg,
                vector_registry,
                vector_embeddings,
                query_vector,
                semantic_floor,
                pool_size,
            )
            .await
        });

        let lexical_asg = self.asg.clone();
        let lexical_registry = self.registry.clone();
        let lexical_query = query.to_owned();
        let k1 = self.config.bm25_k1;
        let b = self.config.bm25_b;
        let lexical_document_terms = self.document_terms.clone();
        let lexical_task = tokio::spawn(async move {
            Self::bm25_search(
                lexical_asg,
                lexical_registry,
                lexical_document_terms,
                &lexical_query,
                k1,
                b,
                pool_size,
            )
            .await
        });

        let semantic = semantic_task.await.unwrap_or_default();
        let lexical = lexical_task.await.unwrap_or_default();

        let seeds = self.ppr_seeds(&semantic, &lexical, context_node);
        let structural_asg = self.asg.clone();
        let structural_registry = self.registry.clone();
        let ppr_config = self.config.ppr.clone();
        let decay_factor = self.decay_factor;
        let last_access = self.last_access.clone();
        let structural = tokio::task::spawn_blocking(move || {
            Self::structural_search(
                structural_asg,
                structural_registry,
                ppr_config,
                seeds,
                pool_size,
                decay_factor,
                last_access,
            )
        })
        .await
        .unwrap_or_default();

        self.fuse(semantic, lexical, structural, top_k)
    }

    fn is_searchable(&self, node_id: usize) -> bool {
        self.registry.chunks.contains_key(&node_id)
    }

    // -----------------------------------------------------------------------
    // Semantic stream
    // -----------------------------------------------------------------------

    async fn semantic_search(
        asg: SharedAsg,
        registry: Arc<ChunkRegistry>,
        embeddings: Arc<RwLock<HashMap<usize, Vec<f64>>>>,
        query_vector: Vec<f64>,
        minimum_score: f64,
        top_k: usize,
    ) -> Vec<SearchResult> {
        let embeddings = embeddings.read().await;
        let mut results = Vec::new();
        for (node_id, node_vector) in embeddings.iter() {
            if asg.get_node(*node_id).is_none() || !registry.chunks.contains_key(node_id) {
                continue;
            }
            let score = cosine_similarity(&query_vector, node_vector);
            if score.is_finite() && score >= minimum_score {
                results.push(SearchResult {
                    node_id: *node_id,
                    score,
                    source: STREAM_NAMES[0],
                });
            }
        }
        sort_results(&mut results);
        results.truncate(top_k);
        results
    }

    // -----------------------------------------------------------------------
    // BM25 stream
    // -----------------------------------------------------------------------

    async fn bm25_search(
        asg: SharedAsg,
        registry: Arc<ChunkRegistry>,
        document_terms_cache: Arc<RwLock<HashMap<usize, Vec<String>>>>,
        query: &str,
        configured_k1: f64,
        configured_b: f64,
        top_k: usize,
    ) -> Vec<SearchResult> {
        let query_terms: Vec<String> = tokenize(query);
        if query_terms.is_empty() {
            return Vec::new();
        }
        let unique_query: HashSet<&str> = query_terms.iter().map(String::as_str).collect();

        // Use cached document terms if available, otherwise fall back to
        // re-tokenizing from searchable text.
        let cached = document_terms_cache.read().await;
        let documents: Vec<(usize, Vec<String>)> = if !cached.is_empty() {
            asg.inner
                .nodes
                .iter()
                .filter(|node| registry.chunks.contains_key(&node.id))
                .filter_map(|node| {
                    cached
                        .get(&node.id)
                        .map(|terms| (node.id, terms.clone()))
                })
                .collect()
        } else {
            drop(cached);
            asg.inner
                .nodes
                .iter()
                .filter(|node| registry.chunks.contains_key(&node.id))
                .map(|node| {
                    let ctx_text = contextual_text(node, &asg.inner);
                    (node.id, tokenize(&ctx_text))
                })
                .collect()
        };
        if documents.is_empty() {
            return Vec::new();
        }

        let average_length = documents
            .iter()
            .map(|(_, terms)| terms.len())
            .sum::<usize>() as f64
            / documents.len() as f64;
        let document_count = documents.len() as f64;
        let k1 = if configured_k1.is_finite() {
            configured_k1.max(0.01)
        } else {
            1.5
        };
        let b = if configured_b.is_finite() {
            configured_b.clamp(0.0, 1.0)
        } else {
            0.75
        };

        let mut document_frequency: HashMap<&str, usize> = HashMap::new();
        for term in &unique_query {
            let count = documents
                .iter()
                .filter(|(_, terms)| terms.iter().any(|candidate| candidate.as_str() == *term))
                .count();
            document_frequency.insert(*term, count);
        }

        let mut results = Vec::new();
        for (node_id, terms) in documents {
            let document_length = terms.len().max(1) as f64;
            let mut frequencies: HashMap<&str, usize> = HashMap::new();
            for term in &terms {
                if unique_query.contains(term.as_str()) {
                    *frequencies.entry(term.as_str()).or_default() += 1;
                }
            }

            let mut score = 0.0;
            for query_term in &query_terms {
                let term = query_term.as_str();
                let tf = *frequencies.get(term).unwrap_or(&0) as f64;
                if tf == 0.0 {
                    continue;
                }
                let df = *document_frequency.get(term).unwrap_or(&0) as f64;
                let idf = (1.0 + (document_count - df + 0.5) / (df + 0.5)).ln();
                let length_norm = 1.0 - b + b * document_length / average_length.max(1.0);
                score += idf * (tf * (k1 + 1.0)) / (tf + k1 * length_norm);
            }

            if score.is_finite() && score > 0.0 {
                results.push(SearchResult {
                    node_id,
                    score,
                    source: STREAM_NAMES[1],
                });
            }
        }
        sort_results(&mut results);
        results.truncate(top_k);
        results
    }

    // -----------------------------------------------------------------------
    // Weighted Personalized PageRank stream
    // -----------------------------------------------------------------------

    fn ppr_seeds(
        &self,
        semantic: &[SearchResult],
        lexical: &[SearchResult],
        context_node: Option<usize>,
    ) -> Vec<(usize, f64)> {
        let mut seeds: HashMap<usize, f64> = HashMap::new();
        if let Some(node_id) = context_node.filter(|id| self.asg.get_node(*id).is_some()) {
            let weight = if self.config.context_seed_weight.is_finite() {
                self.config.context_seed_weight.max(0.0)
            } else {
                4.0
            };
            *seeds.entry(node_id).or_default() += weight;
        }

        let limit = self.config.ppr_seed_candidates.max(1);
        for stream in [semantic, lexical] {
            for (rank, result) in stream.iter().take(limit).enumerate() {
                // Rank-normalized seed weights avoid mixing incomparable cosine
                // and BM25 score scales before PPR.
                *seeds.entry(result.node_id).or_default() += 1.0 / (rank + 1) as f64;
            }
        }
        let mut seeds: Vec<(usize, f64)> = seeds.into_iter().collect();
        seeds.sort_by_key(|(id, _)| *id);
        seeds
    }

    fn structural_search(
        asg: SharedAsg,
        registry: Arc<ChunkRegistry>,
        config: PersonalizedPageRankConfig,
        seeds: Vec<(usize, f64)>,
        top_k: usize,
        decay_factor: f64,
        last_access: Arc<std::sync::Mutex<Instant>>,
    ) -> Vec<SearchResult> {
        let engine = PageRankEngine::from_config(config);
        let scores = engine.personalized_scores(&asg.inner, &seeds);

        // Apply temporal decay to PPR scores so stale code fades over time.
        let mut scores = scores;
        {
            let last = *last_access.lock().unwrap();
            let elapsed = last.elapsed().as_secs_f64();
            if elapsed > 0.0 && decay_factor.is_finite() {
                let decay = decay_factor.powf(elapsed);
                for score in scores.iter_mut() {
                    *score *= decay;
                }
            }
        }

        let mut results: Vec<SearchResult> = scores
            .into_iter()
            .enumerate()
            .filter(|(node_id, score)| {
                *score > 0.0 && registry.chunks.contains_key(node_id)
            })
            .map(|(node_id, score)| SearchResult {
                node_id,
                score,
                source: STREAM_NAMES[2],
            })
            .collect();
        sort_results(&mut results);
        results.truncate(top_k);
        results
    }

    // -----------------------------------------------------------------------
    // ASG structural dependency scores (for RRF tie-breaking)
    // -----------------------------------------------------------------------

    /// Compute per-node ASG structural dependency weights used as a
    /// deterministic tie-breaker inside the RRF fusion.  Only weighted
    /// incoming `calls` and `references` edges count — the two edge kinds
    /// that carry genuine structural signal in a code graph (as opposed to
    /// lexical `contains` nesting).
    ///
    /// Weights are taken from `search.ppr.edge_weights` so they stay
    /// consistent with the PPR structural stream.
    fn compute_structural_dependency_scores(&self) -> HashMap<usize, f64> {
        let mut scores = HashMap::with_capacity(self.asg.inner.nodes.len());
        for node in &self.asg.inner.nodes {
            let weight: f64 = self
                .asg
                .inner
                .reverse_adjacency
                .get(&node.id)
                .map(|edge_indices| {
                    edge_indices
                        .iter()
                        .map(|&edge_idx| {
                            let edge = &self.asg.inner.edges[edge_idx];
                            match edge.kind {
                                EdgeKind::Calls => self.config.ppr.edge_weights.calls,
                                EdgeKind::References => self.config.ppr.edge_weights.references,
                                // Only calls and references are true structural
                                // dependencies; contains/imports/etc. are excluded.
                                _ => 0.0,
                            }
                        })
                        .sum()
                })
                .unwrap_or(0.0);
            scores.insert(node.id, weight);
        }
        scores
    }

    // -----------------------------------------------------------------------
    // Custom RRF
    // -----------------------------------------------------------------------

    fn fuse(
        &self,
        semantic: Vec<SearchResult>,
        lexical: Vec<SearchResult>,
        structural: Vec<SearchResult>,
        top_k: usize,
    ) -> Vec<MergedResult> {
        let original_streams = [&semantic, &lexical, &structural];
        let ranked_streams: Vec<Vec<RankedDoc<usize>>> = original_streams
            .iter()
            .map(|stream| {
                stream
                    .iter()
                    .map(|result| RankedDoc::new(result.node_id, result.score))
                    .collect()
            })
            .collect();

        // Pre-compute per-node structural dependency weights so the RRF
        // fusion can break ties deterministically via ASG topology.
        let structural_dependency_scores = self.compute_structural_dependency_scores();

        let mut merged: Vec<MergedResult> = rrf::fuse(
            &ranked_streams,
            &self.config.rrf,
            Some(&structural_dependency_scores),
        )
        .into_iter()
        .map(|fused| {
            let mut scores = HashMap::new();
            let mut contributions = HashMap::new();
            for (index, stream_name) in STREAM_NAMES.iter().enumerate() {
                if let Some(score) = fused.raw_scores[index] {
                    scores.insert(*stream_name, score);
                }
                let contribution = fused.rrf_contributions[index];
                if contribution > 0.0 {
                    contributions.insert(*stream_name, contribution);
                }
            }
            MergedResult {
                node_id: fused.id,
                rrf_score: fused.fused_score,
                scores,
                rrf_contributions: contributions,
            }
        })
        .collect();
        merged.truncate(top_k);
        merged
    }
}

// ---------------------------------------------------------------------------
// Local text features
// ---------------------------------------------------------------------------

fn searchable_text(node: &Node) -> String {
    // Include tracker_id once (not the name three times), and include kind
    // for type filtering. The body provides lexical coverage of internal
    // concepts. Repeating the body boosts term frequency for code-internal
    // identifiers without the fragile triple-name hack.
    format!(
        "{} {} {}\n{}\n{}",
        node.tracker_id, node.name, node.kind, node.source, node.source
    )
}

/// Contextual text wraps searchable_text with module context from the
/// tracker_id hierarchy. This helps BM25 and semantic search understand
/// what module a chunk belongs to.
fn contextual_text(node: &Node, _asg: &crate::asg::Asg) -> String {
    let base = searchable_text(node);
    // Find the parent module by looking at the tracker_id hierarchy.
    // e.g. "crate::server::mod::fn::handler" → parent module is "crate::server::mod"
    let parts: Vec<&str> = node.tracker_id.split("::").collect();
    if parts.len() > 2 {
        // Include the module path as context prefix for BM25/semantic search.
        let module_context = parts[..parts.len().saturating_sub(1)].join("::");
        format!("[module: {}]\n{}", module_context, base)
    } else {
        base
    }
}

fn tokenize(text: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current = String::new();
    let mut previous_lowercase = false;

    let flush = |current: &mut String, terms: &mut Vec<String>| {
        if !current.is_empty() {
            terms.push(std::mem::take(current).to_lowercase());
        }
    };

    for character in text.chars() {
        if !character.is_alphanumeric() {
            flush(&mut current, &mut terms);
            previous_lowercase = false;
            continue;
        }
        if character.is_uppercase() && previous_lowercase {
            flush(&mut current, &mut terms);
        }
        previous_lowercase = character.is_lowercase();
        current.push(character);
    }
    flush(&mut current, &mut terms);
    terms
}

fn embed_text(text: &str) -> Vec<f64> {
    let mut vector = vec![0.0; EMBEDDING_DIMENSIONS];
    for term in tokenize(text) {
        add_feature(&mut vector, &term, 1.0);
        let characters: Vec<char> = term.chars().collect();
        for trigram in characters.windows(3) {
            let feature: String = trigram.iter().collect();
            add_feature(&mut vector, &feature, 0.2);
        }
    }
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

/// Double-hashing feature insertion eliminates systematic bias from
/// single-hash collision patterns. The second hash (h2) provides an
/// independent offset so that features with colliding h1 indices don't
/// always collide on the same dimension.
fn add_feature(vector: &mut [f64], feature: &str, weight: f64) {
    let mut hasher1 = std::collections::hash_map::DefaultHasher::new();
    feature.hash(&mut hasher1);
    let h1 = hasher1.finish();

    let mut hasher2 = std::collections::hash_map::DefaultHasher::new();
    // Offset the second hash with a different seed to ensure independence.
    0x9e3779b97f4a7c15u64.hash(&mut hasher2);
    feature.hash(&mut hasher2);
    let h2 = hasher2.finish();

    let dim = vector.len();
    let index = (h1 as usize) % dim;
    let sign = if h1 & (1 << 63) == 0 { 1.0 } else { -1.0 };
    // Secondary dimension with h2-offset breaks systematic collision
    // patterns between features that share the same primary index.
    let index2 = ((h1 as usize).wrapping_add(h2 as usize)) % dim;
    let sign2 = if h2 & (1 << 63) == 0 { 1.0 } else { -1.0 };

    vector[index] += sign * weight;
    vector[index2] += sign2 * weight * 0.5;
}

fn cosine_similarity(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn sort_results(results: &mut [SearchResult]) {
    results.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.node_id.cmp(&right.node_id))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_splits_snake_and_camel_case() {
        assert_eq!(
            tokenize("AutocompleteRequest search_engine"),
            vec!["autocomplete", "request", "search", "engine"]
        );
    }

    #[test]
    fn identical_text_has_unit_cosine() {
        let vector = embed_text("personalized page rank");
        assert!((cosine_similarity(&vector, &vector) - 1.0).abs() < 1e-12);
    }
}
