//! Hybrid Retrieval Matrix: AST-weighted lexical × dense vectors × PPR ×
//! Leiden cluster proximity, fused by a 4-stream Reciprocal Rank Fusion loop.
//!
//! Four deliberately independent ranking passes run per query:
//!
//! 1. **Stream A — AST-weighted positional lexical rank.** An in-memory
//!    inverted index maps source tokens to parent AST chunks with
//!    tree-sitter-derived role weights (signatures/structs 3.0×, parameters
//!    and type names 2.0×, body logic 1.0×). When several query keywords hit
//!    one chunk, the literal line distance between the hits applies an
//!    exponential co-location amplifier.
//! 2. **Stream B — dense semantic similarity.** Local neural embeddings
//!    (`BAAI/bge-small-en-v1.5` via `fastembed`) ranked by a hand-written
//!    cosine kernel — no vector database.
//! 3. **Stream C — Personalized PageRank** over the ASG, teleported from the
//!    query candidates and the editor cursor node.
//! 4. **Stream D — Leiden cluster proximity.** The workspace partition from
//!    our custom Leiden implementation yields a deterministic cohesion vector
//!    around the cursor node's community.
//!
//! Fusion uses the canonical RRF formula `Score(c) = Σ 1/(k + r_m(c))` with
//! `k = 60`, plus an ASG-topology tie-break. No source code leaves the
//! process.

pub mod rrf;

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock as AsyncRwLock;
use tree_sitter::{Node as TsNode, Parser};

use crate::asg::{
    leiden::{detect_communities, CommunityStructure, LeidenConfig},
    EdgeKind, Node, PageRankEngine, PersonalizedPageRankConfig, PprIndex, SharedAsg,
};
use crate::compressor::ChunkRegistry;
use crate::recency::RecencyTracker;
use rrf::{RankedDoc, RrfConfig};

/// Stream order: `[lexical, semantic, structural, leiden]`.
const STREAM_NAMES: [&str; 4] = ["lexical", "semantic", "structural", "leiden"];

/// Lightweight local embedding model served entirely in-process.
const EMBEDDING_MODEL: fastembed::EmbeddingModel = fastembed::EmbeddingModel::BGESmallENV15;
/// Output dimensionality of `BAAI/bge-small-en-v1.5`.
const EMBEDDING_DIMENSIONS: usize = 384;
/// Canonical RRF rank offset (spec constant).
const RRF_K: f64 = 60.0;

// ---------------------------------------------------------------------------
// Configuration and result types
// ---------------------------------------------------------------------------

/// Tuning knobs for the four retrieval streams and RRF fusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    /// Candidates generated per requested result before fusion.
    pub candidate_multiplier: usize,
    /// Ignore semantic candidates below this cosine similarity.
    pub semantic_min_score: f64,
    /// Number of lexical/semantic candidates used as PPR teleport seeds.
    pub ppr_seed_candidates: usize,
    /// Relative weight of the editor's current ASG node in the teleport vector.
    pub context_seed_weight: f64,
    /// Weight multiplier for tokens in function signatures and struct
    /// declarations (spec: 3.0×).
    pub signature_token_weight: f64,
    /// Weight multiplier for variable parameters and custom type names
    /// (spec: 2.0×).
    pub type_token_weight: f64,
    /// Weight multiplier for internal body logic and block expressions
    /// (spec: 1.0×).
    pub body_token_weight: f64,
    /// Exponential line-proximity scale: keyword hits `d` lines apart amplify
    /// the chunk score by `exp(-d / proximity_line_scale)`.
    pub proximity_line_scale: f64,
    pub ppr: PersonalizedPageRankConfig,
    pub leiden: LeidenConfig,
    /// Per-stream RRF multipliers ordered `[lexical, semantic, structural,
    /// leiden]`. Uniform weights reproduce the canonical formula exactly.
    pub rrf: RrfConfig,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            candidate_multiplier: 4,
            semantic_min_score: 0.03,
            ppr_seed_candidates: 8,
            context_seed_weight: 4.0,
            signature_token_weight: 3.0,
            type_token_weight: 2.0,
            body_token_weight: 1.0,
            proximity_line_scale: 6.0,
            ppr: PersonalizedPageRankConfig::default(),
            leiden: LeidenConfig::default(),
            rrf: RrfConfig {
                k: RRF_K,
                weights: vec![1.0, 1.0, 1.0, 1.0],
                score_alpha: 0.0,
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
// Stream A: AST-weighted lexical inverted index
// ---------------------------------------------------------------------------

/// One token occurrence inside a parent AST chunk.
#[derive(Debug, Clone)]
pub struct ChunkOccurrence {
    /// Dense ASG node id of the owning chunk.
    pub node_id: usize,
    /// Zero-indexed line of the occurrence within the chunk.
    pub line: usize,
    /// Deterministic role weight (signature 3.0 / type 2.0 / body 1.0).
    pub weight: f64,
}

/// In-memory inverted index: source token -> occurrences in AST chunks.
#[derive(Debug, Default)]
pub struct AstLexicalIndex {
    inverted: HashMap<String, Vec<ChunkOccurrence>>,
}

impl AstLexicalIndex {
    /// Build the index over every searchable chunk, weighting tokens by their
    /// tree-sitter syntax-node role.
    fn build(asg: &crate::asg::Asg, registry: &ChunkRegistry, config: &SearchConfig) -> Self {
        let mut index = Self::default();
        let mut parser = Parser::new();
        if parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .is_err()
        {
            return index;
        }
        for node in &asg.nodes {
            if !registry.chunks.contains_key(&node.id) {
                continue;
            }
            let Some(tree) = parser.parse(node.source.as_bytes(), None) else {
                continue;
            };
            let mut tokens = Vec::new();
            collect_weighted_tokens(
                tree.root_node(),
                &node.source,
                config.body_token_weight,
                config.signature_token_weight,
                config.type_token_weight,
                &mut tokens,
            );
            for (term, weight, line) in tokens {
                index
                    .inverted
                    .entry(term)
                    .or_default()
                    .push(ChunkOccurrence {
                        node_id: node.id,
                        line,
                        weight,
                    });
            }
        }
        index
    }

    /// Score chunks for `query`: weighted term hits amplified exponentially by
    /// the literal line distance between multiple keyword hits in one chunk.
    fn search(
        &self,
        query_terms: &[String],
        pool_size: usize,
        proximity_line_scale: f64,
    ) -> Vec<SearchResult> {
        if query_terms.is_empty() {
            return Vec::new();
        }

        // node_id -> (accumulated weight, hit lines).
        let mut hits: HashMap<usize, (f64, Vec<usize>)> = HashMap::new();
        let mut seen_terms: HashSet<&str> = HashSet::new();
        for term in query_terms {
            if !seen_terms.insert(term.as_str()) {
                continue;
            }
            if let Some(occurrences) = self.inverted.get(term) {
                for occurrence in occurrences {
                    let entry = hits.entry(occurrence.node_id).or_default();
                    entry.0 += occurrence.weight;
                    entry.1.push(occurrence.line);
                }
            }
        }

        let scale = if proximity_line_scale.is_finite() && proximity_line_scale > 0.0 {
            proximity_line_scale
        } else {
            6.0
        };

        let mut results: Vec<SearchResult> = hits
            .into_iter()
            .map(|(node_id, (mut score, mut lines))| {
                if lines.len() > 1 {
                    lines.sort_unstable();
                    // Literal mathematical distance between the first and last
                    // keyword hit; shorter spans scale the score exponentially.
                    let span = (lines[lines.len() - 1].saturating_sub(lines[0])) as f64;
                    score *= (-span / scale).exp();
                }
                SearchResult {
                    node_id,
                    score,
                    source: STREAM_NAMES[0],
                }
            })
            .filter(|result| result.score.is_finite() && result.score > 0.0)
            .collect();

        sort_results(&mut results);
        results.truncate(pool_size);
        results
    }
}

/// Tree-sitter syntax-role weight for a node kind.
///
/// Function signatures and struct/trait/enum declarations earn the signature
/// weight; variable parameters and custom type names earn the type weight;
/// everything else (internal body logic, block expressions) stays at the
/// inherited baseline.
fn syntax_role_weight(kind: &str, signature: f64, type_weight: f64, body: f64) -> f64 {
    match kind {
        "function_item" | "function_signature_item" | "struct_item" | "enum_item"
        | "trait_item" | "impl_item" | "type_item" => signature,
        "parameter" | "parameters" | "type_identifier" | "scoped_type_identifier"
        | "generic_type" | "field_declaration" | "enum_variant" | "where_predicate" => {
            type_weight
        }
        _ => body,
    }
}

const IDENTIFIER_KINDS: [&str; 5] = [
    "identifier",
    "field_identifier",
    "type_identifier",
    "shorthand_field_identifier",
    "property_identifier",
];

/// Recursively collect `(lowercase token, weight, line)` triples.
///
/// The carried weight is the maximum role weight along the ancestor path,
/// except that body `block`s reset to the baseline so a function's signature
/// boost never leaks into its internal logic.
fn collect_weighted_tokens(
    node: TsNode<'_>,
    source: &str,
    inherited: f64,
    signature: f64,
    type_weight: f64,
    out: &mut Vec<(String, f64, usize)>,
) {
    let kind = node.kind();
    let level = if kind == "block" {
        inherited.min(1.0_f64.max(f64::MIN_POSITIVE))
    } else {
        syntax_role_weight(kind, signature, type_weight, 1.0).max(inherited)
    };

    if IDENTIFIER_KINDS.contains(&kind) {
        if let Ok(text) = node.utf8_text(source.as_bytes()) {
            out.push((text.to_lowercase(), level, node.start_position().row));
        }
        return;
    }

    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        collect_weighted_tokens(child, source, level, signature, type_weight, out);
    }
}

// ---------------------------------------------------------------------------
// Dense vector helpers
// ---------------------------------------------------------------------------

/// Hyper-fast cosine similarity over raw `f32` slices. Single pass, no
/// allocations, no normalization assumptions beyond non-zero magnitudes.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let (mut dot, mut norm_a, mut norm_b) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..len {
        let x = a[i];
        let y = b[i];
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denominator = norm_a.sqrt() * norm_b.sqrt();
    if denominator > f32::MIN_POSITIVE {
        dot / denominator
    } else {
        0.0
    }
}

/// Deterministic offline fallback embedding (double-hashed bag of features).
/// Used only when the neural model cannot be initialized; keeps Stream B
/// functional with degraded semantics.
fn embed_text_fallback(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0f32; EMBEDDING_DIMENSIONS];
    for term in tokenize(text) {
        add_feature(&mut vector, &term, 1.0);
        let characters: Vec<char> = term.chars().collect();
        for trigram in characters.windows(3) {
            let feature: String = trigram.iter().collect();
            add_feature(&mut vector, &feature, 0.2);
        }
    }
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > f32::MIN_POSITIVE {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

/// Double-hashing feature insertion eliminates systematic bias from
/// single-hash collision patterns.
fn add_feature(vector: &mut [f32], feature: &str, weight: f32) {
    let mut hasher1 = std::collections::hash_map::DefaultHasher::new();
    feature.hash(&mut hasher1);
    let h1 = hasher1.finish();

    let mut hasher2 = std::collections::hash_map::DefaultHasher::new();
    0x9e3779b97f4a7c15u64.hash(&mut hasher2);
    feature.hash(&mut hasher2);
    let h2 = hasher2.finish();

    let dim = vector.len();
    let index = (h1 as usize) % dim;
    let sign = if h1 & (1 << 63) == 0 { 1.0 } else { -1.0 };
    let index2 = ((h1 as usize).wrapping_add(h2 as usize)) % dim;
    let sign2 = if h2 & (1 << 63) == 0 { 1.0 } else { -1.0 };

    vector[index] += sign * weight;
    vector[index2] += sign2 * weight * 0.5;
}

// ---------------------------------------------------------------------------
// Search engine
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct SearchEngine {
    asg: SharedAsg,
    registry: Arc<ChunkRegistry>,
    config: SearchConfig,
    /// Stream A: token -> chunk occurrences with AST role weights.
    lexical_index: Arc<AsyncRwLock<AstLexicalIndex>>,
    /// Stream B: dense node embeddings (model-space `f32` vectors).
    embeddings: Arc<AsyncRwLock<HashMap<usize, Vec<f32>>>>,
    /// The local neural embedder held directly in runtime server state.
    embedder: Arc<RwLock<Option<fastembed::TextEmbedding>>>,
    /// Stream D: Leiden community assignment over the workspace.
    communities: Arc<AsyncRwLock<Option<CommunityStructure>>>,
    /// Stream C: precomputed PPR adjacency for the current graph snapshot.
    /// Built once per (re)index instead of per query — the old path rebuilt
    /// the full O(E) adjacency on every keystroke.
    ppr_index: Arc<RwLock<PprIndex>>,
    /// Sliding-window recency tracker that boosts recently touched nodes in
    /// the PPR teleport vector.
    recency: Arc<RecencyTracker>,
}

impl SearchEngine {
    pub fn new(asg: SharedAsg, registry: ChunkRegistry) -> Self {
        Self::with_config(asg, registry, SearchConfig::default())
    }

    pub fn with_config(asg: SharedAsg, registry: ChunkRegistry, config: SearchConfig) -> Self {
        let ppr_index = PprIndex::build(&asg.inner, &config.ppr.edge_weights);
        Self {
            asg,
            registry: Arc::new(registry),
            config,
            lexical_index: Arc::new(AsyncRwLock::new(AstLexicalIndex::default())),
            embeddings: Arc::new(AsyncRwLock::new(HashMap::new())),
            embedder: Arc::new(RwLock::new(None)),
            communities: Arc::new(AsyncRwLock::new(None)),
            ppr_index: Arc::new(RwLock::new(ppr_index)),
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

    /// Initialize the local `fastembed` model (downloads weights on first run,
    /// cached on disk afterwards). Safe to call repeatedly.
    pub async fn init_embedder(&self) -> bool {
        if self.embedder.read().unwrap().is_some() {
            return true;
        }
        let slot = self.embedder.clone();
        tokio::task::spawn_blocking(move || {
            let options = fastembed::InitOptions::new(EMBEDDING_MODEL)
                .with_show_download_progress(false)
                .with_cache_dir(fastembed_cache_dir());
            match fastembed::TextEmbedding::try_new(options) {
                Ok(model) => {
                    *slot.write().unwrap() = Some(model);
                    true
                }
                Err(error) => {
                    tracing::warn!(
                        "fastembed init failed, falling back to local features: {error}"
                    );
                    false
                }
            }
        })
        .await
        .unwrap_or(false)
    }

    /// Embed a batch of texts through the live model, falling back to the
    /// deterministic local embedder when the model is unavailable.
    async fn embed_batch(&self, texts: Vec<String>) -> Vec<Vec<f32>> {
        if texts.is_empty() {
            return Vec::new();
        }
        let slot = self.embedder.clone();
        let count = texts.len();
        let owned = texts.clone();
        let neural = tokio::task::spawn_blocking(move || {
            let guard = slot.read().unwrap();
            guard.as_ref().map(|model| model.embed(owned, None))
        })
        .await;

        match neural {
            Ok(Some(Ok(vectors))) if vectors.len() == count => vectors,
            _ => texts
                .iter()
                .map(|text| embed_text_fallback(text))
                .collect(),
        }
    }

    /// Precompute every retrieval structure once: AST-weighted lexical index,
    /// dense embeddings, and the Leiden community partition.
    pub async fn precompute_embeddings(&self) {
        self.init_embedder().await;

        let lexical = AstLexicalIndex::build(&self.asg.inner, &self.registry, &self.config);
        *self.lexical_index.write().await = lexical;

        let texts: Vec<(usize, String)> = self
            .asg
            .inner
            .nodes
            .iter()
            .filter(|node| self.is_searchable(node.id))
            .map(|node| (node.id, contextual_text(node, &self.asg.inner)))
            .collect();
        let vectors = self
            .embed_batch(texts.iter().map(|(_, text)| text.clone()).collect())
            .await;
        let mut embeddings: HashMap<usize, Vec<f32>> = HashMap::with_capacity(texts.len());
        for ((node_id, _), vector) in texts.into_iter().zip(vectors) {
            embeddings.insert(node_id, vector);
        }
        *self.embeddings.write().await = embeddings;

        let asg = self.asg.clone();
        let leiden_config = self.config.leiden.clone();
        let structure = tokio::task::spawn_blocking(move || {
            detect_communities(&asg.inner, &leiden_config)
        })
        .await
        .unwrap_or_else(|_| CommunityStructure {
            assignment: Vec::new(),
            community_count: 0,
        });
        *self.communities.write().await = Some(structure);

        // Refresh the cached PPR adjacency so Stream C matches the same
        // graph snapshot the lexical index, embeddings, and communities
        // were just built from.
        *self.ppr_index.write().unwrap() =
            PprIndex::build(&self.asg.inner, &self.config.ppr.edge_weights);
    }

    /// Search without editor context.
    pub async fn search(&self, query: &str, top_k: usize) -> Vec<MergedResult> {
        self.search_with_context(query, top_k, None).await
    }

    /// Run the four ranking passes concurrently, then merge them with the
    /// canonical Reciprocal Rank Fusion formula (`k = 60`).
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

        let pool_size = top_k
            .saturating_mul(self.config.candidate_multiplier.max(1))
            .max(top_k)
            .min(self.asg.inner.nodes.len());
        let query_terms = tokenize(query);

        // ---- Streams A + B run in parallel tasks --------------------------
        let lexical_index = self.lexical_index.clone();
        let lexical_query = query_terms.clone();
        let lexical_pool = pool_size;
        let lexical_scale = self.config.proximity_line_scale;
        let lexical_task = tokio::spawn(async move {
            let index = lexical_index.read().await;
            index.search(&lexical_query, lexical_pool, lexical_scale)
        });

        let vector_engine = self.clone();
        let vector_query = query.to_owned();
        let vector_floor = self.config.semantic_min_score.max(0.0) as f32;
        let vector_pool = pool_size;
        let semantic_task = tokio::spawn(async move {
            vector_engine
                .semantic_search(&vector_query, vector_floor, vector_pool)
                .await
        });

        let lexical = lexical_task.await.unwrap_or_default();
        let semantic = semantic_task.await.unwrap_or_default();

        // ---- Stream C: PPR teleported from query candidates + cursor ------
        let seeds = self.ppr_seeds(&lexical, &semantic, context_node);
        let ppr_registry = self.registry.clone();
        let ppr_config = self.config.ppr.clone();
        let ppr_index = self.ppr_index.clone();
        let structural = tokio::task::spawn_blocking(move || {
            Self::structural_search(ppr_index, ppr_registry, ppr_config, seeds, pool_size)
        })
        .await
        .unwrap_or_default();

        // ---- Stream D: Leiden cluster proximity ---------------------------
        let leiden_anchor =
            context_node.or_else(|| {
                lexical
                    .first()
                    .or_else(|| semantic.first())
                    .map(|result| result.node_id)
            });
        let leiden = self.leiden_proximity(leiden_anchor, pool_size).await;

        self.fuse(lexical, semantic, structural, leiden, top_k)
    }

    fn is_searchable(&self, node_id: usize) -> bool {
        self.registry.chunks.contains_key(&node_id)
    }

    // -----------------------------------------------------------------------
    // Stream B: dense vector semantic search
    // -----------------------------------------------------------------------

    async fn semantic_search(
        &self,
        query: &str,
        minimum_score: f32,
        top_k: usize,
    ) -> Vec<SearchResult> {
        let mut query_vector = self.embed_batch(vec![query.to_owned()]).await;
        let Some(query_vector) = query_vector.pop() else {
            return Vec::new();
        };

        let embeddings = self.embeddings.read().await;
        let mut results = Vec::new();
        for (&node_id, node_vector) in embeddings.iter() {
            if !self.registry.chunks.contains_key(&node_id) {
                continue;
            }
            let score = cosine_similarity(&query_vector, node_vector) as f64;
            if score.is_finite() && score >= minimum_score as f64 {
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
    // Stream C: Personalized PageRank
    // -----------------------------------------------------------------------

    fn ppr_seeds(
        &self,
        lexical: &[SearchResult],
        semantic: &[SearchResult],
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
        for stream in [lexical, semantic] {
            for (rank, result) in stream.iter().take(limit).enumerate() {
                // Rank-normalized seed weights avoid mixing incomparable
                // cosine and lexical score scales before PPR.
                *seeds.entry(result.node_id).or_default() += 1.0 / (rank + 1) as f64;
            }
        }
        let mut seeds: Vec<(usize, f64)> = seeds.into_iter().collect();
        seeds.sort_by_key(|(id, _)| *id);
        seeds
    }

    fn structural_search(
        ppr_index: Arc<RwLock<PprIndex>>,
        registry: Arc<ChunkRegistry>,
        config: PersonalizedPageRankConfig,
        seeds: Vec<(usize, f64)>,
        top_k: usize,
    ) -> Vec<SearchResult> {
        let engine = PageRankEngine::from_config(config);
        let index = ppr_index.read().unwrap();
        let scores = engine.personalized_scores_on(&index, &seeds);

        let mut results: Vec<SearchResult> = scores
            .into_iter()
            .enumerate()
            .filter(|(node_id, score)| *score > 0.0 && registry.chunks.contains_key(node_id))
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
    // Stream D: Leiden cluster proximity
    // -----------------------------------------------------------------------

    /// Deterministic proximity vector: chunks inside the anchor's community
    /// score 1.0; neighbouring communities score proportionally to normalized
    /// inter-community edge cohesion; everything else scores 0.0.
    async fn leiden_proximity(
        &self,
        anchor_node: Option<usize>,
        top_k: usize,
    ) -> Vec<SearchResult> {
        let communities = self.communities.read().await;
        let Some(structure) = communities.as_ref() else {
            return Vec::new();
        };
        let Some(anchor) = anchor_node.and_then(|id| structure.community_of(id)) else {
            return Vec::new();
        };

        let scores = structure.proximity_scores(&self.asg.inner, anchor);
        let mut results: Vec<SearchResult> = scores
            .into_iter()
            .enumerate()
            .filter(|(node_id, score)| *score > 0.0 && self.is_searchable(*node_id))
            .map(|(node_id, score)| SearchResult {
                node_id,
                score,
                source: STREAM_NAMES[3],
            })
            .collect();
        sort_results(&mut results);
        results.truncate(top_k);
        results
    }

    // -----------------------------------------------------------------------
    // ASG structural dependency scores (RRF tie-breaking)
    // -----------------------------------------------------------------------

    /// Weighted incoming `calls` + `references` edges — the two edge kinds
    /// that carry genuine structural signal — used to break exact RRF ties.
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
    // 4-stream Reciprocal Rank Fusion
    // -----------------------------------------------------------------------

    fn fuse(
        &self,
        lexical: Vec<SearchResult>,
        semantic: Vec<SearchResult>,
        structural: Vec<SearchResult>,
        leiden: Vec<SearchResult>,
        top_k: usize,
    ) -> Vec<MergedResult> {
        let original_streams = [&lexical, &semantic, &structural, &leiden];
        let ranked_streams: Vec<Vec<RankedDoc<usize>>> = original_streams
            .iter()
            .map(|stream| {
                stream
                    .iter()
                    .map(|result| RankedDoc::new(result.node_id, result.score))
                    .collect()
            })
            .collect();

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

fn sort_results(results: &mut [SearchResult]) {
    results.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.node_id.cmp(&right.node_id))
    });
}

// ---------------------------------------------------------------------------
// Text features
// ---------------------------------------------------------------------------

fn searchable_text(node: &Node) -> String {
    format!(
        "{} {} {}\n{}\n{}",
        node.tracker_id, node.name, node.kind, node.source, node.source
    )
}

/// Contextual text wraps searchable_text with module context from the
/// tracker_id hierarchy so both streams understand what module a chunk
/// belongs to.
fn contextual_text(node: &Node, _asg: &crate::asg::Asg) -> String {
    let base = searchable_text(node);
    let parts: Vec<&str> = node.tracker_id.split("::").collect();
    if parts.len() > 2 {
        let module_context = parts[..parts.len().saturating_sub(1)].join("::");
        format!("[module: {}]\n{}", module_context, base)
    } else {
        base
    }
}

/// Split text into lowercase sub-word terms (camelCase and snake_case aware).
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

/// Model cache directory so the embedding weights download exactly once.
fn fastembed_cache_dir() -> PathBuf {
    std::env::var_os("TOKEN_SAVER_EMBED_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("token-saver-fastembed"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_basics() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn tokenizer_splits_snake_and_camel_case() {
        assert_eq!(
            tokenize("AutocompleteRequest search_engine"),
            vec!["autocomplete", "request", "search", "engine"]
        );
    }

    #[test]
    fn fallback_embeddings_are_unit_length_and_deterministic() {
        let a = embed_text_fallback("fn retrieve_chunks(query)");
        let b = embed_text_fallback("fn retrieve_chunks(query)");
        assert_eq!(a, b);
        let norm: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }
}
