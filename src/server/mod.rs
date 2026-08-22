//! Low-latency HTTP/SSE server for token-budgeted code retrieval.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use tower_http::cors::{Any, CorsLayer};
use tower_http::cors::AllowOrigin;
use tracing::{debug, info, warn};

use crate::asg::{Asg, SharedAsg};
use crate::ast::{AstChunker, MerkleTree, TrigramIndex};
use crate::compressor::ChunkRegistry;
use crate::config::TokenSaverConfig;
use crate::memory::MemoryStore;
use crate::terminal::{ExecuteRequest, ExecuteResponse};
use crate::persistence::PersistentStore;
use crate::search::SearchEngine;
use crate::tracker::{ContextTracker, CursorPayload};
use crate::watcher::reindex::SharedIndexes;

// ---------------------------------------------------------------------------
// API types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutocompleteRequest {
    pub file_path: String,
    pub line: usize,
    pub column: usize,
    pub query: String,
    pub request_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChunk {
    pub text: String,
    pub done: bool,
    pub node_id: Option<usize>,
    pub pagerank: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    pub context_node_id: Option<usize>,
}

fn default_top_k() -> usize {
    10
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub node_id: usize,
    pub tracker_id: String,
    pub name: String,
    pub kind: String,
    pub file_path: String,
    pub pagerank: f64,
    pub rrf_score: f64,
    pub scores: HashMap<&'static str, f64>,
    pub rrf_contributions: HashMap<&'static str, f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResponse {
    pub query: String,
    pub hits: Vec<SearchHit>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub workspace: String,
    pub asg_nodes: usize,
    pub asg_edges: usize,
    pub compressed_chunks: usize,
    pub ast_chunks: usize,
    pub merkle_root: Option<String>,
    pub memory_count: usize,
}

// Memory API types

#[derive(Debug, Clone, Deserialize)]
pub struct SaveMemoryRequest {
    pub content: String,
    pub namespace: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SaveMemoryResponse {
    pub id: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RecallMemoryRequest {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecallMemoryResponse {
    pub memories: Vec<crate::memory::Memory>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ListMemoryResponse {
    pub memories: Vec<crate::memory::Memory>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ListMemoryQuery {
    pub namespace: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ForgetMemoryResponse {
    pub forgotten: bool,
}

// ---------------------------------------------------------------------------
// Simple in-memory rate limiter (token bucket per IP)
// ---------------------------------------------------------------------------

/// A lightweight token-bucket rate limiter keyed by IP address.
/// Each IP gets `max_tokens` tokens that refill at `refill_per_sec` rate.
#[derive(Clone)]
pub struct RateLimiter {
    buckets: Arc<tokio::sync::Mutex<HashMap<String, RateBucket>>>,
    max_tokens: u32,
    refill_per_sec: u32,
}

struct RateBucket {
    tokens: f64,
    last_refill: std::time::Instant,
}

impl RateLimiter {
    pub fn new(max_tokens: u32, refill_per_sec: u32) -> Self {
        Self {
            buckets: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            max_tokens: max_tokens.max(1),
            refill_per_sec: refill_per_sec.max(1),
        }
    }

    /// Check if a request from the given key is allowed.
    /// Returns true if the request is allowed, false if rate-limited.
    pub async fn allow(&self, key: &str) -> bool {
        let mut buckets = self.buckets.lock().await;
        let now = std::time::Instant::now();
        let bucket = buckets.entry(key.to_string()).or_insert_with(|| RateBucket {
            tokens: self.max_tokens as f64,
            last_refill: now,
        });

        // Refill tokens based on elapsed time.
        let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * self.refill_per_sec as f64)
            .min(self.max_tokens as f64);
        bucket.last_refill = now;

        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ActiveRequest {
    generation: u64,
    token: CancellationToken,
}

#[derive(Clone)]
pub struct ServerState {
    pub asg: SharedAsg,
    pub registry: Arc<ChunkRegistry>,
    pub search_engine: Arc<SearchEngine>,
    pub tracker: Arc<ContextTracker>,
    pub indexes: SharedIndexes,
    pub active_requests: Arc<tokio::sync::Mutex<HashMap<String, ActiveRequest>>>,
    pub request_counter: Arc<AtomicU64>,
    pub workspace: Arc<std::path::PathBuf>,
    pub debounce_ms: u64,
    pub memory_store: Arc<tokio::sync::Mutex<MemoryStore>>,
    pub persistent_store: Option<Arc<PersistentStore>>,
    pub started_at: std::time::Instant,
    pub rate_limiter: RateLimiter,
    /// Total HTTP requests served (all endpoints).
    pub total_requests: Arc<AtomicU64>,
    /// Total tokens estimated across all context/retrieval responses.
    pub total_tokens_served: Arc<AtomicU64>,
}

impl ServerState {
    /// Backwards-compatible state constructor using default runtime settings.
    pub fn new(
        asg: Asg,
        registry: ChunkRegistry,
        search_engine: SearchEngine,
        ast_trigram: TrigramIndex,
        ast_merkle: MerkleTree,
    ) -> Self {
        let config = TokenSaverConfig::default();
        let workspace = std::env::current_dir().unwrap_or_else(|_| ".".into());
        Self::with_config(
            asg,
            registry,
            search_engine,
            ast_trigram,
            ast_merkle,
            workspace,
            &config,
        )
    }

    pub fn with_config(
        asg: Asg,
        registry: ChunkRegistry,
        search_engine: SearchEngine,
        ast_trigram: TrigramIndex,
        ast_merkle: MerkleTree,
        workspace: std::path::PathBuf,
        config: &TokenSaverConfig,
    ) -> Self {
        let shared_asg = SharedAsg::new(asg);
        let tracker = Arc::new(ContextTracker::with_config(
            shared_asg.clone(),
            registry.clone(),
            search_engine.clone(),
            workspace.clone(),
            config.context.clone(),
        ));

        Self {
            asg: shared_asg,
            registry: Arc::new(registry),
            search_engine: Arc::new(search_engine),
            tracker,
            indexes: SharedIndexes::new(ast_merkle, ast_trigram),
            active_requests: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            request_counter: Arc::new(AtomicU64::new(0)),
            workspace: Arc::new(workspace),
            debounce_ms: config.debounce_ms,
            memory_store: Arc::new(tokio::sync::Mutex::new(MemoryStore::new())),
            persistent_store: None,
            started_at: std::time::Instant::now(),
            rate_limiter: RateLimiter::new(60, 10), // 60 burst, 10/sec refill
            total_requests: Arc::new(AtomicU64::new(0)),
            total_tokens_served: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn next_request_id(&self) -> u64 {
        self.request_counter.fetch_add(1, Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Generation-safe debouncing
// ---------------------------------------------------------------------------

/// Debounces by file path. A generation number prevents an older cancelled
/// request from unregistering the newer request that replaced it.
pub struct RequestDebouncer {
    active_requests: Arc<tokio::sync::Mutex<HashMap<String, ActiveRequest>>>,
}

impl RequestDebouncer {
    pub fn new(
        active_requests: Arc<tokio::sync::Mutex<HashMap<String, ActiveRequest>>>,
    ) -> Self {
        Self { active_requests }
    }

    pub async fn register(&self, file_path: &str, generation: u64) -> CancellationToken {
        let token = CancellationToken::new();
        let active = ActiveRequest {
            generation,
            token: token.clone(),
        };
        let mut requests = self.active_requests.lock().await;
        if let Some(previous) = requests.insert(file_path.to_string(), active) {
            previous.token.cancel();
            debug!("cancelled previous request for {file_path}");
        }
        token
    }

    pub async fn unregister(&self, file_path: &str, generation: u64) {
        let mut requests = self.active_requests.lock().await;
        if requests
            .get(file_path)
            .map(|active| active.generation == generation)
            .unwrap_or(false)
        {
            requests.remove(file_path);
        }
    }
}

// ---------------------------------------------------------------------------
// SSE response
// ---------------------------------------------------------------------------

pub struct SseStream {
    rx: tokio::sync::mpsc::Receiver<CompletionChunk>,
}

impl SseStream {
    pub fn new(rx: tokio::sync::mpsc::Receiver<CompletionChunk>) -> Self {
        Self { rx }
    }
}

impl IntoResponse for SseStream {
    fn into_response(self) -> Response {
        // The boolean ensures channel closure emits exactly one final event.
        // The old stream yielded a final event forever and never closed.
        let stream = futures::stream::unfold((self.rx, false), |(mut rx, finished)| async move {
            if finished {
                return None;
            }
            match rx.recv().await {
                Some(chunk) => {
                    let is_done = chunk.done;
                    let json = serde_json::to_string(&chunk).unwrap_or_default();
                    let frame = format!("data: {}\n\n", json);
                    Some((
                        Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(frame)),
                        (rx, is_done),
                    ))
                }
                None => {
                    let final_event = "data: {\"text\":\"\",\"done\":true,\"node_id\":null,\"pagerank\":null}\n\n";
                    Some((
                        Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(final_event)),
                        (rx, true),
                    ))
                }
            }
        });

        Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("Connection", "keep-alive")
            .header("X-Accel-Buffering", "no")
            .body(axum::body::Body::from_stream(stream))
            .expect("valid SSE response")
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

pub async fn autocomplete_handler(
    State(state): State<ServerState>,
    Json(request): Json<AutocompleteRequest>,
) -> Response {
    let generation = state.next_request_id();
    let request_id = request
        .request_id
        .clone()
        .unwrap_or_else(|| format!("req-{generation}"));
    info!(
        "autocomplete {request_id} for {}:{}:{}",
        request.file_path, request.line, request.column
    );

    let debouncer = RequestDebouncer::new(state.active_requests.clone());
    let cancel_token = debouncer
        .register(&request.file_path, generation)
        .await;
    let proceed = tokio::select! {
        _ = tokio::time::sleep(Duration::from_millis(state.debounce_ms)) => true,
        _ = cancel_token.cancelled() => false,
    };
    if !proceed {
        debouncer.unregister(&request.file_path, generation).await;
        return (StatusCode::OK, "cancelled").into_response();
    }

    let cursor = CursorPayload::new(&request.file_path, request.line, request.column);
    let node_id = state.tracker.find_node_at_cursor(&cursor);
    let (node_name, tracker_id, pagerank) = node_id
        .and_then(|id| state.asg.get_node(id))
        .map(|node| {
            (
                Some(node.name.clone()),
                Some(node.tracker_id.clone()),
                Some(node.pagerank),
            )
        })
        .unwrap_or((None, None, None));

    let (tx, rx) = tokio::sync::mpsc::channel::<CompletionChunk>(32);
    let task_state = state.clone();
    let task_file = request.file_path.clone();
    let query = request.query.clone();
    let task_token = cancel_token.clone();

    tokio::spawn(async move {
        let stream_work = async {
            let header_name = tracker_id.or(node_name).unwrap_or_default();
            if tx
                .send(CompletionChunk {
                    text: format!(
                        "// node: {} (pagerank: {:.6})\n",
                        header_name,
                        pagerank.unwrap_or(0.0)
                    ),
                    done: false,
                    node_id,
                    pagerank,
                })
                .await
                .is_err()
            {
                return;
            }

            let surrounding = task_state.tracker.get_surrounding_lines(&cursor);
            if tx
                .send(CompletionChunk {
                    text: format!("{surrounding}\n"),
                    done: false,
                    node_id,
                    pagerank,
                })
                .await
                .is_err()
            {
                return;
            }

            let dependencies = task_state
                .tracker
                .get_compressed_dependencies(&cursor, &query)
                .await;
            for (index, dependency) in dependencies.iter().enumerate() {
                if task_token.is_cancelled() {
                    warn!("autocomplete generation {generation} was cancelled");
                    return;
                }
                if tx
                    .send(CompletionChunk {
                        text: format!(
                            "// --- Compressed Dependency {} ---\n{}\n",
                            index + 1,
                            dependency
                        ),
                        done: false,
                        node_id,
                        pagerank,
                    })
                    .await
                    .is_err()
                {
                    return;
                }
            }

            let _ = tx
                .send(CompletionChunk {
                    text: String::new(),
                    done: true,
                    node_id,
                    pagerank,
                })
                .await;
        };

        stream_work.await;
        RequestDebouncer::new(task_state.active_requests.clone())
            .unregister(&task_file, generation)
            .await;
    });

    SseStream::new(rx).into_response()
}

pub async fn search_handler(
    State(state): State<ServerState>,
    Json(request): Json<SearchRequest>,
) -> impl IntoResponse {
    state.total_requests.fetch_add(1, Ordering::Relaxed);

    // Rate limit check (using a placeholder key since we don't have
    // the client IP in this handler without additional extraction).
    if !state.rate_limiter.allow("global").await {
        return (StatusCode::TOO_MANY_REQUESTS, "rate limited").into_response();
    }

    let top_k = request.top_k.clamp(1, 100);
    let results = state
        .search_engine
        .search_with_context(&request.query, top_k, request.context_node_id)
        .await;

    // Estimate tokens served.
    let tokens_served: usize = results
        .iter()
        .filter_map(|r| state.asg.get_node(r.node_id))
        .map(|n| crate::tracker::estimate_tokens(&n.source))
        .sum();
    state.total_tokens_served.fetch_add(tokens_served as u64, Ordering::Relaxed);

    let hits = results
        .into_iter()
        .filter_map(|result| {
            let node = state.asg.get_node(result.node_id)?;
            Some(SearchHit {
                node_id: node.id,
                tracker_id: node.tracker_id.clone(),
                name: node.name.clone(),
                kind: node.kind.clone(),
                file_path: node.file_path.display().to_string(),
                pagerank: node.pagerank,
                rrf_score: result.rrf_score,
                scores: result.scores,
                rrf_contributions: result.rrf_contributions,
            })
        })
        .collect();
    Json(SearchResponse {
        query: request.query,
        hits,
    }).into_response()
}

pub async fn health_handler(State(state): State<ServerState>) -> impl IntoResponse {
    let trigram = state.indexes.trigram.read().await;
    let merkle = state.indexes.merkle.read().await;
    let memory_count = state.memory_store.lock().await.len();
    Json(HealthResponse {
        status: "ok",
        workspace: state.workspace.display().to_string(),
        asg_nodes: state.asg.inner.nodes.len(),
        asg_edges: state.asg.inner.edges.len(),
        compressed_chunks: state.registry.chunks.len(),
        ast_chunks: trigram.len(),
        merkle_root: merkle.root_hash(),
        memory_count,
    })
}

// ---------------------------------------------------------------------------
// Memory API handlers
// ---------------------------------------------------------------------------

pub async fn save_memory_handler(
    State(state): State<ServerState>,
    Json(request): Json<SaveMemoryRequest>,
) -> impl IntoResponse {
    let id = state.memory_store.lock().await.save(&request.content, request.namespace, None);
    Json(SaveMemoryResponse { id })
}

pub async fn recall_memory_handler(
    State(state): State<ServerState>,
    Json(request): Json<RecallMemoryRequest>,
) -> impl IntoResponse {
    let memories = state.memory_store.lock().await.recall(&request.query, request.top_k);
    Json(RecallMemoryResponse { memories })
}

pub async fn forget_memory_handler(
    State(state): State<ServerState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let forgotten = state.memory_store.lock().await.forget(&id);
    Json(ForgetMemoryResponse { forgotten })
}

pub async fn list_memory_handler(
    State(state): State<ServerState>,
    Query(query): Query<ListMemoryQuery>,
) -> impl IntoResponse {
    let memories = state.memory_store.lock().await.list(query.namespace.as_deref());
    Json(ListMemoryResponse { memories })
}

// ---------------------------------------------------------------------------
// Stats endpoint
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct StatsResponse {
    pub workspace: String,
    pub asg_nodes: usize,
    pub asg_edges: usize,
    pub asg_edge_kinds: HashMap<String, usize>,
    pub compressed_chunks: usize,
    pub bytes_saved: usize,
    pub trigram_index_size: usize,
    pub merkle_root: Option<String>,
    pub memory_count: usize,
    pub uptime_secs: f64,
}

pub async fn stats_handler(State(state): State<ServerState>) -> impl IntoResponse {
    let trigram = state.indexes.trigram.read().await;
    let merkle = state.indexes.merkle.read().await;
    let mem = state.memory_store.lock().await;

    let mut edge_kinds: HashMap<String, usize> = HashMap::new();
    for edge in &state.asg.inner.edges {
        let label = match edge.kind {
            crate::asg::EdgeKind::Calls => "calls",
            crate::asg::EdgeKind::Contains => "contains",
            crate::asg::EdgeKind::Imports => "imports",
            crate::asg::EdgeKind::References => "references",
            crate::asg::EdgeKind::Implements => "implements",
            crate::asg::EdgeKind::FieldOf => "field_of",
            crate::asg::EdgeKind::VariantOf => "variant_of",
            crate::asg::EdgeKind::Bridge => "bridge",
        };
        *edge_kinds.entry(label.to_string()).or_default() += 1;
    }

    let bytes_saved: usize = state
        .registry
        .chunks
        .iter()
        .map(|entry| entry.value().bytes_saved())
        .sum();

    Json(StatsResponse {
        workspace: state.workspace.display().to_string(),
        asg_nodes: state.asg.inner.nodes.len(),
        asg_edges: state.asg.inner.edges.len(),
        asg_edge_kinds: edge_kinds,
        compressed_chunks: state.registry.chunks.len(),
        bytes_saved,
        trigram_index_size: trigram.len(),
        merkle_root: merkle.root_hash(),
        memory_count: mem.len(),
        uptime_secs: state.started_at.elapsed().as_secs_f64(),
    })
}

// ---------------------------------------------------------------------------
// Context endpoint
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ContextRequest {
    pub file_path: String,
    pub line: usize,
    pub column: usize,
    /// Optional query for context-aware dependency retrieval.
    #[serde(default)]
    pub query: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextResponse {
    pub node_id: Option<usize>,
    pub node_name: Option<String>,
    pub pagerank: Option<f64>,
    pub surrounding_lines: String,
    pub compressed_deps: Vec<String>,
    pub total_tokens: usize,
}

pub async fn context_handler(
    State(state): State<ServerState>,
    Json(request): Json<ContextRequest>,
) -> impl IntoResponse {
    let cursor = CursorPayload::new(&request.file_path, request.line, request.column);
    let node_id = state.tracker.find_node_at_cursor(&cursor);
    let (node_name, pagerank) = node_id
        .and_then(|id| state.asg.get_node(id))
        .map(|node| (Some(node.name.clone()), Some(node.pagerank)))
        .unwrap_or((None, None));

    let surrounding = state.tracker.get_surrounding_lines(&cursor);
    let query = request.query.unwrap_or_default();
    let compressed_deps = state
        .tracker
        .get_compressed_dependencies(&cursor, &query)
        .await;

    let total_tokens = crate::tracker::estimate_tokens(&surrounding)
        + compressed_deps
            .iter()
            .map(|d| crate::tracker::estimate_tokens(d))
            .sum::<usize>();

    Json(ContextResponse {
        node_id,
        node_name,
        pagerank,
        surrounding_lines: surrounding,
        compressed_deps,
        total_tokens,
    })
}

// ---------------------------------------------------------------------------
// Graph exploration endpoint
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct GraphNodeResponse {
    pub node_id: usize,
    pub tracker_id: String,
    pub name: String,
    pub kind: String,
    pub file_path: String,
    pub pagerank: f64,
    pub range: (usize, usize),
    pub incoming_edges: Vec<GraphEdge>,
    pub outgoing_edges: Vec<GraphEdge>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphEdge {
    pub node_id: usize,
    pub node_name: String,
    pub kind: String,
}

pub async fn graph_node_handler(
    State(state): State<ServerState>,
    Path(node_id): Path<usize>,
) -> Response {
    let Some(node) = state.asg.get_node(node_id) else {
        return (StatusCode::NOT_FOUND, "node not found").into_response();
    };

    let incoming: Vec<GraphEdge> = state
        .asg
        .inner
        .reverse_adjacency
        .get(&node_id)
        .map(|edge_ids| {
            edge_ids
                .iter()
                .filter_map(|&eid| {
                    let edge = &state.asg.inner.edges[eid];
                    state
                        .asg
                        .get_node(edge.from)
                        .map(|n| GraphEdge {
                            node_id: n.id,
                            node_name: n.name.clone(),
                            kind: format!("{:?}", edge.kind),
                        })
                })
                .collect()
        })
        .unwrap_or_default();

    let outgoing: Vec<GraphEdge> = state
        .asg
        .inner
        .adjacency
        .get(&node_id)
        .map(|edge_ids| {
            edge_ids
                .iter()
                .filter_map(|&eid| {
                    let edge = &state.asg.inner.edges[eid];
                    state
                        .asg
                        .get_node(edge.to)
                        .map(|n| GraphEdge {
                            node_id: n.id,
                            node_name: n.name.clone(),
                            kind: format!("{:?}", edge.kind),
                        })
                })
                .collect()
        })
        .unwrap_or_default();

    Json(GraphNodeResponse {
        node_id: node.id,
        tracker_id: node.tracker_id.clone(),
        name: node.name.clone(),
        kind: node.kind.clone(),
        file_path: node.file_path.display().to_string(),
        pagerank: node.pagerank,
        range: node.range,
        incoming_edges: incoming,
        outgoing_edges: outgoing,
    })
    .into_response()
}

// ---------------------------------------------------------------------------
// Manual reindex endpoint
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ReindexResponse {
    pub status: &'static str,
    pub message: String,
}

pub async fn reindex_handler(
    State(state): State<ServerState>,
) -> impl IntoResponse {
    let workspace = state.workspace.clone();
    let indexes = state.indexes.clone();

    // Run a full re-chunk + index update in the background.
    tokio::spawn(async move {
        let mut chunker = match AstChunker::new() {
            Ok(c) => c.with_crate_root(workspace.as_path()),
            Err(e) => {
                warn!("reindex: failed to create chunker: {e}");
                return;
            }
        };

        let chunks = match chunker.chunk_dir(workspace.as_path()) {
            Ok(c) => c,
            Err(e) => {
                warn!("reindex: failed to chunk workspace: {e}");
                return;
            }
        };

        // Rebuild trigram index.
        let mut trigram = indexes.trigram.write().await;
        *trigram = TrigramIndex::new(3);
        trigram.index(&chunks);
        drop(trigram);

        // Rebuild merkle tree.
        let new_merkle = MerkleTree::build(&chunks);
        *indexes.merkle.write().await = new_merkle;

        info!("manual reindex complete: {} chunks", chunks.len());
    });

    Json(ReindexResponse {
        status: "ok",
        message: "reindex started in background".to_string(),
    })
}

// ---------------------------------------------------------------------------
// Update memory endpoint
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct UpdateMemoryRequest {
    pub importance: Option<f64>,
    pub namespace: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdateMemoryResponse {
    pub updated: bool,
}

pub async fn update_memory_handler(
    State(state): State<ServerState>,
    Path(id): Path<String>,
    Json(request): Json<UpdateMemoryRequest>,
) -> impl IntoResponse {
    let updated = state.memory_store.lock().await.update(
        &id,
        request.importance,
        Some(request.namespace),
    );
    Json(UpdateMemoryResponse { updated })
}

// ---------------------------------------------------------------------------
// Token-savings metrics endpoint
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct MetricsResponse {
    pub total_requests: u64,
    pub total_tokens_served: u64,
    pub compressed_chunks: usize,
    pub bytes_saved: usize,
    pub original_bytes: usize,
    pub compression_ratio: f64,
    pub uptime_secs: f64,
    pub requests_per_second: f64,
    pub memory_count: usize,
    pub asg_nodes: usize,
    pub asg_edges: usize,
}

pub async fn metrics_handler(State(state): State<ServerState>) -> impl IntoResponse {
    let total_requests = state.total_requests.load(Ordering::Relaxed);
    let total_tokens = state.total_tokens_served.load(Ordering::Relaxed);
    let uptime = state.started_at.elapsed().as_secs_f64();

    let mut bytes_saved = 0usize;
    let mut original_bytes = 0usize;
    for entry in state.registry.chunks.iter() {
        let chunk = entry.value();
        bytes_saved += chunk.bytes_saved();
        original_bytes += chunk.original_bytes;
    }

    let compression_ratio = if original_bytes > 0 {
        bytes_saved as f64 / original_bytes as f64
    } else {
        0.0
    };

    let rps = if uptime > 0.0 {
        total_requests as f64 / uptime
    } else {
        0.0
    };

    let mem = state.memory_store.lock().await;
    Json(MetricsResponse {
        total_requests,
        total_tokens_served: total_tokens,
        compressed_chunks: state.registry.chunks.len(),
        bytes_saved,
        original_bytes,
        compression_ratio,
        uptime_secs: uptime,
        requests_per_second: rps,
        memory_count: mem.len(),
        asg_nodes: state.asg.inner.nodes.len(),
        asg_edges: state.asg.inner.edges.len(),
    })
}

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

pub async fn execute_handler(
    State(state): State<ServerState>,
    Json(request): Json<ExecuteRequest>,
) -> impl IntoResponse {
    state.total_requests.fetch_add(1, Ordering::Relaxed);
    let workspace = state.workspace.as_path();
    let response: ExecuteResponse = crate::terminal::execute(workspace, request).await;
    Json(response).into_response()
}

pub fn build_router(state: ServerState, cors_origin: Option<&str>) -> Router {
    let router = Router::new()
        .route("/health", get(health_handler))
        .route("/v1/stats", get(stats_handler))
        .route("/v1/search", post(search_handler))
        .route("/v1/autocomplete", post(autocomplete_handler))
        .route("/v1/context", post(context_handler))
        .route("/v1/graph/{node_id}", get(graph_node_handler))
        .route("/v1/reindex", post(reindex_handler))
        .route("/v1/metrics", get(metrics_handler))
        .route("/v1/memories", post(save_memory_handler).get(list_memory_handler))
        .route("/v1/memories/recall", post(recall_memory_handler))
        .route("/v1/memories/{id}", delete(forget_memory_handler))
        .route("/v1/memories/{id}/update", post(update_memory_handler))
        .route("/v1/execute", post(execute_handler))
        .with_state(state);

    // Apply CORS layer if an origin is configured.
    if let Some(origin) = cors_origin {
        let allow_origin = if origin == "*" {
            AllowOrigin::any()
        } else {
            AllowOrigin::exact(
                origin.parse().unwrap_or_else(|_| {
                    axum::http::HeaderValue::from_bytes(b"*").unwrap()
                })
            )
        };
        let cors = CorsLayer::new()
            .allow_origin(allow_origin)
            .allow_methods(Any)
            .allow_headers(Any);
        router.layer(cors)
    } else {
        router
    }
}

pub async fn run_server() -> anyhow::Result<()> {
    run_server_with_config(TokenSaverConfig::load()?).await
}

pub async fn run_server_with_config(config: TokenSaverConfig) -> anyhow::Result<()> {
    let workspace = config.canonical_workspace()?;
    info!("indexing workspace {}", workspace.display());

    // Try to load from persistent store first.
    let db_path = workspace.join(".token-saver.db");
    let persistent_store = if db_path.exists() {
        match PersistentStore::open(&db_path) {
            Ok(store) => {
                info!("persistent store found at {}", db_path.display());
                Some(store)
            }
            Err(e) => {
                warn!("failed to open persistent store: {e}");
                None
            }
        }
    } else {
        None
    };

    let asg = if let Some(ref store) = persistent_store {
        match store.load_asg() {
            Ok(Some(asg)) => {
                info!("loaded ASG from persistent store: {} nodes, {} edges", asg.nodes.len(), asg.edges.len());
                asg
            }
            Ok(None) => {
                info!("no ASG in persistent store, building from source");
                build_asg(&workspace, &config)?
            }
            Err(e) => {
                warn!("failed to load ASG from persistent store: {e}");
                build_asg(&workspace, &config)?
            }
        }
    } else {
        build_asg(&workspace, &config)?
    };
    info!("semantic ASG ready: {} nodes, {} edges", asg.nodes.len(), asg.edges.len());

    let mut compressor = crate::compressor::ChunkCompressor::new();
    let registry = compressor.compress_asg(&asg);
    let saved_bytes: usize = registry
        .chunks
        .iter()
        .map(|entry| entry.value().bytes_saved())
        .sum();
    info!("compressed {} chunks ({} bytes saved)", registry.chunks.len(), saved_bytes);

    let shared_asg = SharedAsg::new(asg.clone());
    let search_engine = SearchEngine::with_config(
        shared_asg,
        registry.clone(),
        config.search.clone(),
    );
    search_engine.precompute_embeddings().await;

    let mut chunker = AstChunker::new()?.with_crate_root(&workspace);
    let chunks = chunker.chunk_dir(&workspace)?;
    let ast_merkle = MerkleTree::build(&chunks);
    let mut ast_trigram = TrigramIndex::new(3);
    ast_trigram.index(&chunks);
    info!(
        "AST index ready: {} chunks, merkle {}",
        chunks.len(),
        ast_merkle.root_hash().unwrap_or_default()
    );

    // Load memory store from persistent storage.
    let memory_store = if let Some(ref store) = persistent_store {
        match store.load_memories() {
            Ok(ms) => {
                info!("loaded {} memories from persistent store", ms.len());
                ms
            }
            Err(e) => {
                warn!("failed to load memories from persistent store: {e}");
                MemoryStore::new()
            }
        }
    } else {
        MemoryStore::new()
    };

    let mut state = ServerState::with_config(
        asg,
        registry,
        search_engine,
        ast_trigram,
        ast_merkle,
        workspace.clone(),
        &config,
    );
    *state.memory_store.lock().await = memory_store;
    state.persistent_store = persistent_store.map(Arc::new);

    // Save to persistent store if available.
    if let Some(ref store) = state.persistent_store {
        if let Err(e) = store.save_asg(&state.asg.inner) {
            warn!("failed to save ASG to persistent store: {e}");
        }
        let mem = state.memory_store.lock().await;
        if let Err(e) = store.save_memories(&mem) {
            warn!("failed to save memories to persistent store: {e}");
        }
    }

    // Start file watcher with reindex worker in the background.
    let watcher_workspace = workspace.clone();
    let watcher_indexes = state.indexes.clone();
    let watcher_debounce = std::time::Duration::from_millis(
        std::env::var("TOKEN_SAVER_WATCHER_DEBOUNCE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(config.watcher_debounce_ms),
    );
    tokio::spawn(async move {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let watcher_task = crate::watcher::watch_workspace(watcher_workspace.clone(), tx);
        let reindex_task = crate::watcher::reindex::run_reindex_worker(
            watcher_workspace,
            watcher_indexes,
            rx,
            watcher_debounce,
        );

        tokio::select! {
            result = watcher_task => {
                if let Err(e) = result {
                    warn!("file watcher failed: {e}");
                }
            }
            _ = reindex_task => {
                info!("reindex worker exited");
            }
        }
    });

    // Periodic memory pruning — every 10 minutes, remove memories whose
    // effective decay score has fallen below 0.01.
    {
        let prune_store = state.memory_store.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(600));
            loop {
                interval.tick().await;
                let pruned = prune_store.lock().await.prune_decayed(0.01);
                if pruned > 0 {
                    info!("periodic prune: removed {pruned} decayed memories");
                }
            }
        });
    }

    let cors_origin = config.cors_origin.as_deref();
    let listener = tokio::net::TcpListener::bind(&config.bind).await?;
    info!("Token Saver listening on http://{}", config.bind);

    // Graceful shutdown on Ctrl+C — save persistent state before exiting.
    let shutdown_state = state.clone();
    tokio::select! {
        result = axum::serve(listener, build_router(state, cors_origin)) => result?,
        _ = tokio::signal::ctrl_c() => {
            info!("received Ctrl+C, shutting down gracefully");
            // Persist ASG and memories to redb on shutdown.
            if let Some(ref store) = shutdown_state.persistent_store {
                info!("saving ASG and memories to persistent store before exit");
                if let Err(e) = store.save_asg(&shutdown_state.asg.inner) {
                    warn!("failed to save ASG on shutdown: {e}");
                }
                let mem = shutdown_state.memory_store.lock().await;
                if let Err(e) = store.save_memories(&mem) {
                    warn!("failed to save memories on shutdown: {e}");
                } else {
                    info!("saved {} memories on shutdown", mem.len());
                }
            }
        }
    }
    Ok(())
}

fn build_asg(workspace: &std::path::Path, config: &TokenSaverConfig) -> anyhow::Result<Asg> {
    Ok(crate::asg::build_asg_from_dir_with_config(
        workspace,
        config.search.ppr.clone(),
    )?)
}

/// Run the MCP server (for --mcp CLI flag). Loads config, builds the ASG,
/// and starts the MCP JSON-RPC server over stdio.
pub async fn run_mcp_server_with_config(config: TokenSaverConfig) -> anyhow::Result<()> {
    let workspace = config.canonical_workspace()?;
    info!("indexing workspace {} for MCP server", workspace.display());

    let asg = crate::asg::build_asg_from_dir_with_config(
        &workspace,
        config.search.ppr.clone(),
    )?;
    info!("semantic ASG ready: {} nodes, {} edges", asg.nodes.len(), asg.edges.len());

    let mut compressor = crate::compressor::ChunkCompressor::new();
    let registry = compressor.compress_asg(&asg);

    let shared_asg = SharedAsg::new(asg);
    let search_engine = Arc::new(SearchEngine::with_config(
        shared_asg.clone(),
        registry,
        config.search.clone(),
    ));
    search_engine.precompute_embeddings().await;

    let memory_store = Arc::new(tokio::sync::Mutex::new(MemoryStore::new()));

    crate::mcp::run_mcp_server(search_engine, shared_asg, memory_store).await
}
