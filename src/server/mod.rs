//! Low-latency HTTP/SSE server for token-budgeted code retrieval.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::asg::{Asg, SharedAsg};
use crate::ast::{AstChunker, MerkleTree, TrigramIndex};
use crate::compressor::ChunkRegistry;
use crate::config::TokenSaverConfig;
use crate::search::SearchEngine;
use crate::tracker::{ContextTracker, CursorPayload};

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
    pub ast_trigram: Arc<TrigramIndex>,
    pub ast_merkle: Arc<MerkleTree>,
    pub active_requests: Arc<tokio::sync::Mutex<HashMap<String, ActiveRequest>>>,
    pub request_counter: Arc<AtomicU64>,
    pub workspace: Arc<std::path::PathBuf>,
    pub debounce_ms: u64,
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
            ast_trigram: Arc::new(ast_trigram),
            ast_merkle: Arc::new(ast_merkle),
            active_requests: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            request_counter: Arc::new(AtomicU64::new(0)),
            workspace: Arc::new(workspace),
            debounce_ms: config.debounce_ms,
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
                    let frame = format!("data: {json}\n\n");
                    Some((
                        Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(frame)),
                        (rx, is_done),
                    ))
                }
                None => Some((
                    Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(
                        "data: {\"text\":\"\",\"done\":true,\"node_id\":null,\"pagerank\":null}\n\n",
                    )),
                    (rx, true),
                )),
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
    let top_k = request.top_k.clamp(1, 100);
    let results = state
        .search_engine
        .search_with_context(&request.query, top_k, request.context_node_id)
        .await;
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
    })
}

pub async fn health_handler(State(state): State<ServerState>) -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok",
        workspace: state.workspace.display().to_string(),
        asg_nodes: state.asg.inner.nodes.len(),
        asg_edges: state.asg.inner.edges.len(),
        compressed_chunks: state.registry.chunks.len(),
        ast_chunks: state.ast_trigram.len(),
        merkle_root: state.ast_merkle.root_hash(),
    })
}

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

pub fn build_router(state: ServerState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/search", post(search_handler))
        .route("/v1/autocomplete", post(autocomplete_handler))
        .with_state(state)
}

pub async fn run_server() -> anyhow::Result<()> {
    run_server_with_config(TokenSaverConfig::load()?).await
}

pub async fn run_server_with_config(config: TokenSaverConfig) -> anyhow::Result<()> {
    let workspace = config.canonical_workspace()?;
    info!("indexing workspace {}", workspace.display());

    let asg = crate::asg::build_asg_from_dir_with_config(
        &workspace,
        config.search.ppr.clone(),
    )?;
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

    let state = ServerState::with_config(
        asg,
        registry,
        search_engine,
        ast_trigram,
        ast_merkle,
        workspace,
        &config,
    );
    let listener = tokio::net::TcpListener::bind(&config.bind).await?;
    info!("Token Saver listening on http://{}", config.bind);
    axum::serve(listener, build_router(state)).await?;
    Ok(())
}
