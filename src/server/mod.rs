//! Phase 5: Low-Latency Streaming Completion Server
//!
//! An asynchronous completion server using axum and tokio.
//! Exposes a single POST endpoint `/v1/autocomplete` with:
//!   - 150ms request debouncer using CancellationToken
//!   - SSE streaming of completion chunks

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::asg::{Asg, SharedAsg};
use crate::ast::{AstChunker, MerkleTree, TrigramIndex};
use crate::compressor::ChunkRegistry;
use crate::search::SearchEngine;
use crate::tracker::{ContextTracker, CursorPayload};

// ---------------------------------------------------------------------------
// Request / Response Types
// ---------------------------------------------------------------------------

/// Request body for the autocomplete endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutocompleteRequest {
    /// The file path being edited.
    pub file_path: String,
    /// Cursor line (0-indexed).
    pub line: usize,
    /// Cursor column (0-indexed).
    pub column: usize,
    /// The partial text / query to complete.
    pub query: String,
    /// Optional request ID for tracing.
    pub request_id: Option<String>,
}

/// A single SSE chunk in the streaming response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChunk {
    /// The completion text for this chunk.
    pub text: String,
    /// Whether this is the final chunk.
    pub done: bool,
    /// The node ID this chunk is based on, if any.
    pub node_id: Option<usize>,
    /// The PageRank score of the source node.
    pub pagerank: Option<f64>,
}

// ---------------------------------------------------------------------------
// Server State
// ---------------------------------------------------------------------------

/// Shared application state for the completion server.
#[derive(Clone)]
pub struct ServerState {
    /// The ASG wrapped in a shared handle.
    pub asg: SharedAsg,
    /// The chunk registry for compressed source.
    pub registry: Arc<ChunkRegistry>,
    /// The search engine for RRF retrieval.
    pub search_engine: Arc<SearchEngine>,
    /// The context tracker for cursor mapping.
    pub tracker: Arc<ContextTracker>,
    /// Local trigram index over AST chunks (lexical/hybrid search pillar).
    pub ast_trigram: Arc<TrigramIndex>,
    /// Merkle tree fingerprint of the indexed workspace (change detection).
    pub ast_merkle: Arc<MerkleTree>,
    /// Active request cancellation tokens, keyed by file_path.
    pub active_requests: Arc<tokio::sync::Mutex<HashMap<String, CancellationToken>>>,
    /// Monotonic request-id counter.
    pub request_counter: Arc<AtomicU64>,
}

impl ServerState {
    pub fn new(
        asg: Asg,
        registry: ChunkRegistry,
        search_engine: SearchEngine,
        ast_trigram: TrigramIndex,
        ast_merkle: MerkleTree,
    ) -> Self {
        let shared_asg = SharedAsg::new(asg);
        let tracker = Arc::new(ContextTracker::new(
            shared_asg.clone(),
            registry.clone(),
            search_engine.clone(),
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
        }
    }

    /// Allocate the next monotonic request ID.
    pub fn next_request_id(&self) -> u64 {
        self.request_counter.fetch_add(1, Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Debouncer
// ---------------------------------------------------------------------------

/// Debounces incoming requests by file path.
/// If a request for the same file is already in-flight, it is cancelled
/// and replaced with the new one. The 150ms delay ensures we only process
/// the latest keystroke.
pub struct RequestDebouncer {
    active_requests: Arc<tokio::sync::Mutex<HashMap<String, CancellationToken>>>,
}

impl RequestDebouncer {
    pub fn new(
        active_requests: Arc<tokio::sync::Mutex<HashMap<String, CancellationToken>>>,
    ) -> Self {
        Self { active_requests }
    }

    /// Register a new request, cancelling any previous request for the same file.
    /// Returns a CancellationToken that will be cancelled if a newer request arrives.
    pub async fn register(&self, file_path: &str) -> CancellationToken {
        let token = CancellationToken::new();
        let mut requests = self.active_requests.lock().await;
        if let Some(old_token) = requests.insert(file_path.to_string(), token.clone()) {
            old_token.cancel();
            debug!("Cancelled previous request for {}", file_path);
        }
        token
    }

    /// Unregister a request after it completes.
    pub async fn unregister(&self, file_path: &str) {
        let mut requests = self.active_requests.lock().await;
        requests.remove(file_path);
    }
}

// ---------------------------------------------------------------------------
// SSE Streaming Response
// ---------------------------------------------------------------------------

/// A streaming SSE response that yields completion chunks.
///
/// The receiver side of the mpsc channel is turned into a `Stream` of
/// `text/event-stream` frames, then wrapped in an `axum::body::Body`.
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
        let stream = futures::stream::unfold(self.rx, |mut rx| async move {
            match rx.recv().await {
                Some(chunk) => {
                    let json = serde_json::to_string(&chunk).unwrap_or_default();
                    let frame = format!("data: {json}\n\n");
                    Some((
                        Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(frame)),
                        rx,
                    ))
                }
                // Channel closed: emit a final `done` event and end the stream.
                None => {
                    let final_frame = axum::body::Bytes::from(
                        "data: {\"text\":\"\",\"done\":true,\"node_id\":null,\"pagerank\":null}\n\n",
                    );
                    Some((Ok::<_, std::convert::Infallible>(final_frame), rx))
                }
            }
        });

        let body = axum::body::Body::from_stream(stream);

        Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("Connection", "keep-alive")
            .header("X-Accel-Buffering", "no")
            .body(body)
            .expect("valid SSE response")
    }
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/// POST /v1/autocomplete handler.
///
/// Debounces overlapping keystroke requests (150ms settle window), then streams
/// the assembled completion context chunk-by-chunk over SSE.
pub async fn autocomplete_handler(
    State(state): State<ServerState>,
    Json(req): Json<AutocompleteRequest>,
) -> Response {
    let request_id = req
        .request_id
        .clone()
        .unwrap_or_else(|| format!("req-{}", state.next_request_id()));

    info!(
        "Received autocomplete request {} for {}:{}:{}",
        request_id, req.file_path, req.line, req.column
    );

    // Debounce: cancel any in-flight request for this file.
    let debouncer = RequestDebouncer::new(state.active_requests.clone());
    let cancel_token = debouncer.register(&req.file_path).await;

    // Wait 150ms; a newer keystroke for the same file cancels this token and
    // drops this request.
    let watch_token = cancel_token.clone();
    let proceed = tokio::select! {
        _ = tokio::time::sleep(Duration::from_millis(150)) => true,
        _ = watch_token.cancelled() => false,
    };

    if !proceed {
        debug!("Request {} cancelled by a newer keystroke", request_id);
        debouncer.unregister(&req.file_path).await;
        return (StatusCode::OK, "cancelled").into_response();
    }

    // Build the cursor payload and locate the node under the cursor.
    let cursor = CursorPayload::new(&req.file_path, req.line, req.column);
    let node_id = state.tracker.find_node_at_cursor(&cursor);
    let (node_name, pagerank) = match node_id.and_then(|id| state.asg.get_node(id)) {
        Some(node) => (Some(node.name.clone()), Some(node.pagerank)),
        None => (None, None),
    };

    debug!(
        "Cursor at node {:?} (name: {:?}, pagerank: {:?})",
        node_id, node_name, pagerank
    );

    // Channel over which completion chunks are streamed.
    let (tx, rx) = tokio::sync::mpsc::channel::<CompletionChunk>(32);

    // Spawn the completion assembly task; it can be dropped (tx closed) if the
    // client disconnects.
    let state = state.clone();
    let cursor = cursor.clone();
    let query = req.query.clone();
    let task_token = cancel_token.clone();
    let task_id = request_id.clone();

    tokio::spawn(async move {
        // Chunk 1: a small header identifying the node under the cursor.
        if tx
            .send(CompletionChunk {
                text: format!(
                    "// node: {} (pagerank: {:.6})\n",
                    node_name.clone().unwrap_or_default(),
                    pagerank.unwrap_or(0.0)
                ),
                done: false,
                node_id,
                pagerank,
            })
            .await
            .is_err()
        {
            return; // Client disconnected.
        }

        // Chunk 2: raw (uncompressed) 20-line surrounding context.
        let surrounding = state.tracker.get_surrounding_lines(&cursor);
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

        // Chunks 3..N: top-3 compressed dependency nodes in token-saver notation.
        let deps = state
            .tracker
            .get_compressed_dependencies(&cursor, &query)
            .await;

        for (i, dep) in deps.iter().enumerate() {
            if task_token.is_cancelled() {
                warn!("Completion task {} cancelled by a newer request", task_id);
                break;
            }

            let header = match i {
                0 => "// --- Compressed Dependency (top 1) ---\n",
                1 => "// --- Compressed Dependency (top 2) ---\n",
                _ => "// --- Compressed Dependency (top 3) ---\n",
            };

            if tx
                .send(CompletionChunk {
                    text: format!("{header}{dep}\n"),
                    done: false,
                    node_id,
                    pagerank,
                })
                .await
                .is_err()
            {
                return;
            }

            // Small inter-chunk delay for a natural streaming cadence.
            tokio::time::sleep(Duration::from_millis(8)).await;
        }

        // Final chunk.
        let _ = tx
            .send(CompletionChunk {
                text: String::new(),
                done: true,
                node_id,
                pagerank,
            })
            .await;
    });

    // The streaming task holds its own copies of state; unregister this request
    // so the next keystroke starts clean.
    debouncer.unregister(&req.file_path).await;

    SseStream::new(rx).into_response()
}

// ---------------------------------------------------------------------------
// Server Builder
// ---------------------------------------------------------------------------

/// Build and return the axum router for the completion server.
pub fn build_router(state: ServerState) -> Router {
    Router::new()
        .route("/v1/autocomplete", post(autocomplete_handler))
        .with_state(state)
}

/// Run the completion server on the given address.
pub async fn run_server() -> anyhow::Result<()> {
    info!("Starting Token-Saver completion server");

    // Build the ASG from the workspace source directory.
    let workspace_dir = std::path::Path::new("/workspaces/Token-saver");
    let asg = crate::asg::build_asg_from_dir(workspace_dir)?;

    // Build the lossless compressed chunk registry (Phase 2).
    let mut chunk_compressor = crate::compressor::ChunkCompressor::new();
    let registry = chunk_compressor.compress_asg(&asg);

    // Build the multi-vector RRF search engine (Phase 3).
    let shared_asg = SharedAsg::new(asg.clone());
    let search_engine = SearchEngine::new(shared_asg.clone(), registry.clone());
    search_engine.precompute_embeddings().await;

    // Build the Cursor-style AST index (Phase 6): chunk the workspace, build
    // the Merkle fingerprint, and index trigrams for lexical/hybrid search.
    let mut chunker = AstChunker::new()?;
    let chunks = chunker.chunk_dir(workspace_dir)?;
    let ast_merkle = MerkleTree::build(&chunks);
    let mut ast_trigram = TrigramIndex::new(3);
    ast_trigram.index(&chunks);
    info!(
        "AST index ready: {} chunks, {} trigram docs, merkle root {}",
        chunks.len(),
        ast_trigram.len(),
        ast_merkle.root_hash().unwrap_or_default()
    );

    // Assemble shared state and serve.
    let state = ServerState::new(asg, registry, search_engine, ast_trigram, ast_merkle);
    let app = build_router(state);

    let addr = "0.0.0.0:8080";
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("Listening on http://{addr}");

    axum::serve(listener, app).await?;
    Ok(())
}
