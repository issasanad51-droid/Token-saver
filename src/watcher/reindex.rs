//! Incremental re-indexing worker for file change events.
//!
//! Batches incoming `FileChangeEvent`s with a configurable debounce window,
//! then re-chunks the affected files and updates the Merkle tree, trigram
//! index, and vector sync pipeline incrementally.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn};

use crate::ast::chunker::AstChunker;
use crate::ast::merkle::MerkleTree;
use crate::ast::trigram::TrigramIndex;
use crate::watcher::{ChangeKind, FileChangeEvent};

/// Shared indexes that the reindex worker updates and the server reads.
#[derive(Clone)]
pub struct SharedIndexes {
    /// Current Merkle tree (project fingerprint + change detection).
    pub merkle: Arc<RwLock<MerkleTree>>,
    /// Current trigram index (lexical search).
    pub trigram: Arc<RwLock<TrigramIndex>>,
}

impl SharedIndexes {
    pub fn new(merkle: MerkleTree, trigram: TrigramIndex) -> Self {
        Self {
            merkle: Arc::new(RwLock::new(merkle)),
            trigram: Arc::new(RwLock::new(trigram)),
        }
    }
}

/// The full set of artefacts the reindex worker can refresh.
///
/// When [`RefreshTarget::pipeline`] is `Some`, the worker calls
/// [`crate::server::refresh_pipeline_after_reindex`] on every debounce
/// cycle, so file changes propagate not just to trigram+merkle but also to
/// the live ASG, ChunkRegistry, and SearchEngine caches. When it's `None`,
/// the worker falls back to the legacy trigram+merkle-only refresh.
#[derive(Clone)]
pub struct RefreshTarget {
    pub indexes: SharedIndexes,
    /// Optional full-pipeline refresh hook. Set by the HTTP server at
    /// startup so the file watcher gets the same treatment as `/v1/reindex`.
    /// See [`crate::server::refresh_pipeline_after_reindex`] for the contract.
    pub pipeline: Option<PipelineRefreshHook>,
}

/// Type-erased handle to the full-pipeline refresh function. We use a
/// closure here so the `watcher` crate doesn't need to depend on the
/// `server` crate (which would be a circular dep).
pub type PipelineRefreshHook =
    Arc<dyn Fn() -> futures::future::BoxFuture<'static, anyhow::Result<()>> + Send + Sync>;

impl RefreshTarget {
    pub fn new(indexes: SharedIndexes) -> Self {
        Self {
            indexes,
            pipeline: None,
        }
    }

    pub fn with_pipeline(indexes: SharedIndexes, hook: PipelineRefreshHook) -> Self {
        Self {
            indexes,
            pipeline: Some(hook),
        }
    }
}

/// Stats reported after each re-index cycle.
#[derive(Debug, Clone, Default)]
pub struct ReindexReport {
    pub files_processed: usize,
    pub chunks_added: usize,
    /// Chunks physically removed from the trigram index (i.e. belonged to
    /// files that were deleted or modified).
    pub chunks_removed: usize,
    /// Chunks whose content hash changed between the old and new Merkle tree.
    /// This is distinct from `chunks_removed` — a modified file may produce
    /// the same number of chunks (so `chunks_removed` is 0) but every chunk
    /// has a new hash (so `chunks_changed` is N).
    pub chunks_changed: usize,
}

/// Run the re-index worker: receives file change events, debounces them,
/// and applies incremental updates to the shared indexes.
///
/// If `target.pipeline` is set, the worker calls the full-pipeline refresh
/// hook (which rebuilds the ASG, ChunkRegistry, and SearchEngine caches too)
/// on every debounce cycle, so file changes propagate not just to trigram
/// +merkle but to the live retrieval pipeline as well. When it's `None`,
/// the worker falls back to the legacy trigram+merkle-only refresh.
pub async fn run_reindex_worker(
    workspace: PathBuf,
    target: RefreshTarget,
    mut rx: mpsc::UnboundedReceiver<FileChangeEvent>,
    debounce: Duration,
) {
    info!(
        "reindex worker started for {} (debounce: {}ms, full_pipeline: {})",
        workspace.display(),
        debounce.as_millis(),
        target.pipeline.is_some()
    );

    loop {
        // Wait for the first event.
        let first = match rx.recv().await {
            Some(event) => event,
            None => {
                info!("reindex worker channel closed, shutting down");
                return;
            }
        };

        // Collect events until the debounce window expires.
        let mut pending: Vec<FileChangeEvent> = vec![first];
        let deadline = tokio::time::Instant::now() + debounce;

        loop {
            tokio::select! {
                event = rx.recv() => {
                    match event {
                        Some(e) => pending.push(e),
                        None => {
                            info!("reindex worker channel closed, shutting down");
                            return;
                        }
                    }
                    if tokio::time::Instant::now() >= deadline {
                        break;
                    }
                }
                _ = tokio::time::sleep_until(deadline) => {
                    break;
                }
            }
        }

        // Deduplicate: only care about the latest kind per path. The hook
        // doesn't actually need this set — it does a full rebuild — but we
        // keep it for the legacy fallback path and for the log message.
        let mut changes: HashSet<(PathBuf, ChangeKind)> = HashSet::new();
        for event in pending {
            changes.insert((event.path, event.kind));
        }

        if let Some(hook) = target.pipeline.as_ref() {
            // Full-pipeline refresh: rebuild ASG + registry + search caches
            // + trigram + merkle in one pass. The hook owns the workspace +
            // SharedAsg + ChunkRegistry + SearchEngine, so it can do
            // everything `/v1/reindex` does.
            let started = std::time::Instant::now();
            match hook().await {
                Ok(()) => info!(
                    "reindex cycle complete via full pipeline in {:?} ({} file changes)",
                    started.elapsed(),
                    changes.len()
                ),
                Err(e) => warn!("reindex full-pipeline refresh failed: {e:#}"),
            }
            // The hook already rebuilds trigram + merkle internally, so
            // there's nothing left to do here.
        } else {
            // Legacy fallback: trigram + merkle only (no ASG/registry/search
            // refresh). Retained so the watcher can be used standalone (e.g.
            // in tests that don't have a SearchEngine).
            let report = apply_incremental_update(&workspace, &target.indexes, &changes).await;
            info!(
                "reindex cycle complete: {} files, {} added, {} removed, {} changed",
                report.files_processed,
                report.chunks_added,
                report.chunks_removed,
                report.chunks_changed
            );
        }
    }
}

/// Apply the collected file changes to the shared indexes.
async fn apply_incremental_update(
    workspace: &Path,
    indexes: &SharedIndexes,
    changes: &HashSet<(PathBuf, ChangeKind)>,
) -> ReindexReport {
    let mut report = ReindexReport::default();

    let mut chunker = match AstChunker::new() {
        Ok(c) => c.with_crate_root(workspace),
        Err(e) => {
            warn!("failed to create chunker for reindex: {e}");
            return report;
        }
    };

    // Collect all current chunks from files that still exist, plus
    // identify paths that were deleted.
    let mut new_chunks = Vec::new();
    let mut deleted_paths = HashSet::new();

    for (path, kind) in changes {
        report.files_processed += 1;
        match kind {
            ChangeKind::Deleted => {
                deleted_paths.insert(path.clone());
            }
            ChangeKind::Created | ChangeKind::Modified => {
                if path.exists() {
                    match std::fs::read_to_string(path) {
                        Ok(source) => {
                            let chunks = chunker.chunk_source(path, &source);
                            new_chunks.extend(chunks);
                        }
                        Err(e) => {
                            warn!("failed to read changed file {}: {e}", path.display());
                        }
                    }
                } else {
                    deleted_paths.insert(path.clone());
                }
            }
        }
    }

    if report.files_processed == 0 {
        return report;
    }

    // Update the trigram index: remove old chunks for changed/deleted files,
    // then add the new ones.
    {
        let mut trigram = indexes.trigram.write().await;

        // Remove chunks belonging to changed or deleted files using file-path
        // based removal, which correctly cleans up the inverted index.
        for (path, kind) in changes {
            if *kind == ChangeKind::Deleted || *kind == ChangeKind::Modified {
                let removed = trigram.remove_by_file(path);
                report.chunks_removed += removed;
            }
        }

        // Add/update new chunks.
        for chunk in &new_chunks {
            trigram.add(chunk.clone());
        }
        report.chunks_added = new_chunks.len();
    }

    // Rebuild the Merkle tree from all chunks (changed + unchanged).
    // A full rebuild is simpler and still fast for typical codebases.
    // For very large codebases, a true incremental Merkle update would
    // only rehash the affected leaves.
    {
        // Re-chunk the entire workspace to get the complete chunk list
        // for a consistent Merkle tree. For small-to-medium codebases
        // this is fast enough; for large ones, we'd want true incremental.
        let all_chunks = match chunker.chunk_dir(workspace) {
            Ok(c) => c,
            Err(e) => {
                warn!("failed to re-chunk workspace for Merkle rebuild: {e}");
                return report;
            }
        };

        let old_merkle = indexes.merkle.read().await;
        let new_merkle = MerkleTree::build(&all_chunks);
        let diff = new_merkle.diff(&old_merkle);
        drop(old_merkle);

        report.chunks_changed = diff.changed.len();
        // Note: do NOT overwrite `chunks_removed` with `diff.deleted.len()`.
        // The previous implementation did that, which meant the report's
        // `chunks_removed` reflected only Merkle-tree deletions and erased
        // the trigram-index removal count computed above. The two numbers
        // are different signals — trigram removals count every chunk that
        // was pulled out of the lexical index (file modified OR deleted),
        // while `diff.deleted` counts only chunks that disappeared entirely
        // from the rebuilt Merkle tree. Keep both fields separate.
        let merkle_deleted = diff.deleted.len();

        *indexes.merkle.write().await = new_merkle;

        debug!(
            "merkle diff: {} changed, {} deleted",
            diff.changed.len(),
            merkle_deleted
        );
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_indexes_construction() {
        let merkle = MerkleTree::build(&[]);
        let trigram = TrigramIndex::new(3);
        let indexes = SharedIndexes::new(merkle, trigram);
        assert!(indexes.merkle.blocking_read().root_hash().is_none());
        assert!(indexes.trigram.blocking_read().is_empty());
    }
}
