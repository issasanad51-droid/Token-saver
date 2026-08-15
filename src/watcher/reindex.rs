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

/// Stats reported after each re-index cycle.
#[derive(Debug, Clone)]
pub struct ReindexReport {
    pub files_processed: usize,
    pub chunks_added: usize,
    pub chunks_removed: usize,
    pub chunks_changed: usize,
}

/// Run the re-index worker: receives file change events, debounces them,
/// and applies incremental updates to the shared indexes.
pub async fn run_reindex_worker(
    workspace: PathBuf,
    indexes: SharedIndexes,
    mut rx: mpsc::UnboundedReceiver<FileChangeEvent>,
    debounce: Duration,
) {
    info!("reindex worker started for {} (debounce: {}ms)", workspace.display(), debounce.as_millis());

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

        // Deduplicate: only care about the latest kind per path.
        let mut changes: HashSet<(PathBuf, ChangeKind)> = HashSet::new();
        for event in pending {
            changes.insert((event.path, event.kind));
        }

        let report = apply_incremental_update(&workspace, &indexes, &changes).await;
        info!(
            "reindex cycle complete: {} files, {} added, {} removed, {} changed",
            report.files_processed, report.chunks_added, report.chunks_removed, report.chunks_changed
        );
    }
}

/// Apply the collected file changes to the shared indexes.
async fn apply_incremental_update(
    workspace: &Path,
    indexes: &SharedIndexes,
    changes: &HashSet<(PathBuf, ChangeKind)>,
) -> ReindexReport {
    let mut report = ReindexReport {
        files_processed: 0,
        chunks_added: 0,
        chunks_removed: 0,
        chunks_changed: 0,
    };

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
        report.chunks_removed = diff.deleted.len();

        *indexes.merkle.write().await = new_merkle;

        debug!(
            "merkle diff: {} changed, {} deleted",
            diff.changed.len(),
            diff.deleted.len()
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
