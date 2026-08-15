//! File watcher for incremental re-indexing when source files change.
//!
//! Uses the `notify` crate to watch the workspace directory for .rs file changes.
//! When a change is detected, it triggers incremental re-indexing via the
//! existing Merkle diff pipeline.

pub mod reindex;

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{info, warn};

/// A file change event from the watcher.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChangeEvent {
    pub path: PathBuf,
    pub kind: ChangeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChangeKind {
    Created,
    Modified,
    Deleted,
}

/// Watch a directory for .rs file changes and send events to a channel.
pub async fn watch_workspace(
    workspace: PathBuf,
    tx: mpsc::UnboundedSender<FileChangeEvent>,
) -> anyhow::Result<()> {
    use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};

    let (notify_tx, notify_rx) = std::sync::mpsc::channel();
    let config = notify::Config::default().with_poll_interval(Duration::from_millis(100));
    let mut watcher = RecommendedWatcher::new(notify_tx, config)?;
    watcher.watch(&workspace, RecursiveMode::Recursive)?;

    info!("file watcher started for {}", workspace.display());

    // Spawn a blocking task to receive notify events and forward them.
    tokio::task::spawn_blocking(move || {
        while let Ok(res) = notify_rx.recv() {
            match res {
                Ok(event) => {
                    for path in &event.paths {
                        // Only watch .rs files, skip .git and target.
                        let path_str = path.to_string_lossy();
                        if path_str.contains("/.git/") || path_str.contains("/target/") {
                            continue;
                        }
                        if path.extension().map_or(true, |e| e != "rs") {
                            continue;
                        }

                        let kind = match event.kind {
                            EventKind::Create(_) => ChangeKind::Created,
                            EventKind::Modify(_) => ChangeKind::Modified,
                            EventKind::Remove(_) => ChangeKind::Deleted,
                            _ => ChangeKind::Modified,
                        };

                        if tx
                            .send(FileChangeEvent {
                                path: path.clone(),
                                kind,
                            })
                            .is_err()
                        {
                            warn!("file watcher receiver dropped");
                            return;
                        }
                    }
                }
                Err(e) => {
                    warn!("file watcher error: {e}");
                }
            }
        }
    });

    Ok(())
}
