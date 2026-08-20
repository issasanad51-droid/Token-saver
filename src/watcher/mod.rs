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
                        if !should_watch_path(path) {
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

/// Decide whether a filesystem path emitted by `notify` should be forwarded
/// to the reindex worker.
///
/// Two filters:
/// 1. Path must end in `.rs` (other extensions are ignored — Token-saver
///    only knows how to parse Rust).
/// 2. The path must not pass through a well-known dependency directory
///    (`.git`, `target`, `node_modules`). The check is *component-based*
///    rather than substring-based so it works on both Unix (`/.git/`) and
///    Windows (`\.git\`) separators — the previous implementation only
///    matched the Unix form and would have watched `.git` on Windows.
pub fn should_watch_path(path: &std::path::Path) -> bool {
    if path.extension().map_or(true, |ext| ext != "rs") {
        return false;
    }
    !path.components().any(|component| match component {
        std::path::Component::Normal(segment) => {
            matches!(segment.to_str(), Some(".git" | "target" | "node_modules"))
        }
        _ => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn should_watch_rust_source_in_workspace() {
        assert!(should_watch_path(&PathBuf::from("/repo/src/main.rs")));
        assert!(should_watch_path(&PathBuf::from(
            "/repo/src/deeply/nested/mod.rs"
        )));
    }

    #[test]
    fn should_ignore_non_rust_files() {
        assert!(!should_watch_path(&PathBuf::from("/repo/README.md")));
        assert!(!should_watch_path(&PathBuf::from("/repo/Cargo.toml")));
        assert!(!should_watch_path(&PathBuf::from("/repo/data.json")));
    }

    #[test]
    fn should_ignore_dot_git_directory() {
        // Unix-style separators.
        assert!(!should_watch_path(&PathBuf::from("/repo/.git/config")));
        assert!(!should_watch_path(&PathBuf::from("/repo/.git/HEAD")));
    }

    #[test]
    fn should_ignore_target_and_node_modules() {
        assert!(!should_watch_path(&PathBuf::from(
            "/repo/target/debug/x.rs"
        )));
        assert!(!should_watch_path(&PathBuf::from(
            "/repo/node_modules/pkg/lib.rs"
        )));
    }

    #[test]
    fn should_not_be_fooled_by_path_components_that_contain_target_as_substring() {
        // A directory literally named "my_target" must NOT be filtered out
        // (the previous substring impl would have matched `target` inside
        // `my_target` and skipped legitimate files).
        assert!(should_watch_path(&PathBuf::from("/repo/my_target/lib.rs")));
        assert!(should_watch_path(&PathBuf::from(
            "/repo/.gitignored/lib.rs"
        )));
    }
}
