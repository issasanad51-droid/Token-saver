//! YAML + environment configuration for the complete token-saver pipeline.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

use crate::search::SearchConfig;
use crate::tracker::ContextConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TokenSaverConfig {
    /// Repository/workspace to index. Relative paths resolve from the process
    /// working directory.
    pub workspace: PathBuf,
    /// TCP listener, e.g. `0.0.0.0:8080`.
    pub bind: String,
    /// Keystroke settle window before retrieval starts.
    pub debounce_ms: u64,
    /// Explicit path for the redb persistent store.
    /// Default: `<workspace>/.token-saver.db`
    pub db_path: Option<PathBuf>,
    /// Path to a file containing the 32-byte AES encryption key.
    /// Default: None (random per-process key).
    pub encryption_key_path: Option<PathBuf>,
    /// Debounce window for the file watcher (ms).
    pub watcher_debounce_ms: u64,
    /// Skip files larger than this (KB). Default: 1024 (1 MB).
    pub max_file_size_kb: u64,
    /// Allowed CORS origin. Default: None (no CORS).
    pub cors_origin: Option<String>,
    pub search: SearchConfig,
    pub context: ContextConfig,
}

impl Default for TokenSaverConfig {
    fn default() -> Self {
        Self {
            workspace: PathBuf::from("."),
            bind: "0.0.0.0:8080".to_string(),
            debounce_ms: 150,
            db_path: None,
            encryption_key_path: None,
            watcher_debounce_ms: 500,
            max_file_size_kb: 1024,
            cors_origin: None,
            search: SearchConfig::default(),
            context: ContextConfig::default(),
        }
    }
}

impl TokenSaverConfig {
    /// Load `TOKEN_SAVER_CONFIG` when set, otherwise load `token-saver.yaml` if
    /// present, then apply simple environment overrides.
    pub fn load() -> anyhow::Result<Self> {
        let explicit = std::env::var_os("TOKEN_SAVER_CONFIG").map(PathBuf::from);
        let conventional = PathBuf::from("token-saver.yaml");
        let config_path = explicit.or_else(|| conventional.exists().then_some(conventional));

        let mut config = match config_path {
            Some(path) => Self::from_path(&path)?,
            None => Self::default(),
        };
        if let Some(workspace) = std::env::var_os("TOKEN_SAVER_WORKSPACE") {
            config.workspace = PathBuf::from(workspace);
        }
        if let Ok(bind) = std::env::var("TOKEN_SAVER_BIND") {
            config.bind = bind;
        }
        if let Ok(db_path) = std::env::var("TOKEN_SAVER_DB_PATH") {
            config.db_path = Some(PathBuf::from(db_path));
        }
        if let Ok(key_path) = std::env::var("TOKEN_SAVER_ENCRYPTION_KEY") {
            config.encryption_key_path = Some(PathBuf::from(key_path));
        }
        if let Ok(debounce) = std::env::var("TOKEN_SAVER_WATCHER_DEBOUNCE") {
            if let Ok(ms) = debounce.parse() {
                config.watcher_debounce_ms = ms;
            }
        }
        Ok(config)
    }

    pub fn from_path(path: &Path) -> anyhow::Result<Self> {
        let yaml = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config {}", path.display()))?;
        serde_yaml::from_str(&yaml)
            .with_context(|| format!("invalid token-saver config {}", path.display()))
    }

    pub fn canonical_workspace(&self) -> anyhow::Result<PathBuf> {
        self.workspace
            .canonicalize()
            .with_context(|| format!("workspace does not exist: {}", self.workspace.display()))
    }

    /// Resolve the database path: explicit if set, otherwise `<workspace>/.token-saver.db`.
    pub fn resolve_db_path(&self) -> PathBuf {
        self.db_path
            .clone()
            .unwrap_or_else(|| self.workspace.join(".token-saver.db"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_yaml_uses_nested_defaults() {
        let config: TokenSaverConfig = serde_yaml::from_str(
            "search:\n  rrf:\n    weights: [2.0, 1.0, 0.5]\ncontext:\n  max_dependencies: 3\n",
        )
        .unwrap();
        assert_eq!(config.search.rrf.weights, vec![2.0, 1.0, 0.5]);
        assert_eq!(config.search.rrf.k, 60.0);
        assert_eq!(config.context.max_dependencies, 3);
        assert_eq!(config.context.surrounding_lines, 16);
    }

    #[test]
    fn resolve_db_path_uses_default() {
        let config = TokenSaverConfig::default();
        // Default workspace is "." so db path is "./.token-saver.db"
        assert!(config.resolve_db_path().to_string_lossy().ends_with(".token-saver.db"));
    }

    #[test]
    fn resolve_db_path_uses_explicit() {
        let config = TokenSaverConfig {
            db_path: Some(PathBuf::from("/custom/path.db")),
            ..TokenSaverConfig::default()
        };
        assert_eq!(
            config.resolve_db_path(),
            PathBuf::from("/custom/path.db")
        );
    }

    #[test]
    fn default_watcher_debounce() {
        let config = TokenSaverConfig::default();
        assert_eq!(config.watcher_debounce_ms, 500);
        assert_eq!(config.max_file_size_kb, 1024);
    }
}
