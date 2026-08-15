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
    pub search: SearchConfig,
    pub context: ContextConfig,
}

impl Default for TokenSaverConfig {
    fn default() -> Self {
        Self {
            workspace: PathBuf::from("."),
            bind: "0.0.0.0:8080".to_string(),
            debounce_ms: 150,
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
}
