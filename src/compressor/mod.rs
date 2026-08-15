//! Phase 2: Lossless Token-Saving Chunker
//!
//! Chunks map 1:1 with functional ASG nodes (functions, structs, impl blocks).
//! A dictionary compressor replaces long custom identifiers with compact 1-2
//! character tokens, maintaining an invertible mapping in a thread-safe
//! hydration registry.

use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;

use crate::asg::{Asg, ChunkExtractor, Node};

// ---------------------------------------------------------------------------
// Chunk Registry
// ---------------------------------------------------------------------------

/// A compressed chunk: the original node ID maps to compressed source + dictionary.
#[derive(Debug, Clone)]
pub struct CompressedChunk {
    pub node_id: usize,
    pub compressed_source: String,
    pub dictionary: HashMap<String, String>, // original identifier -> token
    pub reverse_dictionary: HashMap<String, String>, // token -> original identifier
    pub original_bytes: usize,
    /// Bytes sent to the model, including the compact alias legend.
    pub compressed_bytes: usize,
}

impl CompressedChunk {
    /// Render an independently understandable prompt chunk. Previously only the
    /// substituted body was emitted, which saved bytes but gave the model no
    /// way to interpret `$a`/`$b` aliases.
    pub fn prompt_text(&self) -> String {
        if self.dictionary.is_empty() {
            return self.compressed_source.clone();
        }
        let mut aliases: Vec<(&String, &String)> = self.dictionary.iter().collect();
        aliases.sort_by(|(_, left_token), (_, right_token)| left_token.cmp(right_token));
        let legend = aliases
            .into_iter()
            .map(|(original, token)| format!("{token}={original}"))
            .collect::<Vec<_>>()
            .join(",");
        format!("// aliases:{legend}\n{}", self.compressed_source)
    }

    pub fn bytes_saved(&self) -> usize {
        self.original_bytes.saturating_sub(self.compressed_bytes)
    }
}

/// Thread-safe registry mapping compressed node IDs back to original text.
#[derive(Debug, Default, Clone)]
pub struct ChunkRegistry {
    /// node_id -> CompressedChunk
    pub chunks: Arc<DashMap<usize, CompressedChunk>>,
    /// token -> original identifier (global, for hydration)
    pub global_token_map: Arc<DashMap<String, String>>,
}

impl ChunkRegistry {
    pub fn new() -> Self {
        Self {
            chunks: Arc::new(DashMap::new()),
            global_token_map: Arc::new(DashMap::new()),
        }
    }

    /// Register a compressed chunk.
    pub fn register(&self, chunk: CompressedChunk) {
        // Populate global token map.
        for (original, token) in &chunk.dictionary {
            self.global_token_map.insert(token.clone(), original.clone());
        }
        self.chunks.insert(chunk.node_id, chunk);
    }

    /// Hydrate (decompress) a chunk back to original source.
    pub fn hydrate(&self, node_id: usize) -> Option<String> {
        let chunk = self.chunks.get(&node_id)?;
        let mut result = chunk.compressed_source.clone();

        // Replace longest tokens first (`$aa` before `$a`) so one compact alias
        // can never corrupt another during hydration.
        let mut aliases: Vec<(&String, &String)> = chunk.reverse_dictionary.iter().collect();
        aliases.sort_by(|(left, _), (right, _)| right.len().cmp(&left.len()));
        for (token, original) in aliases {
            result = result.replace(token, original);
        }

        Some(result)
    }

    /// Get a compressed chunk by node ID.
    pub fn get(&self, node_id: usize) -> Option<CompressedChunk> {
        self.chunks.get(&node_id).map(|r| r.clone())
    }
}

// ---------------------------------------------------------------------------
// Identifier Extractor
// ---------------------------------------------------------------------------

/// Extracts custom identifiers from source code.
/// An identifier is considered "custom" if it's longer than 3 characters
/// and is not a Rust keyword.
pub struct IdentifierExtractor {
    keywords: std::collections::HashSet<&'static str>,
}

impl Default for IdentifierExtractor {
    fn default() -> Self {
        let keywords: std::collections::HashSet<&'static str> = [
            "as", "break", "const", "continue", "crate", "else", "enum", "extern", "false",
            "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut",
            "pub", "ref", "return", "self", "Self", "static", "struct", "super", "trait",
            "true", "type", "unsafe", "use", "where", "while", "async", "await", "dyn",
            "abstract", "become", "box", "do", "final", "macro", "override", "priv",
            "typeof", "unsized", "virtual", "yield", "try",
        ]
        .iter()
        .copied()
        .collect();

        Self { keywords }
    }
}

impl IdentifierExtractor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract all custom identifiers from source code.
    pub fn extract(&self, source: &str) -> Vec<String> {
        let mut identifiers = Vec::new();
        let mut current = String::new();

        for ch in source.chars() {
            if ch.is_alphanumeric() || ch == '_' {
                current.push(ch);
            } else {
                if self.is_custom_identifier(&current) {
                    identifiers.push(current.clone());
                }
                current.clear();
            }
        }

        // Handle trailing identifier.
        if self.is_custom_identifier(&current) {
            identifiers.push(current);
        }

        // Deduplicate while preserving order.
        let mut seen = std::collections::HashSet::new();
        identifiers.retain(|id| seen.insert(id.clone()));
        identifiers
    }

    fn is_custom_identifier(&self, value: &str) -> bool {
        value.len() > 3
            && value
                .chars()
                .next()
                .map(|first| first == '_' || first.is_alphabetic())
                .unwrap_or(false)
            && !self.keywords.contains(value)
    }
}

// ---------------------------------------------------------------------------
// Token Generator
// ---------------------------------------------------------------------------

/// Generates compact 1-2 character tokens for identifiers.
pub struct TokenGenerator {
    used_tokens: std::collections::HashSet<String>,
    counter: usize,
}

impl Default for TokenGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenGenerator {
    pub fn new() -> Self {
        Self {
            used_tokens: std::collections::HashSet::new(),
            counter: 0,
        }
    }

    /// Generate the next available token.
    /// Tokens are of the form $a, $b, ..., $z, $A, $B, ..., $Z, $0, $1, ..., $9,
    /// then $aa, $ab, etc.
    pub fn next_token(&mut self) -> String {
        loop {
            let token = self.generate_token(self.counter);
            self.counter += 1;
            if !self.used_tokens.contains(&token) {
                self.used_tokens.insert(token.clone());
                return token;
            }
        }
    }

    fn generate_token(&self, mut index: usize) -> String {
        const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        let base = CHARS.len();
        let mut suffix = Vec::new();
        loop {
            suffix.push(CHARS[index % base] as char);
            if index < base {
                break;
            }
            index = index / base - 1;
        }
        suffix.reverse();
        format!("${}", suffix.into_iter().collect::<String>())
    }
}

// ---------------------------------------------------------------------------
// Chunk Compressor
// ---------------------------------------------------------------------------

/// Compresses ASG nodes into token-saved chunks.
pub struct ChunkCompressor {
    extractor: IdentifierExtractor,
    token_gen: TokenGenerator,
}

impl Default for ChunkCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkCompressor {
    pub fn new() -> Self {
        Self {
            extractor: IdentifierExtractor::new(),
            token_gen: TokenGenerator::new(),
        }
    }

    /// Compress all functional chunks in the ASG.
    pub fn compress_asg(&mut self, asg: &Asg) -> ChunkRegistry {
        let registry = ChunkRegistry::new();
        let chunk_ids = asg.extract_chunks();

        for node_id in chunk_ids {
            if let Some(node) = asg.nodes.get(node_id) {
                let chunk = self.compress_node(node);
                registry.register(chunk);
            }
        }

        registry
    }

    /// Compress a single node's source code. Aliases are retained only when
    /// the body plus its legend is strictly smaller than the original source.
    pub fn compress_node(&mut self, node: &Node) -> CompressedChunk {
        self.compress(node.id, &node.source)
    }

    /// Compress a single source string (for ad-hoc use).
    pub fn compress_source(&mut self, source: &str) -> CompressedChunk {
        self.compress(0, source)
    }

    fn compress(&mut self, node_id: usize, source: &str) -> CompressedChunk {
        let identifiers = self.extractor.extract(source);
        let mut dictionary: HashMap<String, String> = HashMap::new();
        let mut reverse_dictionary: HashMap<String, String> = HashMap::new();
        let mut compressed_source = source.to_string();

        for identifier in identifiers {
            let pattern = format!(r"\b{}\b", regex::escape(&identifier));
            let Ok(expression) = regex::Regex::new(&pattern) else { continue };
            let occurrences = expression.find_iter(source).count();
            if occurrences == 0 {
                continue;
            }

            let token = self.token_gen.next_token();
            let body_savings = occurrences.saturating_mul(identifier.len().saturating_sub(token.len()));
            // Approximate this alias's `token=identifier,` legend cost. The
            // final exact-size check below catches interactions and header cost.
            let legend_cost = token.len() + identifier.len() + 2;
            if body_savings <= legend_cost {
                continue;
            }

            compressed_source = expression
                .replace_all(&compressed_source, regex::NoExpand(token.as_str()))
                .to_string();
            dictionary.insert(identifier.clone(), token.clone());
            reverse_dictionary.insert(token, identifier);
        }

        let mut chunk = CompressedChunk {
            node_id,
            compressed_source,
            dictionary,
            reverse_dictionary,
            original_bytes: source.len(),
            compressed_bytes: 0,
        };
        chunk.compressed_bytes = chunk.prompt_text().len();

        // Never call an expansion "compression". Returning the raw source also
        // avoids burdening the model with aliases that provide no net saving.
        if chunk.compressed_bytes >= chunk.original_bytes {
            chunk.compressed_source = source.to_string();
            chunk.dictionary.clear();
            chunk.reverse_dictionary.clear();
            chunk.compressed_bytes = chunk.original_bytes;
        }
        chunk
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hydration_is_lossless_with_prefix_tokens() {
        let registry = ChunkRegistry::new();
        registry.register(CompressedChunk {
            node_id: 7,
            compressed_source: "$aa($a)".to_string(),
            dictionary: HashMap::from([
                ("long_function".to_string(), "$aa".to_string()),
                ("long_value".to_string(), "$a".to_string()),
            ]),
            reverse_dictionary: HashMap::from([
                ("$aa".to_string(), "long_function".to_string()),
                ("$a".to_string(), "long_value".to_string()),
            ]),
            original_bytes: 30,
            compressed_bytes: 7,
        });
        assert_eq!(registry.hydrate(7).as_deref(), Some("long_function(long_value)"));
    }

    #[test]
    fn only_reports_real_prompt_savings() {
        let repeated = "very_long_identifier ".repeat(30);
        let mut compressor = ChunkCompressor::new();
        let chunk = compressor.compress_source(&repeated);
        assert!(chunk.compressed_bytes < chunk.original_bytes);
        assert!(chunk.bytes_saved() > 0);
        assert!(chunk.prompt_text().contains("very_long_identifier"));
    }

    #[test]
    fn short_one_off_identifiers_are_left_alone() {
        let source = "fn once() { let value = 1; }";
        let mut compressor = ChunkCompressor::new();
        let chunk = compressor.compress_source(source);
        assert_eq!(chunk.prompt_text(), source);
        assert_eq!(chunk.bytes_saved(), 0);
    }

    #[test]
    fn token_generator_scales_past_two_character_space() {
        let mut generator = TokenGenerator::new();
        let mut last = String::new();
        for _ in 0..4_000 {
            last = generator.next_token();
        }
        assert!(last.starts_with('$'));
        assert_eq!(generator.used_tokens.len(), 4_000);
    }
}
