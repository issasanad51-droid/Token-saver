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

        // Replace tokens with original identifiers.
        for (token, original) in &chunk.reverse_dictionary {
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
                if current.len() > 3 && !self.keywords.contains(current.as_str()) {
                    identifiers.push(current.clone());
                }
                current.clear();
            }
        }

        // Handle trailing identifier.
        if current.len() > 3 && !self.keywords.contains(current.as_str()) {
            identifiers.push(current);
        }

        // Deduplicate while preserving order.
        let mut seen = std::collections::HashSet::new();
        identifiers.retain(|id| seen.insert(id.clone()));
        identifiers
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

    fn generate_token(&self, index: usize) -> String {
        const CHARS: &[char] = &[
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
            'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
            'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
            'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        ];

        if index < CHARS.len() {
            format!("${}", CHARS[index])
        } else {
            let first = CHARS[index / CHARS.len()];
            let second = CHARS[index % CHARS.len()];
            format!("${}{}", first, second)
        }
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

    /// Compress a single node's source code.
    pub fn compress_node(&mut self, node: &Node) -> CompressedChunk {
        let identifiers = self.extractor.extract(&node.source);
        let mut dictionary: HashMap<String, String> = HashMap::new();
        let mut reverse_dictionary: HashMap<String, String> = HashMap::new();
        let mut compressed_source = node.source.clone();

        for id in identifiers {
            let token = self.token_gen.next_token();
            dictionary.insert(id.clone(), token.clone());
            reverse_dictionary.insert(token.clone(), id.clone());

            // Replace all occurrences of the identifier with the token.
            // Use word-boundary matching to avoid partial replacements.
            let pattern = format!(r"\b{}\b", regex::escape(&id));
            compressed_source = regex::Regex::new(&pattern)
                .unwrap()
                .replace_all(&compressed_source, &token)
                .to_string();
        }

        CompressedChunk {
            node_id: node.id,
            compressed_source,
            dictionary,
            reverse_dictionary,
        }
    }

    /// Compress a single source string (for ad-hoc use).
    pub fn compress_source(&mut self, source: &str) -> CompressedChunk {
        let identifiers = self.extractor.extract(source);
        let mut dictionary: HashMap<String, String> = HashMap::new();
        let mut reverse_dictionary: HashMap<String, String> = HashMap::new();
        let mut compressed_source = source.to_string();

        for id in identifiers {
            let token = self.token_gen.next_token();
            dictionary.insert(id.clone(), token.clone());
            reverse_dictionary.insert(token.clone(), id.clone());

            let pattern = format!(r"\b{}\b", regex::escape(&id));
            compressed_source = regex::Regex::new(&pattern)
                .unwrap()
                .replace_all(&compressed_source, &token)
                .to_string();
        }

        CompressedChunk {
            node_id: 0,
            compressed_source,
            dictionary,
            reverse_dictionary,
        }
    }
}
