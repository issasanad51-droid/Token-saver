//! AST Chunking — tree-sitter based structural splitting.
//!
//! Walks each parsed file's AST and emits self-contained `AstChunk`s, one per
//! logical code entity. Each chunk carries its complete source, byte/line
//! ranges, a stable id, and a content hash (consumed by the Merkle layer for
//! incremental change detection).
//!
//! This is the first of Cursor's four indexing pillars: rather than cutting
//! files into arbitrary text windows, we split along *structural* boundaries
//! (full functions, methods, structs, impls, enums, traits, modules) so the
//! model always receives complete units of context.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use tree_sitter::{Node as TsNode, Parser};

// ---------------------------------------------------------------------------
// Chunk kinds
// ---------------------------------------------------------------------------

/// The kind of logical entity a chunk represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ChunkKind {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Type,
    Const,
    Static,
    Macro,
    Module,
    /// A recognized item we still want standalone (e.g. a top-level statement
    /// block that is not one of the above).
    Other,
}

impl ChunkKind {
    /// Map a tree-sitter node kind to a `ChunkKind`, if it is chunkable.
    pub fn from_ts_kind(kind: &str) -> Option<ChunkKind> {
        match kind {
            "function_item" => Some(ChunkKind::Function),
            "struct_item" => Some(ChunkKind::Struct),
            "enum_item" => Some(ChunkKind::Enum),
            "trait_item" => Some(ChunkKind::Trait),
            "impl_item" => Some(ChunkKind::Impl),
            "type_item" => Some(ChunkKind::Type),
            "const_item" => Some(ChunkKind::Const),
            "static_item" => Some(ChunkKind::Static),
            "macro_definition" => Some(ChunkKind::Macro),
            "mod_item" => Some(ChunkKind::Module),
            _ => None,
        }
    }

    /// Stable short label used inside chunk ids.
    pub fn as_str(&self) -> &'static str {
        match self {
            ChunkKind::Function => "fn",
            ChunkKind::Struct => "struct",
            ChunkKind::Enum => "enum",
            ChunkKind::Trait => "trait",
            ChunkKind::Impl => "impl",
            ChunkKind::Type => "type",
            ChunkKind::Const => "const",
            ChunkKind::Static => "static",
            ChunkKind::Macro => "macro",
            ChunkKind::Module => "mod",
            ChunkKind::Other => "other",
        }
    }
}

// ---------------------------------------------------------------------------
// Chunk
// ---------------------------------------------------------------------------

/// A single self-contained unit of code context.
#[derive(Debug, Clone)]
pub struct AstChunk {
    /// Stable, content-independent id: `<rel_path>::<kind>::<name>`.
    pub id: String,
    /// The file this chunk came from.
    pub file_path: PathBuf,
    /// Display name (function/type name, or synthesized for anon chunks).
    pub name: String,
    /// Logical entity kind.
    pub kind: ChunkKind,
    /// Complete source of the chunk (verbatim from the file).
    pub source: String,
    /// Byte offset range `[start, end)` within the file.
    pub byte_range: (usize, usize),
    /// 0-indexed line range `[start, end)` within the file.
    pub line_range: (usize, usize),
    /// SHA-256 hex of `source` — used for change detection.
    pub content_hash: String,
    /// Optional parent chunk id (e.g. a method inside an `impl`).
    pub parent: Option<String>,
}

impl AstChunk {
    /// Build a stable id from a file path, kind, and name.
    pub fn make_id(file_path: &Path, kind: ChunkKind, name: &str) -> String {
        Self::make_id_ns(file_path, &[], kind, name)
    }

    /// Build a scope-aware id: `<rel_path>[::<scope>...]::<kind>::<name>`.
    ///
    /// `scope` is the nesting path of ancestor items (modules, impl targets,
    /// parent functions). Including it makes ids unique for same-named items
    /// living in different scopes of one file — e.g. `impl A { fn new }` vs
    /// `impl B { fn new }` — which the Merkle/trigram/vector layers all rely on
    /// as their stable key.
    pub fn make_id_ns(
        file_path: &Path,
        scope: &[String],
        kind: ChunkKind,
        name: &str,
    ) -> String {
        let mut id = file_path.to_string_lossy().replace('\\', "/");
        for seg in scope {
            id.push_str("::");
            id.push_str(seg);
        }
        id.push_str("::");
        id.push_str(kind.as_str());
        id.push_str("::");
        id.push_str(name);
        id
    }

    /// Compute the SHA-256 content hash of the source.
    pub fn hash_source(source: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(source.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

// ---------------------------------------------------------------------------
// Chunker
// ---------------------------------------------------------------------------

/// Tree-sitter based AST chunker (Rust-focused, extensible to other grammars).
pub struct AstChunker {
    parser: Parser,
    /// Crate root used to relativize file paths in chunk ids.
    crate_root: PathBuf,
}

impl AstChunker {
    /// Create a chunker with the Rust grammar loaded.
    pub fn new() -> anyhow::Result<Self> {
        let mut parser = Parser::new();
        let rust_lang: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
        parser
            .set_language(&rust_lang)
            .map_err(|e| anyhow::anyhow!("tree-sitter language error: {e}"))?;
        Ok(Self {
            parser,
            crate_root: PathBuf::from("."),
        })
    }

    /// Set the root against which file paths are relativized in chunk ids.
    pub fn with_crate_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.crate_root = root.into();
        self
    }

    /// Chunk a single file's source text.
    pub fn chunk_source(&mut self, file_path: &Path, source: &str) -> Vec<AstChunk> {
        let mut out = Vec::new();
        let Some(tree) = self.parser.parse(source.as_bytes(), None) else {
            return out;
        };
        let rel = file_path
            .strip_prefix(&self.crate_root)
            .unwrap_or(file_path);
        let mut scope: Vec<String> = Vec::new();
        self.walk(tree.root_node(), source, rel, None, &mut scope, &mut out);
        out
    }

    /// Chunk every `.rs` file under `dir`.
    pub fn chunk_dir(&mut self, dir: &Path) -> anyhow::Result<Vec<AstChunk>> {
        let mut chunks = Vec::new();
        for entry in walkdir::WalkDir::new(dir).into_iter().filter_map(Result::ok) {
            let path = entry.path();
            if entry.file_type().is_file() && path.extension().map_or(false, |e| e == "rs") {
                let source = std::fs::read_to_string(path)?;
                chunks.extend(self.chunk_source(path, &source));
            }
        }
        Ok(chunks)
    }

    // -----------------------------------------------------------------------
    // Tree walk
    // -----------------------------------------------------------------------

    fn walk(
        &self,
        node: TsNode,
        source: &str,
        file_path: &Path,
        parent: Option<String>,
        scope: &mut Vec<String>,
        out: &mut Vec<AstChunk>,
    ) {
        // Bare `mod foo;` declarations are resolved through the module path, not
        // as standalone chunks (mirrors the ASG's sparse-graph decision).
        if node.kind() == "mod_item" && node.child_by_field_name("body").is_none() {
            return;
        }

        if let Some(kind) = ChunkKind::from_ts_kind(node.kind()) {
            let name = extract_name(node, source)
                .unwrap_or_else(|| format!("<anon_{}>", node.kind()));
            let start = node.start_byte();
            let end = node.end_byte();
            let src = source.get(start..end).unwrap_or("").to_string();
            let content_hash = AstChunk::hash_source(&src);
            let id = AstChunk::make_id_ns(file_path, scope, kind, &name);
            let chunk = AstChunk {
                id: id.clone(),
                file_path: file_path.to_path_buf(),
                name: name.clone(),
                kind,
                source: src,
                byte_range: (start, end),
                line_range: (node.start_position().row, node.end_position().row + 1),
                content_hash,
                parent: parent.clone(),
            };
            out.push(chunk);

            // Children are namespaced under this item so nested / impl-local /
            // module-local items get unique, stable ids (e.g. `fn new` inside
            // two different `impl` blocks of one file).
            let new_parent = Some(id);
            scope.push(name);
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    self.walk(child, source, file_path, new_parent.clone(), scope, out);
                }
            }
            scope.pop();
        } else {
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    self.walk(child, source, file_path, parent.clone(), scope, out);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract a human-readable name from a tree-sitter item node.
fn extract_name(node: TsNode, source: &str) -> Option<String> {
    // Most items expose a `name` field.
    if let Some(name_node) = node.child_by_field_name("name") {
        return name_node
            .utf8_text(source.as_bytes())
            .ok()
            .map(str::to_string);
    }
    // `impl` blocks have no `name`; synthesize from self type / trait.
    if node.kind() == "impl_item" {
        let self_type = node
            .child_by_field_name("type")
            .and_then(|n| n.utf8_text(source.as_bytes()).ok())
            .unwrap_or("Self");
        let trait_name = node
            .child_by_field_name("trait")
            .and_then(|n| n.utf8_text(source.as_bytes()).ok());
        return match trait_name {
            Some(t) => Some(format!("{t}_for_{self_type}")),
            None => Some(format!("for_{self_type}")),
        };
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
pub fn outer() {
    pub fn inner() {}
}

struct Foo { x: u32 }

impl Foo {
    fn method(&self) {}
}
"#;

    #[test]
    fn chunks_split_on_structural_boundaries() {
        let mut chunker = AstChunker::new().unwrap();
        let chunks = chunker.chunk_source(Path::new("demo.rs"), SAMPLE);

        let kinds: Vec<_> = chunks.iter().map(|c| c.kind).collect();
        assert!(kinds.contains(&ChunkKind::Function));
        assert!(kinds.contains(&ChunkKind::Struct));
        assert!(kinds.contains(&ChunkKind::Impl));

        // The inner function must be its own chunk, parented to `outer`.
        let inner = chunks.iter().find(|c| c.name == "inner").unwrap();
        assert_eq!(inner.kind, ChunkKind::Function);
        assert!(inner.parent.is_some());
        assert!(inner.parent.as_ref().unwrap().contains("::fn::outer"));

        // Every chunk's source is a complete, non-empty slice.
        for c in &chunks {
            assert!(!c.source.trim().is_empty());
            assert_eq!(c.source, SAMPLE.get(c.byte_range.0..c.byte_range.1).unwrap());
        }
    }

    #[test]
    fn content_hash_is_stable() {
        let a = AstChunk::hash_source("pub fn x() {}");
        let b = AstChunk::hash_source("pub fn x() {}");
        let c = AstChunk::hash_source("pub fn y() {}");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
