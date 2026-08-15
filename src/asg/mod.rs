//! Phase 1: Custom Abstract Semantic Graph (ASG) & PageRank Engine
//!
//! Parses source files into ASTs via tree-sitter, constructs an explicit
//! semantic graph with cross-file symbol resolution, and runs a from-scratch
//! PageRank power-iteration over the graph.
//!
//! # v2 foundation
//!
//! `crate::asg::graph` holds the redesigned `AsgGraph` model: string `NodeId`
//! trackers, `petgraph` backing, the `id`/`type`/`body`/`incoming_edges`/
//! `outgoing_edges` YAML schema contract, and lossless YAML round-tripping.
//! This legacy module (Phase 1) is retained for the older `usize`-based
//! pipeline; downstream phases can be migrated onto `AsgGraph` incrementally.

pub mod builder;
pub mod graph;
pub mod pagerank;
pub mod source;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tree_sitter::{Node as TsNode, Parser, Tree};

use crate::compressor::ChunkRegistry;

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

/// Edge kinds that connect ASG nodes semantically.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    /// A call-site node references a definition node.
    Calls,
    /// A node is contained within (owns) another node.
    Contains,
    /// A node imports / references a symbol from another file.
    Imports,
    /// A node implements a trait or interface.
    Implements,
    /// A node is a field of a struct.
    FieldOf,
    /// A node is a variant of an enum.
    VariantOf,
}

/// A semantic node in the ASG.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: usize,
    pub name: String,
    pub kind: String,
    pub source: String,
    pub file_path: PathBuf,
    pub range: (usize, usize), // byte offsets [start, end)
    pub pagerank: f64,
}

/// A directed edge in the ASG.
#[derive(Debug, Clone)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub kind: EdgeKind,
}

/// The complete Abstract Semantic Graph.
#[derive(Debug, Clone, Default)]
pub struct Asg {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    /// Adjacency list: node_id -> list of outgoing edge indices.
    pub adjacency: HashMap<usize, Vec<usize>>,
    /// Reverse adjacency: node_id -> list of incoming edge indices.
    pub reverse_adjacency: HashMap<usize, Vec<usize>>,
    /// Symbol table: fully-qualified name -> node id.
    pub symbol_table: HashMap<String, usize>,
    /// File path -> list of node ids in that file.
    pub file_index: HashMap<PathBuf, Vec<usize>>,
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum AsgError {
    #[error("tree-sitter parse error: {0}")]
    ParseError(String),
    #[error("symbol not found: {0}")]
    SymbolNotFound(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// ASG Builder
// ---------------------------------------------------------------------------

/// Builds an ASG from source files.
pub struct AsgBuilder {
    parser: Parser,
    next_id: usize,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    adjacency: HashMap<usize, Vec<usize>>,
    reverse_adjacency: HashMap<usize, Vec<usize>>,
    symbol_table: HashMap<String, usize>,
    file_index: HashMap<PathBuf, Vec<usize>>,
}

impl AsgBuilder {
    pub fn new() -> Result<Self, AsgError> {
        let mut parser = Parser::new();
        let rust_lang: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
        parser
            .set_language(&rust_lang)
            .map_err(|e| AsgError::ParseError(e.to_string()))?;
        Ok(Self {
            parser,
            next_id: 0,
            nodes: Vec::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
            reverse_adjacency: HashMap::new(),
            symbol_table: HashMap::new(),
            file_index: HashMap::new(),
        })
    }

    /// Parse a single source file and add its nodes/edges to the graph.
    pub fn parse_file(&mut self, path: &Path) -> Result<(), AsgError> {
        let source = std::fs::read_to_string(path)?;
        let tree = self
            .parser
            .parse(&source.as_bytes(), None)
            .ok_or_else(|| AsgError::ParseError(format!("Failed to parse {}", path.display())))?;

        let file_path = path.to_path_buf();
        self.walk_tree(tree.root_node(), &source, &file_path, None);
        Ok(())
    }

    /// Recursively walk a tree-sitter node, creating ASG nodes and edges.
    fn walk_tree(
        &mut self,
        ts_node: TsNode,
        source: &str,
        file_path: &Path,
        parent_id: Option<usize>,
    ) {
        let kind = ts_node.kind();
        let node_source = ts_node
            .utf8_text(source.as_bytes())
            .unwrap_or("")
            .to_string();

        // Determine the node name from the first named child that is an identifier.
        let name = self.extract_name(ts_node, source);

        let id = self.next_id;
        self.next_id += 1;

        let node = Node {
            id,
            name: name.clone(),
            kind: kind.to_string(),
            source: node_source.clone(),
            file_path: file_path.to_path_buf(),
            range: (ts_node.start_byte(), ts_node.end_byte()),
            pagerank: 0.0,
        };

        // Register in symbol table if it's a definition.
        if self.is_definition(ts_node) {
            let qualified = self.qualified_name(&name, file_path);
            self.symbol_table.insert(qualified, id);
        }

        // Register in file index.
        self.file_index
            .entry(file_path.to_path_buf())
            .or_default()
            .push(id);

        self.nodes.push(node);

        // Create edge from parent.
        if let Some(pid) = parent_id {
            self.add_edge(pid, id, EdgeKind::Contains);
        }

        // Recurse into children.
        for i in 0..ts_node.named_child_count() {
            if let Some(child) = ts_node.named_child(i) {
                self.walk_tree(child, source, file_path, Some(id));
            }
        }
    }

    /// Extract a human-readable name from a tree-sitter node.
    fn extract_name(&self, ts_node: TsNode, source: &str) -> String {
        // Look for a child named "name" or an identifier.
        for i in 0..ts_node.named_child_count() {
            if let Some(child) = ts_node.named_child(i) {
                if child.kind() == "identifier" || child.kind() == "type_identifier" {
                    return child
                        .utf8_text(source.as_bytes())
                        .unwrap_or("")
                        .to_string();
                }
            }
        }
        // Fallback: use the node kind.
        ts_node.kind().to_string()
    }

    /// Check if a tree-sitter node represents a definition.
    fn is_definition(&self, ts_node: TsNode) -> bool {
        matches!(
            ts_node.kind(),
            "function_definition"
                | "struct_definition"
                | "enum_definition"
                | "impl_definition"
                | "trait_definition"
                | "type_alias"
                | "let_declaration"
        )
    }

    /// Build a qualified name for symbol resolution.
    fn qualified_name(&self, name: &str, file_path: &Path) -> String {
        let file_stem = file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        format!("{}::{}", file_stem, name)
    }

    /// Add an edge to the graph.
    fn add_edge(&mut self, from: usize, to: usize, kind: EdgeKind) {
        let edge_idx = self.edges.len();
        self.edges.push(Edge { from, to, kind });
        self.adjacency.entry(from).or_default().push(edge_idx);
        self.reverse_adjacency.entry(to).or_default().push(edge_idx);
    }

    /// Resolve cross-file symbol references (CALL -> DEFINITION).
    pub fn resolve_symbols(&mut self) {
        // Collect all call sites (identifier nodes that are not definitions).
        let call_sites: Vec<(usize, String)> = self
            .nodes
            .iter()
            .filter(|n| n.kind == "identifier" && !self.is_definition_kind(&n.kind))
            .map(|n| (n.id, n.name.clone()))
            .collect();

        for (call_id, name) in call_sites {
            // Try to find a matching definition in the symbol table.
            for (qualified, def_id) in &self.symbol_table {
                if qualified.ends_with(&format!("::{}", name)) {
                    self.add_edge(call_id, *def_id, EdgeKind::Calls);
                    break;
                }
            }
        }
    }

    fn is_definition_kind(&self, kind: &str) -> bool {
        matches!(
            kind,
            "function_definition"
                | "struct_definition"
                | "enum_definition"
                | "impl_definition"
                | "trait_definition"
                | "type_alias"
        )
    }

    /// Build the final ASG.
    pub fn build(mut self) -> Asg {
        self.resolve_symbols();
        Asg {
            nodes: self.nodes,
            edges: self.edges,
            adjacency: self.adjacency,
            reverse_adjacency: self.reverse_adjacency,
            symbol_table: self.symbol_table,
            file_index: self.file_index,
        }
    }
}

// ---------------------------------------------------------------------------
// PageRank Engine
// ---------------------------------------------------------------------------

/// Pure-Rust PageRank implementation using power iteration.
pub struct PageRankEngine {
    /// Damping factor (typically 0.85).
    pub damping: f64,
    /// Convergence threshold.
    pub epsilon: f64,
    /// Maximum iterations.
    pub max_iterations: usize,
}

impl Default for PageRankEngine {
    fn default() -> Self {
        Self {
            damping: 0.85,
            epsilon: 1e-6,
            max_iterations: 100,
        }
    }
}

impl PageRankEngine {
    pub fn new(damping: f64, epsilon: f64, max_iterations: usize) -> Self {
        Self {
            damping,
            epsilon,
            max_iterations,
        }
    }

    /// Run PageRank over the ASG, updating each node's `pagerank` field.
    pub fn run(&self, asg: &mut Asg) {
        let n = asg.nodes.len();
        if n == 0 {
            return;
        }

        // Build transition matrix as adjacency lists with out-degrees.
        let mut out_degree: Vec<usize> = vec![0; n];
        for edge in &asg.edges {
            out_degree[edge.from] += 1;
        }

        // Initialize PageRank uniformly.
        let mut pagerank: Vec<f64> = vec![1.0 / n as f64; n];
        let mut new_pagerank: Vec<f64> = vec![0.0; n];

        for _ in 0..self.max_iterations {
            // Reset new values.
            for val in new_pagerank.iter_mut() {
                *val = (1.0 - self.damping) / n as f64;
            }

            // Distribute PageRank along edges.
            for edge in &asg.edges {
                if out_degree[edge.from] > 0 {
                    let contribution = self.damping * pagerank[edge.from] / out_degree[edge.from] as f64;
                    new_pagerank[edge.to] += contribution;
                }
            }

            // Handle dangling nodes (no out-edges): distribute evenly.
            for i in 0..n {
                if out_degree[i] == 0 {
                    new_pagerank[i] += self.damping * pagerank[i] / n as f64;
                }
            }

            // Check convergence.
            let diff: f64 = pagerank
                .iter()
                .zip(new_pagerank.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

            pagerank.clone_from_slice(&new_pagerank);

            if diff < self.epsilon {
                break;
            }
        }

        // Write back to nodes.
        for (node, pr) in asg.nodes.iter_mut().zip(pagerank.iter()) {
            node.pagerank = *pr;
        }
    }
}

// ---------------------------------------------------------------------------
// ASG Extension: Chunk Registry Integration
// ---------------------------------------------------------------------------

/// Trait for extracting functional chunks (functions, structs, impls) from an ASG.
pub trait ChunkExtractor {
    /// Returns node IDs that represent functional chunks.
    fn extract_chunks(&self) -> Vec<usize>;
}

impl ChunkExtractor for Asg {
    fn extract_chunks(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|n| {
                n.kind == "function_definition"
                    || n.kind == "struct_definition"
                    || n.kind == "impl_definition"
                    || n.kind == "enum_definition"
                    || n.kind == "trait_definition"
            })
            .map(|n| n.id)
            .collect()
    }
}

/// Build an ASG from a directory of Rust source files.
pub fn build_asg_from_dir(dir: &Path) -> Result<Asg, AsgError> {
    let mut builder = AsgBuilder::new()?;
    let mut files: Vec<PathBuf> = Vec::new();

    for entry in walkdir::WalkDir::new(dir).into_iter().filter_map(Result::ok) {
        if entry.path().extension().map_or(false, |ext| ext == "rs") {
            files.push(entry.path().to_path_buf());
        }
    }

    files.sort();
    for file in &files {
        builder.parse_file(file)?;
    }

    Ok(builder.build())
}

/// A shared, thread-safe ASG handle for concurrent access.
#[derive(Clone)]
pub struct SharedAsg {
    pub inner: Arc<Asg>,
}

impl SharedAsg {
    pub fn new(asg: Asg) -> Self {
        Self {
            inner: Arc::new(asg),
        }
    }

    pub fn get_node(&self, id: usize) -> Option<&Node> {
        self.inner.nodes.get(id)
    }

    pub fn get_node_by_name(&self, name: &str) -> Option<&Node> {
        self.inner
            .symbol_table
            .get(name)
            .and_then(|id| self.inner.nodes.get(*id))
    }
}
