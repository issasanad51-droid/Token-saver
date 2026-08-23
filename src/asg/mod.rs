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
pub mod leiden;
pub mod pagerank;
pub mod source;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tree_sitter::{Node as TsNode, Parser, Tree};

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

/// Edge kinds that connect ASG nodes semantically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    /// A call-site node references a definition node.
    Calls,
    /// A node is contained within (owns) another node.
    Contains,
    /// A node imports a symbol from another file.
    Imports,
    /// A node's signature or body references a type definition.
    References,
    /// A node implements a trait or interface.
    Implements,
    /// A node is a field of a struct.
    FieldOf,
    /// A node is a variant of an enum.
    VariantOf,
    /// A cross-language bridge edge (SQL query strings, API endpoints).
    Bridge,
}

/// A semantic node in the ASG.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Node {
    /// Dense runtime ID used by the low-latency retrieval path.
    pub id: usize,
    /// Stable global ASG tracker (`crate::module::kind::name`).
    pub tracker_id: String,
    pub name: String,
    pub kind: String,
    pub source: String,
    pub file_path: PathBuf,
    pub range: (usize, usize), // byte offsets [start, end)
    pub pagerank: f64,
}

/// A directed edge in the ASG.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub kind: EdgeKind,
}

/// The complete Abstract Semantic Graph.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
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
// Historical full-AST builder (kept for API compatibility)
// ---------------------------------------------------------------------------

/// Builds an ASG from source files. New workspace indexing uses the sparse v2
/// builder through [`build_asg_from_dir`].
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
            .map_err(|error| AsgError::ParseError(error.to_string()))?;
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

    pub fn parse_file(&mut self, path: &Path) -> Result<(), AsgError> {
        let source = std::fs::read_to_string(path)?;
        let tree = self
            .parser
            .parse(source.as_bytes(), None)
            .ok_or_else(|| AsgError::ParseError(format!("failed to parse {}", path.display())))?;
        self.walk_tree(tree, &source, path);
        Ok(())
    }

    fn walk_tree(&mut self, tree: Tree, source: &str, file_path: &Path) {
        self.walk_node(tree.root_node(), source, file_path, None);
    }

    fn walk_node(
        &mut self,
        ts_node: TsNode,
        source: &str,
        file_path: &Path,
        parent_id: Option<usize>,
    ) {
        let name = self.extract_name(ts_node, source);
        let id = self.next_id;
        self.next_id += 1;
        let tracker_id = self.qualified_name(&name, file_path);

        self.nodes.push(Node {
            id,
            tracker_id: tracker_id.clone(),
            name: name.clone(),
            kind: ts_node.kind().to_string(),
            source: ts_node
                .utf8_text(source.as_bytes())
                .unwrap_or_default()
                .to_string(),
            file_path: file_path.to_path_buf(),
            range: (ts_node.start_byte(), ts_node.end_byte()),
            pagerank: 0.0,
        });

        if self.is_definition(ts_node) {
            self.symbol_table.insert(tracker_id, id);
        }
        self.file_index
            .entry(file_path.to_path_buf())
            .or_default()
            .push(id);
        if let Some(parent) = parent_id {
            self.add_edge(parent, id, EdgeKind::Contains);
        }
        for index in 0..ts_node.named_child_count() {
            if let Some(child) = ts_node.named_child(index) {
                self.walk_node(child, source, file_path, Some(id));
            }
        }
    }

    fn extract_name(&self, ts_node: TsNode, source: &str) -> String {
        if let Some(name) = ts_node.child_by_field_name("name") {
            return name
                .utf8_text(source.as_bytes())
                .unwrap_or_default()
                .to_string();
        }
        for index in 0..ts_node.named_child_count() {
            if let Some(child) = ts_node.named_child(index) {
                if matches!(child.kind(), "identifier" | "type_identifier") {
                    return child
                        .utf8_text(source.as_bytes())
                        .unwrap_or_default()
                        .to_string();
                }
            }
        }
        ts_node.kind().to_string()
    }

    fn is_definition(&self, ts_node: TsNode) -> bool {
        matches!(
            ts_node.kind(),
            "function_item"
                | "struct_item"
                | "enum_item"
                | "impl_item"
                | "trait_item"
                | "type_item"
                | "const_item"
                | "static_item"
                | "macro_definition"
        )
    }

    fn qualified_name(&self, name: &str, file_path: &Path) -> String {
        let file_stem = file_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("unknown");
        format!("{file_stem}::{name}")
    }

    fn add_edge(&mut self, from: usize, to: usize, kind: EdgeKind) {
        let edge_index = self.edges.len();
        self.edges.push(Edge { from, to, kind });
        self.adjacency.entry(from).or_default().push(edge_index);
        self.reverse_adjacency
            .entry(to)
            .or_default()
            .push(edge_index);
    }

    pub fn resolve_symbols(&mut self) {
        let call_sites: Vec<(usize, String)> = self
            .nodes
            .iter()
            .filter(|node| node.kind == "identifier")
            .map(|node| (node.id, node.name.clone()))
            .collect();

        for (call_id, name) in call_sites {
            let suffix = format!("::{name}");
            let target = self
                .symbol_table
                .iter()
                .find(|(qualified, _)| qualified.ends_with(&suffix))
                .map(|(_, definition_id)| *definition_id);
            if let Some(definition_id) = target {
                self.add_edge(call_id, definition_id, EdgeKind::Calls);
            }
        }
    }

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
// Weighted Personalized PageRank Engine
// ---------------------------------------------------------------------------

/// Structural importance assigned to each semantic edge type.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PageRankEdgeWeights {
    pub calls: f64,
    pub references: f64,
    pub contains: f64,
    pub imports: f64,
    pub implements: f64,
    pub field_of: f64,
    pub variant_of: f64,
    pub bridge: f64,
}

impl Default for PageRankEdgeWeights {
    fn default() -> Self {
        Self {
            calls: 1.0,
            references: 0.7,
            contains: 0.15,
            imports: 0.5,
            implements: 0.9,
            field_of: 0.25,
            variant_of: 0.25,
            bridge: 0.6,
        }
    }
}

impl PageRankEdgeWeights {
    fn for_kind(&self, kind: EdgeKind) -> f64 {
        let weight = match kind {
            EdgeKind::Calls => self.calls,
            EdgeKind::References => self.references,
            EdgeKind::Contains => self.contains,
            EdgeKind::Imports => self.imports,
            EdgeKind::Implements => self.implements,
            EdgeKind::FieldOf => self.field_of,
            EdgeKind::VariantOf => self.variant_of,
            EdgeKind::Bridge => self.bridge,
        };
        if weight.is_finite() {
            weight.max(0.0)
        } else {
            0.0
        }
    }
}

/// Tunable weighted Personalized PageRank parameters used by both startup
/// centrality and query-time structural retrieval.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PersonalizedPageRankConfig {
    /// Damping factor. Invalid values are clamped into `[0, 0.99]`.
    pub damping: f64,
    /// L1 convergence threshold.
    pub epsilon: f64,
    /// Hard iteration cap.
    pub max_iterations: usize,
    pub edge_weights: PageRankEdgeWeights,
}

impl Default for PersonalizedPageRankConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            epsilon: 1e-8,
            max_iterations: 100,
            edge_weights: PageRankEdgeWeights::default(),
        }
    }
}

/// Pure-Rust, edge-weighted Personalized PageRank using two dense buffers.
#[derive(Debug, Clone)]
pub struct PageRankEngine {
    pub config: PersonalizedPageRankConfig,
}

impl Default for PageRankEngine {
    fn default() -> Self {
        Self::from_config(PersonalizedPageRankConfig::default())
    }
}

impl PageRankEngine {
    /// Backwards-compatible constructor for the original uniform engine.
    pub fn new(damping: f64, epsilon: f64, max_iterations: usize) -> Self {
        Self::from_config(PersonalizedPageRankConfig {
            damping,
            epsilon,
            max_iterations,
            ..PersonalizedPageRankConfig::default()
        })
    }

    pub fn from_config(config: PersonalizedPageRankConfig) -> Self {
        Self { config }
    }

    pub fn run(&self, asg: &mut Asg) {
        let scores = self.personalized_scores(asg, &[]);
        for (node, score) in asg.nodes.iter_mut().zip(scores) {
            node.pagerank = score;
        }
    }

    /// Compute weighted PPR without mutating the graph.
    ///
    /// Convenience wrapper: builds a one-shot [`PprIndex`] and runs the
    /// iteration on it. Callers on the query hot path should build the index
    /// once (per graph snapshot) and call [`Self::personalized_scores_on`]
    /// instead.
    pub fn personalized_scores(&self, asg: &Asg, seeds: &[(usize, f64)]) -> Vec<f64> {
        let index = PprIndex::build(asg, &self.config.edge_weights);
        self.personalized_scores_on(&index, seeds)
    }
}

/// Precomputed weighted adjacency for repeated PPR queries over a fixed
/// graph snapshot, stored in flat CSR form.
///
/// Building the adjacency is O(E) with one `Vec<Vec<_>>` worth of
/// allocations. Doing that on *every* query (the autocomplete hot path)
/// wasted both the allocation churn and the cache-hostile scatter of
/// nested vectors; `PprIndex` is built once per graph snapshot and shared
/// across queries, so each query becomes a pure numeric sweep over frozen
/// arrays.
#[derive(Debug, Clone, Default)]
pub struct PprIndex {
    node_count: usize,
    /// Flat CSR adjacency: targets of each node's out-edges.
    targets: Vec<usize>,
    /// Edge weights, parallel to `targets`.
    weights: Vec<f64>,
    /// `offsets[node]..offsets[node + 1]` slices this node's out-edges.
    offsets: Vec<usize>,
    /// Total weighted out-degree per node (0 = dangling node).
    out_weight: Vec<f64>,
}

impl PprIndex {
    /// Snapshot the weighted adjacency of `asg` under `weights`.
    pub fn build(asg: &Asg, weights: &PageRankEdgeWeights) -> Self {
        let node_count = asg.nodes.len();
        let mut targets = Vec::with_capacity(asg.edges.len());
        let mut weights_out = Vec::with_capacity(asg.edges.len());
        let mut out_weight = vec![0.0f64; node_count];

        // Bucket edges by source with a counting pass so the flat arrays are
        // ordered by node id without per-node allocations.
        let mut counts = vec![0usize; node_count + 1];
        for edge in &asg.edges {
            if edge.from >= node_count
                || edge.to >= node_count
                || edge.from == edge.to
                || weights.for_kind(edge.kind) <= 0.0
            {
                continue;
            }
            counts[edge.from + 1] += 1;
        }
        let mut running = 0usize;
        for slot in counts.iter_mut() {
            running += *slot;
            *slot = running;
        }
        targets.resize(counts[node_count], 0);
        weights_out.resize(counts[node_count], 0.0);
        let mut cursor = counts.clone();
        for edge in &asg.edges {
            if edge.from >= node_count
                || edge.to >= node_count
                || edge.from == edge.to
                || weights.for_kind(edge.kind) <= 0.0
            {
                continue;
            }
            let weight = weights.for_kind(edge.kind);
            let slot = cursor[edge.from];
            targets[slot] = edge.to;
            weights_out[slot] = weight;
            out_weight[edge.from] += weight;
            cursor[edge.from] += 1;
        }
        let offsets = counts;

        Self {
            node_count,
            targets,
            weights: weights_out,
            offsets,
            out_weight,
        }
    }
}

impl PageRankEngine {
    /// Compute weighted PPR over a precomputed [`PprIndex`] without
    /// rebuilding the adjacency. The seeds are `(node id, weight)` pairs;
    /// they are normalized into the teleport vector internally.
    ///
    /// The iteration indexes several parallel dense buffers by node id;
    /// iterator plumbing would obscure the numeric kernel.
    #[allow(clippy::needless_range_loop)]
    pub fn personalized_scores_on(&self, index: &PprIndex, seeds: &[(usize, f64)]) -> Vec<f64> {
        let node_count = index.node_count;
        if node_count == 0 {
            return Vec::new();
        }

        let mut teleport = vec![0.0; node_count];
        for &(id, weight) in seeds {
            if id < node_count && weight.is_finite() && weight > 0.0 {
                teleport[id] += weight;
            }
        }
        let seed_mass: f64 = teleport.iter().sum();
        if seed_mass > 0.0 {
            for value in &mut teleport {
                *value /= seed_mass;
            }
        } else {
            teleport.fill(1.0 / node_count as f64);
        }

        let damping = if self.config.damping.is_finite() {
            self.config.damping.clamp(0.0, 0.99)
        } else {
            0.85
        };
        let epsilon = if self.config.epsilon.is_finite() {
            self.config.epsilon.max(1e-15)
        } else {
            1e-8
        };
        let max_iterations = self.config.max_iterations.max(1);

        let mut rank = teleport.clone();
        let mut next = vec![0.0; node_count];
        for _ in 0..max_iterations {
            next.fill(0.0);
            let mut dangling_mass = 0.0;

            for from in 0..node_count {
                let out = index.out_weight[from];
                if out == 0.0 {
                    dangling_mass += damping * rank[from];
                    continue;
                }
                let scale = damping * rank[from] / out;
                for slot in index.offsets[from]..index.offsets[from + 1] {
                    next[index.targets[slot]] += scale * index.weights[slot];
                }
            }

            for value in 0..node_count {
                next[value] += ((1.0 - damping) + dangling_mass) * teleport[value];
            }

            let difference: f64 = rank
                .iter()
                .zip(&next)
                .map(|(old, new)| (old - new).abs())
                .sum();
            rank.copy_from_slice(&next);
            if difference < epsilon {
                break;
            }
        }

        let total: f64 = rank.iter().sum();
        if total.is_finite() && total > 0.0 {
            for score in &mut rank {
                *score /= total;
            }
        }
        rank
    }
}

// ---------------------------------------------------------------------------
// Runtime graph assembly
// ---------------------------------------------------------------------------

pub trait ChunkExtractor {
    fn extract_chunks(&self) -> Vec<usize>;
}

impl ChunkExtractor for Asg {
    fn extract_chunks(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|node| {
                matches!(
                    node.kind.as_str(),
                    "fn"
                        | "struct"
                        | "impl"
                        | "enum"
                        | "trait"
                        | "type"
                        | "const"
                        | "static"
                        | "macro"
                        | "mod"
                        | "function_item"
                        | "struct_item"
                        | "impl_item"
                        | "enum_item"
                        | "trait_item"
                )
            })
            .map(|node| node.id)
            .collect()
    }
}

/// Build the sparse semantic ASG used by the live retrieval pipeline.
pub fn build_asg_from_dir(dir: &Path) -> Result<Asg, AsgError> {
    build_asg_from_dir_with_config(dir, PersonalizedPageRankConfig::default())
}

/// Build and rank an ASG with custom weighted-PPR settings.
pub fn build_asg_from_dir_with_config(
    dir: &Path,
    pagerank: PersonalizedPageRankConfig,
) -> Result<Asg, AsgError> {
    let sources = source::SourceSet::from_dir(dir, "rs")?;
    let mut semantic_builder = builder::AsgBuilder::new(&sources)?.with_crate_root(dir);
    semantic_builder.parse()?;
    let semantic = semantic_builder.build();
    let mut runtime = runtime_graph(&semantic);
    // Cross-language semantic bridge edges (SQL strings, API endpoints).
    let bridge_mapper = crate::bridge::BridgeMapper::new();
    let bridge_edges = bridge_mapper.link_bridges(&mut runtime);
    if bridge_edges > 0 {
        tracing::debug!("linked {bridge_edges} cross-language bridge edges");
    }
    PageRankEngine::from_config(pagerank).run(&mut runtime);
    Ok(runtime)
}

fn runtime_graph(graph: &graph::AsgGraph<'_>) -> Asg {
    let mut runtime = Asg::default();
    let mut dense_ids: HashMap<graph::NodeId, usize> = HashMap::new();

    for data in graph.nodes() {
        let id = runtime.nodes.len();
        let name = data
            .id
            .as_str()
            .rsplit("::")
            .next()
            .unwrap_or_default()
            .to_string();
        let kind = node_type_label(data.node_type).to_string();
        let file_path = data.file.clone().unwrap_or_default();

        runtime.nodes.push(Node {
            id,
            tracker_id: data.id.to_string(),
            name: name.clone(),
            kind,
            source: data.body.to_string(),
            file_path: file_path.clone(),
            range: data.range,
            pagerank: data.pagerank_score,
        });
        dense_ids.insert(data.id.clone(), id);
        runtime.file_index.entry(file_path).or_default().push(id);
        runtime.symbol_table.insert(data.id.to_string(), id);
        runtime.symbol_table.entry(name).or_insert(id);
    }

    let mut seen = std::collections::HashSet::new();
    for data in graph.nodes() {
        let Some(&from) = dense_ids.get(&data.id) else {
            continue;
        };
        for edge in graph.outgoing_edges(&data.id) {
            let Some(&to) = dense_ids.get(&edge.node) else {
                continue;
            };
            let kind = runtime_edge_kind(edge.kind);
            if from == to || !seen.insert((from, to, kind)) {
                continue;
            }
            let edge_index = runtime.edges.len();
            runtime.edges.push(Edge { from, to, kind });
            runtime.adjacency.entry(from).or_default().push(edge_index);
            runtime
                .reverse_adjacency
                .entry(to)
                .or_default()
                .push(edge_index);
        }
    }
    runtime
}

fn node_type_label(kind: graph::NodeType) -> &'static str {
    match kind {
        graph::NodeType::Struct => "struct",
        graph::NodeType::Impl => "impl",
        graph::NodeType::Fn => "fn",
        graph::NodeType::Enum => "enum",
        graph::NodeType::Macro => "macro",
        graph::NodeType::Trait => "trait",
        graph::NodeType::Mod => "mod",
        graph::NodeType::Type => "type",
        graph::NodeType::Const => "const",
        graph::NodeType::Static => "static",
        graph::NodeType::Use => "use",
    }
}

fn runtime_edge_kind(kind: graph::EdgeKind) -> EdgeKind {
    match kind {
        graph::EdgeKind::Calls => EdgeKind::Calls,
        graph::EdgeKind::References => EdgeKind::References,
        graph::EdgeKind::Contains => EdgeKind::Contains,
        graph::EdgeKind::Imports => EdgeKind::Imports,
        graph::EdgeKind::Implements => EdgeKind::Implements,
        graph::EdgeKind::FieldOf => EdgeKind::FieldOf,
        graph::EdgeKind::VariantOf => EdgeKind::VariantOf,
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppr_index_matches_oneshot_path() {
        // The cached CSR index must produce byte-identical scores to the
        // one-shot adjacency rebuild it replaced.
        let asg = Asg {
            nodes: (0..8).map(|i| node(i, &format!("n{i}"))).collect(),
            edges: vec![
                Edge { from: 0, to: 1, kind: EdgeKind::Calls },
                Edge { from: 0, to: 2, kind: EdgeKind::References },
                Edge { from: 1, to: 3, kind: EdgeKind::Calls },
                Edge { from: 2, to: 3, kind: EdgeKind::Implements },
                Edge { from: 3, to: 0, kind: EdgeKind::Calls },
                Edge { from: 4, to: 5, kind: EdgeKind::Contains },
                Edge { from: 5, to: 6, kind: EdgeKind::Contains },
                Edge { from: 6, to: 4, kind: EdgeKind::Contains },
                Edge { from: 7, to: 7, kind: EdgeKind::Calls }, // self-loop: skipped
                Edge { from: 9, to: 1, kind: EdgeKind::Calls }, // out-of-range: skipped
            ],
            ..Asg::default()
        };
        let engine = PageRankEngine::default();
        let index = PprIndex::build(&asg, &engine.config.edge_weights);
        for seeds in [
            vec![(0usize, 1.0f64)],
            vec![(3, 1.0), (4, 2.0)],
            Vec::new(), // uniform teleport fallback
        ] {
            let via_index = engine.personalized_scores_on(&index, &seeds);
            let via_graph = engine.personalized_scores(&asg, &seeds);
            assert_eq!(via_index.len(), via_graph.len());
            for (a, b) in via_index.iter().zip(&via_graph) {
                assert!(
                    (a - b).abs() < 1e-12,
                    "scores differ: index={a} graph={b} seeds={seeds:?}"
                );
            }
        }
    }

    #[test]
    fn ppr_index_empty_graph_is_safe() {
        let engine = PageRankEngine::default();
        let index = PprIndex::build(&Asg::default(), &engine.config.edge_weights);
        assert!(engine.personalized_scores_on(&index, &[(0, 1.0)]).is_empty());
    }

    fn node(id: usize, name: &str) -> Node {
        Node {
            id,
            tracker_id: format!("crate::fn::{name}"),
            name: name.to_string(),
            kind: "fn".to_string(),
            source: String::new(),
            file_path: PathBuf::new(),
            range: (0, 0),
            pagerank: 0.0,
        }
    }

    #[test]
    fn weighted_ppr_prefers_call_edge_over_containment() {
        let mut asg = Asg {
            nodes: vec![node(0, "seed"), node(1, "called"), node(2, "contained")],
            edges: vec![
                Edge {
                    from: 0,
                    to: 1,
                    kind: EdgeKind::Calls,
                },
                Edge {
                    from: 0,
                    to: 2,
                    kind: EdgeKind::Contains,
                },
            ],
            ..Asg::default()
        };
        let engine = PageRankEngine::default();
        let scores = engine.personalized_scores(&asg, &[(0, 1.0)]);
        assert!(scores[1] > scores[2]);

        engine.run(&mut asg);
        let total: f64 = asg.nodes.iter().map(|node| node.pagerank).sum();
        assert!((total - 1.0).abs() < 1e-8);
    }

    #[test]
    fn invalid_seed_and_numeric_config_are_safe() {
        let asg = Asg {
            nodes: vec![node(0, "only")],
            ..Asg::default()
        };
        let engine = PageRankEngine::from_config(PersonalizedPageRankConfig {
            damping: f64::NAN,
            epsilon: f64::NAN,
            max_iterations: 0,
            ..PersonalizedPageRankConfig::default()
        });
        let scores = engine.personalized_scores(&asg, &[(99, 1.0), (0, f64::NAN)]);
        assert_eq!(scores, vec![1.0]);
    }
}
