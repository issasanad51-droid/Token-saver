//! v2 Foundational ASG Graph Model
//!
//! `AsgGraph` is the in-memory **directed Abstract Syntax Graph** that
//! underpins the entire token-saver pipeline. It is the single source of
//! truth for structural relationships between code entities and it owns the
//! YAML schema contract used both to hand context to and to receive patches
//! back from the external LLM.
//!
//! # Schema contract (per node)
//!
//! ```yaml
//! - id: crate::server::mod::fn::run_server   # unique global string tracker
//!   type: fn                                 # node type
//!   body: |                                  # isolated clean source, ONLY this entity
//!     pub fn run_server() { /* ... */ }
//!   incoming_edges:
//!     - node: crate::server::main::fn::main
//!       kind: calls
//!   outgoing_edges:
//!     - node: crate::server::state::struct::ServerState
//!       kind: references
//!   pagerank_score: 0.4181
//! ```
//!
//! # Design decisions
//!
//! - **`petgraph::graph::DiGraph` backing store.** Dense, contiguous
//!   `NodeIndex` handles make the PageRank power-iteration (next phase) a
//!   cache-friendly dense-array pass. `NodeIndex` is an internal detail and is
//!   **never serialized** — only the stable string `NodeId` crosses the wire.
//! - **`NodeId` global string tracker.** The stable, serialized identity
//!   (`crate::module::kind::name`). A bidirectional `HashMap` gives O(1)
//!   translation between `NodeId` and `NodeIndex`.
//! - **`Cow<'a, str>` bodies.** `Borrowed` when the graph is built directly
//!   over source buffers (zero-copy parse, zero-copy YAML emission);
//!   `Owned` after a YAML round-trip. Pair the graph with a `SourceSet`
//!   arena (parsing layer, next phase) so every body borrows from one owner
//!   under a single lifetime `'a`.
//! - **Deterministic output.** Edges are sorted by `(node id, kind)` before
//!   serialization so identical graphs always emit byte-identical YAML — this
//!   is what makes the lossless write-back deterministic.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;

use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef as _;
use petgraph::Direction;
use serde::{Deserialize, Serialize};

use super::pagerank::PageRankConfig;

// ---------------------------------------------------------------------------
// Identity: global string tracker
// ---------------------------------------------------------------------------

/// Global string tracker id, e.g. `crate::server::mod::fn::run_server`.
///
/// Serialized `#[serde(transparent)]` so YAML keeps it as a bare scalar
/// (no wrapper object), which keeps the schema hyper-compact.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(pub String);

impl NodeId {
    /// Build a tracker id from a raw string.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Build a fully-qualified tracker id from path segments.
    ///
    /// `NodeId::qualified("crate", &["server", "mod"], "fn", "run_server")`
    /// yields `crate::server::mod::fn::run_server`.
    pub fn qualified(
        crate_root: &str,
        module_path: &[&str],
        entity_kind: &str,
        entity_name: &str,
    ) -> Self {
        let mut parts: Vec<&str> = Vec::with_capacity(module_path.len() + 3);
        parts.push(crate_root);
        parts.extend_from_slice(module_path);
        parts.push(entity_kind);
        parts.push(entity_name);
        Self(parts.join("::"))
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------------
// Discriminants
// ---------------------------------------------------------------------------

/// Discriminant for the `type` field of the schema contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeType {
    Struct,
    Impl,
    Fn,
    Enum,
    Macro,
    Trait,
    Mod,
    Type,
    Const,
    Static,
    Use,
}

/// Discriminant for an edge's `kind` label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EdgeKind {
    /// `from` calls / invokes `to`.
    Calls,
    /// `from` lexically owns `to`.
    Contains,
    /// `from` imports a symbol defined at `to`.
    Imports,
    /// `from` implements trait `to`.
    Implements,
    /// `from` is a field of struct `to`.
    FieldOf,
    /// `from` is a variant of enum `to`.
    VariantOf,
    /// `from`'s type signature references `to`.
    References,
}

// ---------------------------------------------------------------------------
// Serialized schema types (the YAML contract)
// ---------------------------------------------------------------------------

/// A serialized edge reference: the neighbor's `node` id plus the edge `kind`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeRef {
    pub node: NodeId,
    pub kind: EdgeKind,
}

/// The serialized view of a node — exactly the schema contract handed to the
/// LLM and persisted to disk.
///
/// `body` is a `Cow<'a, str>` so serialization emits the borrowed source slice
/// with zero allocation when the graph borrows from a live source buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsgNode<'a> {
    pub id: NodeId,
    #[serde(rename = "type")]
    pub node_type: NodeType,
    #[serde(default, skip_serializing_if = "cow_is_empty")]
    pub body: Cow<'a, str>,
    #[serde(default)]
    pub incoming_edges: Vec<EdgeRef>,
    #[serde(default)]
    pub outgoing_edges: Vec<EdgeRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pagerank_score: Option<f64>,
}

/// Top-level YAML document. `version` lets the schema evolve without breaking
/// on-disk graphs.
#[derive(Debug, Serialize, Deserialize)]
pub struct AsgGraphSnapshot<'a> {
    pub version: u32,
    pub nodes: Vec<AsgNode<'a>>,
}

/// serde helper: `Cow<str>` has no inherent `is_empty` (it is `Deref`-provided
/// from `str`), so expose a free function for `skip_serializing_if`.
fn cow_is_empty(c: &Cow<'_, str>) -> bool {
    c.is_empty()
}

// ---------------------------------------------------------------------------
// In-memory node data (petgraph payload)
// ---------------------------------------------------------------------------

/// Owning data attached to each petgraph node.
#[derive(Debug, Clone)]
pub struct AsgNodeData<'a> {
    pub id: NodeId,
    pub node_type: NodeType,
    pub body: Cow<'a, str>,
    pub pagerank_score: f64,
    /// Originating file (used by the lossless write-back phase).
    pub file: Option<PathBuf>,
    /// Byte range `[start, end)` within `file` — preserved so edits merge back
    /// deterministically without re-parsing the source.
    pub range: (usize, usize),
}

// ---------------------------------------------------------------------------
// The graph
// ---------------------------------------------------------------------------

/// In-memory directed Abstract Syntax Graph.
pub struct AsgGraph<'a> {
    graph: DiGraph<AsgNodeData<'a>, EdgeKind>,
    id_to_index: HashMap<NodeId, NodeIndex>,
    index_to_id: HashMap<NodeIndex, NodeId>,
}

impl<'a> AsgGraph<'a> {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            id_to_index: HashMap::new(),
            index_to_id: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Mutators
    // -----------------------------------------------------------------------

    /// Insert a node. Fails on a duplicate tracker id.
    pub fn add_node(
        &mut self,
        id: NodeId,
        node_type: NodeType,
        body: impl Into<Cow<'a, str>>,
    ) -> Result<NodeIndex, AsgGraphError> {
        if self.id_to_index.contains_key(&id) {
            return Err(AsgGraphError::DuplicateNode(id.to_string()));
        }
        let ix = self.graph.add_node(AsgNodeData {
            id: id.clone(),
            node_type,
            body: body.into(),
            pagerank_score: 0.0,
            file: None,
            range: (0, 0),
        });
        self.id_to_index.insert(id.clone(), ix);
        self.index_to_id.insert(ix, id);
        Ok(ix)
    }

    /// Insert a directed edge between two existing nodes.
    pub fn add_edge(
        &mut self,
        from: &NodeId,
        to: &NodeId,
        kind: EdgeKind,
    ) -> Result<EdgeIndex, AsgGraphError> {
        let f = *self
            .id_to_index
            .get(from)
            .ok_or_else(|| AsgGraphError::NodeNotFound(from.to_string()))?;
        let t = *self
            .id_to_index
            .get(to)
            .ok_or_else(|| AsgGraphError::NodeNotFound(to.to_string()))?;
        Ok(self.graph.add_edge(f, t, kind))
    }

    /// Attach provenance metadata to an existing node (used by the parser and
    /// by the lossless write-back phase).
    pub fn attach_location(
        &mut self,
        id: &NodeId,
        file: PathBuf,
        range: (usize, usize),
    ) -> Result<(), AsgGraphError> {
        let ix = *self
            .id_to_index
            .get(id)
            .ok_or_else(|| AsgGraphError::NodeNotFound(id.to_string()))?;
        let data = &mut self.graph[ix];
        data.file = Some(file);
        data.range = range;
        Ok(())
    }

    /// Apply PageRank scores (computed by the next-phase engine) onto node
    /// data. Scores land in the YAML via `pagerank_score`.
    pub fn stamp_scores<I: IntoIterator<Item = (NodeId, f64)>>(&mut self, scores: I) {
        for (id, score) in scores {
            if let Some(data) = self.node_mut(&id) {
                data.pagerank_score = score;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    #[inline]
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    #[inline]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn node(&self, id: &NodeId) -> Option<&AsgNodeData<'a>> {
        self.id_to_index.get(id).map(|ix| &self.graph[*ix])
    }

    pub fn node_mut(&mut self, id: &NodeId) -> Option<&mut AsgNodeData<'a>> {
        self.id_to_index.get(id).map(|ix| &mut self.graph[*ix])
    }

    pub fn contains(&self, id: &NodeId) -> bool {
        self.id_to_index.contains_key(id)
    }

    /// Iterate all node payloads (dense, contiguous — ideal for the dense
    /// PageRank vector pass).
    pub fn nodes(&self) -> impl Iterator<Item = &AsgNodeData<'a>> {
        self.graph.node_weights()
    }

    /// Schema-view edges leaving `id` (sorted by `(node, kind)`).
    pub fn outgoing_edges(&self, id: &NodeId) -> Vec<EdgeRef> {
        self.edges_of(id).1
    }

    /// Schema-view edges entering `id` (sorted by `(node, kind)`).
    pub fn incoming_edges(&self, id: &NodeId) -> Vec<EdgeRef> {
        self.edges_of(id).0
    }

    /// Translate a tracker id to its dense petgraph handle.
    pub fn index_of(&self, id: &NodeId) -> Option<NodeIndex> {
        self.id_to_index.get(id).copied()
    }

    /// Translate a dense petgraph handle back to its tracker id.
    pub fn id_of(&self, ix: NodeIndex) -> Option<&NodeId> {
        self.index_to_id.get(&ix)
    }

    /// Access the raw petgraph (PageRank / traversal phase).
    pub fn graph(&self) -> &DiGraph<AsgNodeData<'a>, EdgeKind> {
        &self.graph
    }

    /// Mutable access to the raw petgraph.
    pub fn graph_mut(&mut self) -> &mut DiGraph<AsgNodeData<'a>, EdgeKind> {
        &mut self.graph
    }

    // -----------------------------------------------------------------------
    // PageRank (custom power iteration)
    // -----------------------------------------------------------------------

    /// Compute local PageRank centrality via dense power iteration.
    ///
    /// Unlike `petgraph::algo::page_rank` (which is `f32`-weighted and tuned
    /// for web graphs), this is a hand-rolled **f64** engine that runs directly
    /// over the contiguous `NodeIndex` space:
    ///
    /// 1. Build a CSR (compressed sparse row) adjacency once — the code graph
    ///    is static between edits, so the CSR is amortized across queries.
    /// 2. Iterate with exactly two `Vec<f64>` buffers (zero per-iteration
    ///    allocation), handling dangling nodes via uniform redistribution.
    ///
    /// Returns `(NodeId, score)` pairs in dense `NodeIndex` order (deterministic),
    /// normalized so scores sum to 1.0.
    ///
    /// Delegates to the custom kind-weighted + personalized engine in
    /// [`super::pagerank`]; the default config reduces to vanilla PageRank.
    pub fn page_rank(&self, config: &PageRankConfig) -> Vec<(NodeId, f64)> {
        super::pagerank::pagerank(self, config)
    }

    /// Run PageRank and stamp the resulting scores onto node data (also
    /// surfaces in the YAML via `pagerank_score`).
    pub fn compute_page_rank(&mut self, config: &PageRankConfig) {
        super::pagerank::stamp_pagerank(self, config);
    }

    /// Return the top-`k` nodes by stamped `pagerank_score` (descending).
    ///
    /// This is the *structural centrality* stream feeding the RRF fusion —
    /// deeply-integrated utilities and foundational schemas bubble to the top.
    pub fn top_nodes(&self, k: usize) -> Vec<&AsgNodeData<'a>> {
        let mut nodes: Vec<&AsgNodeData<'a>> = self.graph.node_weights().collect();
        nodes.sort_by(|a, b| b.pagerank_score.total_cmp(&a.pagerank_score));
        nodes.truncate(k);
        nodes
    }

    // -----------------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------------

    /// Materialize the schema-contract snapshot. Bodies stay `Borrowed` when
    /// possible, so building the context YAML copies zero source bytes.
    pub fn to_snapshot(&self) -> AsgGraphSnapshot<'a> {
        let nodes = self
            .graph
            .node_indices()
            .map(|ix| {
                let data = &self.graph[ix];
                let (incoming_edges, outgoing_edges) = self.edges_of(&data.id);
                let pagerank_score = (data.pagerank_score != 0.0).then_some(data.pagerank_score);
                AsgNode {
                    id: data.id.clone(),
                    node_type: data.node_type,
                    body: data.body.clone(),
                    incoming_edges,
                    outgoing_edges,
                    pagerank_score,
                }
            })
            .collect();

        AsgGraphSnapshot { version: 1, nodes }
    }

    /// Serialize the full graph to compact YAML.
    pub fn to_yaml(&self) -> Result<String, AsgGraphError> {
        serde_yaml::to_string(&self.to_snapshot()).map_err(AsgGraphError::Yaml)
    }

    /// Deserialize a graph from YAML (bodies become `Owned`).
    pub fn from_yaml(yaml: &str) -> Result<Self, AsgGraphError> {
        let snapshot: AsgGraphSnapshot<'static> =
            serde_yaml::from_str(yaml).map_err(AsgGraphError::Yaml)?;
        let mut graph = Self::new();
        graph.ingest_snapshot(snapshot)?;
        Ok(graph)
    }

    /// Collect a node's incoming/outgoing edges as schema `EdgeRef`s, sorted
    /// by `(node id, kind)` for byte-deterministic serialization.
    fn edges_of(&self, id: &NodeId) -> (Vec<EdgeRef>, Vec<EdgeRef>) {
        let mut incoming = Vec::new();
        let mut outgoing = Vec::new();

        if let Some(ix) = self.id_to_index.get(id) {
            for edge in self.graph.edges_directed(*ix, Direction::Incoming) {
                if let Some(src) = self.index_to_id.get(&edge.source()) {
                    incoming.push(EdgeRef {
                        node: src.clone(),
                        kind: *edge.weight(),
                    });
                }
            }
            for edge in self.graph.edges_directed(*ix, Direction::Outgoing) {
                if let Some(dst) = self.index_to_id.get(&edge.target()) {
                    outgoing.push(EdgeRef {
                        node: dst.clone(),
                        kind: *edge.weight(),
                    });
                }
            }
        }

        incoming.sort_by(|a, b| (a.node.as_str(), a.kind).cmp(&(b.node.as_str(), b.kind)));
        outgoing.sort_by(|a, b| (a.node.as_str(), a.kind).cmp(&(b.node.as_str(), b.kind)));
        (incoming, outgoing)
    }

    /// Rebuild the petgraph from a snapshot. Edges are deduplicated because
    /// `to_snapshot` records each edge twice (once per endpoint).
    fn ingest_snapshot(
        &mut self,
        snapshot: AsgGraphSnapshot<'static>,
    ) -> Result<(), AsgGraphError> {
        let mut pending: Vec<(NodeId, NodeId, EdgeKind)> = Vec::new();

        for node in snapshot.nodes {
            let ix = self.graph.add_node(AsgNodeData {
                id: node.id.clone(),
                node_type: node.node_type,
                body: node.body,
                pagerank_score: node.pagerank_score.unwrap_or(0.0),
                file: None,
                range: (0, 0),
            });
            self.id_to_index.insert(node.id.clone(), ix);
            self.index_to_id.insert(ix, node.id.clone());

            for e in node.outgoing_edges {
                pending.push((node.id.clone(), e.node, e.kind));
            }
            for e in node.incoming_edges {
                pending.push((e.node, node.id.clone(), e.kind));
            }
        }

        let mut seen = std::collections::HashSet::new();
        for (from, to, kind) in pending {
            if from == to || !seen.insert((from.clone(), to.clone(), kind)) {
                continue;
            }
            let f = *self
                .id_to_index
                .get(&from)
                .ok_or_else(|| AsgGraphError::NodeNotFound(from.to_string()))?;
            let t = *self
                .id_to_index
                .get(&to)
                .ok_or_else(|| AsgGraphError::NodeNotFound(to.to_string()))?;
            self.graph.add_edge(f, t, kind);
        }
        Ok(())
    }
}

impl Default for AsgGraph<'_> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum AsgGraphError {
    #[error("yaml error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("duplicate node id: {0}")]
    DuplicateNode(String),
    #[error("node not found: {0}")]
    NodeNotFound(String),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_graph<'a>(source: &'a str) -> AsgGraph<'a> {
        let mut g = AsgGraph::new();
        let fn_id = NodeId::qualified("crate", &["server", "mod"], "fn", "run_server");
        let struct_id = NodeId::qualified("crate", &["server", "state"], "struct", "ServerState");
        let trait_id = NodeId::qualified("crate", &["server", "handler"], "trait", "Handler");

        g.add_node(fn_id.clone(), NodeType::Fn, Cow::Borrowed(source))
            .unwrap();
        g.add_node(
            struct_id.clone(),
            NodeType::Struct,
            Cow::Borrowed("pub struct ServerState {}"),
        )
        .unwrap();
        g.add_node(
            trait_id.clone(),
            NodeType::Trait,
            Cow::Borrowed("pub trait Handler {}"),
        )
        .unwrap();

        g.add_edge(&fn_id, &struct_id, EdgeKind::References)
            .unwrap();
        g.add_edge(&fn_id, &trait_id, EdgeKind::Implements).unwrap();
        g
    }

    #[test]
    fn schema_contract_yaml() {
        let src = "pub fn run_server() { let s = ServerState::new(); }";
        let g = demo_graph(src);
        let yaml = g.to_yaml().unwrap();

        assert!(yaml.contains("id: crate::server::mod::fn::run_server"));
        assert!(yaml.contains("type: fn"));
        assert!(yaml.contains("incoming_edges:"));
        assert!(yaml.contains("outgoing_edges:"));
        assert!(yaml.contains("kind: references"));
        assert!(yaml.contains(src));
    }

    #[test]
    fn round_trip_preserves_graph_and_scores() {
        let src = "pub fn run_server() {}";
        let mut g = demo_graph(src);
        let run_id = NodeId::qualified("crate", &["server", "mod"], "fn", "run_server");
        g.stamp_scores(vec![(run_id.clone(), 0.4181)]);

        let yaml = g.to_yaml().unwrap();
        let restored = AsgGraph::from_yaml(&yaml).unwrap();

        assert_eq!(restored.node_count(), g.node_count());
        assert_eq!(restored.edge_count(), g.edge_count());

        let node = restored.node(&run_id).unwrap();
        assert_eq!(node.node_type, NodeType::Fn);
        assert_eq!(node.body, src);
        assert!((node.pagerank_score - 0.4181).abs() < 1e-9);
        assert_eq!(restored.edge_count(), 2);
    }

    #[test]
    fn zero_copy_borrow() {
        let src = String::from("pub fn x() {}");
        let mut g = AsgGraph::new();
        let id = NodeId::qualified("crate", &[], "fn", "x");
        g.add_node(id.clone(), NodeType::Fn, Cow::Borrowed(src.as_str()))
            .unwrap();

        // Body remains a borrowed slice — zero allocation on the payload.
        assert!(matches!(g.node(&id).unwrap().body, Cow::Borrowed(_)));

        let yaml = g.to_yaml().unwrap();
        assert!(yaml.contains("pub fn x() {}"));
    }

    #[test]
    fn duplicate_id_rejected() {
        let mut g = AsgGraph::new();
        let id = NodeId::qualified("crate", &[], "fn", "x");
        g.add_node(id.clone(), NodeType::Fn, Cow::Borrowed("a"))
            .unwrap();
        assert!(matches!(
            g.add_node(id, NodeType::Fn, Cow::Borrowed("b")),
            Err(AsgGraphError::DuplicateNode(_))
        ));
    }

    #[test]
    fn deterministic_serialization() {
        let src = "pub fn run_server() {}";
        let g1 = demo_graph(src);
        let g2 = demo_graph(src);
        assert_eq!(g1.to_yaml().unwrap(), g2.to_yaml().unwrap());
    }

    // -----------------------------------------------------------------------
    // PageRank
    // -----------------------------------------------------------------------

    /// Star graph: three callers `a`, `b`, `c` all invoke one shared utility `u`.
    /// `u` has in-degree 3 and must rank highest under PageRank.
    fn star_graph<'a>() -> AsgGraph<'a> {
        let mut g = AsgGraph::new();
        let u = NodeId::qualified("crate", &["util"], "fn", "shared");
        let a = NodeId::qualified("crate", &["a"], "fn", "a");
        let b = NodeId::qualified("crate", &["b"], "fn", "b");
        let c = NodeId::qualified("crate", &["c"], "fn", "c");

        for id in [&a, &b, &c, &u] {
            g.add_node(id.clone(), NodeType::Fn, Cow::Borrowed(""))
                .unwrap();
        }
        g.add_edge(&a, &u, EdgeKind::Calls).unwrap();
        g.add_edge(&b, &u, EdgeKind::Calls).unwrap();
        g.add_edge(&c, &u, EdgeKind::Calls).unwrap();
        g
    }

    #[test]
    fn pagerank_hub_ranks_highest_and_normalizes() {
        let g = star_graph();
        let scores: HashMap<NodeId, f64> = g
            .page_rank(&PageRankConfig::default())
            .into_iter()
            .collect();

        let u = NodeId::qualified("crate", &["util"], "fn", "shared");
        let a = NodeId::qualified("crate", &["a"], "fn", "a");

        // The high in-degree utility dominates the callers.
        assert!(scores[&u] > scores[&a]);

        // Scores are a proper probability distribution.
        let total: f64 = scores.values().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn compute_page_rank_stamps_scores() {
        let mut g = star_graph();
        g.compute_page_rank(&PageRankConfig::default());

        let u = NodeId::qualified("crate", &["util"], "fn", "shared");
        assert!(g.node(&u).unwrap().pagerank_score > 0.0);

        // top_nodes exposes the structurally-vital hub first.
        let top = g.top_nodes(1);
        assert_eq!(top[0].id, u);

        // Scores surface in the YAML schema.
        let yaml = g.to_yaml().unwrap();
        assert!(yaml.contains("pagerank_score:"));
    }

    #[test]
    fn pagerank_converges_on_empty_and_singleton() {
        let g = AsgGraph::new();
        assert!(g.page_rank(&PageRankConfig::default()).is_empty());

        let mut g = AsgGraph::new();
        g.add_node(
            NodeId::qualified("crate", &[], "fn", "solo"),
            NodeType::Fn,
            Cow::Borrowed(""),
        )
        .unwrap();
        let scores = g.page_rank(&PageRankConfig::default());
        assert_eq!(scores.len(), 1);
        assert!((scores[0].1 - 1.0).abs() < 1e-9);
    }
}
