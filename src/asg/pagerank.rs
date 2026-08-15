//! Custom PageRank: edge-kind-weighted + personalized teleportation.
//!
//! Two customizations over vanilla PageRank make centrality match the spec's
//! "foundational structure" semantics:
//!
//! 1. **Kind-weighted edges.** A `calls` edge is a much stronger structural
//!    signal than a `contains` (lexical nesting) edge. Outgoing mass is split
//!    by weight, so utilities that are called/referenced a lot accumulate more
//!    centrality than nodes that merely nest other code.
//! 2. **Personalized teleport.** Instead of the uniform `(1 - d)/N` teleport,
//!    the random-surfer mass is distributed by a seed vector `v`. `Seed::Uniform`
//!    recovers vanilla PageRank; `Seed::Foundational` biases toward traits and
//!    widely-`implements`-ed schemas; `Seed::Custom` pins arbitrary nodes.
//!
//! The math (damped, personalized, weighted):
//!
//! ```text
//! pr(v) = (1 - d) * v[v]  +  d * Σ_{u→v} pr(u) * w(u,v) / out_w(u)
//! ```
//! where dangling nodes (no out-weight) redistribute their mass via `v`.

use std::collections::HashMap;
use std::collections::HashSet;

use petgraph::visit::EdgeRef as _;

use super::graph::{AsgGraph, EdgeKind, NodeId, NodeType};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Structural weight of each edge kind. Tune to emphasize call/reference
/// centrality over lexical containment.
#[derive(Debug, Clone)]
pub struct EdgeWeights {
    pub calls: f64,
    pub references: f64,
    pub contains: f64,
    pub imports: f64,
    pub implements: f64,
    pub field_of: f64,
    pub variant_of: f64,
}

impl Default for EdgeWeights {
    fn default() -> Self {
        Self {
            calls: 1.0,
            references: 0.7,
            implements: 0.9,
            imports: 0.5,
            contains: 0.15,
            field_of: 0.25,
            variant_of: 0.25,
        }
    }
}

/// Teleportation (personalization) strategy.
#[derive(Debug, Clone)]
pub enum Seed {
    /// Uniform `1/N` — vanilla PageRank teleport.
    Uniform,
    /// Traits and any node that is the target of an `implements` edge.
    Foundational,
    /// Explicit node ids.
    Custom(Vec<NodeId>),
}

/// Parameters for the weighted personalized PageRank iteration.
#[derive(Debug, Clone)]
pub struct PageRankConfig {
    pub damping: f64,
    pub epsilon: f64,
    pub max_iterations: usize,
    pub edge_weights: EdgeWeights,
    pub seed: Seed,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            epsilon: 1e-8,
            max_iterations: 200,
            edge_weights: EdgeWeights::default(),
            seed: Seed::Uniform,
        }
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Run weighted personalized PageRank over `graph`.
///
/// Returns scores aligned with `graph.nodes()` iteration order (dense petgraph
/// node indices, 0..N), which is what `AsgGraph::stamp_scores` expects.
pub fn pagerank<'a>(graph: &AsgGraph<'a>, config: &PageRankConfig) -> Vec<(NodeId, f64)> {
    let n = graph.node_count();
    if n == 0 {
        return Vec::new();
    }

    // Dense per-node state, indexed by petgraph NodeIndex slot (0..N).
    let indices: Vec<_> = graph.graph().node_indices().collect();
    let ids: Vec<NodeId> = indices
        .iter()
        .map(|ix| graph.graph()[*ix].id.clone())
        .collect();
    let pos_of: HashMap<NodeId, usize> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.clone(), i))
        .collect();

    // Weighted adjacency (compressed as CSR) + weighted out-degree.
    let mut out_w = vec![0.0f64; n];
    let mut edges_out: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for e in graph.graph().edge_references() {
        let src = e.source().index() as usize;
        let dst = e.target().index() as usize;
        if src == dst {
            continue; // self-loops carry no structural signal
        }
        let w = edge_weight(*e.weight(), &config.edge_weights);
        edges_out[src].push((dst, w));
        out_w[src] += w;
    }

    // Personalization vector.
    let v = personalization_vector(&pos_of, graph, &config.seed);

    // Power iteration.
    let d = config.damping;
    let mut pr = v.clone();
    let mut next = vec![0.0f64; n];

    for _ in 0..config.max_iterations {
        for x in next.iter_mut() {
            *x = 0.0;
        }

        let mut dangling_mass = 0.0;
        for u in 0..n {
            if out_w[u] > 0.0 {
                let base = d * pr[u];
                let inv = 1.0 / out_w[u];
                for &(to, w) in &edges_out[u] {
                    next[to] += base * w * inv;
                }
            } else {
                dangling_mass += d * pr[u];
            }
        }

        // Teleport + dangling redistribution through the seed vector.
        for i in 0..n {
            next[i] += (1.0 - d) * v[i] + dangling_mass * v[i];
        }

        let mut diff = 0.0;
        for i in 0..n {
            diff += (pr[i] - next[i]).abs();
        }
        pr.copy_from_slice(&next);
        if diff < config.epsilon {
            break;
        }
    }

    ids.into_iter().zip(pr).collect()
}

/// Same as [`pagerank`], but as a lookup map and stamped into the graph.
pub fn pagerank_map<'a>(
    graph: &AsgGraph<'a>,
    config: &PageRankConfig,
) -> HashMap<NodeId, f64> {
    pagerank(graph, config).into_iter().collect()
}

/// Run PageRank and write the scores back onto the graph's node payloads.
pub fn stamp_pagerank<'a>(graph: &mut AsgGraph<'a>, config: &PageRankConfig) {
    let scores = pagerank(graph, config);
    graph.stamp_scores(scores);
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn edge_weight(kind: EdgeKind, w: &EdgeWeights) -> f64 {
    match kind {
        EdgeKind::Calls => w.calls,
        EdgeKind::References => w.references,
        EdgeKind::Contains => w.contains,
        EdgeKind::Imports => w.imports,
        EdgeKind::Implements => w.implements,
        EdgeKind::FieldOf => w.field_of,
        EdgeKind::VariantOf => w.variant_of,
    }
}

fn personalization_vector<'a>(
    pos_of: &HashMap<NodeId, usize>,
    graph: &AsgGraph<'a>,
    seed: &Seed,
) -> Vec<f64> {
    let n = pos_of.len();
    let mut v = vec![0.0f64; n];

    match seed {
        Seed::Uniform => {
            for x in &mut v {
                *x = 1.0 / n as f64;
            }
        }
        Seed::Custom(seeds) => {
            apply_seeds(&mut v, pos_of, seeds.iter().cloned());
        }
        Seed::Foundational => {
            let mut seeds: Vec<NodeId> = Vec::new();
            for e in graph.graph().edge_references() {
                if *e.weight() == EdgeKind::Implements {
                    if let Some(id) = graph.id_of(e.target()) {
                        seeds.push(id.clone());
                    }
                }
            }
            for data in graph.nodes() {
                if data.node_type == NodeType::Trait {
                    seeds.push(data.id.clone());
                }
            }
            let mut seen = HashSet::new();
            seeds.retain(|s| seen.insert(s.clone()));
            apply_seeds(&mut v, pos_of, seeds.into_iter());
        }
    }
    v
}

fn apply_seeds(
    v: &mut [f64],
    pos_of: &HashMap<NodeId, usize>,
    seeds: impl Iterator<Item = NodeId>,
) {
    let mut total = 0usize;
    for s in seeds {
        if let Some(&p) = pos_of.get(&s) {
            v[p] = 1.0;
            total += 1;
        }
    }
    if total == 0 {
        // Fallback to uniform so we never divide by zero.
        let n = v.len();
        for x in v.iter_mut() {
            *x = 1.0 / n as f64;
        }
    } else {
        for x in v.iter_mut() {
            *x /= total as f64;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asg::graph::NodeType;
    use std::borrow::Cow;

    fn add_node<'a>(g: &mut AsgGraph<'a>, module: &[&str], kind: &str, name: &str, ty: NodeType) -> NodeId {
        let id = NodeId::qualified("crate", module, kind, name);
        g.add_node(id.clone(), ty, Cow::Borrowed("")).unwrap();
        id
    }

    #[test]
    fn high_indegree_hub_outranks_leaves() {
        let mut g = AsgGraph::new();
        let util = add_node(&mut g, &[], "fn", "util", NodeType::Fn);
        let a = add_node(&mut g, &[], "fn", "a", NodeType::Fn);
        let b = add_node(&mut g, &[], "fn", "b", NodeType::Fn);
        let c = add_node(&mut g, &[], "fn", "c", NodeType::Fn);

        for caller in [&a, &b, &c] {
            g.add_edge(caller, &util, EdgeKind::Calls).unwrap();
        }

        let scores = pagerank_map(&g, &PageRankConfig::default());
        assert!(scores[&util] > scores[&a]);
        assert!(scores[&util] > scores[&b]);
        assert!(scores[&util] > scores[&c]);
    }

    #[test]
    fn kind_weighting_biases_centrality() {
        // A single source `s` splits its outgoing mass across two edge kinds:
        // `calls` (w=1.0) vs `contains` (w=0.15). Because the split is w/out_w
        // where out_w = 1.0 + 0.15 = 1.15, the `calls` target receives a ~6.7x
        // larger share of `s`'s mass than the `contains` target — edge-kind
        // weighting must make `called` outrank `nested`.
        let mut g = AsgGraph::new();

        let called = add_node(&mut g, &[], "fn", "called", NodeType::Fn);
        let nested = add_node(&mut g, &[], "fn", "nested", NodeType::Fn);
        let s = add_node(&mut g, &[], "fn", "s", NodeType::Fn);
        // Give `s` some incoming relevance so it has mass to distribute.
        let a = add_node(&mut g, &[], "fn", "a", NodeType::Fn);

        g.add_edge(&s, &called, EdgeKind::Calls).unwrap();
        g.add_edge(&s, &nested, EdgeKind::Contains).unwrap();
        g.add_edge(&a, &s, EdgeKind::Calls).unwrap();

        let scores = pagerank_map(&g, &PageRankConfig::default());
        assert!(
            scores[&called] > scores[&nested],
            "called({}) should outrank nested({})",
            scores[&called],
            scores[&nested]
        );
    }

    #[test]
    fn personalization_boosts_seed() {
        let mut g = AsgGraph::new();
        // A low-degree trait that vanilla PageRank would rank near the bottom.
        let trait_id = add_node(&mut g, &[], "trait", "Handler", NodeType::Trait);
        let hot = add_node(&mut g, &[], "fn", "hot", NodeType::Fn);
        let cold = add_node(&mut g, &[], "fn", "cold", NodeType::Fn);
        g.add_edge(&trait_id, &hot, EdgeKind::Calls).unwrap();

        let uniform = pagerank_map(&g, &PageRankConfig::default());
        let seeded = pagerank_map(
            &g,
            &PageRankConfig {
                seed: Seed::Custom(vec![trait_id.clone()]),
                ..PageRankConfig::default()
            },
        );

        assert!(
            seeded[&trait_id] > uniform[&trait_id],
            "personalization should raise the seed's score"
        );
        // The seed's relative standing flips: it should now outrank the cold leaf.
        assert!(seeded[&trait_id] > seeded[&cold]);
    }

    #[test]
    fn empty_graph_is_safe() {
        let g: AsgGraph<'static> = AsgGraph::new();
        assert!(pagerank(&g, &PageRankConfig::default()).is_empty());
    }

    #[test]
    fn scores_sum_to_one() {
        let mut g = AsgGraph::new();
        let a = add_node(&mut g, &[], "fn", "a", NodeType::Fn);
        let b = add_node(&mut g, &[], "fn", "b", NodeType::Fn);
        g.add_edge(&a, &b, EdgeKind::Calls).unwrap();
        g.add_edge(&b, &a, EdgeKind::References).unwrap();

        let scores = pagerank_map(&g, &PageRankConfig::default());
        let total: f64 = scores.values().sum();
        assert!((total - 1.0).abs() < 1e-6, "total={total}");
    }
}
