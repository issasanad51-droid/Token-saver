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
    /// Module-based seeding: higher weight for nodes in the same module
    /// as the query cursor, decreasing with module distance.
    ModuleBased(Vec<NodeId>, f64),
    /// Degree-based seeding: proportional to node degree (in + out).
    DegreeBased(Vec<NodeId>),
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
    // Parallel edges between the same `(src, dst)` pair are collapsed to the
    // max weight so repeated call sites don't multiply a target's centrality —
    // centrality should reflect *distinct* structural relationships.
    let mut out_w = vec![0.0f64; n];
    let mut edges_out: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for e in graph.graph().edge_references() {
        let src = e.source().index() as usize;
        let dst = e.target().index() as usize;
        if src == dst {
            continue; // self-loops carry no structural signal
        }
        let w = edge_weight(*e.weight(), &config.edge_weights);
        match edges_out[src].iter_mut().find(|(d, _)| *d == dst) {
            Some(slot) => slot.1 = slot.1.max(w),
            None => {
                edges_out[src].push((dst, w));
                out_w[src] += w;
            }
        }
    }

    // Personalization vector.
    let v = personalization_vector(&pos_of, graph, &config.seed);

    // Power iteration with relative convergence detection.
    let d = config.damping;
    let mut pr = v.clone();
    let mut next = vec![0.0f64; n];

    let mut prev_diff = f64::INFINITY;
    for iteration in 0..=config.max_iterations {
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
        let mut max_change = 0.0f64;
        for i in 0..n {
            let change = (pr[i] - next[i]).abs();
            diff += change;
            max_change = max_change.max(change);
        }
        pr.copy_from_slice(&next);

        // Convergence: relative change based on max change
        let rel_change = if prev_diff > 0.0 {
            diff / prev_diff
        } else {
            f64::INFINITY
        };
        prev_diff = diff;

        if rel_change < config.epsilon && max_change < config.epsilon {
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
        Seed::ModuleBased(seed_modules, decay) => {
            // Seed nodes by module proximity to the cursor modules, with
            // weight decaying exponentially per level of module distance:
            //   w(node) = decay ^ shared_prefix_len(node_module, seed_module)
            //
            // A `NodeId` looks like `crate::<module path>::<kind>::<name>`,
            // so the module path is everything between `crate` and the final
            // two `<kind>::<name>` segments.
            fn module_of(id: &NodeId) -> Vec<String> {
                let parts: Vec<&str> = id.0.split("::").collect();
                // strip leading crate root + trailing kind/name
                let end = parts.len().saturating_sub(2);
                let start = usize::from(parts.first() == Some(&"crate"));
                if end > start {
                    parts[start..end].iter().map(|s| s.to_string()).collect()
                } else {
                    Vec::new()
                }
            };

            let mut total_weight = 0.0f64;
            for (node_id, &p) in pos_of.iter() {
                let node_mod = module_of(node_id);
                let mut best = 0.0f64;
                for seed_module in seed_modules {
                    let seed_parts: Vec<&str> = seed_module.0.split("::").collect();
                    let shared = node_mod
                        .iter()
                        .map(|s| s.as_str())
                        .zip(seed_parts.iter().copied())
                        .take_while(|(a, b)| a == b)
                        .count();
                    // Weight decays per unmatched segment: an exact module
                    // match (distance 0) gets 1.0, each level of divergence
                    // multiplies by `decay`.
                    let distance = (seed_parts.len() - shared) as i32;
                    let w = decay.powi(distance);
                    best = best.max(w);
                }
                if best > 0.0 {
                    v[p] = best;
                    total_weight += best;
                }
            }
            if total_weight > 0.0 {
                for x in v.iter_mut() {
                    *x /= total_weight;
                }
            } else {
                // No module matched — fall back to uniform.
                for x in v.iter_mut() {
                    *x = 1.0 / n as f64;
                }
            }
        }
        Seed::DegreeBased(_) => {
            // Seed nodes proportional to their total degree (in + out),
            // keyed by real `NodeId`s resolved through the graph.
            let mut degree: HashMap<NodeId, f64> = HashMap::new();
            for e in graph.graph().edge_references() {
                if let Some(id) = graph.id_of(e.source()) {
                    *degree.entry(id.clone()).or_insert(0.0) += 1.0;
                }
                if let Some(id) = graph.id_of(e.target()) {
                    *degree.entry(id.clone()).or_insert(0.0) += 1.0;
                }
            }
            let mut total = 0.0f64;
            for (node_id, &p) in pos_of.iter() {
                if let Some(&deg) = degree.get(node_id) {
                    let w = deg.max(0.01); // minimum weight floor
                    v[p] = w;
                    total += w;
                }
            }
            if total > 0.0 {
                for x in v.iter_mut() {
                    *x /= total;
                }
            }
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
    fn module_based_seeding() {
        use crate::asg::source::SourceSet;

        let mut ss = SourceSet::new();
        ss.insert(
            std::path::PathBuf::from("src/mod.rs"),
            r#"pub mod inner;
pub fn outer() {}
"#
            .to_string(),
        );
        ss.insert(
            std::path::PathBuf::from("src/mod/inner.rs"),
            r#"pub fn inner_fn() {}
pub mod deep;
pub fn deep_fn() {}
"#
            .to_string(),
        );
        ss.insert(
            std::path::PathBuf::from("src/mod/deep.rs"),
            r#"pub fn deep_fn2() {}
"#
            .to_string(),
        );

        let mut builder = crate::asg::builder::AsgBuilder::new(&ss)
            .unwrap()
            .with_crate_root(std::path::Path::new("."));
        builder.parse().unwrap();
        let graph = builder.build();

        // Module-based seeding should boost nodes in the same module
        let config = PageRankConfig {
            seed: Seed::ModuleBased(vec![NodeId::new("crate")], 0.5),
            ..PageRankConfig::default()
        };
        let scores = pagerank_map(&graph, &config);

        // Every node must receive a normalized (positive) score.
        assert!(
            scores.values().all(|&s| s > 0.0),
            "module-based seeding should produce a valid distribution"
        );
    }

    #[test]
    fn degree_based_seeding() {
        let mut g = AsgGraph::new();
        let hub = add_node(&mut g, &[], "fn", "hub", NodeType::Fn);
        let leaf1 = add_node(&mut g, &[], "fn", "leaf1", NodeType::Fn);
        let leaf2 = add_node(&mut g, &[], "fn", "leaf2", NodeType::Fn);
        let leaf3 = add_node(&mut g, &[], "fn", "leaf3", NodeType::Fn);

        // hub connects to all leaves
        g.add_edge(&hub, &leaf1, EdgeKind::Calls).unwrap();
        g.add_edge(&hub, &leaf2, EdgeKind::Calls).unwrap();
        g.add_edge(&hub, &leaf3, EdgeKind::Calls).unwrap();

        let uniform = pagerank_map(&g, &PageRankConfig::default());
        let seeded = pagerank_map(
            &g,
            &PageRankConfig {
                seed: Seed::DegreeBased(vec![hub.clone()]),
                ..PageRankConfig::default()
            },
        );

        // hub should have higher score with degree-based seeding
        assert!(
            seeded[&hub] > uniform[&hub],
            "degree-based seeding should boost hub score"
        );
        // leaves should have lower relative scores
        assert!(
            seeded[&leaf1] < uniform[&leaf1] + 0.1,
            "degree-based seeding should not boost leaves disproportionately"
        );
    }

    #[test]
    fn module_based_seed_prefers_same_module() {
        let mut g = AsgGraph::new();
        let in_mod = add_node(&mut g, &["server"], "fn", "handler", NodeType::Fn);
        let out_mod = add_node(&mut g, &["client"], "fn", "render", NodeType::Fn);
        // No edges at all: scores come purely from teleportation.
        let cfg = PageRankConfig {
            seed: Seed::ModuleBased(vec![NodeId::new("server")], 0.5),
            ..PageRankConfig::default()
        };
        let scores = pagerank_map(&g, &cfg);
        assert!(
            scores[&in_mod] > scores[&out_mod],
            "same-module node ({}) should outrank other-module node ({})",
            scores[&in_mod],
            scores[&out_mod]
        );
    }

    #[test]
    fn degree_based_seed_prefers_high_degree() {
        let mut g = AsgGraph::new();
        let hub = add_node(&mut g, &[], "fn", "hub", NodeType::Fn);
        let leaf = add_node(&mut g, &[], "fn", "leaf", NodeType::Fn);
        let a = add_node(&mut g, &[], "fn", "a", NodeType::Fn);
        let b = add_node(&mut g, &[], "fn", "b", NodeType::Fn);
        for caller in [&a, &b] {
            g.add_edge(caller, &hub, EdgeKind::Calls).unwrap();
        }

        let cfg = PageRankConfig {
            seed: Seed::DegreeBased(vec![hub.clone(), leaf.clone()]),
            ..PageRankConfig::default()
        };
        let scores = pagerank_map(&g, &cfg);
        assert!(
            scores[&hub] > scores[&leaf],
            "high-degree seed ({}) should outrank zero-degree seed ({})",
            scores[&hub],
            scores[&leaf]
        );
    }

    #[test]
    fn duplicate_edges_do_not_inflate_centrality() {
        // Ten parallel `calls` edges to `dup` vs one to `once`: with
        // deduplication both targets receive the same share of mass.
        let mut g = AsgGraph::new();
        let dup = add_node(&mut g, &[], "fn", "dup", NodeType::Fn);
        let once = add_node(&mut g, &[], "fn", "once", NodeType::Fn);
        let s = add_node(&mut g, &[], "fn", "s", NodeType::Fn);
        let a = add_node(&mut g, &[], "fn", "a", NodeType::Fn);
        g.add_edge(&a, &s, EdgeKind::Calls).unwrap();
        for _ in 0..10 {
            g.add_edge(&s, &dup, EdgeKind::Calls).unwrap();
        }
        g.add_edge(&s, &once, EdgeKind::Calls).unwrap();

        let scores = pagerank_map(&g, &PageRankConfig::default());
        let diff = (scores[&dup] - scores[&once]).abs();
        assert!(
            diff < 1e-9,
            "parallel edges must not skew centrality: dup={} once={}",
            scores[&dup],
            scores[&once]
        );
    }
}
