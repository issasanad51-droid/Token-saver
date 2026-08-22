//! Custom Leiden community detection over the Abstract Semantic Graph.
//!
//! Hand-rolled from scratch (no external graph-partition crates). The
//! algorithm is the classic Leiden recipe adapted to a deterministic,
//! single-threaded setting:
//!
//! 1. **Local moving** — greedy modularity optimization: each node migrates
//!    to the neighbouring community yielding the largest modularity gain
//!    `ΔQ = w(i→C) − γ · tot(C) · k(i) / 2m`. Nodes are visited in dense-id
//!    order and ties are broken by the lowest community id, so identical
//!    graphs always partition identically.
//! 2. **Refinement** — the step that distinguishes Leiden from Louvain:
//!    nodes with *no internal connections* to their assigned community are
//!    split off as singletons. This guarantees communities stay internally
//!    connected and prevents the "badly connected community" artefact that
//!    plain Louvain produces on sparse code graphs.
//! 3. **Aggregation** — communities collapse into super-nodes (self-loops
//!    carry internal weight) and steps 1–2 repeat on the coarser graph until
//!    the modularity gain saturates.
//!
//! The resulting partition powers retrieval Stream D: chunks are ranked by
//! their network cohesion with the community of the editor's cursor node.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{Asg, EdgeKind};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Upper bound for scores assigned to nodes in communities *adjacent* to
/// the seed community. Keeps the retrieval ranking strict: membership in
/// the seed community (`1.0`) always outranks an adjacent community, which
/// always outranks everything else (`0.0`) — even when the bridge into the
/// seed community is the strongest inter-community edge in the graph.
/// Without this cap, `w / max_cohesion` hits `1.0` whenever the seed's
/// bridge dominates, making "different community, weakly connected"
/// indistinguishable from "same community".
const ADJACENT_COMMUNITY_MAX: f64 = 0.5;

/// Tunables for the deterministic Leiden partitioner.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LeidenConfig {
    /// Modularity resolution. `1.0` is canonical Leiden; larger values yield
    /// more, smaller communities.
    pub resolution: f64,
    /// Hard cap on aggregate/refine levels.
    pub max_levels: usize,
    /// Stop early when a level improves modularity by less than this.
    pub min_improvement: f64,
}

impl Default for LeidenConfig {
    fn default() -> Self {
        Self {
            resolution: 1.0,
            max_levels: 8,
            min_improvement: 1e-4,
        }
    }
}

/// Structural weight of each ASG edge kind in the undirected community graph.
/// Mirrors the PPR edge weights: `calls` carries the strongest cohesion.
pub fn edge_weight(kind: EdgeKind) -> f64 {
    match kind {
        EdgeKind::Calls => 1.0,
        EdgeKind::References => 0.7,
        EdgeKind::Implements => 0.9,
        EdgeKind::Imports => 0.5,
        EdgeKind::Contains => 0.15,
        EdgeKind::FieldOf => 0.25,
        EdgeKind::VariantOf => 0.25,
        EdgeKind::Bridge => 0.3,
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// A completed community partition over dense ASG node ids.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityStructure {
    /// Dense node id -> community id (contiguous `0..community_count`).
    pub assignment: Vec<usize>,
    /// Number of distinct communities.
    pub community_count: usize,
}

impl CommunityStructure {
    /// Community id of a node, if it exists.
    pub fn community_of(&self, node_id: usize) -> Option<usize> {
        self.assignment.get(node_id).copied()
    }

    /// Members of a community, ascending by node id.
    pub fn members(&self, community: usize) -> Vec<usize> {
        self.assignment
            .iter()
            .enumerate()
            .filter(|(_, &c)| c == community)
            .map(|(id, _)| id)
            .collect()
    }

    /// Deterministic proximity vector over all nodes for retrieval Stream D.
    ///
    /// Nodes inside `seed_community` score `1.0`. Nodes in directly adjacent
    /// communities score proportionally to the normalized inter-community
    /// edge weight (network cohesion), capped at `ADJACENT_COMMUNITY_MAX`
    /// so they can never tie with or outrank seed-community members.
    /// Everything else scores `0.0`.
    pub fn proximity_scores(&self, asg: &Asg, seed_community: usize) -> Vec<f64> {
        let mut scores = vec![0.0; self.assignment.len()];
        if self.community_count == 0 {
            return scores;
        }

        // Inter-community cohesion matrix: (a, b) -> accumulated edge weight.
        let mut cohesion: HashMap<(usize, usize), f64> = HashMap::new();
        for edge in &asg.edges {
            let (Some(ca), Some(cb)) = (
                self.community_of(edge.from),
                self.community_of(edge.to),
            ) else {
                continue;
            };
            if ca == cb {
                continue;
            }
            let w = edge_weight(edge.kind);
            *cohesion.entry((ca.min(cb), ca.max(cb))).or_default() += w;
        }

        let max_cohesion = cohesion
            .values()
            .copied()
            .fold(0.0f64, |acc, w| acc.max(w));

        for (node, &community) in self.assignment.iter().enumerate() {
            if community == seed_community {
                scores[node] = 1.0;
            } else if max_cohesion > 0.0 {
                let key = (
                    community.min(seed_community),
                    community.max(seed_community),
                );
                if let Some(&w) = cohesion.get(&key) {
                    scores[node] = ADJACENT_COMMUNITY_MAX * (w / max_cohesion);
                }
            }
        }
        scores
    }
}

// ---------------------------------------------------------------------------
// Undirected weighted graph (level representation)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct WeightedGraph {
    n: usize,
    /// Symmetric adjacency: node -> [(neighbor, weight)] sorted by neighbor.
    adj: Vec<Vec<(usize, f64)>>,
    /// Self-loop weight per node (internal mass preserved by aggregation).
    self_loop: Vec<f64>,
    /// Weighted degree per node (includes self-loops once).
    degree: Vec<f64>,
}

impl WeightedGraph {
    fn from_asg(asg: &Asg) -> Self {
        let n = asg.nodes.len();
        let mut adj: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n];
        let mut self_loop = vec![0.0; n];

        for edge in &asg.edges {
            if edge.from >= n || edge.to >= n {
                continue;
            }
            let w = edge_weight(edge.kind);
            if w <= 0.0 {
                continue;
            }
            if edge.from == edge.to {
                self_loop[edge.from] += w;
            } else {
                *adj[edge.from].entry(edge.to).or_default() += w;
                *adj[edge.to].entry(edge.from).or_default() += w;
            }
        }

        let adj: Vec<Vec<(usize, f64)>> = adj
            .into_iter()
            .map(|neighbors| {
                let mut list: Vec<(usize, f64)> = neighbors.into_iter().collect();
                list.sort_by_key(|(id, _)| *id);
                list
            })
            .collect();

        let degree: Vec<f64> = adj
            .iter()
            .zip(&self_loop)
            .map(|(neighbors, sl)| neighbors.iter().map(|(_, w)| w).sum::<f64>() + sl)
            .collect();

        Self {
            n,
            adj,
            self_loop,
            degree,
        }
    }

    fn total_edge_weight(&self) -> f64 {
        // Each undirected edge appears twice in `adj`; self-loops once.
        self.adj
            .iter()
            .map(|neighbors| neighbors.iter().map(|(_, w)| w).sum::<f64>())
            .sum::<f64>()
            / 2.0
            + self.self_loop.iter().sum::<f64>()
    }
}

// ---------------------------------------------------------------------------
// Partitioner
// ---------------------------------------------------------------------------

/// Run the full deterministic Leiden pipeline over `asg`.
pub fn detect_communities(asg: &Asg, config: &LeidenConfig) -> CommunityStructure {
    let graph = WeightedGraph::from_asg(asg);
    let n = graph.n;
    if n == 0 {
        return CommunityStructure {
            assignment: Vec::new(),
            community_count: 0,
        };
    }

    // orig node -> current-level node. Starts as identity.
    let mut node_map: Vec<usize> = (0..n).collect();
    let mut current = graph;

    let resolution = if config.resolution.is_finite() && config.resolution > 0.0 {
        config.resolution
    } else {
        1.0
    };
    let max_levels = config.max_levels.max(1);

    for _ in 0..max_levels {
        let singleton: Vec<usize> = (0..current.n).collect();
        let before = modularity(&current, &singleton, resolution);

        let moved = local_moving(&current, resolution);
        let refined = refine(&current, &moved);
        let after = modularity(&current, &refined, resolution);

        // Flatten the level's partition onto original node ids.
        for orig in node_map.iter_mut() {
            *orig = refined[*orig];
        }

        if after - before < config.min_improvement.abs() || refined.len() == current.n {
            break;
        }

        current = aggregate(&current, &refined);
    }

    // Relabel final coarse ids into contiguous community ids (first-seen order
    // over ascending original node id keeps the labeling deterministic).
    let mut relabel: HashMap<usize, usize> = HashMap::new();
    let assignment: Vec<usize> = node_map
        .iter()
        .map(|&coarse| {
            let next = relabel.len();
            *relabel.entry(coarse).or_insert(next)
        })
        .collect();

    CommunityStructure {
        community_count: relabel.len(),
        assignment,
    }
}

/// Greedy local moving (Louvain phase). Visits nodes in id order until a full
/// sweep makes no migration.
fn local_moving(graph: &WeightedGraph, resolution: f64) -> Vec<usize> {
    let n = graph.n;
    let mut community: Vec<usize> = (0..n).collect();
    let m2 = (2.0 * graph.total_edge_weight()).max(f64::MIN_POSITIVE);
    // Total degree per community (initially each node is its own community).
    let mut tot: Vec<f64> = graph.degree.clone();

    let mut improved = true;
    while improved {
        improved = false;
        for node in 0..n {
            let old = community[node];
            let k_i = graph.degree[node];

            // Tentatively remove the node from its community.
            tot[old] -= k_i;

            // Weight of links from `node` into each neighboring community.
            let mut links: HashMap<usize, f64> = HashMap::new();
            for &(neighbor, w) in &graph.adj[node] {
                if neighbor != node {
                    *links.entry(community[neighbor]).or_default() += w;
                }
            }
            if graph.self_loop[node] > 0.0 {
                *links.entry(old).or_default() += graph.self_loop[node];
            }

            // Modularity gain of joining community C:
            //   ΔQ = w(i→C) − γ · tot(C) · k(i) / 2m
            let mut best_community = old;
            let mut best_gain = f64::NEG_INFINITY;
            // Sorted keys keep tie-breaking deterministic.
            let mut keys: Vec<usize> = links.keys().copied().collect();
            keys.sort_unstable();
            for c in keys {
                let w = links[&c];
                let gain = w - resolution * tot[c] * k_i / m2;
                if gain > best_gain + 1e-12 {
                    best_gain = gain;
                    best_community = c;
                }
            }

            tot[best_community] += k_i;
            if best_community != old {
                community[node] = best_community;
                improved = true;
            }
        }
    }
    community
}

/// Leiden refinement: any node with zero internal degree (no edges to other
/// members of its community) is split off as a singleton so every surviving
/// community is internally connected.
fn refine(graph: &WeightedGraph, community: &[usize]) -> Vec<usize> {
    let n = graph.n;
    let mut refined = community.to_vec();
    let mut next_label = community.iter().copied().max().unwrap_or(0) + 1;

    for node in 0..n {
        let c = refined[node];
        let internal_degree: f64 = graph.adj[node]
            .iter()
            .filter(|&&(neighbor, _)| neighbor != node && refined[neighbor] == c)
            .map(|(_, w)| w)
            .sum();
        if internal_degree <= 0.0 {
            refined[node] = next_label;
            next_label += 1;
        }
    }
    refined
}

/// Collapse each community into a single super-node, preserving internal mass
/// as self-loops and summing inter-community edge weights.
fn aggregate(graph: &WeightedGraph, community: &[usize]) -> WeightedGraph {
    let mut relabel: HashMap<usize, usize> = HashMap::new();
    let label_of = |c: usize, relabel: &mut HashMap<usize, usize>| {
        let next = relabel.len();
        *relabel.entry(c).or_insert(next)
    };

    let mut coarse_adj: Vec<HashMap<usize, f64>> = Vec::new();
    let mut coarse_self: Vec<f64> = Vec::new();

    for node in 0..graph.n {
        let lc = label_of(community[node], &mut relabel);
        if lc >= coarse_adj.len() {
            coarse_adj.push(HashMap::new());
            coarse_self.push(0.0);
        }
        for &(neighbor, w) in &graph.adj[node] {
            let ln = label_of(community[neighbor], &mut relabel);
            if ln >= coarse_adj.len() {
                coarse_adj.push(HashMap::new());
                coarse_self.push(0.0);
            }
            if lc == ln {
                // Internal edge counted once per direction pair; halve below.
                coarse_self[lc] += w / 2.0;
            } else {
                *coarse_adj[lc].entry(ln).or_default() += w;
            }
        }
        coarse_self[lc] += graph.self_loop[node];
    }

    let n = coarse_adj.len();
    let adj: Vec<Vec<(usize, f64)>> = coarse_adj
        .into_iter()
        .map(|neighbors| {
            let mut list: Vec<(usize, f64)> = neighbors.into_iter().collect();
            list.sort_by_key(|(id, _)| *id);
            list
        })
        .collect();
    let degree: Vec<f64> = adj
        .iter()
        .zip(&coarse_self)
        .map(|(neighbors, sl)| neighbors.iter().map(|(_, w)| w).sum::<f64>() + sl)
        .collect();

    WeightedGraph {
        n,
        adj,
        self_loop: coarse_self,
        degree,
    }
}

/// Newman modularity with resolution `γ`:
/// `Q = Σ_c [ in(c)/2m − γ · (tot(c)/2m)² ]`.
fn modularity(graph: &WeightedGraph, community: &[usize], resolution: f64) -> f64 {
    let m2 = (2.0 * graph.total_edge_weight()).max(f64::MIN_POSITIVE);

    let mut internal: Vec<f64> = Vec::new();
    let mut tot: Vec<f64> = Vec::new();
    let mut label: HashMap<usize, usize> = HashMap::new();
    let mut slot = |c: usize, internal: &mut Vec<f64>, tot: &mut Vec<f64>| {
        let next = internal.len();
        *label.entry(c).or_insert_with(|| {
            internal.push(0.0);
            tot.push(0.0);
            next
        })
    };

    for node in 0..graph.n {
        let sc = slot(community[node], &mut internal, &mut tot);
        tot[sc] += graph.degree[node];
        for &(neighbor, w) in &graph.adj[node] {
            if neighbor != node && community[neighbor] == community[node] {
                internal[sc] += w / 2.0; // each internal edge seen twice
            }
        }
        internal[sc] += graph.self_loop[node];
    }

    internal
        .iter()
        .zip(&tot)
        .map(|(&in_c, &tot_c)| in_c / m2 - resolution * (tot_c / m2).powi(2))
        .sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asg::{Edge, Node};
    use std::path::PathBuf;

    fn node(id: usize) -> Node {
        Node {
            id,
            tracker_id: format!("crate::n{id}"),
            name: format!("n{id}"),
            kind: "fn".to_string(),
            source: String::new(),
            file_path: PathBuf::new(),
            range: (0, 0),
            pagerank: 0.0,
        }
    }

    fn asg_with(edges: &[(usize, usize)]) -> Asg {
        let max = edges.iter().flat_map(|&(a, b)| [a, b]).max().unwrap_or(0);
        let mut asg = Asg {
            nodes: (0..=max).map(node).collect(),
            ..Asg::default()
        };
        for &(a, b) in edges {
            asg.edges.push(Edge {
                from: a,
                to: b,
                kind: EdgeKind::Calls,
            });
        }
        asg
    }

    #[test]
    fn two_cliques_split_into_two_communities() {
        // Triangle 0-1-2 and triangle 3-4-5, one weak bridge 2-3.
        let edges = [
            (0, 1),
            (1, 2),
            (0, 2),
            (3, 4),
            (4, 5),
            (3, 5),
            (2, 3),
        ];
        let asg = asg_with(&edges);
        let structure = detect_communities(&asg, &LeidenConfig::default());

        assert_eq!(structure.assignment.len(), 6);
        let c0 = structure.community_of(0).unwrap();
        let c1 = structure.community_of(1).unwrap();
        let c3 = structure.community_of(3).unwrap();
        let c4 = structure.community_of(4).unwrap();
        assert_eq!(c0, c1, "0 and 1 share a clique");
        assert_eq!(c3, c4, "3 and 4 share a clique");
        assert_ne!(c0, c3, "the two cliques should separate");

        // Proximity: members of the seed community outrank the other cluster.
        let scores = structure.proximity_scores(&asg, c0);
        assert_eq!(scores[0], 1.0);
        assert!(scores[3] < 1.0 && scores[3] > 0.0, "bridge community gets partial credit");
    }

    #[test]
    fn deterministic_partition() {
        let edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3)];
        let asg = asg_with(&edges);
        let a = detect_communities(&asg, &LeidenConfig::default());
        let b = detect_communities(&asg, &LeidenConfig::default());
        assert_eq!(a.assignment, b.assignment);
    }

    #[test]
    fn adjacent_scores_strictly_below_seed_and_zero_elsewhere() {
        // Regression: with only two bridged communities the seed's bridge is
        // the strongest (only) inter-community edge, so the old
        // `w / max_cohesion` normalization scored the adjacent community at
        // exactly 1.0 — a full tie with seed membership. Three triangles:
        // 0-1-2 bridged to 3-4-5; 6-7-8 completely disconnected.
        let edges = [
            (0, 1),
            (1, 2),
            (0, 2),
            (3, 4),
            (4, 5),
            (3, 5),
            (2, 3),
            (6, 7),
            (7, 8),
            (6, 8),
        ];
        let asg = asg_with(&edges);
        let structure = detect_communities(&asg, &LeidenConfig::default());

        let seed = structure.community_of(0).unwrap();
        let adjacent = structure.community_of(3).unwrap();
        let far = structure.community_of(6).unwrap();
        assert_ne!(seed, adjacent);
        assert_ne!(seed, far);
        assert_ne!(adjacent, far);

        let scores = structure.proximity_scores(&asg, seed);
        for node in 0..3 {
            assert_eq!(scores[node], 1.0, "seed member {node}");
        }
        for node in 3..6 {
            let s = scores[node];
            assert!(
                s > 0.0 && s < 1.0,
                "adjacent member {node} must get partial credit strictly below 1.0, got {s}"
            );
            assert!(
                s <= ADJACENT_COMMUNITY_MAX,
                "adjacent member {node} must respect the cap, got {s}"
            );
        }
        for node in 6..9 {
            assert_eq!(scores[node], 0.0, "non-adjacent member {node}");
        }
    }

    #[test]
    fn empty_graph_is_safe() {
        let asg = Asg::default();
        let s = detect_communities(&asg, &LeidenConfig::default());
        assert_eq!(s.assignment.len(), 0);
        assert_eq!(s.community_count, 0);
        assert!(s.proximity_scores(&asg, 0).is_empty());
    }
}
