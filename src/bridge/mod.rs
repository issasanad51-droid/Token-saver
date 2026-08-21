//! Cross-Language Semantic Bridge Parsing.
//!
//! Polyglot systems break single-language ASGs. This mapper parses bridges
//! between languages — SQL query strings inside Rust/JS files, API endpoint
//! references across frontend/backend boundaries — and links the nodes that
//! share a bridge into structural graph edges. When a developer queries a
//! backend issue, the cross-language dependency is computed and only the
//! highly relevant matching frontend API call is pulled into context.

use std::collections::HashMap;

use regex::Regex;

use crate::asg::{Asg, Edge, EdgeKind};

/// A parsed cross-language bridge.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Bridge {
    /// A SQL query string (e.g. `query!("SELECT * FROM users")`).
    Sql(String),
    /// An API endpoint reference (e.g. `/api/users`, `http://host/api/x`).
    Api(String),
}

/// Cross-Language Relationship Mapper.
pub struct BridgeMapper {
    sql_re: Regex,
    api_re: Regex,
}

impl Default for BridgeMapper {
    fn default() -> Self {
        Self::new()
    }
}

impl BridgeMapper {
    pub fn new() -> Self {
        // SQL inside macro invocations: query!("..."), sqlx::query!("..."),
        // plus raw strings (r#"..."#) that begin with a SQL keyword.
        let sql_re = Regex::new(
            r##"(?i)(?:query!|query_as!|sqlx::query!|sqlx::query_as!)\s*\(\s*"([^"]*)"|r#?"((?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|WITH)\s[^"]*)"#?"##,
        )
        .expect("valid sql bridge regex");
        // API endpoint references: "/api/...", "http(s)://host/api/...".
        let api_re = Regex::new(r#""((?:/api/|https?://[^"/\s]+/api/)[^"]*)""#)
            .expect("valid api bridge regex");
        Self { sql_re, api_re }
    }

    /// Extract bridges from a source string.
    pub fn extract(&self, source: &str) -> Vec<Bridge> {
        let mut bridges = Vec::new();
        for cap in self.sql_re.captures_iter(source) {
            let sql = cap
                .get(1)
                .or_else(|| cap.get(2))
                .map(|m| m.as_str().trim().to_string())
                .filter(|s| !s.is_empty());
            if let Some(sql) = sql {
                if looks_like_sql(&sql) {
                    bridges.push(Bridge::Sql(sql));
                }
            }
        }
        for cap in self.api_re.captures_iter(source) {
            if let Some(m) = cap.get(1) {
                bridges.push(Bridge::Api(m.as_str().to_string()));
            }
        }
        bridges
    }

    /// Scan every node in the ASG, group nodes by shared bridge, and add
    /// `Bridge` edges between nodes that reference the same bridge target.
    /// Returns the number of edges added.
    pub fn link_bridges(&self, asg: &mut Asg) -> usize {
        let mut by_bridge: HashMap<Bridge, Vec<usize>> = HashMap::new();
        for node in &asg.nodes {
            for bridge in self.extract(&node.source) {
                by_bridge.entry(bridge).or_default().push(node.id);
            }
        }

        let mut added = 0usize;
        let mut seen = std::collections::HashSet::new();
        for (_, node_ids) in by_bridge {
            if node_ids.len() < 2 {
                continue;
            }
            // Link every pair sharing the bridge (directed both ways).
            for i in 0..node_ids.len() {
                for j in 0..node_ids.len() {
                    if i == j {
                        continue;
                    }
                    let (from, to) = (node_ids[i], node_ids[j]);
                    if !seen.insert((from, to)) {
                        continue;
                    }
                    let edge_index = asg.edges.len();
                    asg.edges.push(Edge {
                        from,
                        to,
                        kind: EdgeKind::Bridge,
                    });
                    asg.adjacency.entry(from).or_default().push(edge_index);
                    asg.reverse_adjacency.entry(to).or_default().push(edge_index);
                    added += 1;
                }
            }
        }
        added
    }
}

fn looks_like_sql(sql: &str) -> bool {
    let upper = sql.to_ascii_uppercase();
    [
        "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "WITH", "FROM",
        "WHERE",
    ]
    .iter()
    .any(|kw| upper.contains(kw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_sql_bridge_from_query_macro() {
        let mapper = BridgeMapper::new();
        let src = r#"let users = query!("SELECT * FROM users WHERE id = ?");"#;
        let bridges = mapper.extract(src);
        assert!(bridges.iter().any(|b| matches!(b, Bridge::Sql(s) if s.contains("SELECT"))));
    }

    #[test]
    fn extracts_api_bridge() {
        let mapper = BridgeMapper::new();
        let src = r#"fetch("/api/users/42")"#;
        let bridges = mapper.extract(src);
        assert!(bridges.iter().any(|b| matches!(b, Bridge::Api(s) if s == "/api/users/42")));
    }

    #[test]
    fn links_nodes_sharing_bridge() {
        let mapper = BridgeMapper::new();
        let mut asg = Asg::default();
        asg.nodes.push(crate::asg::Node {
            id: 0,
            tracker_id: "crate::backend::fn::handler".into(),
            name: "handler".into(),
            kind: "fn".into(),
            source: r#"query!("SELECT * FROM users")"#.into(),
            file_path: "backend.rs".into(),
            range: (0, 0),
            pagerank: 0.0,
        });
        asg.nodes.push(crate::asg::Node {
            id: 1,
            tracker_id: "crate::frontend::fn::fetch_users".into(),
            name: "fetch_users".into(),
            kind: "fn".into(),
            source: r#"fetch("/api/users")"#.into(),
            file_path: "frontend.ts".into(),
            range: (0, 0),
            pagerank: 0.0,
        });
        let added = mapper.link_bridges(&mut asg);
        assert!(added > 0);
    }
}