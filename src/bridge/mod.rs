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
        // resource name (SQL table / API path segment) -> nodes mentioning it
        let mut by_resource: HashMap<String, Vec<usize>> = HashMap::new();
        for node in &asg.nodes {
            for bridge in self.extract(&node.source) {
                match &bridge {
                    Bridge::Sql(sql) => {
                        for table in sql_tables(sql) {
                            by_resource
                                .entry(table)
                                .or_default()
                                .push(node.id);
                        }
                    }
                    Bridge::Api(path) => {
                        for resource in api_resources(path) {
                            by_resource
                                .entry(resource)
                                .or_default()
                                .push(node.id);
                        }
                    }
                }
                by_bridge.entry(bridge).or_default().push(node.id);
            }
        }

        let mut seen = std::collections::HashSet::new();
        let connect = |from: usize,
                       to: usize,
                       asg: &mut Asg,
                       seen: &mut std::collections::HashSet<(usize, usize)>| {
            if from == to || !seen.insert((from, to)) {
                return;
            }
            let edge_index = asg.edges.len();
            asg.edges.push(Edge {
                from,
                to,
                kind: EdgeKind::Bridge,
            });
            asg.adjacency.entry(from).or_default().push(edge_index);
            asg.reverse_adjacency.entry(to).or_default().push(edge_index);
        };

        // Exact same bridge string (two identical SQL statements, two
        // callers of the same endpoint): link every ordered pair.
        for node_ids in by_bridge.values() {
            if node_ids.len() < 2 {
                continue;
            }
            for &from in node_ids {
                for &to in node_ids {
                    connect(from, to, asg, &mut seen);
                }
            }
        }

        // Cross-language semantic bridge: a backend SQL statement over table
        // `users` links to frontend API calls touching `/api/users/...`.
        // This is the headline feature — same-resource, different language.
        for node_ids in by_resource.values() {
            if node_ids.len() < 2 {
                continue;
            }
            for &from in node_ids {
                for &to in node_ids {
                    connect(from, to, asg, &mut seen);
                }
            }
        }

        // Every entry in `seen` corresponds to exactly one pushed edge.
        seen.len()
    }
}

/// Extract table names referenced by a SQL statement (`FROM x`, `JOIN x`,
/// `INTO x`, `UPDATE x`), lowercased for case-insensitive matching.
fn sql_tables(sql: &str) -> Vec<String> {
    let mut tables = Vec::new();
    let upper = sql.to_ascii_uppercase();
    for keyword in ["FROM", "JOIN", "INTO", "UPDATE"] {
        let mut search = 0usize;
        while let Some(pos) = upper[search..].find(keyword) {
            let after = search + pos + keyword.len();
            let rest = &sql[after..];
            let trimmed = rest.trim_start();
            let skipped = rest.len() - trimmed.len();
            // Identifier: letters, digits, underscore.
            let ident: String = trimmed
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !ident.is_empty() {
                tables.push(ident.to_lowercase());
            }
            search = after + skipped + ident.len();
        }
    }
    tables.sort();
    tables.dedup();
    tables
}

/// Extract resource segments from an API path (`/api/users/42` -> `users`).
/// Numeric and common verb segments are ignored.
fn api_resources(path: &str) -> Vec<String> {
    let mut resources = Vec::new();
    for segment in path.split('/') {
        if segment.is_empty()
            || segment.chars().all(|c| c.is_ascii_digit())
            || matches!(
                segment,
                "api" | "v1" | "v2" | "v3" | "get" | "post" | "put" | "delete" | "patch"
            )
        {
            continue;
        }
        resources.push(segment.to_lowercase());
    }
    resources.sort();
    resources.dedup();
    resources
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