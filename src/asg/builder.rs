//! Custom ASG builder: semantic entity extraction + cross-file resolution.
//!
//! The parser turns source into a **sparse, semantic** `AsgGraph` rather than a
//! full AST: only meaningful entities become nodes (`fn` / `struct` / `enum` /
//! `trait` / `impl` / `type` / `const` / `static` / `macro` / `mod`), and
//! structural edges are resolved *across files* through a module-aware symbol
//! table:
//!
//! - `calls`       — resolved callee identifiers (incl. `a::b::f()`, `x.y()`)
//! - `references`  — resolved `type_identifier`s appearing in signatures/bodies
//! - `implements`  — `impl Trait for Type` → the resolved `Trait` node
//! - `contains`    — parent item → nested item (impl → fn, mod → item)
//!
//! Bodies borrow directly from `SourceSet`-owned buffers (`Cow::Borrowed`), so
//! parsing is zero-copy: no body is ever allocated or duplicated.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

use tree_sitter::{Node as TsNode, Parser};

use super::graph::{AsgGraph, EdgeKind, NodeId, NodeType};
use super::source::SourceSet;
use crate::asg::AsgError;

// ---------------------------------------------------------------------------
// Pending edges (resolved in a second pass once the symbol table is complete)
// ---------------------------------------------------------------------------

struct PendingCall {
    caller: NodeId,
    callee: String,
    module: Vec<String>,
}

struct PendingRef {
    user: NodeId,
    type_name: String,
    module: Vec<String>,
}

struct PendingImpl {
    impl_id: NodeId,
    trait_name: String,
    module: Vec<String>,
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builds an `AsgGraph` from a `SourceSet`.
pub struct AsgBuilder<'a> {
    parser: Parser,
    source_set: &'a SourceSet,
    crate_root: PathBuf,
    graph: AsgGraph<'a>,
    /// `module::name` and bare `name` -> node id.
    symbols: HashMap<String, NodeId>,
    pending_calls: Vec<PendingCall>,
    pending_refs: Vec<PendingRef>,
    pending_impls: Vec<PendingImpl>,
    /// Non-fatal issues encountered while parsing (e.g. duplicate ids).
    pub warnings: Vec<String>,
}

impl<'a> AsgBuilder<'a> {
    pub fn new(source_set: &'a SourceSet) -> Result<Self, AsgError> {
        let mut parser = Parser::new();
        let rust_lang: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
        parser
            .set_language(&rust_lang)
            .map_err(|e| AsgError::ParseError(e.to_string()))?;
        Ok(Self {
            parser,
            source_set,
            crate_root: PathBuf::from("/"),
            graph: AsgGraph::new(),
            symbols: HashMap::new(),
            pending_calls: Vec::new(),
            pending_refs: Vec::new(),
            pending_impls: Vec::new(),
            warnings: Vec::new(),
        })
    }

    /// Set the root against which file paths are mapped to module paths.
    pub fn with_crate_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.crate_root = root.into();
        self
    }

    /// Access the partially-built graph (useful for debugging).
    pub fn graph(&self) -> &AsgGraph<'a> {
        &self.graph
    }

    /// Parse every file in the source set.
    pub fn parse(&mut self) -> Result<(), AsgError> {
        let mut paths: Vec<PathBuf> = self.source_set.paths().cloned().collect();
        paths.sort();
        for path in paths {
            self.parse_file(&path)?;
        }
        Ok(())
    }

    /// Parse a single file (already present in the source set).
    pub fn parse_file(&mut self, path: &Path) -> Result<(), AsgError> {
        let source: &'a str = self.source_set.get(path).ok_or_else(|| {
            AsgError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{} not in SourceSet", path.display()),
            ))
        })?;

        let tree = self
            .parser
            .parse(source.as_bytes(), None)
            .ok_or_else(|| AsgError::ParseError(format!("failed to parse {}", path.display())))?;

        let module = self.module_path_for(path);
        let root = tree.root_node();
        self.walk_items(root, source, path, &module, None);
        Ok(())
    }

    /// Resolve all deferred edges and produce the final graph.
    pub fn build(mut self) -> AsgGraph<'a> {
        self.resolve_edges();
        self.graph
    }

    // -----------------------------------------------------------------------
    // Module path derivation
    // -----------------------------------------------------------------------

    fn module_path_for(&self, path: &Path) -> Vec<String> {
        let rel = path.strip_prefix(&self.crate_root).unwrap_or(path);
        let mut segs: Vec<String> = rel
            .components()
            .filter_map(|c| match c {
                Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
                _ => None,
            })
            .collect();

        match segs.last().map(String::as_str) {
            Some("mod.rs") | Some("lib.rs") | Some("main.rs") => {
                segs.pop();
            }
            Some(other) => {
                if let Some(stripped) = other.strip_suffix(".rs") {
                    *segs.last_mut().unwrap() = stripped.to_string();
                }
            }
            None => {}
        }
        segs
    }

    // -----------------------------------------------------------------------
    // Item extraction
    // -----------------------------------------------------------------------

    /// Dispatch over the named children of a container (source_file / mod body /
    /// impl body), creating a node for each recognized item.
    fn walk_items(
        &mut self,
        container: TsNode,
        source: &'a str,
        path: &Path,
        module: &[String],
        parent: Option<NodeId>,
    ) {
        for i in 0..container.named_child_count() {
            if let Some(child) = container.named_child(i) {
                self.add_item(child, source, path, module, parent.clone());
            }
        }
    }

    fn add_item(
        &mut self,
        node: TsNode,
        source: &'a str,
        path: &Path,
        module: &[String],
        parent: Option<NodeId>,
    ) {
        match node.kind() {
            "function_item" => self.add_fn(node, source, path, module, parent),
            "struct_item" => self.add_typed(
                node,
                source,
                path,
                module,
                parent,
                NodeType::Struct,
                "struct",
            ),
            "enum_item" => {
                self.add_typed(node, source, path, module, parent, NodeType::Enum, "enum")
            }
            "trait_item" => {
                self.add_typed(node, source, path, module, parent, NodeType::Trait, "trait")
            }
            "type_item" => {
                self.add_typed(node, source, path, module, parent, NodeType::Type, "type")
            }
            "const_item" => {
                self.add_typed(node, source, path, module, parent, NodeType::Const, "const")
            }
            "static_item" => self.add_typed(
                node,
                source,
                path,
                module,
                parent,
                NodeType::Static,
                "static",
            ),
            "macro_definition" => {
                self.add_typed(node, source, path, module, parent, NodeType::Macro, "macro")
            }
            "impl_item" => self.add_impl(node, source, path, module, parent),
            "mod_item" => self.add_mod(node, source, path, module, parent),
            _ => {}
        }
    }

    fn add_fn(
        &mut self,
        node: TsNode,
        source: &'a str,
        path: &Path,
        module: &[String],
        parent: Option<NodeId>,
    ) {
        let Some(name) = field_text(node, "name", source) else {
            return;
        };
        let id = self.scoped_id(module, parent.as_ref(), "fn", &name);
        self.insert_node(node, source, path, module, parent, id, NodeType::Fn);
    }

    fn add_typed(
        &mut self,
        node: TsNode,
        source: &'a str,
        path: &Path,
        module: &[String],
        parent: Option<NodeId>,
        node_type: NodeType,
        kind_label: &str,
    ) {
        let Some(name) = field_text(node, "name", source) else {
            return;
        };
        let id = self.scoped_id(module, parent.as_ref(), kind_label, &name);
        self.insert_node(node, source, path, module, parent, id, node_type);
    }

    /// `impl` nodes have no `name` field, so the tracker id is synthesized from
    /// the self type (and optional trait): `impl::for_AppState`,
    /// `impl::Handler_for_AppState`. Nested functions/consts become children.
    fn add_impl(
        &mut self,
        node: TsNode,
        source: &'a str,
        path: &Path,
        module: &[String],
        parent: Option<NodeId>,
    ) {
        let self_type = field_text(node, "type", source).unwrap_or_else(|| "self".to_string());
        let trait_name = field_text(node, "trait", source);
        let label = match &trait_name {
            Some(t) => format!("{t}_for_{self_type}"),
            None => format!("for_{self_type}"),
        };
        let base_id = self.scoped_id(module, parent.as_ref(), "impl", &label);
        let id = self.unique_id(base_id);

        self.insert_node(
            node,
            source,
            path,
            module,
            parent,
            id.clone(),
            NodeType::Impl,
        );

        if let Some(t) = trait_name {
            self.pending_impls.push(PendingImpl {
                impl_id: id.clone(),
                trait_name: t,
                module: module.to_vec(),
            });
        }

        if let Some(body) = node.child_by_field_name("body") {
            self.walk_items(body, source, path, module, Some(id));
        }
    }

    fn add_mod(
        &mut self,
        node: TsNode,
        source: &'a str,
        path: &Path,
        module: &[String],
        parent: Option<NodeId>,
    ) {
        let Some(name) = field_text(node, "name", source) else {
            return;
        };

        // Bare `mod foo;` file declarations are resolved through the module
        // path, not as graph entities — only inline `mod { .. }` bodies become
        // nodes, keeping the graph sparse and structural.
        let Some(body) = node.child_by_field_name("body") else {
            return;
        };

        let id = self.scoped_id(module, parent.as_ref(), "mod", &name);
        self.insert_node(
            node,
            source,
            path,
            module,
            parent,
            id.clone(),
            NodeType::Mod,
        );

        // Inline module: recurse with an extended module path.
        let mut child_module = module.to_vec();
        child_module.push(name);
        self.walk_items(body, source, path, &child_module, Some(id));
    }

    /// Shared insertion: node creation, provenance, `contains` edge, symbol
    /// registration, and deferred call/ref collection.
    fn insert_node(
        &mut self,
        node: TsNode,
        source: &'a str,
        path: &Path,
        module: &[String],
        parent: Option<NodeId>,
        id: NodeId,
        node_type: NodeType,
    ) {
        if self.graph.contains(&id) {
            self.warnings.push(format!("duplicate id skipped: {id}"));
            return;
        }
        let body = Cow::Borrowed(&source[node.start_byte()..node.end_byte()]);
        let range = (node.start_byte(), node.end_byte());
        let _ = self.graph.add_node(id.clone(), node_type, body);
        let _ = self.graph.attach_location(&id, path.to_path_buf(), range);

        if let Some(p) = parent.as_ref() {
            let _ = self.graph.add_edge(p, &id, EdgeKind::Contains);
        }

        self.register_symbol(&id, module, parent.as_ref());
        self.collect_calls_and_refs(node, source, id.clone(), module);
    }

    /// Build a `crate::…::kind::name` tracker id from an owned module path.
    fn qualified_id(&self, module: &[String], kind: &str, name: &str) -> NodeId {
        let strs: Vec<&str> = module.iter().map(String::as_str).collect();
        NodeId::qualified("crate", &strs, kind, name)
    }

    /// Include the lexical owner in nested item IDs. This prevents common
    /// method names such as `new` from colliding across multiple impl blocks in
    /// the same module.
    fn scoped_id(
        &self,
        module: &[String],
        parent: Option<&NodeId>,
        kind: &str,
        name: &str,
    ) -> NodeId {
        match parent {
            Some(parent) => NodeId::new(format!("{}::{kind}::{name}", parent.as_str())),
            None => self.qualified_id(module, kind, name),
        }
    }

    /// Rust permits multiple inherent impl blocks for one type. Preserve the
    /// compact base tracker for the first and deterministically number later
    /// blocks rather than dropping them as duplicate nodes.
    fn unique_id(&self, base: NodeId) -> NodeId {
        if !self.graph.contains(&base) {
            return base;
        }
        for suffix in 2usize.. {
            let candidate = NodeId::new(format!("{}::block::{suffix}", base.as_str()));
            if !self.graph.contains(&candidate) {
                return candidate;
            }
        }
        unreachable!("unbounded impl tracker suffix space")
    }

    fn register_symbol(&mut self, id: &NodeId, module: &[String], parent: Option<&NodeId>) {
        let name = id.0.rsplit("::").next().unwrap_or("").to_string();

        // Keep the first bare-name binding. A bare method such as `new` is
        // inherently ambiguous; replacing it on every file made resolution
        // depend on traversal order.
        self.symbols
            .entry(name.clone())
            .or_insert_with(|| id.clone());

        let mut parts = module.to_vec();
        parts.push(name.clone());
        self.symbols
            .entry(parts.join("::"))
            .or_insert_with(|| id.clone());

        // Methods also get a `Type::method` alias derived from their impl owner,
        // allowing `AppState::new()` to resolve without a compiler type pass.
        if let Some(parent) = parent {
            if let Some(owner) = impl_owner(parent.as_str()) {
                self.symbols
                    .entry(format!("{owner}::{name}"))
                    .or_insert_with(|| id.clone());
                let mut qualified = module.to_vec();
                qualified.push(owner);
                qualified.push(name);
                self.symbols
                    .entry(qualified.join("::"))
                    .or_insert_with(|| id.clone());
            }
        }
    }

    // -----------------------------------------------------------------------
    // Deferred edge collection
    // -----------------------------------------------------------------------

    /// Scan an item's subtree for call expressions and type references, skipping
    /// nested items (they own their own edges) and the item's own `name` field.
    fn collect_calls_and_refs(
        &mut self,
        item: TsNode,
        source: &'a str,
        owner: NodeId,
        module: &[String],
    ) {
        let name_field = item.child_by_field_name("name");
        for i in 0..item.named_child_count() {
            if let Some(child) = item.named_child(i) {
                if name_field.map_or(false, |n| n == child) {
                    continue;
                }
                if is_item_kind(child.kind()) {
                    continue;
                }
                self.scan_subtree(child, source, owner.clone(), module);
            }
        }
    }

    fn scan_subtree(&mut self, node: TsNode, source: &'a str, owner: NodeId, module: &[String]) {
        match node.kind() {
            "call_expression" => {
                if let Some(callee) = node.child_by_field_name("function") {
                    if let Some(name) = callee_name(callee, source) {
                        self.pending_calls.push(PendingCall {
                            caller: owner.clone(),
                            callee: name,
                            module: module.to_vec(),
                        });
                    }
                }
            }
            "type_identifier" => {
                if let Ok(text) = node.utf8_text(source.as_bytes()) {
                    self.pending_refs.push(PendingRef {
                        user: owner.clone(),
                        type_name: text.to_string(),
                        module: module.to_vec(),
                    });
                }
            }
            _ => {}
        }
        for i in 0..node.named_child_count() {
            if let Some(child) = node.named_child(i) {
                if is_item_kind(child.kind()) {
                    continue;
                }
                self.scan_subtree(child, source, owner.clone(), module);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Second pass: symbol resolution
    // -----------------------------------------------------------------------

    fn resolve_edges(&mut self) {
        for pc in std::mem::take(&mut self.pending_calls) {
            if let Some(target) = self.resolve_name(&pc.callee, &pc.module) {
                if target != pc.caller {
                    let _ = self.graph.add_edge(&pc.caller, &target, EdgeKind::Calls);
                }
            }
        }
        for pr in std::mem::take(&mut self.pending_refs) {
            if let Some(target) = self.resolve_name(&pr.type_name, &pr.module) {
                if target != pr.user {
                    let _ = self.graph.add_edge(&pr.user, &target, EdgeKind::References);
                }
            }
        }
        for pi in std::mem::take(&mut self.pending_impls) {
            if let Some(target) = self.resolve_name(&pi.trait_name, &pi.module) {
                if target != pi.impl_id {
                    let _ = self
                        .graph
                        .add_edge(&pi.impl_id, &target, EdgeKind::Implements);
                }
            }
        }
    }

    /// Resolve a name: try module-qualified keys (walking up the module chain),
    /// then fall back to a global bare-name match.
    fn resolve_name(&self, name: &str, module: &[String]) -> Option<NodeId> {
        let normalized = normalize_symbol(name);
        for i in (0..=module.len()).rev() {
            let prefix = module[..i].join("::");
            let key = if prefix.is_empty() {
                normalized.clone()
            } else {
                format!("{prefix}::{normalized}")
            };
            if let Some(id) = self.symbols.get(&key) {
                return Some(id.clone());
            }
        }
        self.symbols.get(&normalized).cloned().or_else(|| {
            normalized
                .rsplit("::")
                .next()
                .and_then(|bare| self.symbols.get(bare).cloned())
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_item_kind(kind: &str) -> bool {
    matches!(
        kind,
        "function_item"
            | "struct_item"
            | "enum_item"
            | "trait_item"
            | "impl_item"
            | "type_item"
            | "const_item"
            | "static_item"
            | "macro_definition"
            | "mod_item"
    )
}

fn field_text(node: TsNode, field: &str, source: &str) -> Option<String> {
    node.child_by_field_name(field)
        .and_then(|n| n.utf8_text(source.as_bytes()).ok())
        .map(str::to_string)
}

/// Return the self type encoded in an impl tracker such as
/// `crate::state::impl::for_AppState` or `...::impl::Display_for_AppState`.
fn impl_owner(parent: &str) -> Option<String> {
    let marker = "::impl::";
    let label = parent
        .rsplit_once(marker)?
        .1
        .split("::block::")
        .next()
        .unwrap_or_default();
    let owner = label
        .rsplit_once("_for_")
        .map(|(_, owner)| owner)
        .or_else(|| label.strip_prefix("for_"))?;
    Some(normalize_symbol(owner))
}

/// Strip Rust path prefixes/generic arguments down to the stable symbol form
/// used by the lightweight resolver.
fn normalize_symbol(name: &str) -> String {
    let no_generics = name.split('<').next().unwrap_or(name).trim();
    no_generics
        .trim_start_matches("crate::")
        .trim_start_matches("self::")
        .trim_start_matches("super::")
        .replace(' ', "")
}

/// Extract the callable name from a call expression's `function` field.
/// Handles `f()`, `a::b::f()`, `x.f()`, `f::<T>()`.
fn callee_name(callee: TsNode, source: &str) -> Option<String> {
    match callee.kind() {
        "identifier" => callee.utf8_text(source.as_bytes()).ok().map(str::to_string),
        "field_expression" => callee
            .child_by_field_name("field")
            .and_then(|n| n.utf8_text(source.as_bytes()).ok())
            .map(str::to_string),
        "scoped_identifier" => callee
            .utf8_text(source.as_bytes())
            .ok()
            .map(normalize_symbol),
        "generic_function" => callee
            .child_by_field_name("function")
            .and_then(|n| n.utf8_text(source.as_bytes()).ok())
            .map(normalize_symbol),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_sourceset() -> SourceSet {
        let mut ss = SourceSet::new();
        ss.insert(
            PathBuf::from("src/lib.rs"),
            r#"pub mod server;
pub fn main() {
    server::handler::run();
}
"#
            .to_string(),
        );
        ss.insert(
            PathBuf::from("src/server/mod.rs"),
            r#"pub mod handler;
pub mod state;
pub mod service;
"#
            .to_string(),
        );
        ss.insert(
            PathBuf::from("src/server/handler.rs"),
            r#"pub fn run() {
    let s = state::AppState::new();
    service::serve(s);
}
"#
            .to_string(),
        );
        ss.insert(
            PathBuf::from("src/server/state.rs"),
            r#"pub struct AppState { pub db: Database }
pub struct Database;
impl AppState {
    pub fn new() -> AppState {
        AppState { db: Database }
    }
}
"#
            .to_string(),
        );
        ss.insert(
            PathBuf::from("src/server/service.rs"),
            r#"pub fn serve(s: AppState) {}
"#
            .to_string(),
        );
        ss
    }

    fn has_edge(g: &AsgGraph<'_>, from: &NodeId, to: &NodeId, kind: EdgeKind) -> bool {
        g.outgoing_edges(from)
            .iter()
            .any(|e| e.kind == kind && &e.node == to)
    }

    #[test]
    fn builds_cross_file_semantic_graph() {
        let ss = test_sourceset();
        let mut builder = AsgBuilder::new(&ss).unwrap().with_crate_root("src");
        builder.parse().unwrap();
        let g = builder.build();

        let main_id = NodeId::qualified("crate", &[], "fn", "main");
        let run_id = NodeId::qualified("crate", &["server", "handler"], "fn", "run");
        let app_state_id = NodeId::qualified("crate", &["server", "state"], "struct", "AppState");
        let db_id = NodeId::qualified("crate", &["server", "state"], "struct", "Database");
        let serve_id = NodeId::qualified("crate", &["server", "service"], "fn", "serve");
        let impl_id = NodeId::qualified("crate", &["server", "state"], "impl", "for_AppState");
        let new_id = NodeId::new(format!("{}::fn::new", impl_id.as_str()));

        // Entities exist with the expected global tracker ids.
        for id in [
            &main_id,
            &run_id,
            &app_state_id,
            &db_id,
            &new_id,
            &serve_id,
            &impl_id,
        ] {
            assert!(g.contains(id), "missing node {id}");
        }

        // Body isolation + provenance.
        let run = g.node(&run_id).unwrap();
        assert!(run.body.starts_with("pub fn run()"));
        assert!(run.range.0 < run.range.1);
        assert!(run.file.is_some());

        // Cross-file structural edges.
        assert!(has_edge(&g, &main_id, &run_id, EdgeKind::Calls));
        assert!(has_edge(&g, &run_id, &serve_id, EdgeKind::Calls));
        assert!(has_edge(&g, &run_id, &new_id, EdgeKind::Calls));

        // `References` come from *type positions*: `new() -> AppState` return
        // type, `serve(s: AppState)` param type, `impl AppState` self-type,
        // and the `AppState { db: Database }` field type. (`run` only touches
        // `AppState` in expression position, so it correctly has no ref edge.)
        assert!(has_edge(&g, &new_id, &app_state_id, EdgeKind::References));
        assert!(has_edge(&g, &serve_id, &app_state_id, EdgeKind::References));
        assert!(has_edge(&g, &impl_id, &app_state_id, EdgeKind::References));
        assert!(has_edge(&g, &app_state_id, &db_id, EdgeKind::References));
        assert!(has_edge(&g, &impl_id, &new_id, EdgeKind::Contains));

        // No dangling self-edges.
        let g_ref = &g;
        assert!(!has_edge(g_ref, &run_id, &run_id, EdgeKind::Calls));
    }

    #[test]
    fn module_qualified_ids_are_unique() {
        let ss = test_sourceset();
        let mut builder = AsgBuilder::new(&ss).unwrap().with_crate_root("src");
        builder.parse().unwrap();
        let g = builder.build();

        // Two `fn` items named differently across modules must not collide.
        let run_id = NodeId::qualified("crate", &["server", "handler"], "fn", "run");
        let serve_id = NodeId::qualified("crate", &["server", "service"], "fn", "serve");
        assert_ne!(run_id, serve_id);
        assert_eq!(g.node_count(), 7);
    }
}
