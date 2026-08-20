//! Minimal MCP (Model Context Protocol) server for Token-saver.
//!
//! Exposes tools over JSON-RPC via stdio:
//! - `search_code`: Search the codebase using the hybrid pipeline
//! - `save_memory`: Store a fact or design decision
//! - `recall`: Recall stored memories matching a query
//! - `forget_memory`: Delete a memory by id
//! - `health`: Return server health/info
//! - `list_files`: List all indexed files with node counts
//! - `get_context`: Get context around a cursor position
//! - `update_memory`: Update an existing memory's importance or namespace

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::asg::SharedAsg;
use crate::memory::MemoryStore;
use crate::search::SearchEngine;
use crate::tracker::CursorPayload;

// MCP JSON-RPC types
#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Value,
    result: Option<Value>,
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

#[derive(Debug, Serialize)]
struct ToolInfo {
    name: String,
    description: String,
    input_schema: Value,
}

/// Run the MCP server on stdin/stdout.
///
/// `workspace_root` is the canonicalized workspace path used to resolve
/// `get_context` `file_path` arguments safely (paths must stay inside the
/// workspace) and to translate `line`/`column` cursor positions into UTF-8
/// byte offsets against the file's source.
pub async fn run_mcp_server(
    search_engine: Arc<SearchEngine>,
    asg: SharedAsg,
    memory_store: Arc<tokio::sync::Mutex<MemoryStore>>,
    workspace_root: PathBuf,
) -> anyhow::Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let request: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: Value::Null,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: format!("Parse error: {e}"),
                    }),
                };
                writeln!(stdout, "{}", serde_json::to_string(&resp)?)?;
                stdout.flush()?;
                continue;
            }
        };

        let id = request.id.clone().unwrap_or(Value::Null);
        let (result, error) = handle_request(
            request,
            &search_engine,
            &asg,
            &memory_store,
            &workspace_root,
        )
        .await;

        let resp = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result,
            error,
        };
        writeln!(stdout, "{}", serde_json::to_string(&resp)?)?;
        stdout.flush()?;
    }

    Ok(())
}

async fn handle_request(
    req: JsonRpcRequest,
    search_engine: &SearchEngine,
    asg: &SharedAsg,
    memory_store: &Arc<tokio::sync::Mutex<MemoryStore>>,
    workspace_root: &std::path::Path,
) -> (Option<Value>, Option<JsonRpcError>) {
    match req.method.as_str() {
        "initialize" => {
            let result = serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "token-saver",
                    "version": env!("CARGO_PKG_VERSION")
                }
            });
            (Some(result), None)
        }
        "tools/list" => {
            let tools = vec![
                ToolInfo {
                    name: "search_code".to_string(),
                    description: "Search the codebase using hybrid BM25 + PPR + semantic retrieval"
                        .to_string(),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "query": { "type": "string", "description": "Search query" },
                            "top_k": { "type": "integer", "description": "Max results", "default": 10 }
                        },
                        "required": ["query"]
                    }),
                },
                ToolInfo {
                    name: "save_memory".to_string(),
                    description: "Store a fact, design decision, or pattern for later recall"
                        .to_string(),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "content": { "type": "string", "description": "The memory to store" },
                            "namespace": { "type": "string", "description": "Optional scope tag" },
                            "importance": { "type": "number", "description": "Optional importance 0.0-1.0" }
                        },
                        "required": ["content"]
                    }),
                },
                ToolInfo {
                    name: "recall".to_string(),
                    description: "Recall stored memories matching a query".to_string(),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "query": { "type": "string", "description": "Search query for memories" },
                            "top_k": { "type": "integer", "description": "Max results", "default": 10 }
                        },
                        "required": ["query"]
                    }),
                },
                ToolInfo {
                    name: "forget_memory".to_string(),
                    description: "Delete a stored memory by its id".to_string(),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "id": { "type": "string", "description": "The memory id to delete" }
                        },
                        "required": ["id"]
                    }),
                },
                ToolInfo {
                    name: "health".to_string(),
                    description: "Return server health and indexing statistics".to_string(),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {}
                    }),
                },
                ToolInfo {
                    name: "list_files".to_string(),
                    description: "List all indexed source files with their node counts".to_string(),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {}
                    }),
                },
                ToolInfo {
                    name: "get_context".to_string(),
                    description: "Get context around a cursor position in a file".to_string(),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "file_path": { "type": "string", "description": "Path to the source file" },
                            "line": { "type": "integer", "description": "Line number (0-indexed)" },
                            "column": { "type": "integer", "description": "Column number (0-indexed)" }
                        },
                        "required": ["file_path", "line", "column"]
                    }),
                },
                ToolInfo {
                    name: "update_memory".to_string(),
                    description: "Update an existing memory's importance or namespace".to_string(),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "id": { "type": "string", "description": "The memory id to update" },
                            "importance": { "type": "number", "description": "New importance 0.0-1.0" },
                            "namespace": { "type": "string", "description": "New namespace" }
                        },
                        "required": ["id"]
                    }),
                },
            ];
            (Some(serde_json::json!({ "tools": tools })), None)
        }
        "tools/call" => {
            let params = req.params.unwrap_or(Value::Null);
            let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(Value::Null);

            match tool_name {
                "search_code" => {
                    let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                    let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
                    let results = search_engine.search(query, top_k).await;
                    let hits: Vec<serde_json::Value> = results
                        .iter()
                        .filter_map(|r| {
                            asg.get_node(r.node_id).map(|node| {
                                serde_json::json!({
                                    "name": node.name,
                                    "kind": node.kind,
                                    "file": node.file_path.display().to_string(),
                                    "pagerank": node.pagerank,
                                    "rrf_score": r.rrf_score
                                })
                            })
                        })
                        .collect();
                    (
                        Some(serde_json::json!({
                            "content": [{ "type": "text", "text": serde_json::to_string_pretty(&hits).unwrap_or_default() }]
                        })),
                        None,
                    )
                }
                "save_memory" => {
                    let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    let namespace = args
                        .get("namespace")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    let importance = args.get("importance").and_then(|v| v.as_f64());
                    let id = memory_store
                        .lock()
                        .await
                        .save(content, namespace, importance);
                    (
                        Some(serde_json::json!({
                            "content": [{ "type": "text", "text": format!("Memory saved: {}", id) }]
                        })),
                        None,
                    )
                }
                "recall" => {
                    let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                    let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
                    let memories = memory_store.lock().await.recall(query, top_k);
                    (
                        Some(serde_json::json!({
                            "content": [{ "type": "text", "text": serde_json::to_string_pretty(&memories).unwrap_or_default() }]
                        })),
                        None,
                    )
                }
                "forget_memory" => {
                    let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let forgotten = memory_store.lock().await.forget(id);
                    (
                        Some(serde_json::json!({
                            "content": [{ "type": "text", "text": if forgotten { "Memory forgotten" } else { "Memory not found" } }]
                        })),
                        None,
                    )
                }
                "health" => {
                    let mem = memory_store.lock().await;
                    let result = serde_json::json!({
                        "nodes": asg.inner.nodes.len(),
                        "edges": asg.inner.edges.len(),
                        "files": asg.inner.file_index.len(),
                        "symbols": asg.inner.symbol_table.len(),
                        "memories": mem.len()
                    });
                    drop(mem);
                    (
                        Some(serde_json::json!({
                            "content": [{ "type": "text", "text": serde_json::to_string_pretty(&result).unwrap_or_default() }]
                        })),
                        None,
                    )
                }
                "list_files" => {
                    let files: Vec<serde_json::Value> = asg
                        .inner
                        .file_index
                        .iter()
                        .map(|(path, node_ids)| {
                            serde_json::json!({
                                "path": path.display().to_string(),
                                "nodes": node_ids.len()
                            })
                        })
                        .collect();
                    (
                        Some(serde_json::json!({
                            "content": [{ "type": "text", "text": serde_json::to_string_pretty(&files).unwrap_or_default() }]
                        })),
                        None,
                    )
                }
                "get_context" => {
                    let file_path = args.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
                    let line = args.get("line").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    let column = args.get("column").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

                    // Resolve the requested path against the workspace and
                    // reject anything that escapes it (same containment
                    // contract as the HTTP `context_handler`).
                    let resolved = resolve_workspace_path(workspace_root, file_path);
                    let node_info = resolved.and_then(|path| {
                        let source = std::fs::read_to_string(&path).ok()?;
                        let byte_offset =
                            CursorPayload::new(file_path, line, column).to_byte_offset(&source);
                        let node_ids = asg.inner.file_index.get(&path)?;
                        // Pick the *smallest* node whose byte range contains
                        // the cursor — the previous implementation returned the
                        // first node in the file regardless of cursor
                        // position, which made `line` and `column` no-ops.
                        node_ids
                            .iter()
                            .filter_map(|id| asg.get_node(*id))
                            .filter(|node| {
                                byte_offset >= node.range.0 && byte_offset < node.range.1
                            })
                            .min_by_key(|node| node.range.1.saturating_sub(node.range.0))
                            .map(|node| {
                                serde_json::json!({
                                    "name": node.name,
                                    "kind": node.kind,
                                    "tracker_id": node.tracker_id,
                                    "pagerank": node.pagerank,
                                    "range": [node.range.0, node.range.1]
                                })
                            })
                    });

                    let result = serde_json::json!({
                        "file": file_path,
                        "cursor": [line, column],
                        "node": node_info
                    });

                    (
                        Some(serde_json::json!({
                            "content": [{ "type": "text", "text": serde_json::to_string_pretty(&result).unwrap_or_default() }]
                        })),
                        None,
                    )
                }
                "update_memory" => {
                    let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let importance = args.get("importance").and_then(|v| v.as_f64());
                    let namespace = args
                        .get("namespace")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    let updated = memory_store
                        .lock()
                        .await
                        .update(id, importance, Some(namespace));
                    (
                        Some(serde_json::json!({
                            "content": [{ "type": "text", "text": if updated { "Memory updated" } else { "Memory not found" } }]
                        })),
                        None,
                    )
                }
                _ => (
                    None,
                    Some(JsonRpcError {
                        code: -32601,
                        message: format!("Unknown tool: {tool_name}"),
                    }),
                ),
            }
        }
        _ => (
            None,
            Some(JsonRpcError {
                code: -32601,
                message: format!("Method not found: {}", req.method),
            }),
        ),
    }
}

/// Resolve a `file_path` argument against the workspace root and enforce
/// path containment. Returns the canonicalized absolute path only when it
/// lives inside `workspace_root`; returns `None` otherwise (or when the path
/// does not exist on disk, which is normal for unsaved editor buffers).
fn resolve_workspace_path(workspace_root: &std::path::Path, requested: &str) -> Option<PathBuf> {
    let requested = std::path::Path::new(requested);
    let candidate = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        workspace_root.join(requested)
    };
    let canonical = candidate.canonicalize().ok()?;
    canonical.starts_with(workspace_root).then_some(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_rejects_paths_outside_workspace() {
        let tmp = std::env::temp_dir().join(format!(
            "token-saver-mcp-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let outside = tmp.parent().unwrap().join("..").join("..");
        let resolved = resolve_workspace_path(&tmp, &outside.to_string_lossy());
        // The escaped path (after canonicalization) must not start with tmp.
        // Note: if `outside` happens to canonicalize back into tmp's parent
        // chain it's still rejected because it isn't inside tmp.
        assert!(
            resolved.is_none()
                || resolved
                    .as_ref()
                    .map(|p| p.starts_with(&tmp))
                    .unwrap_or(false)
        );
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn resolve_accepts_relative_path_inside_workspace() {
        let tmp = std::env::temp_dir().join(format!(
            "token-saver-mcp-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
                + 1
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let inner = tmp.join("inner.rs");
        std::fs::write(&inner, "fn main() {}").unwrap();
        let resolved = resolve_workspace_path(&tmp, "inner.rs");
        assert_eq!(resolved, Some(inner));
        std::fs::remove_dir_all(&tmp).ok();
    }
}
