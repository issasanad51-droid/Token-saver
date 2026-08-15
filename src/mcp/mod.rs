//! Minimal MCP (Model Context Protocol) server for Token-saver.
//!
//! Exposes three tools over JSON-RPC via stdio:
//! - `search_code`: Search the codebase using the hybrid pipeline
//! - `save_memory`: Store a fact or design decision
//! - `recall`: Recall stored memories matching a query

use std::io::{BufRead, Write};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::memory::MemoryStore;
use crate::search::SearchEngine;
use crate::asg::SharedAsg;
use std::sync::Arc;

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
pub async fn run_mcp_server(
    search_engine: Arc<SearchEngine>,
    asg: SharedAsg,
    memory_store: MemoryStore,
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
        let (result, error) =
            handle_request(request, &search_engine, &asg, &memory_store).await;

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
    memory_store: &MemoryStore,
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
                    description: "Search the codebase using hybrid BM25 + PPR + semantic retrieval".to_string(),
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
                    description: "Store a fact, design decision, or pattern for later recall".to_string(),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "content": { "type": "string", "description": "The memory to store" },
                            "namespace": { "type": "string", "description": "Optional scope tag" }
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
            ];
            (Some(serde_json::json!({ "tools": tools })), None)
        }
        "tools/call" => {
            let params = req.params.unwrap_or(Value::Null);
            let tool_name = params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let args = params
                .get("arguments")
                .cloned()
                .unwrap_or(Value::Null);

            match tool_name {
                "search_code" => {
                    let query = args
                        .get("query")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let top_k = args
                        .get("top_k")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(10) as usize;
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
                    let content = args
                        .get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let namespace = args
                        .get("namespace")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    let id = memory_store.save(content, namespace);
                    (
                        Some(serde_json::json!({
                            "content": [{ "type": "text", "text": format!("Memory saved: {}", id) }]
                        })),
                        None,
                    )
                }
                "recall" => {
                    let query = args
                        .get("query")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let top_k = args
                        .get("top_k")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(10) as usize;
                    let memories = memory_store.recall(query, top_k);
                    (
                        Some(serde_json::json!({
                            "content": [{ "type": "text", "text": serde_json::to_string_pretty(&memories).unwrap_or_default() }]
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
