//! End-to-end accuracy + token-reduction test for the full Token Saver
//! retrieval pipeline: AST chunking -> ASG build -> PPR ranking ->
//! RRF fusion -> compressed dependency packing.
//!
//! We index a small fixture crate, then ask for context at a call site inside
//! `run()`. The system should retrieve only the relevant dependency
//! definitions (`add`, `multiply`) and deliver far fewer tokens than dumping
//! the entire workspace.

use std::fs;

use token_saver::asg::{build_asg_from_dir, SharedAsg};
use token_saver::compressor::ChunkCompressor;
use token_saver::search::{SearchConfig, SearchEngine};
use token_saver::tracker::estimate_tokens;
use token_saver::tracker::{ContextConfig, ContextTracker, CursorPayload};

const MATH_SRC: &str = "\
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn subtract(a: i32, b: i32) -> i32 {
    a - b
}

pub fn multiply(a: i32, b: i32) -> i32 {
    a * b
}

pub fn divide(a: i32, b: i32) -> i32 {
    a / b
}

pub fn power(base: i32, exp: i32) -> i32 {
    let mut result = 1;
    for _ in 0..exp {
        result = result * base;
    }
    result
}

pub fn modulo(a: i32, b: i32) -> i32 {
    a % b
}

pub fn maximum(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}

pub fn minimum(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}
";

const APP_SRC: &str = "\
use crate::math;

pub fn run() -> i32 {
    let x = math::add(2, 3);
    let y = math::multiply(x, 4);
    x + y
}
";

fn write_fixture(root: &std::path::Path) {
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/lib.rs"), "pub mod math;\npub mod app;\n").unwrap();
    fs::write(root.join("src/math.rs"), MATH_SRC).unwrap();
    fs::write(root.join("src/app.rs"), APP_SRC).unwrap();
}

#[tokio::test]
async fn end_to_end_retrieval_accuracy_and_reduction() {
    let root = std::env::temp_dir().join(format!("token_saver_e2e_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    write_fixture(&root);

    // 1. AST -> ASG (with PPR ranking)
    let asg = build_asg_from_dir(&root).expect("ASG build must succeed");
    // 2. AST chunk compression -> ChunkRegistry
    let mut compressor = ChunkCompressor::new();
    let registry = compressor.compress_asg(&asg);
    // 3. Search engine (RRF fusion of semantic + BM25 + structural PPR)
    let shared = SharedAsg::new(asg.clone());
    let search = SearchEngine::with_config(shared.clone(), registry.clone(), SearchConfig::default());
    search.precompute_embeddings().await;
    // 4. Context tracker (cursor -> ASG -> packed compressed deps)
    let tracker = ContextTracker::with_config(
        shared,
        registry,
        search,
        root.clone(),
        ContextConfig::default(),
    );

    // Cursor sits on the `math::add(2, 3)` call inside `run` (0-indexed line 3).
    let cursor = CursorPayload::new("src/app.rs", 3, 17);
    let query = "how is the add function implemented";

    let prompt = tracker.assemble_prompt(&cursor, query).await;

    // ---- Accuracy: the right definitions must be retrieved ----
    assert!(
        prompt.contains("fn add"),
        "delivered prompt must contain the `add` definition:\n{prompt}"
    );
    assert!(
        prompt.contains("fn multiply"),
        "delivered prompt must contain the `multiply` definition:\n{prompt}"
    );

    // ---- Token reduction vs a naive full-source dump ----
    let mut raw_tokens = 0usize;
    for entry in walkdir::WalkDir::new(&root) {
        let path = entry.unwrap().into_path();
        if path.extension().map(|e| e == "rs").unwrap_or(false) {
            let src = fs::read_to_string(&path).unwrap();
            raw_tokens += estimate_tokens(&src);
        }
    }
    let delivered_tokens = estimate_tokens(&prompt);
    let reduction = (raw_tokens.saturating_sub(delivered_tokens)) as f64
        / raw_tokens.max(1) as f64
        * 100.0;

    println!("E2E Token Saver report (AST+ASG+PPR+RRF):");
    println!("  asg nodes              : {}", asg.nodes.len());
    println!("  raw (full dump) tokens : {raw_tokens}");
    println!("  delivered tokens       : {delivered_tokens}");
    println!("  token reduction        : {reduction:.1}%");
    println!("--- delivered prompt ---\n{prompt}");

    assert!(delivered_tokens < raw_tokens, "compression must reduce tokens");
    assert!(
        reduction >= 30.0,
        "expected >=30% reduction, got {reduction:.1}%"
    );

    let _ = fs::remove_dir_all(&root);
}
