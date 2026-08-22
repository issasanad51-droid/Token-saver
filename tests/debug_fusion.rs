//! Debug harness: prints per-stream provenance for the e2e fixture.
use std::fs;

use token_saver::asg::{build_asg_from_dir, SharedAsg};
use token_saver::compressor::ChunkCompressor;
use token_saver::search::{SearchConfig, SearchEngine};

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

#[tokio::test(flavor = "current_thread")]
async fn debug_fusion_provenance() {
    let root = std::env::temp_dir().join(format!("token_saver_dbg_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/lib.rs"), "pub mod math;\npub mod app;\n").unwrap();
    fs::write(root.join("src/math.rs"), MATH_SRC).unwrap();
    fs::write(root.join("src/app.rs"), APP_SRC).unwrap();

    let asg = build_asg_from_dir(&root).unwrap();
    for node in &asg.nodes {
        println!("node {} = {} ({})", node.id, node.tracker_id, node.kind);
    }
    let mut compressor = ChunkCompressor::new();
    let registry = compressor.compress_asg(&asg);
    let shared = SharedAsg::new(asg.clone());
    let search = SearchEngine::with_config(shared.clone(), registry.clone(), SearchConfig::default());
    search.precompute_embeddings().await;

    // Cursor on math::add call inside run.
    let results = search.search_with_context("how is the add function implemented", 10, Some(0)).await;
    println!("--- fused results (context_node=6) ---");
    for r in &results {
        let name = shared.get_node(r.node_id).map(|n| n.name.clone()).unwrap_or_default();
        println!("{:>3} {:<10} rrf={:.5} scores={:?} contrib={:?}",
            r.node_id, name, r.rrf_score, r.scores, r.rrf_contributions);
    }

    let _ = fs::remove_dir_all(&root);
}
