//! End-to-end test for the reindex-refreshes-the-pipeline fix.
//!
//! Writes a tiny Rust workspace to a tempdir, builds an ASG + registry +
//! SearchEngine + SharedIndexes from it, runs a search, mutates one of the
//! source files (adds a new function with a distinctive name), invokes
//! `refresh_pipeline_after_reindex`, runs the search again, and asserts the
//! new code is now visible.
//!
//! Without the ArcSwap refactor + `refresh_after_reindex`, this test would
//! fail because the search engine's cached embeddings/df tables and the
//! ASG snapshot would still reflect the *pre-edit* source.

use std::path::PathBuf;
use std::sync::Arc;

use token_saver::asg::{build_asg_from_dir_with_config, PersonalizedPageRankConfig, SharedAsg};
use token_saver::ast::merkle::MerkleTree;
use token_saver::ast::trigram::TrigramIndex;
use token_saver::compressor::{ChunkCompressor, ChunkRegistry};
use token_saver::search::SearchEngine;
use token_saver::server::refresh_pipeline_after_reindex;
use token_saver::watcher::reindex::SharedIndexes;

/// Write a tiny Rust workspace to `dir` with two files: `lib.rs` and
/// `utils.rs`. The `extra_fn_name` parameter controls whether `utils.rs`
/// contains an extra function with a distinctive name — by varying it
/// between the two builds, we can assert that the search pipeline picks
/// up the new code after a reindex.
///
/// The function name is intentionally composed of nonsense tokens
/// (`zzz_qqq_xxx_marker`) that won't share character trigrams with any
/// real Rust identifier, so BM25/semantic search won't return false
/// positives from the existing `process_value` function.
fn write_workspace(dir: &std::path::Path, extra_fn_name: Option<&str>) -> std::io::Result<()> {
    std::fs::create_dir_all(dir.join("src"))?;
    std::fs::write(
        dir.join("src/lib.rs"),
        "pub mod utils;\npub fn library_entry_point() -> u32 { 42 }\n",
    )?;
    let utils_body = match extra_fn_name {
        Some(name) => {
            format!("pub fn process_value() -> u32 {{ 7 }}\npub fn {name}() -> u32 {{ 99 }}\n")
        }
        None => "pub fn process_value() -> u32 { 7 }\n".to_string(),
    };
    std::fs::write(dir.join("src/utils.rs"), utils_body)?;
    Ok(())
}

/// Build the four artefacts the live pipeline needs from a freshly-written
/// workspace: ASG (with PageRank), ChunkRegistry, TrigramIndex, MerkleTree.
fn build_artefacts(
    workspace: &std::path::Path,
) -> (SharedAsg, Arc<ChunkRegistry>, SearchEngine, SharedIndexes) {
    let ppr = PersonalizedPageRankConfig::default();
    let asg = build_asg_from_dir_with_config(workspace, ppr.clone()).expect("ASG build");
    let shared_asg = SharedAsg::new(asg);

    let mut compressor = ChunkCompressor::new();
    let registry = compressor.compress_asg(&shared_asg.snapshot());
    let registry = Arc::new(registry);

    let search_engine = SearchEngine::with_config(
        shared_asg.clone(),
        (*registry).clone(),
        token_saver::search::SearchConfig::default(),
    );

    // Build trigram + merkle for the SharedIndexes bundle. Token-saver's
    // server normally does this; we replicate the minimal version here.
    let mut chunker = token_saver::ast::chunker::AstChunker::new()
        .map_err(|e| panic!("chunker: {e}"))
        .unwrap()
        .with_crate_root(workspace);
    let chunks = chunker.chunk_dir(workspace).expect("chunk workspace");
    let mut trigram = TrigramIndex::new(3);
    trigram.index(&chunks);
    let merkle = MerkleTree::build(&chunks);
    let indexes = SharedIndexes::new(merkle, trigram);

    (shared_asg, registry, search_engine, indexes)
}

#[tokio::test(flavor = "current_thread")]
async fn reindex_propagates_new_code_to_search_results() {
    // 1. Write a workspace without the `unobtainium_processor` function.
    let dir = std::env::temp_dir().join(format!(
        "token-saver-reindex-test-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    write_workspace(&dir, None).expect("write initial workspace");

    // 2. Build the artefacts and precompute embeddings so the search cache
    //    is *warm* before the reindex — otherwise we wouldn't be testing
    //    cache invalidation, just lazy population.
    let (_asg, registry, search_engine, indexes) = build_artefacts(&dir);
    let search_engine = Arc::new(search_engine);
    search_engine.precompute_embeddings().await;
    let snapshot = search_engine.asg().snapshot();
    let initial_node_count = snapshot.nodes.len();
    assert!(
        initial_node_count > 0,
        "initial ASG should have nodes; got {initial_node_count}"
    );

    // Sanity: before the reindex, *no hit's name or tracker_id contains the
    // marker*. The search engine may return low-confidence fuzzy matches
    // (the semantic-min-score floor is 0.03 and feature-hashed vectors
    // produce incidental cosine overlap), but none of those hits should be
    // the not-yet-existing function.
    let before = search_engine.search("zzz_qqq_xxx_marker", 10).await;
    let before_marker_hits = before
        .iter()
        .filter(|r| {
            search_engine
                .asg()
                .get_node(r.node_id)
                .map(|n| {
                    n.name.contains("zzz_qqq_xxx_marker")
                        || n.tracker_id.contains("zzz_qqq_xxx_marker")
                })
                .unwrap_or(false)
        })
        .count();
    assert_eq!(
        before_marker_hits, 0,
        "no search hit should reference `zzz_qqq_xxx_marker` before reindex"
    );

    // 3. Mutate the workspace on disk — add `zzz_qqq_xxx_marker` to
    //    utils.rs.
    write_workspace(&dir, Some("zzz_qqq_xxx_marker")).expect("write updated workspace");

    // 4. Invoke the same refresh path the file watcher uses.
    refresh_pipeline_after_reindex(
        &dir,
        &PersonalizedPageRankConfig::default(),
        &registry,
        &indexes,
        search_engine.as_ref(),
    )
    .await
    .expect("refresh should succeed");

    // 5. The new ASG should have one more node than before (the new
    //    `unobtainium_processor` fn).
    let post_snapshot = search_engine.asg().snapshot();
    assert_eq!(
        post_snapshot.nodes.len(),
        initial_node_count + 1,
        "post-reindex ASG should have exactly one more node (the new function)"
    );

    // 6. Search again — the new function should now be retrievable.
    let after = search_engine.search("zzz_qqq_xxx_marker", 10).await;
    let after_marker_hits: Vec<_> = after
        .iter()
        .filter_map(|r| {
            let node = search_engine.asg().get_node(r.node_id)?;
            if node.name.contains("zzz_qqq_xxx_marker")
                || node.tracker_id.contains("zzz_qqq_xxx_marker")
            {
                Some(node)
            } else {
                None
            }
        })
        .collect();
    assert!(
        !after_marker_hits.is_empty(),
        "search for `zzz_qqq_xxx_marker` should return at least one hit referencing the new function after reindex; got 0 marker hits out of {} total hits",
        after.len()
    );
    let top_node = &after_marker_hits[0];
    assert!(
        top_node.name.contains("zzz_qqq_xxx_marker"),
        "top marker hit's name should contain `zzz_qqq_xxx_marker`; got name={}",
        top_node.name
    );

    // Cleanup.
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test(flavor = "current_thread")]
async fn reindex_removes_deleted_code_from_search_results() {
    // 1. Write a workspace that contains the `zzz_qqq_xxx_marker` fn.
    let dir = std::env::temp_dir().join(format!(
        "token-saver-reindex-del-test-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    write_workspace(&dir, Some("zzz_qqq_xxx_marker")).expect("write initial workspace");

    let (_asg, registry, search_engine, indexes) = build_artefacts(&dir);
    let search_engine = Arc::new(search_engine);
    search_engine.precompute_embeddings().await;

    // Sanity: the function is initially retrievable.
    let before = search_engine.search("zzz_qqq_xxx_marker", 10).await;
    assert!(
        !before.is_empty(),
        "search should find zzz_qqq_xxx_marker before deletion"
    );

    // 2. Remove the function from utils.rs.
    write_workspace(&dir, None).expect("write updated workspace");

    // 3. Refresh.
    refresh_pipeline_after_reindex(
        &dir,
        &PersonalizedPageRankConfig::default(),
        &registry,
        &indexes,
        search_engine.as_ref(),
    )
    .await
    .expect("refresh should succeed");

    // 4. The function should no longer be retrievable. The search may
    //    still return low-confidence fuzzy matches, but none of them should
    //    reference the now-deleted function name.
    let after = search_engine.search("zzz_qqq_xxx_marker", 10).await;
    let after_marker_hits = after
        .iter()
        .filter(|r| {
            search_engine
                .asg()
                .get_node(r.node_id)
                .map(|n| {
                    n.name.contains("zzz_qqq_xxx_marker")
                        || n.tracker_id.contains("zzz_qqq_xxx_marker")
                })
                .unwrap_or(false)
        })
        .count();
    assert_eq!(
        after_marker_hits, 0,
        "no search hit should reference `zzz_qqq_xxx_marker` after the function was deleted and a reindex ran; got {} marker hits out of {} total hits",
        after_marker_hits,
        after.len()
    );

    std::fs::remove_dir_all(&dir).ok();
}

// PathBuf import is unused on some toolchains but kept for future expansion.
#[allow(dead_code)]
fn _pathbuf_hint() -> PathBuf {
    PathBuf::new()
}
