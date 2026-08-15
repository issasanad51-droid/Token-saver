//! Phase 6: Cursor-style AST Chunking & Incremental Indexing
//!
//! Implements the four pillars of Cursor's codebase indexing:
//!
//! 1. **AST Chunking** ([`chunker`]) — tree-sitter parses each file into an
//!    AST; code is split along logical structural boundaries (full functions,
//!    methods, structs, impls, enums, traits, modules) so the model always
//!    receives *complete* units of context rather than broken lines.
//! 2. **Merkle Tree Change Detection** ([`merkle`]) — a cryptographic hash
//!    tree over all chunks lets us detect exactly which chunks changed between
//!    scans and re-index only those (incremental syncing).
//! 3. **Local Trigram Indexing** ([`trigram`]) — a character n-gram (trigram)
//!    inverted index powers fast lexical/keyword search that combines with
//!    semantic vector search in a hybrid retrieval strategy.
//! 4. **Vector DB Sync & Obfuscation** ([`vector`]) — chunks are embedded,
//!    obfuscated (file names hashed, bodies encrypted), and synced to a remote
//!    vector store (Turbopuffer-style interface) without ever persisting raw
//!    source.

pub mod chunker;
pub mod merkle;
pub mod trigram;
pub mod vector;

pub use chunker::{AstChunk, AstChunker, ChunkKind};
pub use merkle::{DiffResult, MerkleTree};
pub use trigram::{TrigramHit, TrigramIndex};
pub use vector::{MemoryVectorStore, ObfuscatedChunk, Obfuscator, SyncReport, VectorStore, VectorSync};

use std::path::Path;

use rand::RngCore as _;

/// A fully assembled AST index for a workspace.
pub struct IndexBundle {
    /// All chunks extracted from the workspace.
    pub chunks: Vec<AstChunk>,
    /// Merkle tree over the chunks (project fingerprint + change detection).
    pub merkle: MerkleTree,
    /// Local trigram index for lexical/keyword search.
    pub trigram: TrigramIndex,
    /// Vector sync orchestrator (embed + obfuscate + upsert).
    pub sync: VectorSync<MemoryVectorStore>,
    /// Report from the initial sync.
    pub report: SyncReport,
}

/// Build the full AST index for a workspace directory:
/// chunk → merkle → trigram → vector sync.
///
/// This is the end-to-end pipeline Cursor runs on first index. Subsequent
/// syncs should call [`incremental_sync`] with a previously-built
/// [`IndexBundle`] to only re-index changed chunks.
pub fn index_workspace(dir: &Path) -> anyhow::Result<IndexBundle> {
    let mut chunker = AstChunker::new()?;
    let chunks = chunker.chunk_dir(dir)?;

    let merkle = MerkleTree::build(&chunks);

    let mut trigram = TrigramIndex::new(3);
    trigram.index(&chunks);

    // Per-process random key. Production callers that need persistence can
    // construct `VectorSync` directly with a key from their secret store; a
    // hard-coded all-zero key must never be the default.
    let mut key = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut key);
    let store = MemoryVectorStore::default();
    let mut sync = VectorSync::new(&key, store)?;
    let report = sync.sync(&chunks)?;

    Ok(IndexBundle {
        chunks,
        merkle,
        trigram,
        sync,
        report,
    })
}

/// Re-sync an existing index after the workspace changed.
///
/// Re-chunks the directory, diffs the new Merkle tree against the previous
/// one, and pushes only the `changed` chunks (plus purges the `deleted`
/// ones). This is the incremental path Cursor walks every ~10 minutes.
pub fn incremental_sync(
    bundle: &mut IndexBundle,
    dir: &Path,
) -> anyhow::Result<DiffResult> {
    let mut chunker = AstChunker::new()?;
    let new_chunks = chunker.chunk_dir(dir)?;

    let new_merkle = MerkleTree::build(&new_chunks);
    let diff = new_merkle.diff(&bundle.merkle);

    // Update the trigram index for changed/deleted chunks.
    for id in &diff.deleted {
        bundle.trigram.remove(id);
    }
    for chunk in &new_chunks {
        if diff.changed.contains(&chunk.id) {
            bundle.trigram.add(chunk.clone());
        }
    }

    // Re-sync only the changed chunks to the vector store.
    let changed: Vec<AstChunk> = new_chunks
        .iter()
        .filter(|c| diff.changed.contains(&c.id))
        .cloned()
        .collect();
    bundle.sync.sync(&changed)?;
    bundle.sync.delete(&diff.deleted)?;

    // Adopt the new tree + chunk set without parsing the workspace twice.
    bundle.merkle = new_merkle;
    bundle.chunks = new_chunks;

    Ok(diff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_workspace_self_contained() {
        // Index this very crate to prove the pipeline runs end-to-end.
        let bundle = index_workspace(Path::new(".")).expect("index builds");
        assert!(!bundle.chunks.is_empty(), "should extract chunks");
        assert!(bundle.merkle.root_hash().is_some());
        assert!(!bundle.trigram.is_empty());
        // First sync uploads everything, skips nothing.
        assert_eq!(bundle.report.skipped, 0);
        assert_eq!(bundle.report.upserted, bundle.chunks.len());
    }
}
