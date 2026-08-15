//! Merkle Tree Change Detection (Incremental Syncing)
//!
//! This is Cursor's second indexing pillar. A binary hash tree is built over
//! all chunk content hashes. The root hash is the project fingerprint; by
//! comparing per-chunk leaf hashes between scans we can pinpoint *exactly*
//! which chunks changed and re-index only those — no full re-embedding of the
//! codebase on every save.
//!
//! A periodic walk (Cursor does it ~every 10 minutes) recomputes leaf hashes
//! and diffs them against the previous tree. Only the `changed` set is
//! re-embedded/upserted; the `deleted` set is purged from the vector store.

use std::collections::HashMap;

use sha2::{Digest, Sha256};

use crate::ast::chunker::AstChunk;

/// A node in the Merkle tree.
#[derive(Debug, Clone)]
pub struct MerkleNode {
    /// SHA-256 hash of this node's content.
    pub hash: String,
    pub left: Option<Box<MerkleNode>>,
    pub right: Option<Box<MerkleNode>>,
    /// For leaves: the chunk id this leaf represents.
    pub chunk_id: Option<String>,
}

/// A Merkle tree built over a set of chunks.
#[derive(Debug, Clone)]
pub struct MerkleTree {
    root: Option<MerkleNode>,
    /// chunk id -> leaf hash (in the sorted order used to build the tree).
    pub leaf_hashes: HashMap<String, String>,
}

impl MerkleTree {
    /// Build a Merkle tree from chunks. Leaves are ordered by chunk id for
    /// deterministic construction (identical inputs → identical root hash).
    pub fn build(chunks: &[AstChunk]) -> Self {
        let mut leaves: Vec<(String, String)> = chunks
            .iter()
            .map(|c| (c.id.clone(), c.content_hash.clone()))
            .collect();
        leaves.sort_by(|a, b| a.0.cmp(&b.0));

        let leaf_hashes: HashMap<String, String> = leaves.iter().cloned().collect();

        let nodes: Vec<MerkleNode> = leaves
            .into_iter()
            .map(|(id, hash)| MerkleNode {
                hash,
                left: None,
                right: None,
                chunk_id: Some(id),
            })
            .collect();

        let root = build_recursive(nodes);
        Self { root, leaf_hashes }
    }

    /// The root hash — the project's content fingerprint.
    pub fn root_hash(&self) -> Option<String> {
        self.root.as_ref().map(|n| n.hash.clone())
    }

    /// Diff against a previous tree. Returns chunk ids that are new or whose
    /// content changed (`changed`) and ids that existed before but are now
    /// gone (`deleted`).
    pub fn diff(&self, previous: &MerkleTree) -> DiffResult {
        let mut changed = Vec::new();
        let mut deleted = Vec::new();

        for (id, hash) in &self.leaf_hashes {
            match previous.leaf_hashes.get(id) {
                Some(old) if old != hash => changed.push(id.clone()),
                None => changed.push(id.clone()),
                _ => {}
            }
        }
        for id in previous.leaf_hashes.keys() {
            if !self.leaf_hashes.contains_key(id) {
                deleted.push(id.clone());
            }
        }
        DiffResult { changed, deleted }
    }
}

/// Recursively pair leaves/internal nodes, hashing siblings together.
fn build_recursive(mut nodes: Vec<MerkleNode>) -> Option<MerkleNode> {
    if nodes.is_empty() {
        return None;
    }
    if nodes.len() == 1 {
        return Some(nodes.pop().unwrap());
    }

    let mut next_level = Vec::new();
    let mut i = 0;
    while i < nodes.len() {
        let left = nodes[i].clone();
        if i + 1 < nodes.len() {
            let right = nodes[i + 1].clone();
            let combined = format!("{}{}", left.hash, right.hash);
            let hash = hash_str(&combined);
            next_level.push(MerkleNode {
                hash,
                left: Some(Box::new(left)),
                right: Some(Box::new(right)),
                chunk_id: None,
            });
        } else {
            // Odd node out: promote unchanged (its hash already covers content).
            next_level.push(left);
        }
        i += 2;
    }
    build_recursive(next_level)
}

fn hash_str(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Result of diffing two Merkle trees.
#[derive(Debug, Clone, Default)]
pub struct DiffResult {
    /// Chunk ids that are new or whose content changed.
    pub changed: Vec<String>,
    /// Chunk ids that existed before but are gone now.
    pub deleted: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::chunker::{AstChunk, ChunkKind};

    fn fake_chunk(id: &str, content: &str) -> AstChunk {
        AstChunk {
            id: id.to_string(),
            file_path: std::path::PathBuf::from("x.rs"),
            name: id.to_string(),
            kind: ChunkKind::Function,
            source: content.to_string(),
            byte_range: (0, content.len()),
            line_range: (0, 1),
            content_hash: AstChunk::hash_source(content),
            parent: None,
        }
    }

    #[test]
    fn identical_inputs_yield_identical_root() {
        let a = [fake_chunk("f1", "aaa"), fake_chunk("f2", "bbb")];
        let b = [fake_chunk("f2", "bbb"), fake_chunk("f1", "aaa")];
        let t1 = MerkleTree::build(&a);
        let t2 = MerkleTree::build(&b);
        assert_eq!(t1.root_hash(), t2.root_hash());
    }

    #[test]
    fn diff_detects_changed_and_deleted() {
        let before = [fake_chunk("keep", "same"), fake_chunk("drop", "gone")];
        let after = [fake_chunk("keep", "same"), fake_chunk("new", "fresh")];
        let t_before = MerkleTree::build(&before);
        let t_after = MerkleTree::build(&after);

        let d = t_after.diff(&t_before);
        assert!(d.changed.contains(&"new".to_string()));
        assert!(!d.changed.contains(&"keep".to_string()));
        assert!(d.deleted.contains(&"drop".to_string()));
    }
}
