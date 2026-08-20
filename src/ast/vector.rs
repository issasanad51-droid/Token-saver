//! Continuous Vector DB Syncing & Obfuscation
//!
//! This is Cursor's fourth indexing pillar. Once local chunks are ready, they
//! are embedded and synced to a remote vector store (a Turbopuffer-style
//! interface here). To protect privacy, an obfuscation layer runs locally:
//! file names are hashed into opaque aliases and chunk bodies are encrypted
//! with AES-GCM before they ever leave the machine — raw source is never
//! persisted on the server.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use rand::RngCore;
use sha2::{Digest, Sha256};

use crate::ast::chunker::AstChunk;

// ---------------------------------------------------------------------------
// Obfuscator
// ---------------------------------------------------------------------------

/// Encrypts chunk bodies and hashes file paths before they leave the machine.
pub struct Obfuscator {
    cipher: Aes256Gcm,
    /// Secret salt prevents dictionary attacks against predictable file paths.
    alias_salt: [u8; 32],
    /// Cache mapping real file paths to stable opaque aliases.
    file_alias: HashMap<PathBuf, String>,
}

impl Obfuscator {
    /// Create an obfuscator from a 32-byte key.
    pub fn new(key: &[u8; 32]) -> anyhow::Result<Self> {
        let cipher = Aes256Gcm::new(key.into());
        Ok(Self {
            cipher,
            alias_salt: *key,
            file_alias: HashMap::new(),
        })
    }

    /// Deterministically hash a file path into an opaque alias (no real path
    /// ever reaches the cloud).
    pub fn alias_for(&mut self, path: &Path) -> String {
        if let Some(alias) = self.file_alias.get(path) {
            return alias.clone();
        }
        let alias = self.keyed_alias("f", path.to_string_lossy().as_bytes());
        self.file_alias.insert(path.to_path_buf(), alias.clone());
        alias
    }

    /// Hide the local chunk tracker as well as the file path. AST chunk IDs
    /// contain relative paths, so uploading them verbatim would undo the file
    /// alias protection.
    pub fn chunk_alias_for(&self, chunk_id: &str) -> String {
        self.keyed_alias("c", chunk_id.as_bytes())
    }

    fn keyed_alias(&self, prefix: &str, value: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.alias_salt);
        hasher.update(value);
        format!("{prefix}_{:x}", hasher.finalize())
    }

    /// Encrypt a chunk's source. Returns `(nonce_hex, ciphertext_hex)`.
    pub fn encrypt(&self, plaintext: &str) -> anyhow::Result<(String, String)> {
        let mut nonce_bytes = [0u8; 12];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        let ciphertext = self
            .cipher
            .encrypt(nonce, plaintext.as_bytes())
            .map_err(|e| anyhow::anyhow!("encryption failed: {e}"))?;
        Ok((hex_encode(&nonce_bytes), hex_encode(&ciphertext)))
    }

    /// Decrypt back to plaintext (used locally for hydration).
    pub fn decrypt(&self, nonce_hex: &str, ciphertext_hex: &str) -> anyhow::Result<String> {
        let nonce_bytes = hex_decode(nonce_hex)?;
        let ciphertext = hex_decode(ciphertext_hex)?;
        let nonce = Nonce::from_slice(&nonce_bytes);
        let plaintext = self
            .cipher
            .decrypt(nonce, ciphertext.as_ref())
            .map_err(|e| anyhow::anyhow!("decryption failed: {e}"))?;
        String::from_utf8(plaintext).map_err(|e| anyhow::anyhow!("utf8 error: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Obfuscated chunk (the remote representation)
// ---------------------------------------------------------------------------

/// A chunk as stored in the remote vector DB: no raw source, no real path.
#[derive(Debug, Clone)]
pub struct ObfuscatedChunk {
    /// Opaque chunk id (still useful for the local→remote mapping).
    pub chunk_id: String,
    /// Hashed file alias (never the real path).
    pub file_alias: String,
    /// Encrypted body (hex).
    pub encrypted_body: String,
    /// Nonce used for encryption (hex).
    pub nonce: String,
    /// Embedding vector (mock model for now).
    pub embedding: Vec<f64>,
    /// Content hash for dedup / versioning.
    pub content_hash: String,
}

// ---------------------------------------------------------------------------
// Vector store interface (Turbopuffer-style)
// ---------------------------------------------------------------------------

/// A remote vector store client.
pub trait VectorStore {
    /// Upsert (insert or update) obfuscated chunks.
    fn upsert(&mut self, chunks: &[ObfuscatedChunk]) -> anyhow::Result<()>;
    /// Delete chunks by id.
    fn delete(&mut self, chunk_ids: &[String]) -> anyhow::Result<()>;
}

/// In-memory stand-in for a cloud vector DB (swap for Turbopuffer later).
#[derive(Default)]
pub struct MemoryVectorStore {
    pub data: HashMap<String, ObfuscatedChunk>,
}

impl VectorStore for MemoryVectorStore {
    fn upsert(&mut self, chunks: &[ObfuscatedChunk]) -> anyhow::Result<()> {
        for c in chunks {
            self.data.insert(c.chunk_id.clone(), c.clone());
        }
        Ok(())
    }
    fn delete(&mut self, chunk_ids: &[String]) -> anyhow::Result<()> {
        for id in chunk_ids {
            self.data.remove(id);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Sync orchestrator
// ---------------------------------------------------------------------------

/// Orchestrates embedding + obfuscation + sync, applying incremental updates
/// driven by the Merkle diff.
pub struct VectorSync<S: VectorStore> {
    obfuscator: Obfuscator,
    store: S,
    /// Local cache of last-synced content hashes (for incremental upsert).
    pub synced: HashMap<String, String>,
    /// Local chunk tracker -> keyed remote alias.
    remote_ids: HashMap<String, String>,
}

impl<S: VectorStore> VectorSync<S> {
    /// Create a sync orchestrator over `store` with a 32-byte key.
    pub fn new(key: &[u8; 32], store: S) -> anyhow::Result<Self> {
        Ok(Self {
            obfuscator: Obfuscator::new(key)?,
            store,
            synced: HashMap::new(),
            remote_ids: HashMap::new(),
        })
    }

    /// Mock embedding: deterministic hash-based vector (replace with a real
    /// embedding model in production).
    pub fn embed(&self, chunk: &AstChunk) -> Vec<f64> {
        let dim = 64;
        let mut vec = vec![0.0f64; dim];
        let combined = format!("{}{}", chunk.name, chunk.kind.as_str());
        for (i, ch) in combined.chars().enumerate() {
            let idx = (i * 7 + ch as usize) % dim;
            vec[idx] += (ch as u32 as f64) / 256.0;
        }
        let norm: f64 = vec.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }
        vec
    }

    /// Sync a set of chunks, uploading only those whose hash changed since the
    /// last sync (incremental, Merkle-driven).
    pub fn sync(&mut self, chunks: &[AstChunk]) -> anyhow::Result<SyncReport> {
        let mut upserted = Vec::new();
        let mut skipped = 0;
        for chunk in chunks {
            let prev = self.synced.get(&chunk.id);
            if prev == Some(&chunk.content_hash) {
                skipped += 1;
                continue;
            }
            let alias = self.obfuscator.alias_for(&chunk.file_path);
            let remote_id = self.obfuscator.chunk_alias_for(&chunk.id);
            let (nonce, ct) = self.obfuscator.encrypt(&chunk.source)?;
            let ob = ObfuscatedChunk {
                chunk_id: remote_id.clone(),
                file_alias: alias,
                encrypted_body: ct,
                nonce,
                embedding: self.embed(chunk),
                content_hash: chunk.content_hash.clone(),
            };
            upserted.push(ob);
            self.synced
                .insert(chunk.id.clone(), chunk.content_hash.clone());
            self.remote_ids.insert(chunk.id.clone(), remote_id);
        }
        self.store.upsert(&upserted)?;
        Ok(SyncReport {
            upserted: upserted.len(),
            skipped,
        })
    }

    /// Delete chunks that were removed from the codebase.
    pub fn delete(&mut self, chunk_ids: &[String]) -> anyhow::Result<()> {
        let mut remote_ids = Vec::with_capacity(chunk_ids.len());
        for id in chunk_ids {
            self.synced.remove(id);
            let remote_id = self
                .remote_ids
                .remove(id)
                .unwrap_or_else(|| self.obfuscator.chunk_alias_for(id));
            remote_ids.push(remote_id);
        }
        self.store.delete(&remote_ids)
    }

    /// Decrypt a stored chunk's body locally (hydration helper).
    pub fn decrypt_chunk(&self, chunk: &ObfuscatedChunk) -> anyhow::Result<String> {
        self.obfuscator.decrypt(&chunk.nonce, &chunk.encrypted_body)
    }
}

/// Report of a sync operation.
#[derive(Debug, Clone, Default)]
pub struct SyncReport {
    /// Number of chunks actually uploaded.
    pub upserted: usize,
    /// Number of chunks skipped because unchanged.
    pub skipped: usize,
}

// ---------------------------------------------------------------------------
// Hex helpers (avoid an extra dependency)
// ---------------------------------------------------------------------------

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hex_decode(s: &str) -> anyhow::Result<Vec<u8>> {
    if s.len() % 2 != 0 {
        return Err(anyhow::anyhow!("invalid hex length"));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| anyhow::anyhow!("hex decode: {e}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::chunker::{AstChunk, ChunkKind};

    fn chunk(id: &str, src: &str) -> AstChunk {
        AstChunk {
            id: id.to_string(),
            file_path: PathBuf::from("src/lib.rs"),
            name: id.to_string(),
            kind: ChunkKind::Function,
            source: src.to_string(),
            byte_range: (0, src.len()),
            line_range: (0, 1),
            content_hash: AstChunk::hash_source(src),
            parent: None,
        }
    }

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let obf = Obfuscator::new(&[7u8; 32]).unwrap();
        let (nonce, ct) = obf.encrypt("secret source code").unwrap();
        let plain = obf.decrypt(&nonce, &ct).unwrap();
        assert_eq!(plain, "secret source code");
    }

    #[test]
    fn file_alias_is_opaque_and_stable() {
        let mut obf = Obfuscator::new(&[1u8; 32]).unwrap();
        let a = obf.alias_for(Path::new("src/server/mod.rs"));
        let b = obf.alias_for(Path::new("src/server/mod.rs"));
        assert_eq!(a, b);
        assert!(!a.contains("server"));
    }

    #[test]
    fn incremental_sync_skips_unchanged() {
        let key = [3u8; 32];
        let store = MemoryVectorStore::default();
        let mut sync = VectorSync::new(&key, store).unwrap();

        let c = chunk("fn_a", "pub fn a() {}");
        let r1 = sync.sync(&[c.clone()]).unwrap();
        assert_eq!(r1.upserted, 1);
        let remote_id = sync.store.data.keys().next().unwrap();
        assert!(remote_id.starts_with("c_"));
        assert!(!remote_id.contains("fn_a"));

        // Re-sync identical content → skipped.
        let r2 = sync.sync(&[c.clone()]).unwrap();
        assert_eq!(r2.upserted, 0);
        assert_eq!(r2.skipped, 1);

        // Delete it.
        sync.delete(&[c.id.clone()]).unwrap();
        assert!(sync.synced.is_empty());
    }
}
