//! Persistent storage using redb (ACID key-value store).
//!
//! Saves and loads the ASG, embeddings, and chunk registry so the server
//! doesn't need a full rebuild on every restart.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Context as _;
use tracing::{debug, info};

use redb::{Database, TableDefinition};

const DATA_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("data");

const ASG_KEY: &str = "asg";
const EMBEDDINGS_KEY: &str = "embeddings";
#[allow(dead_code)]
const CHUNKS_KEY: &str = "chunks";
const MEMORIES_KEY: &str = "memories";

/// Persistent store backed by redb.
pub struct PersistentStore {
    db: Database,
}

impl PersistentStore {
    /// Open (or create) a persistent store at the given path.
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let db = Database::create(path)?;
        info!("persistent store opened at {}", path.display());
        Ok(Self { db })
    }

    /// Save serialized data under a key.
    pub fn save(&self, key: &str, data: &[u8]) -> anyhow::Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table: redb::Table<&str, &[u8]> = write_txn.open_table(DATA_TABLE)?;
            table.insert(key, data)?;
        }
        write_txn.commit()?;
        debug!("saved {} bytes under key '{}'", data.len(), key);
        Ok(())
    }

    /// Load serialized data for a key.
    pub fn load(&self, key: &str) -> anyhow::Result<Option<Vec<u8>>> {
        let read_txn = self.db.begin_read()?;
        let table: redb::ReadOnlyTable<&str, &[u8]> = read_txn.open_table(DATA_TABLE)?;
        Ok(table.get(key)?.map(|v| v.value().to_vec()))
    }

    /// Check if a key exists.
    pub fn exists(&self, key: &str) -> bool {
        self.load(key).ok().flatten().is_some()
    }

    /// Save the ASG.
    pub fn save_asg(&self, asg: &crate::asg::Asg) -> anyhow::Result<()> {
        let data =
            serde_json::to_vec(asg).with_context(|| "failed to serialize ASG")?;
        self.save(ASG_KEY, &data)
    }

    /// Load the ASG.
    pub fn load_asg(&self) -> anyhow::Result<Option<crate::asg::Asg>> {
        match self.load(ASG_KEY)? {
            Some(data) => Ok(Some(
                serde_json::from_slice(&data)
                    .with_context(|| "failed to deserialize ASG")?,
            )),
            None => Ok(None),
        }
    }

    /// Save embeddings.
    pub fn save_embeddings(
        &self,
        embeddings: &HashMap<usize, Vec<f64>>,
    ) -> anyhow::Result<()> {
        let data = serde_json::to_vec(embeddings)
            .with_context(|| "failed to serialize embeddings")?;
        self.save(EMBEDDINGS_KEY, &data)
    }

    /// Load embeddings.
    pub fn load_embeddings(&self) -> anyhow::Result<Option<HashMap<usize, Vec<f64>>>> {
        match self.load(EMBEDDINGS_KEY)? {
            Some(data) => Ok(Some(
                serde_json::from_slice(&data)
                    .with_context(|| "failed to deserialize embeddings")?,
            )),
            None => Ok(None),
        }
    }

    /// Save all memories.
    pub fn save_memories(&self, memories: &crate::memory::MemoryStore) -> anyhow::Result<()> {
        let all_memories = memories.list(None);
        let data = serde_json::to_vec(&all_memories)
            .with_context(|| "failed to serialize memories")?;
        self.save(MEMORIES_KEY, &data)
    }

    /// Load memories into a MemoryStore.
    pub fn load_memories(&self) -> anyhow::Result<crate::memory::MemoryStore> {
        let store = crate::memory::MemoryStore::new();
        match self.load(MEMORIES_KEY)? {
            Some(data) => {
                let memories: Vec<crate::memory::Memory> = serde_json::from_slice(&data)
                    .with_context(|| "failed to deserialize memories")?;
                for mem in memories {
                    // Re-insert by saving the content, but we need to preserve the
                    // original id and metadata. Use the internal insert approach.
                    store.save(&mem.content, mem.namespace.clone());
                }
            }
            None => {}
        }
        Ok(store)
    }
}
