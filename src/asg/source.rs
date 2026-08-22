//! Zero-copy source arena.
//!
//! Owns the raw text of every parsed file at **stable heap addresses** so that
//! `AsgGraph<'a>` bodies can borrow (`Cow::Borrowed`) from it for the arena's
//! lifetime — the parsing phase never has to copy a body into the graph.
//!
//! # Why `Box<str>`?
//!
//! If the arena stored plain `String`s in a `HashMap`, rehashing/reallocation
//! would move the buffers and silently invalidate every borrowed body slice —
//! a dangling-pointer hazard the borrow checker cannot see (self-referential
//! struct). Wrapping each file in a `Box<str>` pins the `str` contents at a
//! fixed heap address: moving the `Box` (a fat pointer) moves only the pointer,
//! never the data. Slices handed out by [`SourceSet::get`] therefore remain
//! valid for as long as the arena lives.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Arena owning all source buffers for the ASG.
#[derive(Default)]
pub struct SourceSet {
    /// `PathBuf -> Box<str>`: stable-address payloads.
    files: HashMap<PathBuf, Box<str>>,
    /// Insertion-ordered paths for deterministic iteration.
    order: Vec<PathBuf>,
}

impl SourceSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) a file's text.
    ///
    /// The stored `Box<str>` lives on the heap and never moves, so slices from
    /// [`SourceSet::get`] stay valid regardless of later insertions (which may
    /// rehash the index).
    pub fn insert(&mut self, path: PathBuf, text: String) {
        if !self.files.contains_key(&path) {
            self.order.push(path.clone());
        }
        self.files.insert(path, text.into_boxed_str());
    }

    /// Borrow a file's contents. `None` if the path is not present.
    ///
    /// The returned `&str` is valid for `'a` tied to `&'a self` — hand it to
    /// `AsgGraph::add_node` as `Cow::Borrowed(..)` to keep bodies zero-copy.
    pub fn get(&self, path: &Path) -> Option<&str> {
        self.files.get(path).map(|b| b.as_ref())
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Deterministic (insertion-ordered) iteration over stored paths.
    pub fn paths(&self) -> impl Iterator<Item = &PathBuf> {
        self.order.iter()
    }

    /// Recursively load every file with the given extension under `dir`.
    pub fn from_dir(dir: &Path, ext: &str) -> std::io::Result<Self> {
        let mut set = Self::new();
        let walker = walkdir::WalkDir::new(dir).into_iter().filter_entry(|entry| {
            if !entry.file_type().is_dir() {
                return true;
            }
            !matches!(
                entry.file_name().to_string_lossy().as_ref(),
                ".git" | "target" | "node_modules" | ".cache" | ".idea" | ".vscode"
            )
        });
        for entry in walker.filter_map(Result::ok) {
            let path = entry.path();
            if entry.file_type().is_file() && path.extension().is_some_and(|e| e == ext) {
                let text = std::fs::read_to_string(path)?;
                set.insert(path.to_path_buf(), text);
            }
        }
        Ok(set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_roundtrip() {
        let mut sources = SourceSet::new();
        sources.insert(PathBuf::from("a.rs"), "pub fn a() {}".to_string());
        assert_eq!(sources.get(Path::new("a.rs")), Some("pub fn a() {}"));
        assert_eq!(sources.len(), 1);
        assert!(sources.get(Path::new("missing.rs")).is_none());
    }

    #[test]
    fn boxed_str_payloads_do_not_move() {
        let mut sources = SourceSet::new();
        sources.insert(PathBuf::from("a.rs"), "pub fn a() {}".to_string());
        let addr_before = sources.get(Path::new("a.rs")).unwrap().as_ptr() as usize;

        // Insert enough files to force the HashMap index to rehash repeatedly.
        for i in 0..2048 {
            sources.insert(PathBuf::from(format!("f{i}.rs")), format!("pub fn f{i}() {{}}"));
        }

        // The Box<str> payload is heap-pinned: rehashing moves only the pointer
        // slots, never the str contents. (A plain `String` in the map would
        // reallocate and change this address.)
        let addr_after = sources.get(Path::new("a.rs")).unwrap().as_ptr() as usize;
        assert_eq!(addr_before, addr_after);
        assert_eq!(sources.len(), 2049);
        assert_eq!(sources.paths().next().unwrap(), Path::new("a.rs"));
    }

    /// The supported pattern: populate the arena fully, *then* hand borrowed
    /// slices to a graph. Bodies stay zero-copy and valid for the arena's
    /// lifetime.
    #[test]
    fn borrow_slices_after_bulk_insert() {
        let mut sources = SourceSet::new();
        for i in 0..2048 {
            sources.insert(PathBuf::from(format!("f{i}.rs")), format!("pub fn f{i}() {{}}"));
        }

        // Rehash has long since happened; borrows are still coherent.
        let first = sources.get(Path::new("f0.rs")).unwrap();
        assert_eq!(first, "pub fn f0() {}");
        assert_eq!(sources.get(Path::new("f2047.rs")), Some("pub fn f2047() {}"));
    }

    #[test]
    fn insert_replaces_existing() {
        let mut sources = SourceSet::new();
        sources.insert(PathBuf::from("a.rs"), "v1".to_string());
        sources.insert(PathBuf::from("a.rs"), "v2".to_string());
        assert_eq!(sources.len(), 1);
        assert_eq!(sources.get(Path::new("a.rs")), Some("v2"));
    }
}
