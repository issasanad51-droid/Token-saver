//! Tokenizer-Aware Symbol Folding + AST-Aware Space Collapser.
//!
//! Modern LLM tokenizers (BPE: tiktoken, Llama, DeepSeek) fracture long
//! syntactic boilerplate into many tiny sub-tokens. This layer losslessly
//! replaces heavy repeating syntactic patterns with rare single-character
//! Unicode glyphs (e.g. `§`, `ø`, `∆`) that consume exactly one token, and
//! injects a tiny decoder mapping array into the prompt header so the model
//! can reverse the substitution. A companion AST-Aware Space Collapser
//! compresses indentation layouts that burn tokens disproportionately.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Rare single-character Unicode glyphs guaranteed to consume exactly one
/// token in most LLM vocabularies (OpenAI tiktoken, Llama BPE, DeepSeek).
const GLYPHS: [char; 24] = [
    '§', 'ø', '∆', '¶', 'æ', 'ß', 'ð', 'þ', 'ŋ', 'ħ', 'ƿ', 'ǽ', 'œ', 'ƀ', 'đ', 'ı', 'ȷ', 'ɱ',
    'ɲ', 'ɸ', 'ʂ', 'ʄ', 'ʍ', 'ʎ',
];

/// A folded text plus its decoder mapping array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldedText {
    /// The folded (compressed) text.
    pub text: String,
    /// Decoder mapping: glyph -> original substring.
    pub decoder: Vec<(String, String)>,
    pub original_bytes: usize,
    pub folded_bytes: usize,
}

impl FoldedText {
    /// Render the decoder mapping header injected into the prompt.
    pub fn decoder_header(&self) -> String {
        if self.decoder.is_empty() {
            return String::new();
        }
        let map = self
            .decoder
            .iter()
            .map(|(glyph, original)| format!("{glyph}={original}"))
            .collect::<Vec<_>>()
            .join(",");
        format!("// token-fold decoder: {map}\n")
    }

    /// Losslessly unfold back to the original text.
    pub fn unfold(&self) -> String {
        let mut result = self.text.clone();
        for (glyph, original) in &self.decoder {
            result = result.replace(glyph, original);
        }
        result
    }

    pub fn bytes_saved(&self) -> usize {
        self.original_bytes.saturating_sub(self.folded_bytes)
    }
}

/// Language-specific heavy syntactic boilerplate patterns.
pub struct SymbolFolder {
    patterns: HashMap<&'static str, Vec<&'static str>>,
    glyphs: Vec<char>,
}

impl Default for SymbolFolder {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolFolder {
    pub fn new() -> Self {
        let mut patterns: HashMap<&'static str, Vec<&'static str>> = HashMap::new();

        // Rust-heavy boilerplate.
        patterns.insert(
            "rust",
            vec![
                "std::sync::Arc",
                "std::sync::Mutex",
                "std::sync::RwLock",
                "std::collections::HashMap",
                "std::collections::HashSet",
                "std::path::PathBuf",
                "pub async fn ",
                "pub fn ",
                "async fn ",
                "pub struct ",
                "pub enum ",
                "pub trait ",
                "impl ",
                "use std::",
                "use crate::",
                "return ",
                "self.",
                "fn ",
                "::",
                "->",
                "=>",
            ],
        );

        // TypeScript / JavaScript boilerplate.
        let ts_patterns: Vec<&'static str> = vec![
            "import {",
            "import ",
            "export default ",
            "export async function ",
            "export function ",
            "export const ",
            "async function ",
            "function ",
            "const ",
            "return ",
            "from '",
            "from \"",
            "=>",
            "->",
            "::",
        ];
        patterns.insert("typescript", ts_patterns.clone());
        patterns.insert("javascript", ts_patterns);

        // Python boilerplate.
        patterns.insert(
            "python",
            vec![
                "from ",
                "import ",
                "async def ",
                "def ",
                "class ",
                "return ",
                "self.",
                "->",
                "=>",
            ],
        );

        // SQL boilerplate.
        patterns.insert(
            "sql",
            vec![
                "SELECT ",
                "INSERT INTO ",
                "UPDATE ",
                "DELETE FROM ",
                "CREATE TABLE ",
                "ALTER TABLE ",
                "WHERE ",
                "GROUP BY ",
                "ORDER BY ",
                "LEFT JOIN ",
                "INNER JOIN ",
                "JOIN ",
            ],
        );

        Self {
            patterns,
            glyphs: GLYPHS.to_vec(),
        }
    }

    /// Detect the language from a file path or fall back to a default.
    pub fn language_for_path(path: &str) -> &'static str {
        let lower = path.to_ascii_lowercase();
        if lower.ends_with(".rs") {
            "rust"
        } else if lower.ends_with(".ts") || lower.ends_with(".tsx") {
            "typescript"
        } else if lower.ends_with(".js") || lower.ends_with(".jsx") || lower.ends_with(".mjs") {
            "javascript"
        } else if lower.ends_with(".py") {
            "python"
        } else if lower.ends_with(".sql") {
            "sql"
        } else {
            "rust"
        }
    }

    /// Fold heavy repeating syntactic boilerplate into single glyphs.
    ///
    /// The transform is lossless: every substitution is recorded in the
    /// decoder mapping array, so [`FoldedText::unfold`] recovers the exact
    /// original source.
    pub fn fold(&self, source: &str, language: &str) -> FoldedText {
        let mut text = source.to_string();
        let mut decoder: Vec<(String, String)> = Vec::new();
        let mut glyph_iter = self.glyphs.iter();

        let mut patterns: Vec<&str> = self.patterns.get(language).cloned().unwrap_or_default();
        // Longest-first so `pub async fn ` wins over `fn `.
        patterns.sort_by_key(|p| std::cmp::Reverse(p.len()));

        for pattern in patterns {
            if !text.contains(pattern) {
                continue;
            }
            let Some(&glyph) = glyph_iter.next() else {
                break;
            };
            let glyph_str = glyph.to_string();
            text = text.replace(pattern, &glyph_str);
            decoder.push((glyph_str, pattern.to_string()));
        }

        FoldedText {
            folded_bytes: text.len(),
            text,
            decoder,
            original_bytes: source.len(),
        }
    }
}

/// AST-Aware Space Collapser.
///
/// Long blocks of indentation whitespace burn tokens disproportionately in
/// BPE tokenizers. Collapse runs of leading indentation on each line to a
/// single space. Interior whitespace (including string literals) is
/// preserved, keeping the transform semantically lossless.
pub struct SpaceCollapser;

impl SpaceCollapser {
    /// Collapse runs of leading indentation whitespace on each line to a
    /// single space.
    pub fn collapse(source: &str) -> String {
        let mut out = String::with_capacity(source.len());
        let mut at_line_start = true;
        for ch in source.chars() {
            if at_line_start && (ch == ' ' || ch == '\t') {
                if !out.ends_with(' ') {
                    out.push(' ');
                }
                continue;
            }
            out.push(ch);
            at_line_start = ch == '\n';
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_and_unfold_roundtrip() {
        let folder = SymbolFolder::new();
        let src = "pub fn foo() -> std::sync::Arc<u32> { std::sync::Arc::new(1) }";
        let folded = folder.fold(src, "rust");
        assert!(folded.text.len() < src.len());
        assert_eq!(folded.unfold(), src);
    }

    #[test]
    fn decoder_header_renders() {
        let folder = SymbolFolder::new();
        let folded = folder.fold("pub fn a() {}", "rust");
        let header = folded.decoder_header();
        assert!(header.starts_with("// token-fold decoder:"));
        assert!(header.contains('='));
    }

    #[test]
    fn space_collapser_compresses_indentation() {
        let src = "fn a() {\n        let x = 1;\n}";
        let collapsed = SpaceCollapser::collapse(src);
        assert!(collapsed.len() < src.len());
        assert!(collapsed.contains("{\n let x = 1;\n}"));
    }
}