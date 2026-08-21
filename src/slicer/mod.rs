//! Attention-Sparsified "Skeleton" Layouts.
//!
//! When a file is pulled as a contextual dependency, keep the active target
//! function completely intact but strip the bodies of all other functions,
//! replacing them with a single comment placeholder. The model retains a
//! perfect mental map of the codebase's global types, parameters, and
//! signatures while ignoring dead code bodies it wasn't going to modify.

use tree_sitter::{Node as TsNode, Parser};

use crate::asg::AsgError;

/// Skeleton Tree Slicer: strips non-active function bodies from a source file.
pub struct SkeletonSlicer {
    parser: Parser,
}

impl SkeletonSlicer {
    pub fn new() -> Result<Self, AsgError> {
        let mut parser = Parser::new();
        let rust_lang: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
        parser
            .set_language(&rust_lang)
            .map_err(|e| AsgError::ParseError(e.to_string()))?;
        Ok(Self { parser })
    }

    /// Produce a skeleton of `source`. The function whose byte range overlaps
    /// `active_range` (if any) is kept fully intact; every other function body
    /// is replaced with `// [body hidden to save tokens]`.
    pub fn skeleton(&mut self, source: &str, active_range: Option<(usize, usize)>) -> String {
        let tree = match self.parser.parse(source, None) {
            Some(t) => t,
            None => return source.to_string(),
        };
        let mut hides: Vec<(usize, usize)> = Vec::new();
        collect_hidden_bodies(tree.root_node(), active_range, &mut hides);
        if hides.is_empty() {
            return source.to_string();
        }
        hides.sort_unstable();

        // Drop ranges nested inside an already-hidden (outer) body.
        let mut filtered: Vec<(usize, usize)> = Vec::new();
        for (start, end) in hides {
            if let Some(&(_, last_end)) = filtered.last() {
                if start < last_end {
                    continue;
                }
            }
            filtered.push((start, end));
        }

        let mut result = String::with_capacity(source.len());
        let mut pos = 0usize;
        for (start, end) in filtered {
            result.push_str(&source[pos..start]);
            // Preserve the indentation of the body's first line.
            result.push_str(&leading_whitespace(source, start));
            result.push_str("// [body hidden to save tokens]\n");
            pos = end;
        }
        result.push_str(&source[pos..]);
        result
    }

    /// Count how many bodies would be hidden.
    pub fn hidden_body_count(&mut self, source: &str, active_range: Option<(usize, usize)>) -> usize {
        let tree = match self.parser.parse(source, None) {
            Some(t) => t,
            None => return 0,
        };
        let mut hides: Vec<(usize, usize)> = Vec::new();
        collect_hidden_bodies(tree.root_node(), active_range, &mut hides);
        hides.len()
    }
}

fn collect_hidden_bodies(node: TsNode, active: Option<(usize, usize)>, hides: &mut Vec<(usize, usize)>) {
    let kind = node.kind();
    if kind == "function_item" || kind == "function_signature_item" {
        if let Some(body) = node.child_by_field_name("body") {
            let start = body.start_byte();
            let end = body.end_byte();
            let is_active = active
                .map(|(a, b)| start >= a && end <= b)
                .unwrap_or(false);
            if !is_active {
                hides.push((start, end));
            }
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_hidden_bodies(child, active, hides);
    }
}

fn leading_whitespace(source: &str, byte_offset: usize) -> String {
    let line_start = source[..byte_offset].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let line = &source[line_start..byte_offset];
    line.chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skeleton_hides_other_bodies_keeps_active() {
        let src = "fn a() {\n    let x = 1;\n}\n\nfn b() {\n    let y = 2;\n}\n";
        let mut slicer = SkeletonSlicer::new().unwrap();
        // Active range must cover the entire body of fn b (from '{' to '}').
        let fn_b_start = src.find("fn b").unwrap();
        let body_start = fn_b_start + src[fn_b_start..].find('{').unwrap();
        let body_end = fn_b_start + src[fn_b_start..].rfind('}').unwrap() + 1;
        let active = Some((body_start, body_end));
        let skeleton = slicer.skeleton(src, active);
        assert!(skeleton.contains("fn a()"));
        assert!(skeleton.contains("// [body hidden to save tokens]"));
        assert!(skeleton.contains("let y = 2;"));
        assert!(!skeleton.contains("let x = 1;"));
    }

    #[test]
    fn skeleton_keeps_all_when_no_functions() {
        let src = "// just a comment\n";
        let mut slicer = SkeletonSlicer::new().unwrap();
        assert_eq!(slicer.skeleton(src, None), src);
    }
}