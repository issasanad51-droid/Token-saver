//! Custom RRF fusion: weighted + score-aware Reciprocal Rank Fusion.
//!
//! Stock RRF is `score(d) = Σ 1/(k + rank_d)` — it treats every retrieval
//! stream equally and discards raw scores entirely. Two customizations lift
//! retrieval quality:
//!
//! 1. **Per-stream weight.** `w_s / (k + rank)` lets the caller trust the
//!    semantic vector stream more than the lexical BM25 stream (e.g.
//!    `weights = [1.2, 1.0, 0.9]`). With `weights = [1, 1, 1]` and
//!    `score_alpha = 0` this reduces to the exact spec formula.
//! 2. **Score-aware blending.** Raw scores (min-max normalized per stream) are
//!    folded in via `score_alpha`. Two documents at adjacent ranks but with
//!    wildly different raw scores are now distinguished — this is a well-known
//!    RRF weakness that score blending repairs.
//!
//! Generic over document ids so it works with both the legacy `usize` pipeline
//! and the v2 `NodeId` tracker.

use std::collections::HashMap;
use std::hash::Hash;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A document as ranked by a single retrieval stream (sorted best-first).
#[derive(Debug, Clone)]
pub struct RankedDoc<Id> {
    pub id: Id,
    pub score: f64,
}

impl<Id> RankedDoc<Id> {
    pub fn new(id: Id, score: f64) -> Self {
        Self { id, score }
    }
}

/// A fused result with provenance (per-stream RRF + raw scores).
#[derive(Debug, Clone)]
pub struct FusedDoc<Id> {
    pub id: Id,
    pub fused_score: f64,
    /// RRF contribution from each stream (0.0 if absent from that stream).
    pub rrf_contributions: Vec<f64>,
    /// Original raw score per stream.
    pub raw_scores: Vec<Option<f64>>,
}

/// Fusion parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RrfConfig {
    /// Rank offset constant (spec default: 60).
    pub k: f64,
    /// Per-stream multipliers, e.g. `[semantic, structural, lexical]`.
    pub weights: Vec<f64>,
    /// Blend factor for normalized raw scores (0.0 = pure RRF).
    pub score_alpha: f64,
}

impl Default for RrfConfig {
    fn default() -> Self {
        Self {
            k: 60.0,
            weights: vec![1.0, 1.0, 1.0],
            score_alpha: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Fusion
// ---------------------------------------------------------------------------

/// Fuse `streams` (each sorted best-first) into a single ranked list.
///
/// `config.weights` is length-matched to `streams`; a length mismatch falls
/// back to uniform weights.
pub fn fuse<Id: Clone + Eq + Hash>(
    streams: &[Vec<RankedDoc<Id>>],
    config: &RrfConfig,
) -> Vec<FusedDoc<Id>> {
    let n_streams = streams.len();
    if n_streams == 0 {
        return Vec::new();
    }

    let ws: Vec<f64> = if config.weights.len() == n_streams {
        config.weights.clone()
    } else {
        vec![1.0; n_streams]
    };

    // Per-stream min/max for normalization.
    let ranges: Vec<(f64, f64)> = streams
        .iter()
        .map(|s| {
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            for d in s {
                min = min.min(d.score);
                max = max.max(d.score);
            }
            (min, max)
        })
        .collect();

    let mut acc: HashMap<Id, (f64, Vec<f64>, Vec<Option<f64>>)> = HashMap::new();

    for (si, stream) in streams.iter().enumerate() {
        let w = ws[si];
        let (min, max) = ranges[si];
        let span = max - min;

        for (rank, doc) in stream.iter().enumerate() {
            let entry = acc
                .entry(doc.id.clone())
                .or_insert_with(|| (0.0, vec![0.0; n_streams], vec![None; n_streams]));

            // If the doc appears twice in one stream, keep the best (first) rank.
            if entry.1[si] != 0.0 {
                continue;
            }

            let rrf = w / (config.k + rank as f64);
            entry.0 += rrf;
            entry.1[si] = rrf;
            entry.2[si] = Some(doc.score);

            if config.score_alpha > 0.0 && span > 0.0 {
                let normalized = (doc.score - min) / span;
                entry.0 += config.score_alpha * w * normalized;
            }
        }
    }

    let mut out: Vec<FusedDoc<Id>> = acc
        .into_iter()
        .map(|(id, (fused_score, rrf_contributions, raw_scores))| FusedDoc {
            id,
            fused_score,
            rrf_contributions,
            raw_scores,
        })
        .collect();

    out.sort_by(|a, b| {
        b.fused_score
            .partial_cmp(&a.fused_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

/// Convenience for the canonical 3-stream pipeline
/// (semantic vector, structural PageRank, lexical BM25).
pub fn fuse_three<Id: Clone + Eq + Hash>(
    semantic: Vec<RankedDoc<Id>>,
    structural: Vec<RankedDoc<Id>>,
    lexical: Vec<RankedDoc<Id>>,
    config: &RrfConfig,
) -> Vec<FusedDoc<Id>> {
    fuse(&[semantic, structural, lexical], config)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_formula_with_default_config() {
        let streams = vec![
            vec![RankedDoc::new(1u32, 1.0)],
            vec![RankedDoc::new(1u32, 1.0)],
            vec![RankedDoc::new(1u32, 1.0)],
        ];
        let out = fuse(&streams, &RrfConfig::default());
        assert_eq!(out.len(), 1);
        // Present in all 3 streams at rank 0 -> 3 * (1/60).
        assert!((out[0].fused_score - 3.0 / 60.0).abs() < 1e-12);
        assert_eq!(out[0].rrf_contributions, vec![1.0 / 60.0; 3]);
    }

    #[test]
    fn per_stream_weights_change_ranking() {
        // Doc A only in stream 0 (semantic) — boosted by weight 2.
        // Doc B only in stream 1 (structural) — weight 1.
        let a = 1u32;
        let b = 2u32;
        let streams = vec![
            vec![RankedDoc::new(a, 1.0)],
            vec![RankedDoc::new(b, 1.0)],
            vec![],
        ];
        let cfg = RrfConfig {
            weights: vec![2.0, 1.0, 1.0],
            ..RrfConfig::default()
        };
        let out = fuse(&streams, &cfg);
        assert_eq!(out[0].id, a, "weighted stream should dominate");
        assert_eq!(out[1].id, b);

        // Uniform weights -> tie broken only by insertion order.
        let out = fuse(&streams, &RrfConfig::default());
        assert!((out[0].fused_score - out[1].fused_score).abs() < 1e-12);
    }

    #[test]
    fn score_blending_amplifies_confidence() {
        // Same topology as pure RRF, but stream scores differ widely.
        let a = 1u32;
        let b = 2u32;
        let streams = vec![
            vec![RankedDoc::new(a, 1000.0), RankedDoc::new(b, 1.0)],
            vec![],
            vec![],
        ];

        let pure = fuse(&streams, &RrfConfig::default());
        let blended = fuse(
            &streams,
            &RrfConfig {
                weights: vec![1.0, 0.0, 0.0],
                score_alpha: 0.5,
                ..RrfConfig::default()
            },
        );

        let margin_pure = pure[0].fused_score - pure[1].fused_score;
        let margin_blended = blended[0].fused_score - blended[1].fused_score;
        assert!(
            margin_blended > margin_pure,
            "score blending should widen the gap (pure={margin_pure}, blended={margin_blended})"
        );
        assert_eq!(blended[0].id, a);
        assert_eq!(blended[1].id, b);
    }

    #[test]
    fn duplicate_in_stream_keeps_best_rank() {
        let streams = vec![vec![
            RankedDoc::new(1u32, 100.0),
            RankedDoc::new(1u32, 50.0),
            RankedDoc::new(2u32, 1.0),
        ]];
        let out = fuse(&streams, &RrfConfig::default());
        // Doc 1 counted once, at rank 0.
        assert!((out[0].fused_score - 1.0 / 60.0).abs() < 1e-12);
        assert_eq!(out[0].rrf_contributions[0], 1.0 / 60.0);
    }

    #[test]
    fn empty_and_absent_streams_are_safe() {
        assert!(fuse::<u32>(&[], &RrfConfig::default()).is_empty());

        let streams = vec![
            vec![RankedDoc::new(1u32, 1.0)],
            vec![],
            vec![RankedDoc::new(2u32, 1.0)],
        ];
        let out = fuse(&streams, &RrfConfig::default());
        assert_eq!(out.len(), 2);
        assert!((out[0].fused_score - 1.0 / 60.0).abs() < 1e-12);
    }
}
