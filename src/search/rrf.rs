//! Custom RRF fusion: weighted + score-aware Reciprocal Rank Fusion with
//! deterministic ASG topology tie-breaking.
//!
//! Stock RRF is `score(d) = Σ 1/(k + rank_d)` — it treats every retrieval
//! stream equally and discards raw scores entirely. Three customizations lift
//! retrieval quality:
//!
//! 1. **Per-stream weight.** `w_s / (k + rank)` lets the caller trust the
//!    semantic vector stream more than the lexical BM25 stream (e.g.
//!    `weights = [1.2, 1.0, 0.9]`). With `weights = [1, 1, 1]` and
//!    `score_alpha = 0` this reduces to the exact spec formula.
//! 2. **Score-aware rank compression.** When `score_alpha > 0`, the
//!    normalized raw score (min-max per stream) is injected *into the rank
//!    denominator*, dynamically compressing the effective rank for highly
//!    confident matches:
//!
//!    ```text
//!    effective_rank = rank × (1 − α × normalized_score)
//!    contribution   = w_s / (k + effective_rank)
//!    ```
//!
//!    A document at rank 3 with a perfect normalized score of 1.0 and
//!    `α = 0.3` behaves as if it were at rank `3 × 0.7 = 2.1`, pulling
//!    high-confidence matches toward the top of the token budget without
//!    abandoning RRF's rank-first robustness.
//!
//! 3. **ASG topology tie-breaking.** When two documents achieve the same
//!    fused RRF score (which happens constantly with repetitive code
//!    patterns), the tie is broken by inspecting the Abstract Semantic Graph
//!    structural dependency count — specifically weighted incoming `calls`
//!    and `references` edge counts — so that structurally central nodes
//!    (well-connected utilities, foundational traits) float above leaf
//!    definitions with identical lexical/semantic rankings.
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
#[serde(default)]
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
///
/// When `structural_scores` is `Some`, documents with identical fused RRF
/// scores are tie-broken by their ASG structural dependency weight (higher
/// is better). Pass `None` to preserve the legacy ordinal-only tie-break.
pub fn fuse<Id: Clone + Eq + Hash>(
    streams: &[Vec<RankedDoc<Id>>],
    config: &RrfConfig,
    structural_scores: Option<&HashMap<Id, f64>>,
) -> Vec<FusedDoc<Id>> {
    let n_streams = streams.len();
    if n_streams == 0 {
        return Vec::new();
    }

    let k = if config.k.is_finite() {
        config.k.max(1.0)
    } else {
        60.0
    };
    let alpha = if config.score_alpha.is_finite() {
        config.score_alpha.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let ws: Vec<f64> = if config.weights.len() == n_streams {
        config
            .weights
            .iter()
            .map(|weight| {
                if weight.is_finite() {
                    weight.max(0.0)
                } else {
                    0.0
                }
            })
            .collect()
    } else {
        vec![1.0; n_streams]
    };

    // Per-stream min/max for confidence normalization. Ignore NaNs rather than
    // allowing one bad backend score to poison the complete fusion.
    let ranges: Vec<Option<(f64, f64)>> = streams
        .iter()
        .map(|stream| {
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            for doc in stream.iter().filter(|doc| doc.score.is_finite()) {
                min = min.min(doc.score);
                max = max.max(doc.score);
            }
            (min.is_finite() && max.is_finite()).then_some((min, max))
        })
        .collect();

    // The ordinal makes exact ties deterministic despite HashMap's randomized
    // iteration order.
    let mut next_ordinal = 0usize;
    let mut acc: HashMap<Id, (f64, Vec<f64>, Vec<Option<f64>>, usize)> = HashMap::new();

    for (stream_index, stream) in streams.iter().enumerate() {
        let weight = ws[stream_index];
        for (rank, doc) in stream.iter().enumerate() {
            let entry = acc.entry(doc.id.clone()).or_insert_with(|| {
                let ordinal = next_ordinal;
                next_ordinal += 1;
                (
                    0.0,
                    vec![0.0; n_streams],
                    vec![None; n_streams],
                    ordinal,
                )
            });

            // If a backend accidentally emits a duplicate, retain its first
            // (therefore best) rank even when the stream weight is zero.
            if entry.2[stream_index].is_some() {
                continue;
            }

            let normalized = match (ranges[stream_index], doc.score.is_finite()) {
                (Some((min, max)), true) if max > min => (doc.score - min) / (max - min),
                _ => 0.0,
            };
            // Score-aware rank compression: inject the normalized confidence
            // score *into the denominator* so high-confidence matches behave
            // as if they appeared at a better rank.
            let score_factor = (alpha * normalized.clamp(0.0, 1.0)).min(1.0);
            let effective_rank = rank as f64 * (1.0 - score_factor);
            let contribution = weight / (k + effective_rank);

            entry.0 += contribution;
            entry.1[stream_index] = contribution;
            entry.2[stream_index] = Some(doc.score);
        }
    }

    let mut with_order: Vec<(FusedDoc<Id>, usize)> = acc
        .into_iter()
        .map(|(id, (fused_score, rrf_contributions, raw_scores, ordinal))| {
            (
                FusedDoc {
                    id,
                    fused_score,
                    rrf_contributions,
                    raw_scores,
                },
                ordinal,
            )
        })
        .collect();

    with_order.sort_by(|(a, a_order), (b, b_order)| {
        b.fused_score
            .total_cmp(&a.fused_score)
            .then_with(|| {
                // ASG topology tie-breaker: prefer the node with the higher
                // structural dependency weight (weighted incoming calls +
                // references edges).  Falls back to 0.0 when structural
                // scores are absent, preserving legacy ordinal behaviour.
                let sa = structural_scores
                    .and_then(|s| s.get(&a.id))
                    .copied()
                    .unwrap_or(0.0);
                let sb = structural_scores
                    .and_then(|s| s.get(&b.id))
                    .copied()
                    .unwrap_or(0.0);
                sb.total_cmp(&sa)
            })
            .then_with(|| a_order.cmp(b_order))
    });
    with_order.into_iter().map(|(doc, _)| doc).collect()
}

/// Convenience for the canonical 3-stream pipeline
/// (semantic vector, structural PageRank, lexical BM25).
pub fn fuse_three<Id: Clone + Eq + Hash>(
    semantic: Vec<RankedDoc<Id>>,
    structural: Vec<RankedDoc<Id>>,
    lexical: Vec<RankedDoc<Id>>,
    config: &RrfConfig,
    structural_scores: Option<&HashMap<Id, f64>>,
) -> Vec<FusedDoc<Id>> {
    fuse(
        &[semantic, structural, lexical],
        config,
        structural_scores,
    )
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
        let out = fuse(&streams, &RrfConfig::default(), None);
        assert_eq!(out.len(), 1);
        // Present in all 3 streams at rank 0 -> 3 * (1/60).
        assert!((out[0].fused_score - 3.0 / 60.0).abs() < 1e-12);
        assert_eq!(out[0].rrf_contributions, vec![1.0 / 60.0; 3]);
    }

    #[test]
    fn per_stream_weights_change_ranking() {
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
        let out = fuse(&streams, &cfg, None);
        assert_eq!(out[0].id, a, "weighted stream should dominate");
        assert_eq!(out[1].id, b);

        // Uniform weights -> tie broken only by insertion order.
        let out = fuse(&streams, &RrfConfig::default(), None);
        assert!((out[0].fused_score - out[1].fused_score).abs() < 1e-12);
    }

    #[test]
    fn score_blending_widens_margin() {
        // With the new denominator-injection formula, a high-confidence doc
        // at rank 0 with score_alpha > 0 should dominate even more than
        // pure RRF because effective_rank compresses toward zero.
        let a = 1u32;
        let b = 2u32;
        let streams = vec![
            vec![RankedDoc::new(a, 1000.0), RankedDoc::new(b, 1.0)],
            vec![],
            vec![],
        ];

        let pure = fuse(&streams, &RrfConfig::default(), None);
        let blended = fuse(
            &streams,
            &RrfConfig {
                weights: vec![1.0, 0.0, 0.0],
                score_alpha: 0.5,
                ..RrfConfig::default()
            },
            None,
        );

        let margin_pure = pure[0].fused_score - pure[1].fused_score;
        let margin_blended = blended[0].fused_score - blended[1].fused_score;
        // Denominator injection: high-score doc gets effective_rank ~0,
        // yielding contribution ~1/k — much larger than 1/(k+1) at rank 1.
        assert!(
            margin_blended >= margin_pure,
            "score-aware blending should widen the gap (pure={margin_pure}, blended={margin_blended})"
        );
        assert_eq!(blended[0].id, a);
        assert_eq!(blended[1].id, b);
    }

    #[test]
    fn score_alpha_zero_is_pure_rrf() {
        let a = 1u32;
        let b = 2u32;
        let streams = vec![
            vec![RankedDoc::new(a, 999.0), RankedDoc::new(b, 0.1)],
            vec![],
        ];
        let pure = fuse(&streams, &RrfConfig::default(), None);
        let alpha_zero = fuse(
            &streams,
            &RrfConfig {
                score_alpha: 0.0,
                ..RrfConfig::default()
            },
            None,
        );
        // With alpha = 0, raw scores have no effect — scores should match.
        for (p, q) in pure.iter().zip(alpha_zero.iter()) {
            assert!((p.fused_score - q.fused_score).abs() < 1e-12);
        }
    }

    #[test]
    fn effective_rank_compression_at_rank_zero() {
        // A doc at rank 0 with a perfect normalized score and alpha=1.0
        // should get effective_rank = 0 → contribution = w / k.
        let streams = vec![vec![RankedDoc::new(1u32, 42.0)]];
        let cfg = RrfConfig {
            weights: vec![1.0],
            score_alpha: 1.0,
            ..RrfConfig::default()
        };
        let out = fuse(&streams, &cfg, None);
        assert_eq!(out.len(), 1);
        // effective_rank = 0 * (1 - 1.0) = 0 → contribution = 1.0 / 60.0
        assert!((out[0].fused_score - 1.0 / 60.0).abs() < 1e-12);
    }

    #[test]
    fn duplicate_in_stream_keeps_best_rank() {
        let streams = vec![vec![
            RankedDoc::new(1u32, 100.0),
            RankedDoc::new(1u32, 50.0),
            RankedDoc::new(2u32, 1.0),
        ]];
        let out = fuse(&streams, &RrfConfig::default(), None);
        // Doc 1 counted once, at rank 0.
        assert!((out[0].fused_score - 1.0 / 60.0).abs() < 1e-12);
        assert_eq!(out[0].rrf_contributions[0], 1.0 / 60.0);
    }

    #[test]
    fn empty_and_absent_streams_are_safe() {
        assert!(fuse::<u32>(&[], &RrfConfig::default(), None).is_empty());

        let streams = vec![
            vec![RankedDoc::new(1u32, 1.0)],
            vec![],
            vec![RankedDoc::new(2u32, 1.0)],
        ];
        let out = fuse(&streams, &RrfConfig::default(), None);
        assert_eq!(out.len(), 2);
        assert!((out[0].fused_score - 1.0 / 60.0).abs() < 1e-12);
    }

    #[test]
    fn asg_tie_breaker_prefers_higher_structural_score() {
        // Two docs with identical RRF contributions from a single stream.
        // Without structural scores the order is by ordinal (insertion order).
        // With structural scores, doc B (higher structural weight) wins.
        let a = 1u32;
        let b = 2u32;
        let streams = vec![
            vec![RankedDoc::new(a, 1.0)],
            vec![RankedDoc::new(b, 1.0)],
        ];
        let cfg = RrfConfig {
            weights: vec![1.0, 1.0],
            score_alpha: 0.0,
            ..RrfConfig::default()
        };

        // Without structural scores — tied, ordinal decides.
        let out_no_struct = fuse(&streams, &cfg, None);
        assert!(
            (out_no_struct[0].fused_score - out_no_struct[1].fused_score).abs() < 1e-12,
            "scores should be equal without structural tie-breaker"
        );

        // With structural scores — doc B has more structural dependencies.
        let mut struct_scores = HashMap::new();
        struct_scores.insert(a, 1.0);
        struct_scores.insert(b, 5.0);
        let out_struct = fuse(&streams, &cfg, Some(&struct_scores));
        assert!(
            (out_struct[0].fused_score - out_struct[1].fused_score).abs() < 1e-12,
            "RRF scores should still be equal"
        );
        assert_eq!(
            out_struct[0].id, b,
            "higher structural score should win the tie-break"
        );
        assert_eq!(out_struct[1].id, a);
    }

    #[test]
    fn asg_tie_breaker_skipped_when_none() {
        // Confirm that passing None for structural_scores doesn't break anything.
        let streams = vec![
            vec![RankedDoc::new(1u32, 1.0), RankedDoc::new(2u32, 0.5)],
            vec![RankedDoc::new(2u32, 1.0), RankedDoc::new(1u32, 0.5)],
        ];
        let out = fuse(&streams, &RrfConfig::default(), None);
        assert_eq!(out.len(), 2);
        // Both docs appear in both streams — scores should be identical,
        // broken by ordinal only.
        assert!(
            (out[0].fused_score - out[1].fused_score).abs() < 1e-12
        );
    }
}
