//! Sliding Window Turn-Summarization buffer + recency-boosted PPR teleport.
//!
//! Nodes edited or queried recently receive a temporary "recency boost" in
//! the Personalized PageRank teleport vector. The boost decays exponentially
//! with a configurable half-life, and the sliding window is summarized into a
//! tight semantic markdown summary as the coding session moves forward.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use serde::Serialize;

use crate::asg::SharedAsg;

/// A single recency event in the sliding window (serializable).
#[derive(Debug, Clone, Serialize)]
pub struct RecencyEvent {
    pub node_id: usize,
    pub tracker_id: String,
    pub name: String,
    pub kind: String,
    pub file_path: String,
    /// Seconds since the event was recorded.
    pub age_secs: f64,
    pub action: String,
}

struct RecencyEventInternal {
    node_id: usize,
    tracker_id: String,
    name: String,
    kind: String,
    file_path: String,
    timestamp: Instant,
    action: String,
}

/// Sliding-window recency tracker with exponential decay.
pub struct RecencyTracker {
    events: Mutex<Vec<RecencyEventInternal>>,
    /// Half-life of the recency boost (default 120s).
    pub half_life_secs: f64,
    /// Sliding window length (default 600s).
    pub window_secs: f64,
    /// Maximum boost applied to a just-touched node.
    pub max_boost: f64,
}

impl Default for RecencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl RecencyTracker {
    pub fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            half_life_secs: 120.0,
            window_secs: 600.0,
            max_boost: 3.0,
        }
    }

    /// Record an interaction with a node.
    pub fn record(&self, node_id: usize, action: &str, asg: &SharedAsg) {
        let Some(node) = asg.get_node(node_id) else {
            return;
        };
        let mut events = self.events.lock().unwrap();
        let now = Instant::now();
        // Remove stale events outside the sliding window.
        events.retain(|e| now.duration_since(e.timestamp).as_secs_f64() < self.window_secs);
        events.push(RecencyEventInternal {
            node_id,
            tracker_id: node.tracker_id.clone(),
            name: node.name.clone(),
            kind: node.kind.clone(),
            file_path: node.file_path.display().to_string(),
            timestamp: now,
            action: action.to_string(),
        });
    }

    /// Exponential recency boost for a node: `max_boost * 2^(-age/half_life)`.
    pub fn recency_boost(&self, node_id: usize) -> f64 {
        let events = self.events.lock().unwrap();
        let now = Instant::now();
        let mut best = 0.0f64;
        for e in events.iter() {
            if e.node_id != node_id {
                continue;
            }
            let age = now.duration_since(e.timestamp).as_secs_f64();
            if age >= self.window_secs {
                continue;
            }
            let boost = self.boost_for_age(age);
            if boost > best {
                best = boost;
            }
        }
        best
    }

    /// Apply recency boosts to a PPR seed vector in place.
    pub fn apply_boosts(&self, seeds: &mut Vec<(usize, f64)>) {
        let events = self.events.lock().unwrap();
        if events.is_empty() {
            return;
        }
        let now = Instant::now();
        let mut boosts: HashMap<usize, f64> = HashMap::new();
        for e in events.iter() {
            let age = now.duration_since(e.timestamp).as_secs_f64();
            if age >= self.window_secs {
                continue;
            }
            let boost = self.boost_for_age(age);
            let entry = boosts.entry(e.node_id).or_default();
            if boost > *entry {
                *entry = boost;
            }
        }
        for (node_id, boost) in boosts {
            if let Some(slot) = seeds.iter_mut().find(|(id, _)| *id == node_id) {
                slot.1 += boost;
            } else {
                seeds.push((node_id, boost));
            }
        }
    }

    /// Produce a tight semantic markdown summary of the sliding window.
    pub fn summarize_window(&self) -> String {
        let events = self.events.lock().unwrap();
        if events.is_empty() {
            return "No recent activity in the sliding window.".to_string();
        }
        let now = Instant::now();
        let mut lines: Vec<String> = Vec::new();
        for e in events.iter().rev().take(20) {
            let age = now.duration_since(e.timestamp).as_secs_f64();
            let boost = self.boost_for_age(age);
            lines.push(format!(
                "- `{}` ({}) [{}] — {}s ago, boost {:.2}",
                e.name, e.kind, e.action, age as u64, boost
            ));
        }
        format!("## Recent activity (sliding window)\n{}", lines.join("\n"))
    }

    /// List recent events (for the `/v1/recency` endpoint).
    pub fn events(&self) -> Vec<RecencyEvent> {
        let events = self.events.lock().unwrap();
        let now = Instant::now();
        events
            .iter()
            .rev()
            .take(50)
            .map(|e| RecencyEvent {
                node_id: e.node_id,
                tracker_id: e.tracker_id.clone(),
                name: e.name.clone(),
                kind: e.kind.clone(),
                file_path: e.file_path.clone(),
                age_secs: now.duration_since(e.timestamp).as_secs_f64(),
                action: e.action.clone(),
            })
            .collect()
    }

    fn boost_for_age(&self, age_secs: f64) -> f64 {
        if age_secs < 0.0 || !self.half_life_secs.is_finite() || self.half_life_secs <= 0.0 {
            return self.max_boost;
        }
        self.max_boost
            * (-(std::f64::consts::LN_2) * age_secs / self.half_life_secs).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boost_decays_with_age() {
        let tracker = RecencyTracker::new();
        let fresh = tracker.boost_for_age(0.0);
        let old = tracker.boost_for_age(tracker.half_life_secs);
        assert!(fresh > old);
        assert!((old - tracker.max_boost / 2.0).abs() < 1e-9);
    }

    #[test]
    fn apply_boosts_adds_seeds() {
        let tracker = RecencyTracker::new();
        let mut seeds = vec![(1usize, 1.0f64)];
        tracker.apply_boosts(&mut seeds);
        assert_eq!(seeds.len(), 1);
        assert_eq!(seeds[0].1, 1.0);
    }
}