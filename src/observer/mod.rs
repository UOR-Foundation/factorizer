//! Observer module for empirical data collection
//!
//! This module emerges from the need to observe patterns without presumption.

pub mod analyzer;
pub mod collector;
pub mod constants;

use crate::types::{Number, Observation};
use crate::Result;

// Re-export main types
pub use analyzer::Analyzer;
pub use collector::Collector as ObservationCollector; // Alias for compatibility
pub use collector::{Collector, DetailLevel};
pub use constants::ConstantDiscovery;

/// Filter for observations
#[derive(Debug, Clone)]
pub struct ObservationFilter {
    /// Minimum number size
    pub min_size: Option<usize>,
    /// Maximum number size
    pub max_size: Option<usize>,
    /// Pattern type filter
    pub pattern_type: Option<crate::types::PatternKind>,
}

/// Main observer for pattern collection
#[derive(Debug)]
pub struct Observer {
    collector: Collector,
}

impl Observer {
    /// Create a new observer
    pub fn new() -> Self {
        Observer {
            collector: Collector::new(),
        }
    }

    /// Observe patterns in a range of numbers
    pub fn observe_range(&self, range: std::ops::Range<u64>) -> Vec<Observation> {
        self.collector.collect_range(range)
    }

    /// Observe patterns for specific numbers
    pub fn observe_numbers(&self, numbers: &[Number]) -> Vec<Observation> {
        self.collector.collect_numbers(numbers)
    }

    /// Find patterns in observations
    pub fn find_patterns(
        &self,
        observations: &[Observation],
    ) -> Result<Vec<crate::types::Pattern>> {
        Analyzer::find_patterns(observations)
    }
}

impl Default for Observer {
    fn default() -> Self {
        Self::new()
    }
}
