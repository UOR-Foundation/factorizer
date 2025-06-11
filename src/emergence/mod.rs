//! Emergence module - Where patterns reveal themselves
//!
//! This module contains functionality for discovering emergent patterns
//! from empirical observations without imposing structure.

pub mod invariants;
pub mod scaling;
pub mod universal;

use crate::types::{Observation, Pattern};
use crate::Result;

// Re-export main types
pub use invariants::InvariantDiscovery;
pub use scaling::ScalingAnalysis;
pub use universal::UniversalPatterns;

/// Discover all emergent patterns from observations
pub fn discover_all_patterns(observations: &[Observation]) -> Result<EmergentPatterns> {
    let invariants = InvariantDiscovery::find_all(observations)?;
    let scaling_patterns = ScalingAnalysis::analyze_all(observations)?;
    let universal_patterns = UniversalPatterns::discover(observations)?;

    Ok(EmergentPatterns {
        invariants,
        scaling_patterns,
        universal_patterns,
    })
}

/// Collection of all emergent patterns
#[derive(Debug)]
pub struct EmergentPatterns {
    /// Invariant relationships
    pub invariants: Vec<Pattern>,

    /// Scale-dependent patterns
    pub scaling_patterns: Vec<Pattern>,

    /// Universal patterns
    pub universal_patterns: Vec<Pattern>,
}

impl EmergentPatterns {
    /// Get all patterns
    pub fn all_patterns(&self) -> Vec<&Pattern> {
        self.invariants
            .iter()
            .chain(&self.scaling_patterns)
            .chain(&self.universal_patterns)
            .collect()
    }

    /// Count total patterns
    pub fn count(&self) -> usize {
        self.invariants.len() + self.scaling_patterns.len() + self.universal_patterns.len()
    }
}
