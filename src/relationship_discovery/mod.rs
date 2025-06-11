//! Relationship discovery module
//!
//! This module discovers relationships between patterns, observations,
//! and universal constants without imposing structure.

pub mod correlations;
pub mod networks;
pub mod synthesis;

use crate::types::{Observation, Pattern, UniversalConstant};
use crate::Result;

// Re-export main types
pub use correlations::CorrelationAnalysis;
pub use networks::PatternNetwork;
pub use synthesis::RelationshipSynthesis;

/// Discover all relationships in the data
pub fn discover_relationships(
    observations: &[Observation],
    patterns: &[Pattern],
    constants: &[UniversalConstant],
) -> Result<DiscoveredRelationships> {
    // Analyze correlations
    let correlations = CorrelationAnalysis::analyze(observations, patterns)?;

    // Build pattern network
    let network = PatternNetwork::build(patterns, &correlations)?;

    // Synthesize relationships
    let synthesis = RelationshipSynthesis::synthesize(observations, patterns, constants)?;

    Ok(DiscoveredRelationships {
        correlations,
        network,
        synthesis,
    })
}

/// Collection of discovered relationships
#[derive(Debug)]
pub struct DiscoveredRelationships {
    /// Pattern correlations
    pub correlations: Vec<PatternCorrelation>,

    /// Network of pattern relationships
    pub network: PatternNetwork,

    /// Synthesized relationships
    pub synthesis: RelationshipSynthesis,
}

/// Correlation between patterns
#[derive(Debug, Clone)]
pub struct PatternCorrelation {
    /// First pattern ID
    pub pattern_a: String,

    /// Second pattern ID
    pub pattern_b: String,

    /// Correlation strength (-1 to 1)
    pub correlation: f64,

    /// Statistical significance
    pub p_value: f64,

    /// Number of observations
    pub sample_size: usize,
}
