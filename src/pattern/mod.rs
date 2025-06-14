//! Pattern module - The core of The Pattern implementation
//!
//! This module implements the three stages of The Pattern:
//! 1. Recognition - Extract signature from number
//! 2. Formalization - Express in mathematical language
//! 3. Execution - Decode factors from pattern

// Core modules - The three stages of The Pattern
pub mod recognition;
pub mod formalization;
pub mod execution;

// Unified basis system
pub mod basis; // DEPRECATED: Uses f64, limited precision
pub mod basis_exact; // RECOMMENDED: Uses exact arithmetic

// Supporting modules
pub mod advanced;
pub mod expression;
pub mod verification;
pub mod cache;

// Persistence and pre-computed basis modules (for examples)
pub mod basis_persistence; // DEPRECATED: Uses f64, limited precision
pub mod basis_persistence_exact; // RECOMMENDED: Uses exact arithmetic
pub mod precomputed_basis; // Re-exports from basis module

// Pattern implementations
pub mod universal_pattern; // Now uses exact arithmetic internally

// 8-bit stream processor
pub mod stream_processor;
pub mod stream_processor_exact;

// Empirical pattern implementation
pub mod empirical_pattern;

// Direct empirical pattern (no theory, just learned mappings)
pub mod direct_empirical; // DEPRECATED: Limited to 224-bit
pub mod direct_empirical_exact; // RECOMMENDED: Arbitrary precision

// Wave synthesis auto-tuner
pub mod wave_synthesis; // DEPRECATED: Uses f64, limited to 224-bit
pub mod wave_synthesis_exact; // RECOMMENDED: Uses exact arithmetic

use crate::types::{Factors, Formalization, Number, Recognition};
use crate::Result;
use serde::{Deserialize, Serialize};

/// The Pattern - main implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// Discovered patterns from observations
    patterns: Vec<crate::types::Pattern>,

    /// Universal constants discovered
    constants: Vec<crate::types::UniversalConstant>,

    /// Recognition parameters
    recognition_params: RecognitionParams,
}

/// Parameters for pattern recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionParams {
    /// Number of primes for modular DNA
    modular_prime_count: usize,

    /// Size of resonance field
    resonance_field_size: usize,

    /// Threshold for pattern matching
    pattern_threshold: f64,
}

impl Default for RecognitionParams {
    fn default() -> Self {
        RecognitionParams {
            modular_prime_count: 30,
            resonance_field_size: 100,
            pattern_threshold: 0.8,
        }
    }
}

impl Pattern {
    /// Create a new Pattern from observations
    pub fn from_observations(observations: &[crate::types::Observation]) -> Result<Self> {
        use crate::observer::{Analyzer, ConstantDiscovery};

        // Let patterns emerge from data
        let patterns = Analyzer::find_patterns(observations)?;
        let constants = ConstantDiscovery::extract(&patterns);

        Ok(Pattern {
            patterns,
            constants,
            recognition_params: RecognitionParams::default(),
        })
    }

    /// Create from JSON data
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| e.into())
    }

    /// Stage 1: Recognition
    pub fn recognize(&self, n: &Number) -> Result<Recognition> {
        recognition::recognize_with_params(n, &self.constants, &self.recognition_params)
    }

    /// Stage 2: Formalization
    pub fn formalize(&self, recognition: Recognition) -> Result<Formalization> {
        formalization::formalize(recognition, &self.patterns, &self.constants)
    }

    /// Stage 3: Execution
    pub fn execute(&self, formalization: Formalization) -> Result<Factors> {
        execution::execute(formalization, &self.patterns)
    }

    /// Get the discovered patterns
    pub fn patterns(&self) -> &[crate::types::Pattern] {
        &self.patterns
    }

    /// Get the discovered constants
    pub fn constants(&self) -> &[crate::types::UniversalConstant] {
        &self.constants
    }

    /// Discover patterns from observations (static method)
    pub fn discover_from_observations(
        observations: &[crate::types::Observation],
    ) -> Result<Vec<crate::types::Pattern>> {
        use crate::observer::Analyzer;
        Analyzer::find_patterns(observations)
    }

    /// Check if pattern applies to a number
    pub fn applies_to(&self, n: &Number) -> bool {
        // Pattern applies if we can successfully recognize it
        self.recognize(n).is_ok()
    }
}
