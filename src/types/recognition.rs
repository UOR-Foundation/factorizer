//! Recognition and execution types
//!
//! These types represent the stages of pattern recognition and execution.

use crate::types::{Number, PatternSignature, QuantumRegion};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Recognition - what The Pattern sees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recognition {
    /// The extracted signature
    pub signature: PatternSignature,

    /// Identified pattern type
    pub pattern_type: PatternType,

    /// Confidence in recognition (0-1)
    pub confidence: f64,

    /// Quantum neighborhood if identified
    pub quantum_neighborhood: Option<QuantumRegion>,

    /// Additional recognition metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Types of patterns recognized
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum PatternType {
    /// Balanced semiprime (p ≈ q)
    Balanced,

    /// Harmonic pattern (large p/q ratio)
    Harmonic,

    /// Perfect square (p = q)
    Square,

    /// Small factor present
    SmallFactor,

    /// Prime number
    Prime,

    /// Unknown pattern
    Unknown,
}

/// Formalization - mathematical expression of recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Formalization {
    /// The original number
    pub n: Number,

    /// Universal encoding of the pattern
    pub universal_encoding: HashMap<String, f64>,

    /// Resonance peaks in the field
    pub resonance_peaks: Vec<usize>,

    /// Harmonic series expansion
    pub harmonic_series: Vec<f64>,

    /// Pattern matrix representation
    pub pattern_matrix: PatternMatrix,

    /// Decoding strategies to try
    pub strategies: Vec<DecodingStrategy>,

    /// Pattern constraints (stored as JSON)
    #[serde(skip_serializing_if = "Option::is_none")]
    constraints: Option<serde_json::Value>,
}

/// Pattern matrix wrapper for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatrix {
    /// Matrix data as flat vector
    pub data: Vec<f64>,

    /// Matrix dimensions (rows, cols)
    pub shape: (usize, usize),
}

impl PatternMatrix {
    /// Create from ndarray
    pub fn from_array(array: Array2<f64>) -> Self {
        let shape = (array.nrows(), array.ncols());
        let data = array.into_raw_vec();
        PatternMatrix { data, shape }
    }

    /// Convert to ndarray
    pub fn to_array(&self) -> Array2<f64> {
        Array2::from_shape_vec(self.shape, self.data.clone()).expect("Invalid matrix shape")
    }
}

/// Strategies for decoding factors
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DecodingStrategy {
    /// Use resonance peaks
    ResonancePeaks,

    /// Use eigenvalues
    Eigenvalues,

    /// Use harmonic intersection
    HarmonicIntersection,

    /// Use phase relationships
    PhaseRelationships,

    /// Use quantum materialization
    QuantumMaterialization,

    /// Use modular patterns
    ModularPatterns,
}

/// The result of pattern execution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Factors {
    /// First prime factor (p ≤ q)
    pub p: Number,

    /// Second prime factor
    pub q: Number,

    /// How the factors were found
    pub method: String,

    /// Confidence in the factorization
    pub confidence: f64,
}

impl Factors {
    /// Create new factors
    pub fn new(p: Number, q: Number, method: impl Into<String>) -> Self {
        let (p, q) = if p <= q { (p, q) } else { (q, p) };
        Factors {
            p,
            q,
            method: method.into(),
            confidence: 1.0,
        }
    }

    /// Verify the factorization
    pub fn verify(&self, n: &Number) -> bool {
        &self.p * &self.q == *n
    }
}

impl Recognition {
    /// Create new recognition
    pub fn new(signature: PatternSignature, pattern_type: PatternType) -> Self {
        Recognition {
            signature,
            pattern_type,
            confidence: 0.0,
            quantum_neighborhood: None,
            metadata: HashMap::new(),
        }
    }

    /// Set confidence level
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set quantum neighborhood
    pub fn with_quantum_region(mut self, region: QuantumRegion) -> Self {
        self.quantum_neighborhood = Some(region);
        self
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.metadata.insert(key.into(), value);
    }
}

impl Formalization {
    /// Create a new formalization
    pub fn new(
        n: Number,
        universal_encoding: HashMap<String, f64>,
        resonance_peaks: Vec<usize>,
        harmonic_series: Vec<f64>,
        pattern_matrix: PatternMatrix,
        strategies: Vec<DecodingStrategy>,
    ) -> Self {
        Formalization {
            n,
            universal_encoding,
            resonance_peaks,
            harmonic_series,
            pattern_matrix,
            strategies,
            constraints: None,
        }
    }

    /// Add constraints to the formalization
    pub fn add_constraints(
        &mut self,
        constraints: Vec<crate::pattern::expression::PatternConstraint>,
    ) {
        self.constraints =
            Some(serde_json::to_value(constraints).unwrap_or(serde_json::Value::Null));
    }

    /// Get constraints from the formalization
    pub fn get_constraints(&self) -> Option<Vec<crate::pattern::expression::PatternConstraint>> {
        self.constraints.as_ref().and_then(|v| serde_json::from_value(v.clone()).ok())
    }
}
