//! Pattern signature type
//!
//! The signature of a number emerges from empirical observation
//! of what information allows factor recognition.

use crate::types::Number;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pattern signature - the recognizable features of a number
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternSignature {
    /// The number being recognized
    pub value: Number,

    /// Universal components discovered through observation
    pub components: HashMap<String, f64>,

    /// Resonance field - empirically observed
    pub resonance: Vec<f64>,

    /// Modular DNA - always present
    pub modular_dna: Vec<u64>,

    /// Additional patterns that emerge
    pub emergent_features: HashMap<String, serde_json::Value>,
}

impl PatternSignature {
    /// Create a new pattern signature
    pub fn new(n: Number) -> Self {
        PatternSignature {
            value: n,
            components: HashMap::new(),
            resonance: Vec::new(),
            modular_dna: Vec::new(),
            emergent_features: HashMap::new(),
        }
    }

    /// Add a component to the signature
    pub fn add_component(&mut self, name: impl Into<String>, value: f64) {
        self.components.insert(name.into(), value);
    }

    /// Set the resonance field
    pub fn set_resonance(&mut self, field: Vec<f64>) {
        self.resonance = field;
    }

    /// Set the modular DNA
    pub fn set_modular_dna(&mut self, dna: Vec<u64>) {
        self.modular_dna = dna;
    }

    /// Add an emergent feature
    pub fn add_emergent_feature(&mut self, name: impl Into<String>, value: serde_json::Value) {
        self.emergent_features.insert(name.into(), value);
    }

    /// Get a component value
    pub fn get_component(&self, name: &str) -> Option<f64> {
        self.components.get(name).copied()
    }

    /// Calculate signature similarity
    pub fn similarity(&self, other: &PatternSignature) -> f64 {
        let mut total_similarity = 0.0;
        let mut count = 0;

        // Compare components
        for (name, value) in &self.components {
            if let Some(other_value) = other.components.get(name) {
                let diff = (value - other_value).abs();
                let max_val = value.abs().max(other_value.abs()).max(1.0);
                total_similarity += 1.0 - (diff / max_val).min(1.0);
                count += 1;
            }
        }

        // Compare modular DNA
        let dna_similarity =
            self.modular_dna.iter().zip(&other.modular_dna).filter(|(a, b)| a == b).count() as f64
                / self.modular_dna.len().max(1) as f64;

        total_similarity += dna_similarity;
        count += 1;

        if count > 0 {
            total_similarity / count as f64
        } else {
            0.0
        }
    }

    /// Extract key features for pattern matching
    pub fn key_features(&self) -> Vec<(&str, f64)> {
        let mut features: Vec<(&str, f64)> =
            self.components.iter().map(|(k, v)| (k.as_str(), *v)).collect();

        // Sort by absolute value for importance
        features.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        features
    }

    /// Extract multi-dimensional signature components
    pub fn extract_multidimensional(&self) -> HashMap<String, Vec<f64>> {
        let mut dimensions = HashMap::new();

        // Dimension 1: Modular patterns across different bases
        let mut mod_patterns = Vec::new();
        for base in &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29] {
            let pattern = self
                .modular_dna
                .iter()
                .enumerate()
                .filter(|(i, _)| i % base == 0)
                .map(|(_, &v)| v as f64)
                .collect::<Vec<_>>();
            mod_patterns.extend(pattern);
        }
        dimensions.insert("modular_patterns".to_string(), mod_patterns);

        // Dimension 2: Phase relationships
        let phases = extract_phase_relationships(&self.resonance);
        dimensions.insert("phase_relationships".to_string(), phases);

        // Dimension 3: Harmonic decomposition
        let harmonics = extract_harmonic_series(&self.resonance);
        dimensions.insert("harmonic_series".to_string(), harmonics);

        // Dimension 4: Statistical moments
        let moments = calculate_statistical_moments(&self.resonance);
        dimensions.insert("statistical_moments".to_string(), moments);

        dimensions
    }
}

/// Extract phase relationships from resonance field
fn extract_phase_relationships(resonance: &[f64]) -> Vec<f64> {
    let mut phases = Vec::new();

    // Use Hilbert transform approximation for phase extraction
    for i in 1..resonance.len() {
        let phase = (resonance[i] - resonance[i - 1]).atan2(resonance[i]);
        phases.push(phase);
    }

    // Normalize phases to [0, 2π]
    phases.iter_mut().for_each(|p| {
        while *p < 0.0 {
            *p += 2.0 * std::f64::consts::PI;
        }
        while *p > 2.0 * std::f64::consts::PI {
            *p -= 2.0 * std::f64::consts::PI;
        }
    });

    phases
}

/// Extract harmonic series from resonance data
fn extract_harmonic_series(resonance: &[f64]) -> Vec<f64> {
    let mut harmonics = Vec::new();
    let n = resonance.len();

    // Simple DFT for harmonic extraction
    for k in 0..n.min(20) {
        // First 20 harmonics
        let mut real = 0.0;
        let mut imag = 0.0;

        for (i, &val) in resonance.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64;
            real += val * angle.cos();
            imag += val * angle.sin();
        }

        let magnitude = (real * real + imag * imag).sqrt() / n as f64;
        harmonics.push(magnitude);
    }

    harmonics
}

/// Calculate statistical moments of the resonance field
fn calculate_statistical_moments(resonance: &[f64]) -> Vec<f64> {
    if resonance.is_empty() {
        return vec![0.0; 4];
    }

    let n = resonance.len() as f64;

    // Mean
    let mean = resonance.iter().sum::<f64>() / n;

    // Variance
    let variance = resonance.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;

    let std_dev = variance.sqrt();

    // Skewness
    let skewness = if std_dev > 0.0 {
        resonance.iter().map(|x| ((x - mean) / std_dev).powi(3)).sum::<f64>() / n
    } else {
        0.0
    };

    // Kurtosis
    let kurtosis = if std_dev > 0.0 {
        resonance.iter().map(|x| ((x - mean) / std_dev).powi(4)).sum::<f64>() / n - 3.0
    } else {
        0.0
    };

    vec![mean, variance, skewness, kurtosis]
}
