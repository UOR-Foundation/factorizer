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
        let dna_similarity = self.modular_dna.iter()
            .zip(&other.modular_dna)
            .filter(|(a, b)| a == b)
            .count() as f64 / self.modular_dna.len().max(1) as f64;
        
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
        let mut features: Vec<(&str, f64)> = self.components
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();
        
        // Sort by absolute value for importance
        features.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        
        features
    }
}