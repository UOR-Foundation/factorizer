//! Exact arithmetic basis persistence module
//! 
//! This module provides serialization support for the exact arithmetic basis system,
//! allowing pre-computed basis with arbitrary precision to be saved and loaded.

use crate::pattern::basis_exact::BasisExact;
use crate::types::{Number, Rational};
use crate::types::constants::FundamentalConstantsRational;
use crate::error::PatternError;
use crate::Result;
use std::collections::HashMap;
use std::path::Path;
use std::fs;
use serde::{Serialize, Deserialize};

/// Serializable version of exact arithmetic scaling constants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConstantsExact {
    pub alpha: String,      // Serialized as "numerator/denominator"
    pub beta: String,
    pub gamma: String,
    pub delta: String,
    pub epsilon: String,
    pub phi: String,
    pub tau: String,
    pub unity: String,
}

impl From<&crate::pattern::basis_exact::ScalingConstantsExact> for ScalingConstantsExact {
    fn from(constants: &crate::pattern::basis_exact::ScalingConstantsExact) -> Self {
        ScalingConstantsExact {
            alpha: format!("{}/{}", constants.resonance_decay_alpha.numerator(), constants.resonance_decay_alpha.denominator()),
            beta: format!("{}/{}", constants.phase_coupling_beta.numerator(), constants.phase_coupling_beta.denominator()),
            gamma: format!("{}/{}", constants.scale_transition_gamma.numerator(), constants.scale_transition_gamma.denominator()),
            delta: format!("{}/{}", constants.interference_null_delta.numerator(), constants.interference_null_delta.denominator()),
            epsilon: format!("{}/{}", constants.adelic_threshold_epsilon.numerator(), constants.adelic_threshold_epsilon.denominator()),
            phi: format!("{}/{}", constants.golden_ratio_phi.numerator(), constants.golden_ratio_phi.denominator()),
            tau: format!("{}/{}", constants.tribonacci_tau.numerator(), constants.tribonacci_tau.denominator()),
            unity: format!("1/1"), // Unity is always 1/1
        }
    }
}

impl ScalingConstantsExact {
    /// Convert back to FundamentalConstantsRational
    pub fn to_rational(&self, precision_bits: u32) -> Result<FundamentalConstantsRational> {
        let mut constants = FundamentalConstantsRational::new(precision_bits);
        
        // Parse each rational from string
        constants.alpha = self.parse_rational(&self.alpha)?;
        constants.beta = self.parse_rational(&self.beta)?;
        constants.gamma = self.parse_rational(&self.gamma)?;
        constants.delta = self.parse_rational(&self.delta)?;
        constants.epsilon = self.parse_rational(&self.epsilon)?;
        constants.phi = self.parse_rational(&self.phi)?;
        constants.tau = self.parse_rational(&self.tau)?;
        constants.unity = self.parse_rational(&self.unity)?;
        
        Ok(constants)
    }
    
    fn parse_rational(&self, s: &str) -> Result<Rational> {
        let parts: Vec<&str> = s.split('/').collect();
        if parts.len() != 2 {
            return Err(PatternError::ConfigError(format!("Invalid rational format: {}", s)));
        }
        
        let num = Number::from_str_radix(parts[0], 10)
            .map_err(|_| PatternError::ConfigError(format!("Invalid numerator: {}", parts[0])))?;
        let den = Number::from_str_radix(parts[1], 10)
            .map_err(|_| PatternError::ConfigError(format!("Invalid denominator: {}", parts[1])))?;
        
        Ok(Rational::from_ratio(num, den))
    }
}

/// Serializable version of the exact arithmetic Basis
#[derive(Debug, Serialize, Deserialize)]
pub struct SerializableBasisExact {
    /// Base universal coordinates as rational strings [φ, π, e, 1]
    pub base: [String; 4],
    
    /// Universal scaling constants
    pub scaling_constants: ScalingConstantsExact,
    
    /// Pre-computed resonance templates by bit size (as integer strings)
    pub resonance_templates: HashMap<u32, Vec<String>>,
    
    /// Enhanced patterns for specific bit ranges
    pub bit_range_patterns: HashMap<String, ScalingConstantsExact>,
    
    /// Pre-computed harmonic basis functions (as integer strings)
    pub harmonic_basis: Vec<Vec<String>>,
    
    /// Precision in bits
    pub precision_bits: u32,
}

/// Generate and save exact arithmetic basis files
pub fn generate_exact_basis_files(precision_bits: u32) -> Result<()> {
    // Create data/basis directory if it doesn't exist
    let basis_dir = Path::new("data/basis");
    fs::create_dir_all(basis_dir)
        .map_err(|e| PatternError::ConfigError(format!("Failed to create basis directory: {}", e)))?;
    
    // Generate exact basis
    let basis_exact = BasisExact::new(precision_bits);
    
    // Convert to serializable format
    let base_strings = [
        format!("{}/{}", basis_exact.base[0].numerator(), basis_exact.base[0].denominator()),
        format!("{}/{}", basis_exact.base[1].numerator(), basis_exact.base[1].denominator()),
        format!("{}/{}", basis_exact.base[2].numerator(), basis_exact.base[2].denominator()),
        format!("{}/{}", basis_exact.base[3].numerator(), basis_exact.base[3].denominator()),
    ];
    
    let scaling_constants = ScalingConstantsExact::from(&basis_exact.scaling_constants);
    
    // Convert resonance templates
    let mut resonance_templates = HashMap::new();
    for (bits, template) in &basis_exact.resonance_templates {
        let string_template: Vec<String> = template.iter()
            .map(|n| n.to_string())
            .collect();
        resonance_templates.insert(*bits, string_template);
    }
    
    // Convert harmonic basis
    let harmonic_basis: Vec<Vec<String>> = basis_exact.harmonic_basis.iter()
        .map(|harmonic| harmonic.iter().map(|n| n.to_string()).collect())
        .collect();
    
    let serializable = SerializableBasisExact {
        base: base_strings,
        scaling_constants,
        resonance_templates,
        bit_range_patterns: HashMap::new(), // Will be filled with bit-range specific patterns
        harmonic_basis,
        precision_bits,
    };
    
    // Save the exact basis
    let json = serde_json::to_string_pretty(&serializable)
        .map_err(|e| PatternError::ConfigError(format!("Failed to serialize exact basis: {}", e)))?;
    
    fs::write(basis_dir.join("universal_basis_exact.json"), json)
        .map_err(|e| PatternError::ConfigError(format!("Failed to write exact basis file: {}", e)))?;
    
    Ok(())
}