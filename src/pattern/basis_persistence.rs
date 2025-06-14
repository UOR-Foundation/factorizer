//! Basis persistence module for saving and loading pre-computed basis
//! 
//! This module provides serialization support for the basis system,
//! allowing pre-computed basis to be saved and loaded from disk.

use crate::pattern::basis::{Basis, ScalingConstants};
use crate::error::PatternError;
use crate::Result;
use std::collections::HashMap;
use std::path::Path;
use std::fs;
use serde::{Serialize, Deserialize};

/// Serializable version of the Basis struct
#[derive(Debug, Serialize, Deserialize)]
pub struct SerializableBasis {
    /// Base universal coordinates [φ, π, e, 1]
    pub base: [f64; 4],
    
    /// Universal scaling constants
    pub scaling_constants: ScalingConstants,
    
    /// Pre-computed resonance templates by bit size
    pub resonance_templates: HashMap<u32, Vec<f64>>,
    
    /// Enhanced patterns for specific bit ranges
    pub bit_range_patterns: HashMap<(u32, u32), ScalingConstants>,
    
    /// Pre-computed harmonic basis functions
    pub harmonic_basis: Vec<Vec<f64>>,
}

impl From<&Basis> for SerializableBasis {
    fn from(basis: &Basis) -> Self {
        SerializableBasis {
            base: basis.base,
            scaling_constants: basis.scaling_constants.clone(),
            resonance_templates: basis.resonance_templates.clone(),
            bit_range_patterns: basis.bit_range_patterns.clone(),
            harmonic_basis: basis.harmonic_basis.clone(),
        }
    }
}

impl SerializableBasis {
    /// Save basis to JSON file
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| PatternError::ConfigError(format!("Failed to serialize basis: {}", e)))?;
        
        fs::write(path, json)
            .map_err(|e| PatternError::ConfigError(format!("Failed to write basis file: {}", e)))?;
        
        Ok(())
    }
    
    /// Load basis from JSON file
    pub fn load(path: &Path) -> Result<Self> {
        let json = fs::read_to_string(path)
            .map_err(|e| PatternError::ConfigError(format!("Failed to read basis file: {}", e)))?;
        
        let basis = serde_json::from_str(&json)
            .map_err(|e| PatternError::ConfigError(format!("Failed to parse basis file: {}", e)))?;
        
        Ok(basis)
    }
}

/// Generate and save basis files to the data directory
pub fn generate_basis_files() -> Result<()> {
    // Create data/basis directory if it doesn't exist
    let basis_dir = Path::new("data/basis");
    fs::create_dir_all(basis_dir)
        .map_err(|e| PatternError::ConfigError(format!("Failed to create basis directory: {}", e)))?;
    
    // Generate standard basis
    let standard_basis = Basis::new();
    let serializable_standard = SerializableBasis::from(&standard_basis);
    serializable_standard.save(&basis_dir.join("universal_basis.json"))?;
    
    // Generate enhanced basis with additional bit ranges
    let mut enhanced_basis = standard_basis;
    
    // Add bit-range specific patterns
    let bit_ranges = vec![
        (8, 16), (16, 24), (24, 32), (32, 48), (48, 64),
        (64, 80), (80, 96), (96, 112), (112, 128), (128, 144),
        (144, 160), (160, 176), (176, 192), (192, 208), (208, 224),
        (224, 240), (240, 256), (256, 288), (288, 320), (320, 384),
        (384, 448), (448, 512), (512, 768), (768, 1024),
    ];
    
    for (start, end) in bit_ranges {
        let mid_bits = (start + end) as f64 / 2.0;
        
        // Generate bit-range specific constants
        let constants = ScalingConstants {
            resonance_decay_alpha: 1.175 * (mid_bits / 50.0).powf(0.25),
            phase_coupling_beta: 0.199 * (1.0 + mid_bits.ln() / 10.0),
            scale_transition_gamma: 12.416 * mid_bits.sqrt(),
            interference_null_delta: (mid_bits / 100.0).sin().abs(),
            adelic_threshold_epsilon: 4.33 * (1.0 + mid_bits / 200.0),
            golden_ratio_phi: 1.618033988749895,
            tribonacci_tau: 1.839286755214161,
        };
        
        enhanced_basis.bit_range_patterns.insert((start, end), constants);
        
        // Add more resonance templates for key sizes
        for &bits in &[start, (start + end) / 2, end] {
            if !enhanced_basis.resonance_templates.contains_key(&bits) {
                // Generate template inline since the method is private
                let size = ((2.0_f64.powf(bits as f64 / 4.0) as usize).max(64)).min(1024);
                let mut template = vec![0.0; size];
                let phi = 1.618033988749895;
                
                for i in 0..size {
                    let x = i as f64 / size as f64;
                    template[i] = (phi * x * std::f64::consts::PI).sin() * (-x).exp();
                }
                enhanced_basis.resonance_templates.insert(bits, template);
            }
        }
    }
    
    let serializable_enhanced = SerializableBasis::from(&enhanced_basis);
    serializable_enhanced.save(&basis_dir.join("enhanced_basis.json"))?;
    
    Ok(())
}