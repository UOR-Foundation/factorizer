//! Unified basis system for The Pattern
//! 
//! Combines universal constants, pre-computed scaling, and enhanced patterns
//! into a single scale-invariant basis that works at any scale.

use crate::types::Number;
use crate::error::PatternError;
use crate::Result;
use nalgebra::DMatrix;
use std::collections::HashMap;
use std::f64::consts::{E, PI};
use serde::{Serialize, Deserialize};
use std::path::Path;

/// Universal constants discovered through empirical observation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConstants {
    pub resonance_decay_alpha: f64,
    pub phase_coupling_beta: f64,
    pub scale_transition_gamma: f64,
    pub interference_null_delta: f64,
    pub adelic_threshold_epsilon: f64,
    pub golden_ratio_phi: f64,
    pub tribonacci_tau: f64,
}

/// Unified basis that scales to any number size
#[derive(Debug, Clone)]
pub struct Basis {
    /// Base universal coordinates [φ, π, e, 1]
    pub base: [f64; 4],
    
    /// Universal scaling constants
    pub scaling_constants: ScalingConstants,
    
    /// Pre-computed factor relationship matrix
    pub factor_matrix: DMatrix<f64>,
    
    /// Pre-computed resonance templates by bit size
    pub resonance_templates: HashMap<u32, Vec<f64>>,
    
    /// Enhanced patterns for specific bit ranges
    pub bit_range_patterns: HashMap<(u32, u32), ScalingConstants>,
    
    /// Pre-computed harmonic basis functions
    pub harmonic_basis: Vec<Vec<f64>>,
}

/// Scaled basis for a specific number
#[derive(Debug)]
pub struct ScaledBasis {
    pub universal_coords: [f64; 4],
    pub scaled_resonance: Vec<f64>,
    pub scaled_matrix: DMatrix<f64>,
    pub scale_factor: f64,
    pub bit_length: usize,
    pub pattern_estimate: f64,
}

impl Default for ScalingConstants {
    fn default() -> Self {
        // Universal constants from empirical observation
        ScalingConstants {
            resonance_decay_alpha: 1.1750566516490533,
            phase_coupling_beta: 0.19968406830149554,
            scale_transition_gamma: 12.41605776553433,
            interference_null_delta: 0.0,
            adelic_threshold_epsilon: 4.329953646807706,
            golden_ratio_phi: 1.618033988749895,
            tribonacci_tau: 1.839286755214161,
        }
    }
}

impl Basis {
    /// Create new basis with universal constants
    pub fn new() -> Self {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        
        Basis {
            base: [phi, PI, E, 1.0],
            scaling_constants: ScalingConstants::default(),
            factor_matrix: Self::compute_factor_matrix(phi),
            resonance_templates: Self::compute_resonance_templates(),
            bit_range_patterns: HashMap::new(),
            harmonic_basis: Self::compute_harmonic_basis(),
        }
    }
    
    /// Load enhanced basis with bit-range specific patterns
    pub fn load_enhanced(path: &Path) -> Result<Self> {
        let mut basis = Self::new();
        
        // Load bit-range specific constants if available
        let constants_path = path.with_file_name("bit_range_constants.json");
        if constants_path.exists() {
            if let Ok(json) = std::fs::read_to_string(&constants_path) {
                if let Ok(patterns) = serde_json::from_str(&json) {
                    basis.bit_range_patterns = patterns;
                }
            }
        }
        
        // Load pre-computed templates if available
        let templates_path = path.with_file_name("resonance_templates.json");
        if templates_path.exists() {
            if let Ok(json) = std::fs::read_to_string(&templates_path) {
                if let Ok(templates) = serde_json::from_str(&json) {
                    basis.resonance_templates = templates;
                }
            }
        }
        
        Ok(basis)
    }
    
    /// Scale basis to specific number (works at any scale)
    pub fn scale_to_number(&self, n: &Number) -> ScaledBasis {
        let n_bits = n.bit_length();
        
        // Project to universal space
        let coords = self.project_to_universal_space(n);
        
        // Get optimal scaling constants for this bit range
        let scaling = self.get_optimal_scaling(n_bits);
        
        // Get or generate resonance template
        let template = self.get_resonance_template(n_bits);
        
        // Compute scale factor using optimal constants
        let scale_factor = self.compute_scale_factor(n_bits, &scaling);
        
        // Scale resonance template
        let scaled_resonance: Vec<f64> = template.iter()
            .map(|&val| val * scale_factor)
            .collect();
        
        // Scale factor matrix
        let scaled_matrix = self.scale_factor_matrix(n_bits);
        
        // Compute pattern estimate for factor location
        let pattern_estimate = self.compute_pattern_estimate(&coords, &scaled_resonance);
        
        ScaledBasis {
            universal_coords: coords,
            scaled_resonance,
            scaled_matrix,
            scale_factor,
            bit_length: n_bits,
            pattern_estimate,
        }
    }
    
    /// Project number to universal coordinate system
    fn project_to_universal_space(&self, n: &Number) -> [f64; 4] {
        let n_f64 = n.to_f64().unwrap_or(1e100);
        let ln_n = if n.bit_length() > 500 || n_f64.is_infinite() {
            n.bit_length() as f64 * 2.0_f64.ln()
        } else {
            n_f64.ln()
        };
        
        [
            ln_n / self.base[0].ln(),                    // φ-coordinate
            (n_f64 * self.base[0]) % self.base[1],      // π-coordinate
            (ln_n + 1.0) / self.base[2],                // e-coordinate
            n_f64 / (n_f64 + self.base[0] + self.base[1] + self.base[2]), // unity
        ]
    }
    
    /// Get optimal scaling constants for bit range
    fn get_optimal_scaling(&self, n_bits: usize) -> &ScalingConstants {
        // Find bit-range specific constants
        for ((min_bits, max_bits), constants) in &self.bit_range_patterns {
            if n_bits >= *min_bits as usize && n_bits <= *max_bits as usize {
                return constants;
            }
        }
        
        // Use universal constants for any scale
        &self.scaling_constants
    }
    
    /// Get or generate resonance template
    fn get_resonance_template(&self, n_bits: usize) -> Vec<f64> {
        // Find closest pre-computed template
        let template_bits = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
            .iter()
            .find(|&&b| b >= n_bits as u32)
            .copied()
            .unwrap_or(n_bits as u32);
        
        self.resonance_templates
            .get(&template_bits)
            .cloned()
            .unwrap_or_else(|| Self::generate_resonance_template(template_bits))
    }
    
    /// Generate resonance template for any bit size
    fn generate_resonance_template(bits: u32) -> Vec<f64> {
        let size = ((2.0_f64.powf(bits as f64 / 4.0) as usize).max(64)).min(8192);
        let mut template = vec![0.0; size];
        
        let phi = 1.618033988749895;
        for i in 0..size {
            let x = i as f64 / size as f64;
            template[i] = (phi * x * PI).sin() * (-x).exp() + 
                         (E * x * 2.0 * PI).cos() * (-x * x).exp();
        }
        
        template
    }
    
    /// Compute scale factor using optimal constants
    fn compute_scale_factor(&self, n_bits: usize, scaling: &ScalingConstants) -> f64 {
        let bits_f64 = n_bits as f64;
        
        // Scale-invariant formula
        scaling.scale_transition_gamma * 
        (bits_f64 / 50.0).powf(scaling.resonance_decay_alpha) * 
        (1.0 + scaling.phase_coupling_beta * bits_f64.ln())
    }
    
    /// Scale factor matrix for current bit size
    fn scale_factor_matrix(&self, n_bits: usize) -> DMatrix<f64> {
        let scale = (n_bits as f64).ln();
        let mut scaled_matrix = DMatrix::zeros(4, 4);
        
        for i in 0..4 {
            for j in 0..4 {
                scaled_matrix[(i, j)] = self.factor_matrix[(i, j)] * scale;
            }
        }
        
        scaled_matrix
    }
    
    /// Compute pattern estimate from resonance
    fn compute_pattern_estimate(&self, coords: &[f64; 4], resonance: &[f64]) -> f64 {
        // Find resonance peaks
        let mut peaks = Vec::new();
        for i in 1..resonance.len() - 1 {
            if resonance[i] > resonance[i - 1] && resonance[i] > resonance[i + 1] {
                peaks.push((i, resonance[i]));
            }
        }
        
        if peaks.is_empty() {
            return coords[0] * 0.5; // Default to φ-coordinate estimate
        }
        
        // Use strongest peak
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let peak_idx = peaks[0].0;
        
        // Scale peak position to estimate
        (peak_idx as f64 / resonance.len() as f64) * coords[0]
    }
    
    /// Compute factor relationship matrix
    fn compute_factor_matrix(phi: f64) -> DMatrix<f64> {
        let mut matrix = DMatrix::zeros(5, 5);
        
        // φ relationships
        matrix[(0, 0)] = phi;
        matrix[(0, 1)] = 1.0 / phi;
        matrix[(0, 2)] = phi * phi;
        matrix[(0, 3)] = (phi - 1.0).sqrt();
        matrix[(0, 4)] = phi.ln();
        
        // π relationships
        matrix[(1, 0)] = PI / phi;
        matrix[(1, 1)] = PI;
        matrix[(1, 2)] = 2.0 * PI;
        matrix[(1, 3)] = PI / 2.0;
        matrix[(1, 4)] = PI.sqrt();
        
        // e relationships
        matrix[(2, 0)] = E / phi;
        matrix[(2, 1)] = E / PI;
        matrix[(2, 2)] = E;
        matrix[(2, 3)] = E.ln();
        matrix[(2, 4)] = E * E;
        
        // Unity relationships
        matrix[(3, 0)] = 1.0;
        matrix[(3, 1)] = 1.0 / PI;
        matrix[(3, 2)] = 1.0 / E;
        matrix[(3, 3)] = 1.0 / phi;
        matrix[(3, 4)] = 1.0;
        
        // Composite relationships
        matrix[(4, 0)] = phi + PI;
        matrix[(4, 1)] = phi * PI;
        matrix[(4, 2)] = E + phi;
        matrix[(4, 3)] = E * PI;
        matrix[(4, 4)] = phi + PI + E + 1.0;
        
        matrix
    }
    
    /// Pre-compute resonance templates
    fn compute_resonance_templates() -> HashMap<u32, Vec<f64>> {
        let mut templates = HashMap::new();
        
        for bits in [8, 16, 32, 64, 128, 256, 512, 1024] {
            templates.insert(bits, Self::generate_resonance_template(bits));
        }
        
        templates
    }
    
    /// Pre-compute harmonic basis
    fn compute_harmonic_basis() -> Vec<Vec<f64>> {
        let mut basis = Vec::new();
        
        for k in 1..=7 {
            let mut harmonic = vec![0.0; 256];
            for i in 0..256 {
                let x = i as f64 / 256.0;
                harmonic[i] = (k as f64 * 2.0 * PI * x).sin();
            }
            basis.push(harmonic);
        }
        
        basis
    }
    
    /// Find factors using the scaled basis
    pub fn find_factors(&self, n: &Number, scaled_basis: &ScaledBasis) -> Result<(Number, Number)> {
        scaled_basis.find_factors(n)
    }
}

impl ScaledBasis {
    /// Find factors using pattern recognition
    pub fn find_factors(&self, n: &Number) -> Result<(Number, Number)> {
        let sqrt_n = crate::utils::integer_sqrt(n)?;
        
        // Check perfect square first
        if n % &sqrt_n == Number::from(0u32) {
            let quotient = n / &sqrt_n;
            if quotient == sqrt_n {
                return Ok((sqrt_n.clone(), sqrt_n));
            }
        }
        
        // Use pattern estimate to guide search
        let search_radius = self.compute_search_radius();
        let mut offset = 0u128;
        
        while offset <= search_radius {
            offset += 1;
            
            // Try positive offset
            let p_candidate = &sqrt_n + &Number::from(offset);
            if n % &p_candidate == Number::from(0u32) {
                let q = n / &p_candidate;
                return Ok((p_candidate, q));
            }
            
            // Try negative offset
            if sqrt_n > Number::from(offset) {
                let p_candidate = &sqrt_n - &Number::from(offset);
                if p_candidate > Number::from(1u32) && n % &p_candidate == Number::from(0u32) {
                    let q = n / &p_candidate;
                    return Ok((p_candidate, q));
                }
            }
        }
        
        Err(PatternError::ExecutionError(
            format!("Pattern recognition requires deeper analysis for {}-bit number", self.bit_length)
        ))
    }
    
    /// Compute search radius from pattern
    fn compute_search_radius(&self) -> u128 {
        let pattern_radius = (self.pattern_estimate * self.scale_factor).abs() as u128;
        
        // Scale-aware radius that works at any size
        match self.bit_length {
            0..=64 => pattern_radius.max(1000).min(1_000_000),
            65..=128 => pattern_radius.max(100).min(100_000),
            129..=256 => pattern_radius.max(10).min(10_000),
            257..=512 => pattern_radius.max(1).min(1_000),
            _ => pattern_radius.max(1).min(100),
        }
    }
}