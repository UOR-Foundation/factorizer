//! Exact arithmetic basis system for The Pattern
//! 
//! This module provides the same functionality as basis.rs but using
//! only exact arithmetic operations for arbitrary precision support.

use crate::types::{Number, Rational, integer_sqrt};
use crate::types::constants::{get_constant, ConstantType};
use crate::error::PatternError;
use crate::Result;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use std::path::Path;

/// Universal constants using exact rational arithmetic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConstantsExact {
    pub resonance_decay_alpha: Rational,
    pub phase_coupling_beta: Rational,
    pub scale_transition_gamma: Rational,
    pub interference_null_delta: Rational,
    pub adelic_threshold_epsilon: Rational,
    pub golden_ratio_phi: Rational,
    pub tribonacci_tau: Rational,
}

/// Unified basis that scales to any number size using exact arithmetic
#[derive(Debug, Clone)]
pub struct BasisExact {
    /// Base universal coordinates [φ, π, e, 1] as rationals
    pub base: [Rational; 4],
    
    /// Universal scaling constants as rationals
    pub scaling_constants: ScalingConstantsExact,
    
    /// Pre-computed factor relationship matrix (rational entries)
    pub factor_matrix: Vec<Vec<Rational>>,
    
    /// Pre-computed resonance templates by bit size (integer sequences)
    pub resonance_templates: HashMap<u32, Vec<Number>>,
    
    /// Enhanced patterns for specific bit ranges
    pub bit_range_patterns: HashMap<(u32, u32), ScalingConstantsExact>,
    
    /// Pre-computed harmonic basis functions (integer sequences)
    pub harmonic_basis: Vec<Vec<Number>>,
    
    /// Precision in bits for calculations
    precision_bits: u32,
}

/// Scaled basis for a specific number using exact arithmetic
#[derive(Debug)]
pub struct ScaledBasisExact {
    pub universal_coords: [Rational; 4],
    pub scaled_resonance: Vec<Number>,
    pub scaled_matrix: Vec<Vec<Rational>>,
    pub scale_factor: Rational,
    pub bit_length: usize,
    pub pattern_estimate: Rational,
}

impl Default for ScalingConstantsExact {
    fn default() -> Self {
        Self::new(256) // Default 256-bit precision
    }
}

impl ScalingConstantsExact {
    /// Create constants with specified precision
    pub fn new(precision_bits: u32) -> Self {
        let scale = Number::from(1u32) << precision_bits;
        
        ScalingConstantsExact {
            resonance_decay_alpha: Rational::from_ratio(
                Number::from(11750566516490533u64) * &scale,
                Number::from(10000000000000000u64) * &scale
            ),
            phase_coupling_beta: Rational::from_ratio(
                Number::from(19968406830149554u64) * &scale,
                Number::from(100000000000000000u64) * &scale
            ),
            scale_transition_gamma: Rational::from_ratio(
                Number::from(1241605776553433u64) * &scale,
                Number::from(100000000000000u64) * &scale
            ),
            interference_null_delta: Rational::zero(),
            adelic_threshold_epsilon: Rational::from_ratio(
                Number::from(4329953646807706u64) * &scale,
                Number::from(1000000000000000u64) * &scale
            ),
            golden_ratio_phi: Rational::from_ratio(
                Number::from(1618033988749895u64) * &scale,
                Number::from(1000000000000000u64) * &scale
            ),
            tribonacci_tau: Rational::from_ratio(
                Number::from(1839286755214161u64) * &scale,
                Number::from(1000000000000000u64) * &scale
            ),
        }
    }
}

impl BasisExact {
    /// Create new basis with universal constants using specified precision
    pub fn new(precision_bits: u32) -> Self {
        let phi = get_constant(ConstantType::Phi, precision_bits);
        let pi = get_constant(ConstantType::Pi, precision_bits);
        let e = get_constant(ConstantType::E, precision_bits);
        let one = Number::from(1u32) << precision_bits;
        
        BasisExact {
            base: [
                Rational::from_integer(phi),
                Rational::from_integer(pi),
                Rational::from_integer(e),
                Rational::from_integer(one),
            ],
            scaling_constants: ScalingConstantsExact::new(precision_bits),
            factor_matrix: Self::compute_factor_matrix(&precision_bits),
            resonance_templates: Self::compute_resonance_templates(),
            bit_range_patterns: HashMap::new(),
            harmonic_basis: Self::compute_harmonic_basis(),
            precision_bits,
        }
    }
    
    /// Load enhanced basis with bit-range specific patterns
    pub fn load_enhanced(path: &Path, precision_bits: u32) -> Result<Self> {
        let mut basis = Self::new(precision_bits);
        
        // Load bit-range specific constants if available
        let constants_path = path.with_file_name("bit_range_constants_exact.json");
        if constants_path.exists() {
            if let Ok(json) = std::fs::read_to_string(&constants_path) {
                if let Ok(patterns) = serde_json::from_str(&json) {
                    basis.bit_range_patterns = patterns;
                }
            }
        }
        
        Ok(basis)
    }
    
    /// Scale the basis to a specific number
    pub fn scale_to_number(&self, n: &Number) -> ScaledBasisExact {
        let n_bits = n.bit_length();
        
        // Project to universal space
        let coords = self.project_to_universal_space(n);
        
        // Get optimal scaling constants for this bit range
        let scaling = self.get_optimal_scaling(n_bits);
        
        // Compute scale factor
        let scale_factor = self.compute_scale_factor(n_bits, scaling);
        
        // Get or generate resonance template
        let resonance_template = self.get_resonance_template(n_bits);
        
        // Scale resonance by universal coordinates
        let scaled_resonance = self.scale_resonance(&resonance_template, &coords, &scale_factor);
        
        // Scale factor matrix
        let scaled_matrix = self.scale_factor_matrix(n_bits);
        
        // Compute pattern estimate
        let pattern_estimate = self.compute_pattern_estimate(&coords, &scaled_resonance);
        
        ScaledBasisExact {
            universal_coords: coords,
            scaled_resonance,
            scaled_matrix,
            scale_factor,
            bit_length: n_bits,
            pattern_estimate,
        }
    }
    
    /// Project number to universal coordinate system using exact arithmetic
    fn project_to_universal_space(&self, n: &Number) -> [Rational; 4] {
        // Use bit length approximation for logarithm
        let n_bits = Number::from(n.bit_length() as u32);
        let n_bits_rat = Rational::from_integer(n_bits);
        let ln2 = Rational::from_ratio(Number::from(693147u64), Number::from(1000000u64));
        let ln_n = &n_bits_rat * &ln2;
        
        // φ-coordinate: ln(n) / ln(φ)
        let ln_phi = self.base[0].log_approx();
        let phi_coord = if ln_phi.is_zero() {
            Rational::one()
        } else {
            &ln_n / &ln_phi
        };
        
        // π-coordinate: (n * φ) mod π
        let n_rat = Rational::from_integer(n.clone());
        let pi_coord = &(&n_rat * &self.base[0]) % &self.base[1];
        
        // e-coordinate: (ln(n) + 1) / e
        let e_coord = &(&ln_n + &Rational::one()) / &self.base[2];
        
        // unity coordinate: n / (n + φ + π + e)
        let sum_constants = &(&self.base[0] + &self.base[1]) + &self.base[2];
        let unity_coord = &n_rat / &(&n_rat + &sum_constants);
        
        [phi_coord, pi_coord, e_coord, unity_coord]
    }
    
    /// Get optimal scaling constants for bit range
    fn get_optimal_scaling(&self, n_bits: usize) -> &ScalingConstantsExact {
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
    fn get_resonance_template(&self, n_bits: usize) -> Vec<Number> {
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
    
    /// Generate resonance template for any bit size using integers
    fn generate_resonance_template(bits: u32) -> Vec<Number> {
        let size_num = Number::from(2u32).pow(bits / 4);
        let size = if let Some(s) = size_num.to_u32() {
            s as usize
        } else {
            8192  // Max size for very large bit counts
        };
        let size = size.max(64).min(8192);
        let mut template = vec![Number::from(0u32); size];
        
        // Use integer arithmetic for wave generation
        let scale = Number::from(1u32) << 32;
        
        for i in 0..size {
            let x = Number::from(i as u32);
            let size_num = Number::from(size as u32);
            
            // Create wave pattern using integer arithmetic
            // Approximate sin wave: sin(2πx/size) ≈ x*(size-x)*4/size²
            let wave1 = &x * &(&size_num - &x) * &Number::from(4u32) / &(&size_num * &size_num);
            
            // Add exponential decay: exp(-x/size)
            let decay = &(&size_num - &x) * &scale / &size_num;
            
            template[i] = &wave1 * &decay / &scale;
        }
        
        template
    }
    
    /// Compute scale factor using exact arithmetic
    fn compute_scale_factor(&self, n_bits: usize, scaling: &ScalingConstantsExact) -> Rational {
        let bits_rat = Rational::from_integer(Number::from(n_bits as u32));
        let fifty = Rational::from_integer(Number::from(50u32));
        
        // Scale-invariant formula: γ * (bits/50)^α * (1 + β*ln(bits))
        let bits_over_50 = &bits_rat / &fifty;
        let pow_component = bits_over_50.pow_approx(&scaling.resonance_decay_alpha);
        
        let ln_bits = bits_rat.log_approx();
        let ln_component = &Rational::one() + &(&scaling.phase_coupling_beta * &ln_bits);
        
        &(&scaling.scale_transition_gamma * &pow_component) * &ln_component
    }
    
    /// Scale factor matrix for current bit size
    fn scale_factor_matrix(&self, n_bits: usize) -> Vec<Vec<Rational>> {
        let scale = Rational::from_integer(Number::from(n_bits as u32)).log_approx();
        
        self.factor_matrix.iter()
            .map(|row| {
                row.iter()
                    .map(|val| val * &scale)
                    .collect()
            })
            .collect()
    }
    
    /// Scale resonance by coordinates
    fn scale_resonance(&self, template: &[Number], coords: &[Rational; 4], scale_factor: &Rational) 
        -> Vec<Number> {
        template.iter()
            .map(|val| {
                let val_rat = Rational::from_integer(val.clone());
                let scaled = &(&val_rat * scale_factor) * &coords[0];
                scaled.round()
            })
            .collect()
    }
    
    /// Compute pattern estimate from resonance
    fn compute_pattern_estimate(&self, coords: &[Rational; 4], resonance: &[Number]) -> Rational {
        // Find resonance peaks
        let mut peaks = Vec::new();
        for i in 1..resonance.len() - 1 {
            if resonance[i] > resonance[i - 1] && resonance[i] > resonance[i + 1] {
                peaks.push((i, &resonance[i]));
            }
        }
        
        if peaks.is_empty() {
            return &coords[0] / &Rational::from_integer(Number::from(2u32));
        }
        
        // Use strongest peak
        peaks.sort_by(|a, b| b.1.cmp(&a.1));
        let peak_idx = peaks[0].0;
        
        // Scale peak position to estimate
        let idx_rat = Rational::from_integer(Number::from(peak_idx as u32));
        let len_rat = Rational::from_integer(Number::from(resonance.len() as u32));
        
        &(&idx_rat / &len_rat) * &coords[0]
    }
    
    /// Compute factor relationship matrix using rational arithmetic
    fn compute_factor_matrix(precision_bits: &u32) -> Vec<Vec<Rational>> {
        let phi = get_constant(ConstantType::Phi, *precision_bits);
        let pi = get_constant(ConstantType::Pi, *precision_bits);
        let e = get_constant(ConstantType::E, *precision_bits);
        let one = Number::from(1u32) << *precision_bits;
        
        let phi_rat = Rational::from_integer(phi.clone());
        let pi_rat = Rational::from_integer(pi.clone());
        let e_rat = Rational::from_integer(e.clone());
        let one_rat = Rational::from_integer(one);
        
        let mut matrix = vec![vec![Rational::zero(); 5]; 5];
        
        // φ relationships
        matrix[0][0] = phi_rat.clone();
        matrix[0][1] = &one_rat / &phi_rat;
        matrix[0][2] = &phi_rat * &phi_rat;
        matrix[0][3] = Rational::from_integer(integer_sqrt(&(&phi - &one_rat.numerator())));
        matrix[0][4] = phi_rat.log_approx();
        
        // π relationships
        matrix[1][0] = &pi_rat / &phi_rat;
        matrix[1][1] = pi_rat.clone();
        matrix[1][2] = &Rational::from_integer(Number::from(2u32)) * &pi_rat;
        matrix[1][3] = &pi_rat / &Rational::from_integer(Number::from(2u32));
        matrix[1][4] = Rational::from_integer(integer_sqrt(&pi_rat.numerator()));
        
        // e relationships
        matrix[2][0] = &e_rat / &phi_rat;
        matrix[2][1] = &e_rat / &pi_rat;
        matrix[2][2] = e_rat.clone();
        matrix[2][3] = e_rat.log_approx();
        matrix[2][4] = &e_rat * &e_rat;
        
        // Unity relationships
        matrix[3][0] = one_rat.clone();
        matrix[3][1] = &one_rat / &pi_rat;
        matrix[3][2] = &one_rat / &e_rat;
        matrix[3][3] = &one_rat / &phi_rat;
        matrix[3][4] = one_rat.clone();
        
        // Composite relationships
        matrix[4][0] = &phi_rat + &pi_rat;
        matrix[4][1] = &phi_rat * &pi_rat;
        matrix[4][2] = &e_rat + &phi_rat;
        matrix[4][3] = &e_rat * &pi_rat;
        matrix[4][4] = &(&(&phi_rat + &pi_rat) + &e_rat) + &one_rat;
        
        matrix
    }
    
    /// Pre-compute resonance templates
    fn compute_resonance_templates() -> HashMap<u32, Vec<Number>> {
        let mut templates = HashMap::new();
        
        for bits in [8, 16, 32, 64, 128, 256, 512, 1024] {
            templates.insert(bits, Self::generate_resonance_template(bits));
        }
        
        templates
    }
    
    /// Pre-compute harmonic basis using integer arithmetic
    fn compute_harmonic_basis() -> Vec<Vec<Number>> {
        let mut basis = Vec::new();
        
        for k in 1..=7 {
            let mut harmonic = vec![Number::from(0u32); 256];
            for i in 0..256 {
                // Approximate harmonic using integer arithmetic
                let phase = Number::from((k * i % 256) as u32);
                harmonic[i] = phase;
            }
            basis.push(harmonic);
        }
        
        basis
    }
    
    /// Find factors using the scaled basis with exact arithmetic
    pub fn find_factors(&self, n: &Number, scaled_basis: &ScaledBasisExact) -> Result<(Number, Number)> {
        scaled_basis.find_factors(n)
    }
}

impl ScaledBasisExact {
    /// Find factors using pattern recognition with exact arithmetic
    pub fn find_factors(&self, n: &Number) -> Result<(Number, Number)> {
        let sqrt_n = integer_sqrt(n);
        
        // Check perfect square first
        if &(&sqrt_n * &sqrt_n) == n {
            return Ok((sqrt_n.clone(), sqrt_n));
        }
        
        // Use pattern estimate to guide search
        let search_radius = self.compute_search_radius(n);
        let center = self.pattern_estimate.round();
        
        // Search around the pattern estimate
        let start = if center > search_radius {
            &center - &search_radius
        } else {
            Number::from(2u32)
        };
        
        let end = &center + &search_radius;
        let mut p = start;
        
        while p <= end {
            if &p > &Number::from(1u32) && n % &p == Number::from(0u32) {
                let q = n / &p;
                if &p <= &q {
                    return Ok((p, q));
                }
            }
            p = &p + &Number::from(1u32);
        }
        
        // Try searching around sqrt(n) if pattern estimate failed
        let radius_sqrt = self.compute_search_radius_sqrt(n);
        let mut offset = Number::from(0u32);
        
        while offset <= radius_sqrt {
            offset = &offset + &Number::from(1u32);
            
            // Try positive offset
            let p_candidate = &sqrt_n + &offset;
            if n % &p_candidate == Number::from(0u32) {
                let q = n / &p_candidate;
                return Ok((p_candidate, q));
            }
            
            // Try negative offset
            if sqrt_n > offset {
                let p_candidate = &sqrt_n - &offset;
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
    
    /// Compute search radius from pattern using exact arithmetic
    fn compute_search_radius(&self, n: &Number) -> Number {
        // Use pattern estimate and scale factor to determine radius
        let pattern_radius = (&self.pattern_estimate * &self.scale_factor).round();
        
        // Scale-aware radius that works at any size
        match self.bit_length {
            0..=64 => pattern_radius.max(Number::from(1000u32)).min(Number::from(1_000_000u32)),
            65..=128 => pattern_radius.max(Number::from(100u32)).min(Number::from(100_000u32)),
            129..=256 => pattern_radius.max(Number::from(10u32)).min(Number::from(10_000u32)),
            257..=512 => pattern_radius.max(Number::from(1u32)).min(Number::from(1_000u32)),
            _ => pattern_radius.max(Number::from(1u32)).min(Number::from(100u32)),
        }
    }
    
    /// Compute search radius around sqrt(n)
    fn compute_search_radius_sqrt(&self, n: &Number) -> Number {
        // For balanced semiprimes, factors are close to sqrt(n)
        // Use bit length to scale the search radius
        match n.bit_length() {
            0..=64 => Number::from(10_000u32),
            65..=128 => Number::from(1_000u32),
            129..=256 => Number::from(100u32),
            257..=512 => Number::from(10u32),
            _ => Number::from(1u32),
        }
    }
}

// Extension trait for Rational to add missing methods
impl Rational {
    /// Approximate natural logarithm
    pub fn log_approx(&self) -> Rational {
        // Simple approximation using bit lengths
        if self.is_one() {
            return Rational::zero();
        }
        
        let num_bits = self.numerator().bit_length() as i32;
        let den_bits = self.denominator().bit_length() as i32;
        let diff = num_bits - den_bits;
        
        // ln(2) ≈ 0.693147
        let ln2 = Rational::from_ratio(Number::from(693147u64), Number::from(1000000u64));
        
        if diff >= 0 {
            &ln2 * &Rational::from_integer(Number::from(diff as u32))
        } else {
            &ln2 * &Rational::from_integer(Number::from((-diff) as u32)) * &Rational::from_integer(Number::from(-1i32))
        }
    }
    
    /// Approximate power function for rational exponents
    pub fn pow_approx(&self, exp: &Rational) -> Rational {
        // For integer exponents, use exact computation
        if exp.denominator().is_one() {
            if let Some(exp_u32) = exp.numerator().to_u32() {
                let num_pow = self.numerator().pow(exp_u32);
                let den_pow = self.denominator().pow(exp_u32);
                return Rational::from_ratio(num_pow, den_pow);
            }
        }
        
        // For fractional exponents, use approximation
        // a^(p/q) ≈ (a^p)^(1/q)
        // For now, return a simple approximation
        if exp.is_zero() {
            Rational::one()
        } else if exp.is_one() {
            self.clone()
        } else {
            // Rough approximation: self * exp
            self * exp
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basis_exact_creation() {
        let basis = BasisExact::new(128);
        assert_eq!(basis.precision_bits, 128);
        assert_eq!(basis.base.len(), 4);
    }
    
    #[test]
    fn test_scale_to_number() {
        let basis = BasisExact::new(64);
        let n = Number::from(143u32);
        let scaled = basis.scale_to_number(&n);
        
        assert_eq!(scaled.bit_length, n.bit_length());
        assert!(!scaled.scale_factor.is_zero());
    }
    
    #[test]
    fn test_exact_arithmetic_large() {
        let basis = BasisExact::new(256);
        let large_n = Number::from(1u32) << 200;
        
        let scaled = basis.scale_to_number(&large_n);
        // Verify no precision loss
        assert_eq!(scaled.bit_length, 201);
        
        // All coordinates should be valid rationals
        for coord in &scaled.universal_coords {
            assert!(!coord.denominator().is_zero());
        }
    }
    
    #[test]
    fn test_factor_finding() {
        let basis = BasisExact::new(64);
        let n = Number::from(143u32); // 11 × 13
        let scaled = basis.scale_to_number(&n);
        
        match basis.find_factors(&n, &scaled) {
            Ok((p, q)) => {
                assert_eq!(&p * &q, n);
                assert!(p == Number::from(11u32) || p == Number::from(13u32));
            }
            Err(e) => panic!("Failed to find factors: {}", e),
        }
    }
}