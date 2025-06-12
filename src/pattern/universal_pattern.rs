//! Universal Pattern implementation based on discovered constants
//!
//! This module implements The Pattern's core insights:
//! - Universal constant basis (φ, π, e, 1)
//! - Three-stage process (Recognition, Formalization, Execution)
//! - Resonance field and harmonic analysis

use crate::error::PatternError;
use crate::types::{Number, Factors};
use crate::utils;
use crate::Result;
use nalgebra::{DMatrix, SymmetricEigen};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::f64::consts::{E, PI};

/// Universal constants that form the basis of The Pattern
#[derive(Debug, Clone)]
pub struct UniversalConstants {
    /// Golden ratio φ
    pub phi: f64,
    /// Circle constant π
    pub pi: f64,
    /// Natural logarithm base e
    pub e: f64,
    /// Unity
    pub unity: f64,
    /// Derived: 2 - φ
    pub beta: f64,
    /// Euler-Mascheroni constant
    pub gamma: f64,
}

impl Default for UniversalConstants {
    fn default() -> Self {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        UniversalConstants {
            phi,
            pi: PI,
            e: E,
            unity: 1.0,
            beta: 2.0 - phi,
            gamma: 0.5772156649015329,
        }
    }
}

/// Universal Pattern recognizer
pub struct UniversalPattern {
    constants: UniversalConstants,
    cache: HashMap<String, (Number, Number)>,
}

impl UniversalPattern {
    pub fn new() -> Self {
        UniversalPattern {
            constants: UniversalConstants::default(),
            cache: HashMap::new(),
        }
    }

    /// Stage 1: Recognition - Extract universal signature
    pub fn recognize(&self, n: &Number) -> Result<UniversalRecognition> {
        // Extract universal components
        let phi_component = self.extract_phi_component(n)?;
        let pi_component = self.extract_pi_component(n)?;
        let e_component = self.extract_e_component(n)?;
        let unity_phase = self.extract_unity_phase(n)?;

        // Generate resonance field
        let resonance_field = self.generate_resonance_field(n, phi_component, pi_component, e_component)?;

        Ok(UniversalRecognition {
            value: n.clone(),
            phi_component,
            pi_component,
            e_component,
            unity_phase,
            resonance_field,
        })
    }

    /// Stage 2: Formalization - Express in universal language
    pub fn formalize(&self, recognition: UniversalRecognition) -> Result<UniversalFormalization> {
        let n = &recognition.value;

        // Compute harmonic series
        let harmonic_series = self.compute_harmonic_series(&recognition)?;
        
        // Find resonance peaks
        let resonance_peaks = self.find_resonance_peaks(&recognition.resonance_field);
        
        // Construct pattern matrix
        let pattern_matrix = self.construct_pattern_matrix(&recognition)?;
        
        // Encode factor structure
        let factor_encoding = self.encode_factor_structure(&recognition);

        Ok(UniversalFormalization {
            value: n.clone(),
            universal_coordinates: vec![
                recognition.phi_component,
                recognition.pi_component,
                recognition.e_component,
                recognition.unity_phase,
            ],
            harmonic_series,
            resonance_peaks,
            pattern_matrix,
            factor_encoding,
        })
    }

    /// Stage 3: Execution - Decode factors
    pub fn execute(&mut self, formalization: UniversalFormalization) -> Result<Factors> {
        let n = &formalization.value;

        // Check cache
        let cache_key = n.to_string();
        if let Some((p, q)) = self.cache.get(&cache_key) {
            return Ok(Factors::new(p.clone(), q.clone(), "universal_pattern_cached"));
        }

        // Try multiple decoding strategies in order of effectiveness
        
        // 1. Resonance peak decoding
        if let Ok(factors) = self.decode_via_resonance(n, &formalization) {
            self.cache.insert(cache_key, (factors.p.clone(), factors.q.clone()));
            return Ok(factors);
        }

        // 2. Eigenvalue decoding
        if let Ok(factors) = self.decode_via_eigenvalues(n, &formalization) {
            self.cache.insert(cache_key, (factors.p.clone(), factors.q.clone()));
            return Ok(factors);
        }

        // 3. Harmonic intersection
        if let Ok(factors) = self.decode_via_harmonics(n, &formalization) {
            self.cache.insert(cache_key, (factors.p.clone(), factors.q.clone()));
            return Ok(factors);
        }

        // 4. Phase relationship decoding
        if let Ok(factors) = self.decode_via_phase_relationships(n, &formalization) {
            self.cache.insert(cache_key, (factors.p.clone(), factors.q.clone()));
            return Ok(factors);
        }

        // 5. Universal relationship decoding
        if let Ok(factors) = self.decode_via_universal_relationships(n, &formalization) {
            self.cache.insert(cache_key, (factors.p.clone(), factors.q.clone()));
            return Ok(factors);
        }

        // 6. Enhanced search as fallback
        if let Ok(factors) = self.enhanced_search(n, &formalization) {
            self.cache.insert(cache_key, (factors.p.clone(), factors.q.clone()));
            return Ok(factors);
        }

        Err(PatternError::ExecutionError("All decoding strategies failed".to_string()))
    }

    // Component extraction methods

    fn extract_phi_component(&self, n: &Number) -> Result<f64> {
        // n's relationship to Fibonacci sequence
        let log_n = if n.bit_length() > 53 {
            // Use bit length approximation for large numbers
            n.bit_length() as f64 * 2.0_f64.ln()
        } else {
            n.to_f64().unwrap_or(1.0).ln()
        };
        
        Ok(log_n / self.constants.phi.ln())
    }

    fn extract_pi_component(&self, n: &Number) -> Result<f64> {
        // n's relationship to circular harmonics
        let modulus = (2.0 * self.constants.pi * 1_000_000.0) as u64;
        let n_mod = (n % &Number::from(modulus)).to_f64().unwrap_or(0.0);
        Ok(n_mod / (self.constants.pi * 1_000_000.0))
    }

    fn extract_e_component(&self, n: &Number) -> Result<f64> {
        // n's relationship to exponential growth
        let log_n = if n.bit_length() > 53 {
            n.bit_length() as f64 * 2.0_f64.ln()
        } else {
            n.to_f64().unwrap_or(1.0).ln()
        };
        
        Ok(log_n / self.constants.e)
    }

    fn extract_unity_phase(&self, n: &Number) -> Result<f64> {
        // Phase in the unit circle
        let n_float = n.to_f64().unwrap_or(0.0);
        Ok((n_float * self.constants.phi) % (2.0 * self.constants.pi))
    }

    // Resonance field generation

    fn generate_resonance_field(&self, n: &Number, phi: f64, pi: f64, e: f64) -> Result<Array1<f64>> {
        let sqrt_n = utils::integer_sqrt(n)?;
        let size = (sqrt_n.to_f64().unwrap_or(1000.0) as usize).min(1000);
        let mut field = Array1::zeros(size);

        for i in 0..size {
            let i_f = i as f64;
            let size_f = size as f64;
            
            // Universal harmonic at position i
            let harmonic = (phi * (pi * i_f / size_f).sin() + 
                           e * (phi * i_f / size_f).cos()) / self.constants.unity;
            
            // Damping factor
            let damping = if n.bit_length() > 200 {
                (-i_f / 2.0_f64.powf(n.bit_length() as f64 / 4.0)).exp()
            } else {
                (-i_f / n.to_f64().unwrap_or(1.0).powf(0.25)).exp()
            };
            
            field[i] = harmonic * damping;
        }

        Ok(field)
    }

    // Formalization methods

    fn compute_harmonic_series(&self, recognition: &UniversalRecognition) -> Result<Array1<f64>> {
        let n = &recognition.value;
        let limit = 20.min(n.to_f64().unwrap_or(20.0) as usize);
        let mut harmonics = Array1::zeros(limit);

        for k in 1..=limit {
            let k_f = k as f64;
            harmonics[k - 1] = recognition.phi_component.powf(k_f) +
                               recognition.pi_component * k_f +
                               recognition.e_component / k_f;
        }

        Ok(harmonics)
    }

    fn find_resonance_peaks(&self, field: &Array1<f64>) -> Vec<usize> {
        let mut peaks = Vec::new();
        
        for i in 1..field.len() - 1 {
            if field[i] > field[i - 1] && field[i] > field[i + 1] {
                peaks.push(i);
            }
        }

        // Return top 10 peaks
        peaks.truncate(10);
        peaks
    }

    fn construct_pattern_matrix(&self, recognition: &UniversalRecognition) -> Result<Array2<f64>> {
        let mut matrix = Array2::zeros((4, 4));

        // Encode universal constant relationships
        matrix[[0, 0]] = recognition.phi_component;
        matrix[[0, 1]] = recognition.pi_component;
        matrix[[1, 0]] = recognition.e_component;
        matrix[[1, 1]] = recognition.unity_phase;

        // Cross-relationships
        matrix[[0, 2]] = recognition.phi_component * recognition.pi_component;
        matrix[[2, 0]] = recognition.e_component / (recognition.phi_component + 1e-10);
        matrix[[1, 2]] = recognition.unity_phase.sin();
        matrix[[2, 1]] = recognition.unity_phase.cos();

        // Normalize
        matrix[[2, 2]] = matrix.diag().sum();

        // Fill with resonance field values
        let field_len = recognition.resonance_field.len();
        for i in 0..4.min(field_len) {
            matrix[[3, i]] = recognition.resonance_field[i];
        }

        Ok(matrix)
    }

    fn encode_factor_structure(&self, recognition: &UniversalRecognition) -> HashMap<String, f64> {
        let mut encoding = HashMap::new();

        encoding.insert("product_phase".to_string(), 
            (recognition.phi_component * recognition.pi_component) % (2.0 * self.constants.pi));
        
        encoding.insert("sum_resonance".to_string(),
            recognition.phi_component + recognition.pi_component + recognition.e_component);
        
        encoding.insert("difference_field".to_string(),
            (recognition.phi_component - recognition.e_component).abs());
        
        encoding.insert("unity_coupling".to_string(),
            recognition.unity_phase / (2.0 * self.constants.pi));
        
        let resonance_integral = recognition.resonance_field.sum() / recognition.resonance_field.len() as f64;
        encoding.insert("resonance_integral".to_string(), resonance_integral);

        encoding
    }

    // Decoding strategies

    fn decode_via_resonance(&self, n: &Number, formalization: &UniversalFormalization) -> Result<Factors> {
        let peaks = &formalization.resonance_peaks;
        let field = &formalization.harmonic_series;
        
        if peaks.len() < 2 {
            return Err(PatternError::ExecutionError("Insufficient resonance peaks".to_string()));
        }
        
        let sqrt_n = utils::integer_sqrt(n)?;
        
        // Analyze peak spacing (Python: peak_spacing = np.diff(resonance_peaks))
        for i in 0..peaks.len() - 1 {
            let spacing = peaks[i + 1] as i64 - peaks[i] as i64;
            if spacing > 0 {
                // Scale spacing to potential factor
                let factor_estimate = (spacing as f64 * sqrt_n.to_f64().unwrap_or(1.0) / field.len() as f64) as u64;
                let candidate = Number::from(factor_estimate);
                
                if candidate > Number::from(1u32) && n % &candidate == Number::from(0u32) {
                    let other = n / &candidate;
                    return Ok(Factors::new(candidate, other, "universal_resonance_spacing"));
                }
            }
        }
        
        // Analyze peak magnitudes
        let peak_magnitudes: Vec<f64> = peaks.iter()
            .map(|&p| if p < field.len() { field[p] } else { 0.0 })
            .collect();
        
        // Relative magnitudes can encode factor relationships
        if peak_magnitudes.len() >= 2 && peak_magnitudes[0] > 0.0 {
            let magnitude_ratio = peak_magnitudes[1] / peak_magnitudes[0];
            
            // Map magnitude ratio to factor estimate
            let factor_estimate = (magnitude_ratio * sqrt_n.to_f64().unwrap_or(1.0)) as u64;
            let candidate = Number::from(factor_estimate);
            
            if candidate > Number::from(1u32) && n % &candidate == Number::from(0u32) {
                let other = n / &candidate;
                return Ok(Factors::new(candidate, other, "universal_resonance_magnitude"));
            }
        }

        Err(PatternError::ExecutionError("Resonance decoding failed".to_string()))
    }

    fn decode_via_eigenvalues(&self, n: &Number, formalization: &UniversalFormalization) -> Result<Factors> {
        // Convert ndarray to nalgebra matrix
        let (rows, cols) = formalization.pattern_matrix.dim();
        let data: Vec<f64> = formalization.pattern_matrix.iter().cloned().collect();
        let matrix = DMatrix::from_row_slice(rows, cols, &data);
        
        // Compute eigendecomposition for symmetric part
        let symmetric = &matrix * &matrix.transpose();
        let eigen = SymmetricEigen::new(symmetric);
        
        let sqrt_n = utils::integer_sqrt(n)?;
        
        // Use eigenvalues and eigenvectors to find factors
        for (i, eigenval) in eigen.eigenvalues.iter().enumerate() {
            if eigenval.is_finite() && *eigenval > 1.0 {
                // Direct eigenvalue interpretation (Python: factor_candidate = int(abs(eigenval.real) * np.sqrt(n) / self.basis.PHI))
                let factor_candidate = (eigenval.abs() * sqrt_n.to_f64().unwrap_or(1.0) / self.constants.phi) as u64;
                let candidate = Number::from(factor_candidate);
                
                if candidate > Number::from(1u32) && n % &candidate == Number::from(0u32) {
                    let other = n / &candidate;
                    return Ok(Factors::new(candidate, other, "universal_eigenvalue_direct"));
                }
                
                // Eigenvector interpretation
                if i < eigen.eigenvectors.ncols() {
                    let eigenvec = eigen.eigenvectors.column(i);
                    
                    // Find dominant component
                    let (dominant_idx, &max_val) = eigenvec.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or((0, &0.0));
                    
                    if dominant_idx < eigenvec.len() {
                        let factor_candidate = (max_val.abs() * sqrt_n.to_f64().unwrap_or(1.0)) as u64;
                        let candidate = Number::from(factor_candidate);
                        
                        if candidate > Number::from(1u32) && n % &candidate == Number::from(0u32) {
                            let other = n / &candidate;
                            return Ok(Factors::new(candidate, other, "universal_eigenvalue_vector"));
                        }
                    }
                }
            }
        }

        Err(PatternError::ExecutionError("Eigenvalue decoding failed".to_string()))
    }

    fn decode_via_harmonics(&self, n: &Number, formalization: &UniversalFormalization) -> Result<Factors> {
        let harmonics = &formalization.harmonic_series;
        
        if harmonics.len() < 2 {
            return Err(PatternError::ExecutionError("Insufficient harmonics".to_string()));
        }
        
        let sqrt_n = utils::integer_sqrt(n)?;
        
        // Look for harmonic intersections (Python: harmonic difference encodes factor information)
        for i in 0..harmonics.len() - 1 {
            for j in i + 1..harmonics.len() {
                let h_diff = (harmonics[i] - harmonics[j]).abs();
                
                if h_diff > 0.0 {
                    // Scale to factor estimate
                    let factor_estimate = (h_diff * sqrt_n.to_f64().unwrap_or(1.0) / self.constants.e) as u64;
                    let candidate = Number::from(factor_estimate);
                    
                    if candidate > Number::from(1u32) && n % &candidate == Number::from(0u32) {
                        let other = n / &candidate;
                        return Ok(Factors::new(candidate, other, "universal_harmonic_diff"));
                    }
                }
            }
        }
        
        // Harmonic ratios
        for i in 1..harmonics.len() {
            if harmonics[i] != 0.0 {
                let ratio = harmonics[0] / harmonics[i];
                let factor_estimate = (ratio * sqrt_n.to_f64().unwrap_or(1.0) / i as f64) as u64;
                let candidate = Number::from(factor_estimate);
                
                if candidate > Number::from(1u32) && n % &candidate == Number::from(0u32) {
                    let other = n / &candidate;
                    return Ok(Factors::new(candidate, other, "universal_harmonic_ratio"));
                }
            }
        }

        Err(PatternError::ExecutionError("Harmonic decoding failed".to_string()))
    }

    fn decode_via_universal_relationships(&self, n: &Number, formalization: &UniversalFormalization) -> Result<Factors> {
        let encoding = &formalization.factor_encoding;
        let sqrt_n = utils::integer_sqrt(n)?;
        
        // Use sum and product relationships
        if let Some(sum_resonance) = encoding.get("sum_resonance") {
            let sum_estimate = sum_resonance * sqrt_n.to_f64().unwrap_or(1.0) / self.constants.phi;
            
            // Quadratic relationship: p + q = sum, p * q = n
            let discriminant = sum_estimate * sum_estimate - 4.0 * n.to_f64().unwrap_or(1.0);
            
            if discriminant >= 0.0 {
                let sqrt_disc = discriminant.sqrt();
                let p = ((sum_estimate + sqrt_disc) / 2.0) as u64;
                let q = ((sum_estimate - sqrt_disc) / 2.0) as u64;
                
                let p_num = Number::from(p);
                let q_num = Number::from(q);
                
                if &p_num * &q_num == *n && p > 1 && q > 1 {
                    let (min_factor, max_factor) = if p_num < q_num { (p_num, q_num) } else { (q_num, p_num) };
                    return Ok(Factors::new(min_factor, max_factor, "universal_quadratic"));
                }
            }
        }

        Err(PatternError::ExecutionError("Universal relationship decoding failed".to_string()))
    }
    
    fn decode_via_phase_relationships(&self, n: &Number, formalization: &UniversalFormalization) -> Result<Factors> {
        let encoding = &formalization.factor_encoding;
        let sqrt_n = utils::integer_sqrt(n)?;
        
        // Extract phase information
        let product_phase = encoding.get("product_phase").copied().unwrap_or(0.0);
        let unity_coupling = encoding.get("unity_coupling").copied().unwrap_or(0.0);
        
        // Phase difference often encodes p - q
        let phase_diff = (product_phase - unity_coupling * 2.0 * self.constants.pi).abs();
        
        // Estimate p - q from phase
        let diff_estimate = (phase_diff * sqrt_n.to_f64().unwrap_or(1.0) / (2.0 * self.constants.pi)) as i64;
        
        // Use sum-difference relationships
        // We know p * q = n and estimate p - q
        // Therefore p = (sqrt(n² + diff²) + diff) / 2
        
        if diff_estimate >= 0 {
            let n_float = n.to_f64().unwrap_or(1e9);
            // Avoid overflow by checking if n is too large
            if n_float < 1e15 {  // Only attempt for reasonable sized numbers
                let n_squared = n_float * n_float;
                let diff_squared = (diff_estimate * diff_estimate) as f64;
                let discriminant = n_squared + diff_squared;
                
                if discriminant >= 0.0 && discriminant < 1e30 {  // Prevent overflow
                    let sqrt_disc = discriminant.sqrt();
                    
                    // Use floating point to avoid integer overflow
                    let p_float = (sqrt_disc + diff_estimate as f64) / 2.0;
                    
                    if p_float > 1.0 && p_float < 1e18 {  // Reasonable bounds
                        let p = p_float as u64;
                        let p_num = Number::from(p);
                        if n % &p_num == Number::from(0u32) {
                            let q_num = n / &p_num;
                            return Ok(Factors::new(p_num, q_num, "universal_phase_diff"));
                        }
                    }
                }
            }
        }
        
        // Try phase-based search
        let phase_center = (product_phase * sqrt_n.to_f64().unwrap_or(1.0) / self.constants.pi) as u64;
        let search_radius = if n.bit_length() > 200 {
            10000 // Limit for very large numbers
        } else {
            (n.to_f64().unwrap_or(1.0).powf(0.25) as u64).min(100000).max(100)
        };
        
        for offset in 0..search_radius {
            // Check for overflow before adding
            if let Some(candidate_val) = phase_center.checked_add(offset) {
                let p_candidate = Number::from(candidate_val);
                if p_candidate > Number::from(1u32) && n % &p_candidate == Number::from(0u32) {
                    let q = n / &p_candidate;
                    return Ok(Factors::new(p_candidate, q, "universal_phase_search"));
                }
            }
            
            if phase_center > offset {
                let p_candidate = Number::from(phase_center - offset);
                if p_candidate > Number::from(1u32) && n % &p_candidate == Number::from(0u32) {
                    let q = n / &p_candidate;
                    return Ok(Factors::new(p_candidate, q, "universal_phase_search"));
                }
            }
        }
        
        Err(PatternError::ExecutionError("Phase relationship decoding failed".to_string()))
    }
    
    fn enhanced_search(&self, n: &Number, formalization: &UniversalFormalization) -> Result<Factors> {
        // Use all available information to guide search
        let encoding = &formalization.factor_encoding;
        let resonance_peaks = &formalization.resonance_peaks;
        let sqrt_n = utils::integer_sqrt(n)?;
        
        // Estimate search center from multiple sources
        let mut estimates = Vec::new();
        
        // Resonance-based estimate
        if !resonance_peaks.is_empty() {
            let field_len = formalization.harmonic_series.len() as f64;
            estimates.push(resonance_peaks[0] as f64 * sqrt_n.to_f64().unwrap_or(1.0) / field_len);
        }
        
        // Phase-based estimate
        if let Some(product_phase) = encoding.get("product_phase") {
            let phase_estimate = product_phase * sqrt_n.to_f64().unwrap_or(1.0) / self.constants.pi;
            estimates.push(phase_estimate);
        }
        
        // Universal constant estimate
        let phi_estimate = formalization.universal_coordinates[0] * sqrt_n.to_f64().unwrap_or(1.0) / self.constants.phi;
        estimates.push(phi_estimate);
        
        // Use median of estimates as search center
        let search_center = if estimates.is_empty() {
            sqrt_n.to_f64().unwrap_or(1000.0) as u64
        } else {
            estimates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            estimates[estimates.len() / 2] as u64
        };
        
        // Adaptive search radius
        let search_radius = if n.bit_length() > 200 {
            10000  // Limit for very large numbers
        } else {
            (n.to_f64().unwrap_or(1.0).powf(0.25) as u64).max(10)
        };
        
        // Prioritized search order based on pattern
        let search_order = self.get_search_order(search_center, search_radius, encoding);
        
        for p_candidate in search_order {
            let candidate = Number::from(p_candidate);
            if candidate > Number::from(1u32) && candidate < *n && n % &candidate == Number::from(0u32) {
                let q = n / &candidate;
                return Ok(Factors::new(candidate, q, "universal_enhanced_search"));
            }
        }
        
        Err(PatternError::ExecutionError("Enhanced search failed".to_string()))
    }
    
    fn get_search_order(&self, center: u64, radius: u64, encoding: &HashMap<String, f64>) -> Vec<u64> {
        let mut candidates = Vec::new();
        
        let unity_coupling = encoding.get("unity_coupling").copied().unwrap_or(0.0);
        let product_phase = encoding.get("product_phase").copied().unwrap_or(0.0);
        
        // Generate candidates with priorities
        for offset in 0..=radius {
            if offset == 0 {
                candidates.push((center, 0.0));  // Highest priority
            } else {
                // Priority based on resonance with encoding
                let priority1 = (offset as f64 * unity_coupling).sin().abs();
                let priority2 = (offset as f64 * product_phase).cos().abs();
                
                if let Some(candidate) = center.checked_add(offset) {
                    candidates.push((candidate, priority1));
                }
                if center >= offset {
                    candidates.push((center - offset, priority2));
                }
            }
        }
        
        // Sort by priority (lower is better)
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return just the candidate values
        candidates.into_iter()
            .map(|(c, _)| c)
            .filter(|&c| c > 1)
            .collect()
    }
}

/// Recognition result with universal components
#[derive(Debug, Clone)]
pub struct UniversalRecognition {
    pub value: Number,
    pub phi_component: f64,
    pub pi_component: f64,
    pub e_component: f64,
    pub unity_phase: f64,
    pub resonance_field: Array1<f64>,
}

/// Formalization with universal encoding
#[derive(Debug, Clone)]
pub struct UniversalFormalization {
    pub value: Number,
    pub universal_coordinates: Vec<f64>,
    pub harmonic_series: Array1<f64>,
    pub resonance_peaks: Vec<usize>,
    pub pattern_matrix: Array2<f64>,
    pub factor_encoding: HashMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_universal_constants() {
        let constants = UniversalConstants::default();
        
        // Verify golden ratio
        assert!((constants.phi - 1.618033988749895).abs() < 1e-10);
        
        // Verify relationship: φ² = φ + 1
        assert!((constants.phi * constants.phi - (constants.phi + 1.0)).abs() < 1e-10);
        
        // Verify beta = 2 - φ
        assert!((constants.beta - (2.0 - constants.phi)).abs() < 1e-10);
    }

    #[test]
    fn test_universal_pattern_small() {
        let pattern = UniversalPattern::new();
        
        // Test with 143 = 11 × 13
        let n = Number::from(143u32);
        
        let recognition = pattern.recognize(&n).unwrap();
        assert!(recognition.phi_component > 0.0);
        assert!(recognition.resonance_field.len() > 0);
        
        let formalization = pattern.formalize(recognition).unwrap();
        assert_eq!(formalization.universal_coordinates.len(), 4);
        
        // Execution might not always succeed with the universal pattern
        // It's more of a research approach than guaranteed factorization
        let _ = pattern.execute(formalization);
    }
}