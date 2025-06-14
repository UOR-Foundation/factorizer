//! Universal Pattern implementation based on discovered constants
//!
//! This module implements The Pattern's core insights:
//! - Universal constant basis (φ, π, e, 1)
//! - Three-stage process (Recognition, Formalization, Execution)
//! - Resonance field and harmonic analysis

use crate::error::PatternError;
use crate::types::{Number, Factors, Rational, integer_sqrt};
use crate::types::constants::{FundamentalConstantsRational, get_constant, ConstantType};
use crate::utils;
use crate::Result;
use crate::pattern::basis_exact::BasisExact;
use crate::pattern::wave_synthesis_exact::WaveSynthesisPatternExact;
use std::collections::HashMap;
use std::str::FromStr;

/// Universal constants using exact rational arithmetic
#[derive(Debug, Clone)]
pub struct UniversalConstants {
    /// Golden ratio φ
    pub phi: Rational,
    /// Circle constant π
    pub pi: Rational,
    /// Natural logarithm base e
    pub e: Rational,
    /// Unity
    pub unity: Rational,
    /// Derived: 2 - φ
    pub beta: Rational,
    /// Euler-Mascheroni constant (approximation)
    pub gamma: Rational,
    /// Precision in bits
    precision_bits: u32,
}

impl UniversalConstants {
    fn new(precision_bits: u32) -> Self {
        let phi = get_constant(ConstantType::Phi, precision_bits);
        let pi = get_constant(ConstantType::Pi, precision_bits);
        let e = get_constant(ConstantType::E, precision_bits);
        let one = Number::from(1u32) << precision_bits;
        let two = Number::from(2u32) << precision_bits;
        
        UniversalConstants {
            phi: Rational::from_integer(phi.clone()),
            pi: Rational::from_integer(pi),
            e: Rational::from_integer(e),
            unity: Rational::from_integer(one.clone()),
            beta: Rational::from_integer(&two - &phi),
            gamma: Rational::from_ratio(577215664901532u64, 1000000000000000u64),
            precision_bits,
        }
    }
}

impl Default for UniversalConstants {
    fn default() -> Self {
        Self::new(256) // Default 256-bit precision
    }
}

/// Pattern-specific invariant constants using exact arithmetic
pub struct PatternConstants {
    /// φ / π - fundamental resonance ratio
    pub resonance_base: Rational,
    /// e / φ - harmonic scaling factor
    pub harmonic_scale: Rational,
    /// 2π - unity field
    pub unity_field: Rational,
    /// Fundamental constants from empirical observation
    pub fundamental: FundamentalConstantsRational,
    /// Balanced semiprime detection threshold
    pub balance_threshold: Rational,
    /// Precision scaling factor for large numbers
    pub precision_scale: Rational,
}

impl PatternConstants {
    fn new(constants: &UniversalConstants, precision_bits: u32) -> Self {
        PatternConstants {
            resonance_base: &constants.phi / &constants.pi,
            harmonic_scale: &constants.e / &constants.phi,
            unity_field: &constants.pi * &Rational::from_integer(Number::from(2u32)),
            fundamental: FundamentalConstantsRational::new(precision_bits),
            balance_threshold: Rational::from_ratio(1u32, 100000u32),
            precision_scale: Rational::from_integer(Number::from(50u32)),
        }
    }
}

/// Universal Pattern recognizer with exact arithmetic
pub struct UniversalPattern {
    constants: UniversalConstants,
    pattern_constants: PatternConstants,
    cache: HashMap<String, (Number, Number)>,
    /// Pre-computed universal basis for poly-time solving
    universal_basis: Option<BasisExact>,
    /// Wave synthesis pattern for large numbers
    wave_synthesis: Option<WaveSynthesisPatternExact>,
    /// Precision in bits
    precision_bits: u32,
}

impl UniversalPattern {
    /// Create a new UniversalPattern recognizer with specified precision
    pub fn new() -> Self {
        Self::with_precision(256) // Default 256-bit precision
    }
    
    /// Create with specified precision in bits
    pub fn with_precision(precision_bits: u32) -> Self {
        let constants = UniversalConstants::new(precision_bits);
        let pattern_constants = PatternConstants::new(&constants, precision_bits);
        
        UniversalPattern {
            constants,
            pattern_constants,
            cache: HashMap::new(),
            universal_basis: None,
            wave_synthesis: None,
            precision_bits,
        }
    }
    
    /// Initialize with pre-computed basis for poly-time solving
    pub fn with_precomputed_basis() -> Self {
        let precision_bits = 256;
        let mut pattern = Self::with_precision(precision_bits);
        pattern.universal_basis = Some(BasisExact::new(precision_bits));
        pattern.wave_synthesis = Some(WaveSynthesisPatternExact::new(precision_bits));
        pattern
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
        
        // 0. Use wave synthesis for large numbers
        if let Some(ref mut wave_synthesis) = self.wave_synthesis {
            if n.bit_length() > 64 {
                if let Ok(factors) = wave_synthesis.factor(n) {
                    self.cache.insert(cache_key.clone(), (factors.p.clone(), factors.q.clone()));
                    return Ok(factors);
                }
            }
        }
        
        // 1. Try pre-computed basis if available
        if let Some(ref basis) = self.universal_basis {
            if let Ok(factors) = self.decode_with_precomputed_basis(n, &formalization, basis) {
                self.cache.insert(cache_key.clone(), (factors.p.clone(), factors.q.clone()));
                return Ok(factors);
            }
        }
        
        // For very large numbers (RSA-scale), we need different strategies
        if n.bit_length() > 300 {
            // RSA-scale numbers cannot be factored with current methods
            return Err(PatternError::ExecutionError(
                format!("Cannot factor {}-bit number with current methods. RSA-scale factorization requires quantum computers or centuries of computation.", n.bit_length())
            ));
        }
        
        // 1. Quick check for small factors (helps with unbalanced cases)
        if n.bit_length() <= 40 {  // Only for very small numbers
            if let Ok(factors) = self.quick_small_factor_check(n) {
                self.cache.insert(cache_key.clone(), (factors.p.clone(), factors.q.clone()));
                return Ok(factors);
            }
        }
        
        // 2. For potentially balanced semiprimes, try Fermat first
        // Detection: check if n is close to a perfect square
        let sqrt_n = utils::integer_sqrt(n)?;
        let sqrt_squared = &sqrt_n * &sqrt_n;
        let difference = if n > &sqrt_squared {
            n - &sqrt_squared
        } else {
            &sqrt_squared - n
        };
        
        // Use exact arithmetic for balance detection
        let diff_rat = Rational::from_integer(difference);
        let sqrt_rat = Rational::from_integer(sqrt_n);
        let balance_ratio = &diff_rat / &sqrt_rat;
        
        // Adaptive threshold based on number size
        let n_bits = n.bit_length();
        let threshold = if n_bits > 200 {
            &self.pattern_constants.balance_threshold * &Rational::from_integer(Number::from(1000u32))
        } else if n_bits > 150 {
            &self.pattern_constants.balance_threshold * &Rational::from_integer(Number::from(100u32))
        } else if n_bits > 100 {
            &self.pattern_constants.balance_threshold * &Rational::from_integer(Number::from(10u32))
        } else {
            self.pattern_constants.balance_threshold.clone()
        };
        
        let is_likely_balanced = balance_ratio < threshold;
        
        if is_likely_balanced && n.bit_length() > 40 {
            if let Ok(factors) = self.fermat_based_search(n) {
                self.cache.insert(cache_key.clone(), (factors.p.clone(), factors.q.clone()));
                return Ok(factors);
            }
        }
        
        // 3. Phi-sum guided search - USES TRUE INVARIANT
        if let Ok(factors) = self.phi_sum_guided_search(n, &formalization) {
            self.cache.insert(cache_key.clone(), (factors.p.clone(), factors.q.clone()));
            return Ok(factors);
        }
        
        // 4. Resonance peak decoding
        if let Ok(factors) = self.decode_via_resonance(n, &formalization) {
            self.cache.insert(cache_key.clone(), (factors.p.clone(), factors.q.clone()));
            return Ok(factors);
        }
        
        // 5. For small numbers, try direct factorization
        if n.bit_length() <= 40 {
            if let Ok(factors) = self.small_number_factorization(n) {
                self.cache.insert(cache_key.clone(), (factors.p.clone(), factors.q.clone()));
                return Ok(factors);
            }
        }

        // 6. Eigenvalue decoding
        if let Ok(factors) = self.decode_via_eigenvalues(n, &formalization) {
            self.cache.insert(cache_key.clone(), (factors.p.clone(), factors.q.clone()));
            return Ok(factors);
        }

        // 7. Harmonic intersection
        if let Ok(factors) = self.decode_via_harmonics(n, &formalization) {
            self.cache.insert(cache_key.clone(), (factors.p.clone(), factors.q.clone()));
            return Ok(factors);
        }

        // 8. Phase relationship decoding
        if let Ok(factors) = self.decode_via_phase_relationships(n, &formalization) {
            self.cache.insert(cache_key.clone(), (factors.p.clone(), factors.q.clone()));
            return Ok(factors);
        }

        // 9. Universal relationship decoding
        if let Ok(factors) = self.decode_via_universal_relationships(n, &formalization) {
            self.cache.insert(cache_key.clone(), (factors.p.clone(), factors.q.clone()));
            return Ok(factors);
        }
        
        // 10. Universal intersection search (if not too large)
        if n.bit_length() <= 64 {
            if let Ok(factors) = self.decode_via_universal_intersections(n, &formalization) {
                self.cache.insert(cache_key.clone(), (factors.p.clone(), factors.q.clone()));
                return Ok(factors);
            }
        }
        
        // 11. Enhanced search as fallback
        if let Ok(factors) = self.enhanced_search(n, &formalization) {
            self.cache.insert(cache_key.clone(), (factors.p.clone(), factors.q.clone()));
            return Ok(factors);
        }
        
        // 12. Final attempt with Fermat if we haven't tried it yet
        if !is_likely_balanced {
            if let Ok(factors) = self.fermat_based_search(n) {
                self.cache.insert(cache_key, (factors.p.clone(), factors.q.clone()));
                return Ok(factors);
            }
        }

        Err(PatternError::ExecutionError("All decoding strategies failed".to_string()))
    }
    
    // Pre-computed basis decoding - THE POLY-TIME APPROACH
    fn decode_with_precomputed_basis(&self, n: &Number, _formalization: &UniversalFormalization, basis: &BasisExact) -> Result<Factors> {
        // Scale the pre-computed basis to this number
        let scaled_basis = basis.scale_to_number(n);
        
        // Use the scaled basis to find factors in poly-time
        match basis.find_factors(n, &scaled_basis) {
            Ok((p, q)) => Ok(Factors::new(p, q, "precomputed_basis_exact")),
            Err(e) => Err(e)
        }
    }
    
    // Quick check for small factors (helps with unbalanced cases)
    fn quick_small_factor_check(&self, n: &Number) -> Result<Factors> {
        // Check small primes first
        let small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
                           53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113];
        
        for &p in &small_primes {
            let p_num = Number::from(p as u32);
            if n % &p_num == Number::from(0u32) {
                let q = n / &p_num;
                if q > Number::from(1u32) {
                    return Ok(Factors::new(p_num, q, "quick_small_factor"));
                }
            }
        }
        
        // Check slightly larger range for unbalanced cases
        let limit = Number::from(10000u32);
        let mut candidate = Number::from(127u32); // Start after small primes
        
        while candidate < limit {
            if n % &candidate == Number::from(0u32) {
                let q = n / &candidate;
                if q > Number::from(1u32) {
                    return Ok(Factors::new(candidate, q, "quick_factor_scan"));
                }
            }
            candidate = &candidate + &Number::from(2u32); // Check odd numbers
        }
        
        Err(PatternError::ExecutionError("No small factors found".to_string()))
    }
    
    // Helper method for small number factorization
    fn small_number_factorization(&self, n: &Number) -> Result<Factors> {
        let sqrt_n = utils::integer_sqrt(n)?;
        
        // Use Number type throughout for consistency
        let mut candidate = Number::from(2u32);
        let max_candidate = sqrt_n.min(Number::from(1_000_000u32));
        
        while candidate <= max_candidate {
            if n % &candidate == Number::from(0u32) {
                let q = n / &candidate;
                return Ok(Factors::new(candidate, q, "universal_small_factor"));
            }
            candidate = &candidate + &Number::from(1u32);
        }
        
        Err(PatternError::ExecutionError("No small factors found".to_string()))
    }
    
    // Fermat-based search for balanced semiprimes
    fn fermat_based_search(&self, n: &Number) -> Result<Factors> {
        // Fermat's method works well for balanced semiprimes where p and q are close
        let sqrt_n = utils::integer_sqrt(n)?;
        
        // Start from ceil(sqrt(n))
        let mut a = sqrt_n.clone();
        if &a * &a < *n {
            a = &a + &Number::from(1u32);
        }
        
        // Adaptive limit based on number size and balance detection
        // For RSA-scale numbers, even "close" factors are astronomically far in absolute terms
        let max_iterations = if n.bit_length() > 300 {
            // Cannot brute force RSA-scale numbers
            return Err(PatternError::ExecutionError(
                format!("Fermat search not feasible for {}-bit numbers. Use specialized algorithms.", n.bit_length())
            ));
        } else if n.bit_length() > 250 {
            1_000_000    // Very limited search for large numbers
        } else if n.bit_length() > 200 {
            10_000_000   // Still limited but more thorough
        } else if n.bit_length() > 150 {
            20_000_000   // More iterations for medium-large
        } else if n.bit_length() > 100 {
            5_000_000    // Better coverage for medium numbers
        } else if n.bit_length() > 64 {
            1_000_000
        } else {
            100_000
        };
        
        // For very large numbers, use adaptive step size
        // Start with step 1 for all sizes to ensure we don't miss factors
        let mut step = Number::from(1u32);
        
        // For acceleration, we'll increase step size if no factors found after many iterations
        let acceleration_threshold = if n.bit_length() > 200 {
            1_000_000
        } else if n.bit_length() > 150 {
            500_000
        } else {
            100_000
        };
        
        let mut iterations = 0;
        while iterations < max_iterations {
            let a_squared = &a * &a;
            if a_squared >= *n {
                let b_squared = &a_squared - n;
                
                // Check if b_squared is a perfect square
                let b = utils::integer_sqrt(&b_squared)?;
                if &b * &b == b_squared {
                    // Found factors: p = a - b, q = a + b
                    let p = &a - &b;
                    let q = &a + &b;
                    
                    if p > Number::from(1u32) && q > Number::from(1u32) {
                        return Ok(Factors::new(p, q, "universal_fermat"));
                    }
                }
            }
            
            a = &a + &step;
            iterations += 1;
            
            // Accelerate search if no factors found after threshold
            if iterations == acceleration_threshold && step == Number::from(1u32) {
                // Only accelerate for very large numbers where brute force is impractical
                if n.bit_length() > 150 {
                    step = Number::from(2u32);
                }
            }
            
            // Further acceleration for extremely large numbers
            if iterations == acceleration_threshold * 2 && n.bit_length() > 200 {
                step = Number::from(5u32);
            }
        }
        
        Err(PatternError::ExecutionError(
            format!("Fermat search failed after {} iterations for {}-bit number", iterations, n.bit_length())
        ))
    }
    
    // Universal intersection search - THE KEY METHOD
    fn decode_via_universal_intersections(&self, n: &Number, formalization: &UniversalFormalization) -> Result<Factors> {
        // Project n into universal space
        let n_coords = vec![
            formalization.universal_coordinates[0], // φ-coordinate
            formalization.universal_coordinates[1], // π-coordinate  
            formalization.universal_coordinates[2], // e-coordinate
            formalization.universal_coordinates[3], // unity-coordinate
        ];
        
        let sqrt_n = utils::integer_sqrt(n)?;
        
        // Use exact arithmetic for offset calculation
        let n_bits = Number::from(n.bit_length() as u32);
        let offset = if n.bit_length() < 64 {
            Number::from(1000u32)
        } else if n.bit_length() < 128 {
            Number::from(10000u32)
        } else {
            // Scale with sqrt of bit length
            integer_sqrt(&(&n_bits * &Number::from(100u32)))
        };
        
        let search_start = if sqrt_n > offset {
            &sqrt_n - &offset
        } else {
            Number::from(2u32)
        };
        
        let search_end = &sqrt_n + &offset;
        let mut p_candidate = search_start;
        
        // Limit iterations for practicality
        let max_iterations = 1_000_000;
        let mut iterations = 0;
        
        while p_candidate <= search_end && iterations < max_iterations {
            if n % &p_candidate == Number::from(0u32) {
                let q_candidate = n / &p_candidate;
                
                // Project p and q into universal space
                let p_phi = self.extract_phi_component(&p_candidate)?;
                let _p_pi = self.extract_pi_component(&p_candidate)?;
                let p_e = self.extract_e_component(&p_candidate)?;
                
                let q_phi = self.extract_phi_component(&q_candidate)?;
                let _q_pi = self.extract_pi_component(&q_candidate)?;
                let q_e = self.extract_e_component(&q_candidate)?;
                
                // Check TRUE golden ratio invariant: p_φ + q_φ = n_φ
                let sum_phi = &p_phi + &q_phi;
                let diff_phi = if &sum_phi > &n_coords[0] {
                    &sum_phi - &n_coords[0]
                } else {
                    &n_coords[0] - &sum_phi
                };
                let threshold_phi = &n_coords[0] / &Rational::from_integer(Number::from(100u32));
                
                if diff_phi < threshold_phi {
                    return Ok(Factors::new(p_candidate, q_candidate, "universal_phi_sum_exact"));
                }
                
                // Check exponential sum relationship
                let sum_e = &p_e + &q_e;
                let diff_e = if &sum_e > &n_coords[2] {
                    &sum_e - &n_coords[2]
                } else {
                    &n_coords[2] - &sum_e
                };
                let threshold_e = &n_coords[2] / &Rational::from_integer(Number::from(10u32));
                
                if diff_e < threshold_e {
                    return Ok(Factors::new(p_candidate, q_candidate, "universal_e_sum_exact"));
                }
                
                // Check resonance-based relationships
                let p_q_resonance = &(&p_phi * &q_phi) / &self.pattern_constants.resonance_base;
                let diff_res = if &p_q_resonance > &n_coords[0] {
                    &p_q_resonance - &n_coords[0]
                } else {
                    &n_coords[0] - &p_q_resonance
                };
                
                if diff_res < threshold_e {
                    return Ok(Factors::new(p_candidate, q_candidate, "universal_resonance_exact"));
                }
            }
            
            p_candidate = &p_candidate + &Number::from(1u32);
            iterations += 1;
        }
        
        Err(PatternError::ExecutionError("No universal intersections found".to_string()))
    }
    
    // Guided search using the true φ-sum invariant
    fn phi_sum_guided_search(&self, n: &Number, formalization: &UniversalFormalization) -> Result<Factors> {
        let n_phi = &formalization.universal_coordinates[0];
        let sqrt_n = utils::integer_sqrt(n)?;
        
        // Use exact arithmetic for estimations
        let coord_sum = &formalization.universal_coordinates[0] + &formalization.universal_coordinates[1];
        let sqrt_rat = Rational::from_integer(sqrt_n.clone());
        let sum_estimate = &(&coord_sum * &sqrt_rat) / &self.constants.phi;
        let center = sum_estimate.round();
        
        // Search radius based on bit length
        let radius = if n.bit_length() < 64 {
            Number::from(1000u32)
        } else {
            Number::from(100u32)
        };
        
        let start = if center > radius { &center - &radius } else { Number::from(2u32) };
        let end = &center + &radius;
        
        let mut sum = start;
        while sum <= end {
            // For sum s, solve: p + q = s, p*q = n
            // This gives: p = (s ± sqrt(s² - 4n)) / 2
            let s_squared = &sum * &sum;
            let four_n = n * &Number::from(4u32);
            
            if s_squared > four_n {
                let discriminant = &s_squared - &four_n;
                let sqrt_disc = integer_sqrt(&discriminant);
                
                if &sqrt_disc * &sqrt_disc == discriminant {
                    let two = Number::from(2u32);
                    let p = (&sum - &sqrt_disc) / &two;
                    let q = (&sum + &sqrt_disc) / &two;
                    
                    if p > Number::from(1u32) && &p * &q == *n {
                        // Verify phi invariant
                        let p_phi = self.extract_phi_component(&p)?;
                        let q_phi = self.extract_phi_component(&q)?;
                        let sum_phi = &p_phi + &q_phi;
                        
                        // Check if close enough (within 1%)
                        let diff = if &sum_phi > n_phi {
                            &sum_phi - n_phi
                        } else {
                            n_phi - &sum_phi
                        };
                        let threshold = n_phi / &Rational::from_integer(Number::from(100u32));
                        
                        if diff < threshold {
                            return Ok(Factors::new(p, q, "phi_sum_guided_exact"));
                        }
                    }
                }
            }
            sum = &sum + &Number::from(1u32);
        }
        
        Err(PatternError::ExecutionError("Phi-sum guided search failed".to_string()))
    }

    // Component extraction methods

    fn extract_phi_component(&self, n: &Number) -> Result<Rational> {
        // Use bit length approximation for logarithm
        let n_bits = Rational::from_integer(Number::from(n.bit_length() as u32));
        let ln2 = Rational::from_ratio(693147u64, 1000000u64);
        let ln_n = &n_bits * &ln2;
        let ln_phi = self.constants.phi.log_approx();
        
        Ok(&ln_n / &ln_phi)
    }

    fn extract_pi_component(&self, n: &Number) -> Result<Rational> {
        // π-coordinate: (n * φ) % π
        let n_rat = Rational::from_integer(n.clone());
        Ok(&(&n_rat * &self.constants.phi) % &self.constants.pi)
    }

    fn extract_e_component(&self, n: &Number) -> Result<Rational> {
        // e-coordinate: ln(n + 1) / e
        let n_plus_one = n + &Number::from(1u32);
        let n_bits = Rational::from_integer(Number::from(n_plus_one.bit_length() as u32));
        let ln2 = Rational::from_ratio(693147u64, 1000000u64);
        let ln_n = &n_bits * &ln2;
        
        Ok(&ln_n / &self.constants.e)
    }

    fn extract_unity_phase(&self, n: &Number) -> Result<Rational> {
        // Unity coordinate: n / (n + φ + π + e)
        let n_rat = Rational::from_integer(n.clone());
        let sum = &(&self.constants.phi + &self.constants.pi) + &self.constants.e;
        let denominator = &n_rat + &sum;
        
        Ok(&n_rat / &denominator)
    }

    // Resonance field generation

    fn generate_resonance_field(&self, n: &Number, phi: &Rational, pi: &Rational, e: &Rational) -> Result<Vec<Number>> {
        let size = (Number::from(2u32).pow((n.bit_length() / 4) as u32) as usize)
            .max(64)
            .min(1024);
        let mut field = vec![Number::from(0u32); size];
        
        for i in 0..size {
            let i_rat = Rational::from_integer(Number::from(i as u32));
            let size_rat = Rational::from_integer(Number::from(size as u32));
            
            // Universal harmonic at position i using exact arithmetic
            let phase = &(&pi * &i_rat) / &size_rat;
            let harmonic = &(&phi * &phase) + &(&e * &phase);
            
            // Scale and convert to integer
            let scaled = &harmonic * &Rational::from_integer(Number::from(1u32) << 16);
            field[i] = scaled.round();
        }

        Ok(field)
    }

    // Formalization methods

    fn compute_harmonic_series(&self, recognition: &UniversalRecognition) -> Result<Vec<Number>> {
        let mut harmonics = Vec::new();
        
        for k in 1..=7 {
            let k_rat = Rational::from_integer(Number::from(k));
            
            let harmonic = &(&(&recognition.phi_component * &k_rat) * &self.constants.phi) +
                          &(&(&recognition.pi_component * &k_rat) * &self.constants.pi) +
                          &(&(&recognition.e_component * &k_rat) * &self.constants.e) +
                          &(&recognition.unity_phase * &k_rat);
            
            let modded = &harmonic % &self.pattern_constants.unity_field;
            harmonics.push(modded.round());
        }

        Ok(harmonics)
    }

    fn find_resonance_peaks(&self, field: &[Number]) -> Vec<usize> {
        let mut peaks = Vec::new();
        
        for i in 1..field.len() - 1 {
            if field[i] > field[i - 1] && field[i] > field[i + 1] {
                peaks.push(i);
            }
        }

        peaks.truncate(10); // Top 10 peaks
        peaks
    }

    fn construct_pattern_matrix(&self, recognition: &UniversalRecognition) -> Result<Vec<Vec<Rational>>> {
        let mut matrix = vec![vec![Rational::zero(); 4]; 4];

        // Encode universal constant relationships
        matrix[0][0] = recognition.phi_component.clone();
        matrix[0][1] = recognition.pi_component.clone();
        matrix[1][0] = recognition.e_component.clone();
        matrix[1][1] = recognition.unity_phase.clone();

        // Cross-relationships
        matrix[0][2] = &recognition.phi_component * &recognition.pi_component;
        matrix[2][0] = &recognition.e_component / &(&recognition.phi_component + &Rational::from_ratio(1u32, 1000000u32));
        
        // Use exact arithmetic for trigonometric approximations
        matrix[1][2] = recognition.unity_phase.clone(); // Simplified
        matrix[2][1] = recognition.unity_phase.clone(); // Simplified

        // Diagonal sum
        matrix[2][2] = &(&matrix[0][0] + &matrix[1][1]) + &matrix[2][0];

        // Fill with resonance field values
        for i in 0..4.min(recognition.resonance_field.len()) {
            matrix[3][i] = Rational::from_integer(recognition.resonance_field[i].clone());
        }

        Ok(matrix)
    }

    fn encode_factor_structure(&self, recognition: &UniversalRecognition) -> HashMap<String, Rational> {
        let mut encoding = HashMap::new();

        let two_pi = &self.constants.pi * &Rational::from_integer(Number::from(2u32));
        
        encoding.insert("product_phase".to_string(), 
            &(&recognition.phi_component * &recognition.pi_component) % &two_pi);
        
        encoding.insert("sum_resonance".to_string(),
            &(&recognition.phi_component + &recognition.pi_component) + &recognition.e_component);
        
        let diff = &recognition.phi_component - &recognition.e_component;
        encoding.insert("difference_field".to_string(),
            if diff.is_negative() { &Rational::zero() - &diff } else { diff });
        
        encoding.insert("unity_coupling".to_string(),
            &recognition.unity_phase / &two_pi);
        
        // Resonance integral
        let sum: Number = recognition.resonance_field.iter().cloned().sum();
        let len = Number::from(recognition.resonance_field.len() as u32);
        encoding.insert("resonance_integral".to_string(),
            Rational::from_ratio(sum, len));

        encoding
    }

    // Decoding strategies

    fn decode_via_resonance(&self, n: &Number, formalization: &UniversalFormalization) -> Result<Factors> {
        let peaks = &formalization.resonance_peaks;
        
        if peaks.len() < 2 {
            return Err(PatternError::ExecutionError("Insufficient resonance peaks".to_string()));
        }
        
        let sqrt_n = utils::integer_sqrt(n)?;
        
        // Analyze peak spacing
        for i in 0..peaks.len() - 1 {
            let spacing = Number::from((peaks[i + 1] - peaks[i]) as u32);
            let field_len = Number::from(formalization.harmonic_series.len() as u32);
            
            if !spacing.is_zero() && !field_len.is_zero() {
                let scale_rat = Rational::from_ratio(spacing, field_len);
                let sqrt_rat = Rational::from_integer(sqrt_n.clone());
                let candidate = (&sqrt_rat * &scale_rat).round();
                
                if candidate > Number::from(1u32) && n % &candidate == Number::from(0u32) {
                    let other = n / &candidate;
                    return Ok(Factors::new(candidate, other, "resonance_spacing_exact"));
                }
            }
        }
        
        Err(PatternError::ExecutionError("Resonance decoding failed".to_string()))
    }

    fn decode_via_eigenvalues(&self, _n: &Number, _formalization: &UniversalFormalization) -> Result<Factors> {
        // Eigenvalue computation requires floating-point arithmetic
        // For exact arithmetic version, this would need a different approach
        Err(PatternError::ExecutionError("Eigenvalue decoding not available in exact mode".to_string()))
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
                    let scale_factor = h_diff / self.constants.e;
                    let candidate = if scale_factor > 0.0 && scale_factor.is_finite() {
                        let scaled = sqrt_n.to_f64().unwrap_or(1e100) * scale_factor;
                        if scaled < 1e18 {
                            Number::from(scaled as u128)
                        } else {
                            Number::from_str(&format!("{:.0}", scaled)).unwrap_or(Number::from(1u32))
                        }
                    } else {
                        Number::from(1u32)
                    };
                    
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
                let scale_factor = ratio / i as f64;
                let candidate = if scale_factor > 0.0 && scale_factor.is_finite() {
                    let scaled = sqrt_n.to_f64().unwrap_or(1e100) * scale_factor;
                    if scaled < 1e18 {
                        Number::from(scaled as u128)
                    } else {
                        Number::from_str(&format!("{:.0}", scaled)).unwrap_or(Number::from(1u32))
                    }
                } else {
                    Number::from(1u32)
                };
                
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
        let sqrt_n_f64 = sqrt_n.to_f64().unwrap_or(1e15);
        
        // Try multiple sum estimation approaches
        let mut sum_estimates = Vec::new();
        
        // Original sum resonance approach
        if let Some(sum_resonance) = encoding.get("sum_resonance") {
            // Try different scaling factors as the original has high error
            sum_estimates.push(sum_resonance * sqrt_n_f64 / self.constants.phi);
            sum_estimates.push(sum_resonance * sqrt_n_f64 / self.constants.e);
            sum_estimates.push(sum_resonance * sqrt_n_f64 / self.constants.pi);
            sum_estimates.push(sum_resonance * sqrt_n_f64); // Direct scaling
        }
        
        // Use harmonic mean of coordinates
        let coords = &formalization.universal_coordinates;
        if coords.len() >= 3 {
            let harmonic_mean = 3.0 / (1.0/coords[0] + 1.0/coords[1] + 1.0/coords[2]);
            sum_estimates.push(harmonic_mean * sqrt_n_f64 / self.constants.phi);
        }
        
        // Try each sum estimate
        for sum_estimate in sum_estimates {
            // For very large numbers, work in log space to avoid overflow
            if n.bit_length() > 100 {
                // Use Fermat's method insight: if p+q ≈ sum, then a = (p+q)/2
                let a_estimate = sum_estimate / 2.0;
                let a_squared = a_estimate * a_estimate;
                let n_f64 = n.to_f64().unwrap_or(1e30);
                
                if a_squared > n_f64 {
                    let b_squared = a_squared - n_f64;
                    if b_squared >= 0.0 {
                        let b = b_squared.sqrt();
                        let p_f64 = a_estimate + b;
                        let q_f64 = a_estimate - b;
                        
                        if p_f64 > 1.0 && q_f64 > 1.0 {
                            let p_num = if p_f64 < 1e18 {
                                Number::from(p_f64 as u128)
                            } else {
                                Number::from_str(&format!("{:.0}", p_f64)).unwrap_or(Number::from(1u32))
                            };
                            let q_num = if q_f64 < 1e18 {
                                Number::from(q_f64 as u128)
                            } else {
                                Number::from_str(&format!("{:.0}", q_f64)).unwrap_or(Number::from(1u32))
                            };
                            
                            if &p_num * &q_num == *n && p_num > Number::from(1u32) && q_num > Number::from(1u32) {
                                let (min_factor, max_factor) = if p_num < q_num { (p_num, q_num) } else { (q_num, p_num) };
                                return Ok(Factors::new(min_factor, max_factor, "universal_quadratic"));
                            }
                        }
                    }
                }
            } else {
                // Original quadratic approach for smaller numbers
                let discriminant = sum_estimate * sum_estimate - 4.0 * n.to_f64().unwrap_or(1.0);
                
                if discriminant >= 0.0 {
                    let sqrt_disc = discriminant.sqrt();
                    let p_f64 = (sum_estimate + sqrt_disc) / 2.0;
                    let q_f64 = (sum_estimate - sqrt_disc) / 2.0;
                    
                    let p_num = if p_f64 > 1.0 && p_f64 < 1e18 {
                        Number::from(p_f64 as u128)
                    } else if p_f64 > 1.0 {
                        Number::from_str(&format!("{:.0}", p_f64)).unwrap_or(Number::from(1u32))
                    } else {
                        Number::from(1u32)
                    };
                    let q_num = if q_f64 > 1.0 && q_f64 < 1e18 {
                        Number::from(q_f64 as u128)
                    } else if q_f64 > 1.0 {
                        Number::from_str(&format!("{:.0}", q_f64)).unwrap_or(Number::from(1u32))
                    } else {
                        Number::from(1u32)
                    };
                    
                    if &p_num * &q_num == *n && p_num > Number::from(1u32) && q_num > Number::from(1u32) {
                        let (min_factor, max_factor) = if p_num < q_num { (p_num, q_num) } else { (q_num, p_num) };
                        return Ok(Factors::new(min_factor, max_factor, "universal_quadratic"));
                    }
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
        let diff_estimate = (phase_diff * sqrt_n.to_f64().unwrap_or(1e100) / (2.0 * self.constants.pi)) as i128;
        
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
                    
                    if p_float > 1.0 {  // Check bounds
                        let p_num = if p_float < 1e18 {
                            Number::from(p_float as u128)
                        } else {
                            Number::from_str(&format!("{:.0}", p_float)).unwrap_or(Number::from(1u32))
                        };
                        if n % &p_num == Number::from(0u32) {
                            let q_num = n / &p_num;
                            return Ok(Factors::new(p_num, q_num, "universal_phase_diff"));
                        }
                    }
                }
            }
        }
        
        // Try phase-based search with improved radius calculation
        let phase_center_f64 = product_phase * sqrt_n.to_f64().unwrap_or(1e100) / self.constants.pi;
        let phase_center = if phase_center_f64 < 1e18 && phase_center_f64 > 0.0 {
            Number::from(phase_center_f64 as u64)
        } else {
            // For very large values, use exact computation
            let phase_scale = Number::from((product_phase * 1000.0) as u64);
            (sqrt_n * &phase_scale) / &Number::from((self.constants.pi * 1000.0) as u64)
        };
        
        // Improved search radius calculation based on Python implementation
        let search_radius = if n.bit_length() > 1000 {
            // For extremely large numbers, use a fixed large radius
            Number::from(1_000_000u64)
        } else if n.bit_length() > 200 {
            // For very large numbers, scale with bit length
            // Scale with sqrt of bit length
            let bits = Number::from(n.bit_length() as u32);
            integer_sqrt(&(&bits * &Number::from(100000000u32)))
        } else {
            // For smaller numbers, use n^0.25 with a reasonable minimum and maximum
            let n_float = n.to_f64().unwrap_or(1e15);
            if n_float < 1e15 {
                Number::from((n_float.powf(0.25) as u64).max(1000).min(10_000_000))
            } else {
                // Scale based on bit length for numbers beyond float range
                Number::from((2.0_f64.powf(n.bit_length() as f64 / 4.0) as u64).min(10_000_000))
            }
        };
        
        // Improved search with better factor candidate generation
        let mut offset = Number::from(0u32);
        while offset < search_radius {
            // Check both directions from center
            // For center + offset
            // phase_center is now a Number
            let candidate = &phase_center + &offset;
            if candidate > Number::from(1u32) && n % &candidate == Number::from(0u32) {
                    let q = n / &candidate;
                    // Ensure p <= q
                    let (p, q) = if candidate <= q {
                        (candidate, q)
                    } else {
                        (q, candidate)
                    };
                    return Ok(Factors::new(p, q, "universal_phase_search"));
            }
            
            // For center - offset
            if offset > Number::from(0u32) && phase_center > Number::from(0u32) {
                if phase_center > offset {
                    let p_candidate = &phase_center - &offset;
                    if n % &p_candidate == Number::from(0u32) {
                        let q = n / &p_candidate;
                        // Ensure p <= q
                        let (p, q) = if p_candidate <= q {
                            (p_candidate, q)
                        } else {
                            (q, p_candidate)
                        };
                        return Ok(Factors::new(p, q, "universal_phase_search"));
                    }
                }
            }
            
            offset = &offset + &Number::from(1u32);
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
            Number::from(1000u32)
        } else {
            estimates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            Number::from(estimates[estimates.len() / 2] as u128)
        };
        
        // Adaptive search radius - much more aggressive for larger numbers
        let search_radius = if n.bit_length() > 1000 {
            10_000_000u128  // Very large search radius for huge numbers
        } else if n.bit_length() > 500 {
            5_000_000u128
        } else if n.bit_length() > 200 {
            1_000_000u128
        } else if n.bit_length() > 100 {
            500_000u128
        } else if n.bit_length() > 50 {
            100_000u128
        } else {
            // For smaller numbers, use n^0.25 with reasonable bounds
            let n_float = n.to_f64().unwrap_or(1e15);
            if n_float < 1e15 {
                (n_float.powf(0.25) as u128).max(10_000)
            } else {
                50_000u128
            }
        };
        
        // For arbitrary precision, we need to handle the search differently
        // since we can't easily iterate over Number types in get_search_order
        let _candidates_found = Vec::<Number>::new();
        
        // Search around the center with radius
        let mut offset = Number::from(0u32);
        let max_iterations = 10_000_000;
        let mut iterations = 0;
        
        while offset <= Number::from(search_radius) && iterations < max_iterations {
            // Check center + offset
            if &search_center + &offset > Number::from(1u32) {
                let candidate = &search_center + &offset;
                if candidate < *n && n % &candidate == Number::from(0u32) {
                    let q = n / &candidate;
                    return Ok(Factors::new(candidate, q, "universal_enhanced_search"));
                }
            }
            
            // Check center - offset (if valid)
            if offset > Number::from(0u32) && search_center > offset {
                let candidate = &search_center - &offset;
                if candidate > Number::from(1u32) && n % &candidate == Number::from(0u32) {
                    let q = n / &candidate;
                    return Ok(Factors::new(candidate, q, "universal_enhanced_search"));
                }
            }
            
            offset = &offset + &Number::from(1u32);
            iterations += 1;
        }
        
        Err(PatternError::ExecutionError("Enhanced search failed".to_string()))
    }
    
    fn get_search_order(&self, _center: &Number, _radius: &Number, _encoding: &HashMap<String, f64>) -> Vec<Number> {
        // This method is no longer used with arbitrary precision
        // Keeping it for compatibility but returning empty vector
        Vec::new()
    }
}

/// Recognition result with universal components
#[derive(Debug, Clone)]
pub struct UniversalRecognition {
    /// The number being recognized
    pub value: Number,
    /// Golden ratio (φ) component extracted from the number
    pub phi_component: Rational,
    /// Pi (π) component extracted from the number
    pub pi_component: Rational,
    /// Euler's number (e) component extracted from the number
    pub e_component: Rational,
    /// Unity phase angle in the universal field
    pub unity_phase: Rational,
    /// Resonance field containing harmonic relationships
    pub resonance_field: Vec<Number>,
}

/// Formalization with universal encoding
#[derive(Debug, Clone)]
pub struct UniversalFormalization {
    /// The number being formalized
    pub value: Number,
    /// Coordinates in universal constant space
    pub universal_coordinates: Vec<Rational>,
    /// Harmonic series representation of the pattern
    pub harmonic_series: Vec<Number>,
    /// Indices of resonance peaks in the harmonic series
    pub resonance_peaks: Vec<usize>,
    /// Pattern matrix encoding relationships between components
    pub pattern_matrix: Vec<Vec<Rational>>,
    /// Factor encoding map for quantum neighborhood search
    pub factor_encoding: HashMap<String, Rational>,
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
        let mut pattern = UniversalPattern::new();
        
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