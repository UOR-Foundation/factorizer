//! Stage 3: Execution - Decode factors from formalized pattern
//!
//! This module executes the pattern to manifest factors.

use crate::error::PatternError;
use crate::pattern::expression::{Expression, PatternConstraint};
use crate::pattern::verification::verify_factors;
use crate::types::recognition::{DecodingStrategy, Factors, Formalization};
use crate::types::{Number, Pattern, PatternKind};
use crate::types::quantum_enhanced::{DistributionType, EnhancedQuantumRegion};
use crate::utils;
use crate::Result;
use nalgebra::{DMatrix, SymmetricEigen};
use std::collections::HashMap;

/// Execute the formalized pattern to decode factors
pub fn execute(formalization: Formalization, patterns: &[Pattern]) -> Result<Factors> {
    let n = &formalization.n;

    // First try constraint-based decoding if available
    if let Some(constraints) = formalization.get_constraints() {
        if let Ok(mut factors) = decode_with_constraints(&formalization, &constraints, patterns) {
            if factors.verify(n) {
                // Run comprehensive verification
                if let Ok(verification) = verify_factors(&factors, n, patterns, &factors.method) {
                    if verification.is_valid {
                        factors.confidence = verification.confidence;
                        return Ok(factors);
                    }
                }
            }
        }
    }

    // Try pattern-guided decoding
    if let Ok(mut factors) = pattern_guided_decoding(&formalization, patterns) {
        if factors.verify(n) {
            if let Ok(verification) = verify_factors(&factors, n, patterns, &factors.method) {
                if verification.is_valid {
                    factors.confidence = verification.confidence;
                    return Ok(factors);
                }
            }
        }
    }

    // Try each decoding strategy
    for strategy in &formalization.strategies {
        match decode_with_strategy(&formalization, strategy, patterns) {
            Ok(mut factors) if factors.verify(n) => {
                if let Ok(verification) = verify_factors(&factors, n, patterns, &factors.method) {
                    if verification.is_valid {
                        factors.confidence = verification.confidence;
                        return Ok(factors);
                    }
                }
            },
            _ => continue, // Try next strategy
        }
    }

    // Try advanced decoding methods
    if let Ok(mut factors) = advanced_resonance_decoding(&formalization, patterns) {
        if factors.verify(n) {
            if let Ok(verification) = verify_factors(&factors, n, patterns, &factors.method) {
                if verification.is_valid {
                    factors.confidence = verification.confidence;
                    return Ok(factors);
                }
            }
        }
    }

    if let Ok(mut factors) = harmonic_interference_decoding(&formalization) {
        if factors.verify(n) {
            if let Ok(verification) = verify_factors(&factors, n, patterns, &factors.method) {
                if verification.is_valid {
                    factors.confidence = verification.confidence;
                    return Ok(factors);
                }
            }
        }
    }

    if let Ok(mut factors) = quantum_collapse_decoding(&formalization) {
        if factors.verify(n) {
            if let Ok(verification) = verify_factors(&factors, n, patterns, &factors.method) {
                if verification.is_valid {
                    factors.confidence = verification.confidence;
                    return Ok(factors);
                }
            }
        }
    }

    // Try enhanced quantum search methods
    if let Ok(mut factors) = enhanced_quantum_search(&formalization, patterns) {
        if factors.verify(n) {
            if let Ok(verification) = verify_factors(&factors, n, patterns, &factors.method) {
                if verification.is_valid {
                    factors.confidence = verification.confidence;
                    return Ok(factors);
                }
            }
        }
    }

    // Try multi-level quantum search
    if let Ok(mut factors) = multi_level_quantum_search(&formalization, patterns) {
        if factors.verify(n) {
            if let Ok(verification) = verify_factors(&factors, n, patterns, &factors.method) {
                if verification.is_valid {
                    factors.confidence = verification.confidence;
                    return Ok(factors);
                }
            }
        }
    }

    // Try parallel quantum search for large numbers
    if n.bit_length() > 64 {
        if let Ok(mut factors) = parallel_quantum_search(&formalization, patterns, 8) {
            if factors.verify(n) {
                if let Ok(verification) = verify_factors(&factors, n, patterns, &factors.method) {
                    if verification.is_valid {
                        factors.confidence = verification.confidence;
                        return Ok(factors);
                    }
                }
            }
        }
    }

    // Try Universal Pattern as last resort before primality check
    if let Ok(mut factors) = universal_pattern_decoding(&formalization) {
        if factors.verify(n) {
            if let Ok(verification) = verify_factors(&factors, n, patterns, &factors.method) {
                if verification.is_valid {
                    factors.confidence = verification.confidence;
                    return Ok(factors);
                }
            }
        }
    }

    // If no strategy worked, check if prime
    if utils::is_probable_prime(n, 25) {
        return Ok(Factors::new(
            n.clone(),
            Number::from(1u32),
            "prime_recognition",
        ));
    }

    // Pattern execution failed
    Err(PatternError::ExecutionError(
        "No decoding strategy successfully found factors".to_string(),
    ))
}

/// Decode factors using a specific strategy
fn decode_with_strategy(
    formalization: &Formalization,
    strategy: &DecodingStrategy,
    _patterns: &[Pattern],
) -> Result<Factors> {
    match strategy {
        DecodingStrategy::ResonancePeaks => decode_resonance_peaks(formalization),
        DecodingStrategy::Eigenvalues => decode_eigenvalues(formalization),
        DecodingStrategy::HarmonicIntersection => decode_harmonic_intersection(formalization),
        DecodingStrategy::PhaseRelationships => decode_phase_relationships(formalization),
        DecodingStrategy::QuantumMaterialization => decode_quantum_materialization(formalization),
        DecodingStrategy::ModularPatterns => decode_modular_patterns(formalization),
    }
}

/// Decode using resonance peaks
fn decode_resonance_peaks(formalization: &Formalization) -> Result<Factors> {
    let n = &formalization.n;
    let sqrt_n = utils::integer_sqrt(n)?;

    for &peak_idx in &formalization.resonance_peaks {
        // Map peak index to potential factor position
        let field_size = formalization.harmonic_series.len();
        if field_size == 0 {
            continue;
        }

        let position_ratio = peak_idx as f64 / field_size as f64;
        let offset = (position_ratio * sqrt_n.to_f64().unwrap_or(1.0)) as u64;

        let candidate = &sqrt_n + &Number::from(offset);

        if candidate > Number::from(1u32) && n % &candidate == Number::from(0u32) {
            let other = n / &candidate;
            return Ok(Factors::new(candidate, other, "resonance_peaks"));
        }
    }

    Err(PatternError::ExecutionError(
        "Resonance peaks did not yield factors".to_string(),
    ))
}

/// Decode using eigenvalues of pattern matrix
fn decode_eigenvalues(formalization: &Formalization) -> Result<Factors> {
    let n = &formalization.n;
    let matrix_data = &formalization.pattern_matrix.data;
    let (rows, cols) = formalization.pattern_matrix.shape;

    // Convert to nalgebra matrix
    let matrix = DMatrix::from_row_slice(rows, cols, matrix_data);

    // Compute eigendecomposition for symmetric part
    let symmetric = &matrix * &matrix.transpose();
    let eigen = SymmetricEigen::new(symmetric);

    // Use eigenvalues to find factors
    for eigenval in eigen.eigenvalues.iter() {
        if eigenval.is_finite() && *eigenval > 0.0 {
            // Map eigenvalue to potential factor
            let factor_candidate = (*eigenval * n.to_f64().unwrap_or(1.0).sqrt()) as u64;
            let candidate = Number::from(factor_candidate);

            if candidate > Number::from(1u32) && n % &candidate == Number::from(0u32) {
                let other = n / &candidate;
                return Ok(Factors::new(candidate, other, "eigenvalues"));
            }
        }
    }

    Err(PatternError::ExecutionError(
        "Eigenvalues did not yield factors".to_string(),
    ))
}

/// Decode using harmonic intersection
fn decode_harmonic_intersection(formalization: &Formalization) -> Result<Factors> {
    let n = &formalization.n;
    let harmonics = &formalization.harmonic_series;

    if harmonics.len() < 2 {
        return Err(PatternError::ExecutionError(
            "Insufficient harmonics".to_string(),
        ));
    }

    // Look for intersections in harmonic series
    for i in 0..harmonics.len() - 1 {
        for j in i + 1..harmonics.len() {
            let ratio = harmonics[i] / harmonics[j];

            if ratio.is_finite() && ratio > 0.0 {
                // Map ratio to potential factor
                let factor_candidate = (ratio * n.to_f64().unwrap_or(1.0).powf(0.3)) as u64;
                let candidate = Number::from(factor_candidate);

                if candidate > Number::from(1u32) && n % &candidate == Number::from(0u32) {
                    let other = n / &candidate;
                    return Ok(Factors::new(candidate, other, "harmonic_intersection"));
                }
            }
        }
    }

    Err(PatternError::ExecutionError(
        "Harmonic intersection did not yield factors".to_string(),
    ))
}

/// Decode using phase relationships
fn decode_phase_relationships(formalization: &Formalization) -> Result<Factors> {
    let n = &formalization.n;
    let encoding = &formalization.universal_encoding;

    // Extract phase information
    let product_phase = encoding.get("product_phase").copied().unwrap_or(0.0);
    let unity_coupling = encoding.get("unity_coupling").copied().unwrap_or(0.0);

    if product_phase != 0.0 && unity_coupling != 0.0 {
        // Phase difference encodes factor relationship
        let phase_diff = (product_phase - unity_coupling * 2.0 * std::f64::consts::PI).abs();

        let factor_ratio = phase_diff.exp();
        let sqrt_n = utils::integer_sqrt(n)?;

        // Use ratio to find factors
        let p_estimate = sqrt_n.to_f64().unwrap_or(1.0) * factor_ratio.sqrt();
        let q_estimate = sqrt_n.to_f64().unwrap_or(1.0) / factor_ratio.sqrt();

        // Check nearby integers
        for offset in -10..=10 {
            let p_candidate = Number::from((p_estimate + offset as f64) as u64);

            if p_candidate > Number::from(1u32) && n % &p_candidate == Number::from(0u32) {
                let q = n / &p_candidate;
                return Ok(Factors::new(p_candidate, q, "phase_relationships"));
            }

            // Also check q candidates
            let q_candidate = Number::from((q_estimate + offset as f64) as u64);
            if q_candidate > Number::from(1u32) && n % &q_candidate == Number::from(0u32) {
                let p = n / &q_candidate;
                return Ok(Factors::new(p, q_candidate, "phase_relationships"));
            }
        }
    }

    Err(PatternError::ExecutionError(
        "Phase relationships did not yield factors".to_string(),
    ))
}

/// Decode using quantum materialization
fn decode_quantum_materialization(formalization: &Formalization) -> Result<Factors> {
    let n = &formalization.n;
    let encoding = &formalization.universal_encoding;

    // Use universal encoding to identify quantum neighborhood
    let sum_resonance = encoding.get("sum_resonance").copied().unwrap_or(0.0);
    let sqrt_n = utils::integer_sqrt(n)?;

    // The Pattern reveals approximate factor sum
    let sum_estimate = sum_resonance * sqrt_n.to_f64().unwrap_or(1.0) / 1.618033988749895; // φ

    // Use quadratic formula: p + q = sum, p * q = n
    let discriminant = sum_estimate * sum_estimate - 4.0 * n.to_f64().unwrap_or(1.0);

    if discriminant >= 0.0 {
        let sqrt_disc = discriminant.sqrt();
        let p = ((sum_estimate + sqrt_disc) / 2.0) as u64;
        let q = ((sum_estimate - sqrt_disc) / 2.0) as u64;

        let p_num = Number::from(p);
        let q_num = Number::from(q);

        if &p_num * &q_num == *n && p_num > Number::from(1u32) && q_num > Number::from(1u32) {
            return Ok(Factors::new(p_num, q_num, "quantum_materialization"));
        }

        // Search nearby for exact factors
        let search_radius = (sqrt_disc * 0.1).max(100.0) as u64;

        for offset in 0..=search_radius {
            for sign_p in &[-1i64, 1] {
                for sign_q in &[-1i64, 1] {
                    let p_candidate =
                        Number::from((p as i64 + sign_p * offset as i64).unsigned_abs());
                    let q_candidate =
                        Number::from((q as i64 + sign_q * offset as i64).unsigned_abs());

                    if &p_candidate * &q_candidate == *n
                        && p_candidate > Number::from(1u32)
                        && q_candidate > Number::from(1u32)
                    {
                        return Ok(Factors::new(
                            p_candidate,
                            q_candidate,
                            "quantum_materialization",
                        ));
                    }
                }
            }
        }
    }

    Err(PatternError::ExecutionError(
        "Quantum materialization did not yield factors".to_string(),
    ))
}

/// Decode using modular patterns
fn decode_modular_patterns(formalization: &Formalization) -> Result<Factors> {
    let n = &formalization.n;

    // Check small primes directly
    let primes = utils::generate_primes(1000);

    for prime in primes {
        if n % &prime == Number::from(0u32) && n != &prime {
            let other = n / &prime;

            // Check if other factor is also prime
            if utils::is_probable_prime(&other, 20) {
                return Ok(Factors::new(prime, other, "modular_patterns"));
            }
        }
    }

    Err(PatternError::ExecutionError(
        "Modular patterns did not yield factors".to_string(),
    ))
}

/// Advanced resonance-based decoding using multiple resonance fields
fn advanced_resonance_decoding(
    formalization: &Formalization,
    patterns: &[Pattern],
) -> Result<Factors> {
    let n = &formalization.n;
    let sqrt_n = utils::integer_sqrt(n)?;

    // Combine resonance information from multiple sources
    let mut resonance_map: HashMap<usize, f64> = HashMap::new();

    // Primary resonance peaks
    for &peak in &formalization.resonance_peaks {
        *resonance_map.entry(peak).or_insert(0.0) += 1.0;
    }

    // Harmonic series resonances
    for (i, &harmonic) in formalization.harmonic_series.iter().enumerate() {
        if harmonic.abs() > 0.5 {
            *resonance_map.entry(i).or_insert(0.0) += harmonic.abs();
        }
    }

    // Pattern-specific resonances
    for pattern in patterns {
        if pattern.applies_to(n) {
            match &pattern.kind {
                PatternKind::Emergent => {
                    // Balanced patterns have strong resonance near sqrt(n)
                    let center_idx = formalization.harmonic_series.len() / 2;
                    *resonance_map.entry(center_idx).or_insert(0.0) += pattern.frequency;
                },
                PatternKind::Harmonic { base_frequency, .. } => {
                    // Harmonic patterns create periodic resonances
                    let period = (base_frequency * 10.0) as usize;
                    for i in (0..formalization.harmonic_series.len()).step_by(period.max(1)) {
                        *resonance_map.entry(i).or_insert(0.0) += pattern.frequency * 0.5;
                    }
                },
                _ => {},
            }
        }
    }

    // Sort resonance points by strength
    let mut resonance_points: Vec<(usize, f64)> = resonance_map.into_iter().collect();
    resonance_points.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Try the strongest resonance points
    for (idx, strength) in resonance_points.iter().take(20) {
        if *strength < 0.5 {
            continue;
        }

        let field_size = formalization.harmonic_series.len().max(100) as f64;
        let position_ratio = *idx as f64 / field_size;

        // Multiple mapping strategies
        let mappings = vec![
            position_ratio * sqrt_n.to_f64().unwrap_or(1.0),
            position_ratio.powf(2.0) * sqrt_n.to_f64().unwrap_or(1.0),
            position_ratio.sqrt() * sqrt_n.to_f64().unwrap_or(1.0),
            (position_ratio * std::f64::consts::PI).sin() * sqrt_n.to_f64().unwrap_or(1.0),
        ];

        for mapping in mappings {
            let offset = mapping as i64;
            for delta in -5..=5 {
                let candidate = if offset + delta >= 0 {
                    &sqrt_n + &Number::from((offset + delta).unsigned_abs())
                } else {
                    let abs_val = (offset + delta).unsigned_abs();
                    if abs_val < sqrt_n.as_integer().to_u64().unwrap_or(0) {
                        &sqrt_n - &Number::from(abs_val)
                    } else {
                        continue;
                    }
                };

                if candidate > Number::from(1u32) && n % &candidate == Number::from(0u32) {
                    let other = n / &candidate;
                    return Ok(Factors::new(candidate, other, "advanced_resonance"));
                }
            }
        }
    }

    Err(PatternError::ExecutionError(
        "Advanced resonance decoding did not yield factors".to_string(),
    ))
}

/// Harmonic interference decoding
fn harmonic_interference_decoding(formalization: &Formalization) -> Result<Factors> {
    let n = &formalization.n;
    let harmonics = &formalization.harmonic_series;

    if harmonics.len() < 3 {
        return Err(PatternError::ExecutionError(
            "Insufficient harmonics for interference".to_string(),
        ));
    }

    // Find interference patterns
    let mut interference_points = Vec::new();

    for i in 0..harmonics.len() - 2 {
        for j in i + 1..harmonics.len() - 1 {
            for k in j + 1..harmonics.len() {
                // Three-wave interference
                let interference = harmonics[i] * harmonics[j] * harmonics[k];

                if interference.abs() > 0.1 {
                    // Constructive interference indicates factor relationship
                    let freq_i = i as f64 / harmonics.len() as f64;
                    let freq_j = j as f64 / harmonics.len() as f64;
                    let freq_k = k as f64 / harmonics.len() as f64;

                    // Beat frequency encodes factor information
                    let beat = (freq_i - freq_j).abs() + (freq_j - freq_k).abs();
                    interference_points.push((beat, interference));
                }
            }
        }
    }

    // Sort by interference strength
    interference_points
        .sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));

    let sqrt_n = utils::integer_sqrt(n)?;

    // Use interference patterns to find factors
    for (beat, strength) in interference_points.iter().take(10) {
        let factor_estimate = sqrt_n.to_f64().unwrap_or(1.0) * (1.0 + beat * strength.signum());

        // Search around estimate
        let search_radius = (factor_estimate * 0.01).max(100.0) as u64;

        for offset in 0..=search_radius {
            let candidate = Number::from((factor_estimate + offset as f64) as u64);

            if candidate > Number::from(1u32)
                && candidate < *n
                && n % &candidate == Number::from(0u32)
            {
                let other = n / &candidate;
                return Ok(Factors::new(candidate, other, "harmonic_interference"));
            }

            if offset > 0 {
                let candidate = Number::from((factor_estimate - offset as f64).abs() as u64);
                if candidate > Number::from(1u32) && n % &candidate == Number::from(0u32) {
                    let other = n / &candidate;
                    return Ok(Factors::new(candidate, other, "harmonic_interference"));
                }
            }
        }
    }

    Err(PatternError::ExecutionError(
        "Harmonic interference did not yield factors".to_string(),
    ))
}

/// Quantum collapse simulation decoding
fn quantum_collapse_decoding(formalization: &Formalization) -> Result<Factors> {
    let n = &formalization.n;
    let encoding = &formalization.universal_encoding;

    // Extract quantum state parameters
    let phi_component = encoding.get("phi_component").copied().unwrap_or(1.618);
    let unity_coupling = encoding.get("unity_coupling").copied().unwrap_or(0.0);
    let resonance_integral = encoding.get("resonance_integral").copied().unwrap_or(0.0);

    // Simulate quantum state collapse
    let sqrt_n = utils::integer_sqrt(n)?;
    let n_float = n.to_f64().unwrap_or(1e9);

    // Wave function parameters
    let alpha = phi_component;
    let beta = unity_coupling * 2.0 * std::f64::consts::PI;
    let gamma = resonance_integral;

    // Collapse probabilities at different positions
    let num_samples = 50;
    let mut collapse_points = Vec::new();

    for i in 0..num_samples {
        let x = i as f64 / num_samples as f64;

        // Quantum wave function
        let psi = alpha * (beta * x).cos() + gamma * (beta * x * 2.0).sin();
        let probability = psi * psi;

        if probability > 0.1 {
            // Map to factor space
            let factor_pos = sqrt_n.to_f64().unwrap_or(1.0) * (0.5 + x);
            collapse_points.push((factor_pos, probability));
        }
    }

    // Sort by probability
    collapse_points.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Try collapse points
    for (position, prob) in collapse_points.iter().take(20) {
        if *prob < 0.1 {
            continue;
        }

        let base_candidate = *position as u64;

        // Quantum uncertainty principle - search within uncertainty bounds
        let uncertainty = (n_float.ln() / (2.0 * std::f64::consts::PI)).max(10.0) as u64;

        for delta in 0..=uncertainty {
            for sign in &[-1i64, 1] {
                let candidate_val = (base_candidate as i64 + sign * delta as i64).unsigned_abs();
                let candidate = Number::from(candidate_val);

                if candidate > Number::from(1u32)
                    && candidate < *n
                    && n % &candidate == Number::from(0u32)
                {
                    let other = n / &candidate;
                    return Ok(Factors::new(candidate, other, "quantum_collapse"));
                }
            }
        }
    }

    Err(PatternError::ExecutionError(
        "Quantum collapse did not yield factors".to_string(),
    ))
}

/// Pattern-guided factor extraction
fn pattern_guided_decoding(formalization: &Formalization, patterns: &[Pattern]) -> Result<Factors> {
    let n = &formalization.n;

    // Find the most applicable pattern
    let mut best_pattern = None;
    let mut best_score = 0.0;

    for pattern in patterns {
        if pattern.applies_to(n) {
            let score = pattern.frequency;
            if score > best_score {
                best_score = score;
                best_pattern = Some(pattern);
            }
        }
    }

    if let Some(pattern) = best_pattern {
        match &pattern.kind {
            PatternKind::Power { base, exponent } => {
                // For power patterns, one factor is base^k for some k < exponent
                let base_num = Number::from(*base);
                let mut current = base_num.clone();

                for _k in 1..*exponent {
                    if n % &current == Number::from(0u32) {
                        let other = n / &current;
                        if other > Number::from(1u32) {
                            return Ok(Factors::new(current, other, "pattern_guided_power"));
                        }
                    }
                    current = &current * &base_num;
                }
            },

            PatternKind::Mersenne { p, .. } => {
                // For Mersenne numbers, check if it's a Mersenne prime
                if utils::is_probable_prime(n, 25) {
                    return Ok(Factors::new(
                        n.clone(),
                        Number::from(1u32),
                        "mersenne_prime",
                    ));
                }

                // Otherwise, use specialized Mersenne factorization
                let two = Number::from(2u32);
                for k in 2..(*p / 2) {
                    let factor = &two.pow(k) - &Number::from(1u32);
                    if n % &factor == Number::from(0u32) {
                        let other = n / &factor;
                        return Ok(Factors::new(factor, other, "pattern_guided_mersenne"));
                    }
                }
            },

            PatternKind::Fibonacci { index, .. } => {
                // For Fibonacci pattern numbers, use GCD with nearby Fibonacci numbers
                let fibs = generate_fibonacci_numbers(*index + 10);

                let start_idx = (*index).saturating_sub(5);
                let end_idx = (*index + 5).min(fibs.len());
                for fib in &fibs[start_idx..end_idx] {
                    let gcd = utils::gcd(n, fib);
                    if gcd > Number::from(1u32) && gcd < *n {
                        let other = n / &gcd;
                        return Ok(Factors::new(gcd, other, "pattern_guided_fibonacci"));
                    }
                }
            },

            _ => {
                // For other patterns, use pattern frequency as a guide
                let sqrt_n = utils::integer_sqrt(n)?;
                let offset = (sqrt_n.to_f64().unwrap_or(1.0) * pattern.frequency) as u64;

                for delta in 0..=100 {
                    // Use saturating add to prevent overflow
                    let candidate_offset = offset.saturating_add(delta);
                    let candidate = &sqrt_n + &Number::from(candidate_offset);
                    if n % &candidate == Number::from(0u32) {
                        let other = n / &candidate;
                        return Ok(Factors::new(candidate, other, "pattern_guided"));
                    }

                    if offset > delta {
                        let candidate = &sqrt_n + &Number::from(offset - delta);
                        if n % &candidate == Number::from(0u32) {
                            let other = n / &candidate;
                            return Ok(Factors::new(candidate, other, "pattern_guided"));
                        }
                    }
                }
            },
        }
    }

    Err(PatternError::ExecutionError(
        "Pattern-guided decoding did not yield factors".to_string(),
    ))
}

/// Decode using mathematical constraints
fn decode_with_constraints(
    formalization: &Formalization,
    constraints: &[PatternConstraint],
    _patterns: &[Pattern],
) -> Result<Factors> {
    let n = &formalization.n;
    let sqrt_n = utils::integer_sqrt(n)?;
    let n_float = n.to_f64().unwrap_or(1e9);

    // Build variable bindings for constraint evaluation
    let mut bindings = HashMap::new();
    bindings.insert("n".to_string(), n_float);
    bindings.insert("sqrt_n".to_string(), sqrt_n.to_f64().unwrap_or(1e4));

    // Extract bounds from constraints
    let mut p_min = 2.0;
    let mut p_max = n_float;

    for constraint in constraints {
        match (&constraint.lhs, &constraint.relation, &constraint.rhs) {
            (Expression::Variable(var), _, Expression::Constant(val)) if var == "p" => {
                match constraint.relation {
                    crate::pattern::expression::ConstraintRelation::GreaterEqual => {
                        p_min = f64::max(p_min, *val)
                    },
                    crate::pattern::expression::ConstraintRelation::LessEqual => {
                        p_max = f64::min(p_max, *val)
                    },
                    _ => {},
                }
            },
            _ => {},
        }
    }

    // Search within constrained bounds
    let search_start = f64::max(p_min, 2.0) as u64;
    let search_end = p_max.min(sqrt_n.to_f64().unwrap_or(1e9) * 2.0) as u64;

    // Adaptive step size based on number size
    let step = if n.bit_length() > 100 {
        ((search_end - search_start) / 10000).max(1)
    } else {
        1
    };

    for p_val in (search_start..=search_end).step_by(step as usize) {
        let p = Number::from(p_val);

        if n % &p == Number::from(0u32) {
            let q = n / &p;

            // Verify constraints
            bindings.insert("p".to_string(), p.to_f64().unwrap_or(0.0));
            bindings.insert("q".to_string(), q.to_f64().unwrap_or(0.0));

            let mut constraints_satisfied = true;
            for constraint in constraints {
                if let (Ok(lhs_val), Ok(rhs_val)) = (
                    constraint.lhs.evaluate(&bindings),
                    constraint.rhs.evaluate(&bindings),
                ) {
                    let satisfied = match &constraint.relation {
                        crate::pattern::expression::ConstraintRelation::Equal => {
                            (lhs_val - rhs_val).abs() < 1e-6
                        },
                        crate::pattern::expression::ConstraintRelation::LessThan => {
                            lhs_val < rhs_val
                        },
                        crate::pattern::expression::ConstraintRelation::GreaterThan => {
                            lhs_val > rhs_val
                        },
                        crate::pattern::expression::ConstraintRelation::LessEqual => {
                            lhs_val <= rhs_val
                        },
                        crate::pattern::expression::ConstraintRelation::GreaterEqual => {
                            lhs_val >= rhs_val
                        },
                        crate::pattern::expression::ConstraintRelation::Approximately(tol) => {
                            (lhs_val - rhs_val).abs() < *tol
                        },
                    };

                    if !satisfied && constraint.confidence > 0.5 {
                        constraints_satisfied = false;
                        break;
                    }
                }
            }

            if constraints_satisfied {
                return Ok(Factors::new(p, q, "constraint_based"));
            }
        }
    }

    Err(PatternError::ExecutionError(
        "Constraint-based decoding did not yield factors".to_string(),
    ))
}

/// Generate Fibonacci numbers up to index n
fn generate_fibonacci_numbers(n: usize) -> Vec<Number> {
    let mut fibs = vec![Number::from(0u32), Number::from(1u32)];

    for i in 2..=n {
        let next = &fibs[i - 1] + &fibs[i - 2];
        fibs.push(next);
    }

    fibs
}

/// Decode using Universal Pattern approach
fn universal_pattern_decoding(formalization: &Formalization) -> Result<Factors> {
    use crate::pattern::universal_pattern::{UniversalPattern, UniversalRecognition};
    
    let n = &formalization.n;
    let mut pattern = UniversalPattern::new();
    
    // Create UniversalRecognition from the formalization data
    use crate::types::{Rational, Number};
    
    // Helper to convert f64 to Rational
    let f64_to_rational = |f: f64| -> Rational {
        // Convert f64 to rational by scaling up and creating ratio
        let scaled = (f * 1_000_000_000.0) as i64;
        Rational::from_ratio(scaled.abs() as u64, 1_000_000_000u64)
    };
    
    let universal_recognition = UniversalRecognition {
        value: n.clone(),
        phi_component: formalization.universal_encoding.get("phi_component")
            .map(|&x| f64_to_rational(x))
            .unwrap_or(Rational::from_ratio(1618033989u64, 1000000000u64)),
        pi_component: formalization.universal_encoding.get("pi_component")
            .map(|&x| f64_to_rational(x))
            .unwrap_or(Rational::from_ratio(3141592654u64, 1000000000u64)),
        e_component: formalization.universal_encoding.get("e_component")
            .map(|&x| f64_to_rational(x))
            .unwrap_or(Rational::from_ratio(2718281828u64, 1000000000u64)),
        unity_phase: formalization.universal_encoding.get("unity_coupling")
            .map(|&x| f64_to_rational(x * 2.0 * std::f64::consts::PI))
            .unwrap_or(Rational::zero()),
        resonance_field: formalization.harmonic_series.iter()
            .map(|&x| Number::from((x * 1_000_000.0) as u64))
            .collect(),
    };
    
    // Formalize using universal pattern
    match pattern.formalize(universal_recognition) {
        Ok(universal_formalization) => {
            // Execute to get factors
            pattern.execute(universal_formalization)
        },
        Err(_) => {
            // Fallback: try direct universal constant relationships
            let sqrt_n = utils::integer_sqrt(n)?;
            
            // Golden ratio search
            let phi = 1.618033988749895;
            let candidates = vec![
                (sqrt_n.to_f64().unwrap_or(1.0) * phi) as u64,
                (sqrt_n.to_f64().unwrap_or(1.0) / phi) as u64,
                (sqrt_n.to_f64().unwrap_or(1.0) * phi * phi) as u64,
                (sqrt_n.to_f64().unwrap_or(1.0) / (phi * phi)) as u64,
            ];
            
            for candidate_val in candidates {
                let candidate = Number::from(candidate_val);
                if candidate > Number::from(1u32) && n % &candidate == Number::from(0u32) {
                    let other = n / &candidate;
                    return Ok(Factors::new(candidate, other, "universal_pattern_direct"));
                }
            }
            
            Err(PatternError::ExecutionError(
                "Universal pattern decoding did not yield factors".to_string(),
            ))
        }
    }
}

// ========================================================================
// Enhanced Execution Strategies (consolidated from execution_enhanced.rs)
// ========================================================================

/// Enhanced decoding using adaptive quantum search
fn enhanced_quantum_search(
    formalization: &Formalization,
    patterns: &[Pattern],
) -> Result<Factors> {
    let n = &formalization.n;

    // Create enhanced quantum region
    let mut quantum_region = create_quantum_region_from_formalization(formalization, patterns)?;

    // Get initial search candidates
    let candidates = quantum_region.get_search_candidates(20);

    for candidate in candidates {
        // Check if it's a factor
        if candidate > Number::from(1u32) && &candidate < n && n % &candidate == Number::from(0u32)
        {
            let other = n / &candidate;

            // Found a factor, update quantum region
            quantum_region.update(&candidate, true, None);

            return Ok(Factors::new(candidate, other, "enhanced_quantum_search"));
        } else {
            // Not a factor, update region
            quantum_region.update(&candidate, false, None);
        }
    }

    // If initial candidates failed, use adaptive search
    adaptive_quantum_search(formalization, &mut quantum_region, patterns)
}

/// Create quantum region from formalization
fn create_quantum_region_from_formalization(
    formalization: &Formalization,
    patterns: &[Pattern],
) -> Result<EnhancedQuantumRegion> {
    let n = &formalization.n;
    let sqrt_n = utils::integer_sqrt(n)?;

    // Determine center from resonance peaks
    let center = if !formalization.resonance_peaks.is_empty() {
        let peak_idx = formalization.resonance_peaks[0];
        let field_size = formalization.harmonic_series.len().max(100) as f64;
        let ratio = peak_idx as f64 / field_size;
        &sqrt_n * Number::from((1.0 + ratio * 0.1) as u64)
    } else {
        sqrt_n
    };

    // Create region with pattern analysis
    let mut region = EnhancedQuantumRegion::from_pattern_analysis(center, patterns, n);

    // Set distribution type based on harmonic analysis
    if has_multiple_harmonics(&formalization.harmonic_series) {
        region.distribution_type = DistributionType::MultiModal;
    }

    Ok(region)
}

/// Check if harmonic series indicates multiple modes
fn has_multiple_harmonics(harmonics: &[f64]) -> bool {
    if harmonics.len() < 3 {
        return false;
    }

    // Count significant harmonics
    let threshold = 0.3;
    let significant = harmonics.iter().filter(|&&h| h.abs() > threshold).count();

    significant > 2
}

/// Adaptive quantum search with learning
fn adaptive_quantum_search(
    formalization: &Formalization,
    quantum_region: &mut EnhancedQuantumRegion,
    patterns: &[Pattern],
) -> Result<Factors> {
    let n = &formalization.n;
    let max_iterations = 100;

    for iteration in 0..max_iterations {
        // Adjust search strategy based on confidence
        let num_candidates = if quantum_region.confidence_metrics.overall > 0.7 {
            5 // High confidence, focus search
        } else if quantum_region.confidence_metrics.overall > 0.4 {
            10 // Medium confidence
        } else {
            20 // Low confidence, explore more
        };

        // Get next batch of candidates
        let candidates = quantum_region.get_search_candidates(num_candidates);

        for candidate in candidates {
            if candidate > Number::from(1u32) && &candidate < n {
                if n % &candidate == Number::from(0u32) {
                    let other = n / &candidate;

                    // Found factor! Update region with success
                    let matching_pattern = patterns.iter().find(|p| p.applies_to(n));
                    quantum_region.update(&candidate, true, matching_pattern);

                    // Verify and return
                    let factors =
                        Factors::new(candidate.clone(), other.clone(), "adaptive_quantum_search");
                    if let Ok(verification) =
                        verify_factors(&factors, n, patterns, "adaptive_quantum_search")
                    {
                        if verification.is_valid {
                            return Ok(factors);
                        }
                    }
                } else {
                    // Update region with failure
                    quantum_region.update(&candidate, false, None);
                }
            }
        }

        // Check if we should change strategy
        if iteration % 10 == 9 {
            adjust_search_strategy(quantum_region, iteration);
        }
    }

    Err(PatternError::ExecutionError(
        "Adaptive quantum search exhausted".to_string(),
    ))
}

/// Adjust search strategy based on performance
fn adjust_search_strategy(quantum_region: &mut EnhancedQuantumRegion, iteration: usize) {
    let confidence = quantum_region.confidence_metrics.overall;

    if confidence < 0.3 && iteration > 20 {
        // Low confidence after many iterations - try different distribution
        match quantum_region.distribution_type {
            DistributionType::Gaussian => {
                quantum_region.distribution_type = DistributionType::MultiModal;
            },
            DistributionType::MultiModal => {
                quantum_region.distribution_type = DistributionType::Empirical;
            },
            _ => {},
        }
    }

    // Adjust radius based on miss rate
    if quantum_region.observations.total_observations > 0 {
        let success_rate = quantum_region.observations.successes.len() as f64
            / quantum_region.observations.total_observations as f64;

        if success_rate < 0.01 {
            // Very low success rate - expand search
            quantum_region.radius.growth_rate = 2.0;
        }
    }
}

/// Multi-level quantum search
fn multi_level_quantum_search(
    formalization: &Formalization,
    patterns: &[Pattern],
) -> Result<Factors> {
    let n = &formalization.n;

    // Level 1: Coarse search with large regions
    let coarse_region = create_coarse_quantum_region(n, patterns)?;
    if let Ok(factors) = search_quantum_level(formalization, coarse_region, patterns, "coarse") {
        return Ok(factors);
    }

    // Level 2: Medium resolution
    let medium_regions = create_medium_quantum_regions(formalization, patterns)?;
    for region in medium_regions {
        if let Ok(factors) = search_quantum_level(formalization, region, patterns, "medium") {
            return Ok(factors);
        }
    }

    // Level 3: Fine-grained search
    let fine_regions = create_fine_quantum_regions(formalization, patterns)?;
    for region in fine_regions {
        if let Ok(factors) = search_quantum_level(formalization, region, patterns, "fine") {
            return Ok(factors);
        }
    }

    Err(PatternError::ExecutionError(
        "Multi-level quantum search failed".to_string(),
    ))
}

/// Create coarse quantum region for initial search
fn create_coarse_quantum_region(
    n: &Number,
    _patterns: &[Pattern],
) -> Result<EnhancedQuantumRegion> {
    let sqrt_n = utils::integer_sqrt(n)?;
    let radius = &sqrt_n / &Number::from(10u32);

    Ok(EnhancedQuantumRegion::new(sqrt_n, radius, n))
}

/// Create medium resolution quantum regions
fn create_medium_quantum_regions(
    formalization: &Formalization,
    _patterns: &[Pattern],
) -> Result<Vec<EnhancedQuantumRegion>> {
    let n = &formalization.n;
    let sqrt_n = utils::integer_sqrt(n)?;
    let mut regions = Vec::new();

    // Create regions around resonance peaks
    for &peak_idx in formalization.resonance_peaks.iter().take(3) {
        let field_size = formalization.harmonic_series.len().max(100) as f64;
        let ratio = peak_idx as f64 / field_size;
        let center = &sqrt_n * Number::from((1.0 + ratio * 0.2) as u64);
        let radius = &sqrt_n / &Number::from(50u32);

        let region = EnhancedQuantumRegion::new(center, radius, n);
        regions.push(region);
    }

    Ok(regions)
}

/// Create fine-grained quantum regions
fn create_fine_quantum_regions(
    formalization: &Formalization,
    patterns: &[Pattern],
) -> Result<Vec<EnhancedQuantumRegion>> {
    let n = &formalization.n;
    let mut regions = Vec::new();

    // Create regions based on pattern predictions
    for pattern in patterns.iter().filter(|p| p.applies_to(n)) {
        match &pattern.kind {
            PatternKind::Emergent => {
                let sqrt_n = utils::integer_sqrt(n)?;
                let center = sqrt_n.clone();
                let radius = Number::from(1000u32);
                regions.push(EnhancedQuantumRegion::new(center, radius, n));
            },
            PatternKind::Harmonic { base_frequency, .. } => {
                let sqrt_n = utils::integer_sqrt(n)?;
                let offset = (base_frequency * 1000.0) as u64;
                let center = &sqrt_n + &Number::from(offset);
                let radius = Number::from(offset / 10);

                let mut region = EnhancedQuantumRegion::new(center, radius, n);
                region.distribution_type = DistributionType::MultiModal;
                regions.push(region);
            },
            _ => {},
        }
    }

    Ok(regions)
}

/// Search within a quantum level
fn search_quantum_level(
    formalization: &Formalization,
    mut quantum_region: EnhancedQuantumRegion,
    _patterns: &[Pattern],
    level: &str,
) -> Result<Factors> {
    let n = &formalization.n;
    let max_attempts = match level {
        "coarse" => 50,
        "medium" => 100,
        "fine" => 200,
        _ => 100,
    };

    for _ in 0..max_attempts {
        let candidates = quantum_region.get_search_candidates(10);

        for candidate in &candidates {
            if candidate > &Number::from(1u32)
                && candidate < n
                && n % candidate == Number::from(0u32)
            {
                let other = n / candidate;
                let method = format!("quantum_search_{}", level);
                return Ok(Factors::new(candidate.clone(), other, method));
            }
        }

        // Update quantum region based on failures
        for candidate in &candidates {
            quantum_region.update(candidate, false, None);
        }
    }

    Err(PatternError::ExecutionError(format!(
        "Quantum search at {} level failed",
        level
    )))
}

/// Parallel quantum search across multiple regions
fn parallel_quantum_search(
    formalization: &Formalization,
    patterns: &[Pattern],
    num_regions: usize,
) -> Result<Factors> {
    use rayon::prelude::*;

    let n = &formalization.n;

    // Create multiple quantum regions
    let regions = create_parallel_quantum_regions(formalization, patterns, num_regions)?;

    // Search in parallel
    let result = regions.into_par_iter().find_map_any(|region| {
        // Each thread searches its region
        for _ in 0..50 {
            let candidates = region.get_search_candidates(5);

            for candidate in candidates {
                if candidate > Number::from(1u32)
                    && &candidate < n
                    && n % &candidate == Number::from(0u32)
                {
                    let other = n / &candidate;
                    return Some(Factors::new(candidate, other, "parallel_quantum_search"));
                }
            }
        }
        None
    });

    result.ok_or_else(|| PatternError::ExecutionError("Parallel quantum search failed".to_string()))
}

/// Create regions for parallel search
fn create_parallel_quantum_regions(
    formalization: &Formalization,
    _patterns: &[Pattern],
    num_regions: usize,
) -> Result<Vec<EnhancedQuantumRegion>> {
    let n = &formalization.n;
    let sqrt_n = utils::integer_sqrt(n)?;
    let mut regions = Vec::new();

    // Create regions at different scales
    for i in 0..num_regions {
        let offset_factor = 1.0 + (i as f64 * 0.1);
        let center = &sqrt_n * Number::from(offset_factor as u64);
        let radius = &sqrt_n / &Number::from(20u32);

        let mut region = EnhancedQuantumRegion::new(center, radius, n);

        // Vary distribution types
        region.distribution_type = match i % 3 {
            0 => DistributionType::Gaussian,
            1 => DistributionType::MultiModal,
            _ => DistributionType::Skewed,
        };

        regions.push(region);
    }

    Ok(regions)
}
