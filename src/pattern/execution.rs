//! Stage 3: Execution - Decode factors from formalized pattern
//!
//! This module executes the pattern to manifest factors.

use crate::error::PatternError;
use crate::types::recognition::{DecodingStrategy, Factors, Formalization};
use crate::types::{Number, Pattern};
use crate::utils;
use crate::Result;
use nalgebra::{DMatrix, SymmetricEigen};

/// Execute the formalized pattern to decode factors
pub fn execute(formalization: Formalization, patterns: &[Pattern]) -> Result<Factors> {
    let n = &formalization.n;

    // Try each decoding strategy
    for strategy in &formalization.strategies {
        match decode_with_strategy(&formalization, strategy) {
            Ok(factors) if factors.verify(n) => return Ok(factors),
            _ => continue, // Try next strategy
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
                        Number::from((p as i64 + sign_p * offset as i64).abs() as u64);
                    let q_candidate =
                        Number::from((q as i64 + sign_q * offset as i64).abs() as u64);

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
