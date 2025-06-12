//! Stage 1: Recognition - Extract pattern signature from number
//!
//! This module implements pattern recognition based on empirical observation.

use crate::pattern::RecognitionParams;
use crate::types::quantum_enhanced::EnhancedQuantumRegion;
use crate::types::{
    Number, PatternSignature, PatternType, QuantumRegion, Recognition, UniversalConstant,
};
use crate::utils;
use crate::Result;
use std::f64::consts::{E, PI};

/// Recognize the pattern signature of a number (with patterns)
pub fn recognize(n: Number, patterns: &[crate::types::Pattern]) -> Result<Recognition> {
    // Extract constants from patterns
    let constants = crate::observer::ConstantDiscovery::extract(patterns);
    let params = RecognitionParams::default();

    recognize_with_params(&n, &constants, &params)
}

/// Recognize the pattern signature of a number (with explicit parameters)
pub fn recognize_with_params(
    n: &Number,
    constants: &[UniversalConstant],
    params: &RecognitionParams,
) -> Result<Recognition> {
    // Extract signature
    let signature = extract_signature(n, constants, params)?;

    // Identify pattern type
    let pattern_type = identify_pattern_type(&signature)?;

    // Calculate confidence
    let confidence = calculate_confidence(&signature, pattern_type);

    // Detect quantum neighborhood if applicable
    let quantum_neighborhood = detect_quantum_region(n, &signature, pattern_type)?;

    // Create recognition
    let mut recognition = Recognition::new(signature, pattern_type).with_confidence(confidence);

    if let Some(region) = quantum_neighborhood {
        recognition = recognition.with_quantum_region(region);
    }

    Ok(recognition)
}

/// Extract the pattern signature from a number
fn extract_signature(
    n: &Number,
    constants: &[UniversalConstant],
    params: &RecognitionParams,
) -> Result<PatternSignature> {
    let mut signature = PatternSignature::new(n.clone());

    // Extract universal components
    extract_universal_components(&mut signature, n, constants)?;

    // Extract modular DNA
    let modular_dna = extract_modular_dna(n, params.modular_prime_count)?;
    signature.set_modular_dna(modular_dna);

    // Generate resonance field
    let resonance = generate_resonance_field(n, &signature, params.resonance_field_size)?;
    signature.set_resonance(resonance);

    // Extract emergent features
    extract_emergent_features(&mut signature, n)?;

    // Extract multi-dimensional components
    let multidim = signature.extract_multidimensional();
    for (dim_name, values) in multidim {
        // Store key statistics from each dimension
        if !values.is_empty() {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            signature.add_component(format!("{}_mean", dim_name), mean);
            signature.add_component(format!("{}_max", dim_name), max);
        }
    }

    // Check for advanced patterns
    let advanced_patterns = crate::pattern::advanced::detect_advanced_patterns(n);
    for pattern in advanced_patterns {
        signature.add_emergent_feature(
            format!("advanced_pattern_{}", pattern.id),
            serde_json::json!({
                "kind": format!("{:?}", pattern.kind),
                "description": pattern.description,
                "frequency": pattern.frequency
            }),
        );
    }

    Ok(signature)
}

/// Extract universal components based on discovered constants
fn extract_universal_components(
    signature: &mut PatternSignature,
    n: &Number,
    constants: &[UniversalConstant],
) -> Result<()> {
    // Golden ratio component (φ)
    let phi = constants
        .iter()
        .find(|c| c.name == "phi")
        .map(|c| c.value)
        .unwrap_or(1.618033988749895);

    let log_n = n.to_f64().map(|v| v.ln()).unwrap_or_else(|| n.bit_length() as f64 * std::f64::consts::LN_2);

    let phi_component = log_n / phi.ln();
    signature.add_component("phi_component", phi_component);

    // Pi component (π)
    let pi_component = (n.as_integer().to_u64().unwrap_or(0) % 1000000) as f64 / (PI * 100000.0);
    signature.add_component("pi_component", pi_component);

    // E component (e)
    let e_component = log_n / E;
    signature.add_component("e_component", e_component);

    // Unity phase
    let unity_phase = (n.as_integer().to_u64().unwrap_or(0) as f64 * phi) % (2.0 * PI);
    signature.add_component("unity_phase", unity_phase);

    // Add any other discovered constants
    for constant in constants {
        if constant.universality > 0.5 {
            let component = log_n / constant.value;
            signature.add_component(format!("{}_component", constant.name), component);
        }
    }

    Ok(())
}

/// Extract modular DNA - the number's identity across prime moduli
fn extract_modular_dna(n: &Number, prime_count: usize) -> Result<Vec<u64>> {
    let primes = utils::generate_primes(prime_count);

    let modular_dna: Vec<u64> =
        primes.iter().map(|p| (n % p).as_integer().to_u64().unwrap_or(0)).collect();

    Ok(modular_dna)
}

/// Generate resonance field for the number
fn generate_resonance_field(
    n: &Number,
    signature: &PatternSignature,
    field_size: usize,
) -> Result<Vec<f64>> {
    let mut field = vec![0.0; field_size];

    let phi = signature.get_component("phi_component").unwrap_or(1.0);
    let pi = signature.get_component("pi_component").unwrap_or(1.0);
    let e = signature.get_component("e_component").unwrap_or(1.0);

    let sqrt_n = utils::integer_sqrt(n)?;
    let damping_factor = if n.bit_length() > 100 {
        1.0 / (n.bit_length() as f64).sqrt()
    } else {
        1.0 / sqrt_n.to_f64().unwrap_or(1.0).sqrt()
    };

    for i in 0..field_size {
        let x = i as f64 / field_size as f64;

        // Universal harmonic
        let harmonic = phi * (pi * x * 2.0 * PI).sin() + e * (phi * x * 2.0 * PI).cos();

        // Apply damping
        field[i] = harmonic * (-(i as f64) * damping_factor).exp();
    }

    // Normalize
    let max_val = field.iter().map(|v| v.abs()).fold(0.0, f64::max);
    if max_val > 0.0 {
        for v in &mut field {
            *v /= max_val;
        }
    }

    Ok(field)
}

/// Extract emergent features that appear in observations
fn extract_emergent_features(signature: &mut PatternSignature, n: &Number) -> Result<()> {
    // Bit length feature
    signature.add_emergent_feature("bit_length", serde_json::json!(n.bit_length()));

    // Digit sum in base 10
    let digit_sum: u32 = n.to_string().chars().filter_map(|c| c.to_digit(10)).sum();
    signature.add_emergent_feature("digit_sum", serde_json::json!(digit_sum));

    // Last digits pattern
    let last_digits = n.as_integer().to_u64().unwrap_or(0) % 1000;
    signature.add_emergent_feature("last_digits", serde_json::json!(last_digits));

    Ok(())
}

/// Identify the pattern type from signature
fn identify_pattern_type(signature: &PatternSignature) -> Result<PatternType> {
    let n = &signature.value;

    // Check for small factors first
    let small_primes = utils::generate_primes(100);
    for p in &small_primes {
        if n % p == Number::from(0u32) && n != p {
            return Ok(PatternType::SmallFactor);
        }
    }
    
    // Quick primality check for small numbers
    if n.bit_length() <= 32 && utils::is_probable_prime(n, 20) {
        return Ok(PatternType::Prime);
    }

    // Check modular DNA for patterns
    let dna_variety = signature.modular_dna.iter().collect::<std::collections::HashSet<_>>().len()
        as f64
        / signature.modular_dna.len() as f64;

    // High variety suggests prime
    if dna_variety > 0.9 {
        // Additional primality indicators
        let resonance_sum: f64 = signature.resonance.iter().map(|v| v.abs()).sum();
        if resonance_sum < 0.1 {
            return Ok(PatternType::Prime);
        }
    }

    // Check for harmonic pattern (large imbalance)
    let resonance_variance: f64 =
        signature.resonance.iter().map(|v| v * v).sum::<f64>() / signature.resonance.len() as f64;

    if resonance_variance > 0.5 {
        return Ok(PatternType::Harmonic);
    }

    // Check for perfect square
    let sqrt_n = utils::integer_sqrt(n)?;
    if &sqrt_n * &sqrt_n == *n && utils::is_probable_prime(&sqrt_n, 20) {
        return Ok(PatternType::Square);
    }

    // Default to balanced for semiprimes
    if n.bit_length() > 10 && dna_variety > 0.5 {
        return Ok(PatternType::Balanced);
    }

    Ok(PatternType::Unknown)
}

/// Calculate confidence in the recognition
fn calculate_confidence(signature: &PatternSignature, pattern_type: PatternType) -> f64 {
    let mut confidence: f64 = 0.0;

    // Base confidence from pattern type
    confidence += match pattern_type {
        PatternType::SmallFactor => 1.0,
        PatternType::Square => 0.95,
        PatternType::Prime => 0.8,
        PatternType::Balanced => 0.7,
        PatternType::Harmonic => 0.6,
        PatternType::Unknown => 0.3,
    };

    // Adjust based on resonance field quality
    let resonance_strength: f64 = signature.resonance.iter().map(|v| v.abs()).fold(0.0, f64::max);

    if resonance_strength > 0.8 {
        confidence *= 1.1;
    }

    // Adjust based on component values
    let component_sum: f64 = signature.components.values().sum();
    if component_sum.is_finite() && component_sum.abs() < 1000.0 {
        confidence *= 1.05;
    }

    confidence.min(1.0)
}

/// Detect quantum neighborhood where factors exist
fn detect_quantum_region(
    n: &Number,
    signature: &PatternSignature,
    pattern_type: PatternType,
) -> Result<Option<QuantumRegion>> {
    match pattern_type {
        PatternType::Balanced => {
            // For balanced semiprimes, quantum region is near sqrt(n)
            let sqrt_n = utils::integer_sqrt(n)?;

            // Estimate offset from pattern
            let phi_component = signature.get_component("phi_component").unwrap_or(1.0);
            let offset_estimate = sqrt_n.to_f64().unwrap_or(1.0) * phi_component / 1000.0;

            let center = &sqrt_n + &Number::from(offset_estimate as u64);
            let radius = Number::from((offset_estimate * 0.1).max(1000.0) as u64);

            let mut region = QuantumRegion::new(center, radius);
            region.confidence = 0.8;

            Ok(Some(region))
        },
        PatternType::Harmonic => {
            // For harmonic patterns, factors are far apart
            let sqrt_n = utils::integer_sqrt(n)?;
            let center = &sqrt_n / &Number::from(10u32);
            let radius = &center / &Number::from(2u32);

            let mut region = QuantumRegion::new(center, radius);
            region.confidence = 0.6;

            Ok(Some(region))
        },
        _ => Ok(None),
    }
}

/// Detect enhanced quantum neighborhood with adaptive distributions
pub fn detect_enhanced_quantum_region(
    n: &Number,
    signature: &PatternSignature,
    pattern_type: PatternType,
    patterns: &[crate::types::Pattern],
) -> Result<Option<EnhancedQuantumRegion>> {
    match pattern_type {
        PatternType::Balanced => {
            // For balanced semiprimes, quantum region is near sqrt(n)
            let sqrt_n = utils::integer_sqrt(n)?;

            // Estimate offset from pattern
            let phi_component = signature.get_component("phi_component").unwrap_or(1.0);
            let offset_estimate = sqrt_n.to_f64().unwrap_or(1.0) * phi_component / 1000.0;

            let center = &sqrt_n + &Number::from(offset_estimate as u64);
            let _initial_radius = Number::from((offset_estimate * 0.1).max(1000.0) as u64);

            let region = EnhancedQuantumRegion::from_pattern_analysis(center, patterns, n);
            Ok(Some(region))
        },
        PatternType::Harmonic => {
            // For harmonic patterns, use multi-modal distribution
            let sqrt_n = utils::integer_sqrt(n)?;
            let center = &sqrt_n / &Number::from(10u32);

            let mut region = EnhancedQuantumRegion::from_pattern_analysis(center, patterns, n);
            region.distribution_type = crate::types::quantum_enhanced::DistributionType::MultiModal;

            Ok(Some(region))
        },
        PatternType::SmallFactor => {
            // For numbers with small factors, use skewed distribution
            let _sqrt_n = utils::integer_sqrt(n)?;
            let center = Number::from(1000u32); // Start search near small primes

            let mut region = EnhancedQuantumRegion::new(center, Number::from(500u32), n);
            region.distribution_type = crate::types::quantum_enhanced::DistributionType::Skewed;

            Ok(Some(region))
        },
        _ => Ok(None),
    }
}
