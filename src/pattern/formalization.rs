//! Stage 2: Formalization - Express recognition in mathematical language
//!
//! This module translates pattern recognition into mathematical structures.

use crate::pattern::expression::{
    create_pattern_constraints, pattern_to_expression, Expression, PatternConstraint,
};
use crate::types::recognition::{
    DecodingStrategy, Formalization, PatternMatrix, PatternType, Recognition,
};
use crate::types::{Pattern, UniversalConstant};
use crate::Result;
use ndarray::Array2;
use std::collections::HashMap;

/// Formalize the recognition into mathematical expression
pub fn formalize(
    recognition: Recognition,
    patterns: &[Pattern],
    constants: &[UniversalConstant],
) -> Result<Formalization> {
    let n = recognition.signature.value.clone();

    // Build universal encoding
    let universal_encoding = build_universal_encoding(&recognition, constants)?;

    // Find resonance peaks
    let resonance_peaks = find_resonance_peaks(&recognition.signature.resonance)?;

    // Compute harmonic series
    let harmonic_series = compute_harmonic_series(&recognition, constants)?;

    // Build pattern matrix
    let pattern_matrix = build_pattern_matrix(&recognition, patterns)?;

    // Generate mathematical constraints
    let constraints = generate_constraints(&recognition, patterns)?;

    // Select and rank strategies
    let strategies = select_and_rank_strategies(&recognition, &constraints)?;

    // Create formalization with constraints
    let mut formalization = Formalization::new(
        n,
        universal_encoding,
        resonance_peaks,
        harmonic_series,
        pattern_matrix,
        strategies,
    );

    // Store constraints
    formalization.add_constraints(constraints);

    Ok(formalization)
}

/// Build universal encoding from recognition
fn build_universal_encoding(
    recognition: &Recognition,
    constants: &[UniversalConstant],
) -> Result<HashMap<String, f64>> {
    let mut encoding = HashMap::new();

    // Include all components from signature
    for (name, value) in &recognition.signature.components {
        encoding.insert(name.clone(), *value);
    }

    // Add derived encodings
    let phi = recognition.signature.get_component("phi_component").unwrap_or(1.0);
    let pi = recognition.signature.get_component("pi_component").unwrap_or(1.0);
    let e = recognition.signature.get_component("e_component").unwrap_or(1.0);

    // Product phase
    encoding.insert(
        "product_phase".to_string(),
        (phi * pi) % (2.0 * std::f64::consts::PI),
    );

    // Sum resonance
    encoding.insert("sum_resonance".to_string(), phi + pi + e);

    // Difference field
    encoding.insert("difference_field".to_string(), (phi - e).abs());

    // Unity coupling
    let unity = recognition.signature.get_component("unity_phase").unwrap_or(0.0);
    encoding.insert(
        "unity_coupling".to_string(),
        unity / (2.0 * std::f64::consts::PI),
    );

    // Resonance integral
    let resonance_sum: f64 = recognition.signature.resonance.iter().sum();
    let resonance_integral = resonance_sum / recognition.signature.resonance.len() as f64;
    encoding.insert("resonance_integral".to_string(), resonance_integral);

    // Add constant-based encodings
    for constant in constants {
        if constant.universality > 0.5 {
            let value = recognition
                .signature
                .get_component(&format!("{}_component", constant.name))
                .unwrap_or(0.0);
            encoding.insert(
                format!("{}_encoding", constant.name),
                value * constant.value,
            );
        }
    }

    Ok(encoding)
}

/// Find peaks in the resonance field
fn find_resonance_peaks(resonance: &[f64]) -> Result<Vec<usize>> {
    let mut peaks = Vec::new();

    if resonance.len() < 3 {
        return Ok(peaks);
    }

    // Find local maxima
    for i in 1..resonance.len() - 1 {
        if resonance[i] > resonance[i - 1] && resonance[i] > resonance[i + 1] {
            peaks.push(i);
        }
    }

    // Sort by amplitude
    peaks.sort_by(|&a, &b| {
        resonance[b]
            .abs()
            .partial_cmp(&resonance[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Keep top peaks
    peaks.truncate(10);

    Ok(peaks)
}

/// Compute harmonic series expansion
fn compute_harmonic_series(
    recognition: &Recognition,
    constants: &[UniversalConstant],
) -> Result<Vec<f64>> {
    let mut series = Vec::new();
    let n = &recognition.signature.value;

    // Base frequency from phi component
    let base_freq = recognition.signature.get_component("phi_component").unwrap_or(1.0);

    // Generate harmonics
    for k in 1i32..=20 {
        let harmonic = base_freq.powi(k)
            + recognition
                .signature
                .get_component("pi_component")
                .unwrap_or(0.0)
                .powi(k)
            + recognition.signature.get_component("e_component").unwrap_or(0.0).powi(k);

        // Scale by k-th root of n
        let scale = if n.bit_length() > 100 {
            1.0 / (k as f64 * n.bit_length() as f64).sqrt()
        } else {
            n.to_f64().unwrap_or(1.0).powf(1.0 / k as f64)
        };

        series.push(harmonic * scale);
    }

    // Apply constant modulation
    for (i, constant) in constants.iter().enumerate() {
        if constant.universality > 0.7 && i < series.len() {
            series[i] *= constant.value;
        }
    }

    Ok(series)
}

/// Build pattern matrix representation
fn build_pattern_matrix(recognition: &Recognition, patterns: &[Pattern]) -> Result<PatternMatrix> {
    let size = 4; // 4x4 matrix
    let mut matrix = Array2::zeros((size, size));

    // Fill with signature components
    let components: Vec<f64> = recognition.signature.components.values().copied().collect();
    for (i, &value) in components.iter().take(size * size).enumerate() {
        let row = i / size;
        let col = i % size;
        matrix[[row, col]] = value;
    }

    // Add resonance field data
    let field_len = recognition.signature.resonance.len();
    if field_len > 0 {
        matrix[[3, 0]] = recognition.signature.resonance[0];
        matrix[[3, 1]] = recognition.signature.resonance[field_len / 2];
        matrix[[3, 2]] = recognition.signature.resonance[field_len - 1];
        matrix[[3, 3]] = recognition.signature.resonance.iter().sum::<f64>() / field_len as f64;
    }

    // Apply pattern-specific transformations
    for pattern in patterns {
        if pattern.applies_to(&recognition.signature.value) {
            // Scale by pattern frequency
            let scale = pattern.frequency;
            for value in matrix.iter_mut() {
                *value *= scale;
            }
        }
    }

    Ok(PatternMatrix::from_array(matrix))
}

/// Select decoding strategies based on pattern type
fn select_strategies(pattern_type: PatternType) -> Vec<DecodingStrategy> {
    match pattern_type {
        PatternType::Balanced => vec![
            DecodingStrategy::ResonancePeaks,
            DecodingStrategy::Eigenvalues,
            DecodingStrategy::HarmonicIntersection,
            DecodingStrategy::QuantumMaterialization,
        ],
        PatternType::Harmonic => vec![
            DecodingStrategy::HarmonicIntersection,
            DecodingStrategy::ModularPatterns,
            DecodingStrategy::PhaseRelationships,
        ],
        PatternType::SmallFactor => vec![
            DecodingStrategy::ModularPatterns,
            DecodingStrategy::ResonancePeaks,
        ],
        PatternType::Square => vec![
            DecodingStrategy::Eigenvalues,
            DecodingStrategy::ResonancePeaks,
        ],
        PatternType::Prime => vec![
            // No strategies for primes
        ],
        PatternType::Unknown => vec![
            DecodingStrategy::ResonancePeaks,
            DecodingStrategy::Eigenvalues,
            DecodingStrategy::ModularPatterns,
        ],
    }
}

/// Generate mathematical constraints from recognition
fn generate_constraints(
    recognition: &Recognition,
    patterns: &[Pattern],
) -> Result<Vec<PatternConstraint>> {
    let mut all_constraints = Vec::new();

    // Find matching patterns
    for pattern in patterns {
        if pattern.applies_to(&recognition.signature.value) {
            // Convert pattern to expression
            let expr = pattern_to_expression(pattern);

            // Create pattern-specific constraints
            let constraints = create_pattern_constraints(pattern, &[]);
            all_constraints.extend(constraints);

            // Add expression-based constraint
            all_constraints.push(PatternConstraint {
                id: format!("pattern_{}_expr", pattern.id),
                lhs: expr.clone(),
                relation: crate::pattern::expression::ConstraintRelation::Equal,
                rhs: Expression::var("n"),
                confidence: pattern.frequency,
            });
        }
    }

    // Add recognition-based constraints
    if let Some(quantum_region) = &recognition.quantum_neighborhood {
        // Quantum neighborhood constraint
        let (lower, upper) = quantum_region.bounds();
        all_constraints.push(PatternConstraint {
            id: "quantum_bounds".to_string(),
            lhs: Expression::var("p"),
            relation: crate::pattern::expression::ConstraintRelation::GreaterEqual,
            rhs: Expression::Constant(lower.to_f64().unwrap_or(1.0)),
            confidence: quantum_region.confidence,
        });
        all_constraints.push(PatternConstraint {
            id: "quantum_bounds_upper".to_string(),
            lhs: Expression::var("p"),
            relation: crate::pattern::expression::ConstraintRelation::LessEqual,
            rhs: Expression::Constant(upper.to_f64().unwrap_or(1e9)),
            confidence: quantum_region.confidence,
        });
    }

    Ok(all_constraints)
}

/// Select and rank strategies based on constraints
fn select_and_rank_strategies(
    recognition: &Recognition,
    constraints: &[PatternConstraint],
) -> Result<Vec<DecodingStrategy>> {
    let mut strategies = select_strategies(recognition.pattern_type);

    // Rank strategies based on constraint confidence
    let mut strategy_scores: HashMap<DecodingStrategy, f64> = HashMap::new();

    for strategy in &strategies {
        let mut score = match strategy {
            DecodingStrategy::ResonancePeaks => 0.8,
            DecodingStrategy::Eigenvalues => 0.7,
            DecodingStrategy::HarmonicIntersection => 0.6,
            DecodingStrategy::ModularPatterns => 0.5,
            DecodingStrategy::PhaseRelationships => 0.4,
            DecodingStrategy::QuantumMaterialization => 0.9,
        };

        // Adjust score based on constraints
        for constraint in constraints {
            if constraint.confidence > 0.8 {
                score *= 1.1;
            }
        }

        // Adjust based on recognition confidence
        score *= recognition.confidence;

        strategy_scores.insert(*strategy, score);
    }

    // Sort strategies by score
    strategies.sort_by(|a, b| {
        strategy_scores
            .get(b)
            .unwrap_or(&0.0)
            .partial_cmp(strategy_scores.get(a).unwrap_or(&0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Add fallback strategies if needed
    if strategies.len() < 3 {
        if !strategies.contains(&DecodingStrategy::ModularPatterns) {
            strategies.push(DecodingStrategy::ModularPatterns);
        }
        if !strategies.contains(&DecodingStrategy::ResonancePeaks) {
            strategies.push(DecodingStrategy::ResonancePeaks);
        }
    }

    Ok(strategies)
}
