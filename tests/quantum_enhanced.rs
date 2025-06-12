//! Tests for enhanced quantum search functionality

use rust_pattern_solver::observer::ConstantDiscovery;
use rust_pattern_solver::pattern::{
    execution_enhanced::{
        enhanced_quantum_search, multi_level_quantum_search, parallel_quantum_search,
    },
    formalization::formalize,
    recognition::recognize,
};
use rust_pattern_solver::types::pattern::ScaleRange;
use rust_pattern_solver::types::{Number, Pattern, PatternKind};

#[test]
fn test_enhanced_quantum_search_balanced() {
    // Test with balanced semiprime 143 = 11 × 13
    let n = Number::from(143u32);
    let patterns = create_test_patterns();

    // Create recognition and formalization
    let recognition = recognize(n.clone(), &patterns).unwrap();
    let constants = ConstantDiscovery::extract(&patterns);
    let formalization = formalize(recognition, &patterns, &constants).unwrap();

    // Try enhanced quantum search
    let result = enhanced_quantum_search(&formalization, &patterns);
    assert!(result.is_ok());

    let factors = result.unwrap();
    assert!(factors.verify(&n));
    assert!(
        (factors.p == Number::from(11u32) && factors.q == Number::from(13u32))
            || (factors.p == Number::from(13u32) && factors.q == Number::from(11u32))
    );
}

#[test]
fn test_multi_level_quantum_search() {
    // Test with larger semiprime 323 = 17 × 19
    let n = Number::from(323u32);
    let patterns = create_test_patterns();

    let recognition = recognize(n.clone(), &patterns).unwrap();
    let constants = ConstantDiscovery::extract(&patterns);
    let formalization = formalize(recognition, &patterns, &constants).unwrap();

    let result = multi_level_quantum_search(&formalization, &patterns);
    assert!(result.is_ok());

    let factors = result.unwrap();
    assert!(factors.verify(&n));
}

#[test]
fn test_parallel_quantum_search() {
    // Test with 899 = 29 × 31
    let n = Number::from(899u32);
    let patterns = create_test_patterns();

    let recognition = recognize(n.clone(), &patterns).unwrap();
    let constants = ConstantDiscovery::extract(&patterns);
    let formalization = formalize(recognition, &patterns, &constants).unwrap();

    let result = parallel_quantum_search(&formalization, &patterns);
    assert!(result.is_ok());

    let factors = result.unwrap();
    assert!(factors.verify(&n));
}

#[test]
fn test_adaptive_quantum_region() {
    use rust_pattern_solver::types::quantum_enhanced::EnhancedQuantumRegion;

    let n = Number::from(10001u32);
    let center = Number::from(100u32);
    let patterns = create_test_patterns();

    let mut region = EnhancedQuantumRegion::from_pattern_analysis(center, &patterns, &n);

    // Test probability calculation
    let p1 = region.probability_at(&Number::from(100u32));
    let p2 = region.probability_at(&Number::from(110u32));
    assert!(p1 > 0.0);
    assert!(p2 > 0.0);

    // Test updating with observations
    region.update(&Number::from(105u32), true, None);
    region.update(&Number::from(120u32), false, None);

    // Test search candidates
    let candidates = region.get_search_candidates(5);
    assert!(!candidates.is_empty());
}

#[test]
fn test_distribution_adaptation() {
    use rust_pattern_solver::types::quantum_enhanced::{DistributionType, EnhancedQuantumRegion};

    let n = Number::from(100000u32);
    let center = Number::from(316u32); // ~sqrt(100000)

    let mut region = EnhancedQuantumRegion::new(center, Number::from(50u32), &n);
    assert_eq!(region.distribution_type, DistributionType::Gaussian);

    // Add multiple successful observations at different modes
    for i in 0..15 {
        let offset = if i < 5 {
            10
        } else if i < 10 {
            -20
        } else {
            30
        };
        region.update(&Number::from((316 + offset) as u32), true, None);
    }

    // After enough observations, it should detect multi-modality
    let candidates = region.get_search_candidates(10);
    assert!(candidates.len() > 3); // Should have multiple modes
}

#[test]
fn test_quantum_search_with_harmonic_pattern() {
    // Test with number having harmonic pattern
    let n = Number::from(2047u32); // 23 × 89 (large imbalance)

    let mut patterns = create_test_patterns();
    patterns.push(Pattern {
        id: "harmonic_test".to_string(),
        kind: PatternKind::Harmonic {
            base_frequency: 0.5,
            harmonics: vec![1.0, 0.5, 0.25],
        },
        frequency: 0.8,
        description: "Test harmonic pattern".to_string(),
        parameters: vec![],
        scale_range: ScaleRange {
            min_bits: 1,
            max_bits: 100,
            unbounded: false,
        },
    });

    let recognition = recognize(n.clone(), &patterns).unwrap();
    let constants = ConstantDiscovery::extract(&patterns);
    let formalization = formalize(recognition, &patterns, &constants).unwrap();

    // For harmonic patterns, the full execution pipeline should find factors
    // even if enhanced quantum search alone might not be optimal
    let result = rust_pattern_solver::pattern::execution::execute(formalization, &patterns);
    match result {
        Ok(factors) => {
            assert!(factors.verify(&n));
            assert_eq!(&factors.p * &factors.q, n);
            // Verify we found the right factors (23 and 89)
            assert!(
                (factors.p == Number::from(23u32) && factors.q == Number::from(89u32))
                || (factors.p == Number::from(89u32) && factors.q == Number::from(23u32))
            );
        }
        Err(e) => {
            panic!("Execution failed for harmonic pattern: {:?}", e);
        }
    }
}

#[test]
#[ignore] // This test is slow
fn test_large_number_quantum_search() {
    // Test with larger number
    let n = Number::from(1000003u32) * Number::from(1000033u32); // ~10^12
    let patterns = create_test_patterns();

    let recognition = recognize(n.clone(), &patterns).unwrap();
    let constants = ConstantDiscovery::extract(&patterns);
    let formalization = formalize(recognition, &patterns, &constants).unwrap();

    // Parallel quantum search should be used for large numbers
    let result = parallel_quantum_search(&formalization, &patterns);
    assert!(result.is_ok());

    let factors = result.unwrap();
    assert!(factors.verify(&n));
}

fn create_test_patterns() -> Vec<Pattern> {
    vec![
        Pattern {
            id: "emergent_1".to_string(),
            kind: PatternKind::Emergent,
            frequency: 0.7,
            description: "Basic emergent pattern".to_string(),
            parameters: vec![],
            scale_range: ScaleRange {
                min_bits: 1,
                max_bits: 100,
                unbounded: false,
            },
        },
        Pattern {
            id: "power_2".to_string(),
            kind: PatternKind::Power {
                base: 2,
                exponent: 10,
            },
            frequency: 0.3,
            description: "Powers of 2".to_string(),
            parameters: vec![],
            scale_range: ScaleRange {
                min_bits: 1,
                max_bits: 100,
                unbounded: false,
            },
        },
    ]
}
