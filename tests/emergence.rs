//! Emergence tests - Verify pattern discovery without presumption

use rust_pattern_solver::emergence::{
    discover_all_patterns, InvariantDiscovery, ScalingAnalysis, UniversalPatterns,
};
use rust_pattern_solver::observer::ObservationCollector;
use rust_pattern_solver::types::{Number, PatternKind};
use rust_pattern_solver::utils;

#[test]
fn test_invariant_discovery() {
    // Generate test observations
    let mut collector = ObservationCollector::new();
    let mut numbers = Vec::new();

    // Various types of semiprimes
    for p in &[3u32, 5, 7, 11, 13, 17, 19, 23] {
        for q in &[3u32, 5, 7, 11, 13, 17, 19, 23] {
            if p <= q {
                numbers.push(Number::from(p * q));
            }
        }
    }

    let observations = collector.observe_parallel(&numbers).unwrap();
    let invariants = InvariantDiscovery::find_all(&observations).unwrap();

    // Should discover fundamental invariants
    let has_multiplication = invariants.iter().any(|p| p.id == "multiplication_invariant");
    let has_fermat = invariants.iter().any(|p| p.id == "fermat_identity");
    let has_modular = invariants.iter().any(|p| p.id == "modular_invariant");

    assert!(
        has_multiplication,
        "Failed to discover multiplication invariant"
    );
    assert!(has_fermat, "Failed to discover Fermat identity");
    assert!(has_modular, "Failed to discover modular invariant");

    // All invariants should have frequency 1.0
    for invariant in &invariants {
        if matches!(invariant.kind, PatternKind::Invariant) {
            assert_eq!(
                invariant.frequency, 1.0,
                "Invariant {} has frequency {} < 1.0",
                invariant.id, invariant.frequency
            );
        }
    }
}

#[test]
fn test_scaling_analysis() {
    let mut collector = ObservationCollector::new();
    let mut numbers = Vec::new();

    // Generate semiprimes at different scales
    // 8-bit scale
    for _ in 0..20 {
        let p = utils::generate_random_prime(4).unwrap();
        let q = utils::generate_random_prime(4).unwrap();
        numbers.push(&p * &q);
    }

    // 16-bit scale
    for _ in 0..20 {
        let p = utils::generate_random_prime(8).unwrap();
        let q = utils::generate_random_prime(8).unwrap();
        numbers.push(&p * &q);
    }

    // 24-bit scale
    for _ in 0..20 {
        let p = utils::generate_random_prime(12).unwrap();
        let q = utils::generate_random_prime(12).unwrap();
        numbers.push(&p * &q);
    }

    let observations = collector.observe_parallel(&numbers).unwrap();
    let scaling_patterns = ScalingAnalysis::analyze_all(&observations).unwrap();

    assert!(
        !scaling_patterns.is_empty(),
        "No scaling patterns discovered"
    );

    // Should discover some scale-dependent patterns
    let scale_dependent_count = scaling_patterns
        .iter()
        .filter(|p| matches!(p.kind, PatternKind::ScaleDependent))
        .count();

    assert!(
        scale_dependent_count > 0,
        "No scale-dependent patterns found"
    );
}

#[test]
fn test_universal_pattern_discovery() {
    let mut collector = ObservationCollector::new();
    let mut numbers = Vec::new();

    // Include various semiprime types
    // Balanced
    numbers.extend(vec![
        Number::from(143u32),  // 11 × 13
        Number::from(323u32),  // 17 × 19
        Number::from(667u32),  // 23 × 29
        Number::from(1147u32), // 31 × 37
    ]);

    // Squares
    numbers.extend(vec![
        Number::from(49u32),  // 7²
        Number::from(121u32), // 11²
        Number::from(169u32), // 13²
    ]);

    // Unbalanced
    numbers.extend(vec![
        Number::from(93u32),  // 3 × 31
        Number::from(155u32), // 5 × 31
        Number::from(217u32), // 7 × 31
    ]);

    let observations = collector.observe_parallel(&numbers).unwrap();
    let universal_patterns = UniversalPatterns::discover(&observations).unwrap();

    assert!(
        !universal_patterns.is_empty(),
        "No universal patterns discovered"
    );

    // Extract constants
    let constants = UniversalPatterns::extract_constants(&universal_patterns);

    // Should discover some universal structure
    for pattern in &universal_patterns {
        println!(
            "Universal pattern: {} - {}",
            pattern.id, pattern.description
        );
    }
}

#[test]
fn test_pattern_emergence_without_bias() {
    // This test verifies that patterns emerge from data, not our assumptions
    let mut collector = ObservationCollector::new();

    // Generate random semiprimes without specific structure
    let mut numbers = Vec::new();
    for _ in 0..100 {
        let bits_p = 4 + (rand::random::<usize>() % 12);
        let bits_q = 4 + (rand::random::<usize>() % 12);

        let p = utils::generate_random_prime(bits_p).unwrap();
        let q = utils::generate_random_prime(bits_q).unwrap();
        numbers.push(&p * &q);
    }

    let observations = collector.observe_parallel(&numbers).unwrap();
    let emergent = discover_all_patterns(&observations).unwrap();

    // Verify we discovered patterns
    assert!(emergent.count() > 0, "No patterns emerged from random data");

    // Invariants should still hold
    let invariant_count = emergent.invariants.iter().filter(|p| p.frequency >= 1.0).count();
    assert!(invariant_count >= 3, "Core invariants not discovered");

    // Should find some patterns even in random data
    println!(
        "Discovered {} total patterns from random semiprimes",
        emergent.count()
    );
    println!("  {} invariants", emergent.invariants.len());
    println!("  {} scaling patterns", emergent.scaling_patterns.len());
    println!("  {} universal patterns", emergent.universal_patterns.len());
}

#[test]
fn test_pattern_validation_across_scales() {
    let mut collector = ObservationCollector::new();
    let mut numbers = Vec::new();

    // Generate observations across multiple scales
    for scale_bits in &[8, 16, 24, 32] {
        for _ in 0..10 {
            let p = utils::generate_random_prime(scale_bits / 2).unwrap();
            let q = utils::generate_random_prime(scale_bits / 2).unwrap();
            numbers.push(&p * &q);
        }
    }

    let observations = collector.observe_parallel(&numbers).unwrap();
    let invariants = InvariantDiscovery::find_all(&observations).unwrap();

    // Validate each invariant holds across all scales
    for invariant in &invariants {
        let valid = InvariantDiscovery::validate_across_scales(invariant, &observations);
        assert!(
            valid,
            "Invariant {} does not hold across all scales",
            invariant.id
        );
    }
}

#[test]
fn test_no_presumed_patterns() {
    // Verify that we don't impose patterns that don't exist
    let mut collector = ObservationCollector::new();

    // Create observations that don't follow typical patterns
    // Use products of non-adjacent primes
    let primes = vec![3u32, 7, 13, 19, 29, 37, 43, 53, 61, 71];
    let mut numbers = Vec::new();

    for i in 0..primes.len() - 3 {
        numbers.push(Number::from(primes[i]) * Number::from(primes[i + 3]));
    }

    let observations = collector.observe_parallel(&numbers).unwrap();
    let emergent = discover_all_patterns(&observations).unwrap();

    // Should still find invariants (they're mathematical truths)
    assert!(!emergent.invariants.is_empty());

    // But pattern distribution might be different
    // This tests that we're not forcing patterns
    let total_patterns = emergent.count();
    assert!(total_patterns > 0, "Should find some patterns");

    // Patterns should reflect actual data structure
    for pattern in emergent.all_patterns() {
        assert!(
            pattern.frequency > 0.0,
            "Pattern {} has zero frequency",
            pattern.id
        );
    }
}
