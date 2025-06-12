//! Demonstration of enhanced quantum search for factorization
//!
//! This example shows how the adaptive quantum search learns from observations
//! and adjusts its probability distributions to find factors more efficiently.

use rust_pattern_solver::observer::ConstantDiscovery;
use rust_pattern_solver::pattern::{
    execution_enhanced::enhanced_quantum_search, formalization::formalize, recognition::recognize,
};
use rust_pattern_solver::types::pattern::ScaleRange;
use rust_pattern_solver::types::{Number, Pattern, PatternKind};

fn main() {
    println!("Enhanced Quantum Search Demonstration");
    println!("====================================\n");

    // Test with various numbers
    let test_numbers = vec![
        ("143", "11 × 13 (balanced)"),
        ("323", "17 × 19 (balanced)"),
        ("2047", "23 × 89 (harmonic)"),
        ("10001", "73 × 137 (balanced)"),
        ("123456789", "3^2 × 3607 × 3803 (composite)"),
    ];

    // Create test patterns
    let patterns = create_patterns();
    let constants = ConstantDiscovery::extract(&patterns);

    for (n_str, description) in test_numbers {
        println!("Testing: {} = {}", n_str, description);

        let n = Number::from_str_radix(n_str, 10).unwrap();

        // Step 1: Recognition
        match recognize(n.clone(), &patterns) {
            Ok(recognition) => {
                println!("  Recognition: {:?}", recognition.pattern_type);

                // Step 2: Formalization
                match formalize(recognition, &patterns, &constants) {
                    Ok(formalization) => {
                        println!(
                            "  Formalization strategies: {} available",
                            formalization.strategies.len()
                        );

                        // Step 3: Enhanced quantum search
                        match enhanced_quantum_search(&formalization, &patterns) {
                            Ok(factors) => {
                                println!(
                                    "  ✓ Factors found: {} × {} = {}",
                                    factors.p, factors.q, n
                                );
                                println!("    Method: {}", factors.method);
                                println!("    Confidence: {:.2}%\n", factors.confidence * 100.0);
                            },
                            Err(e) => {
                                println!("  ✗ Failed: {}\n", e);
                            },
                        }
                    },
                    Err(e) => {
                        println!("  Formalization failed: {}\n", e);
                    },
                }
            },
            Err(e) => {
                println!("  Recognition failed: {}\n", e);
            },
        }
    }

    // Demonstrate adaptive learning
    demonstrate_adaptive_learning();
}

fn create_patterns() -> Vec<Pattern> {
    vec![
        Pattern {
            id: "balanced_semiprime".to_string(),
            kind: PatternKind::Emergent,
            frequency: 0.7,
            description: "Balanced semiprime pattern".to_string(),
            parameters: vec![],
            scale_range: ScaleRange {
                min_bits: 8,
                max_bits: 512,
                unbounded: true,
            },
        },
        Pattern {
            id: "harmonic_pattern".to_string(),
            kind: PatternKind::Harmonic {
                base_frequency: 0.618,
                harmonics: vec![1.0, 0.618, 0.382],
            },
            frequency: 0.4,
            description: "Harmonic imbalance pattern".to_string(),
            parameters: vec![],
            scale_range: ScaleRange {
                min_bits: 8,
                max_bits: 512,
                unbounded: true,
            },
        },
        Pattern {
            id: "power_pattern".to_string(),
            kind: PatternKind::Power {
                base: 2,
                exponent: 10,
            },
            frequency: 0.3,
            description: "Power of prime pattern".to_string(),
            parameters: vec![],
            scale_range: ScaleRange {
                min_bits: 10,
                max_bits: 1024,
                unbounded: false,
            },
        },
    ]
}

fn demonstrate_adaptive_learning() {
    use rust_pattern_solver::types::quantum_enhanced::EnhancedQuantumRegion;

    println!("\nAdaptive Learning Demonstration");
    println!("==============================\n");

    let n = Number::from(10007u32);
    let sqrt_n = rust_pattern_solver::utils::integer_sqrt(&n).unwrap();

    // Create quantum region
    let mut region = EnhancedQuantumRegion::new(sqrt_n.clone(), Number::from(50u32), &n);

    println!("Initial quantum region:");
    println!("  Center: {}", region.center);
    println!("  Radius: {}", region.radius.current);
    println!("  Distribution: {:?}", region.distribution_type);
    println!(
        "  Confidence: {:.2}%\n",
        region.confidence_metrics.overall * 100.0
    );

    // Simulate observations
    let observations = vec![
        (Number::from(100u32), false, "miss"),
        (Number::from(105u32), false, "miss"),
        (Number::from(95u32), false, "miss"),
        (Number::from(59u32), true, "HIT! (factor found)"),
    ];

    for (i, (location, success, label)) in observations.iter().enumerate() {
        println!(
            "Observation {}: {} at {} from center",
            i + 1,
            label,
            if location >= &region.center {
                location - &region.center
            } else {
                &region.center - location
            }
        );

        region.update(location, *success, None);

        println!(
            "  Updated confidence: {:.2}%",
            region.confidence_metrics.overall * 100.0
        );
        println!(
            "  Radius: {} ({})",
            region.radius.current,
            if region.radius.miss_count > 0 {
                format!("{} consecutive misses", region.radius.miss_count)
            } else {
                "reset".to_string()
            }
        );

        if *success {
            println!("  ✓ Factor found! Adaptive learning successful.\n");
        }
    }

    // Show final search candidates
    println!("Final search candidates based on learning:");
    let candidates = region.get_search_candidates(5);
    for (i, candidate) in candidates.iter().enumerate() {
        println!("  {}: {}", i + 1, candidate);
    }
}
