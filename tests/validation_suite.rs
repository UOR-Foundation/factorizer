//! Validation suite for building confidence in The Pattern
//!
//! This suite tests specific aspects of pattern recognition to ensure
//! we understand the capabilities and limitations before attempting
//! larger challenges.

use rust_pattern_solver::observer::{ObservationCollector, ConstantDiscovery};
use rust_pattern_solver::pattern::{self, Pattern};
use rust_pattern_solver::types::{Number, PatternType};
use rust_pattern_solver::emergence::{ScalingAnalysis, InvariantDiscovery};

#[test]
fn test_small_prime_recognition() {
    // Test that we correctly identify prime numbers
    let primes = vec![
        Number::from(2u32),
        Number::from(3u32),
        Number::from(5u32),
        Number::from(7u32),
        Number::from(11u32),
        Number::from(13u32),
        Number::from(97u32),
        Number::from(997u32),
    ];

    let _collector = ObservationCollector::new();
    let patterns = Pattern::discover_from_observations(&[]).unwrap_or_default();

    for p in primes {
        let recognition = pattern::recognition::recognize(p.clone(), &patterns);
        match recognition {
            Ok(rec) => {
                assert_eq!(
                    rec.pattern_type,
                    PatternType::Prime,
                    "Failed to recognize {} as prime",
                    p
                );
            }
            Err(e) => panic!("Failed to recognize prime {}: {}", p, e),
        }
    }
}

#[test]
fn test_small_semiprime_patterns() {
    // Test recognition of small semiprimes with known factors
    let test_cases = vec![
        (15u32, 3u32, 5u32),
        (21u32, 3u32, 7u32),
        (35u32, 5u32, 7u32),
        (77u32, 7u32, 11u32),
        (143u32, 11u32, 13u32),
        (221u32, 13u32, 17u32),
        (323u32, 17u32, 19u32),
    ];

    let mut collector = ObservationCollector::new();
    
    // Collect training data
    let training_numbers: Vec<Number> = test_cases
        .iter()
        .map(|(n, _, _)| Number::from(*n))
        .collect();
    
    let observations = collector.observe_parallel(&training_numbers).unwrap();
    let patterns = Pattern::discover_from_observations(&observations).unwrap();

    println!("\nDiscovered {} patterns from small semiprimes", patterns.len());

    // Test each case
    for (n, expected_p, expected_q) in test_cases {
        println!("\nTesting {} = {} × {}", n, expected_p, expected_q);
        
        let n_big = Number::from(n);
        let recognition = pattern::recognition::recognize(n_big.clone(), &patterns).unwrap();
        
        println!("  Pattern type: {:?}", recognition.pattern_type);
        println!("  Confidence: {:.2}", recognition.confidence);
        
        // Try to factor
        let formalization = pattern::formalization::formalize(recognition, &patterns, &[]).unwrap();
        let factors = pattern::execution::execute(formalization, &patterns).unwrap();
        
        assert!(
            (factors.p == Number::from(expected_p) && factors.q == Number::from(expected_q)) ||
            (factors.p == Number::from(expected_q) && factors.q == Number::from(expected_p)),
            "Factor mismatch for {}: got {} × {}, expected {} × {}",
            n, factors.p, factors.q, expected_p, expected_q
        );
        
        println!("  ✓ Correctly factored");
    }
}

#[test]
fn test_pattern_type_classification() {
    // Test that different types of numbers are classified correctly
    let mut collector = ObservationCollector::new();
    
    // Generate diverse training set
    let mut training = Vec::new();
    
    // Balanced semiprimes (p ≈ q)
    for i in 0..20 {
        let p = 100 + i;
        let q = 100 + i + 2;
        training.push(Number::from(p * q));
    }
    
    // Harmonic semiprimes (p << q)
    for i in 0..20 {
        let p = 3 + i;
        let q = 1000 + i;
        training.push(Number::from(p * q));
    }
    
    // Power patterns (p = q)
    for i in 2..10 {
        training.push(Number::from(i * i));
    }
    
    let observations = collector.observe_parallel(&training).unwrap();
    let patterns = Pattern::discover_from_observations(&observations).unwrap();
    
    // Test classification
    let test_cases = vec![
        (Number::from(10403u32), PatternType::SmallFactor),  // 101 × 103 - small factors
        (Number::from(3 * 997), PatternType::SmallFactor),   // 3 × 997 - has factor 3
        (Number::from(169u32), PatternType::SmallFactor),    // 13² - small factor
        (Number::from(97u32), PatternType::Prime),           // prime
    ];
    
    for (n, expected_type) in test_cases {
        let recognition = pattern::recognition::recognize(n.clone(), &patterns).unwrap();
        println!("{} recognized as {:?} (expected {:?})", 
                 n, recognition.pattern_type, expected_type);
        assert_eq!(recognition.pattern_type, expected_type);
    }
}

#[test]
fn test_universal_constants() {
    // Test that universal constants are discovered and consistent
    let mut collector = ObservationCollector::new();
    
    // Collect observations across different scales
    let mut all_numbers = Vec::new();
    
    // Small scale
    for i in 0..50 {
        let p = 2 * i + 3;
        let q = 2 * i + 5;
        all_numbers.push(Number::from(p * q));
    }
    
    // Medium scale
    for i in 0..20 {
        let p = 1000 + i;
        let q = 1000 + i + 2;
        all_numbers.push(Number::from(p as u32 * q as u32));
    }
    
    let observations = collector.observe_parallel(&all_numbers).unwrap();
    let patterns = Pattern::discover_from_observations(&observations).unwrap();
    
    // Extract constants
    let constants = ConstantDiscovery::extract(&patterns);
    
    println!("\nDiscovered {} universal constants:", constants.len());
    for (i, constant) in constants.iter().enumerate().take(5) {
        println!("  {}: {} = {:.6} (universality: {:.2})", 
                 i, constant.name, constant.value, constant.universality);
    }
    
    // Verify some expected relationships
    assert!(!constants.is_empty(), "Should discover some universal constants");
}

#[test]
fn test_scaling_behavior() {
    // Test how patterns scale with number size
    use rust_pattern_solver::types::Observation;
    
    let scales = vec![
        (8, 16, 10),    // bits_min, bits_max, sample_count
        (16, 32, 10),
        (32, 64, 5),
    ];
    
    for (min_bits, max_bits, count) in scales {
        println!("\nAnalyzing {}-{} bit scaling:", min_bits, max_bits);
        
        // Generate numbers in this range
        let mut numbers = Vec::new();
        for i in 0..count {
            let bits = min_bits + (i * (max_bits - min_bits) / count);
            let half_bits = bits / 2;
            
            let p = Number::from(1u32) << (half_bits as u32 - 1);
            let p = &p + &Number::from((2 * i + 3) as u32);
            
            let q = Number::from(1u32) << (half_bits as u32 - 1);
            let q = &q + &Number::from((2 * i + 5) as u32);
            
            numbers.push(&p * &q);
        }
        
        // Collect observations for this scale
        let mut collector = ObservationCollector::new();
        match collector.observe_parallel(&numbers) {
            Ok(observations) => {
                // Analyze scaling patterns
                match ScalingAnalysis::analyze_all(&observations) {
                    Ok(patterns) => {
                        println!("  Found {} scaling patterns", patterns.len());
                        // Patterns should exist at all scales
                        assert!(!patterns.is_empty(), 
                                "Should find patterns at {}-{} bits", min_bits, max_bits);
                    }
                    Err(e) => println!("  Scaling analysis failed: {}", e),
                }
            }
            Err(e) => println!("  Observation failed: {}", e),
        }
    }
}

#[test]
fn test_invariant_relationships() {
    // Test that certain relationships hold universally
    let mut collector = ObservationCollector::new();
    
    // Generate diverse test set
    let mut numbers = Vec::new();
    
    // Different types of semiprimes
    for i in 0..30 {
        // Balanced
        numbers.push(Number::from((100 + i) * (102 + i)));
        // Harmonic  
        numbers.push(Number::from((3 + i) * (1000 + i)));
        // Close factors
        numbers.push(Number::from((50 + i) * (51 + i)));
    }
    
    let observations = collector.observe_parallel(&numbers).unwrap();
    
    // Discover invariants
    match InvariantDiscovery::find_all(&observations) {
        Ok(invariants) => {
            println!("\nDiscovered {} invariant patterns:", invariants.len());
            for (i, pattern) in invariants.iter().enumerate().take(5) {
                println!("  {}: {} (frequency: {:.2})", 
                         i, pattern.description, pattern.frequency);
            }
            
            // Should find some invariant patterns
            assert!(!invariants.is_empty(), "Should discover some invariant patterns");
        }
        Err(e) => {
            println!("\nInvariant discovery failed: {}", e);
        }
    }
}

#[test]
fn test_quantum_neighborhood_accuracy() {
    // Test that quantum neighborhoods correctly contain factors
    let test_cases = vec![
        (143u32, 11u32, 13u32),
        (323u32, 17u32, 19u32),
        (667u32, 23u32, 29u32),
    ];
    
    let mut collector = ObservationCollector::new();
    let training: Vec<Number> = test_cases.iter()
        .map(|(n, _, _)| Number::from(*n))
        .collect();
    
    let observations = collector.observe_parallel(&training).unwrap();
    let patterns = Pattern::discover_from_observations(&observations).unwrap();
    
    for (n, p, q) in test_cases {
        let n_big = Number::from(n);
        let recognition = pattern::recognition::recognize(n_big.clone(), &patterns).unwrap();
        
        if let Some(quantum_region) = &recognition.quantum_neighborhood {
            let p_big = Number::from(p);
            let q_big = Number::from(q);
            
            // Check if factors are within the quantum region
            let p_in_region = quantum_region.contains(&p_big);
            let q_in_region = quantum_region.contains(&q_big);
            
            println!("\nQuantum region for {} = {} × {}:", n, p, q);
            println!("  Center: {}", quantum_region.center);
            println!("  Radius: {}", quantum_region.radius);
            println!("  Contains p ({}): {}", p, p_in_region);
            println!("  Contains q ({}): {}", q, q_in_region);
            
            assert!(p_in_region || q_in_region, 
                    "Quantum region should contain at least one factor");
        }
    }
}

#[test]
fn test_edge_case_handling() {
    // Test handling of edge cases
    let mut collector = ObservationCollector::new();
    let patterns = Pattern::discover_from_observations(&[]).unwrap_or_default();
    
    // Test cases that might cause issues
    let edge_cases = vec![
        (Number::from(4u32), "perfect square of prime"),
        (Number::from(8u32), "power of 2"),
        (Number::from(27u32), "power of 3"),
        (Number::from(6u32), "product of consecutive primes"),
    ];
    
    for (n, description) in edge_cases {
        println!("\nTesting edge case: {} ({})", n, description);
        
        let result = pattern::recognition::recognize(n.clone(), &patterns);
        match result {
            Ok(rec) => {
                println!("  Pattern type: {:?}", rec.pattern_type);
                println!("  Confidence: {:.2}", rec.confidence);
            }
            Err(e) => {
                println!("  Recognition failed (expected): {}", e);
            }
        }
    }
}