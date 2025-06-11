//! Integration tests - Full pipeline from observation to execution

use rust_pattern_solver::{
    observer::ObservationCollector,
    pattern::{recognition, formalization, execution},
    emergence::discover_all_patterns,
    types::{Number, Pattern},
};
use std::time::Instant;

#[test]
fn test_full_pipeline_small_numbers() {
    // Full pipeline test: Observe → Discover → Recognize → Formalize → Execute
    
    // Step 1: Observe
    let mut collector = ObservationCollector::new();
    let training_numbers: Vec<Number> = vec![
        15, 21, 35, 77, 91, 143, 221, 323, 437, 667, 899
    ].into_iter().map(Number::from).collect();
    
    let observations = collector.observe_parallel(&training_numbers).unwrap();
    assert_eq!(observations.len(), training_numbers.len());
    
    // Step 2: Discover patterns
    let emergent = discover_all_patterns(&observations).unwrap();
    assert!(emergent.count() > 0);
    
    let patterns = Pattern::discover_from_observations(&observations).unwrap();
    assert!(!patterns.is_empty());
    
    // Step 3-5: Test recognition, formalization, and execution on new numbers
    let test_cases = vec![
        (391u32, 17u32, 23u32),
        (493u32, 17u32, 29u32),
        (551u32, 19u32, 29u32),
        (713u32, 23u32, 31u32),
    ];
    
    for (n, expected_p, expected_q) in test_cases {
        let n_big = Number::from(n);
        
        // Recognition
        let recognition = recognition::recognize(n_big.clone(), &patterns).unwrap();
        assert!(recognition.confidence > 0.5);
        
        // Formalization
        let constants = rust_pattern_solver::emergence::UniversalPatterns::extract_constants(
            &emergent.universal_patterns
        );
        let formalized = formalization::formalize(recognition, &patterns, &constants).unwrap();
        
        // Execution
        let factors = execution::execute(formalized, &patterns).unwrap();
        assert!(factors.verify(&n_big));
        
        // Verify correct factors
        let (p, q) = if factors.p <= factors.q {
            (factors.p, factors.q)
        } else {
            (factors.q.clone(), factors.p.clone())
        };
        
        assert_eq!(p, Number::from(expected_p));
        assert_eq!(q, Number::from(expected_q));
    }
}

#[test]
fn test_performance_scaling() {
    // Test how performance scales with number size
    let mut collector = ObservationCollector::new();
    
    // Generate training data at different scales
    let mut training_numbers = Vec::new();
    
    // 8-bit scale
    for _ in 0..20 {
        let p = rust_pattern_solver::utils::generate_random_prime(4).unwrap();
        let q = rust_pattern_solver::utils::generate_random_prime(4).unwrap();
        training_numbers.push(&p * &q);
    }
    
    // 16-bit scale
    for _ in 0..20 {
        let p = rust_pattern_solver::utils::generate_random_prime(8).unwrap();
        let q = rust_pattern_solver::utils::generate_random_prime(8).unwrap();
        training_numbers.push(&p * &q);
    }
    
    let observations = collector.observe_parallel(&training_numbers).unwrap();
    let patterns = Pattern::discover_from_observations(&observations).unwrap();
    
    // Test recognition speed at different scales
    let test_scales = vec![8, 16, 24];
    
    for bits in test_scales {
        let mut times = Vec::new();
        
        for _ in 0..10 {
            let p = rust_pattern_solver::utils::generate_random_prime(bits / 2).unwrap();
            let q = rust_pattern_solver::utils::generate_random_prime(bits / 2).unwrap();
            let n = &p * &q;
            
            let start = Instant::now();
            let _ = recognition::recognize(n, &patterns);
            times.push(start.elapsed());
        }
        
        let avg_time = times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / times.len() as f64;
        println!("{}-bit recognition: {:.3}ms average", bits, avg_time * 1000.0);
        
        // Recognition should be fast
        assert!(avg_time < 0.1, "{}-bit recognition too slow", bits);
    }
}

#[test]
fn test_mixed_number_types() {
    // Test with a mix of different semiprime types
    let mut collector = ObservationCollector::new();
    
    let training_numbers = vec![
        // Balanced
        Number::from(143u32),   // 11 × 13
        Number::from(323u32),   // 17 × 19
        // Squares
        Number::from(49u32),    // 7²
        Number::from(121u32),   // 11²
        // Small factor
        Number::from(14u32),    // 2 × 7
        Number::from(22u32),    // 2 × 11
        // Unbalanced
        Number::from(93u32),    // 3 × 31
        Number::from(217u32),   // 7 × 31
    ];
    
    let observations = collector.observe_parallel(&training_numbers).unwrap();
    let emergent = discover_all_patterns(&observations).unwrap();
    let patterns = Pattern::discover_from_observations(&observations).unwrap();
    let constants = rust_pattern_solver::emergence::UniversalPatterns::extract_constants(
        &emergent.universal_patterns
    );
    
    // Test each type
    let test_cases = vec![
        (Number::from(209u32), "balanced"),      // 11 × 19
        (Number::from(169u32), "square"),        // 13²
        (Number::from(34u32), "small_factor"),   // 2 × 17
        (Number::from(155u32), "unbalanced"),    // 5 × 31
    ];
    
    for (n, expected_type) in test_cases {
        let recognition = recognition::recognize(n.clone(), &patterns).unwrap();
        let formalized = formalization::formalize(recognition, &patterns, &constants).unwrap();
        
        match execution::execute(formalized, &patterns) {
            Ok(factors) => {
                assert!(factors.verify(&n));
                println!("{} ({}) factored successfully using {}", 
                    n, expected_type, factors.method);
            }
            Err(e) => {
                panic!("Failed to factor {} ({}): {}", n, expected_type, e);
            }
        }
    }
}

#[test]
fn test_pattern_consistency() {
    // Verify patterns remain consistent across multiple runs
    let mut collector = ObservationCollector::new();
    
    let numbers: Vec<Number> = (0..50)
        .map(|i| {
            let p = 2 * i + 3;
            let q = 2 * i + 5;
            Number::from(p * q)
        })
        .collect();
    
    // Run pattern discovery multiple times
    let observations = collector.observe_parallel(&numbers).unwrap();
    
    let patterns1 = Pattern::discover_from_observations(&observations).unwrap();
    let patterns2 = Pattern::discover_from_observations(&observations).unwrap();
    
    // Should discover same number of patterns
    assert_eq!(patterns1.len(), patterns2.len());
    
    // Pattern discovery should be deterministic
    let emergent1 = discover_all_patterns(&observations).unwrap();
    let emergent2 = discover_all_patterns(&observations).unwrap();
    
    assert_eq!(emergent1.invariants.len(), emergent2.invariants.len());
}

#[test]
fn test_error_handling() {
    let patterns = Vec::new(); // Empty patterns
    
    // Should handle recognition with no patterns gracefully
    let n = Number::from(143u32);
    let result = recognition::recognize(n, &patterns);
    assert!(result.is_ok()); // Should still work, just with less confidence
    
    // Should handle invalid inputs
    let mut collector = ObservationCollector::new();
    
    // Can't observe 0
    assert!(collector.observe_single(Number::from(0u32)).is_err());
    
    // Can't observe 1
    assert!(collector.observe_single(Number::from(1u32)).is_err());
    
    // Can't observe negative (if we could construct them)
    // Primes should be recognized but not factored
    let prime = Number::from(97u32);
    let obs_result = collector.observe_single(prime);
    assert!(obs_result.is_err()); // Can't observe a prime as semiprime
}

#[test]
fn test_batch_processing() {
    // Test batch processing efficiency
    let mut collector = ObservationCollector::new();
    
    // Generate 1000 test semiprimes
    let mut numbers = Vec::new();
    for _ in 0..1000 {
        let p = rust_pattern_solver::utils::generate_random_prime(8).unwrap();
        let q = rust_pattern_solver::utils::generate_random_prime(8).unwrap();
        numbers.push(&p * &q);
    }
    
    let start = Instant::now();
    let observations = collector.observe_parallel(&numbers).unwrap();
    let observe_time = start.elapsed();
    
    assert_eq!(observations.len(), numbers.len());
    println!("Observed {} numbers in {:.2}s ({:.0} per second)",
        numbers.len(),
        observe_time.as_secs_f64(),
        numbers.len() as f64 / observe_time.as_secs_f64()
    );
    
    // Should be reasonably fast
    assert!(observe_time.as_secs() < 10, "Batch observation too slow");
}