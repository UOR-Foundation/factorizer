//! Correctness tests - Verify The Pattern produces correct results

use rust_pattern_solver::observer::ObservationCollector;
use rust_pattern_solver::pattern::{execution, formalization, recognition};
use rust_pattern_solver::types::{Number, Pattern};

#[test]
fn test_small_semiprimes() {
    let test_cases = vec![
        (15u32, 3u32, 5u32),
        (21u32, 3u32, 7u32),
        (35u32, 5u32, 7u32),
        (77u32, 7u32, 11u32),
        (91u32, 7u32, 13u32),
        (143u32, 11u32, 13u32),
        (221u32, 13u32, 17u32),
        (323u32, 17u32, 19u32),
    ];

    // Collect observations for pattern discovery
    let mut collector = ObservationCollector::new();
    let numbers: Vec<Number> = test_cases.iter().map(|(n, _, _)| Number::from(*n)).collect();

    let observations = collector.observe_parallel(&numbers).unwrap();
    let patterns = Pattern::discover_from_observations(&observations).unwrap();

    // Test each case
    for (n, expected_p, expected_q) in test_cases {
        let n_big = Number::from(n);

        // Full pattern execution
        let recognition = recognition::recognize(n_big.clone(), &patterns).unwrap();
        let formalized = formalization::formalize(recognition, &patterns, &[]).unwrap();
        let factors = execution::execute(formalized, &patterns).unwrap();

        // Verify correctness
        assert!(factors.verify(&n_big), "Factors don't multiply to n");

        // Check we found the right factors (order doesn't matter)
        let (p, q) = if factors.p <= factors.q {
            (factors.p, factors.q)
        } else {
            (factors.q.clone(), factors.p.clone())
        };

        assert_eq!(
            p,
            Number::from(expected_p),
            "Wrong smaller factor for {}",
            n
        );
        assert_eq!(q, Number::from(expected_q), "Wrong larger factor for {}", n);
    }
}

#[test]
fn test_perfect_squares() {
    let squares = vec![
        (4u32, 2u32),
        (9u32, 3u32),
        (25u32, 5u32),
        (49u32, 7u32),
        (121u32, 11u32),
        (169u32, 13u32),
    ];

    // Include squares in training data
    let mut collector = ObservationCollector::new();
    let numbers: Vec<Number> = squares.iter().map(|(n, _)| Number::from(*n)).collect();

    let observations = collector.observe_parallel(&numbers).unwrap();
    let patterns = Pattern::discover_from_observations(&observations).unwrap();

    for (n, root) in squares {
        let n_big = Number::from(n);

        let recognition = recognition::recognize(n_big.clone(), &patterns).unwrap();
        let formalized = formalization::formalize(recognition, &patterns, &[]).unwrap();
        let factors = execution::execute(formalized, &patterns).unwrap();

        assert!(factors.verify(&n_big));
        assert_eq!(factors.p, Number::from(root));
        assert_eq!(factors.q, Number::from(root));
    }
}

#[test]
fn test_prime_recognition() {
    let primes = vec![2u32, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];

    // Need some semiprimes for pattern discovery
    let mut collector = ObservationCollector::new();
    let semiprimes = vec![
        Number::from(15u32),
        Number::from(21u32),
        Number::from(35u32),
    ];

    let observations = collector.observe_parallel(&semiprimes).unwrap();
    let patterns = Pattern::discover_from_observations(&observations).unwrap();

    for p in primes {
        let n = Number::from(p);
        let recognition = recognition::recognize(n.clone(), &patterns).unwrap();

        assert_eq!(
            recognition.pattern_type,
            rust_pattern_solver::types::PatternType::Prime,
            "Failed to recognize {} as prime",
            p
        );
    }
}

#[test]
fn test_small_factor_cases() {
    let cases = vec![
        (6u32, 2u32, 3u32),
        (10u32, 2u32, 5u32),
        (14u32, 2u32, 7u32),
        (22u32, 2u32, 11u32),
        (26u32, 2u32, 13u32),
        (33u32, 3u32, 11u32),
        (39u32, 3u32, 13u32),
    ];

    let mut collector = ObservationCollector::new();
    let numbers: Vec<Number> = cases.iter().map(|(n, _, _)| Number::from(*n)).collect();

    let observations = collector.observe_parallel(&numbers).unwrap();
    let patterns = Pattern::discover_from_observations(&observations).unwrap();

    for (n, expected_p, expected_q) in cases {
        let n_big = Number::from(n);

        let recognition = recognition::recognize(n_big.clone(), &patterns).unwrap();
        let formalized = formalization::formalize(recognition, &patterns, &[]).unwrap();
        let factors = execution::execute(formalized, &patterns).unwrap();

        assert!(factors.verify(&n_big));

        // Verify we found the expected factors
        let found_p = factors.p.min(factors.q.clone());
        let found_q = factors.p.max(factors.q);

        assert_eq!(found_p, Number::from(expected_p));
        assert_eq!(found_q, Number::from(expected_q));
    }
}

#[test]
fn test_larger_semiprimes() {
    // Test with some larger numbers
    let cases = vec![
        (10403u64, 101u64, 103u64), // Balanced twins
        (10609u64, 103u64, 103u64), // Perfect square
        (11021u64, 103u64, 107u64), // Close primes
        (11663u64, 107u64, 109u64), // Consecutive primes
    ];

    let mut collector = ObservationCollector::new();
    let numbers: Vec<Number> = cases.iter().map(|(n, _, _)| Number::from(*n)).collect();

    let observations = collector.observe_parallel(&numbers).unwrap();
    let patterns = Pattern::discover_from_observations(&observations).unwrap();

    for (n, expected_p, expected_q) in cases {
        let n_big = Number::from(n);

        let recognition = recognition::recognize(n_big.clone(), &patterns).unwrap();
        let formalized = formalization::formalize(recognition, &patterns, &[]).unwrap();

        match execution::execute(formalized, &patterns) {
            Ok(factors) => {
                assert!(factors.verify(&n_big));

                let found_p = factors.p.min(factors.q.clone());
                let found_q = factors.p.max(factors.q);

                assert_eq!(found_p, Number::from(expected_p));
                assert_eq!(found_q, Number::from(expected_q));
            },
            Err(e) => {
                panic!("Failed to factor {}: {}", n, e);
            },
        }
    }
}

#[test]
fn test_edge_cases() {
    let mut collector = ObservationCollector::new();

    // Test 4 = 2×2 (smallest semiprime)
    let four = Number::from(4u32);
    let obs = collector.observe_single(four.clone()).unwrap();
    assert_eq!(obs.p, Number::from(2u32));
    assert_eq!(obs.q, Number::from(2u32));

    // Test that 1 is handled correctly
    let one = Number::from(1u32);
    assert!(collector.observe_single(one).is_err());

    // Test that 0 is handled correctly
    let zero = Number::from(0u32);
    assert!(collector.observe_single(zero).is_err());
}
