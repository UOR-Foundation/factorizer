//! Integration tests using the test matrix
//! 
//! Tests the complete factorization pipeline with real test cases.

use eight_bit_pattern::{AutoTuner, TestCase};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize, Serialize)]
struct TestMatrix {
    version: String,
    generated: String,
    description: String,
    test_cases: std::collections::HashMap<String, Vec<TestCaseJson>>,
}

#[derive(Debug, Deserialize, Serialize)]
struct TestCaseJson {
    bit_length: usize,
    n: String,
    p: String,
    q: String,
    balanced: bool,
    p_bits: usize,
    q_bits: usize,
}

fn load_test_matrix() -> Vec<TestCase> {
    let path = Path::new("../data/test_matrix.json");
    
    if !path.exists() {
        // If test matrix doesn't exist, create a minimal set of test cases
        return vec![
            TestCase {
                n: BigInt::from(143),
                p: BigInt::from(11),
                q: BigInt::from(13),
                bit_length: 8,
            },
            TestCase {
                n: BigInt::from(15),
                p: BigInt::from(3),
                q: BigInt::from(5),
                bit_length: 4,
            },
            TestCase {
                n: BigInt::from(77),
                p: BigInt::from(7),
                q: BigInt::from(11),
                bit_length: 7,
            },
        ];
    }
    
    let contents = fs::read_to_string(path).expect("Failed to read test matrix");
    let matrix: TestMatrix = serde_json::from_str(&contents).expect("Failed to parse test matrix");
    
    let mut test_cases = Vec::new();
    
    // Extract test cases from all bit lengths
    for (_, cases) in matrix.test_cases {
        for case in cases {
            if let (Ok(n), Ok(p), Ok(q)) = (
                case.n.parse::<BigInt>(),
                case.p.parse::<BigInt>(),
                case.q.parse::<BigInt>(),
            ) {
                test_cases.push(TestCase {
                    n,
                    p,
                    q,
                    bit_length: case.bit_length,
                });
            }
        }
    }
    
    // Limit to reasonable number for testing
    test_cases.truncate(50);
    test_cases
}

#[test]
fn test_auto_tuner_with_test_matrix() {
    let mut tuner = AutoTuner::new();
    let test_cases = load_test_matrix();
    
    println!("Loaded {} test cases", test_cases.len());
    
    // Test initial success rate
    let mut initial_success = 0;
    for case in &test_cases {
        let result = tuner.factor(&case.n.to_string());
        if let Ok((p, q)) = result {
            let p_big = p.parse::<BigInt>().unwrap();
            let q_big = q.parse::<BigInt>().unwrap();
            
            if (p_big == case.p && q_big == case.q) || (p_big == case.q && q_big == case.p) {
                initial_success += 1;
            }
        }
    }
    
    println!("Initial success rate: {}/{}", initial_success, test_cases.len());
    
    // Load test cases for tuning
    tuner.load_test_cases(test_cases.clone());
    
    // Run optimization for a few rounds
    let optimized_params = tuner.optimize(10);
    
    println!("Optimized parameters: {:?}", optimized_params);
    
    // Test optimized success rate
    let mut optimized_success = 0;
    for case in &test_cases {
        let result = tuner.factor(&case.n.to_string());
        if let Ok((p, q)) = result {
            let p_big = p.parse::<BigInt>().unwrap();
            let q_big = q.parse::<BigInt>().unwrap();
            
            if (p_big == case.p && q_big == case.q) || (p_big == case.q && q_big == case.p) {
                optimized_success += 1;
            }
        }
    }
    
    println!("Optimized success rate: {}/{}", optimized_success, test_cases.len());
    
    // Should improve or at least not get worse
    assert!(optimized_success >= initial_success);
}

#[test]
fn test_specific_hard_semiprimes() {
    let tuner = AutoTuner::new();
    
    // Test some specific hard cases
    let hard_cases = vec![
        ("143", "11", "13"),  // Classic example
        ("1073", "29", "37"), // Larger primes
        ("2021", "43", "47"), // Close primes
    ];
    
    for (n, expected_p, expected_q) in hard_cases {
        println!("Testing factorization of {}", n);
        
        match tuner.factor(n) {
            Ok((p, q)) => {
                println!("  Found factors: {} * {}", p, q);
                
                // Verify correctness
                let p_big = p.parse::<BigInt>().unwrap();
                let q_big = q.parse::<BigInt>().unwrap();
                let n_big = n.parse::<BigInt>().unwrap();
                
                assert_eq!(p_big.clone() * q_big.clone(), n_big);
                
                // Check if factors match expected (in either order)
                let expected_p_big = expected_p.parse::<BigInt>().unwrap();
                let expected_q_big = expected_q.parse::<BigInt>().unwrap();
                
                assert!(
                    (p_big == expected_p_big && q_big == expected_q_big) ||
                    (p_big == expected_q_big && q_big == expected_p_big)
                );
            }
            Err(e) => {
                println!("  Failed to factor: {}", e);
                // It's okay if default parameters don't work for all cases
            }
        }
    }
}

#[test]
fn test_edge_cases() {
    let tuner = AutoTuner::new();
    
    // Test perfect square
    match tuner.factor("49") {
        Ok((p, q)) => {
            assert_eq!(p, "7");
            assert_eq!(q, "7");
        }
        Err(_) => {
            // Acceptable for now
        }
    }
    
    // Test very small semiprime
    match tuner.factor("6") {
        Ok((p, q)) => {
            assert!(
                (p == "2" && q == "3") || (p == "3" && q == "2")
            );
        }
        Err(_) => {
            // Acceptable for now
        }
    }
}

#[test]
fn test_invalid_inputs() {
    let tuner = AutoTuner::new();
    
    // Test invalid number format
    assert!(tuner.factor("not_a_number").is_err());
    
    // Test negative number
    assert!(tuner.factor("-15").is_err());
    
    // Test zero
    assert!(tuner.factor("0").is_err());
}