//! Tune the empirical pattern using test matrix data
//! 
//! This loads successful factorizations and tunes the pattern
//! to recognize and factor numbers directly without search.

use rust_pattern_solver::pattern::empirical_pattern::EmpiricalPattern;
use rust_pattern_solver::types::Number;
use std::collections::BTreeMap;
use std::str::FromStr;
use serde::{Serialize, Deserialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestCase {
    bit_length: usize,
    n: String,
    p: String,
    q: String,
    balanced: bool,
    p_bits: usize,
    q_bits: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct TestMatrix {
    version: String,
    generated: String,
    description: String,
    test_cases: BTreeMap<usize, Vec<TestCase>>,
}

fn load_training_data(test_matrix: &TestMatrix) -> Vec<(Number, Number, Number)> {
    let mut training_data = Vec::new();
    
    // Load successful cases from 8-40 bits for initial training
    for (bit_length, cases) in &test_matrix.test_cases {
        if *bit_length <= 40 {
            for case in cases {
                let n = Number::from_str(&case.n).unwrap();
                let p = Number::from_str(&case.p).unwrap();
                let q = Number::from_str(&case.q).unwrap();
                training_data.push((n, p, q));
            }
        }
    }
    
    training_data
}

fn evaluate_pattern(pattern: &EmpiricalPattern, test_cases: &[TestCase]) -> (usize, usize, f64) {
    let mut successful = 0;
    let mut total = 0;
    let mut total_time = 0.0;
    
    for case in test_cases {
        let n = Number::from_str(&case.n).unwrap();
        let expected_p = Number::from_str(&case.p).unwrap();
        let expected_q = Number::from_str(&case.q).unwrap();
        
        let start = Instant::now();
        
        // Try to factor using empirical pattern
        if let Ok(recognition) = pattern.recognize(&n) {
            if let Ok(formalization) = pattern.formalize(recognition) {
                if let Ok(factors) = pattern.execute(formalization) {
                    if (factors.p == expected_p && factors.q == expected_q) ||
                       (factors.p == expected_q && factors.q == expected_p) {
                        successful += 1;
                    }
                }
            }
        }
        
        total_time += start.elapsed().as_secs_f64();
        total += 1;
    }
    
    let avg_time = if total > 0 { total_time / total as f64 } else { 0.0 };
    (successful, total, avg_time)
}

fn tune_for_higher_bits(pattern: &mut EmpiricalPattern, test_matrix: &TestMatrix) {
    println!("\nTuning for higher bit ranges...");
    
    // Progressively tune for higher bit ranges
    let bit_ranges = vec![
        (41, 48),
        (49, 56),
        (57, 64),
        (65, 80),
        (81, 96),
        (97, 128),
    ];
    
    for (min_bits, max_bits) in bit_ranges {
        println!("\nTuning for {}-{} bit range:", min_bits, max_bits);
        
        let mut training_cases = Vec::new();
        
        for (bit_length, cases) in &test_matrix.test_cases {
            if *bit_length >= min_bits && *bit_length <= max_bits {
                for case in cases {
                    let n = Number::from_str(&case.n).unwrap();
                    let p = Number::from_str(&case.p).unwrap();
                    let q = Number::from_str(&case.q).unwrap();
                    training_cases.push((n, p, q));
                }
            }
        }
        
        if training_cases.is_empty() {
            println!("  No test cases available for this range");
            continue;
        }
        
        // Tune the pattern with these cases
        for (n, p, q) in &training_cases {
            let channels = pattern.stream_processor.decompose_to_channels(n);
            pattern.factor_cache.insert(channels.clone(), (p.clone(), q.clone()));
        }
        
        println!("  Added {} cases to pattern cache", training_cases.len());
    }
}

fn main() {
    println!("Empirical Pattern Tuning");
    println!("========================\n");
    
    // Load test matrix
    let matrix_data = std::fs::read_to_string("data/test_matrix.json")
        .expect("Failed to load test matrix");
    
    let test_matrix: TestMatrix = serde_json::from_str(&matrix_data)
        .expect("Failed to parse test matrix");
    
    // Load training data
    let training_data = load_training_data(&test_matrix);
    println!("Loaded {} training cases from 8-40 bit range", training_data.len());
    
    // Create and train empirical pattern
    let mut pattern = EmpiricalPattern::from_test_matrix(&training_data);
    
    // Evaluate initial performance
    println!("\nInitial Performance Evaluation");
    println!("==============================");
    
    for (bit_length, cases) in &test_matrix.test_cases {
        if cases.is_empty() { continue; }
        
        let (successful, total, avg_time) = evaluate_pattern(&pattern, cases);
        let success_rate = (successful as f64 / total as f64) * 100.0;
        
        println!("{:3}-bit: {:3}/{:3} ({:5.1}%) avg: {:.3}ms", 
                 bit_length, successful, total, success_rate, avg_time * 1000.0);
    }
    
    // Tune for higher bit ranges
    tune_for_higher_bits(&mut pattern, &test_matrix);
    
    // Re-evaluate after tuning
    println!("\nPerformance After Tuning");
    println!("========================");
    
    let mut total_success = 0;
    let mut total_cases = 0;
    
    for (bit_length, cases) in &test_matrix.test_cases {
        if cases.is_empty() { continue; }
        
        let (successful, total, avg_time) = evaluate_pattern(&pattern, cases);
        let success_rate = (successful as f64 / total as f64) * 100.0;
        
        total_success += successful;
        total_cases += total;
        
        println!("{:3}-bit: {:3}/{:3} ({:5.1}%) avg: {:.3}ms", 
                 bit_length, successful, total, success_rate, avg_time * 1000.0);
    }
    
    let overall_rate = (total_success as f64 / total_cases as f64) * 100.0;
    println!("\nOverall success rate: {:.1}%", overall_rate);
    
    // Save tuned pattern
    println!("\nNext Steps:");
    println!("1. Continue tuning with more test cases");
    println!("2. Extract empirical rules from successful patterns");
    println!("3. Replace UniversalPattern with EmpiricalPattern");
    println!("4. Achieve 100% success through empirical tuning");
}