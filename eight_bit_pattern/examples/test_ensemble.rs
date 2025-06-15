//! Test the ensemble voting system
//!
//! Run with: cargo run --example test_ensemble

use eight_bit_pattern::{
    EnsembleVoter, VotingStrategy, TunerParams, TestCase
};
use num_bigint::BigInt;
use std::time::Instant;

fn main() {
    println!("=== Testing Ensemble Voting System ===\n");
    
    // Test cases
    let test_cases = vec![
        TestCase { n: BigInt::from(15u32), p: BigInt::from(3), q: BigInt::from(5), bit_length: 4 },
        TestCase { n: BigInt::from(35u32), p: BigInt::from(5), q: BigInt::from(7), bit_length: 6 },
        TestCase { n: BigInt::from(143u32), p: BigInt::from(11), q: BigInt::from(13), bit_length: 8 },
        TestCase { n: BigInt::from(323u32), p: BigInt::from(17), q: BigInt::from(19), bit_length: 9 },
        TestCase { n: BigInt::from(437u32), p: BigInt::from(19), q: BigInt::from(23), bit_length: 9 },
        TestCase { n: BigInt::from(667u32), p: BigInt::from(23), q: BigInt::from(29), bit_length: 10 },
        TestCase { n: BigInt::from(899u32), p: BigInt::from(29), q: BigInt::from(31), bit_length: 10 },
        TestCase { n: BigInt::from(1073u32), p: BigInt::from(29), q: BigInt::from(37), bit_length: 11 },
    ];
    
    let params = TunerParams::default();
    
    // Test different voting strategies
    let strategies = vec![
        (VotingStrategy::Majority, "Majority Voting"),
        (VotingStrategy::WeightedBySuccess, "Weighted by Success"),
        (VotingStrategy::AdaptiveBySize, "Adaptive by Size"),
        (VotingStrategy::FirstSuccess, "First Success"),
    ];
    
    for (strategy, name) in strategies {
        println!("\n=== {} ===", name);
        
        let ensemble = EnsembleVoter::new(strategy);
        let stats = ensemble.get_statistics();
        
        println!("Ensemble contains {} constant sets", stats.num_constant_sets);
        println!("Average success rate: {:.1}%", stats.average_success_rate * 100.0);
        println!("Bit coverage: {}-{} bits\n", stats.bit_coverage.0, stats.bit_coverage.1);
        
        let mut successes = 0;
        let mut total_time = 0u128;
        
        for test_case in &test_cases {
            let start = Instant::now();
            let result = ensemble.recognize_factors(&test_case.n, &params);
            let elapsed = start.elapsed();
            total_time += elapsed.as_micros();
            
            let success = if let Some(factors) = result {
                let correct = (factors.p == test_case.p && factors.q == test_case.q) ||
                             (factors.p == test_case.q && factors.q == test_case.p);
                
                if correct {
                    successes += 1;
                    print!("✓");
                } else {
                    print!("✗");
                }
                correct
            } else {
                print!("✗");
                false
            };
            
            if !success && test_case.bit_length <= 10 {
                println!(" {} = {} × {} - FAILED", test_case.n, test_case.p, test_case.q);
            }
        }
        
        println!("\n\nSuccess rate: {}/{} ({:.1}%)", 
            successes, test_cases.len(), 
            successes as f64 / test_cases.len() as f64 * 100.0);
        println!("Average time: {:.1} μs", total_time as f64 / test_cases.len() as f64);
    }
    
    // Test custom constant set addition
    println!("\n\n=== Testing Custom Constant Set ===");
    
    let mut ensemble = EnsembleVoter::new(VotingStrategy::FirstSuccess);
    
    // Add a custom constant set optimized for small primes
    ensemble.add_constant_set(eight_bit_pattern::EnsembleConstantSet {
        values: [1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0],
        success_rate: 0.7,
        optimal_bit_range: (4, 12),
        name: "Small Primes".to_string(),
    });
    
    let stats = ensemble.get_statistics();
    println!("After adding custom set: {} constant sets", stats.num_constant_sets);
    
    // Test on a specific number
    let n = BigInt::from(35u32);
    println!("\nTesting N = 35 with augmented ensemble:");
    
    if let Some(factors) = ensemble.recognize_factors(&n, &params) {
        println!("Found factors: {} × {}", factors.p, factors.q);
        if factors.verify(&n) {
            println!("✓ Verification passed!");
        }
    } else {
        println!("✗ No factors found");
    }
}