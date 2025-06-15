//! Test advanced resonance-based factor extraction
//!
//! Run with: cargo run --example test_advanced

use eight_bit_pattern::{
    TunerParams, compute_basis, recognize_factors_advanced,
    TestCase
};
use num_bigint::BigInt;
use std::time::Instant;

fn main() {
    println!("=== Testing Advanced Resonance Extraction ===\n");
    
    // Test cases of increasing difficulty
    let test_cases = vec![
        TestCase { n: BigInt::from(143u32), p: BigInt::from(11), q: BigInt::from(13), bit_length: 8 },
        TestCase { n: BigInt::from(9409u32), p: BigInt::from(97), q: BigInt::from(97), bit_length: 14 },
        TestCase { n: BigInt::from(10403u32), p: BigInt::from(101), q: BigInt::from(103), bit_length: 14 },
        TestCase { n: BigInt::from(169649u32), p: BigInt::from(379), q: BigInt::from(447), bit_length: 18 },
        TestCase { n: BigInt::from(1649821u64), p: BigInt::from(1123), q: BigInt::from(1469), bit_length: 21 },
    ];
    
    // Create basis and parameters
    let params = TunerParams::default();
    let basis = compute_basis(128, &params);
    
    println!("Testing {} semiprimes with advanced resonance extraction:\n", test_cases.len());
    
    let mut successes = 0;
    let mut total_time = 0u128;
    
    for test_case in &test_cases {
        print!("N = {} ({} bits): ", test_case.n, test_case.bit_length);
        
        let start = Instant::now();
        let result = recognize_factors_advanced(&test_case.n, &basis, &params);
        let elapsed = start.elapsed();
        
        total_time += elapsed.as_micros();
        
        match result {
            Some(factors) => {
                let correct = (factors.p == test_case.p && factors.q == test_case.q) ||
                             (factors.p == test_case.q && factors.q == test_case.p);
                
                if correct {
                    println!("✓ SUCCESS - {} × {} in {:?}", factors.p, factors.q, elapsed);
                    successes += 1;
                } else {
                    println!("✗ INCORRECT - Found {} × {} (expected {} × {})", 
                        factors.p, factors.q, test_case.p, test_case.q);
                }
            }
            None => {
                println!("✗ FAILED - No factors found in {:?}", elapsed);
            }
        }
    }
    
    println!("\n=== Summary ===");
    println!("Success rate: {}/{} ({:.1}%)", 
        successes, test_cases.len(), 
        (successes as f64 / test_cases.len() as f64) * 100.0
    );
    println!("Average time: {:.1} μs", total_time as f64 / test_cases.len() as f64);
    
    // Test comparison with standard method
    println!("\n=== Comparison with Standard Method ===");
    
    for test_case in test_cases.iter().take(3) {
        println!("\nN = {}:", test_case.n);
        
        // Standard method
        let start = Instant::now();
        let standard_result = eight_bit_pattern::recognize_factors(&test_case.n, &basis, &params);
        let standard_time = start.elapsed();
        
        // Advanced method
        let start = Instant::now();
        let advanced_result = recognize_factors_advanced(&test_case.n, &basis, &params);
        let advanced_time = start.elapsed();
        
        println!("  Standard: {} in {:?}", 
            if standard_result.is_some() { "Success" } else { "Failed" },
            standard_time
        );
        println!("  Advanced: {} in {:?}",
            if advanced_result.is_some() { "Success" } else { "Failed" },
            advanced_time
        );
        
        if advanced_time < standard_time {
            println!("  Advanced is {:.1}x faster", 
                standard_time.as_nanos() as f64 / advanced_time.as_nanos() as f64);
        }
    }
}