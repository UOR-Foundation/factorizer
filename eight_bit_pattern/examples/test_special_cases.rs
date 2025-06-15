//! Test special case detection
//!
//! Run with: cargo run --example test_special_cases

use eight_bit_pattern::{
    detect_special_cases, try_special_cases, SpecialCase, 
    TunerParams, compute_basis, recognize_factors,
    decompose, detect_aligned_channels, extract_factors
};
use num_bigint::BigInt;
use std::time::Instant;

fn main() {
    println!("=== Testing Special Case Detection ===\n");
    
    // Test cases with known special properties
    let test_cases = vec![
        // Perfect squares
        (BigInt::from(9409u32), "97²", SpecialCase::PerfectSquare),
        (BigInt::from(10201u32), "101²", SpecialCase::PerfectSquare),
        (BigInt::from(12769u32), "113²", SpecialCase::PerfectSquare),
        
        // Twin primes
        (BigInt::from(35u32), "5 × 7", SpecialCase::TwinPrimes),
        (BigInt::from(143u32), "11 × 13", SpecialCase::TwinPrimes),
        (BigInt::from(323u32), "17 × 19", SpecialCase::TwinPrimes),
        (BigInt::from(899u32), "29 × 31", SpecialCase::TwinPrimes),
        
        // Sophie Germain primes (p, 2p+1)
        (BigInt::from(253u32), "11 × 23", SpecialCase::SophieGermain),
        (BigInt::from(1081u32), "23 × 47", SpecialCase::SophieGermain),
        
        // Cousin primes (differ by 4)
        (BigInt::from(21u32), "3 × 7", SpecialCase::CousinPrimes),
        (BigInt::from(77u32), "7 × 11", SpecialCase::CousinPrimes),
        (BigInt::from(221u32), "13 × 17", SpecialCase::CousinPrimes),
        
        // Sexy primes (differ by 6)
        (BigInt::from(91u32), "7 × 13", SpecialCase::SexyPrimes),
        (BigInt::from(187u32), "11 × 17", SpecialCase::SexyPrimes),
        (BigInt::from(403u32), "13 × 31", SpecialCase::SexyPrimes),
    ];
    
    println!("Direct Special Case Detection:\n");
    
    let mut detected_count = 0;
    for (n, description, expected_type) in &test_cases {
        let special_cases = detect_special_cases(n);
        
        print!("N = {:6} ({}): ", n, description);
        
        if special_cases.is_empty() {
            println!("No special case detected");
        } else {
            detected_count += 1;
            for result in &special_cases {
                let correct = result.case_type == *expected_type;
                let symbol = if correct { "✓" } else { "✗" };
                
                println!("{} {:?} with confidence {:.2}", 
                    symbol, result.case_type, result.confidence);
                
                if result.confidence >= 0.99 {
                    println!("  Factors: {} × {}", result.factors.p, result.factors.q);
                }
            }
        }
    }
    
    println!("\nDetection rate: {}/{} ({:.1}%)\n", 
        detected_count, test_cases.len(),
        detected_count as f64 / test_cases.len() as f64 * 100.0);
    
    // Test integrated pattern recognition with special cases
    println!("\n=== Testing Integrated Pattern Recognition ===\n");
    
    let params = TunerParams::default();
    
    let mut standard_time = 0u128;
    let mut special_time = 0u128;
    let mut special_success = 0;
    
    for (n, description, _) in test_cases.iter().take(10) {
        print!("N = {} ({}): ", n, description);
        
        // Time with special case detection (integrated)
        let start = Instant::now();
        let result_with_special = recognize_factors(n, &params);
        special_time += start.elapsed().as_micros();
        
        // Time without special case detection
        let start = Instant::now();
        let basis = compute_basis(n, &params);
        let channels = decompose(n);
        let peaks = detect_aligned_channels(n, &basis, &params);
        let result_without_special = extract_factors(n, &peaks, &channels, &params);
        standard_time += start.elapsed().as_micros();
        
        match (result_with_special, result_without_special) {
            (Some(f1), Some(f2)) => {
                if f1.verify(n) && f2.verify(n) {
                    println!("Both methods succeeded");
                    special_success += 1;
                }
            }
            (Some(f1), None) => {
                if f1.verify(n) {
                    println!("✓ Special case succeeded where standard failed!");
                    special_success += 1;
                }
            }
            (None, Some(_)) => {
                println!("Standard succeeded where special failed");
            }
            (None, None) => {
                println!("Both methods failed");
            }
        }
    }
    
    println!("\n=== Performance Comparison ===");
    println!("With special cases: {:.1} μs average", special_time as f64 / 10.0);
    println!("Without special cases: {:.1} μs average", standard_time as f64 / 10.0);
    
    if special_time < standard_time {
        let speedup = standard_time as f64 / special_time as f64;
        println!("Special case detection provides {:.1}x speedup", speedup);
    }
    
    println!("\nSpecial case success rate: {}/10", special_success);
    
    // Test specific optimizations
    println!("\n=== Specific Optimization Tests ===\n");
    
    // Large perfect square
    let large_square = BigInt::from(1000003) * BigInt::from(1000003);
    println!("Large perfect square: {}", large_square);
    
    let start = Instant::now();
    if let Some(factors) = try_special_cases(&large_square) {
        let elapsed = start.elapsed();
        println!("  Detected in {:?}", elapsed);
        println!("  Factors: {} × {}", factors.p, factors.q);
    }
    
    // Large twin primes
    let twin1 = BigInt::from(1000037u64);
    let twin2 = BigInt::from(1000039u64);
    let large_twin = &twin1 * &twin2;
    println!("\nLarge twin primes: {} = {} × {}", large_twin, twin1, twin2);
    
    let start = Instant::now();
    if let Some(factors) = try_special_cases(&large_twin) {
        let elapsed = start.elapsed();
        println!("  Detected in {:?}", elapsed);
        println!("  Factors: {} × {}", factors.p, factors.q);
    }
}