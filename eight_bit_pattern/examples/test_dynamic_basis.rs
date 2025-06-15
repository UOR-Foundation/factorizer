//! Test dynamic basis sizing refactor
//!
//! Run with: cargo run --example test_dynamic_basis

use eight_bit_pattern::{compute_basis, recognize_factors, TunerParams, bit_size};
use num_bigint::BigInt;

fn main() {
    println!("=== Testing Dynamic Basis Sizing ===\n");
    
    let params = TunerParams::default();
    
    // Test various bit sizes
    let test_cases = vec![
        ("8-bit", BigInt::from(143u32)),          // 11 × 13
        ("16-bit", BigInt::from(58081u32)),       // 241 × 241
        ("32-bit", BigInt::from(3215031751u64)),  // 56599 × 56809
        ("64-bit", BigInt::from(u64::MAX - 10)),
        ("128-bit", BigInt::from(1u128) << 120),
        ("256-bit", BigInt::from(1u128) << 250),
        ("512-bit", BigInt::from(1u128) << 500),
        ("1024-bit", BigInt::from(1u128) << 1020),
    ];
    
    for (name, n) in test_cases {
        let bits = bit_size(&n);
        let basis = compute_basis(&n, &params);
        let expected_channels = ((bits + 7) / 8).max(32);
        
        println!("{} number ({} bits):", name, bits);
        println!("  Basis channels: {}", basis.num_channels);
        println!("  Expected: >= {}", expected_channels);
        println!("  Status: {}", 
            if basis.num_channels >= expected_channels { "✓ PASS" } else { "✗ FAIL" }
        );
        
        // Try to factor small numbers
        if bits <= 32 {
            let result = recognize_factors(&n, &params);
            if let Some(factors) = result {
                if factors.verify(&n) {
                    println!("  Factorization: {} = {} × {}", n, factors.p, factors.q);
                }
            }
        }
        
        println!();
    }
    
    // Test that basis grows appropriately
    println!("=== Channel Count Scaling ===\n");
    
    for power in (0..=10).map(|i| i * 100) {
        let n = BigInt::from(1u128) << power;
        let bits = bit_size(&n);
        let basis = compute_basis(&n, &params);
        
        println!("2^{} ({} bits): {} channels", power, bits, basis.num_channels);
    }
}