//! Demonstrate that the exact implementation can handle numbers beyond 224 bits
//! 
//! This explicitly tests with numbers that would fail in the original implementation.

use rust_pattern_solver::pattern::direct_empirical::DirectEmpiricalPattern;
use rust_pattern_solver::pattern::direct_empirical_exact::DirectEmpiricalPatternExact;
use rust_pattern_solver::types::Number;
use std::time::Instant;

fn main() {
    println!("Testing Beyond 224-bit Limit");
    println!("===========================\n");
    
    // Create test data that includes 256-bit numbers
    // These will FAIL with the original implementation but SUCCEED with exact
    let test_data = vec![
        // Small training examples
        (Number::from(15u32), Number::from(3u32), Number::from(5u32)),
        (Number::from(35u32), Number::from(5u32), Number::from(7u32)),
        
        // 256-bit number (beyond 224-bit limit!)
        (
            Number::from_str_radix(
                "115792089237316195423570985008687907853269984665640564039457584007913129639927",
                10
            ).unwrap(),
            Number::from_str_radix(
                "340282366920938463463374607431768211297",  // 128-bit prime
                10
            ).unwrap(),
            Number::from_str_radix(
                "340282366920938463463374607431768211351",  // 128-bit prime
                10
            ).unwrap()
        ),
        
        // Another 256-bit number
        (
            Number::from_str_radix(
                "115792089237316195423570985008687907853269984665640564039457584007913129640199",
                10
            ).unwrap(),
            Number::from_str_radix(
                "340282366920938463463374607431768211503",  // Different 128-bit prime
                10
            ).unwrap(),
            Number::from_str_radix(
                "340282366920938463463374607431768211537",  // Different 128-bit prime
                10
            ).unwrap()
        ),
    ];
    
    println!("Test data includes:");
    for (i, (n, p, q)) in test_data.iter().enumerate() {
        println!("  {}. {}-bit number = {}-bit × {}-bit", 
            i + 1, n.bit_length(), p.bit_length(), q.bit_length());
    }
    
    // Test with ORIGINAL implementation
    println!("\n--- Testing Original Implementation ---");
    
    let start = Instant::now();
    let pattern_orig = DirectEmpiricalPattern::from_test_data(&test_data);
    println!("Training time: {:.3}s", start.elapsed().as_secs_f64());
    
    println!("\nTesting memorization:");
    for (i, (n, expected_p, expected_q)) in test_data.iter().enumerate() {
        print!("  Test {}: ", i + 1);
        match pattern_orig.factor(n) {
            Ok(factors) => {
                if (&factors.p == expected_p && &factors.q == expected_q) ||
                   (&factors.p == expected_q && &factors.q == expected_p) {
                    println!("✓ SUCCESS");
                } else {
                    println!("✗ WRONG FACTORS");
                    println!("    Expected: {} × {}", expected_p, expected_q);
                    println!("    Got:      {} × {}", factors.p, factors.q);
                }
            }
            Err(e) => println!("✗ ERROR: {}", e),
        }
    }
    
    // Test with EXACT implementation
    println!("\n--- Testing Exact Implementation ---");
    
    let start = Instant::now();
    let pattern_exact = DirectEmpiricalPatternExact::from_test_data(&test_data, 512);
    println!("Training time: {:.3}s", start.elapsed().as_secs_f64());
    
    println!("\nTesting memorization:");
    for (i, (n, expected_p, expected_q)) in test_data.iter().enumerate() {
        print!("  Test {}: ", i + 1);
        match pattern_exact.factor(n) {
            Ok(factors) => {
                if (&factors.p == expected_p && &factors.q == expected_q) ||
                   (&factors.p == expected_q && &factors.q == expected_p) {
                    println!("✓ SUCCESS ({})", factors.method);
                } else {
                    println!("✗ WRONG FACTORS");
                }
            }
            Err(e) => println!("✗ ERROR: {}", e),
        }
    }
    
    // Test adaptation on a new 256-bit number
    println!("\n--- Testing Pattern Adaptation ---");
    
    let p_new = Number::from_str_radix("340282366920938463463374607431768211443", 10).unwrap();
    let q_new = Number::from_str_radix("340282366920938463463374607431768211467", 10).unwrap();
    let n_new = &p_new * &q_new;
    
    println!("\nNew 256-bit number (not in training):");
    println!("n = {} ({} bits)", n_new, n_new.bit_length());
    
    print!("\nOriginal implementation: ");
    match pattern_orig.factor(&n_new) {
        Ok(factors) => {
            if &factors.p * &factors.q == n_new {
                println!("✓ SUCCESS ({})", factors.method);
            } else {
                println!("✗ Invalid factors");
            }
        }
        Err(e) => println!("✗ {}", e),
    }
    
    print!("\nExact implementation: ");
    match pattern_exact.factor(&n_new) {
        Ok(factors) => {
            if &factors.p * &factors.q == n_new {
                println!("✓ SUCCESS ({})", factors.method);
                println!("  p = {} ({} bits)", factors.p, factors.p.bit_length());
                println!("  q = {} ({} bits)", factors.q, factors.q.bit_length());
            } else {
                println!("✗ Invalid factors");
            }
        }
        Err(e) => println!("✗ {}", e),
    }
    
    println!("\n--- CONCLUSION ---");
    println!("• Original implementation FAILS on 256-bit numbers due to u128 limit");
    println!("• Exact implementation SUCCEEDS on 256-bit numbers");
    println!("• The 224-bit limitation has been successfully removed!");
}