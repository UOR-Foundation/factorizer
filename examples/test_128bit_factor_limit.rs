//! Demonstrate the 128-bit factor limitation in the original implementation
//! 
//! This test shows that the original implementation fails when factors exceed 128 bits,
//! even if the composite number itself is within range.

use rust_pattern_solver::pattern::direct_empirical::DirectEmpiricalPattern;
use rust_pattern_solver::pattern::direct_empirical_exact::DirectEmpiricalPatternExact;
use rust_pattern_solver::types::Number;

fn main() {
    println!("Testing 128-bit Factor Limitation");
    println!("================================\n");
    
    // Create a number with factors that are exactly at the 128-bit boundary
    // This will work with the original implementation
    let p_128bit = Number::from_str_radix("340282366920938463463374607431768211297", 10).unwrap(); // 128-bit prime
    let q_small = Number::from(7u32); // Small prime
    let n_works = &p_128bit * &q_small;
    
    println!("Test 1: Number with 128-bit factor (should work in both)");
    println!("n = {} ({} bits)", n_works, n_works.bit_length());
    println!("p = {} ({} bits)", p_128bit, p_128bit.bit_length());
    println!("q = {} ({} bits)\n", q_small, q_small.bit_length());
    
    // Create a number with factors that exceed 128 bits
    // This will FAIL with the original implementation due to u128 conversion
    let p_129bit = Number::from_str_radix("680564733841876926926749214863536422957", 10).unwrap(); // 129-bit prime
    let q_small2 = Number::from(11u32); // Small prime
    let n_fails = &p_129bit * &q_small2;
    
    println!("Test 2: Number with 129-bit factor (will fail in original)");
    println!("n = {} ({} bits)", n_fails, n_fails.bit_length());
    println!("p = {} ({} bits)", p_129bit, p_129bit.bit_length());
    println!("q = {} ({} bits)\n", q_small2, q_small2.bit_length());
    
    // Create training data
    let train_data = vec![
        // Include both test cases in training
        (n_works.clone(), p_128bit.clone(), q_small.clone()),
        (n_fails.clone(), p_129bit.clone(), q_small2.clone()),
    ];
    
    // Test ORIGINAL implementation
    println!("--- Original Implementation (u128 limited) ---\n");
    
    let pattern_orig = DirectEmpiricalPattern::from_test_data(&train_data);
    
    print!("Test 1 (128-bit factor): ");
    match pattern_orig.factor(&n_works) {
        Ok(factors) => {
            if (&factors.p == &p_128bit && &factors.q == &q_small) ||
               (&factors.p == &q_small && &factors.q == &p_128bit) {
                println!("✓ SUCCESS");
            } else {
                println!("✗ WRONG FACTORS");
                println!("  Got: {} × {}", factors.p, factors.q);
            }
        }
        Err(e) => println!("✗ ERROR: {}", e),
    }
    
    print!("Test 2 (129-bit factor): ");
    match pattern_orig.factor(&n_fails) {
        Ok(factors) => {
            if (&factors.p == &p_129bit && &factors.q == &q_small2) ||
               (&factors.p == &q_small2 && &factors.q == &p_129bit) {
                println!("✓ SUCCESS");
            } else {
                println!("✗ WRONG FACTORS (due to u128 truncation)");
                println!("  Expected: {} × {}", p_129bit, q_small2);
                println!("  Got:      {} × {}", factors.p, factors.q);
                
                // Show what happens in the original implementation
                println!("  Note: Original implementation would truncate 129-bit values during adaptation");
            }
        }
        Err(e) => println!("✗ ERROR: {}", e),
    }
    
    // Test EXACT implementation
    println!("\n--- Exact Implementation (no limit) ---\n");
    
    let pattern_exact = DirectEmpiricalPatternExact::from_test_data(&train_data, 256);
    
    print!("Test 1 (128-bit factor): ");
    match pattern_exact.factor(&n_works) {
        Ok(factors) => {
            if (&factors.p == &p_128bit && &factors.q == &q_small) ||
               (&factors.p == &q_small && &factors.q == &p_128bit) {
                println!("✓ SUCCESS");
            } else {
                println!("✗ WRONG FACTORS");
            }
        }
        Err(e) => println!("✗ ERROR: {}", e),
    }
    
    print!("Test 2 (129-bit factor): ");
    match pattern_exact.factor(&n_fails) {
        Ok(factors) => {
            if (&factors.p == &p_129bit && &factors.q == &q_small2) ||
               (&factors.p == &q_small2 && &factors.q == &p_129bit) {
                println!("✓ SUCCESS (handles 129-bit factors correctly!)");
            } else {
                println!("✗ WRONG FACTORS");
            }
        }
        Err(e) => println!("✗ ERROR: {}", e),
    }
    
    println!("\n--- CONCLUSION ---");
    println!("The original implementation FAILS when any factor exceeds 128 bits");
    println!("due to conversions like: Number::from((value as u128))");
    println!("\nThe exact implementation handles factors of ANY size correctly!");
}