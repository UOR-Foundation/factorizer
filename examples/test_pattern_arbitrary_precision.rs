//! Test The Pattern's ability to handle arbitrary precision semiprimes
//! 
//! This evaluates the refactored implementation against the test matrix
//! to ensure it can factor semiprimes of any size without precision limits.

use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use std::str::FromStr;

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

fn main() {
    println!("Testing The Pattern with Arbitrary Precision");
    println!("===========================================\n");
    
    // Load test matrix
    let matrix_data = std::fs::read_to_string("data/test_matrix.json")
        .expect("Failed to load test matrix");
    
    let test_matrix: TestMatrix = serde_json::from_str(&matrix_data)
        .expect("Failed to parse test matrix");
    
    // Initialize pattern with pre-computed basis
    let mut pattern = UniversalPattern::with_precomputed_basis();
    
    // Test specific bit ranges to verify arbitrary precision
    let test_ranges = vec![
        (8, 64),     // Small numbers (should work)
        (128, 128),  // At the old limit
        (224, 224),  // Just before the old limit  
        (232, 256),  // Beyond the old 224-bit limit
        (512, 512),  // Large numbers
    ];
    
    println!("Testing critical bit ranges:\n");
    
    for (min_bits, max_bits) in test_ranges {
        println!("Testing {}-{} bit range:", min_bits, max_bits);
        println!("{}", "-".repeat(50));
        
        let mut total = 0;
        let mut success = 0;
        
        // Test all cases in this range
        for (bit_length, cases) in &test_matrix.test_cases {
            if *bit_length >= min_bits && *bit_length <= max_bits {
                for case in cases.iter().take(3) { // Test first 3 cases per bit length
                    total += 1;
                    
                    let n = Number::from_str(&case.n).unwrap();
                    let expected_p = Number::from_str(&case.p).unwrap();
                    let expected_q = Number::from_str(&case.q).unwrap();
                    
                    print!("  {}-bit: ", bit_length);
                    
                    let start = Instant::now();
                    let timeout = Duration::from_secs(30);
                    
                    // Try to factor with timeout
                    let result = std::thread::scope(|s| {
                        let handle = s.spawn(|| {
                            pattern.recognize(&n)
                                .and_then(|r| pattern.formalize(r))
                                .and_then(|f| pattern.execute(f))
                        });
                        
                        let start_time = Instant::now();
                        loop {
                            if handle.is_finished() {
                                return handle.join().unwrap();
                            }
                            
                            if start_time.elapsed() > timeout {
                                return Err(rust_pattern_solver::error::PatternError::ExecutionError(
                                    "Timeout".to_string()
                                ));
                            }
                            
                            std::thread::sleep(Duration::from_millis(100));
                        }
                    });
                    
                    let elapsed = start.elapsed();
                    
                    match result {
                        Ok(factors) => {
                            if (factors.p == expected_p && factors.q == expected_q) ||
                               (factors.p == expected_q && factors.q == expected_p) {
                                success += 1;
                                println!("✓ SUCCESS in {:.3}ms ({})", 
                                    elapsed.as_secs_f64() * 1000.0, 
                                    factors.method);
                                    
                                // Verify no precision loss
                                let product = &factors.p * &factors.q;
                                if product != n {
                                    println!("    WARNING: Product verification failed!");
                                }
                            } else {
                                println!("✗ WRONG FACTORS");
                                println!("    Expected: {} × {}", expected_p, expected_q);
                                println!("    Got:      {} × {}", factors.p, factors.q);
                            }
                        }
                        Err(e) => {
                            println!("✗ ERROR: {}", e);
                            
                            // Check if this is due to precision limits
                            if n.bit_length() > 224 {
                                println!("    Note: This is beyond the old 224-bit limit");
                            }
                        }
                    }
                }
            }
        }
        
        if total > 0 {
            let rate = (success as f64 / total as f64) * 100.0;
            println!("\nSuccess rate: {}/{} ({:.1}%)\n", success, total, rate);
            
            if min_bits > 224 && success == 0 {
                println!("⚠️  CRITICAL: Pattern cannot handle numbers beyond 224 bits!");
                println!("   This indicates the arbitrary precision refactoring is incomplete.\n");
            }
        }
    }
    
    // Test a specific large number to demonstrate the issue
    println!("\nDirect Test of Large Numbers:");
    println!("{}", "-".repeat(50));
    
    // Create a 256-bit semiprime
    let p_128 = Number::from_str_radix("340282366920938463463374607431768211503", 10).unwrap();
    let q_128 = Number::from_str_radix("340282366920938463463374607431768211527", 10).unwrap();
    let n_256 = &p_128 * &q_128;
    
    println!("Testing {}-bit number directly:", n_256.bit_length());
    println!("n = {}", n_256);
    
    let start = Instant::now();
    match pattern.recognize(&n_256)
        .and_then(|r| pattern.formalize(r))
        .and_then(|f| pattern.execute(f)) {
        Ok(factors) => {
            if (factors.p == p_128 && factors.q == q_128) ||
               (factors.p == q_128 && factors.q == p_128) {
                println!("✓ SUCCESS! The Pattern CAN handle 256-bit numbers!");
                println!("  Time: {:.3}ms", start.elapsed().as_secs_f64() * 1000.0);
                println!("  Method: {}", factors.method);
            } else {
                println!("✗ Wrong factors returned");
            }
        }
        Err(e) => {
            println!("✗ FAILED: {}", e);
            println!("  This suggests The Pattern still has precision limitations.");
        }
    }
    
    println!("\nConclusion:");
    println!("-----------");
    println!("If The Pattern fails on numbers > 224 bits, it means:");
    println!("1. The UniversalPattern still has u128/f64 conversions limiting precision");
    println!("2. The refactoring to support arbitrary precision is incomplete");
    println!("3. We need to audit and fix all precision-limiting code paths");
}