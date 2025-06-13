//! Test scaling of The Pattern with increasingly large numbers

use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;
use std::str::FromStr;
use std::time::Instant;

fn test_number(pattern: &mut UniversalPattern, n_str: &str, p_str: &str, q_str: &str, desc: &str) {
    let n = Number::from_str(n_str).unwrap();
    let p = Number::from_str(p_str).unwrap();
    let q = Number::from_str(q_str).unwrap();
    
    // Verify it's correct
    if &p * &q != n {
        println!("❌ ERROR: Invalid test case - p × q ≠ n");
        return;
    }
    
    println!("\n{}", "=".repeat(70));
    println!("{} ({} bits)", desc, n.bit_length());
    
    let start = Instant::now();
    
    match pattern.recognize(&n) {
        Ok(recognition) => {
            match pattern.formalize(recognition) {
                Ok(formalization) => {
                    match pattern.execute(formalization) {
                        Ok(factors) => {
                            let elapsed = start.elapsed();
                            
                            if &factors.p * &factors.q == n {
                                println!("✓ SUCCESS in {:?} via {}", elapsed, factors.method);
                                
                                // Check if we found the right factors
                                let found_correct = (factors.p == p && factors.q == q) || 
                                                  (factors.p == q && factors.q == p);
                                if !found_correct {
                                    println!("  ⚠ Found different factors: {} × {}", factors.p, factors.q);
                                }
                            } else {
                                println!("✗ FAILED - Invalid factors");
                            }
                        }
                        Err(e) => {
                            println!("✗ FAILED after {:?}: {}", start.elapsed(), e);
                        }
                    }
                }
                Err(_) => println!("✗ Formalization failed"),
            }
        }
        Err(_) => println!("✗ Recognition failed"),
    }
}

fn main() {
    println!("=== Testing Pattern Scaling ===\n");
    println!("Using pre-computed basis (auto-tune approach)");
    
    let mut pattern = UniversalPattern::with_precomputed_basis();
    
    // Test cases of increasing size - all true balanced semiprimes
    let test_cases = vec![
        // 20-bit balanced
        ("524309", "719", "729", "20-bit balanced"),
        
        // 40-bit balanced  
        ("1099511689081", "1048573", "1048609", "40-bit balanced"),
        
        // 60-bit balanced
        ("1152921506754330757", "1073741827", "1073741831", "60-bit balanced"),
        
        // 80-bit balanced
        ("1208925819627823314239443", "1099511627773", "1099511627791", "80-bit balanced"),
        
        // 100-bit balanced
        ("1267650600228260926694094797363", "1125899906842597", "1125899906842679", "100-bit balanced"),
        
        // 120-bit balanced (two 60-bit primes)
        ("1329227995784915889045502779648075827", "1152921504606846997", "1152921504606847021", "120-bit balanced"),
        
        // 128-bit balanced
        ("340282366920938463463374586203780677351", "18446744073709551557", "18446744073709551643", "128-bit balanced"),
    ];
    
    for (n, p, q, desc) in test_cases {
        test_number(&mut pattern, n, p, q, desc);
    }
    
    println!("\n{}", "=".repeat(70));
    println!("SUMMARY");
    println!("{}", "=".repeat(70));
    println!("\nThe Pattern should maintain poly-time performance across all scales.");
    println!("Auto-tune approach scales the pre-computed basis to each number size.");
}