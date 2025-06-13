//! Progressive testing from our current capabilities up to RSA challenges
//! This tests the universal pattern on increasingly large semiprimes

use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;
use std::time::Instant;
use std::str::FromStr;

fn test_number(pattern: &mut UniversalPattern, n: Number, description: &str) -> bool {
    println!("\n{}", "=".repeat(80));
    println!("Testing: {} ", description);
    println!("Number: {} ({} bits)", n, n.bit_length());
    println!("{}", "=".repeat(80));
    
    let start = Instant::now();
    
    // Recognition
    let recog_start = Instant::now();
    match pattern.recognize(&n) {
        Ok(recognition) => {
            println!("Recognition completed in {:?}", recog_start.elapsed());
            println!("  φ-component: {:.6}", recognition.phi_component);
            println!("  π-component: {:.6}", recognition.pi_component);
            println!("  e-component: {:.6}", recognition.e_component);
            
            // Formalization
            let formal_start = Instant::now();
            match pattern.formalize(recognition) {
                Ok(formalization) => {
                    println!("Formalization completed in {:?}", formal_start.elapsed());
                    println!("  Resonance peaks: {}", formalization.resonance_peaks.len());
                    
                    // Execution
                    let exec_start = Instant::now();
                    match pattern.execute(formalization) {
                        Ok(factors) => {
                            println!("✓ Factorization successful in {:?}", exec_start.elapsed());
                            println!("  Factors: {} × {}", factors.p, factors.q);
                            println!("  Method: {}", factors.method);
                            println!("  Total time: {:?}", start.elapsed());
                            
                            // Verify
                            if &factors.p * &factors.q == n {
                                println!("  ✓ Verification passed");
                                return true;
                            } else {
                                println!("  ✗ Verification FAILED!");
                                return false;
                            }
                        }
                        Err(e) => {
                            println!("✗ Execution failed after {:?}: {}", exec_start.elapsed(), e);
                            println!("  Total time: {:?}", start.elapsed());
                            return false;
                        }
                    }
                }
                Err(e) => {
                    println!("✗ Formalization failed: {}", e);
                    return false;
                }
            }
        }
        Err(e) => {
            println!("✗ Recognition failed: {}", e);
            return false;
        }
    }
}

fn main() {
    let mut pattern = UniversalPattern::new();
    let mut successes = 0;
    let mut failures = 0;
    
    println!("=== Universal Pattern RSA Progression ===\n");
    
    // Stage 1: Verify our current capabilities (up to 64-bit)
    println!("\n=== Stage 1: Current Capabilities (up to 64-bit) ===");
    
    let stage1_tests = vec![
        // 32-bit balanced semiprime
        (Number::from(2147673613u64), "2147673613 = 46337 × 46349 (32-bit twin primes)"),
        
        // 48-bit semiprime
        (Number::from(140737436084957u64), "140737436084957 = 11863283 × 11863279 (48-bit)"),
        
        // 64-bit semiprime
        (Number::from(9223372012704246007u64), "9223372012704246007 = 3037000493 × 3037000499 (64-bit)"),
    ];
    
    for (n, desc) in stage1_tests {
        if test_number(&mut pattern, n, desc) {
            successes += 1;
        } else {
            failures += 1;
        }
    }
    
    // Stage 2: Bridge to larger numbers (80-128 bit)
    println!("\n\n=== Stage 2: Bridge Numbers (80-128 bit) ===");
    
    let stage2_tests = vec![
        // 80-bit semiprime
        (
            Number::from_str("1208925819627823314239443").unwrap(),
            "80-bit: 1099511627791 × 1099511627773"
        ),
        
        // 96-bit semiprime
        (
            Number::from_str("79228162514280100192239747807").unwrap(),
            "96-bit: 281474976710677 × 281474976710691"
        ),
        
        // 112-bit semiprime  
        (
            Number::from_str("5192296858534833249022831287600429").unwrap(),
            "112-bit: 72057594037927961 × 72057594037927989"
        ),
        
        // 128-bit semiprime
        (
            Number::from_str("340282366920938464127457394085312069931").unwrap(),
            "128-bit: 18446744073709551629 × 18446744073709551639"
        ),
    ];
    
    for (n, desc) in stage2_tests {
        if test_number(&mut pattern, n, desc) {
            successes += 1;
        } else {
            failures += 1;
        }
    }
    
    // Stage 3: Pre-RSA challenges (150-250 bit)
    println!("\n\n=== Stage 3: Pre-RSA Challenges (150-250 bit) ===");
    
    let stage3_tests = vec![
        // 160-bit semiprime
        (
            Number::from_str("1461501637330902918203684832716283019655932542983").unwrap(),
            "160-bit semiprime"
        ),
        
        // 200-bit semiprime  
        (
            Number::from_str("1606938044258990275541962092341162602522202993782792835301399").unwrap(),
            "200-bit semiprime"
        ),
    ];
    
    for (n, desc) in stage3_tests {
        if test_number(&mut pattern, n, desc) {
            successes += 1;
        } else {
            failures += 1;
        }
    }
    
    // Stage 4: Small RSA challenges
    println!("\n\n=== Stage 4: Small RSA Challenges ===");
    
    // RSA-100 (330 bits)
    let rsa100 = Number::from_str(
        "1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139"
    ).unwrap();
    
    if test_number(&mut pattern, rsa100, "RSA-100 (330 bits)") {
        successes += 1;
        println!("\n🎉 RSA-100 FACTORED! The Pattern works at RSA scale!");
    } else {
        failures += 1;
        println!("\nRSA-100 not yet factored. Need further tuning.");
    }
    
    // Summary
    println!("\n\n{}", "=".repeat(80));
    println!("=== SUMMARY ===");
    println!("{}", "=".repeat(80));
    println!("Total tests: {}", successes + failures);
    println!("Successes: {} ✓", successes);
    println!("Failures: {} ✗", failures);
    println!("Success rate: {:.1}%", (successes as f64 / (successes + failures) as f64) * 100.0);
    
    if failures == 0 {
        println!("\n✨ All tests passed! The Pattern is ready for larger challenges!");
    } else {
        println!("\n⚠️  Some tests failed. Analysis needed for larger numbers.");
    }
}