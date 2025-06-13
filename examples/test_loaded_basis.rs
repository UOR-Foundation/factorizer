//! Test the pattern with loaded pre-computed basis

use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;
use std::str::FromStr;
use std::time::Instant;

fn main() {
    println!("=== Testing Pattern with Loaded Basis ===\n");
    
    // Create pattern - this will load from file
    println!("Loading pre-computed basis from disk...");
    let start = Instant::now();
    let mut pattern = UniversalPattern::with_precomputed_basis();
    println!("Basis loaded in {:?}\n", start.elapsed());
    
    // Test cases of increasing size
    let test_cases = vec![
        ("143", "11 × 13"),
        ("9223372012704246007", "2147483647 × 4294967197"),
        ("10403", "101 × 103"),
        ("25217", "151 × 167"),
        ("142763", "367 × 389"),
    ];
    
    println!("Testing with loaded basis:");
    println!("{}", "=".repeat(70));
    
    for (n_str, expected) in test_cases {
        let n = Number::from_str(n_str).unwrap();
        print!("n = {} ({}): ", n, expected);
        
        let start = Instant::now();
        match pattern.recognize(&n) {
            Ok(recognition) => {
                match pattern.formalize(recognition) {
                    Ok(formalization) => {
                        match pattern.execute(formalization) {
                            Ok(factors) => {
                                let elapsed = start.elapsed();
                                let verified = &factors.p * &factors.q == n;
                                if verified {
                                    println!("✓ {} × {} in {:?}", factors.p, factors.q, elapsed);
                                } else {
                                    println!("✗ Invalid: {} × {} = {}", factors.p, factors.q, &factors.p * &factors.q);
                                }
                            }
                            Err(e) => println!("✗ Execution: {}", e),
                        }
                    }
                    Err(e) => println!("✗ Formalization: {}", e),
                }
            }
            Err(e) => println!("✗ Recognition: {}", e),
        }
    }
    
    println!("\n{}", "=".repeat(70));
    println!("Testing larger numbers with loaded basis:");
    println!("{}", "=".repeat(70));
    
    // Test with RSA-like number
    let rsa_test = "1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139";
    let n = Number::from_str(rsa_test).unwrap();
    println!("\nTesting 100-digit RSA number:");
    println!("n = {}", n);
    
    let start = Instant::now();
    match pattern.recognize(&n) {
        Ok(recognition) => {
            match pattern.formalize(recognition) {
                Ok(formalization) => {
                    match pattern.execute(formalization) {
                        Ok(factors) => {
                            let elapsed = start.elapsed();
                            println!("✓ Factored in {:?} via {}", elapsed, factors.method);
                            println!("  p = {}", factors.p);
                            println!("  q = {}", factors.q);
                            
                            // Verify
                            if &factors.p * &factors.q == n {
                                println!("  ✓ Verification passed");
                            } else {
                                println!("  ✗ Verification failed");
                            }
                        }
                        Err(e) => println!("✗ Execution failed after {:?}: {}", start.elapsed(), e),
                    }
                }
                Err(e) => println!("✗ Formalization failed: {}", e),
            }
        }
        Err(e) => println!("✗ Recognition failed: {}", e),
    }
    
    println!("\nNote: The pre-computed basis is loaded from data/basis/universal_basis.json");
    println!("This avoids regenerating the basis every time, improving startup performance.");
}