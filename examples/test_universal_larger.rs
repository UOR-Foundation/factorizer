//! Test Universal Pattern with larger numbers and RSA-like semiprimes

use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;
use std::time::Instant;

fn test_number(pattern: &mut UniversalPattern, n: Number, description: &str) {
    println!("\nTesting: {} ({})", n, description);
    println!("Bit length: {}", n.bit_length());
    
    let start = Instant::now();
    
    // Stage 1: Recognition
    match pattern.recognize(&n) {
        Ok(recognition) => {
            println!("Recognition completed in {:?}", start.elapsed());
            println!("  φ-component: {:.6}", recognition.phi_component);
            println!("  π-component: {:.6}", recognition.pi_component);
            println!("  e-component: {:.6}", recognition.e_component);
            
            // Stage 2: Formalization
            let form_start = Instant::now();
            match pattern.formalize(recognition) {
                Ok(formalization) => {
                    println!("Formalization completed in {:?}", form_start.elapsed());
                    println!("  Resonance peaks: {:?}", formalization.resonance_peaks.len());
                    
                    // Stage 3: Execution
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
                            } else {
                                println!("  ✗ Verification FAILED!");
                            }
                        }
                        Err(e) => {
                            println!("✗ Execution failed after {:?}: {}", exec_start.elapsed(), e);
                            println!("  Total time: {:?}", start.elapsed());
                        }
                    }
                }
                Err(e) => {
                    println!("✗ Formalization failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("✗ Recognition failed: {}", e);
        }
    }
}

fn main() -> rust_pattern_solver::Result<()> {
    println!("=== Universal Pattern - Larger Number Tests ===");
    
    let mut pattern = UniversalPattern::new();
    
    // Test cases of increasing difficulty
    let test_cases = vec![
        // Small balanced semiprimes
        (Number::from(35u32), "5 × 7"),
        (Number::from(77u32), "7 × 11"),
        (Number::from(221u32), "13 × 17"),
        
        // Medium semiprimes
        (Number::from(9409u32), "97² - prime square"),
        (Number::from(25217u32), "151 × 167 - twin-like primes"),
        (Number::from(142763u32), "367 × 389 - consecutive primes"),
        
        // Larger semiprimes (32-bit range)
        (Number::from(1073676289u64), "32767² - Mersenne prime square"),
        (Number::from(2147673613u64), "46337 × 46349 - twin primes"),
        
        // 48-bit semiprime
        (Number::from(140737436084957u64), "11863283 × 11863279"),
        
        // 56-bit semiprime  
        (Number::from(36028792748750231u64), "189812507 × 189812533"),
        
        // 64-bit semiprime (if it can handle it)
        (Number::from(9223372012704246007u64), "3037000493 × 3037000499"),
    ];
    
    for (n, desc) in test_cases {
        test_number(&mut pattern, n, desc);
    }
    
    // Test with RSA-8 (the smallest RSA challenge that was factored)
    println!("\n=== RSA-8 Test ===");
    let rsa8 = Number::from(4189u32); // 59 × 71 (not actual RSA-8, but similar size for testing)
    test_number(&mut pattern, rsa8, "59 × 71 (RSA-8 sized)");
    
    // Test special cases
    println!("\n=== Special Cases ===");
    
    // Golden ratio related
    let golden_semi = Number::from(987u32); // Fibonacci number, 3 × 7 × 47
    test_number(&mut pattern, golden_semi, "987 - Fibonacci number");
    
    // Perfect square minus 1
    let square_minus_1 = Number::from(9999u32); // 100² - 1 = 99 × 101
    test_number(&mut pattern, square_minus_1, "100² - 1");
    
    Ok(())
}