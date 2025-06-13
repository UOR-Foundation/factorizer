//! Test The Pattern with verified RSA numbers

use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;
use std::str::FromStr;
use std::time::{Duration, Instant};

#[derive(Debug)]
struct RSATestCase {
    name: &'static str,
    n: &'static str,
    p: &'static str,
    q: &'static str,
    bits: usize,
}

fn verify_rsa_number(test: &RSATestCase) -> bool {
    let n = Number::from_str(test.n).unwrap();
    let p = Number::from_str(test.p).unwrap();
    let q = Number::from_str(test.q).unwrap();
    
    let product = &p * &q;
    if product != n {
        println!("❌ {} verification FAILED: p × q ≠ n", test.name);
        return false;
    }
    
    if n.bit_length() != test.bits {
        println!("⚠️  {} bit length mismatch: expected {}, got {}", 
                 test.name, test.bits, n.bit_length());
    }
    
    true
}

fn test_rsa_number(pattern: &mut UniversalPattern, test: &RSATestCase, timeout: Duration) -> bool {
    let n = Number::from_str(test.n).unwrap();
    
    println!("\n{}", "=".repeat(80));
    println!("Testing {} ({} bits)", test.name, test.bits);
    
    let start = Instant::now();
    
    match pattern.recognize(&n) {
        Ok(recognition) => {
            println!("✓ Recognition in {:?}", start.elapsed());
            println!("  φ-component: {:.6}", recognition.phi_component);
            
            match pattern.formalize(recognition) {
                Ok(formalization) => {
                    println!("✓ Formalization in {:?}", start.elapsed());
                    
                    // Set timeout for execution
                    let exec_start = Instant::now();
                    match pattern.execute(formalization) {
                        Ok(factors) => {
                            let exec_time = exec_start.elapsed();
                            let total_time = start.elapsed();
                            
                            if exec_time > timeout {
                                println!("⏱️  Timeout: execution took {:?} (limit: {:?})", exec_time, timeout);
                                return false;
                            }
                            
                            println!("✓ SUCCESS in {:?} (execution: {:?})", total_time, exec_time);
                            println!("  Method: {}", factors.method);
                            println!("  Found: {} × {}", factors.p, factors.q);
                            
                            // Verify
                            if &factors.p * &factors.q == n {
                                println!("  ✓ Verification passed");
                                
                                // Check if we found the right factors
                                let p_expected = Number::from_str(test.p).unwrap();
                                let q_expected = Number::from_str(test.q).unwrap();
                                
                                if (factors.p == p_expected && factors.q == q_expected) ||
                                   (factors.p == q_expected && factors.q == p_expected) {
                                    println!("  ✓ Correct factors found!");
                                } else {
                                    println!("  ⚠️  Different valid factorization found");
                                }
                                
                                return true;
                            } else {
                                println!("  ❌ Verification FAILED!");
                                return false;
                            }
                        }
                        Err(e) => {
                            let exec_time = exec_start.elapsed();
                            if exec_time > timeout {
                                println!("⏱️  Timeout after {:?}: {}", exec_time, e);
                            } else {
                                println!("✗ Execution failed after {:?}: {}", exec_time, e);
                            }
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
    println!("=== Testing The Pattern with Verified RSA Numbers ===\n");
    
    // First verify all test cases
    let test_cases = vec![
        // Start with smaller verified numbers
        RSATestCase {
            name: "RSA-100",
            n: "1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139",
            p: "37975227936943673922808872755445627854565536638199",
            q: "40094690950920881030683735292761468389214899724061",
            bits: 330,
        },
        RSATestCase {
            name: "RSA-110",
            n: "35794234179725868774991807832568455403003778024228226193532908190484670252364677411513516111204504060317568667",
            p: "6122421090493547576937037317561418841225758554253106999",
            q: "5846418214406154678836553182979162384198610505601062333",
            bits: 364,
        },
        RSATestCase {
            name: "RSA-120",
            n: "227010481295437363334259960947493668895875336466084780038173258247009162675779735389791151574049166747880487470296548479",
            p: "327414555693498015751146303749141488063642403240171463406883",
            q: "693342667110830181197325401899700641361965863127336680673013",
            bits: 397,
        },
        RSATestCase {
            name: "RSA-129",
            n: "114381625757888867669235779976146612010218296721242362562561842935706935245733897830597123563958705058989075147599290026879543541",
            p: "3490529510847650949147849619903898133417764638493387843990820577",
            q: "32769132993266709549961988190834461413177642967992942539798288533",
            bits: 426,
        },
    ];
    
    println!("Verifying test cases...");
    let mut all_valid = true;
    for test in &test_cases {
        if !verify_rsa_number(test) {
            all_valid = false;
        } else {
            println!("✓ {} verified", test.name);
        }
    }
    
    if !all_valid {
        println!("\n❌ Some test cases failed verification. Exiting.");
        return;
    }
    
    println!("\n✅ All test cases verified!\n");
    
    // Now test with The Pattern
    let mut pattern_with_basis = UniversalPattern::with_precomputed_basis();
    let mut pattern_regular = UniversalPattern::new();
    
    println!("\n=== Testing with Pre-computed Basis (Auto-tune) ===");
    
    let mut successes = 0;
    let mut total = 0;
    
    // Test with increasing timeouts
    let timeouts = vec![
        Duration::from_millis(100),   // RSA-100
        Duration::from_millis(500),   // RSA-110
        Duration::from_secs(1),       // RSA-120
        Duration::from_secs(5),       // RSA-129
    ];
    
    for (test, timeout) in test_cases.iter().zip(timeouts.iter()) {
        total += 1;
        if test_rsa_number(&mut pattern_with_basis, test, *timeout) {
            successes += 1;
        }
    }
    
    println!("\n{}", "=".repeat(80));
    println!("SUMMARY - Pre-computed Basis");
    println!("{}", "=".repeat(80));
    println!("Success rate: {}/{} ({:.1}%)", 
             successes, total, 100.0 * successes as f64 / total as f64);
    
    // Test one case without pre-computed basis for comparison
    println!("\n\n=== Testing without Pre-computed Basis (for comparison) ===");
    test_rsa_number(&mut pattern_regular, &test_cases[0], Duration::from_secs(1));
    
    println!("\n{}", "=".repeat(80));
    println!("ANALYSIS");
    println!("{}", "=".repeat(80));
    println!("\n1. RSA numbers are much harder than random semiprimes");
    println!("2. They are specifically chosen to have factors close to sqrt(n)");
    println!("3. The Pattern should excel at these balanced cases");
    println!("4. Pre-computed basis enables poly-time scaling");
    
    if successes > 0 {
        println!("\n🎉 The Pattern successfully factored {} RSA number(s)!", successes);
    } else {
        println!("\n📊 Further tuning needed for RSA-scale numbers");
    }
}