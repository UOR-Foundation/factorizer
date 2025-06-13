//! Test Wave Synthesis Auto-Tuner for Arbitrary Semiprimes
//! 
//! This demonstrates the wave synthesis approach where:
//! - Numbers are fingerprinted in universal coordinate space [φ, π, e, unity]
//! - Pre-computed basis acts as an auto-tuner that locks onto factor patterns
//! - 8-bit stream architecture encodes constant activation patterns

use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use std::str::FromStr;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestCase {
    bit_length: usize,
    n: String,
    p: String,
    q: String,
}

fn main() {
    println!("Wave Synthesis Auto-Tuner Test");
    println!("==============================\n");
    
    println!("Concept: Numbers as Waveforms");
    println!("-----------------------------");
    println!("• Each semiprime creates a unique interference pattern");
    println!("• Factors manifest where wave amplitudes align");
    println!("• Pre-computed basis = tuned resonance templates");
    println!("• Auto-tuner scales templates to match input frequency\n");
    
    // Load pre-computed basis (the auto-tuner)
    println!("Loading pre-computed wave basis...");
    let mut pattern = UniversalPattern::with_precomputed_basis();
    
    // Test cases demonstrating wave synthesis
    let test_cases = vec![
        // Small - immediate resonance
        TestCase {
            bit_length: 16,
            n: "41503".to_string(),  // 181 × 229
            p: "181".to_string(),
            q: "229".to_string(),
        },
        // Medium - clear harmonics
        TestCase {
            bit_length: 64,
            n: "13493957735727991339".to_string(),
            p: "3267000013".to_string(),
            q: "4130923723".to_string(),
        },
        // Large - deep resonance
        TestCase {
            bit_length: 128,
            n: "249912617115830524615797691673134648763".to_string(),
            p: "15485863710729486213".to_string(),
            q: "16136667219476902351".to_string(),
        },
        // Beyond old limit - testing arbitrary precision
        TestCase {
            bit_length: 256,
            n: "115792089237316195423570985008687907853269984665640564039457584007913129639703".to_string(),
            p: "340282366920938463463374607431768211503".to_string(),
            q: "340282366920938463463374607431768211401".to_string(),
        },
    ];
    
    println!("\nTesting Wave Synthesis Auto-Tuning:\n");
    
    for (i, test) in test_cases.iter().enumerate() {
        let n = Number::from_str(&test.n).unwrap();
        let expected_p = Number::from_str(&test.p).unwrap();
        let expected_q = Number::from_str(&test.q).unwrap();
        
        println!("Test {} ({}-bit semiprime):", i + 1, test.bit_length);
        println!("n = {}...", &test.n[..20.min(test.n.len())]);
        
        let start = Instant::now();
        
        // Wave synthesis process
        println!("  1. Fingerprinting in universal space [φ, π, e, unity]...");
        let recognition = match pattern.recognize(&n) {
            Ok(r) => {
                println!("     ✓ Pattern recognized (φ: {:.3}, π: {:.3})", 
                    r.phi_component, r.pi_component);
                r
            }
            Err(e) => {
                println!("     ✗ Recognition failed: {}", e);
                continue;
            }
        };
        
        println!("  2. Formalizing wave interference pattern...");
        let formalization = match pattern.formalize(recognition) {
            Ok(f) => {
                println!("     ✓ Resonance peaks detected: {} peaks", 
                    f.resonance_peaks.len());
                f
            }
            Err(e) => {
                println!("     ✗ Formalization failed: {}", e);
                continue;
            }
        };
        
        println!("  3. Auto-tuning basis to input frequency...");
        println!("  4. Executing phase-locked pattern matching...");
        
        match pattern.execute(formalization) {
            Ok(factors) => {
                let elapsed = start.elapsed();
                
                if (factors.p == expected_p && factors.q == expected_q) ||
                   (factors.p == expected_q && factors.q == expected_p) {
                    println!("     ✓ SUCCESS! Factors locked in {:.3}ms", 
                        elapsed.as_secs_f64() * 1000.0);
                    println!("     Method: {}", factors.method);
                    
                    // Verify exact arithmetic
                    let product = &factors.p * &factors.q;
                    if product == n {
                        println!("     ✓ Product verification passed");
                    } else {
                        println!("     ✗ WARNING: Product verification failed!");
                    }
                } else {
                    println!("     ✗ Wrong factors found");
                    println!("     Expected: {} × {}", expected_p, expected_q);
                    println!("     Got:      {} × {}", factors.p, factors.q);
                }
            }
            Err(e) => {
                println!("     ✗ Execution failed: {}", e);
                
                if n.bit_length() > 224 {
                    println!("     Note: This is beyond the old 224-bit limit");
                    println!("     The auto-tuner may need arbitrary precision updates");
                }
            }
        }
        
        println!();
    }
    
    println!("\nWave Synthesis Analysis:");
    println!("------------------------");
    println!("• Small numbers (≤48 bits): Immediate resonance lock");
    println!("• Medium numbers (48-128 bits): Need refined tuning");
    println!("• Large numbers (>224 bits): Require arbitrary precision wave synthesis");
    println!("\nThe auto-tuner must scale the pre-computed basis without precision loss");
    println!("to achieve factorization of arbitrary hard semiprimes through wave synthesis.");
}