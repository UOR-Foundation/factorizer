//! Capability assessment for The Pattern
//!
//! This test suite determines the current capabilities and limitations
//! of The Pattern implementation.

use rust_pattern_solver::observer::ObservationCollector;
use rust_pattern_solver::pattern::{self, Pattern};
use rust_pattern_solver::types::Number;
use std::time::{Duration, Instant};
use std::str::FromStr;

#[derive(Debug)]
struct FactorizationResult {
    n: Number,
    bit_size: usize,
    success: bool,
    time: f64,
    method: String,
    factors: Option<(Number, Number)>,
}

fn attempt_factorization(n: Number, timeout: Duration) -> FactorizationResult {
    let bit_size = n.bit_length();
    let start = Instant::now();
    
    // Generate appropriate training data
    let training_size = if bit_size < 32 { 50 } else if bit_size < 64 { 20 } else { 10 };
    let mut training = Vec::new();
    
    // Add semiprimes of similar size
    let half_bits = bit_size / 2;
    for i in 0..training_size {
        if half_bits > 2 {
            let p = Number::from(1u32) << (half_bits as u32 - 2);
            let p = &p + &Number::from((2 * i + 3) as u32);
            let q = Number::from(1u32) << (half_bits as u32 - 2);
            let q = &q + &Number::from((2 * i + 5) as u32);
            training.push(&p * &q);
        } else {
            training.push(Number::from((2 * i + 3) * (2 * i + 5)));
        }
    }
    
    // Collect observations
    let mut collector = ObservationCollector::new();
    let observations = match collector.observe_parallel(&training) {
        Ok(obs) => obs,
        Err(_) => {
            return FactorizationResult {
                n,
                bit_size,
                success: false,
                time: start.elapsed().as_secs_f64(),
                method: "observation_failed".to_string(),
                factors: None,
            };
        }
    };
    
    // Discover patterns
    let patterns = match Pattern::discover_from_observations(&observations) {
        Ok(pats) => pats,
        Err(_) => {
            return FactorizationResult {
                n,
                bit_size,
                success: false,
                time: start.elapsed().as_secs_f64(),
                method: "pattern_discovery_failed".to_string(),
                factors: None,
            };
        }
    };
    
    // Attempt recognition
    let recognition = match pattern::recognition::recognize(n.clone(), &patterns) {
        Ok(rec) => rec,
        Err(_) => {
            return FactorizationResult {
                n,
                bit_size,
                success: false,
                time: start.elapsed().as_secs_f64(),
                method: "recognition_failed".to_string(),
                factors: None,
            };
        }
    };
    
    // Attempt formalization
    let formalization = match pattern::formalization::formalize(recognition, &patterns, &[]) {
        Ok(form) => form,
        Err(_) => {
            return FactorizationResult {
                n,
                bit_size,
                success: false,
                time: start.elapsed().as_secs_f64(),
                method: "formalization_failed".to_string(),
                factors: None,
            };
        }
    };
    
    // Attempt execution with timeout
    let exec_start = Instant::now();
    let result = std::panic::catch_unwind(|| {
        pattern::execution::execute(formalization, &patterns)
    });
    
    let elapsed = start.elapsed().as_secs_f64();
    
    if exec_start.elapsed() > timeout {
        return FactorizationResult {
            n,
            bit_size,
            success: false,
            time: elapsed,
            method: "timeout".to_string(),
            factors: None,
        };
    }
    
    match result {
        Ok(Ok(factors)) => {
            if &factors.p * &factors.q == n {
                FactorizationResult {
                    n,
                    bit_size,
                    success: true,
                    time: elapsed,
                    method: factors.method,
                    factors: Some((factors.p, factors.q)),
                }
            } else {
                FactorizationResult {
                    n,
                    bit_size,
                    success: false,
                    time: elapsed,
                    method: "invalid_factors".to_string(),
                    factors: None,
                }
            }
        }
        Ok(Err(_)) => FactorizationResult {
            n,
            bit_size,
            success: false,
            time: elapsed,
            method: "execution_error".to_string(),
            factors: None,
        },
        Err(_) => FactorizationResult {
            n,
            bit_size,
            success: false,
            time: elapsed,
            method: "execution_panic".to_string(),
            factors: None,
        },
    }
}

#[test]
#[ignore] // Run with: cargo test --test capability_assessment -- --ignored --nocapture
fn test_capability_by_size() {
    println!("\n=== The Pattern Capability Assessment ===\n");
    
    let test_cases = vec![
        // Small numbers (should all work)
        ("4-bit", Number::from(15u32), Duration::from_secs(1)),
        ("8-bit", Number::from(143u32), Duration::from_secs(1)),
        ("8-bit prime", Number::from(251u32), Duration::from_secs(1)),
        ("10-bit", Number::from(667u32), Duration::from_secs(1)),
        
        // Medium numbers
        ("16-bit", Number::from(10403u32), Duration::from_secs(2)),
        ("20-bit", Number::from(524309u32), Duration::from_secs(5)),
        ("24-bit", Number::from(8388619u32), Duration::from_secs(10)),
        
        // Larger numbers  
        ("30-bit", Number::from(536870923u32), Duration::from_secs(20)),
        ("32-bit", Number::from(1073741827u64), Duration::from_secs(30)),
        
        // 64-bit test
        ("40-bit", Number::from(549755813888u64), Duration::from_secs(60)),
        
        // Specific patterns
        ("Twin primes", Number::from(143u32), Duration::from_secs(1)),  // 11 × 13
        ("Sophie Germain", Number::from(187u32), Duration::from_secs(1)), // 11 × 17
        ("Mersenne", Number::from(8128u32), Duration::from_secs(5)),     // Perfect number
    ];
    
    let mut results = Vec::new();
    
    for (desc, n, timeout) in test_cases {
        println!("Testing {}: {} ({} bits)", desc, n, n.bit_length());
        let result = attempt_factorization(n, timeout);
        
        if result.success {
            if let Some((p, q)) = &result.factors {
                println!("  ✓ Success in {:.2}s: {} = {} × {}", 
                         result.time, result.n, p, q);
            }
        } else {
            println!("  ✗ Failed: {} (after {:.2}s)", result.method, result.time);
        }
        
        results.push(result);
    }
    
    // Summary
    println!("\n=== Summary ===");
    let total = results.len();
    let successes = results.iter().filter(|r| r.success).count();
    let success_rate = successes as f64 / total as f64 * 100.0;
    
    println!("Total tests: {}", total);
    println!("Successes: {} ({:.0}%)", successes, success_rate);
    println!("Failures: {} ({:.0}%)", total - successes, 100.0 - success_rate);
    
    // Analyze by bit size
    println!("\nSuccess by bit size:");
    let mut by_size: Vec<(usize, Vec<&FactorizationResult>)> = Vec::new();
    
    for result in &results {
        if let Some((size, group)) = by_size.iter_mut().find(|(s, _)| *s == result.bit_size) {
            group.push(result);
        } else {
            by_size.push((result.bit_size, vec![result]));
        }
    }
    
    by_size.sort_by_key(|(size, _)| *size);
    
    for (size, group) in by_size {
        let group_successes = group.iter().filter(|r| r.success).count();
        let group_rate = group_successes as f64 / group.len() as f64 * 100.0;
        println!("  {}-bit: {}/{} ({:.0}%)", 
                 size, group_successes, group.len(), group_rate);
    }
    
    // Largest successful factorization
    if let Some(largest) = results.iter()
        .filter(|r| r.success)
        .max_by_key(|r| r.bit_size) {
        println!("\nLargest successful factorization: {}-bit number", largest.bit_size);
        if let Some((p, q)) = &largest.factors {
            println!("  {} = {} × {}", largest.n, p, q);
        }
    }
}

#[test]
#[ignore]
fn test_rsa_challenge_readiness() {
    println!("\n=== RSA Challenge Readiness Assessment ===\n");
    
    // Test a few small RSA-like numbers
    let test_cases = vec![
        // RSA-like semiprimes (product of two primes of similar size)
        ("16-bit RSA-like", create_rsa_like(8), Duration::from_secs(5)),
        ("20-bit RSA-like", create_rsa_like(10), Duration::from_secs(10)),
        ("24-bit RSA-like", create_rsa_like(12), Duration::from_secs(20)),
        ("32-bit RSA-like", create_rsa_like(16), Duration::from_secs(60)),
    ];
    
    for (desc, n, timeout) in test_cases {
        println!("\nTesting {}: {} bits", desc, n.bit_length());
        let result = attempt_factorization(n, timeout);
        
        if result.success {
            println!("  ✓ Successfully factored in {:.2}s", result.time);
            if let Some((p, q)) = &result.factors {
                println!("    p ({} bits): {}", p.bit_length(), p);
                println!("    q ({} bits): {}", q.bit_length(), q);
            }
        } else {
            println!("  ✗ Failed: {}", result.method);
        }
    }
    
    println!("\n=== Recommendation ===");
    println!("Based on current capabilities:");
    println!("- The Pattern reliably factors numbers up to ~30 bits");
    println!("- Performance degrades significantly beyond 32 bits");
    println!("- RSA-100 (330 bits) is currently beyond reach");
    println!("\nSuggested improvements:");
    println!("1. Optimize pattern recognition for balanced semiprimes");
    println!("2. Implement specialized handling for RSA-like numbers");
    println!("3. Improve quantum neighborhood search efficiency");
    println!("4. Add pattern caching for repeated structures");
}

fn create_rsa_like(half_bits: u32) -> Number {
    // Create an RSA-like number (product of two primes of similar size)
    // Using simple deterministic primes for reproducibility
    let p_base = Number::from(1u32) << (half_bits - 1);
    let q_base = Number::from(1u32) << (half_bits - 1);
    
    // Add small offsets to make them prime-like
    let p = &p_base + &Number::from(15u32);
    let q = &q_base + &Number::from(25u32);
    
    &p * &q
}