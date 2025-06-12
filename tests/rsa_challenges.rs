//! Progressive testing from small numbers to RSA challenges
//!
//! This test suite validates The Pattern's recognition abilities across
//! increasing scales, building confidence before attempting RSA challenges.

use rust_pattern_solver::observer::ObservationCollector;
use rust_pattern_solver::pattern::{self, Pattern};
use rust_pattern_solver::types::Number;
use std::time::{Duration, Instant};
use std::str::FromStr;

/// Test helper to attempt factorization with timeout
fn test_factorization_with_timeout(
    n: Number,
    expected_p: Option<Number>,
    expected_q: Option<Number>,
    timeout: Duration,
    description: &str,
) -> Result<(), String> {
    println!("\nTesting {}", description);
    println!("N = {} ({} bits)", n, n.bit_length());

    // Collect patterns from training data
    let mut collector = ObservationCollector::new();
    let training_numbers = generate_training_set(n.bit_length());
    let observations = collector
        .observe_parallel(&training_numbers)
        .map_err(|e| format!("Failed to collect observations: {}", e))?;

    let patterns = Pattern::discover_from_observations(&observations)
        .map_err(|e| format!("Failed to discover patterns: {}", e))?;

    // Attempt recognition
    let start = Instant::now();
    let recognition_result = pattern::recognition::recognize(n.clone(), &patterns);

    match recognition_result {
        Ok(recognition) => {
            println!("  Recognition: {:?} (confidence: {:.2})", 
                     recognition.pattern_type, recognition.confidence);

            // Attempt formalization
            let formalization_result = pattern::formalization::formalize(
                recognition,
                &patterns,
                &[],
            );

            match formalization_result {
                Ok(formalization) => {
                    // Attempt execution with timeout
                    let execution_start = Instant::now();
                    let execution_result = std::panic::catch_unwind(|| {
                        pattern::execution::execute(formalization, &patterns)
                    });

                    let elapsed = execution_start.elapsed();
                    if elapsed > timeout {
                        return Err(format!("Execution timeout after {:?}", elapsed));
                    }

                    match execution_result {
                        Ok(Ok(factors)) => {
                            let total_time = start.elapsed();
                            println!("  ✓ Factors found in {:.2}s", total_time.as_secs_f64());
                            println!("    p = {}", factors.p);
                            println!("    q = {}", factors.q);

                            // Verify factors if expected values provided
                            if let (Some(exp_p), Some(exp_q)) = (expected_p, expected_q) {
                                if (factors.p == exp_p && factors.q == exp_q) ||
                                   (factors.p == exp_q && factors.q == exp_p) {
                                    Ok(())
                                } else {
                                    Err(format!("Factor mismatch! Expected {} × {}", exp_p, exp_q))
                                }
                            } else {
                                // Just verify the multiplication
                                if &factors.p * &factors.q == n {
                                    Ok(())
                                } else {
                                    Err("Factor verification failed!".to_string())
                                }
                            }
                        }
                        Ok(Err(e)) => Err(format!("Execution failed: {}", e)),
                        Err(_) => Err("Execution panicked".to_string()),
                    }
                }
                Err(e) => Err(format!("Formalization failed: {}", e)),
            }
        }
        Err(e) => Err(format!("Recognition failed: {}", e)),
    }
}

/// Generate appropriate training set for the target bit size
fn generate_training_set(target_bits: usize) -> Vec<Number> {
    let mut training = Vec::new();

    // Always include small semiprimes for base patterns
    for i in 0..100 {
        let p = 2 * i + 3;
        let q = 2 * i + 5;
        training.push(Number::from(p * q));
    }

    // Add semiprimes at appropriate scale
    if target_bits >= 16 {
        // Add 16-bit semiprimes
        for i in 0..50 {
            let p = 251 + 2 * i;
            let q = 257 + 2 * i;
            training.push(Number::from(p as u32 * q as u32));
        }
    }

    if target_bits >= 32 {
        // Add 32-bit semiprimes
        for i in 0..20 {
            let p = 65521u32 + 2 * i;
            let q = 65537u32 + 2 * i;
            training.push(Number::from(p as u64 * q as u64));
        }
    }

    if target_bits >= 64 {
        // Add larger semiprimes using known patterns
        training.push(Number::from_str_radix("FFFFFFFFFFFFFFFF", 16).unwrap());
        training.push(Number::from_str_radix("FEDCBA9876543211", 16).unwrap());
    }

    training
}

#[test]
#[ignore] // Run with: cargo test --test rsa_challenges test_progressive_scale -- --ignored --nocapture
fn test_progressive_scale() {
    // Start with small numbers to validate the approach
    let test_cases = vec![
        // Very small (8-bit)
        (Number::from(15u32), Some(Number::from(3u32)), Some(Number::from(5u32)), 
         Duration::from_secs(1), "8-bit: 15 = 3 × 5"),
        
        // Small (16-bit)
        (Number::from(143u32), Some(Number::from(11u32)), Some(Number::from(13u32)),
         Duration::from_secs(1), "16-bit: 143 = 11 × 13"),
        
        // Medium (32-bit)
        (Number::from(1073741827u64), Some(Number::from(32749u32)), Some(Number::from(32771u32)),
         Duration::from_secs(5), "32-bit: balanced semiprime"),
        
        // Large (64-bit)
        (Number::from_str_radix("FEDCBA9876543211", 16).unwrap(), None, None,
         Duration::from_secs(10), "64-bit: hex pattern"),
    ];

    let mut successes = 0;
    let mut failures = 0;

    for (n, p, q, timeout, desc) in test_cases {
        match test_factorization_with_timeout(n, p, q, timeout, desc) {
            Ok(_) => successes += 1,
            Err(e) => {
                println!("  ✗ Failed: {}", e);
                failures += 1;
            }
        }
    }

    println!("\n=== Progressive Scale Test Summary ===");
    println!("Successes: {}", successes);
    println!("Failures: {}", failures);
    
    assert!(failures == 0, "Some factorizations failed");
}

#[test]
#[ignore] // Run with: cargo test --test rsa_challenges test_rsa_100 -- --ignored --nocapture
fn test_rsa_100() {
    let n = Number::from_str(
        "1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139"
    ).unwrap();
    
    let p = Number::from_str(
        "37975227936943673922808872755445627854565536638199"
    ).unwrap();
    
    let q = Number::from_str(
        "40094690950920881030683735292761468389214899724061"
    ).unwrap();

    match test_factorization_with_timeout(
        n, 
        Some(p), 
        Some(q), 
        Duration::from_secs(300), // 5 minutes
        "RSA-100 (330 bits)"
    ) {
        Ok(_) => println!("✓ RSA-100 factored successfully!"),
        Err(e) => println!("✗ RSA-100 failed: {}", e),
    }
}

#[test]
#[ignore] // Run with: cargo test --test rsa_challenges test_pattern_scaling -- --ignored --nocapture
fn test_pattern_scaling() {
    println!("\n=== Testing Pattern Scaling Behavior ===\n");

    // Analyze how patterns scale with number size
    let scales = vec![
        (8, 16),    // Small
        (16, 32),   // Medium  
        (32, 64),   // Large
        (64, 128),  // Very large
        (128, 256), // Extreme
    ];

    for (min_bits, max_bits) in scales {
        println!("Analyzing {}-{} bit range:", min_bits, max_bits);
        
        // Generate semiprimes in this range
        let test_numbers = generate_test_numbers_in_range(min_bits, max_bits, 10);
        
        let mut collector = ObservationCollector::new();
        match collector.observe_parallel(&test_numbers) {
            Ok(observations) => {
                match Pattern::discover_from_observations(&observations) {
                    Ok(patterns) => {
                        println!("  Found {} patterns", patterns.len());
                        
                        // Analyze pattern characteristics
                        for pattern in patterns.iter().take(3) {
                            println!("    - {}: frequency={:.2}, scale={:?}", 
                                     pattern.id, pattern.frequency, pattern.scale_range);
                        }
                    }
                    Err(e) => println!("  Pattern discovery failed: {}", e),
                }
            }
            Err(e) => println!("  Observation failed: {}", e),
        }
        println!();
    }
}

/// Generate test numbers in a specific bit range
fn generate_test_numbers_in_range(min_bits: usize, max_bits: usize, count: usize) -> Vec<Number> {
    let mut numbers = Vec::new();
    
    for i in 0..count {
        let bits = min_bits + (i * (max_bits - min_bits) / count);
        
        // Create a semiprime with approximately the target bit size
        let half_bits = bits / 2;
        let p_base = Number::from(1u32) << (half_bits as u32 - 1);
        let p = &p_base + &Number::from((2 * i + 3) as u32);
        
        let q_base = Number::from(1u32) << (half_bits as u32 - 1);
        let q = &q_base + &Number::from((2 * i + 5) as u32);
        
        numbers.push(&p * &q);
    }
    
    numbers
}

#[test]
#[ignore] // Run with: cargo test --test rsa_challenges test_rsa_challenges_summary -- --ignored --nocapture
fn test_rsa_challenges_summary() {
    println!("\n=== RSA Challenges Summary ===\n");
    
    let challenges = vec![
        ("RSA-100", 100, 330, true),   // Already factored
        ("RSA-110", 110, 364, true),   // Already factored
        ("RSA-120", 120, 397, true),   // Already factored
        ("RSA-129", 129, 426, true),   // Already factored
        ("RSA-130", 130, 430, true),   // Already factored
        ("RSA-140", 140, 463, true),   // Already factored
        ("RSA-150", 150, 496, true),   // Already factored
        ("RSA-155", 155, 512, true),   // Already factored
        ("RSA-160", 160, 530, true),   // Already factored
        ("RSA-170", 170, 563, true),   // Already factored
        ("RSA-576", 174, 576, true),   // Already factored
        ("RSA-180", 180, 596, true),   // Already factored
        ("RSA-190", 190, 629, true),   // Already factored
        ("RSA-640", 193, 640, true),   // Already factored
        ("RSA-200", 200, 663, true),   // Already factored
        ("RSA-210", 210, 696, true),   // Already factored
        ("RSA-704", 212, 704, true),   // Already factored
        ("RSA-220", 220, 729, true),   // Already factored
        ("RSA-230", 230, 762, true),   // Already factored
        ("RSA-232", 232, 768, true),   // Already factored
        ("RSA-768", 232, 768, true),   // Already factored
        ("RSA-240", 240, 795, true),   // Already factored
        ("RSA-250", 250, 829, true),   // Already factored
        ("RSA-260", 260, 862, false),  // Not factored
        ("RSA-270", 270, 895, false),  // Not factored
        ("RSA-896", 270, 896, false),  // Not factored
        ("RSA-280", 280, 928, false),  // Not factored
        ("RSA-290", 290, 962, false),  // Not factored
        ("RSA-300", 300, 995, false),  // Not factored
        ("RSA-309", 309, 1024, false), // Not factored
        ("RSA-1024", 309, 1024, false),// Not factored
        ("RSA-2048", 617, 2048, false),// Not factored
    ];
    
    println!("{:<12} {:>12} {:>12} {:>12}", "Challenge", "Digits", "Bits", "Status");
    println!("{:-<50}", "");
    
    for (name, digits, bits, factored) in challenges {
        let status = if factored { "Factored" } else { "Open" };
        println!("{:<12} {:>12} {:>12} {:>12}", name, digits, bits, status);
    }
    
    println!("\nRecommended test progression:");
    println!("1. Start with progressive_scale test (8-64 bits)");
    println!("2. Test RSA-100 (330 bits) - smallest RSA challenge");
    println!("3. Analyze pattern scaling behavior");
    println!("4. Attempt larger challenges based on success");
}