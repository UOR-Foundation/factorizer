//! Test different threshold configurations to optimize pattern detection
//!
//! Run with: cargo run --example test_tuning_thresholds

use eight_bit_pattern::{
    recognize_factors, TunerParams, decompose, compute_basis,
    detect_aligned_channels, extract_factors
};
use num_bigint::BigInt;
use std::time::Instant;

fn main() {
    println!("=== Testing Threshold Configurations ===\n");
    
    // Generate test cases
    let test_cases = generate_test_semiprimes();
    println!("Generated {} test semiprimes\n", test_cases.len());
    
    // Test different configurations
    let configs = vec![
        ("Default", TunerParams::default()),
        ("Tight Alignment", TunerParams {
            alignment_threshold: 5,
            ..Default::default()
        }),
        ("Loose Alignment", TunerParams {
            alignment_threshold: 50,
            ..Default::default()
        }),
        ("High Coupling", TunerParams {
            coupling_strength: 0.8,
            ..Default::default()
        }),
        ("Low Coupling", TunerParams {
            coupling_strength: 0.2,
            ..Default::default()
        }),
        ("Conservative", TunerParams {
            alignment_threshold: 10,
            coupling_strength: 0.3,
            position_weight_boost: 1.2,
            ..Default::default()
        }),
        ("Aggressive", TunerParams {
            alignment_threshold: 30,
            coupling_strength: 0.6,
            position_weight_boost: 2.0,
            ..Default::default()
        }),
    ];
    
    for (name, params) in configs {
        println!("=== Configuration: {} ===", name);
        println!("  alignment_threshold: {}", params.alignment_threshold);
        println!("  coupling_strength: {:.1}", params.coupling_strength);
        println!("  position_weight_boost: {:.1}", params.position_weight_boost);
        
        let mut success_count = 0;
        let mut total_time = 0u128;
        let mut failures = Vec::new();
        
        for (n, p, q) in &test_cases {
            let start = Instant::now();
            
            if let Some(factors) = recognize_factors(n, &params) {
                if factors.verify(n) && 
                   ((factors.p == *p && factors.q == *q) || (factors.p == *q && factors.q == *p)) {
                    success_count += 1;
                } else {
                    failures.push((n.clone(), p.clone(), q.clone()));
                }
            } else {
                failures.push((n.clone(), p.clone(), q.clone()));
            }
            
            total_time += start.elapsed().as_micros();
        }
        
        println!("  Success rate: {}/{} ({:.1}%)", 
            success_count, test_cases.len(),
            100.0 * success_count as f64 / test_cases.len() as f64);
        println!("  Average time: {:.1} μs", total_time as f64 / test_cases.len() as f64);
        
        if !failures.is_empty() && failures.len() <= 5 {
            println!("  Failed cases:");
            for (n, p, q) in failures.iter().take(5) {
                println!("    {} = {} × {}", n, p, q);
            }
        }
        
        println!();
    }
    
    // Test with pattern detection order variations
    println!("=== Testing Pattern Detection Order ===\n");
    
    test_detection_order_simple_first(&test_cases);
    test_detection_order_hierarchical_last(&test_cases);
}

fn generate_test_semiprimes() -> Vec<(BigInt, BigInt, BigInt)> {
    let mut semiprimes = Vec::new();
    
    // Small primes for testing
    let primes: Vec<u32> = vec![
        3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
        79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157,
        163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239,
        241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331
    ];
    
    // Generate various types of semiprimes
    for i in 0..primes.len() {
        // Perfect squares
        if i < 10 {
            let p = BigInt::from(primes[i]);
            semiprimes.push((p.clone() * &p, p.clone(), p.clone()));
        }
        
        // Close factors
        if i < primes.len() - 1 {
            let p = BigInt::from(primes[i]);
            let q = BigInt::from(primes[i + 1]);
            semiprimes.push((p.clone() * &q, p, q));
        }
        
        // Unbalanced factors
        if i < primes.len() / 2 {
            let p = BigInt::from(primes[i]);
            let q = BigInt::from(primes[primes.len() - 1 - i]);
            if p < q {
                semiprimes.push((p.clone() * &q, p, q));
            }
        }
    }
    
    semiprimes
}

fn test_detection_order_simple_first(test_cases: &[(BigInt, BigInt, BigInt)]) {
    println!("Testing with simple patterns prioritized:");
    
    let params = TunerParams::default();
    let mut success_count = 0;
    
    for (n, p, q) in test_cases.iter().take(20) { // Test subset
        let channels = decompose(n);
        let basis = compute_basis(n, &params);
        
        // Use standard detection
        let peaks = detect_aligned_channels(n, &basis, &params);
        
        // Try extracting with simple patterns first
        if let Some(factors) = extract_factors(n, &peaks, &channels, &params) {
            if factors.verify(n) && 
               ((factors.p == *p && factors.q == *q) || (factors.p == *q && factors.q == *p)) {
                success_count += 1;
            }
        }
    }
    
    println!("  Success rate: {}/{} ({:.1}%)\n", 
        success_count, 20,
        100.0 * success_count as f64 / 20.0);
}

fn test_detection_order_hierarchical_last(test_cases: &[(BigInt, BigInt, BigInt)]) {
    println!("Testing with hierarchical patterns as fallback:");
    
    let params = TunerParams {
        alignment_threshold: 15,
        coupling_strength: 0.4,
        position_weight_boost: 1.5,
        ..Default::default()
    };
    
    let mut success_count = 0;
    
    for (n, p, q) in test_cases.iter().take(20) { // Test subset
        if let Some(factors) = recognize_factors(n, &params) {
            if factors.verify(n) && 
               ((factors.p == *p && factors.q == *q) || (factors.p == *q && factors.q == *p)) {
                success_count += 1;
            }
        }
    }
    
    println!("  Success rate: {}/{} ({:.1}%)\n", 
        success_count, 20,
        100.0 * success_count as f64 / 20.0);
}