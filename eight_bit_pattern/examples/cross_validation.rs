//! Cross-validation suite with unseen semiprimes
//!
//! Run with: cargo run --example cross_validation

use eight_bit_pattern::{
    TunerParams, compute_basis, recognize_factors,
    EnsembleVoter, VotingStrategy, TestCase
};
use num_bigint::BigInt;
use num_traits::{One, Zero};
use std::collections::HashSet;
use std::time::Instant;

fn main() {
    println!("=== Cross-Validation Suite for The 8-Bit Pattern ===\n");
    
    // Generate unseen semiprimes not in the original test matrix
    let unseen_cases = generate_unseen_semiprimes();
    println!("Generated {} unseen semiprimes for validation\n", unseen_cases.len());
    
    let params = TunerParams::default();
    
    // Test different approaches
    println!("=== Method Performance on Unseen Data ===\n");
    
    // 1. Standard pattern recognition
    test_method(
        "Standard Pattern Recognition",
        &unseen_cases,
        |n| recognize_factors(n, &params),
    );
    
    // 2. Advanced resonance extraction
    test_method(
        "Advanced Resonance Extraction",
        &unseen_cases,
        |n| {
            let basis = compute_basis(n, &params);
            eight_bit_pattern::recognize_factors_advanced(n, &basis, &params)
        },
    );
    
    // 3. Ensemble voting
    let ensemble = EnsembleVoter::new(VotingStrategy::AdaptiveBySize);
    test_method(
        "Ensemble Voting (Adaptive)",
        &unseen_cases,
        |n| ensemble.recognize_factors(n, &params),
    );
    
    // 4. Special cases only
    test_method(
        "Special Cases Only",
        &unseen_cases,
        |n| eight_bit_pattern::try_special_cases(n),
    );
    
    // Detailed analysis by category
    println!("\n=== Performance by Semiprime Category ===\n");
    
    analyze_by_category(&unseen_cases, &basis, &params);
    
    // Failure analysis
    println!("\n=== Failure Analysis ===\n");
    
    analyze_failures(&unseen_cases, &basis, &params);
    
    // Generalization assessment
    println!("\n=== Generalization Assessment ===\n");
    
    assess_generalization(&unseen_cases, &basis, &params);
}

/// Generate unseen semiprimes not typically in test sets
fn generate_unseen_semiprimes() -> Vec<TestCase> {
    let mut cases = Vec::new();
    let mut seen = HashSet::new();
    
    // Small primes for combinations
    let small_primes = vec![
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
        53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    ];
    
    // Generate combinations avoiding common test cases
    for i in 2..small_primes.len() {
        for j in i..small_primes.len() {
            let p = BigInt::from(small_primes[i]);
            let q = BigInt::from(small_primes[j]);
            let n = &p * &q;
            
            // Skip if this is a common test case
            if n == BigInt::from(15) || n == BigInt::from(35) || 
               n == BigInt::from(143) || n == BigInt::from(323) {
                continue;
            }
            
            if !seen.contains(&n) && n.bits() <= 20 {
                seen.insert(n.clone());
                cases.push(TestCase {
                    n: n.clone(),
                    p: p.clone(),
                    q: q.clone(),
                    bit_length: n.bits() as usize,
                });
            }
        }
    }
    
    // Add some medium-sized unseen cases
    let medium_primes = vec![
        101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
        151, 157, 163, 167, 173, 179, 181, 191, 193, 197
    ];
    
    for i in 0..5 {
        for j in i+1..6 {
            let p = BigInt::from(medium_primes[i]);
            let q = BigInt::from(medium_primes[j]);
            let n = &p * &q;
            
            cases.push(TestCase {
                n: n.clone(),
                p: p.clone(),
                q: q.clone(),
                bit_length: n.bits() as usize,
            });
        }
    }
    
    // Add some special structure semiprimes
    // Mersenne-like: 2^k - 1
    for k in vec![5, 7, 11, 13] {
        let p = (BigInt::one() << k) - 1;
        let q = (BigInt::one() << (k + 2)) - 1;
        let n = &p * &q;
        
        if is_prime(&p) && is_prime(&q) {
            cases.push(TestCase {
                n: n.clone(),
                p: p.clone(),
                q: q.clone(),
                bit_length: n.bits() as usize,
            });
        }
    }
    
    // Sort by bit length
    cases.sort_by_key(|c| c.bit_length);
    cases
}

/// Simple primality test (for small numbers)
fn is_prime(n: &BigInt) -> bool {
    if n <= &BigInt::one() {
        return false;
    }
    if n == &BigInt::from(2) {
        return true;
    }
    if n % 2 == BigInt::zero() {
        return false;
    }
    
    let sqrt_n = n.sqrt();
    let mut d = BigInt::from(3);
    while d <= sqrt_n {
        if n % &d == BigInt::zero() {
            return false;
        }
        d += 2;
    }
    true
}

/// Test a factorization method
fn test_method<F>(name: &str, cases: &[TestCase], method: F)
where
    F: Fn(&BigInt) -> Option<eight_bit_pattern::Factors>,
{
    println!("Testing {}:", name);
    
    let mut successes = 0;
    let mut total_time = 0u128;
    let mut bit_success: std::collections::HashMap<usize, (usize, usize)> = 
        std::collections::HashMap::new();
    
    for case in cases {
        let start = Instant::now();
        let result = method(&case.n);
        let elapsed = start.elapsed().as_micros();
        total_time += elapsed;
        
        if let Some(factors) = result {
            if factors.verify(&case.n) {
                successes += 1;
                let entry = bit_success.entry(case.bit_length).or_insert((0, 0));
                entry.0 += 1;
            }
        }
        
        let entry = bit_success.entry(case.bit_length).or_insert((0, 0));
        entry.1 += 1;
    }
    
    println!("  Overall success: {}/{} ({:.1}%)", 
        successes, cases.len(), 
        successes as f64 / cases.len() as f64 * 100.0);
    println!("  Average time: {:.1} μs", total_time as f64 / cases.len() as f64);
    
    // Show success by bit size
    let mut bit_sizes: Vec<_> = bit_success.keys().cloned().collect();
    bit_sizes.sort();
    
    println!("  Success by bit size:");
    for size in bit_sizes {
        let (succ, total) = bit_success[&size];
        println!("    {}-bit: {}/{} ({:.1}%)", 
            size, succ, total, succ as f64 / total as f64 * 100.0);
    }
    println!();
}

/// Analyze performance by semiprime category
fn analyze_by_category(
    cases: &[TestCase],
    basis: &eight_bit_pattern::Basis,
    params: &TunerParams,
) {
    // Categorize semiprimes
    let mut categories: std::collections::HashMap<String, Vec<&TestCase>> = 
        std::collections::HashMap::new();
    
    for case in cases {
        // Check special properties
        let special_cases = eight_bit_pattern::detect_special_cases(&case.n);
        
        if !special_cases.is_empty() {
            for sc in special_cases {
                let category = format!("{:?}", sc.case_type);
                categories.entry(category).or_default().push(case);
            }
        } else {
            // Categorize by factor relationship
            let diff = if case.p > case.q {
                &case.p - &case.q
            } else {
                &case.q - &case.p
            };
            
            let category = if diff < BigInt::from(10) {
                "Close factors".to_string()
            } else if &case.p * 2 > case.q && &case.q * 2 > case.p {
                "Balanced factors".to_string()
            } else {
                "Unbalanced factors".to_string()
            };
            
            categories.entry(category).or_default().push(case);
        }
    }
    
    // Test each category
    for (category, cat_cases) in categories {
        println!("Category: {} ({} cases)", category, cat_cases.len());
        
        let mut successes = 0;
        for case in &cat_cases {
            if let Some(factors) = recognize_factors(&case.n, basis, params) {
                if factors.verify(&case.n) {
                    successes += 1;
                }
            }
        }
        
        println!("  Success rate: {}/{} ({:.1}%)\n", 
            successes, cat_cases.len(),
            successes as f64 / cat_cases.len() as f64 * 100.0);
    }
}

/// Analyze failure patterns
fn analyze_failures(
    cases: &[TestCase],
    basis: &eight_bit_pattern::Basis,
    params: &TunerParams,
) {
    let mut failures = Vec::new();
    
    for case in cases {
        let result = recognize_factors(&case.n, basis, params);
        
        if result.is_none() || !result.as_ref().unwrap().verify(&case.n) {
            failures.push(case);
        }
    }
    
    if failures.is_empty() {
        println!("No failures detected!");
        return;
    }
    
    println!("Failed on {} cases:", failures.len());
    
    // Analyze failure patterns
    let mut failure_patterns = std::collections::HashMap::new();
    
    for case in &failures {
        // Check bit patterns
        let channels = eight_bit_pattern::decompose(&case.n);
        let pattern_hash = channels.iter().take(4).fold(0u32, |acc, &ch| {
            (acc << 8) | ch as u32
        });
        
        *failure_patterns.entry(pattern_hash).or_insert(0) += 1;
    }
    
    // Show common failure patterns
    let mut patterns: Vec<_> = failure_patterns.into_iter().collect();
    patterns.sort_by(|a, b| b.1.cmp(&a.1));
    
    println!("\nMost common failure patterns:");
    for (pattern, count) in patterns.iter().take(5) {
        println!("  Pattern 0x{:08X}: {} failures", pattern, count);
    }
    
    // Show example failures
    println!("\nExample failures:");
    for case in failures.iter().take(5) {
        println!("  {} = {} × {} ({} bits)", 
            case.n, case.p, case.q, case.bit_length);
    }
}

/// Assess generalization capability
fn assess_generalization(
    cases: &[TestCase],
    basis: &eight_bit_pattern::Basis,
    params: &TunerParams,
) {
    // Test with different parameter variations
    let param_variations = vec![
        ("Default", TunerParams::default()),
        ("Tight alignment", TunerParams {
            alignment_threshold: 2,
            ..TunerParams::default()
        }),
        ("Loose alignment", TunerParams {
            alignment_threshold: 5,
            ..TunerParams::default()
        }),
        ("High coupling", TunerParams {
            phase_coupling_strength: 7,
            ..TunerParams::default()
        }),
    ];
    
    println!("Parameter sensitivity analysis:");
    
    for (name, test_params) in param_variations {
        let mut successes = 0;
        
        for case in cases {
            if let Some(factors) = recognize_factors(&case.n, basis, &test_params) {
                if factors.verify(&case.n) {
                    successes += 1;
                }
            }
        }
        
        println!("  {}: {:.1}% success", 
            name, successes as f64 / cases.len() as f64 * 100.0);
    }
    
    // Statistical summary
    println!("\nGeneralization Summary:");
    
    let standard_success = cases.iter()
        .filter(|case| {
            recognize_factors(&case.n, basis, params)
                .map(|f| f.verify(&case.n))
                .unwrap_or(false)
        })
        .count();
    
    let success_rate = standard_success as f64 / cases.len() as f64;
    
    if success_rate > 0.9 {
        println!("✓ EXCELLENT: The Pattern generalizes well to unseen data ({:.1}%)", 
            success_rate * 100.0);
    } else if success_rate > 0.7 {
        println!("◐ GOOD: Reasonable generalization with room for improvement ({:.1}%)",
            success_rate * 100.0);
    } else if success_rate > 0.5 {
        println!("△ MODERATE: Limited generalization capability ({:.1}%)",
            success_rate * 100.0);
    } else {
        println!("✗ POOR: The Pattern struggles with unseen data ({:.1}%)",
            success_rate * 100.0);
    }
    
    // Recommendations
    println!("\nRecommendations for improvement:");
    if success_rate < 0.9 {
        println!("  - Expand training data to include more diverse semiprimes");
        println!("  - Implement adaptive constant tuning based on number characteristics");
        println!("  - Consider hierarchical pattern recognition for different scales");
    }
}