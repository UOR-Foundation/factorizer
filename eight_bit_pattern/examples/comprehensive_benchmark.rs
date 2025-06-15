//! Comprehensive benchmark and test report for The 8-Bit Pattern
//!
//! Run with: cargo run --example comprehensive_benchmark --release

use eight_bit_pattern::{
    TunerParams, TestCase, Basis, compute_basis, 
    recognize_factors, recognize_factors_advanced,
    recognize_factors_with_diagnostics, DiagnosticAggregator
};
use num_bigint::BigInt;
use std::fs;
use std::path::Path;
use std::time::Instant;
use std::collections::HashMap;

fn main() {
    println!("=== The 8-Bit Pattern - Comprehensive Benchmark Report ===\n");
    println!("Date: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"));
    println!("Implementation: eight_bit_pattern v0.1.0");
    println!();
    
    // Load test cases
    let test_cases = load_test_cases();
    println!("Loaded {} test cases from test matrix", test_cases.len());
    
    // Group by bit size
    let mut cases_by_size: HashMap<usize, Vec<&TestCase>> = HashMap::new();
    for case in &test_cases {
        cases_by_size.entry(case.bit_length).or_default().push(case);
    }
    
    // Create basis with representative large number for maximum coverage
    let params = TunerParams::default();
    let representative_n = BigInt::from(1u128) << 1024; // 1024-bit number
    let basis = compute_basis(&representative_n, &params);
    println!("Created basis with {} channels\n", basis.num_channels);
    
    // Test different approaches
    println!("=== Method Comparison ===\n");
    
    let methods: Vec<(&str, fn(&[TestCase], &Basis, &TunerParams) -> (f64, f64))> = vec![
        ("Standard Pattern Recognition", test_standard_method),
        ("Advanced Resonance Extraction", test_advanced_method),
        ("Hybrid Approach", test_hybrid_method),
    ];
    
    for (name, method) in &methods {
        println!("Testing {}...", name);
        let (success_rate, avg_time) = method(&test_cases, &basis, &params);
        println!("  Success rate: {:.1}%", success_rate * 100.0);
        println!("  Average time: {:.1} μs\n", avg_time);
    }
    
    // Detailed analysis by bit size
    println!("=== Performance by Bit Size ===\n");
    
    let mut sizes: Vec<_> = cases_by_size.keys().copied().collect();
    sizes.sort();
    
    for size in &sizes {
        let cases = &cases_by_size[size];
        let results = test_bit_size(cases, &basis, &params);
        
        println!("{}-bit numbers ({} cases):", size, cases.len());
        println!("  Standard method: {:.1}% success, {:.1} μs avg",
            results.standard_success * 100.0, results.standard_time);
        println!("  Advanced method: {:.1}% success, {:.1} μs avg",
            results.advanced_success * 100.0, results.advanced_time);
        println!("  Best success: {:.1}%", results.best_success * 100.0);
        
        if results.best_success > 0.0 {
            println!("  Example factors found:");
            for (n, p, q) in results.example_factors.iter().take(3) {
                println!("    {} = {} × {}", n, p, q);
            }
        }
        println!();
    }
    
    // Pattern analysis
    println!("=== Pattern Analysis ===\n");
    
    let mut aggregator = DiagnosticAggregator::default();
    let mut pattern_success: HashMap<u8, (usize, usize)> = HashMap::new(); // (total, successful)
    
    for test_case in test_cases.iter().take(50) { // Analyze first 50 for efficiency
        let (result, diagnostics) = recognize_factors_with_diagnostics(
            &test_case.n, &params
        );
        
        aggregator.add(&diagnostics);
        
        // Track pattern success
        for peak in &diagnostics.peaks {
            let entry = pattern_success.entry(peak.aligned_pattern).or_default();
            entry.0 += 1;
            if result.is_some() {
                entry.1 += 1;
            }
        }
    }
    
    // Show most successful patterns
    println!("Most successful patterns:");
    let mut patterns: Vec<_> = pattern_success.iter()
        .filter(|(_, (total, _))| *total > 0)
        .map(|(p, (t, s))| (*p, *s as f64 / *t as f64, *t))
        .collect();
    patterns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    for (pattern, success_rate, count) in patterns.iter().take(10) {
        if *success_rate > 0.0 {
            println!("  Pattern {:08b}: {:.1}% success ({} occurrences)",
                pattern, success_rate * 100.0, count);
            print_active_constants(*pattern);
        }
    }
    
    // Performance characteristics
    println!("\n=== Performance Characteristics ===\n");
    
    let small_cases: Vec<_> = test_cases.iter().filter(|c| c.bit_length <= 20).cloned().collect();
    let medium_cases: Vec<_> = test_cases.iter().filter(|c| c.bit_length > 20 && c.bit_length <= 100).cloned().collect();
    let large_cases: Vec<_> = test_cases.iter().filter(|c| c.bit_length > 100).cloned().collect();
    
    println!("Small numbers (≤20 bits, {} cases):", small_cases.len());
    test_and_report_cases(&small_cases, &basis, &params);
    
    println!("\nMedium numbers (21-100 bits, {} cases):", medium_cases.len());
    test_and_report_cases(&medium_cases, &basis, &params);
    
    println!("\nLarge numbers (>100 bits, {} cases):", large_cases.len());
    test_and_report_cases(&large_cases, &basis, &params);
    
    // Final summary
    println!("\n=== Summary and Recommendations ===\n");
    
    let overall_success = test_standard_method(&test_cases, &basis, &params).0;
    
    if overall_success >= 0.95 {
        println!("✓ EXCELLENT: The implementation achieves {:.1}% success rate!", overall_success * 100.0);
        println!("  The 8-bit pattern theory is validated for practical factorization.");
    } else if overall_success >= 0.50 {
        println!("◐ GOOD: The implementation achieves {:.1}% success rate.", overall_success * 100.0);
        println!("  Further tuning could improve performance:");
        println!("  - Optimize constants for specific bit ranges");
        println!("  - Implement adaptive resonance scaling");
        println!("  - Add special case handling");
    } else if overall_success >= 0.20 {
        println!("△ MODERATE: The implementation achieves {:.1}% success rate.", overall_success * 100.0);
        println!("  Significant improvements needed:");
        println!("  - Re-examine the resonance calculation");
        println!("  - Improve factor extraction algorithms");
        println!("  - Consider ensemble methods");
    } else {
        println!("✗ LIMITED: The implementation achieves only {:.1}% success rate.", overall_success * 100.0);
        println!("  Major revisions required:");
        println!("  - Fundamental algorithm improvements needed");
        println!("  - Constants may need complete re-derivation");
        println!("  - Consider alternative pattern theories");
    }
    
    println!("\nStrengths:");
    if small_cases.len() > 0 {
        let small_success = test_standard_method(&small_cases, &basis, &params).0;
        if small_success > 0.8 {
            println!("  - Excellent performance on small numbers ({:.1}%)", small_success * 100.0);
        }
    }
    
    println!("\nAreas for improvement:");
    if large_cases.len() > 0 {
        let large_success = test_standard_method(&large_cases, &basis, &params).0;
        if large_success < 0.2 {
            println!("  - Large number factorization needs work ({:.1}% success)", large_success * 100.0);
        }
    }
}

// Test methods
fn test_standard_method(cases: &[TestCase], _basis: &Basis, params: &TunerParams) -> (f64, f64) {
    let mut successes = 0;
    let mut total_time = 0u128;
    
    for case in cases {
        let start = Instant::now();
        let result = recognize_factors(&case.n, params);
        total_time += start.elapsed().as_micros();
        
        if let Some(factors) = result {
            if factors.verify(&case.n) {
                successes += 1;
            }
        }
    }
    
    (
        successes as f64 / cases.len() as f64,
        total_time as f64 / cases.len() as f64
    )
}

fn test_advanced_method(cases: &[TestCase], basis: &Basis, params: &TunerParams) -> (f64, f64) {
    let mut successes = 0;
    let mut total_time = 0u128;
    
    for case in cases {
        let start = Instant::now();
        let result = recognize_factors_advanced(&case.n, basis, params);
        total_time += start.elapsed().as_micros();
        
        if let Some(factors) = result {
            if factors.verify(&case.n) {
                successes += 1;
            }
        }
    }
    
    (
        successes as f64 / cases.len() as f64,
        total_time as f64 / cases.len() as f64
    )
}

fn test_hybrid_method(cases: &[TestCase], basis: &Basis, params: &TunerParams) -> (f64, f64) {
    let mut successes = 0;
    let mut total_time = 0u128;
    
    for case in cases {
        let start = Instant::now();
        
        // Try standard first for small numbers
        let result = if case.bit_length <= 20 {
            recognize_factors(&case.n, params)
        } else {
            recognize_factors_advanced(&case.n, basis, params)
        };
        
        total_time += start.elapsed().as_micros();
        
        if let Some(factors) = result {
            if factors.verify(&case.n) {
                successes += 1;
            }
        }
    }
    
    (
        successes as f64 / cases.len() as f64,
        total_time as f64 / cases.len() as f64
    )
}

struct BitSizeResults {
    standard_success: f64,
    standard_time: f64,
    advanced_success: f64,
    advanced_time: f64,
    best_success: f64,
    example_factors: Vec<(BigInt, BigInt, BigInt)>,
}

fn test_bit_size(cases: &[&TestCase], basis: &Basis, params: &TunerParams) -> BitSizeResults {
    let mut standard_success = 0;
    let mut advanced_success = 0;
    let mut standard_time = 0u128;
    let mut advanced_time = 0u128;
    let mut example_factors = Vec::new();
    
    for case in cases {
        // Standard method
        let start = Instant::now();
        let std_result = recognize_factors(&case.n, params);
        standard_time += start.elapsed().as_micros();
        
        if let Some(factors) = std_result {
            if factors.verify(&case.n) {
                standard_success += 1;
                if example_factors.len() < 3 {
                    example_factors.push((case.n.clone(), factors.p, factors.q));
                }
            }
        }
        
        // Advanced method
        let start = Instant::now();
        let adv_result = recognize_factors_advanced(&case.n, basis, params);
        advanced_time += start.elapsed().as_micros();
        
        if let Some(factors) = adv_result {
            if factors.verify(&case.n) {
                advanced_success += 1;
            }
        }
    }
    
    let n = cases.len() as f64;
    BitSizeResults {
        standard_success: standard_success as f64 / n,
        standard_time: standard_time as f64 / n,
        advanced_success: advanced_success as f64 / n,
        advanced_time: advanced_time as f64 / n,
        best_success: (standard_success.max(advanced_success)) as f64 / n,
        example_factors,
    }
}

fn test_and_report_cases(cases: &[TestCase], basis: &Basis, params: &TunerParams) {
    test_and_report(&cases.iter().collect::<Vec<_>>(), basis, params);
}

fn test_and_report(cases: &[&TestCase], basis: &Basis, params: &TunerParams) {
    let (success, time) = test_standard_method(
        &cases.iter().map(|&c| c.clone()).collect::<Vec<_>>(), 
        basis, 
        params
    );
    println!("  Success rate: {:.1}%", success * 100.0);
    println!("  Average time: {:.1} μs", time);
    
    if success > 0.0 && cases.len() > 0 {
        // Show timing distribution
        let mut times = Vec::new();
        for case in cases.iter().take(10) {
            let start = Instant::now();
            let _ = recognize_factors(&case.n, params);
            times.push(start.elapsed().as_micros());
        }
        times.sort();
        if times.len() >= 5 {
            println!("  Timing: min={} μs, median={} μs, max={} μs",
                times[0], times[times.len()/2], times[times.len()-1]);
        }
    }
}

fn print_active_constants(pattern: u8) {
    let constant_names = ["unity", "tau", "phi", "epsilon", "delta", "gamma", "beta", "alpha"];
    let active: Vec<_> = (0..8)
        .filter(|&i| (pattern >> i) & 1 == 1)
        .map(|i| constant_names[i])
        .collect();
    
    if !active.is_empty() {
        println!("    Active constants: {}", active.join(", "));
    }
}

fn load_test_cases() -> Vec<TestCase> {
    // Try to load from test_matrix.json
    let path = Path::new("../data/test_matrix.json");
    
    if path.exists() {
        let contents = fs::read_to_string(path).expect("Failed to read test matrix");
        
        #[derive(serde::Deserialize)]
        struct TestMatrix {
            test_cases: std::collections::HashMap<String, Vec<TestCaseJson>>,
        }
        
        #[derive(serde::Deserialize)]
        struct TestCaseJson {
            bit_length: usize,
            n: String,
            p: String,
            q: String,
        }
        
        let matrix: TestMatrix = serde_json::from_str(&contents).expect("Failed to parse test matrix");
        
        let mut test_cases = Vec::new();
        for (_, cases) in matrix.test_cases {
            for case in cases {
                if let (Ok(n), Ok(p), Ok(q)) = (
                    case.n.parse::<BigInt>(),
                    case.p.parse::<BigInt>(),
                    case.q.parse::<BigInt>(),
                ) {
                    test_cases.push(TestCase {
                        n,
                        p,
                        q,
                        bit_length: case.bit_length,
                    });
                }
            }
        }
        
        test_cases
    } else {
        // Default test cases
        vec![
            TestCase { n: BigInt::from(15), p: BigInt::from(3), q: BigInt::from(5), bit_length: 4 },
            TestCase { n: BigInt::from(35), p: BigInt::from(5), q: BigInt::from(7), bit_length: 6 },
            TestCase { n: BigInt::from(143), p: BigInt::from(11), q: BigInt::from(13), bit_length: 8 },
        ]
    }
}