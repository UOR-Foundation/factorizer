//! Test parallel channel processing performance
//!
//! Run with: cargo run --example test_parallel --release

use eight_bit_pattern::{
    recognize_factors, recognize_factors_parallel,
    TunerParams, compute_basis, TestCase
};
use num_bigint::BigInt;
use std::time::Instant;
use rayon::ThreadPoolBuilder;

fn main() {
    println!("=== Parallel Channel Processing Performance Test ===\n");
    
    // Configure thread pool
    let num_threads = num_cpus::get();
    ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();
    
    println!("Using {} threads for parallel processing\n", num_threads);
    
    // Generate test cases of various sizes
    let test_cases = generate_test_cases();
    let params = TunerParams::default();
    let basis = compute_basis(128, &params);
    
    // Warmup
    println!("Warming up...");
    for case in test_cases.iter().take(10) {
        let _ = recognize_factors(&case.n, &basis, &params);
        let _ = recognize_factors_parallel(&case.n, &basis, &params);
    }
    
    println!("\n=== Performance Comparison ===\n");
    
    // Group by bit size
    let mut grouped: std::collections::HashMap<usize, Vec<&TestCase>> = std::collections::HashMap::new();
    for case in &test_cases {
        grouped.entry(case.bit_length).or_default().push(case);
    }
    
    let mut bit_sizes: Vec<_> = grouped.keys().copied().collect();
    bit_sizes.sort();
    
    for bit_size in bit_sizes {
        let cases = &grouped[&bit_size];
        if cases.is_empty() { continue; }
        
        println!("{}-bit numbers ({} cases):", bit_size, cases.len());
        
        // Test serial version
        let mut serial_time = 0u128;
        let mut serial_success = 0;
        
        for case in cases {
            let start = Instant::now();
            if let Some(factors) = recognize_factors(&case.n, &basis, &params) {
                if factors.verify(&case.n) {
                    serial_success += 1;
                }
            }
            serial_time += start.elapsed().as_micros();
        }
        
        // Test parallel version
        let mut parallel_time = 0u128;
        let mut parallel_success = 0;
        
        for case in cases {
            let start = Instant::now();
            if let Some(factors) = recognize_factors_parallel(&case.n, &basis, &params) {
                if factors.verify(&case.n) {
                    parallel_success += 1;
                }
            }
            parallel_time += start.elapsed().as_micros();
        }
        
        let serial_avg = serial_time as f64 / cases.len() as f64;
        let parallel_avg = parallel_time as f64 / cases.len() as f64;
        let speedup = serial_avg / parallel_avg;
        
        println!("  Serial:   {:.1} μs/number ({}/{} success)", 
            serial_avg, serial_success, cases.len());
        println!("  Parallel: {:.1} μs/number ({}/{} success)", 
            parallel_avg, parallel_success, cases.len());
        println!("  Speedup:  {:.2}x", speedup);
        
        if speedup < 0.9 {
            println!("  Note: Parallel is slower for this bit size (overhead > benefit)");
        }
        
        println!();
    }
    
    // Test very large numbers
    println!("\n=== Large Number Performance ===\n");
    
    test_large_number("64-bit", BigInt::from(u64::MAX), &basis, &params);
    test_large_number("128-bit", BigInt::from(1u128) << 127, &basis, &params);
    test_large_number("256-bit", BigInt::from(1u128) << 255, &basis, &params);
}

fn test_large_number(label: &str, n: BigInt, basis: &eight_bit_pattern::Basis, params: &TunerParams) {
    println!("{} number:", label);
    
    let start = Instant::now();
    let serial_channels = eight_bit_pattern::decompose(&n);
    let serial_time = start.elapsed();
    
    let start = Instant::now();
    let parallel_channels = eight_bit_pattern::decompose_parallel(&n);
    let parallel_time = start.elapsed();
    
    println!("  Channel decomposition:");
    println!("    Serial:   {:?}", serial_time);
    println!("    Parallel: {:?}", parallel_time);
    println!("    Speedup:  {:.2}x", serial_time.as_nanos() as f64 / parallel_time.as_nanos() as f64);
    
    assert_eq!(serial_channels, parallel_channels, "Channel mismatch!");
    
    // Full recognition test
    let start = Instant::now();
    let _ = recognize_factors(&n, basis, params);
    let serial_total = start.elapsed();
    
    let start = Instant::now();
    let _ = recognize_factors_parallel(&n, basis, params);
    let parallel_total = start.elapsed();
    
    println!("  Full recognition:");
    println!("    Serial:   {:?}", serial_total);
    println!("    Parallel: {:?}", parallel_total);
    println!("    Speedup:  {:.2}x", serial_total.as_nanos() as f64 / parallel_total.as_nanos() as f64);
    println!();
}

fn generate_test_cases() -> Vec<TestCase> {
    let mut cases = Vec::new();
    
    // Small cases (where parallel might have overhead)
    let small_primes = vec![3, 5, 7, 11, 13, 17, 19, 23, 29, 31];
    for i in 0..small_primes.len() {
        for j in i..small_primes.len() {
            let p = BigInt::from(small_primes[i]);
            let q = BigInt::from(small_primes[j]);
            let n = &p * &q;
            let bit_length = n.bits() as usize;
            cases.push(TestCase { n, p, q, bit_length });
        }
    }
    
    // Medium cases
    let medium_primes = vec![97, 101, 103, 107, 109, 113, 127, 131, 137, 139];
    for i in 0..medium_primes.len().min(5) {
        for j in i..medium_primes.len().min(5) {
            let p = BigInt::from(medium_primes[i]);
            let q = BigInt::from(medium_primes[j]);
            let n = &p * &q;
            let bit_length = n.bits() as usize;
            cases.push(TestCase { n, p, q, bit_length });
        }
    }
    
    // Larger cases (where parallel should help)
    let large_primes = vec![
        1009u32, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061
    ];
    for i in 0..large_primes.len().min(3) {
        for j in i..large_primes.len().min(3) {
            let p = BigInt::from(large_primes[i]);
            let q = BigInt::from(large_primes[j]);
            let n = &p * &q;
            let bit_length = n.bits() as usize;
            cases.push(TestCase { n, p, q, bit_length });
        }
    }
    
    cases
}