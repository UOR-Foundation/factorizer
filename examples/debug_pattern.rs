//! Debug and tune The Pattern implementation
//! This provides detailed diagnostics to identify where improvements are needed

use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;
use rust_pattern_solver::utils;
use std::str::FromStr;
use std::time::{Duration, Instant};
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct DebugMetrics {
    n: Number,
    n_bits: usize,
    p: Number,
    q: Number,
    balance_ratio: f64,
    
    // Recognition metrics
    phi_component: f64,
    pi_component: f64,
    e_component: f64,
    unity_phase: f64,
    resonance_peaks: usize,
    
    // Execution metrics
    method_used: String,
    success: bool,
    execution_time: Duration,
    
    // Detailed metrics
    phi_sum_error: f64,
    search_radius_used: Option<f64>,
    iterations_used: Option<usize>,
    distance_from_sqrt: Number,
}

fn debug_factorization(pattern: &mut UniversalPattern, n: &Number, p: &Number, q: &Number) -> DebugMetrics {
    let start = Instant::now();
    let sqrt_n = utils::integer_sqrt(n).unwrap();
    
    // Calculate metrics
    let diff = if p > q { p - q } else { q - p };
    let balance_ratio = diff.to_f64().unwrap_or(1e50) / sqrt_n.to_f64().unwrap_or(1e100);
    
    let p_dist = if p > &sqrt_n { p - &sqrt_n } else { &sqrt_n - p };
    let q_dist = if q > &sqrt_n { q - &sqrt_n } else { &sqrt_n - q };
    let distance_from_sqrt = if p_dist > q_dist { p_dist } else { q_dist };
    
    // Recognition
    let recognition = match pattern.recognize(n) {
        Ok(r) => r,
        Err(_) => {
            return DebugMetrics {
                n: n.clone(),
                n_bits: n.bit_length(),
                p: p.clone(),
                q: q.clone(),
                balance_ratio,
                phi_component: 0.0,
                pi_component: 0.0,
                e_component: 0.0,
                unity_phase: 0.0,
                resonance_peaks: 0,
                method_used: "recognition_failed".to_string(),
                success: false,
                execution_time: start.elapsed(),
                phi_sum_error: 0.0,
                search_radius_used: None,
                iterations_used: None,
                distance_from_sqrt,
            };
        }
    };
    
    let phi_component = recognition.phi_component;
    let pi_component = recognition.pi_component;
    let e_component = recognition.e_component;
    let unity_phase = recognition.unity_phase;
    
    // Calculate phi sum error
    let p_phi = p.to_f64().unwrap_or(1e100).ln() / 1.618033988749895_f64.ln();
    let q_phi = q.to_f64().unwrap_or(1e100).ln() / 1.618033988749895_f64.ln();
    let phi_sum_error = ((p_phi + q_phi) - phi_component).abs();
    
    // Formalization
    let formalization = match pattern.formalize(recognition) {
        Ok(f) => f,
        Err(_) => {
            return DebugMetrics {
                n: n.clone(),
                n_bits: n.bit_length(),
                p: p.clone(),
                q: q.clone(),
                balance_ratio,
                phi_component,
                pi_component,
                e_component,
                unity_phase,
                resonance_peaks: 0,
                method_used: "formalization_failed".to_string(),
                success: false,
                execution_time: start.elapsed(),
                phi_sum_error,
                search_radius_used: None,
                iterations_used: None,
                distance_from_sqrt,
            };
        }
    };
    
    let resonance_peaks = formalization.resonance_peaks.len();
    
    // Execution
    let exec_start = Instant::now();
    let (success, method_used) = match pattern.execute(formalization) {
        Ok(factors) => {
            let correct = &factors.p * &factors.q == *n;
            (correct, factors.method)
        }
        Err(_) => (false, "all_failed".to_string()),
    };
    
    DebugMetrics {
        n: n.clone(),
        n_bits: n.bit_length(),
        p: p.clone(),
        q: q.clone(),
        balance_ratio,
        phi_component,
        pi_component,
        e_component,
        unity_phase,
        resonance_peaks,
        method_used,
        success,
        execution_time: exec_start.elapsed(),
        phi_sum_error,
        search_radius_used: None, // Would need to instrument the code
        iterations_used: None,     // Would need to instrument the code
        distance_from_sqrt,
    }
}

fn analyze_metrics(metrics: &[DebugMetrics]) {
    println!("\n{}", "=".repeat(80));
    println!("PATTERN ANALYSIS");
    println!("{}", "=".repeat(80));
    
    // Success rate by method
    let mut method_stats: HashMap<String, (usize, usize)> = HashMap::new();
    for m in metrics {
        let entry = method_stats.entry(m.method_used.clone()).or_insert((0, 0));
        entry.1 += 1;
        if m.success {
            entry.0 += 1;
        }
    }
    
    println!("\nSuccess by method:");
    for (method, (succ, total)) in method_stats {
        println!("  {}: {}/{} ({:.1}%)", 
                 method, succ, total, 
                 100.0 * succ as f64 / total as f64);
    }
    
    // Timing analysis
    println!("\nTiming by bit size (successful only):");
    println!("  Bits | Method              | Avg Time    | Max Time");
    println!("  -----|---------------------|-------------|------------");
    
    let mut bit_buckets: HashMap<usize, Vec<&DebugMetrics>> = HashMap::new();
    for m in metrics {
        if m.success {
            let bucket = (m.n_bits / 32) * 32;
            bit_buckets.entry(bucket).or_insert_with(Vec::new).push(m);
        }
    }
    
    let mut buckets: Vec<_> = bit_buckets.keys().collect();
    buckets.sort();
    
    for bucket in buckets {
        let bucket_metrics = &bit_buckets[bucket];
        if !bucket_metrics.is_empty() {
            let avg_time: Duration = bucket_metrics.iter()
                .map(|m| m.execution_time)
                .sum::<Duration>() / bucket_metrics.len() as u32;
            
            let max_time = bucket_metrics.iter()
                .map(|m| m.execution_time)
                .max()
                .unwrap();
            
            let primary_method = bucket_metrics.iter()
                .map(|m| &m.method_used)
                .max_by_key(|m| bucket_metrics.iter().filter(|x| &x.method_used == *m).count())
                .unwrap();
            
            println!("  {:3}-{:3} | {:19} | {:11.1?} | {:11.1?}", 
                     bucket, bucket + 31, primary_method, avg_time, max_time);
        }
    }
    
    // Pattern component analysis
    println!("\nPattern components vs success:");
    println!("  Status  | Avg φ-comp | Avg φ-error | Avg Balance");
    println!("  --------|------------|-------------|-------------");
    
    let successful: Vec<_> = metrics.iter().filter(|m| m.success).collect();
    let failed: Vec<_> = metrics.iter().filter(|m| !m.success).collect();
    
    if !successful.is_empty() {
        let avg_phi: f64 = successful.iter().map(|m| m.phi_component).sum::<f64>() / successful.len() as f64;
        let avg_error: f64 = successful.iter().map(|m| m.phi_sum_error).sum::<f64>() / successful.len() as f64;
        let avg_balance: f64 = successful.iter().map(|m| m.balance_ratio.log10()).sum::<f64>() / successful.len() as f64;
        println!("  Success | {:10.2} | {:11.6} | 10^{:8.1}", avg_phi, avg_error, avg_balance);
    }
    
    if !failed.is_empty() {
        let avg_phi: f64 = failed.iter().map(|m| m.phi_component).sum::<f64>() / failed.len() as f64;
        let avg_error: f64 = failed.iter().map(|m| m.phi_sum_error).sum::<f64>() / failed.len() as f64;
        let avg_balance: f64 = failed.iter().map(|m| m.balance_ratio.log10()).sum::<f64>() / failed.len() as f64;
        println!("  Failed  | {:10.2} | {:11.6} | 10^{:8.1}", avg_phi, avg_error, avg_balance);
    }
    
    // Distance from sqrt(n) analysis
    println!("\nDistance from sqrt(n) analysis:");
    println!("  Bits | Balance      | Distance    | Success");
    println!("  -----|--------------|-------------|--------");
    
    for m in metrics {
        let dist_bits = m.distance_from_sqrt.bit_length();
        println!("  {:3}  | {:12.2e} | 2^{:9} | {}", 
                 m.n_bits, 
                 m.balance_ratio,
                 dist_bits,
                 if m.success { "✓" } else { "✗" });
    }
}

fn suggest_improvements(metrics: &[DebugMetrics]) {
    println!("\n{}", "=".repeat(80));
    println!("SUGGESTED IMPROVEMENTS");
    println!("{}", "=".repeat(80));
    
    // Analyze failure patterns
    let failures: Vec<_> = metrics.iter().filter(|m| !m.success).collect();
    
    if !failures.is_empty() {
        println!("\nFailure patterns:");
        
        // Check if failures are related to balance ratio
        let balance_threshold = 1e-4;
        let unbalanced_failures = failures.iter()
            .filter(|m| m.balance_ratio > balance_threshold)
            .count();
        
        if unbalanced_failures > failures.len() / 2 {
            println!("  ⚠ Most failures are unbalanced semiprimes (balance > {:.0e})", balance_threshold);
            println!("  → Consider implementing specialized methods for unbalanced cases");
        }
        
        // Check if failures are at specific bit sizes
        let failure_bits: Vec<_> = failures.iter().map(|m| m.n_bits).collect();
        let min_fail = failure_bits.iter().min().unwrap_or(&0);
        println!("  ⚠ Failures start at {} bits", min_fail);
        
        if *min_fail > 128 {
            println!("  → Need better large number handling");
            println!("  → Consider increasing search radius for large numbers");
        }
    }
    
    // Analyze timing
    let slow_threshold = Duration::from_secs(1);
    let slow_cases: Vec<_> = metrics.iter()
        .filter(|m| m.execution_time > slow_threshold)
        .collect();
    
    if !slow_cases.is_empty() {
        println!("\nPerformance issues:");
        println!("  ⚠ {} cases took > 1 second", slow_cases.len());
        
        let avg_slow_bits = slow_cases.iter()
            .map(|m| m.n_bits)
            .sum::<usize>() / slow_cases.len();
        
        println!("  → Average bit size of slow cases: {}", avg_slow_bits);
        println!("  → Consider optimizing for {}-bit range", avg_slow_bits);
    }
    
    // Analyze phi sum error
    let high_error_cases: Vec<_> = metrics.iter()
        .filter(|m| m.phi_sum_error > 0.1)
        .collect();
    
    if !high_error_cases.is_empty() {
        println!("\nPattern accuracy issues:");
        println!("  ⚠ {} cases have high φ-sum error (> 0.1)", high_error_cases.len());
        println!("  → The φ-sum invariant may need adjustment for certain number types");
    }
}

fn main() {
    println!("=== Debug and Tune The Pattern ===\n");
    
    let mut pattern = UniversalPattern::new();
    let mut all_metrics = Vec::new();
    
    // Test suite with various number types
    let test_cases = vec![
        // Small balanced
        (Number::from(15u32), Number::from(3u32), Number::from(5u32), "Small balanced"),
        (Number::from(143u32), Number::from(11u32), Number::from(13u32), "Twin primes"),
        
        // Medium balanced
        (Number::from(9223372012704246007u64), Number::from(3037000493u64), Number::from(3037000499u64), "64-bit balanced"),
        
        // Large balanced
        (
            Number::from_str("1208925819627823314239443").unwrap(),
            Number::from_str("1099511627773").unwrap(),
            Number::from_str("1099511627791").unwrap(),
            "80-bit balanced"
        ),
        
        // Very large balanced
        (
            Number::from_str("340282366920938464127457394085312069931").unwrap(),
            Number::from_str("18446744073709551629").unwrap(),
            Number::from_str("18446744073709551639").unwrap(),
            "128-bit balanced"
        ),
        
        // 160-bit balanced
        (
            Number::from_str("1461501637330902918203812978853162170348451400437").unwrap(),
            Number::from_str("1208925819614629174706227").unwrap(),
            Number::from_str("1208925819614629174706231").unwrap(),
            "160-bit balanced"
        ),
        
        // Unbalanced cases
        (Number::from(35u32), Number::from(5u32), Number::from(7u32), "Small unbalanced"),
        (Number::from(997003u64), Number::from(997u64), Number::from(1003u64), "Medium unbalanced"),
        
        // Special cases
        (Number::from(25u32), Number::from(5u32), Number::from(5u32), "Perfect square"),
        (Number::from(77u32), Number::from(7u32), Number::from(11u32), "7×11"),
    ];
    
    println!("Running debug tests...\n");
    
    for (n_input, p, q, desc) in test_cases {
        // Handle both n provided and p,q provided cases
        let n = if n_input == Number::from(0u32) {
            &p * &q
        } else {
            n_input
        };
        
        println!("Testing: {} (n={}, {} bits)", desc, n, n.bit_length());
        let metrics = debug_factorization(&mut pattern, &n, &p, &q);
        
        println!("  Result: {} via {} in {:?}", 
                 if metrics.success { "✓" } else { "✗" },
                 metrics.method_used,
                 metrics.execution_time);
        
        all_metrics.push(metrics);
    }
    
    // Analyze results
    analyze_metrics(&all_metrics);
    suggest_improvements(&all_metrics);
    
    println!("\n{}", "=".repeat(80));
    println!("TUNING RECOMMENDATIONS");
    println!("{}", "=".repeat(80));
    
    println!("\n1. Balance Detection:");
    println!("   - Current threshold may be too strict for large numbers");
    println!("   - Consider adaptive thresholds based on bit size");
    
    println!("\n2. Search Radius:");
    println!("   - Needs better scaling for 100+ bit numbers");
    println!("   - Consider using empirical constants more effectively");
    
    println!("\n3. Method Selection:");
    println!("   - Fermat works well for balanced cases");
    println!("   - Need better methods for unbalanced semiprimes");
    
    println!("\n4. Pre-computed Basis:");
    println!("   - Should implement the poly-time approach");
    println!("   - Use universal constants for scaling");
}