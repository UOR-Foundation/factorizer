//! Diagnostic factorization test with detailed metrics for tuning scaling
//! This provides comprehensive information about each factorization attempt

use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;
use rust_pattern_solver::utils;
use std::str::FromStr;
use std::time::{Duration, Instant};
use std::collections::HashMap;

#[derive(Debug)]
struct FactorizationDiagnostics {
    n_bits: usize,
    balance_ratio: f64,
    phi_component: f64,
    pi_component: f64,
    e_component: f64,
    unity_phase: f64,
    resonance_peaks: usize,
    recognition_time: Duration,
    formalization_time: Duration,
    execution_time: Duration,
    method_used: String,
    success: bool,
    sqrt_n_bits: usize,
    factor_distance: Option<Number>,
    fermat_iterations: Option<usize>,
    search_radius: Option<Number>,
}

fn collect_diagnostics(
    pattern: &mut UniversalPattern,
    n: &Number,
    expected_p: &Number,
    expected_q: &Number,
) -> FactorizationDiagnostics {
    let sqrt_n = utils::integer_sqrt(n).unwrap();
    let sqrt_n_bits = sqrt_n.bit_length();
    
    // Calculate balance ratio
    let diff = if expected_p > expected_q {
        expected_p - expected_q
    } else {
        expected_q - expected_p
    };
    let balance_ratio = diff.to_f64().unwrap_or(1e50) / sqrt_n.to_f64().unwrap_or(1e100);
    
    // Calculate factor distance from sqrt(n)
    let p_dist = if expected_p > &sqrt_n {
        expected_p - &sqrt_n
    } else {
        &sqrt_n - expected_p
    };
    let q_dist = if expected_q > &sqrt_n {
        expected_q - &sqrt_n
    } else {
        &sqrt_n - expected_q
    };
    let factor_distance = if p_dist > q_dist { p_dist } else { q_dist };
    
    // Recognition phase
    let start = Instant::now();
    let recognition = match pattern.recognize(n) {
        Ok(r) => r,
        Err(_) => {
            return FactorizationDiagnostics {
                n_bits: n.bit_length(),
                balance_ratio,
                phi_component: 0.0,
                pi_component: 0.0,
                e_component: 0.0,
                unity_phase: 0.0,
                resonance_peaks: 0,
                recognition_time: start.elapsed(),
                formalization_time: Duration::from_secs(0),
                execution_time: Duration::from_secs(0),
                method_used: "recognition_failed".to_string(),
                success: false,
                sqrt_n_bits,
                factor_distance: Some(factor_distance),
                fermat_iterations: None,
                search_radius: None,
            };
        }
    };
    let recognition_time = start.elapsed();
    
    // Formalization phase
    let start = Instant::now();
    let formalization = match pattern.formalize(recognition.clone()) {
        Ok(f) => f,
        Err(_) => {
            return FactorizationDiagnostics {
                n_bits: n.bit_length(),
                balance_ratio,
                phi_component: recognition.phi_component,
                pi_component: recognition.pi_component,
                e_component: recognition.e_component,
                unity_phase: recognition.unity_phase,
                resonance_peaks: 0,
                recognition_time,
                formalization_time: start.elapsed(),
                execution_time: Duration::from_secs(0),
                method_used: "formalization_failed".to_string(),
                success: false,
                sqrt_n_bits,
                factor_distance: Some(factor_distance),
                fermat_iterations: None,
                search_radius: None,
            };
        }
    };
    let formalization_time = start.elapsed();
    
    // Execution phase
    let start = Instant::now();
    let (success, method_used) = match pattern.execute(formalization.clone()) {
        Ok(factors) => {
            let correct = &factors.p * &factors.q == *n;
            (correct, factors.method)
        }
        Err(_) => (false, "all_methods_failed".to_string()),
    };
    let execution_time = start.elapsed();
    
    FactorizationDiagnostics {
        n_bits: n.bit_length(),
        balance_ratio,
        phi_component: recognition.phi_component,
        pi_component: recognition.pi_component,
        e_component: recognition.e_component,
        unity_phase: recognition.unity_phase,
        resonance_peaks: formalization.resonance_peaks.len(),
        recognition_time,
        formalization_time,
        execution_time,
        method_used,
        success,
        sqrt_n_bits,
        factor_distance: Some(factor_distance),
        fermat_iterations: None, // Would need to instrument the code to get this
        search_radius: None,     // Would need to instrument the code to get this
    }
}

fn analyze_scaling(diagnostics: &[FactorizationDiagnostics]) {
    println!("\n{}", "=".repeat(80));
    println!("SCALING ANALYSIS");
    println!("{}", "=".repeat(80));
    
    // Group by success/failure
    let successful: Vec<_> = diagnostics.iter().filter(|d| d.success).collect();
    let failed: Vec<_> = diagnostics.iter().filter(|d| !d.success).collect();
    
    println!("\nSuccess rate by bit size:");
    let mut bit_groups: HashMap<usize, (usize, usize)> = HashMap::new();
    for d in diagnostics {
        let bucket = (d.n_bits / 32) * 32; // Group by 32-bit buckets
        let entry = bit_groups.entry(bucket).or_insert((0, 0));
        entry.1 += 1; // total
        if d.success {
            entry.0 += 1; // successes
        }
    }
    
    let mut buckets: Vec<_> = bit_groups.keys().collect();
    buckets.sort();
    
    for bucket in buckets {
        let (succ, total) = bit_groups[bucket];
        println!("  {}-{} bits: {}/{} ({:.1}%)", 
                 bucket, bucket + 31, succ, total, 
                 100.0 * succ as f64 / total as f64);
    }
    
    // Analyze patterns in failures
    if !failed.is_empty() {
        println!("\nFailure analysis:");
        println!("  Bits | Balance Ratio | φ-component | Method Failed");
        println!("  -----|---------------|-------------|---------------");
        for d in &failed {
            println!("  {:4} | {:13.2e} | {:11.2} | {}", 
                     d.n_bits, d.balance_ratio, d.phi_component, d.method_used);
        }
    }
    
    // Analyze timing patterns
    if !successful.is_empty() {
        println!("\nTiming analysis for successful factorizations:");
        println!("  Bits | Balance | Recognition | Formalization | Execution | Method");
        println!("  -----|---------|-------------|---------------|-----------|-------");
        for d in &successful {
            println!("  {:4} | {:7.0e} | {:11.1?} | {:13.1?} | {:9.1?} | {}", 
                     d.n_bits, 
                     d.balance_ratio,
                     d.recognition_time,
                     d.formalization_time,
                     d.execution_time,
                     d.method_used);
        }
    }
    
    // Analyze universal coordinates scaling
    println!("\nUniversal coordinate scaling:");
    println!("  Bits | φ-comp | π-comp | e-comp | unity | peaks");
    println!("  -----|--------|--------|--------|-------|-------");
    for d in diagnostics {
        println!("  {:4} | {:6.1} | {:6.2} | {:6.2} | {:5.3} | {:5}",
                 d.n_bits,
                 d.phi_component,
                 d.pi_component,
                 d.e_component,
                 d.unity_phase,
                 d.resonance_peaks);
    }
    
    // Look for scaling relationships
    println!("\nScaling relationships:");
    
    // φ-component vs bits
    let phi_scaling: Vec<(f64, f64)> = diagnostics.iter()
        .map(|d| (d.n_bits as f64, d.phi_component))
        .collect();
    
    if phi_scaling.len() > 2 {
        // Simple linear regression
        let n = phi_scaling.len() as f64;
        let sum_x: f64 = phi_scaling.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = phi_scaling.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = phi_scaling.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = phi_scaling.iter().map(|(x, _)| x * x).sum();
        
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;
        
        println!("  φ-component ≈ {:.3} * bits + {:.3}", slope, intercept);
        
        // Check if it's closer to logarithmic
        let log_scaling: Vec<(f64, f64)> = diagnostics.iter()
            .map(|d| (d.n_bits as f64, d.phi_component / d.n_bits as f64))
            .collect();
        
        let ratio_avg: f64 = log_scaling.iter().map(|(_, r)| r).sum::<f64>() / log_scaling.len() as f64;
        println!("  φ-component / bits ≈ {:.6} (logarithmic relationship)", ratio_avg);
    }
}

fn main() {
    println!("=== Diagnostic Factorization Test ===\n");
    println!("This test collects detailed metrics to help tune scaling parameters.\n");
    
    let mut pattern = UniversalPattern::new();
    let mut diagnostics = Vec::new();
    
    // Test cases with known factors
    let test_cases = vec![
        // Small cases
        (Number::from(15u32), Number::from(3u32), Number::from(5u32), "15 = 3 × 5"),
        (Number::from(21u32), Number::from(3u32), Number::from(7u32), "21 = 3 × 7"),
        (Number::from(143u32), Number::from(11u32), Number::from(13u32), "143 = 11 × 13"),
        
        // 32-bit
        (Number::from(2147673613u64), Number::from(46337u64), Number::from(46349u64), "32-bit balanced"),
        
        // 48-bit
        (Number::from(140737436084957u64), Number::from(11863283u64), Number::from(11863279u64), "48-bit balanced"),
        
        // 64-bit
        (Number::from(9223372012704246007u64), Number::from(3037000493u64), Number::from(3037000499u64), "64-bit balanced"),
        
        // 80-bit
        (
            Number::from_str("1208925819627823314239443").unwrap(),
            Number::from_str("1099511627773").unwrap(),
            Number::from_str("1099511627791").unwrap(),
            "80-bit balanced"
        ),
        
        // 96-bit
        (
            Number::from_str("79228162514280100192239747807").unwrap(),
            Number::from_str("281474976710677").unwrap(),
            Number::from_str("281474976710691").unwrap(),
            "96-bit balanced"
        ),
        
        // 128-bit
        (
            Number::from_str("340282366920938464127457394085312069931").unwrap(),
            Number::from_str("18446744073709551629").unwrap(),
            Number::from_str("18446744073709551639").unwrap(),
            "128-bit balanced"
        ),
        
        // Unbalanced cases
        (Number::from(35u32), Number::from(5u32), Number::from(7u32), "35 = 5 × 7 unbalanced"),
        (Number::from(997003u64), Number::from(997u64), Number::from(1003u64), "~20-bit mild unbalanced"),
    ];
    
    for (n, p, q, desc) in test_cases {
        println!("\n{}", "=".repeat(60));
        println!("Testing: {}", desc);
        println!("n = {} ({} bits)", n, n.bit_length());
        
        let diag = collect_diagnostics(&mut pattern, &n, &p, &q);
        
        println!("Balance ratio: {:.2e}", diag.balance_ratio);
        println!("Recognition: {:?}", diag.recognition_time);
        println!("  φ={:.2}, π={:.2}, e={:.2}, unity={:.3}", 
                 diag.phi_component, diag.pi_component, diag.e_component, diag.unity_phase);
        println!("Formalization: {:?} ({} peaks)", diag.formalization_time, diag.resonance_peaks);
        println!("Execution: {:?}", diag.execution_time);
        
        if diag.success {
            println!("✓ Success via {}", diag.method_used);
        } else {
            println!("✗ Failed at {}", diag.method_used);
        }
        
        diagnostics.push(diag);
    }
    
    // Perform scaling analysis
    analyze_scaling(&diagnostics);
    
    println!("\n{}", "=".repeat(80));
    println!("KEY OBSERVATIONS FOR TUNING:");
    println!("1. Look at where factorization starts failing (bit threshold)");
    println!("2. Note the relationship between φ-component and bit size");
    println!("3. Observe which methods succeed at different scales");
    println!("4. Check if balance ratio threshold needs adjustment");
    println!("5. Monitor execution times to identify performance bottlenecks");
}