//! Test scale invariance of The 8-Bit Pattern across different bit sizes
//!
//! Run with: cargo run --example scale_invariance_test

use eight_bit_pattern::{
    TunerParams, TestCase, compute_basis, recognize_factors, 
    decompose
};
use num_bigint::BigInt;
use std::fs;
use std::path::Path;
use std::collections::HashMap;

fn main() {
    println!("=== Scale Invariance Testing for The 8-Bit Pattern ===\n");
    
    // Load test cases
    let test_cases = load_test_cases();
    println!("Loaded {} test cases from test matrix", test_cases.len());
    
    // Group by bit size ranges
    let ranges = vec![
        (1, 8, "Tiny (1-8 bits)"),
        (9, 16, "Small (9-16 bits)"),
        (17, 32, "Medium (17-32 bits)"),
        (33, 64, "Large (33-64 bits)"),
        (65, 128, "Very Large (65-128 bits)"),
        (129, 256, "Huge (129-256 bits)"),
        (257, 512, "Massive (257-512 bits)"),
        (513, 1024, "Gigantic (513-1024 bits)"),
    ];
    
    let params = TunerParams::default();
    let basis = compute_basis(128, &params);
    
    println!("\n=== Success Rate by Scale ===\n");
    
    let mut scale_results = Vec::new();
    
    for (min_bits, max_bits, name) in &ranges {
        let cases_in_range: Vec<_> = test_cases.iter()
            .filter(|c| c.bit_length >= *min_bits && c.bit_length <= *max_bits)
            .collect();
        
        if cases_in_range.is_empty() {
            continue;
        }
        
        let (success_rate, avg_bits, pattern_stats) = analyze_scale(&cases_in_range, &basis, &params);
        
        println!("{} - {} cases:", name, cases_in_range.len());
        println!("  Success rate: {:.1}%", success_rate * 100.0);
        println!("  Average bit size: {:.0}", avg_bits);
        
        scale_results.push((
            *min_bits,
            *max_bits,
            name.to_string(),
            success_rate,
            cases_in_range.len(),
            pattern_stats,
        ));
    }
    
    // Analyze pattern consistency across scales
    println!("\n=== Pattern Consistency Analysis ===\n");
    
    analyze_pattern_consistency(&scale_results);
    
    // Test adaptive scaling
    println!("\n=== Adaptive Scaling Test ===\n");
    
    test_adaptive_scaling(&test_cases, &basis);
    
    // Analyze channel behavior at different scales
    println!("\n=== Channel Behavior by Scale ===\n");
    
    analyze_channel_behavior(&test_cases, &basis, &params);
    
    // Summary and recommendations
    println!("\n=== Summary and Recommendations ===\n");
    
    provide_scale_recommendations(&scale_results);
}

struct PatternStats {
    common_patterns: HashMap<u8, usize>,
    channel_utilization: [usize; 32],
    avg_peaks_per_number: f64,
}

fn analyze_scale(
    cases: &[&TestCase], 
    basis: &eight_bit_pattern::Basis, 
    params: &TunerParams
) -> (f64, f64, PatternStats) {
    let mut successes = 0;
    let mut total_bits = 0;
    let mut pattern_freq = HashMap::new();
    let mut channel_usage = [0usize; 32];
    let mut total_peaks = 0;
    
    for case in cases {
        total_bits += case.bit_length;
        
        // Test factorization
        let result = recognize_factors(&case.n, basis, params);
        if let Some(factors) = result {
            if factors.verify(&case.n) {
                successes += 1;
            }
        }
        
        // Analyze patterns
        let (_, diagnostics) = eight_bit_pattern::recognize_factors_with_diagnostics(
            &case.n, basis, params
        );
        
        total_peaks += diagnostics.peaks.len();
        
        for peak in &diagnostics.peaks {
            *pattern_freq.entry(peak.aligned_pattern).or_default() += 1;
        }
        
        // Track channel usage
        let channels = decompose(&case.n);
        for i in 0..channels.len().min(32) {
            if channels[i] != 0 {
                channel_usage[i] += 1;
            }
        }
    }
    
    let n = cases.len() as f64;
    let stats = PatternStats {
        common_patterns: pattern_freq,
        channel_utilization: channel_usage,
        avg_peaks_per_number: total_peaks as f64 / n,
    };
    
    (
        successes as f64 / n,
        total_bits as f64 / n,
        stats,
    )
}

fn analyze_pattern_consistency(results: &[(usize, usize, String, f64, usize, PatternStats)]) {
    // Find patterns that appear consistently across scales
    let mut global_patterns: HashMap<u8, Vec<(String, usize)>> = HashMap::new();
    
    for (_, _, scale_name, _, _, stats) in results {
        for (&pattern, &count) in &stats.common_patterns {
            global_patterns.entry(pattern).or_default().push((scale_name.clone(), count));
        }
    }
    
    // Find universally appearing patterns
    let universal_patterns: Vec<_> = global_patterns.iter()
        .filter(|(_, occurrences)| occurrences.len() >= 3) // Appears in at least 3 scales
        .collect();
    
    println!("Universal patterns (appearing across multiple scales):");
    for (&pattern, occurrences) in universal_patterns.iter().take(10) {
        println!("\n  Pattern {:08b}:", pattern);
        print_active_constants(pattern);
        
        for (scale, count) in occurrences.iter() {
            println!("    {}: {} occurrences", scale, count);
        }
    }
    
    // Analyze channel utilization trends
    println!("\n\nChannel utilization trends:");
    
    for (min_bits, max_bits, scale_name, _, case_count, stats) in results {
        if *case_count == 0 {
            continue;
        }
        
        let most_used_channel = stats.channel_utilization.iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        let utilization_rate = stats.channel_utilization.iter()
            .take(8) // Focus on first 8 channels
            .sum::<usize>() as f64 / (*case_count * 8) as f64;
        
        println!("  {}: Most active channel: {}, Utilization: {:.1}%",
            scale_name, most_used_channel, utilization_rate * 100.0);
    }
}

fn test_adaptive_scaling(test_cases: &[TestCase], basis: &eight_bit_pattern::Basis) {
    // Test different parameter configurations for different scales
    let scale_configs = vec![
        (20, TunerParams {
            alignment_threshold: 2,  // Lower threshold for small numbers
            phase_coupling_strength: 5,  // More coupling for small numbers
            ..TunerParams::default()
        }, "Small numbers (<20 bits)"),
        (64, TunerParams {
            alignment_threshold: 3,  // Default threshold
            phase_coupling_strength: 3,  // Default coupling
            ..TunerParams::default()
        }, "Medium numbers (20-64 bits)"),
        (256, TunerParams {
            alignment_threshold: 4,  // Higher threshold for large numbers
            phase_coupling_strength: 1,  // Less coupling for large numbers
            ..TunerParams::default()
        }, "Large numbers (64-256 bits)"),
    ];
    
    println!("Testing adaptive parameter configurations:\n");
    
    for (max_bits, params, description) in scale_configs {
        let cases: Vec<_> = test_cases.iter()
            .filter(|c| c.bit_length <= max_bits)
            .take(50) // Limit for speed
            .collect();
        
        if cases.is_empty() {
            continue;
        }
        
        let mut successes = 0;
        for case in &cases {
            let result = recognize_factors(&case.n, basis, &params);
            if let Some(factors) = result {
                if factors.verify(&case.n) {
                    successes += 1;
                }
            }
        }
        
        let success_rate = successes as f64 / cases.len() as f64;
        println!("  {}: {:.1}% success rate", description, success_rate * 100.0);
        println!("    Parameters: alignment_threshold={}, phase_coupling={}", 
            params.alignment_threshold, params.phase_coupling_strength);
    }
}

fn analyze_channel_behavior(
    test_cases: &[TestCase], 
    basis: &eight_bit_pattern::Basis,
    params: &TunerParams
) {
    // Sample numbers from different scales
    let samples = vec![
        (8, "8-bit"),
        (16, "16-bit"),
        (32, "32-bit"),
        (64, "64-bit"),
        (128, "128-bit"),
    ];
    
    for (target_bits, label) in samples {
        if let Some(case) = test_cases.iter().find(|c| c.bit_length == target_bits) {
            println!("\nExample {} number: {}", label, case.n);
            
            let channels = decompose(&case.n);
            let active_channels = channels.iter().take(16).filter(|&&c| c != 0).count();
            
            println!("  Active channels: {} of {}", active_channels, channels.len().min(16));
            println!("  First 8 channels: {:?}", &channels[..8.min(channels.len())]);
            
            // Show pattern distribution
            let (_, diagnostics) = eight_bit_pattern::recognize_factors_with_diagnostics(
                &case.n, basis, params
            );
            
            if !diagnostics.peaks.is_empty() {
                println!("  Detected {} peaks", diagnostics.peaks.len());
                let avg_strength: f64 = diagnostics.peaks.iter()
                    .map(|p| p.alignment_strength as f64)
                    .sum::<f64>() / diagnostics.peaks.len() as f64;
                println!("  Average peak strength: {:.2}", avg_strength);
            }
        }
    }
}

fn provide_scale_recommendations(results: &[(usize, usize, String, f64, usize, PatternStats)]) {
    // Find scale with best performance
    let best_scale = results.iter()
        .filter(|(_, _, _, _, count, _)| *count > 10) // Minimum sample size
        .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap());
    
    if let Some((min_bits, max_bits, name, success_rate, _, _)) = best_scale {
        println!("Best performance: {} with {:.1}% success rate", name, success_rate * 100.0);
        
        if *success_rate > 0.8 {
            println!("\n✓ The Pattern shows strong scale invariance in the {}-{} bit range", 
                min_bits, max_bits);
        }
    }
    
    // Analyze scale degradation
    let small_success = results.iter()
        .find(|(min, max, _, _, _, _)| *min <= 32 && *max >= 16)
        .map(|(_, _, _, rate, _, _)| rate)
        .unwrap_or(&0.0);
    
    let large_success = results.iter()
        .find(|(min, _, _, _, _, _)| *min > 64)
        .map(|(_, _, _, rate, _, _)| rate)
        .unwrap_or(&0.0);
    
    if *small_success > *large_success + 0.3 {
        println!("\n⚠ Significant performance degradation at larger scales detected");
        println!("  Recommendations:");
        println!("  - Implement scale-adaptive constant tuning");
        println!("  - Use different resonance calculations for large numbers");
        println!("  - Consider hierarchical pattern recognition");
    } else if (small_success - large_success).abs() < 0.1 {
        println!("\n✓ Excellent scale invariance - pattern works consistently across scales");
    }
    
    // Pattern consistency recommendations
    println!("\nPattern consistency insights:");
    let avg_peaks: Vec<_> = results.iter()
        .map(|(_, _, name, _, _, stats)| (name, stats.avg_peaks_per_number))
        .collect();
    
    for (name, avg) in avg_peaks {
        if avg > 5.0 {
            println!("  {} has high pattern density ({:.1} peaks/number)", name, avg);
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
        vec![]
    }
}