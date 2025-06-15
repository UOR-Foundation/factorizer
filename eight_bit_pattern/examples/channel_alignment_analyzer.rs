//! Channel alignment analyzer to optimize detection thresholds
//!
//! Run with: cargo run --example channel_alignment_analyzer

use eight_bit_pattern::{
    TunerParams, TestCase, compute_basis, decompose,
    recognize_factors_with_diagnostics
};
use num_bigint::BigInt;
use std::fs;
use std::path::Path;
use std::collections::HashMap;

fn main() {
    println!("=== Channel Alignment Analyzer for The 8-Bit Pattern ===\n");
    
    // Load test cases
    let test_cases = load_test_cases();
    let successful_cases: Vec<_> = test_cases.iter()
        .filter(|c| c.bit_length <= 16) // Focus on cases we can solve
        .take(50)
        .collect();
    
    println!("Analyzing {} successful test cases\n", successful_cases.len());
    
    let params = TunerParams::default();
    let basis = compute_basis(32, &params);
    
    // Analyze channel alignment patterns
    let mut alignment_stats = AlignmentStats::new();
    
    for case in &successful_cases {
        analyze_case(case, &basis, &params, &mut alignment_stats);
    }
    
    // Report findings
    println!("\n=== Channel Alignment Analysis ===\n");
    
    println!("1. Channel Value Distribution:");
    print_channel_distribution(&alignment_stats.channel_values);
    
    println!("\n2. Factor Location Patterns:");
    print_factor_patterns(&alignment_stats.factor_positions);
    
    println!("\n3. Channel Correlation with Factors:");
    print_channel_correlations(&alignment_stats);
    
    println!("\n4. Optimal Threshold Analysis:");
    analyze_optimal_thresholds(&alignment_stats);
    
    println!("\n5. Pattern Alignment Strategies:");
    suggest_alignment_strategies(&alignment_stats);
}

struct AlignmentStats {
    // Channel value -> frequency
    channel_values: HashMap<u8, usize>,
    // Factor -> channel positions where it appears
    factor_positions: HashMap<u32, Vec<usize>>,
    // Channel position -> successful factor extractions
    channel_success: HashMap<usize, usize>,
    // Alignment pattern -> success count
    alignment_patterns: HashMap<Vec<u8>, usize>,
    // Distance between factor appearances
    factor_distances: Vec<usize>,
}

impl AlignmentStats {
    fn new() -> Self {
        Self {
            channel_values: HashMap::new(),
            factor_positions: HashMap::new(),
            channel_success: HashMap::new(),
            alignment_patterns: HashMap::new(),
            factor_distances: Vec::new(),
        }
    }
}

fn analyze_case(
    case: &TestCase,
    basis: &eight_bit_pattern::Basis,
    params: &TunerParams,
    stats: &mut AlignmentStats,
) {
    let channels = decompose(&case.n);
    
    // Record channel values
    for &ch in &channels {
        *stats.channel_values.entry(ch).or_default() += 1;
    }
    
    // Check for direct factor encoding
    let p = case.p.to_u32_digits().1.get(0).copied().unwrap_or(0);
    let q = case.q.to_u32_digits().1.get(0).copied().unwrap_or(0);
    
    // Find where factors appear in channels
    for (pos, &ch) in channels.iter().enumerate() {
        if ch as u32 == p % 256 || ch as u32 == q % 256 {
            stats.factor_positions.entry(p).or_default().push(pos);
            stats.factor_positions.entry(q).or_default().push(pos);
            *stats.channel_success.entry(pos).or_default() += 1;
        }
    }
    
    // Run pattern recognition to see what patterns align
    let (result, diagnostics) = recognize_factors_with_diagnostics(&case.n, basis, params);
    
    if result.is_some() {
        // Record successful alignment patterns
        for peak in &diagnostics.peaks {
            let pattern: Vec<u8> = (peak.start_channel..=peak.end_channel)
                .filter_map(|i| channels.get(i).copied())
                .collect();
            
            *stats.alignment_patterns.entry(pattern).or_default() += 1;
        }
    }
    
    // Calculate distances between factor appearances
    if let Some(p_positions) = stats.factor_positions.get(&p) {
        for i in 1..p_positions.len() {
            stats.factor_distances.push(p_positions[i] - p_positions[i-1]);
        }
    }
}

fn print_channel_distribution(values: &HashMap<u8, usize>) {
    let mut sorted: Vec<_> = values.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    
    println!("  Most common channel values:");
    for (val, count) in sorted.iter().take(10) {
        let bar = "█".repeat((*count / 10).min(50));
        println!("    Value {:3}: {} ({})", val, bar, count);
    }
    
    // Analyze value ranges
    let mut ranges = [0usize; 8];
    for (&val, &count) in values {
        ranges[(val / 32) as usize] += count;
    }
    
    println!("\n  Value range distribution:");
    for (i, &count) in ranges.iter().enumerate() {
        let start = i * 32;
        let end = (i + 1) * 32 - 1;
        println!("    Range {:3}-{:3}: {} occurrences", start, end, count);
    }
}

fn print_factor_patterns(positions: &HashMap<u32, Vec<usize>>) {
    println!("  Factors appearing in channels:");
    
    let mut position_freq: HashMap<usize, usize> = HashMap::new();
    for (factor, positions) in positions {
        if *factor < 256 {
            for &pos in positions {
                *position_freq.entry(pos).or_default() += 1;
            }
        }
    }
    
    let mut sorted: Vec<_> = position_freq.iter().collect();
    sorted.sort_by_key(|&(&pos, _)| pos);
    
    for (&pos, &count) in sorted.iter().take(16) {
        println!("    Channel {}: {} factor appearances", pos, count);
    }
}

fn print_channel_correlations(stats: &AlignmentStats) {
    println!("  Channel success rates:");
    
    let mut success_rates: Vec<_> = stats.channel_success.iter()
        .map(|(&pos, &success)| (pos, success))
        .collect();
    success_rates.sort_by(|a, b| b.1.cmp(&a.1));
    
    for (pos, success) in success_rates.iter().take(8) {
        println!("    Channel {}: {} successful extractions", pos, success);
    }
    
    // Analyze pattern lengths
    if !stats.alignment_patterns.is_empty() {
        let avg_length: f64 = stats.alignment_patterns.keys()
            .map(|p| p.len() as f64)
            .sum::<f64>() / stats.alignment_patterns.len() as f64;
        
        println!("\n  Average successful pattern length: {:.1} channels", avg_length);
    }
}

fn analyze_optimal_thresholds(stats: &AlignmentStats) {
    // Analyze factor distances to find optimal alignment threshold
    if !stats.factor_distances.is_empty() {
        let mut distances = stats.factor_distances.clone();
        distances.sort();
        
        let median = distances[distances.len() / 2];
        let min = distances[0];
        let max = distances[distances.len() - 1];
        
        println!("  Factor appearance distances:");
        println!("    Minimum: {} channels", min);
        println!("    Median:  {} channels", median);
        println!("    Maximum: {} channels", max);
        
        // Suggest threshold based on common distances
        let suggested_threshold = median.max(2).min(8);
        println!("\n  Suggested alignment_threshold: {} channels", suggested_threshold);
    }
    
    // Analyze successful pattern characteristics
    if !stats.alignment_patterns.is_empty() {
        let mut pattern_sizes: HashMap<usize, usize> = HashMap::new();
        for (pattern, &count) in &stats.alignment_patterns {
            *pattern_sizes.entry(pattern.len()).or_default() += count;
        }
        
        println!("\n  Successful pattern sizes:");
        let mut sorted: Vec<_> = pattern_sizes.iter().collect();
        sorted.sort_by_key(|&(&size, _)| size);
        
        for (&size, &count) in &sorted {
            println!("    {} channels: {} successes", size, count);
        }
    }
}

fn suggest_alignment_strategies(stats: &AlignmentStats) {
    println!("\n  Recommended alignment strategies:");
    
    // Check if factors appear directly in channels
    let direct_encoding = stats.factor_positions.values()
        .any(|positions| !positions.is_empty());
    
    if direct_encoding {
        println!("  ✓ Direct factor encoding detected in channels");
        println!("    - Check channel values modulo small primes");
        println!("    - Use GCD between aligned channel values");
    }
    
    // Check for periodic patterns
    if stats.factor_distances.len() > 5 {
        let avg_distance = stats.factor_distances.iter().sum::<usize>() as f64 
            / stats.factor_distances.len() as f64;
        
        println!("\n  ✓ Periodic factor appearances detected");
        println!("    - Average distance: {:.1} channels", avg_distance);
        println!("    - Consider sliding window of size {}", avg_distance.ceil() as usize);
    }
    
    // Suggest parameter adjustments
    println!("\n  Parameter tuning suggestions:");
    
    if let Some(&max_success_pos) = stats.channel_success.keys().max() {
        if max_success_pos > 8 {
            println!("    - Increase phase_coupling_strength to check more channels");
        }
    }
    
    let value_concentration = stats.channel_values.len() as f64 / 256.0;
    if value_concentration < 0.5 {
        println!("    - Channel values are concentrated ({:.0}% of range used)", 
            value_concentration * 100.0);
        println!("    - Consider adjusting resonance_scaling_shift");
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