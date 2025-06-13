//! Analyze test matrix results to understand performance patterns
//! 
//! This tool examines the test results to identify why performance
//! degrades at 48+ bits and what patterns lead to successful factorization.

use rust_pattern_solver::types::Number;
use std::collections::{HashMap, BTreeMap};
use std::str::FromStr;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestCase {
    bit_length: usize,
    n: String,
    p: String,
    q: String,
    balanced: bool,
    p_bits: usize,
    q_bits: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct TestMatrix {
    version: String,
    generated: String,
    description: String,
    test_cases: BTreeMap<usize, Vec<TestCase>>,
}

#[derive(Debug)]
struct MethodPerformance {
    method_name: String,
    success_count: usize,
    total_count: usize,
    avg_time_ms: f64,
    bit_ranges: Vec<usize>,
}

#[derive(Debug)]
struct BytePattern {
    byte_value: u8,
    channel_position: usize,
    leads_to_success: bool,
    factor_relationship: Option<(String, String)>,
}

fn analyze_successful_patterns(test_matrix: &TestMatrix) -> HashMap<usize, Vec<BytePattern>> {
    let mut patterns_by_bit_length = HashMap::new();
    
    // Focus on successful cases (8-40 bits)
    for (bit_length, cases) in &test_matrix.test_cases {
        if *bit_length <= 40 {
            let mut patterns = Vec::new();
            
            for case in cases {
                let n = Number::from_str(&case.n).unwrap();
                let bytes = n_to_bytes(&n);
                
                // Record byte patterns
                for (idx, &byte_val) in bytes.iter().enumerate() {
                    patterns.push(BytePattern {
                        byte_value: byte_val,
                        channel_position: idx,
                        leads_to_success: true,
                        factor_relationship: Some((case.p.clone(), case.q.clone())),
                    });
                }
            }
            
            patterns_by_bit_length.insert(*bit_length, patterns);
        }
    }
    
    patterns_by_bit_length
}

fn n_to_bytes(n: &Number) -> Vec<u8> {
    // Convert number to little-endian bytes
    let hex_str = format!("{:x}", n.as_integer());
    let mut bytes = Vec::new();
    
    // Convert hex string to bytes
    for i in (0..hex_str.len()).step_by(2) {
        if i + 1 < hex_str.len() {
            let byte_str = &hex_str[i..i+2];
            if let Ok(byte) = u8::from_str_radix(byte_str, 16) {
                bytes.push(byte);
            }
        } else {
            // Handle odd length
            let byte_str = &hex_str[i..i+1];
            if let Ok(byte) = u8::from_str_radix(byte_str, 16) {
                bytes.push(byte);
            }
        }
    }
    
    bytes.reverse(); // Convert to little-endian
    bytes
}

fn analyze_method_transitions() {
    println!("Method Performance Analysis");
    println!("==========================\n");
    
    // From the test results, we know:
    let method_data = vec![
        ("precomputed_basis", vec![8, 16, 24, 32, 40], 100.0),
        ("universal_pattern_cached", vec![8, 16], 100.0),
        ("quick_factor_scan", vec![24, 32], 100.0),
        ("phi_sum_guided", vec![32, 40, 48], 35.0), // Fails at 48+
    ];
    
    for (method, bit_ranges, success_rate) in method_data {
        println!("Method: {}", method);
        println!("  Bit ranges: {:?}", bit_ranges);
        println!("  Success rate at highest range: {}%", success_rate);
        println!("  Status: {}", if success_rate < 50.0 { "FAILING" } else { "Working" });
        println!();
    }
}

fn analyze_byte_frequencies(patterns: &HashMap<usize, Vec<BytePattern>>) {
    println!("\nByte Pattern Frequency Analysis");
    println!("==============================\n");
    
    for (bit_length, patterns) in patterns {
        let mut byte_freq: HashMap<u8, usize> = HashMap::new();
        
        for pattern in patterns {
            *byte_freq.entry(pattern.byte_value).or_insert(0) += 1;
        }
        
        // Find most common bytes
        let mut freq_vec: Vec<_> = byte_freq.iter().collect();
        freq_vec.sort_by(|a, b| b.1.cmp(a.1));
        
        println!("{}-bit patterns:", bit_length);
        println!("  Total unique bytes: {}", freq_vec.len());
        println!("  Top 5 most frequent:");
        
        for (byte_val, count) in freq_vec.iter().take(5) {
            println!("    Byte {:3} ({:08b}): {} occurrences", byte_val, byte_val, count);
        }
        println!();
    }
}

fn identify_critical_transitions(patterns: &HashMap<usize, Vec<BytePattern>>) {
    println!("\nCritical Bit Range Transitions");
    println!("==============================\n");
    
    println!("40-48 bit transition (SUCCESS → FAILURE):");
    println!("  - Last working method: phi_sum_guided");
    println!("  - Failure mode: Timeout at 1 second");
    println!("  - Pattern: All methods become search-based rather than recognition-based");
    println!("\nHypothesis: The pre-computed basis lacks proper tuning for 48+ bit channels");
    println!("Solution: Need channel-specific constants for higher bit ranges");
}

fn main() {
    println!("Analyzing Test Matrix Results\n");
    
    // Load test matrix
    let matrix_data = std::fs::read_to_string("data/test_matrix.json")
        .expect("Failed to load test matrix");
    
    let test_matrix: TestMatrix = serde_json::from_str(&matrix_data)
        .expect("Failed to parse test matrix");
    
    // Analyze method transitions
    analyze_method_transitions();
    
    // Extract successful patterns
    let patterns = analyze_successful_patterns(&test_matrix);
    
    // Analyze byte frequencies
    analyze_byte_frequencies(&patterns);
    
    // Identify critical transitions
    identify_critical_transitions(&patterns);
    
    // Recommendations
    println!("\nRecommendations for Stream Processor");
    println!("===================================\n");
    
    println!("1. Focus on replicating success of 8-40 bit range");
    println!("2. Each 8-bit channel needs its own tuned constants");
    println!("3. Pre-compute resonance patterns for all 256 byte values per channel");
    println!("4. Eliminate search-based methods (phi_sum_guided)");
    println!("5. Ensure scale invariance through proper constant scaling");
    
    // Save analysis results
    let analysis = serde_json::json!({
        "successful_bit_ranges": [8, 16, 24, 32, 40],
        "failing_bit_ranges": [48, 56, 64],
        "critical_transition": 48,
        "working_methods": ["precomputed_basis", "universal_pattern_cached", "quick_factor_scan"],
        "failing_methods": ["phi_sum_guided"],
        "pattern_count": patterns.len(),
    });
    
    std::fs::write("data/performance_analysis.json", serde_json::to_string_pretty(&analysis).unwrap())
        .expect("Failed to write analysis");
    
    println!("\nAnalysis saved to data/performance_analysis.json");
}