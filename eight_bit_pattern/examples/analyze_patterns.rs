//! Pattern analysis tool for understanding which bit patterns correlate with factors
//!
//! Run with: cargo run --example analyze_patterns

use eight_bit_pattern::{
    AutoTuner, TestCase, TunerParams, compute_basis, 
    recognize_factors_with_diagnostics, DiagnosticAggregator
};
use num_bigint::BigInt;
use std::fs;
use std::path::Path;

fn main() {
    println!("=== Pattern Analysis for The 8-Bit Pattern ===\n");
    
    // Load test cases
    let test_cases = load_test_cases();
    println!("Loaded {} test cases", test_cases.len());
    
    // Create basis with default parameters
    let params = TunerParams::default();
    let basis = compute_basis(128, &params);
    println!("Created basis with {} channels", basis.num_channels);
    
    // Create diagnostic aggregator
    let mut aggregator = DiagnosticAggregator::default();
    
    // Run factorization on each test case
    println!("\nAnalyzing patterns...");
    for (i, test_case) in test_cases.iter().enumerate() {
        if i % 10 == 0 {
            print!(".");
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
        
        let (result, diagnostics) = recognize_factors_with_diagnostics(
            &test_case.n,
            &basis,
            &params
        );
        
        // Add to aggregator
        aggregator.add(&diagnostics);
        
        // Verify if successful
        if let Some(factors) = result {
            let correct = (factors.p == test_case.p && factors.q == test_case.q) ||
                         (factors.p == test_case.q && factors.q == test_case.p);
            if !correct {
                println!("\nWARNING: Incorrect factors for {}", test_case.n);
            }
        }
    }
    
    println!("\n\n=== Analysis Results ===\n");
    
    // Overall statistics
    println!("Overall: {}", aggregator.overall_stats());
    
    // Success rate by bit size
    println!("\nSuccess rate by bit size:");
    let mut sizes: Vec<_> = aggregator.success_by_size.iter().collect();
    sizes.sort_by_key(|(size, _)| *size);
    for (size, (success, total)) in sizes {
        let rate = if *total > 0 {
            *success as f64 / *total as f64 * 100.0
        } else {
            0.0
        };
        println!("  {:3}-bit: {:3}/{:3} ({:5.1}%)", size, success, total, rate);
    }
    
    // Most common patterns globally
    println!("\nMost common patterns at peaks:");
    let mut patterns: Vec<_> = aggregator.global_pattern_frequency.iter().collect();
    patterns.sort_by(|a, b| b.1.cmp(a.1));
    for (pattern, count) in patterns.iter().take(10) {
        println!("  Pattern {:08b} ({}): {} occurrences", pattern, pattern, count);
        print_active_constants(**pattern);
    }
    
    // Patterns most correlated with success
    println!("\nPatterns most correlated with success:");
    let success_patterns = aggregator.success_patterns();
    for (pattern, success_rate) in success_patterns.iter().take(10) {
        let total = aggregator.global_pattern_frequency.get(pattern).unwrap_or(&0);
        let success = aggregator.success_pattern_frequency.get(pattern).unwrap_or(&0);
        println!(
            "  Pattern {:08b} ({}): {:.1}% success rate ({}/{})",
            pattern, pattern, success_rate * 100.0, success, total
        );
        print_active_constants(*pattern);
    }
    
    // Analyze which constants are most active in successful patterns
    println!("\nConstant activation in successful patterns:");
    let mut constant_success = [0usize; 8];
    let mut constant_total = [0usize; 8];
    
    for (pattern, count) in &aggregator.success_pattern_frequency {
        for bit in 0..8 {
            if (pattern >> bit) & 1 == 1 {
                constant_success[bit] += count;
            }
        }
    }
    
    for (pattern, count) in &aggregator.global_pattern_frequency {
        for bit in 0..8 {
            if (pattern >> bit) & 1 == 1 {
                constant_total[bit] += count;
            }
        }
    }
    
    let constant_names = ["unity", "tau", "phi", "epsilon", "delta", "gamma", "beta", "alpha"];
    for (i, name) in constant_names.iter().enumerate() {
        let rate = if constant_total[i] > 0 {
            constant_success[i] as f64 / constant_total[i] as f64 * 100.0
        } else {
            0.0
        };
        println!("  Bit {} ({}): {:.1}% success when active", i, name, rate);
    }
    
    // Recommendations
    println!("\n=== Tuning Recommendations ===\n");
    
    if aggregator.total_success == 0 {
        println!("No successful factorizations found. Consider:");
        println!("- Adjusting initial constant values");
        println!("- Lowering alignment threshold");
        println!("- Modifying resonance scaling");
    } else {
        println!("Based on pattern analysis:");
        
        // Find most successful pattern
        if let Some((pattern, _)) = success_patterns.first() {
            println!("- Focus on pattern {:08b} which has highest success rate", pattern);
            println!("- This pattern activates:");
            print_active_constants(*pattern);
        }
        
        // Recommend constant adjustments
        println!("\n- Consider increasing weights for constants with high success rates");
        println!("- Consider decreasing weights for constants rarely in successful patterns");
    }
}

fn print_active_constants(pattern: u8) {
    let constant_names = ["unity", "tau", "phi", "epsilon", "delta", "gamma", "beta", "alpha"];
    let active: Vec<_> = (0..8)
        .filter(|&i| (pattern >> i) & 1 == 1)
        .map(|i| constant_names[i])
        .collect();
    
    if !active.is_empty() {
        println!("    Active: {}", active.join(", "));
    }
}

fn load_test_cases() -> Vec<TestCase> {
    // Try to load from test_matrix.json
    let path = Path::new("../data/test_matrix.json");
    
    if path.exists() {
        // Load and parse test matrix
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
        
        // Limit for analysis
        test_cases.truncate(100);
        test_cases
    } else {
        // Use default test cases
        vec![
            TestCase { n: BigInt::from(15), p: BigInt::from(3), q: BigInt::from(5), bit_length: 4 },
            TestCase { n: BigInt::from(21), p: BigInt::from(3), q: BigInt::from(7), bit_length: 5 },
            TestCase { n: BigInt::from(35), p: BigInt::from(5), q: BigInt::from(7), bit_length: 6 },
            TestCase { n: BigInt::from(77), p: BigInt::from(7), q: BigInt::from(11), bit_length: 7 },
            TestCase { n: BigInt::from(143), p: BigInt::from(11), q: BigInt::from(13), bit_length: 8 },
            TestCase { n: BigInt::from(221), p: BigInt::from(13), q: BigInt::from(17), bit_length: 8 },
            TestCase { n: BigInt::from(323), p: BigInt::from(17), q: BigInt::from(19), bit_length: 9 },
            TestCase { n: BigInt::from(437), p: BigInt::from(19), q: BigInt::from(23), bit_length: 9 },
        ]
    }
}