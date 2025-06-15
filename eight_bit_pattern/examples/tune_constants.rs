//! Tune the 8 fundamental constants using pattern frequency analysis
//!
//! Run with: cargo run --example tune_constants

use eight_bit_pattern::{
    TuningConfig, ConstantTuner, TestCase
};
use num_bigint::BigInt;
use std::fs;
use std::path::Path;

fn main() {
    println!("=== Constant Tuning for The 8-Bit Pattern ===\n");
    
    // Load test cases
    let mut test_cases = load_test_cases();
    // Filter to only small numbers for faster tuning
    test_cases.retain(|case| case.bit_length <= 32);
    println!("Loaded {} test cases (filtered to ≤32 bits)", test_cases.len());
    
    // Create tuning configuration
    let mut config = TuningConfig::default();
    config.iterations = 20;  // Reduced for faster testing
    config.learning_rate = 0.01;  // More aggressive learning rate
    config.batch_size = 10;  // Smaller batch for faster evaluation
    config.target_success_rate = 0.50;  // More realistic target
    
    println!("\nTuning configuration:");
    println!("  Iterations: {}", config.iterations);
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Batch size: {}", config.batch_size);
    println!("  Target success rate: {:.0}%", config.target_success_rate * 100.0);
    
    // Create and run tuner
    let mut tuner = ConstantTuner::new(config, test_cases);
    let result = tuner.tune();
    
    // Print results
    result.print_summary();
    
    // Show comparison with initial constants
    println!("\n=== Constant Value Changes ===");
    let initial = [1.0, 1.839287, 1.618034, 0.5, 0.159155, 6.283185, 0.199612, 14.134725];
    let names = ["unity", "tau", "phi", "epsilon", "delta", "gamma", "beta", "alpha"];
    
    for i in 0..8 {
        let initial_val = initial[i];
        let tuned_val = result.best_constants.values[i];
        let change = ((tuned_val - initial_val) / initial_val) * 100.0;
        
        println!("  {} ({}): {:.6} → {:.6} ({:+.1}%)", 
            names[i], i, initial_val, tuned_val, change);
    }
    
    // Suggest next steps
    println!("\n=== Recommendations ===");
    if result.final_success_rate >= 0.80 {
        println!("✓ Target success rate achieved!");
        println!("  Consider:");
        println!("  - Testing on larger numbers");
        println!("  - Fine-tuning with smaller learning rate");
        println!("  - Implementing ensemble methods");
    } else {
        println!("✗ Target success rate not achieved");
        println!("  Consider:");
        println!("  - Increasing iterations");
        println!("  - Adjusting learning rate");
        println!("  - Modifying the resonance calculation");
        println!("  - Improving factor extraction logic");
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
        
        test_cases
    } else {
        // Use default test cases
        println!("Warning: test_matrix.json not found, using default test cases");
        vec![
            TestCase { n: BigInt::from(15), p: BigInt::from(3), q: BigInt::from(5), bit_length: 4 },
            TestCase { n: BigInt::from(21), p: BigInt::from(3), q: BigInt::from(7), bit_length: 5 },
            TestCase { n: BigInt::from(35), p: BigInt::from(5), q: BigInt::from(7), bit_length: 6 },
            TestCase { n: BigInt::from(77), p: BigInt::from(7), q: BigInt::from(11), bit_length: 7 },
            TestCase { n: BigInt::from(143), p: BigInt::from(11), q: BigInt::from(13), bit_length: 8 },
            TestCase { n: BigInt::from(221), p: BigInt::from(13), q: BigInt::from(17), bit_length: 8 },
            TestCase { n: BigInt::from(323), p: BigInt::from(17), q: BigInt::from(19), bit_length: 9 },
            TestCase { n: BigInt::from(437), p: BigInt::from(19), q: BigInt::from(23), bit_length: 9 },
            TestCase { n: BigInt::from(667), p: BigInt::from(23), q: BigInt::from(29), bit_length: 10 },
            TestCase { n: BigInt::from(899), p: BigInt::from(29), q: BigInt::from(31), bit_length: 10 },
        ]
    }
}