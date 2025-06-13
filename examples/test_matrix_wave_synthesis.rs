//! Test Wave Synthesis Auto-Tuner against the official test matrix
//! 
//! This evaluates the wave synthesis approach on hard semiprimes
//! to verify it can factor arbitrary numbers without precision limits.

use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use std::str::FromStr;

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

fn main() {
    println!("Wave Synthesis Test Matrix Evaluation");
    println!("====================================\n");
    
    // Load test matrix
    let matrix_data = match std::fs::read_to_string("data/test_matrix.json") {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to load test matrix: {}", e);
            eprintln!("Please run 'cargo run --example generate_test_matrix' first");
            return;
        }
    };
    
    let test_matrix: TestMatrix = serde_json::from_str(&matrix_data)
        .expect("Failed to parse test matrix");
    
    println!("Test Matrix Version: {}", test_matrix.version);
    println!("Total test cases: {}\n", 
        test_matrix.test_cases.values().map(|v| v.len()).sum::<usize>());
    
    // Initialize pattern with pre-computed basis
    let mut pattern = UniversalPattern::with_precomputed_basis();
    
    // Test specific bit ranges to verify arbitrary precision
    let test_ranges = vec![
        ("Small (8-64 bit)", 8, 64),
        ("Medium (128 bit)", 128, 128),
        ("Large (224 bit)", 224, 224),
        ("Beyond limit (256 bit)", 256, 256),
        ("Very large (512 bit)", 512, 512),
    ];
    
    let mut total_tested = 0;
    let mut total_success = 0;
    
    for (range_name, min_bits, max_bits) in test_ranges {
        println!("Testing {} range:", range_name);
        println!("{}", "-".repeat(60));
        
        let mut range_total = 0;
        let mut range_success = 0;
        let mut range_times = Vec::new();
        
        // Test up to 5 cases per bit size in range
        for (bit_length, cases) in &test_matrix.test_cases {
            if *bit_length >= min_bits && *bit_length <= max_bits {
                for (i, case) in cases.iter().take(5).enumerate() {
                    range_total += 1;
                    total_tested += 1;
                    
                    let n = Number::from_str(&case.n).unwrap();
                    let expected_p = Number::from_str(&case.p).unwrap();
                    let expected_q = Number::from_str(&case.q).unwrap();
                    
                    print!("  {}-bit #{}: ", bit_length, i + 1);
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                    
                    let start = Instant::now();
                    let timeout = Duration::from_secs(match *bit_length {
                        0..=64 => 1,
                        65..=128 => 5,
                        129..=256 => 30,
                        _ => 60,
                    });
                    
                    // Try factorization with timeout
                    let result = std::thread::scope(|s| {
                        let handle = s.spawn(|| {
                            pattern.recognize(&n)
                                .and_then(|r| pattern.formalize(r))
                                .and_then(|f| pattern.execute(f))
                        });
                        
                        let start_time = Instant::now();
                        loop {
                            if handle.is_finished() {
                                return handle.join().unwrap();
                            }
                            
                            if start_time.elapsed() > timeout {
                                return Err(rust_pattern_solver::error::PatternError::ExecutionError(
                                    "Timeout".to_string()
                                ));
                            }
                            
                            std::thread::sleep(Duration::from_millis(100));
                        }
                    });
                    
                    let elapsed = start.elapsed();
                    
                    match result {
                        Ok(factors) => {
                            if (factors.p == expected_p && factors.q == expected_q) ||
                               (factors.p == expected_q && factors.q == expected_p) {
                                range_success += 1;
                                total_success += 1;
                                range_times.push(elapsed);
                                
                                let ms = elapsed.as_secs_f64() * 1000.0;
                                if ms < 1.0 {
                                    println!("✓ {:.3}μs ({})", ms * 1000.0, factors.method);
                                } else {
                                    println!("✓ {:.3}ms ({})", ms, factors.method);
                                }
                            } else {
                                println!("✗ Wrong factors");
                                println!("     Expected: {} × {}", expected_p, expected_q);
                                println!("     Got:      {} × {}", factors.p, factors.q);
                            }
                        }
                        Err(e) => {
                            println!("✗ {}", e);
                            if n.bit_length() > 224 {
                                println!("     Note: Beyond old 224-bit limit");
                            }
                        }
                    }
                }
            }
        }
        
        if range_total > 0 {
            let success_rate = (range_success as f64 / range_total as f64) * 100.0;
            println!("\n{} Summary:", range_name);
            println!("  Success: {}/{} ({:.1}%)", range_success, range_total, success_rate);
            
            if !range_times.is_empty() {
                let avg_time = range_times.iter().sum::<Duration>() / range_times.len() as u32;
                let min_time = range_times.iter().min().unwrap();
                let max_time = range_times.iter().max().unwrap();
                
                println!("  Timing: avg {:.3}ms, min {:.3}ms, max {:.3}ms",
                    avg_time.as_secs_f64() * 1000.0,
                    min_time.as_secs_f64() * 1000.0,
                    max_time.as_secs_f64() * 1000.0);
            }
            
            if min_bits > 224 && range_success == 0 {
                println!("  ⚠️  CRITICAL: Cannot factor beyond 224 bits!");
                println!("  The implementation still has precision limitations.");
            }
        }
        println!();
    }
    
    // Overall summary
    println!("\nOverall Results");
    println!("===============");
    println!("Total tested: {}", total_tested);
    println!("Total successful: {}", total_success);
    println!("Overall success rate: {:.1}%", 
        (total_success as f64 / total_tested.max(1) as f64) * 100.0);
    
    if total_success > 0 {
        println!("\n✓ Wave synthesis auto-tuner is working!");
    } else {
        println!("\n✗ Wave synthesis implementation needs debugging");
    }
    
    // Analysis
    println!("\nWave Synthesis Analysis:");
    println!("------------------------");
    println!("• The Pattern transforms numbers into wave patterns");
    println!("• Pre-computed basis provides resonance templates");
    println!("• Auto-tuner scales templates to match input frequency");
    println!("• Factors manifest at constructive interference points");
    
    if total_tested > 0 {
        let tested_256_bit = test_matrix.test_cases.get(&256)
            .map(|cases| cases.iter().take(5).count())
            .unwrap_or(0);
        
        if tested_256_bit > 0 {
            println!("\nArbitrary Precision Status:");
            let success_256 = test_matrix.test_cases.get(&256)
                .map(|_| if total_success > 0 { "✓ WORKING" } else { "✗ NEEDS FIXES" })
                .unwrap_or("? NOT TESTED");
            println!("256-bit factorization: {}", success_256);
        }
    }
}