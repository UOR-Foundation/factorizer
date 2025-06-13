//! Test matrix evaluation framework
//! 
//! This evaluates The Pattern implementation against the authoritative test matrix
//! without using the known factors except for verification.

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

#[derive(Debug, Clone)]
struct TestResult {
    bit_length: usize,
    success: bool,
    time: Duration,
    method: Option<String>,
    balanced: bool,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct EvaluationSummary {
    bit_length: usize,
    total_cases: usize,
    successful: usize,
    failed: usize,
    success_rate: f64,
    #[serde(serialize_with = "serialize_duration")]
    avg_time: Duration,
    #[serde(serialize_with = "serialize_duration")]
    min_time: Duration,
    #[serde(serialize_with = "serialize_duration")]
    max_time: Duration,
    balanced_success_rate: f64,
    unbalanced_success_rate: f64,
}

fn serialize_duration<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_f64(duration.as_secs_f64())
}

fn evaluate_test_case(pattern: &mut UniversalPattern, test_case: &TestCase, timeout: Duration) -> TestResult {
    let n = Number::from_str(&test_case.n).expect("Invalid number in test case");
    
    let start = Instant::now();
    
    // Run with timeout
    let result = std::thread::scope(|s| {
        let handle = s.spawn(|| {
            // Recognition
            let recognition = pattern.recognize(&n)?;
            
            // Formalization
            let formalization = pattern.formalize(recognition)?;
            
            // Execution
            pattern.execute(formalization)
        });
        
        // Wait with timeout
        loop {
            if handle.is_finished() {
                return handle.join().unwrap();
            }
            
            std::thread::sleep(Duration::from_millis(100));
            let elapsed = start.elapsed();
            
            if elapsed > timeout {
                return Err(rust_pattern_solver::error::PatternError::ExecutionError(
                    "Timeout".to_string()
                ));
            }
        }
    });
    
    let time = start.elapsed();
    
    match result {
        Ok(factors) => {
            // Verify without using the known factors during computation
            let expected_p = Number::from_str(&test_case.p).unwrap();
            let expected_q = Number::from_str(&test_case.q).unwrap();
            
            let success = (factors.p == expected_p && factors.q == expected_q) ||
                         (factors.p == expected_q && factors.q == expected_p);
            
            TestResult {
                bit_length: test_case.bit_length,
                success,
                time,
                method: Some(factors.method.clone()),
                balanced: test_case.balanced,
                error: if !success { Some("Incorrect factors".to_string()) } else { None },
            }
        }
        Err(e) => TestResult {
            bit_length: test_case.bit_length,
            success: false,
            time,
            method: None,
            balanced: test_case.balanced,
            error: Some(e.to_string()),
        }
    }
}

fn summarize_results(bit_length: usize, results: &[TestResult]) -> EvaluationSummary {
    let total_cases = results.len();
    let successful = results.iter().filter(|r| r.success).count();
    let failed = total_cases - successful;
    
    let success_rate = if total_cases > 0 {
        successful as f64 / total_cases as f64
    } else {
        0.0
    };
    
    let times: Vec<Duration> = results.iter()
        .filter(|r| r.success)
        .map(|r| r.time)
        .collect();
    
    let avg_time = if !times.is_empty() {
        times.iter().sum::<Duration>() / times.len() as u32
    } else {
        Duration::from_secs(0)
    };
    
    let min_time = times.iter().min().copied().unwrap_or(Duration::from_secs(0));
    let max_time = times.iter().max().copied().unwrap_or(Duration::from_secs(0));
    
    // Calculate balanced vs unbalanced success rates
    let balanced_results: Vec<&TestResult> = results.iter().filter(|r| r.balanced).collect();
    let balanced_success = balanced_results.iter().filter(|r| r.success).count();
    let balanced_success_rate = if !balanced_results.is_empty() {
        balanced_success as f64 / balanced_results.len() as f64
    } else {
        0.0
    };
    
    let unbalanced_results: Vec<&TestResult> = results.iter().filter(|r| !r.balanced).collect();
    let unbalanced_success = unbalanced_results.iter().filter(|r| r.success).count();
    let unbalanced_success_rate = if !unbalanced_results.is_empty() {
        unbalanced_success as f64 / unbalanced_results.len() as f64
    } else {
        0.0
    };
    
    EvaluationSummary {
        bit_length,
        total_cases,
        successful,
        failed,
        success_rate,
        avg_time,
        min_time,
        max_time,
        balanced_success_rate,
        unbalanced_success_rate,
    }
}

fn main() {
    println!("The Pattern Test Matrix Evaluation");
    println!("==================================\n");
    
    // Load test matrix
    let matrix_data = std::fs::read_to_string("data/test_matrix.json")
        .expect("Failed to load test matrix. Run generate_test_matrix first.");
    
    let test_matrix: TestMatrix = serde_json::from_str(&matrix_data)
        .expect("Failed to parse test matrix");
    
    println!("Test Matrix Version: {}", test_matrix.version);
    println!("Generated: {}", test_matrix.generated);
    println!();
    
    // Initialize pattern with pre-computed basis
    let mut pattern = UniversalPattern::with_precomputed_basis();
    
    // Set timeout based on bit length
    let get_timeout = |bits: usize| -> Duration {
        match bits {
            0..=64 => Duration::from_secs(1),
            65..=128 => Duration::from_secs(5),
            129..=256 => Duration::from_secs(30),
            257..=512 => Duration::from_secs(120),
            _ => Duration::from_secs(300),
        }
    };
    
    let mut all_summaries = Vec::new();
    
    // Evaluate each bit length
    for (bit_length, test_cases) in &test_matrix.test_cases {
        println!("Evaluating {}-bit test cases ({} cases)...", bit_length, test_cases.len());
        
        let timeout = get_timeout(*bit_length);
        let mut results = Vec::new();
        
        for (i, test_case) in test_cases.iter().enumerate() {
            print!("  Case {}/{}: ", i + 1, test_cases.len());
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
            
            let result = evaluate_test_case(&mut pattern, test_case, timeout);
            
            if result.success {
                let time_ms = result.time.as_secs_f64() * 1000.0;
                if time_ms < 1.0 {
                    println!("✓ ({:.3}μs, {})", time_ms * 1000.0, result.method.as_ref().unwrap());
                } else if time_ms < 1000.0 {
                    println!("✓ ({:.3}ms, {})", time_ms, result.method.as_ref().unwrap());
                } else {
                    println!("✓ ({:.3}s, {})", result.time.as_secs_f64(), result.method.as_ref().unwrap());
                }
            } else {
                println!("✗ ({})", result.error.as_ref().unwrap());
            }
            
            results.push(result);
        }
        
        let summary = summarize_results(*bit_length, &results);
        println!("\n{}-bit Summary:", bit_length);
        println!("  Success rate: {:.1}%", summary.success_rate * 100.0);
        println!("  Balanced: {:.1}%", summary.balanced_success_rate * 100.0);
        println!("  Unbalanced: {:.1}%", summary.unbalanced_success_rate * 100.0);
        let avg_ms = summary.avg_time.as_secs_f64() * 1000.0;
        let min_ms = summary.min_time.as_secs_f64() * 1000.0;
        let max_ms = summary.max_time.as_secs_f64() * 1000.0;
        
        if avg_ms < 1.0 {
            println!("  Avg time: {:.3}μs", avg_ms * 1000.0);
            println!("  Time range: {:.3}μs - {:.3}μs", min_ms * 1000.0, max_ms * 1000.0);
        } else if avg_ms < 1000.0 {
            println!("  Avg time: {:.3}ms", avg_ms);
            println!("  Time range: {:.3}ms - {:.3}ms", min_ms, max_ms);
        } else {
            println!("  Avg time: {:.3}s", summary.avg_time.as_secs_f64());
            println!("  Time range: {:.3}s - {:.3}s", 
                     summary.min_time.as_secs_f64(), 
                     summary.max_time.as_secs_f64());
        }
        println!();
        
        all_summaries.push(summary);
    }
    
    // Overall summary
    println!("\nOverall Results");
    println!("===============");
    
    let total_cases: usize = all_summaries.iter().map(|s| s.total_cases).sum();
    let total_success: usize = all_summaries.iter().map(|s| s.successful).sum();
    let overall_rate = total_success as f64 / total_cases as f64;
    
    println!("Total test cases: {}", total_cases);
    println!("Total successful: {}", total_success);
    println!("Overall success rate: {:.1}%", overall_rate * 100.0);
    
    // Performance by bit range
    println!("\nPerformance by Bit Range:");
    println!("Bits     | Success | Balanced | Unbalanced | Avg Time");
    println!("---------|---------|----------|------------|----------");
    
    for summary in &all_summaries {
        let avg_ms = summary.avg_time.as_secs_f64() * 1000.0;
        let time_str = if avg_ms < 1.0 {
            format!("{:7.3}μs", avg_ms * 1000.0)
        } else if avg_ms < 1000.0 {
            format!("{:7.3}ms", avg_ms)
        } else {
            format!("{:7.3}s", summary.avg_time.as_secs_f64())
        };
        
        println!("{:8} | {:6.1}% | {:7.1}% | {:9.1}% | {}",
                 summary.bit_length,
                 summary.success_rate * 100.0,
                 summary.balanced_success_rate * 100.0,
                 summary.unbalanced_success_rate * 100.0,
                 time_str);
    }
    
    // Save results
    let results_json = serde_json::to_string_pretty(&all_summaries).unwrap();
    std::fs::write("data/test_matrix_results.json", results_json).unwrap();
    
    println!("\nResults saved to data/test_matrix_results.json");
}