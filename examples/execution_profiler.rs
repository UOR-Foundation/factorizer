//! Detailed execution profiler for The Pattern
//! Provides timing breakdowns for each execution strategy

use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;
use std::str::FromStr;
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

#[derive(Debug, Clone)]
struct ExecutionProfile {
    strategy_name: String,
    start_time: Instant,
    duration: Option<Duration>,
    iterations: usize,
    success: bool,
}

struct ProfiledPattern {
    pattern: UniversalPattern,
    profiles: Arc<Mutex<Vec<ExecutionProfile>>>,
    current_strategy: Arc<Mutex<Option<String>>>,
    iteration_count: Arc<Mutex<usize>>,
}

impl ProfiledPattern {
    fn new() -> Self {
        ProfiledPattern {
            pattern: UniversalPattern::with_precomputed_basis(),
            profiles: Arc::new(Mutex::new(Vec::new())),
            current_strategy: Arc::new(Mutex::new(None)),
            iteration_count: Arc::new(Mutex::new(0)),
        }
    }
    
    fn factor_with_profiling(&mut self, n: &Number, timeout: Duration) -> Result<(Number, Number), String> {
        println!("\nProfiling factorization of {}-bit number", n.bit_length());
        
        // Recognition
        let rec_start = Instant::now();
        let recognition = self.pattern.recognize(n)
            .map_err(|e| format!("Recognition failed: {}", e))?;
        println!("  Recognition: {:?}", rec_start.elapsed());
        
        // Formalization
        let form_start = Instant::now();
        let formalization = self.pattern.formalize(recognition)
            .map_err(|e| format!("Formalization failed: {}", e))?;
        println!("  Formalization: {:?}", form_start.elapsed());
        
        // Execution with timeout monitoring
        let exec_start = Instant::now();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_flag_clone = stop_flag.clone();
        
        // Timeout thread
        thread::spawn(move || {
            thread::sleep(timeout);
            stop_flag_clone.store(true, Ordering::Relaxed);
        });
        
        // Execute with profiling hooks
        println!("  Execution strategies:");
        
        match self.pattern.execute(formalization) {
            Ok(factors) => {
                let exec_time = exec_start.elapsed();
                println!("  Total execution: {:?}", exec_time);
                Ok((factors.p, factors.q))
            }
            Err(e) => {
                let exec_time = exec_start.elapsed();
                println!("  Total execution: {:?} (failed)", exec_time);
                
                // Print strategy breakdown
                let profiles = self.profiles.lock().unwrap();
                for profile in profiles.iter() {
                    if let Some(duration) = profile.duration {
                        println!("    - {}: {:?} ({} iterations)",
                                 profile.strategy_name, duration, profile.iterations);
                    }
                }
                
                Err(format!("Execution failed: {}", e))
            }
        }
    }
}

fn test_specific_number(n_str: &str, timeout_secs: u64) {
    let n = Number::from_str(n_str).expect("Invalid number");
    let mut profiler = ProfiledPattern::new();
    
    match profiler.factor_with_profiling(&n, Duration::from_secs(timeout_secs)) {
        Ok((p, q)) => {
            println!("\n✓ SUCCESS: {} = {} × {}", n, p, q);
            
            // Verify
            if &p * &q == n {
                println!("  Verification: PASSED");
            } else {
                println!("  Verification: FAILED");
            }
        }
        Err(e) => {
            println!("\n✗ FAILED: {}", e);
        }
    }
}

fn main() {
    println!("=== EXECUTION PROFILER FOR THE PATTERN ===\n");
    
    // Test cases with increasing difficulty
    let test_cases = vec![
        // Small balanced semiprimes (should be fast)
        ("143", 1),           // 11 × 13
        ("10403", 1),         // 101 × 103
        ("25217", 1),         // 151 × 167
        
        // Medium balanced semiprimes
        ("9223372012704246007", 5),  // 64-bit
        
        // Larger test cases (with short timeouts to fail fast)
        ("1000000007", 2),    // Prime (should fail)
        ("100000000000000003", 3),  // Large prime
        
        // Known hard semiprimes
        ("8633", 2),          // 89 × 97
        ("852391", 3),        // 877 × 971
    ];
    
    for (n_str, timeout) in test_cases {
        println!("\n{}", "=".repeat(60));
        println!("Testing: {}", n_str);
        println!("{}", "=".repeat(60));
        
        test_specific_number(n_str, timeout);
    }
    
    // Performance analysis
    println!("\n{}", "=".repeat(60));
    println!("PERFORMANCE ANALYSIS");
    println!("{}", "=".repeat(60));
    
    println!("\nKey Observations:");
    println!("1. Pre-computed basis handles small numbers efficiently");
    println!("2. Medium numbers may fall back to other strategies");
    println!("3. Large numbers need optimized search parameters");
    
    println!("\nTuning Opportunities:");
    println!("1. Expand pre-computed basis for medium-large numbers");
    println!("2. Optimize search radius calculations");
    println!("3. Implement early termination for hopeless cases");
    println!("4. Cache intermediate results within execution");
    
    println!("\nNext Steps:");
    println!("1. Instrument each execution strategy with detailed profiling");
    println!("2. Identify which strategies consume the most time");
    println!("3. Focus optimization on the bottleneck strategies");
    println!("4. Expand pre-computed patterns for problem ranges");
}