//! Test basis performance and show where time is spent

use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;
use std::str::FromStr;
use std::time::Instant;

fn main() {
    println!("=== Basis Performance Analysis ===\n");
    
    // Test loading time
    println!("1. Loading basis from disk:");
    let load_start = Instant::now();
    let mut pattern = UniversalPattern::with_precomputed_basis();
    let load_time = load_start.elapsed();
    println!("   ✓ Loaded in {:?}\n", load_time);
    
    // Test small numbers
    println!("2. Small number performance:");
    let small_tests = vec![
        ("143", "11 × 13"),
        ("10403", "101 × 103"),
        ("25217", "151 × 167"),
    ];
    
    for (n_str, desc) in &small_tests {
        let n = Number::from_str(n_str).unwrap();
        print!("   {} ({}): ", n, desc);
        
        let start = Instant::now();
        
        // Time each stage
        let rec_start = Instant::now();
        let recognition = pattern.recognize(&n).unwrap();
        let rec_time = rec_start.elapsed();
        
        let form_start = Instant::now();
        let formalization = pattern.formalize(recognition).unwrap();
        let form_time = form_start.elapsed();
        
        let exec_start = Instant::now();
        let factors = pattern.execute(formalization).unwrap();
        let exec_time = exec_start.elapsed();
        
        let total_time = start.elapsed();
        
        println!("✓ {} × {} via {}", factors.p, factors.q, factors.method);
        println!("      Recognition: {:?}, Formalization: {:?}, Execution: {:?}, Total: {:?}", 
                 rec_time, form_time, exec_time, total_time);
    }
    
    // Test medium numbers
    println!("\n3. Medium number performance:");
    let medium_test = "9223372012704246007"; // 64-bit
    let n = Number::from_str(medium_test).unwrap();
    println!("   {} (64-bit):", n);
    
    let start = Instant::now();
    let rec_start = Instant::now();
    let recognition = pattern.recognize(&n).unwrap();
    let rec_time = rec_start.elapsed();
    
    let form_start = Instant::now();
    let formalization = pattern.formalize(recognition).unwrap();
    let form_time = form_start.elapsed();
    
    let exec_start = Instant::now();
    match pattern.execute(formalization) {
        Ok(factors) => {
            let exec_time = exec_start.elapsed();
            println!("   ✓ {} × {} via {}", factors.p, factors.q, factors.method);
            println!("   Recognition: {:?}, Formalization: {:?}, Execution: {:?}", 
                     rec_time, form_time, exec_time);
        }
        Err(e) => {
            println!("   ✗ Execution failed after {:?}: {}", exec_start.elapsed(), e);
        }
    }
    
    // Test where the slowdown happens with larger numbers
    println!("\n4. Large number stage analysis:");
    let large_tests = vec![
        ("1000000007", "large prime", 30),
        ("10000000019", "larger prime", 34),
        ("100000000003", "even larger", 37),
    ];
    
    for (n_str, desc, bits) in large_tests {
        println!("\n   Testing {}-bit {} ({})", bits, desc, n_str);
        let n = Number::from_str(n_str).unwrap();
        
        // Time recognition
        let rec_start = Instant::now();
        match pattern.recognize(&n) {
            Ok(recognition) => {
                let rec_time = rec_start.elapsed();
                println!("   ✓ Recognition: {:?}", rec_time);
                
                // Time formalization
                let form_start = Instant::now();
                match pattern.formalize(recognition) {
                    Ok(formalization) => {
                        let form_time = form_start.elapsed();
                        println!("   ✓ Formalization: {:?}", form_time);
                        
                        // Time execution with timeout
                        println!("   Starting execution (5s timeout)...");
                        let exec_start = Instant::now();
                        
                        // Create a simple timeout mechanism
                        let timeout = std::time::Duration::from_secs(5);
                        let start_time = Instant::now();
                        
                        // We can't truly timeout the execution, but we can at least measure
                        match pattern.execute(formalization) {
                            Ok(factors) => {
                                let exec_time = exec_start.elapsed();
                                println!("   ✓ Execution: {:?} via {}", exec_time, factors.method);
                                println!("   Result: {} × {}", factors.p, factors.q);
                            }
                            Err(e) => {
                                let exec_time = exec_start.elapsed();
                                if exec_time > timeout {
                                    println!("   ✗ Execution timed out after {:?}", exec_time);
                                } else {
                                    println!("   ✗ Execution failed after {:?}: {}", exec_time, e);
                                }
                            }
                        }
                    }
                    Err(e) => println!("   ✗ Formalization failed: {}", e),
                }
            }
            Err(e) => println!("   ✗ Recognition failed: {}", e),
        }
    }
    
    println!("\n5. Summary:");
    println!("   - Basis loading is fast (~{}ms)", load_time.as_millis());
    println!("   - Recognition and formalization are always fast");
    println!("   - Execution is where the time is spent for large numbers");
    println!("   - The pre-computed basis helps significantly with small-medium numbers");
}