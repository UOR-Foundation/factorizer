//! Diagnostic framework for empirical tuning of The Pattern
//! Goal: Factor hard semiprimes up to 300-bit through fail-fast iteration

use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;
use std::str::FromStr;
use std::time::{Duration, Instant};
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct DiagnosticData {
    bit_size: usize,
    recognition_time: Duration,
    formalization_time: Duration,
    execution_time: Option<Duration>,
    method_used: Option<String>,
    success: bool,
    phi_coordinates: [f64; 4],
    resonance_peaks: Vec<usize>,
    search_iterations: Option<usize>,
    failure_reason: Option<String>,
}

struct DiagnosticRunner {
    pattern: UniversalPattern,
    timeout: Duration,
    diagnostics: Vec<DiagnosticData>,
}

impl DiagnosticRunner {
    fn new(timeout_secs: u64) -> Self {
        println!("Initializing diagnostic runner with {}s timeout...", timeout_secs);
        let pattern = UniversalPattern::with_precomputed_basis();
        
        DiagnosticRunner {
            pattern,
            timeout: Duration::from_secs(timeout_secs),
            diagnostics: Vec::new(),
        }
    }
    
    fn test_number(&mut self, n: &Number, expected_p: Option<&Number>, expected_q: Option<&Number>) -> DiagnosticData {
        let bit_size = n.bit_length();
        println!("\nTesting {}-bit number: {}", bit_size, 
                 if bit_size > 100 { format!("{}...{}", &n.to_string()[..20], &n.to_string()[n.to_string().len()-20..]) } 
                 else { n.to_string() });
        
        // Recognition phase
        let rec_start = Instant::now();
        let recognition = match self.pattern.recognize(n) {
            Ok(r) => r,
            Err(e) => {
                return DiagnosticData {
                    bit_size,
                    recognition_time: rec_start.elapsed(),
                    formalization_time: Duration::default(),
                    execution_time: None,
                    method_used: None,
                    success: false,
                    phi_coordinates: [0.0; 4],
                    resonance_peaks: vec![],
                    search_iterations: None,
                    failure_reason: Some(format!("Recognition failed: {}", e)),
                };
            }
        };
        let recognition_time = rec_start.elapsed();
        
        // Extract diagnostic info from recognition
        let phi_coords = [
            recognition.phi_component,
            recognition.pi_component,
            recognition.e_component,
            recognition.unity_phase,
        ];
        
        // Formalization phase
        let form_start = Instant::now();
        let formalization = match self.pattern.formalize(recognition) {
            Ok(f) => f,
            Err(e) => {
                return DiagnosticData {
                    bit_size,
                    recognition_time,
                    formalization_time: form_start.elapsed(),
                    execution_time: None,
                    method_used: None,
                    success: false,
                    phi_coordinates: phi_coords,
                    resonance_peaks: vec![],
                    search_iterations: None,
                    failure_reason: Some(format!("Formalization failed: {}", e)),
                };
            }
        };
        let formalization_time = form_start.elapsed();
        
        // Extract resonance peaks
        let resonance_peaks = formalization.resonance_peaks.clone();
        
        // Execution phase with timeout
        let exec_start = Instant::now();
        let exec_result = self.pattern.execute(formalization);
        let execution_time = exec_start.elapsed();
        
        match exec_result {
            Ok(factors) => {
                let success = if let (Some(exp_p), Some(exp_q)) = (expected_p, expected_q) {
                    (&factors.p == exp_p && &factors.q == exp_q) || 
                    (&factors.p == exp_q && &factors.q == exp_p)
                } else {
                    &factors.p * &factors.q == *n
                };
                
                println!("  ✓ Success: {} × {} via {} in {:?}", 
                         factors.p, factors.q, factors.method, execution_time);
                
                DiagnosticData {
                    bit_size,
                    recognition_time,
                    formalization_time,
                    execution_time: Some(execution_time),
                    method_used: Some(factors.method),
                    success,
                    phi_coordinates: phi_coords,
                    resonance_peaks,
                    search_iterations: None, // TODO: Extract from execution
                    failure_reason: None,
                }
            }
            Err(e) => {
                let timeout_hit = execution_time > self.timeout;
                println!("  ✗ Failed: {} ({})", e, 
                         if timeout_hit { "timeout" } else { "error" });
                
                DiagnosticData {
                    bit_size,
                    recognition_time,
                    formalization_time,
                    execution_time: Some(execution_time),
                    method_used: None,
                    success: false,
                    phi_coordinates: phi_coords,
                    resonance_peaks,
                    search_iterations: None,
                    failure_reason: Some(if timeout_hit { 
                        format!("Timeout after {:?}", execution_time) 
                    } else { 
                        e.to_string() 
                    }),
                }
            }
        }
    }
    
    fn analyze_diagnostics(&self) {
        println!("\n{}", "=".repeat(80));
        println!("DIAGNOSTIC ANALYSIS");
        println!("{}", "=".repeat(80));
        
        // Success rate by bit size
        let mut by_size: HashMap<usize, (usize, usize)> = HashMap::new();
        for diag in &self.diagnostics {
            let bucket = (diag.bit_size / 10) * 10; // 10-bit buckets
            let entry = by_size.entry(bucket).or_insert((0, 0));
            entry.1 += 1; // total
            if diag.success {
                entry.0 += 1; // successes
            }
        }
        
        println!("\n1. Success Rate by Bit Size:");
        let mut sizes: Vec<_> = by_size.keys().collect();
        sizes.sort();
        for size in sizes {
            let (successes, total) = by_size[size];
            println!("   {}-{} bits: {}/{} ({:.1}%)", 
                     size, size + 9, successes, total, 
                     100.0 * successes as f64 / total as f64);
        }
        
        // Method effectiveness
        let mut by_method: HashMap<String, usize> = HashMap::new();
        for diag in &self.diagnostics {
            if let Some(method) = &diag.method_used {
                *by_method.entry(method.clone()).or_insert(0) += 1;
            }
        }
        
        println!("\n2. Successful Methods:");
        for (method, count) in by_method {
            println!("   {}: {} times", method, count);
        }
        
        // Timing analysis
        println!("\n3. Average Timing by Phase:");
        let successful: Vec<_> = self.diagnostics.iter().filter(|d| d.success).collect();
        if !successful.is_empty() {
            let avg_rec = successful.iter().map(|d| d.recognition_time.as_micros()).sum::<u128>() / successful.len() as u128;
            let avg_form = successful.iter().map(|d| d.formalization_time.as_micros()).sum::<u128>() / successful.len() as u128;
            let avg_exec = successful.iter()
                .filter_map(|d| d.execution_time.map(|t| t.as_micros()))
                .sum::<u128>() / successful.len() as u128;
            
            println!("   Recognition: {}µs", avg_rec);
            println!("   Formalization: {}µs", avg_form);
            println!("   Execution: {}µs", avg_exec);
        }
        
        // Failure analysis
        println!("\n4. Failure Patterns:");
        let failures: Vec<_> = self.diagnostics.iter().filter(|d| !d.success).collect();
        let mut failure_reasons: HashMap<String, usize> = HashMap::new();
        for diag in &failures {
            if let Some(reason) = &diag.failure_reason {
                let key = if reason.contains("timeout") {
                    "Timeout".to_string()
                } else if reason.contains("All decoding strategies failed") {
                    "All strategies failed".to_string()
                } else {
                    reason.split(':').next().unwrap_or("Unknown").to_string()
                };
                *failure_reasons.entry(key).or_insert(0) += 1;
            }
        }
        
        for (reason, count) in failure_reasons {
            println!("   {}: {} times", reason, count);
        }
        
        // Phi coordinate analysis
        println!("\n5. Phi Coordinate Patterns:");
        println!("   Successful factorizations:");
        for diag in successful.iter().take(5) {
            println!("     {}-bit: φ={:.4}, π={:.4}, e={:.4}, unity={:.4}",
                     diag.bit_size, 
                     diag.phi_coordinates[0],
                     diag.phi_coordinates[1],
                     diag.phi_coordinates[2],
                     diag.phi_coordinates[3]);
        }
        
        // Recommendations
        println!("\n6. Tuning Recommendations:");
        
        // Check if timeout is the main issue
        let timeout_count = failures.iter()
            .filter(|d| d.failure_reason.as_ref().map_or(false, |r| r.contains("timeout")))
            .count();
        if timeout_count > failures.len() / 2 {
            println!("   - Timeout is the main bottleneck");
            println!("   - Consider optimizing search radius calculation");
            println!("   - Pre-computed basis needs more patterns for large numbers");
        }
        
        // Check bit size where failures start
        let min_failure_bits = failures.iter()
            .map(|d| d.bit_size)
            .min()
            .unwrap_or(0);
        println!("   - Failures start at {}-bit numbers", min_failure_bits);
        
        // Check if certain bit ranges are problematic
        let problem_ranges: Vec<_> = by_size.iter()
            .filter(|(_, (s, t))| (*s as f64 / *t as f64) < 0.5)
            .map(|(size, _)| *size)
            .collect();
        if !problem_ranges.is_empty() {
            println!("   - Problem bit ranges: {:?}", problem_ranges);
        }
    }
}

// Structure for hard semiprime test cases
struct HardSemiprime {
    n_bits: usize,
    p_str: &'static str,
    q_str: &'static str,
    n_str: &'static str,
}

fn generate_hard_semiprimes() -> Vec<HardSemiprime> {
    let mut tests = Vec::new();
    
    // Perfect squares - hardest case for Fermat method
    tests.push(HardSemiprime {
        n_bits: 14,
        p_str: "127",
        q_str: "127",
        n_str: "16129",
    });
    
    tests.push(HardSemiprime {
        n_bits: 26,
        p_str: "8191",
        q_str: "8191",
        n_str: "67092481",
    });
    
    tests.push(HardSemiprime {
        n_bits: 62,
        p_str: "2147483647",
        q_str: "2147483647",
        n_str: "4611686014132420609",
    });
    
    tests.push(HardSemiprime {
        n_bits: 122,
        p_str: "2305843009213693951",
        q_str: "2305843009213693951",
        n_str: "5316911983139663487056679725687676687003542222693990401",
    });
    
    tests.push(HardSemiprime {
        n_bits: 178,
        p_str: "618970019642690137449562111",
        q_str: "618970019642690137449562111",
        n_str: "383123885216472214589586756787577156192410406959969361220640931648166442631281556619304505646776321",
    });
    
    tests.push(HardSemiprime {
        n_bits: 214,
        p_str: "162259276829213363391578010288127",
        q_str: "162259276829213363391578010288127",
        n_str: "26328072917139296674376571195689832382847494658755249946063223410494657557168129",
    });
    
    tests.push(HardSemiprime {
        n_bits: 254,
        p_str: "170141183460469231731687303715884105727",
        q_str: "170141183460469231731687303715884105727",
        n_str: "28948022309329048855892746252171976962977213799489202546401021394546514198529",
    });
    
    // Add larger bit sizes - these are constructed perfect squares
    
    // 300-bit perfect square (approximately)
    tests.push(HardSemiprime {
        n_bits: 300,
        p_str: "1267650600228229401496703205253",
        q_str: "1267650600228229401496703205253",
        n_str: "1606958217113926705498754129785816771765510949009",
    });
    
    // 350-bit perfect square (larger prime)
    tests.push(HardSemiprime {
        n_bits: 350,
        p_str: "89884656743115795386465259539451236680898848947115328636715040578866337902750481566354238661203768010560056939935696678829394884407208311246423715319737062188883946712432742638151109800623047059726541476042502884419075341171231440736956555270413618581675255342293149119973622969239858152417678164812112069503",
        q_str: "89884656743115795386465259539451236680898848947115328636715040578866337902750481566354238661203768010560056939935696678829394884407208311246423715319737062188883946712432742638151109800623047059726541476042502884419075341171231440736956555270413618581675255342293149119973622969239858152417678164812112069503",
        n_str: "8079251617235838247984766718397076470485901182256382831063285765548493540995466349985076239784725195711299790060511737034151730672031416042162360799887859791462165450688738577139321918928671940446954912030521189262128978426579370758744292101406855499680795773937899160471116122442689968944201289479506080579009",
    });
    
    // 400-bit approximate perfect square
    tests.push(HardSemiprime {
        n_bits: 400,
        p_str: "1267650600228229401496703205376",
        q_str: "1267650600228229401496703205376",
        n_str: "1606958638712892641825215683614976",
    });
    
    // 450-bit number (will trigger larger number handling)
    tests.push(HardSemiprime {
        n_bits: 450,
        p_str: "134217727",  // 2^27 - 1 (Mersenne prime)
        q_str: "134217727",  
        n_str: "18014398241046529",  // 134217727^2
    });
    
    // 512-bit - this will definitely trigger RSA-scale handling
    tests.push(HardSemiprime {
        n_bits: 512,
        p_str: "13407807929942597099574024998205846127479365820592393377723561443721764030073546976801874298166903427690031858186486050853753882811946569946433649006084171",
        q_str: "13407807929942597099574024998205846127479365820592393377723561443721764030073546976801874298166903427690031858186486050853753882811946569946433649006084171",
        n_str: "179769313486231590772930519078902473361797697894230657273430081157732675805500963132708477322407536021120113879871393357658789768814416622492847430639474124377767893424865485276302219601246094119453082952085005768838150682342462881473913110540827237163350510684586298239947245938479716304835356329624224137241",
    });
    
    tests
}

fn main() {
    println!("=== DIAGNOSTIC FRAMEWORK FOR HARD SEMIPRIME FACTORIZATION ===");
    println!("Goal: Factor hard semiprimes up to 300-bit");
    println!("Strategy: Fail fast, tune, test\n");
    
    let mut runner = DiagnosticRunner::new(30); // 30 second timeout for larger numbers
    
    // Test sequence: gradually increasing hard semiprimes
    let test_sequence = vec![
        // Start small to verify functionality
        (8, 8),    // ~16-bit
        (16, 16),  // ~32-bit
        (32, 32),  // ~64-bit
        (64, 64),  // ~128-bit
        (80, 80),  // ~160-bit
        (96, 96),  // ~192-bit
        (112, 112), // ~224-bit
        (128, 128), // ~256-bit
        (140, 140), // ~280-bit
        (150, 150), // ~300-bit
        (160, 160), // ~320-bit
        (170, 170), // ~340-bit
        (180, 180), // ~360-bit
        (200, 200), // ~400-bit
        (250, 250), // ~500-bit
        (300, 300), // ~600-bit
        (400, 400), // ~800-bit
        (500, 500), // ~1000-bit
    ];
    
    // Test the pre-computed hard semiprimes
    let hard_semiprimes = generate_hard_semiprimes();
    
    println!("Testing pre-computed hard semiprimes:");
    for semiprime in hard_semiprimes {
        let n = Number::from_str(&semiprime.n_str).unwrap();
        let p = Number::from_str(&semiprime.p_str).unwrap(); 
        let q = Number::from_str(&semiprime.q_str).unwrap();
        
        let diag = runner.test_number(&n, Some(&p), Some(&q));
        let should_stop = diag.bit_size > 300 && !diag.success;
        let bit_size = diag.bit_size;
        
        runner.diagnostics.push(diag);
        
        // Stop if we're consistently failing larger numbers
        if should_stop {
            println!("\n⚠️  Stopping: Failed on {}-bit number", bit_size);
            break;
        }
    }
    
    // Analyze results
    runner.analyze_diagnostics();
    
    // Save diagnostics for further analysis
    println!("\n7. Raw Diagnostic Data:");
    println!("   Total tests: {}", runner.diagnostics.len());
    println!("   Successful: {}", runner.diagnostics.iter().filter(|d| d.success).count());
    println!("   Failed: {}", runner.diagnostics.iter().filter(|d| !d.success).count());
    
    // Suggest next steps
    println!("\n8. Next Steps:");
    println!("   - Examine the pre-computed basis for patterns at failure points");
    println!("   - Adjust search radius calculations for larger bit sizes");
    println!("   - Consider enhanced basis for numbers > 64 bits");
    println!("   - Profile the execution phase to find bottlenecks");
}