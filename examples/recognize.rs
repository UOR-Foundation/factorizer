//! Example: Recognize patterns in numbers
//!
//! This example demonstrates pattern recognition - finding signatures
//! and identifying quantum neighborhoods where factors exist.

use rust_pattern_solver::observer::ObservationCollector;
use rust_pattern_solver::pattern::{execution, formalization, recognition, Pattern};
use rust_pattern_solver::types::Number;

fn main() -> rust_pattern_solver::Result<()> {
    println!("=== The Pattern: Recognition Stage ===\n");

    // First, collect some observations to establish patterns
    let mut collector = ObservationCollector::new();

    let training_set = vec![
        Number::from(15u32),  // 3 × 5
        Number::from(21u32),  // 3 × 7
        Number::from(35u32),  // 5 × 7
        Number::from(77u32),  // 7 × 11
        Number::from(143u32), // 11 × 13
        Number::from(221u32), // 13 × 17
        Number::from(323u32), // 17 × 19
        Number::from(437u32), // 19 × 23
        Number::from(667u32), // 23 × 29
        Number::from(899u32), // 29 × 31
    ];

    println!("Collecting training observations...");
    let observations = collector.observe_parallel(&training_set)?;
    println!("Collected {} observations\n", observations.len());

    // Discover patterns from observations
    let patterns = Pattern::discover_from_observations(&observations)?;
    println!("Discovered {} patterns\n", patterns.len());

    // Now demonstrate recognition on new numbers
    let test_cases = vec![
        (Number::from(391u32), "17 × 23"),  // Balanced
        (Number::from(35u32), "5 × 7"),     // Small balanced
        (Number::from(91u32), "7 × 13"),    // Harmonic
        (Number::from(93u32), "3 × 31"),    // Unbalanced
        (Number::from(2021u32), "43 × 47"), // Large balanced
    ];

    for (n, factors) in test_cases {
        println!("Recognizing pattern for {} = {}:", &n, factors);

        // Stage 1: Recognition
        match recognition::recognize(n.clone(), &patterns) {
            Ok(recognition) => {
                println!("  Pattern type: {:?}", recognition.pattern_type);
                println!("  Confidence: {:.2}", recognition.confidence);
                println!("  Quantum neighborhood: [");

                // Show key signature components
                for (name, value) in recognition.signature.components.iter().take(5) {
                    println!("    {}: {:.4}", name, value);
                }
                println!("  ]");

                // Show resonance field summary
                let field_len = recognition.signature.resonance.len();
                let max_resonance = recognition
                    .signature
                    .resonance
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                println!(
                    "  Resonance field: {} points, max: {:.3}",
                    field_len, max_resonance
                );

                // Continue to formalization
                match formalization::formalize(recognition, &patterns, &[]) {
                    Ok(formalization) => {
                        println!(
                            "  Formalized with {} strategies",
                            formalization.strategies.len()
                        );

                        // Try execution
                        match execution::execute(formalization, &patterns) {
                            Ok(factors) => {
                                println!(
                                    "  ✓ Factors found: {} × {} (method: {})",
                                    factors.p, factors.q, factors.method
                                );
                            },
                            Err(e) => {
                                println!("  ✗ Execution failed: {}", e);
                            },
                        }
                    },
                    Err(e) => println!("  Formalization failed: {}", e),
                }
            },
            Err(e) => println!("  Recognition failed: {}", e),
        }
        println!();
    }

    // Demonstrate recognition on a prime
    println!("Recognizing pattern for prime 97:");
    match recognition::recognize(Number::from(97u32), &patterns) {
        Ok(rec) => {
            println!("  Pattern type: {:?}", rec.pattern_type);
            println!("  Confidence: {:.2}", rec.confidence);
            if rec.pattern_type == rust_pattern_solver::types::PatternType::Prime {
                println!("  ✓ Correctly identified as prime!");
            }
        },
        Err(e) => println!("  Recognition failed: {}", e),
    }

    // Demonstrate batch recognition
    println!("\nBatch recognition performance test:");
    let batch_size = 100;
    let mut batch = Vec::new();

    for _ in 0..batch_size {
        let p = rust_pattern_solver::utils::generate_random_prime(8)?;
        let q = rust_pattern_solver::utils::generate_random_prime(8)?;
        batch.push(&p * &q);
    }

    let start = std::time::Instant::now();
    let mut success_count = 0;

    for n in &batch {
        if let Ok(rec) = recognition::recognize(n.clone(), &patterns) {
            if rec.confidence > 0.5 {
                success_count += 1;
            }
        }
    }

    let elapsed = start.elapsed();
    println!(
        "  Recognized {}/{} numbers in {:.2}s ({:.2} per second)",
        success_count,
        batch_size,
        elapsed.as_secs_f64(),
        batch_size as f64 / elapsed.as_secs_f64()
    );

    Ok(())
}
