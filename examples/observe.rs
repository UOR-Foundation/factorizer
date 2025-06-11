//! Example: Observe factorizations and collect data
//!
//! This example demonstrates how to observe factorizations and collect
//! comprehensive data without imposing any preconceptions.

use rust_pattern_solver::observer::{ObservationCollector, ObservationFilter};
use rust_pattern_solver::types::Number;
use rust_pattern_solver::utils;

fn main() -> rust_pattern_solver::Result<()> {
    println!("=== The Pattern: Observation Stage ===\n");

    // Create observation collector
    let mut collector = ObservationCollector::new();

    // Example 1: Observe a single factorization
    println!("Observing single factorization:");
    let n = Number::from(143u32); // 11 × 13
    match collector.observe_single(n.clone()) {
        Ok(obs) => {
            println!("n = {}", obs.n);
            println!("p = {}, q = {}", obs.p, obs.q);
            println!("Pattern type: {:?}", obs.scale.pattern_type);
            println!("Balance ratio: {:.3}", obs.scale.balance_ratio);
            println!("Offset from sqrt(n): {}", obs.derived.offset);
            println!("Modular signature: {:?}\n", obs.modular.modular_signature);
        },
        Err(e) => println!("Failed to observe {}: {}\n", n, e),
    }

    // Example 2: Observe multiple factorizations
    println!("Observing multiple factorizations:");
    let test_cases = vec![
        Number::from(15u32),  // 3 × 5
        Number::from(21u32),  // 3 × 7
        Number::from(35u32),  // 5 × 7
        Number::from(77u32),  // 7 × 11
        Number::from(143u32), // 11 × 13
        Number::from(221u32), // 13 × 17
        Number::from(323u32), // 17 × 19
        Number::from(437u32), // 19 × 23
    ];

    match collector.observe_parallel(&test_cases) {
        Ok(observations) => {
            println!("Collected {} observations", observations.len());

            // Analyze patterns
            let balanced_count =
                observations.iter().filter(|obs| obs.scale.balance_ratio < 1.1).count();
            println!(
                "Balanced semiprimes: {}/{}",
                balanced_count,
                observations.len()
            );

            // Show offset distribution
            let avg_offset_ratio: f64 =
                observations.iter().map(|obs| obs.derived.offset_ratio.abs()).sum::<f64>()
                    / observations.len() as f64;
            println!("Average offset ratio: {:.4}", avg_offset_ratio);
        },
        Err(e) => println!("Failed to observe: {}", e),
    }

    // Example 3: Filtered observation
    println!("\nFiltered observation (16-bit balanced semiprimes):");
    let filter = ObservationFilter {
        min_bits: Some(14),
        max_bits: Some(18),
        pattern_type: Some(rust_pattern_solver::types::observation::PatternClass::Balanced),
        max_balance_ratio: Some(1.2),
    };

    // Generate test semiprimes in range
    let mut filtered_cases = Vec::new();
    for _ in 0..20 {
        let p = utils::generate_random_prime(8)?;
        let q = utils::generate_random_prime(8)?;
        let n = &p * &q;
        if n.bit_length() >= 14 && n.bit_length() <= 18 {
            filtered_cases.push(n);
        }
    }

    match collector.observe_filtered(&filtered_cases, &filter) {
        Ok(filtered_obs) => {
            println!("Found {} matching observations:", filtered_obs.len());
            for obs in filtered_obs.iter().take(3) {
                println!(
                    "  {} = {} × {} (balance: {:.3})",
                    obs.n, obs.p, obs.q, obs.scale.balance_ratio
                );
            }
        },
        Err(e) => println!("Failed to observe: {}", e),
    }

    // Example 4: Large number observation
    println!("\nObserving larger numbers:");
    let large_cases = vec![
        // Generate some 32-bit semiprimes
        utils::generate_random_prime(16)? * utils::generate_random_prime(16)?,
        utils::generate_random_prime(20)? * utils::generate_random_prime(12)?,
        utils::generate_random_prime(24)? * utils::generate_random_prime(8)?,
    ];

    match collector.observe_parallel(&large_cases) {
        Ok(large_obs) => {
            for obs in large_obs {
                println!(
                    "  {} bits: balance={:.3}, gap={}",
                    obs.scale.bit_length, obs.scale.balance_ratio, obs.scale.prime_gap
                );
            }
        },
        Err(e) => println!("Failed to observe: {}", e),
    }

    // Save observations
    println!("\nSaving observations to data/observations.json");
    if let Err(e) = collector.save_to_file("data/observations.json") {
        println!("Failed to save: {}", e);
    }

    Ok(())
}
