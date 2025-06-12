//! Example: Visualize The Pattern
//!
//! This example demonstrates visualization of pattern signatures,
//! resonance fields, and quantum neighborhoods.

use rust_pattern_solver::observer::ObservationCollector;
use rust_pattern_solver::pattern::{recognition, Pattern};
use rust_pattern_solver::types::Number;

fn main() -> rust_pattern_solver::Result<()> {
    println!("=== The Pattern: Visualization ===\n");

    // Collect some observations for pattern discovery
    let mut collector = ObservationCollector::new();
    let training_numbers: Vec<Number> = (0..10)
        .map(|i| {
            let p = 2 * i + 3;
            let q = 2 * i + 5;
            Number::from(p * q)
        })
        .collect();

    let observations = collector.observe_parallel(&training_numbers)?;
    let patterns = Pattern::discover_from_observations(&observations)?;

    // Visualize different types of numbers
    let test_cases = vec![
        (Number::from(143u32), "11 × 13", "Balanced"),
        (Number::from(91u32), "7 × 13", "Harmonic"),
        (Number::from(93u32), "3 × 31", "Unbalanced"),
        (Number::from(121u32), "11²", "Square"),
        (Number::from(97u32), "prime", "Prime"),
    ];

    for (n, desc, expected_type) in test_cases {
        println!("╔═══════════════════════════════════════════════════════╗");
        println!(
            "║ Visualizing: {} = {} ({}){}║",
            n,
            desc,
            expected_type,
            " ".repeat(26 - n.to_string().len() - desc.len() - expected_type.len())
        );
        println!("╚═══════════════════════════════════════════════════════╝");

        match recognition::recognize(n.clone(), &patterns) {
            Ok(recognition) => {
                // Basic info
                println!("\nPattern Type: {:?}", recognition.pattern_type);
                println!("Confidence: {:.2}%", recognition.confidence * 100.0);

                // Visualize signature components
                println!("\n📊 Signature Components:");
                visualize_components(&recognition.signature.components);

                // Visualize resonance field
                println!("\n🌊 Resonance Field:");
                visualize_resonance_field(&recognition.signature.resonance);

                // Visualize quantum neighborhood
                println!("\n⚛️  Quantum Neighborhood:");
                if let Some(ref quantum_region) = recognition.quantum_neighborhood {
                    visualize_quantum_region(quantum_region);
                } else {
                    println!("   (not identified)");
                }

                // Show pattern DNA
                println!("\n🧬 Pattern DNA:");
                if recognition.signature.modular_dna.len() >= 10 {
                    print!("   Modular: [");
                    for (i, &val) in recognition.signature.modular_dna.iter().take(10).enumerate() {
                        if i > 0 {
                            print!(", ");
                        }
                        print!("{}", val);
                    }
                    println!(", ...]");
                } else {
                    println!("   Modular: {:?}", recognition.signature.modular_dna);
                }
            },
            Err(e) => println!("\n❌ Recognition failed: {}", e),
        }

        println!("\n");
    }

    // Visualize pattern relationships
    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║                 Pattern Relationships                  ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    // Show how patterns relate to each other
    let n1 = Number::from(143u32); // 11 × 13
    let n2 = Number::from(221u32); // 13 × 17
    let n3 = Number::from(323u32); // 17 × 19

    println!("Comparing consecutive balanced semiprimes:");

    if let (Ok(r1), Ok(r2), Ok(r3)) = (
        recognition::recognize(n1.clone(), &patterns),
        recognition::recognize(n2.clone(), &patterns),
        recognition::recognize(n3.clone(), &patterns),
    ) {
        println!("\n{} → {} → {}", n1, n2, n3);

        // Compare phi components
        let phi1 = r1.signature.get_component("phi_component").unwrap_or(0.0);
        let phi2 = r2.signature.get_component("phi_component").unwrap_or(0.0);
        let phi3 = r3.signature.get_component("phi_component").unwrap_or(0.0);

        println!("\nφ evolution: {:.4} → {:.4} → {:.4}", phi1, phi2, phi3);

        // Show resonance correlation
        if r1.signature.resonance.len() == r2.signature.resonance.len() {
            let correlation = compute_correlation(&r1.signature.resonance, &r2.signature.resonance);
            println!("Resonance correlation (143↔221): {:.3}", correlation);
        }

        if r2.signature.resonance.len() == r3.signature.resonance.len() {
            let correlation = compute_correlation(&r2.signature.resonance, &r3.signature.resonance);
            println!("Resonance correlation (221↔323): {:.3}", correlation);
        }
    }

    Ok(())
}

fn visualize_components(components: &std::collections::HashMap<String, f64>) {
    let mut sorted: Vec<_> = components.iter().collect();
    sorted.sort_by_key(|&(k, _)| k);

    for (name, value) in sorted.iter().take(8) {
        let bar_length = (value.abs() * 20.0).min(40.0) as usize;
        let bar = "█".repeat(bar_length);
        println!("   {:>15}: {:>8.4} {}", name, value, bar);
    }
}

fn visualize_resonance_field(resonance: &[f64]) {
    if resonance.is_empty() {
        println!("   (empty)");
        return;
    }

    // Find min/max for normalization
    let max_val = resonance.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b.abs()));
    let min_val = resonance.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    // Create ASCII visualization
    let width = 50;
    let height = 8;
    let mut grid = vec![vec![' '; width]; height];

    // Plot resonance values
    for (i, &val) in resonance.iter().enumerate() {
        let x = (i * width / resonance.len()).min(width - 1);
        let normalized =
            if max_val > min_val { (val - min_val) / (max_val - min_val) } else { 0.5 };
        let y = ((1.0 - normalized) * (height - 1) as f64) as usize;

        grid[y][x] = match normalized {
            n if n > 0.8 => '█',
            n if n > 0.6 => '▓',
            n if n > 0.4 => '▒',
            n if n > 0.2 => '░',
            _ => '·',
        };
    }

    // Print grid
    for row in grid {
        print!("   ");
        for ch in row {
            print!("{}", ch);
        }
        println!();
    }

    println!(
        "   Min: {:.3}, Max: {:.3}, Points: {}",
        min_val,
        max_val,
        resonance.len()
    );
}

fn visualize_quantum_region(region: &rust_pattern_solver::types::QuantumRegion) {
    println!("   Center: {}", region.center);
    println!("   Radius: {}", region.radius);
    println!("   Peak offset: {}", region.peak_offset);
    println!("   Confidence: {:.2}%", region.confidence * 100.0);

    if !region.probability_distribution.is_empty() {
        println!(
            "   Probability distribution: {} points",
            region.probability_distribution.len()
        );
        let max_prob = region.probability_distribution.iter().fold(0.0f64, |a, &b| a.max(b));
        println!("   Peak probability: {:.4}", max_prob);
    }
}

fn compute_correlation(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let n = a.len() as f64;
    let sum_a: f64 = a.iter().sum();
    let sum_b: f64 = b.iter().sum();
    let sum_ab: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let sum_a2: f64 = a.iter().map(|x| x * x).sum();
    let sum_b2: f64 = b.iter().map(|y| y * y).sum();

    let numerator = n * sum_ab - sum_a * sum_b;
    let denominator = ((n * sum_a2 - sum_a * sum_a) * (n * sum_b2 - sum_b * sum_b)).sqrt();

    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}
