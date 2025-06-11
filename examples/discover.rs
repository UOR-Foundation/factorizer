//! Example: Discover emergent patterns and universal constants
//!
//! This example demonstrates the discovery of emergent patterns from
//! empirical observations without imposing structure.

use rust_pattern_solver::emergence::{discover_all_patterns, UniversalPatterns};
use rust_pattern_solver::observer::ObservationCollector;
use rust_pattern_solver::types::{Number, PatternKind};
use rust_pattern_solver::utils;

fn main() -> rust_pattern_solver::Result<()> {
    println!("=== The Pattern: Discovery Stage ===\n");
    
    // Generate comprehensive observation dataset
    println!("Generating observation dataset...");
    let mut collector = ObservationCollector::new();
    let mut test_numbers = Vec::new();
    
    // Small balanced semiprimes
    for i in 2..20 {
        for j in i..20 {
            if utils::is_probable_prime(&Number::from(i as u32), 10) &&
               utils::is_probable_prime(&Number::from(j as u32), 10) {
                test_numbers.push(Number::from(i as u32) * Number::from(j as u32));
            }
        }
    }
    
    // Medium semiprimes (16-bit factors)
    for _ in 0..50 {
        let p = utils::generate_random_prime(8)?;
        let q = utils::generate_random_prime(8)?;
        test_numbers.push(&p * &q);
    }
    
    // Larger semiprimes (32-bit factors)
    for _ in 0..20 {
        let p = utils::generate_random_prime(16)?;
        let q = utils::generate_random_prime(16)?;
        test_numbers.push(&p * &q);
    }
    
    println!("Collecting {} observations...", test_numbers.len());
    let observations = collector.observe_parallel(&test_numbers)?;
    println!("Successfully observed {} numbers\n", observations.len());
    
    // Discover emergent patterns
    println!("Discovering emergent patterns...");
    let emergent = discover_all_patterns(&observations)?;
    
    println!("\n=== Discovered Patterns ===");
    println!("Total patterns found: {}\n", emergent.count());
    
    // Display invariants
    println!("Invariant Relationships ({}):", emergent.invariants.len());
    for pattern in &emergent.invariants {
        println!("  - {}: {}", pattern.id, pattern.description);
        println!("    Frequency: {:.1}%", pattern.frequency * 100.0);
        if !pattern.scale_range.unbounded {
            println!("    Scale: {}-{} bits", 
                pattern.scale_range.min_bits,
                pattern.scale_range.max_bits);
        }
    }
    
    // Display scaling patterns
    println!("\nScale-Dependent Patterns ({}):", emergent.scaling_patterns.len());
    for pattern in &emergent.scaling_patterns {
        println!("  - {}: {}", pattern.id, pattern.description);
        if !pattern.parameters.is_empty() {
            println!("    Parameters: {:?}", pattern.parameters);
        }
    }
    
    // Display universal patterns
    println!("\nUniversal Patterns ({}):", emergent.universal_patterns.len());
    for pattern in &emergent.universal_patterns {
        println!("  - {}: {}", pattern.id, pattern.description);
        println!("    Frequency: {:.1}%", pattern.frequency * 100.0);
        if !pattern.parameters.is_empty() {
            println!("    Parameters: {:?}", pattern.parameters);
        }
    }
    
    // Extract universal constants
    println!("\n=== Universal Constants ===");
    let constants = UniversalPatterns::extract_constants(&emergent.universal_patterns);
    
    if constants.is_empty() {
        println!("No universal constants discovered yet.");
        println!("(More observations needed to reveal universal structure)");
    } else {
        for constant in &constants {
            println!("  {}: {:.6}", constant.name, constant.value);
            println!("    Universality: {:.1}%", constant.universality * 100.0);
            println!("    {}", constant.description);
        }
    }
    
    // Analyze pattern statistics
    println!("\n=== Pattern Analysis ===");
    
    let invariant_count = emergent.invariants.iter()
        .filter(|p| p.frequency >= 1.0)
        .count();
    println!("Perfect invariants: {}", invariant_count);
    
    let high_freq_patterns = emergent.all_patterns().into_iter()
        .filter(|p| p.frequency >= 0.8)
        .count();
    println!("High-frequency patterns (≥80%): {}", high_freq_patterns);
    
    // Check for specific interesting patterns
    let has_phi = emergent.universal_patterns.iter()
        .any(|p| p.id.contains("phi"));
    let has_pi = emergent.universal_patterns.iter()
        .any(|p| p.id.contains("pi"));
    let has_e = emergent.universal_patterns.iter()
        .any(|p| p.id.contains("exponential") || p.id.contains("e_"));
    
    println!("\nUniversal constant relationships found:");
    println!("  Golden ratio (φ): {}", if has_phi { "Yes" } else { "No" });
    println!("  Pi (π): {}", if has_pi { "Yes" } else { "No" });
    println!("  Euler's number (e): {}", if has_e { "Yes" } else { "No" });
    
    // Demonstrate pattern evolution with scale
    println!("\n=== Pattern Evolution with Scale ===");
    
    // Group observations by bit length
    let mut by_scale = std::collections::HashMap::new();
    for obs in &observations {
        by_scale.entry(obs.scale.bit_length)
            .or_insert_with(Vec::new)
            .push(obs);
    }
    
    // Show how patterns change
    let mut scales: Vec<_> = by_scale.keys().copied().collect();
    scales.sort();
    
    for &scale in scales.iter().take(5) {
        if let Some(obs_at_scale) = by_scale.get(&scale) {
            let balanced_ratio = obs_at_scale.iter()
                .filter(|obs| obs.scale.balance_ratio < 1.1)
                .count() as f64 / obs_at_scale.len() as f64;
            
            println!("  {} bits: {:.1}% balanced (n={})",
                scale, balanced_ratio * 100.0, obs_at_scale.len());
        }
    }
    
    // Save discovered patterns
    println!("\n=== Saving Results ===");
    println!("Patterns have been discovered and analyzed.");
    println!("In a production system, these would be saved to:");
    println!("  - data/patterns.json");
    println!("  - data/universal_constants.json");
    
    Ok(())
}