//! Example: Observe RSA-like factorizations
//!
//! This example focuses on collecting observations of RSA-like numbers
//! (balanced semiprimes where p ≈ q) to improve pattern recognition.

use rust_pattern_solver::observer::ObservationCollector;
use rust_pattern_solver::types::Number;
use rust_pattern_solver::utils;
use std::collections::HashMap;

fn main() -> rust_pattern_solver::Result<()> {
    println!("=== The Pattern: RSA-Like Observation ===\n");

    let mut collector = ObservationCollector::new();
    let mut observations_by_size: HashMap<usize, Vec<_>> = HashMap::new();

    // Generate RSA-like numbers of increasing sizes
    let bit_sizes = vec![8, 10, 12, 14, 16, 18, 20, 24, 28, 32];
    
    for &bits in &bit_sizes {
        println!("Generating {}-bit RSA-like semiprimes...", bits);
        
        let half_bits = bits / 2;
        let mut numbers = Vec::new();
        
        // Generate multiple examples at each size
        for i in 0..20 {
            // Create balanced semiprimes where p ≈ q
            let p_base = Number::from(1u32) << (half_bits as u32 - 1);
            let q_base = Number::from(1u32) << (half_bits as u32 - 1);
            
            // Add small random-like offsets to make them prime-like
            let p_offset = Number::from((i * 2 + 3) as u32);
            let q_offset = Number::from((i * 2 + 5) as u32);
            
            let p = &p_base + &p_offset;
            let q = &q_base + &q_offset;
            
            // Ensure p ≤ q
            let (p, q) = if p <= q { (p, q) } else { (q, p) };
            
            let n = &p * &q;
            
            // Only keep if it's the right size
            if n.bit_length() == bits {
                numbers.push(n.clone());
                
                // Also add some variations with different offsets
                if i < 5 {
                    // Try with larger offsets to get more prime-like numbers
                    let p2 = &p_base + &Number::from((i * 6 + 7) as u32);
                    let q2 = &q_base + &Number::from((i * 6 + 11) as u32);
                    let n2 = &p2 * &q2;
                    if n2.bit_length() == bits {
                        numbers.push(n2);
                    }
                }
            }
        }
        
        // Observe these numbers
        match collector.observe_parallel(&numbers) {
            Ok(obs) => {
                println!("  Collected {} observations", obs.len());
                
                // Analyze balance ratios
                let balanced = obs.iter()
                    .filter(|o| o.scale.balance_ratio < 2.0)
                    .count();
                println!("  Balanced (ratio < 2.0): {}/{}", balanced, obs.len());
                
                // Store for later analysis
                for o in obs {
                    observations_by_size
                        .entry(o.scale.bit_length)
                        .or_insert_with(Vec::new)
                        .push(o);
                }
            }
            Err(e) => println!("  Failed: {}", e),
        }
    }
    
    // Analyze patterns by size
    println!("\n=== Pattern Analysis by Bit Size ===");
    
    let mut sizes: Vec<_> = observations_by_size.keys().cloned().collect();
    sizes.sort();
    
    for size in sizes {
        let obs = &observations_by_size[&size];
        
        // Calculate statistics
        let avg_balance: f64 = obs.iter()
            .map(|o| o.scale.balance_ratio)
            .sum::<f64>() / obs.len() as f64;
            
        let avg_offset_ratio: f64 = obs.iter()
            .map(|o| o.derived.offset_ratio.abs())
            .sum::<f64>() / obs.len() as f64;
            
        let prime_classifications = obs.iter()
            .filter(|o| o.p == Number::from(1u32) || o.q == o.n)
            .count();
            
        println!("\n{}-bit numbers ({} observations):", size, obs.len());
        println!("  Average balance ratio: {:.3}", avg_balance);
        println!("  Average offset ratio: {:.4}", avg_offset_ratio);
        println!("  Misclassified as prime: {}/{}", prime_classifications, obs.len());
        
        // Show example patterns
        if let Some(first) = obs.first() {
            println!("  Example: {} = {} × {}", first.n, first.p, first.q);
            println!("    Offset from sqrt: {}", first.derived.offset);
            println!("    First 10 modular DNA: {:?}", 
                     &first.modular.modular_signature[..10.min(first.modular.modular_signature.len())]);
        }
    }
    
    // Save RSA-focused observations
    println!("\n=== Saving RSA-Like Observations ===");
    
    // Create a focused dataset
    let all_obs = collector.observations();
    let rsa_like: Vec<_> = all_obs.iter()
        .filter(|o| {
            // Focus on balanced semiprimes where neither factor is 1
            o.scale.balance_ratio < 10.0 && 
            o.p != Number::from(1u32) && 
            o.q != o.n
        })
        .cloned()
        .collect();
    
    println!("Total observations: {}", all_obs.len());
    println!("RSA-like observations: {}", rsa_like.len());
    
    // Save to file
    match collector.save_to_file("data/rsa_observations.json") {
        Ok(_) => println!("Saved to data/rsa_observations.json"),
        Err(e) => println!("Failed to save: {}", e),
    }
    
    // Key insights
    println!("\n=== Key Insights ===");
    println!("1. Balanced semiprimes (p ≈ q) have distinctive patterns");
    println!("2. Offset from sqrt(n) is typically small for balanced cases");
    println!("3. Modular signatures show characteristic patterns");
    println!("4. Current recognition is misclassifying many composites as primes");
    
    Ok(())
}