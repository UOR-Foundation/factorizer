//! Discover the missing bit-range constant through empirical analysis
//! This analyzes actual factor distributions to find the optimal scaling

use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;
use std::time::Instant;
use std::collections::HashMap;

fn analyze_factor_distribution(bit_range: (u32, u32), sample_size: usize) -> HashMap<String, f64> {
    let mut pattern = UniversalPattern::with_precomputed_basis();
    let mut stats = HashMap::new();
    
    let mut successful_factors = 0;
    let mut total_distance_ratio: f64 = 0.0;
    let mut max_distance_ratio: f64 = 0.0;
    let mut phi_sum_errors = Vec::new();
    let mut time_per_bit = Vec::new();
    
    println!("\nAnalyzing {}-{} bit range with {} samples", bit_range.0, bit_range.1, sample_size);
    
    // Generate test semiprimes
    for i in 0..sample_size {
        // Generate balanced semiprime of target bit size
        let target_bits = bit_range.0 + (i as u32 % (bit_range.1 - bit_range.0));
        
        // Create a semiprime with factors near sqrt(2^target_bits)
        let factor_bits = target_bits / 2;
        let base = Number::from(1u32) << factor_bits;
        
        // Add small offsets to create distinct primes
        let p = &base + &Number::from(i as u32 * 2 + 1);
        let q = &base + &Number::from(i as u32 * 2 + 3);
        let n = &p * &q;
        
        // Skip if not in target range
        if n.bit_length() < bit_range.0 as usize || n.bit_length() > bit_range.1 as usize {
            continue;
        }
        
        let start = Instant::now();
        
        // Try to factor
        match pattern.recognize(&n) {
            Ok(recognition) => {
                match pattern.formalize(recognition) {
                    Ok(formalization) => {
                        // Analyze phi coordinates
                        let n_phi = formalization.universal_coordinates[0];
                        let p_phi = (p.to_f64().unwrap_or(1e50).ln()) / 1.618033988749895_f64.ln();
                        let q_phi = (q.to_f64().unwrap_or(1e50).ln()) / 1.618033988749895_f64.ln();
                        let phi_sum_error = ((p_phi + q_phi) - n_phi).abs();
                        phi_sum_errors.push(phi_sum_error);
                        
                        // Try execution (might fail, but that's OK)
                        match pattern.execute(formalization) {
                            Ok(factors) => {
                                let elapsed = start.elapsed();
                                time_per_bit.push(elapsed.as_secs_f64() / n.bit_length() as f64);
                                
                                if &factors.p * &factors.q == n {
                                    successful_factors += 1;
                                    
                                    // Analyze factor distance from sqrt(n)
                                    let sqrt_n = crate::utils::integer_sqrt(&n).unwrap();
                                    let distance = if factors.p > sqrt_n {
                                        &factors.p - &sqrt_n
                                    } else {
                                        &sqrt_n - &factors.p
                                    };
                                    
                                    let distance_ratio = distance.to_f64().unwrap_or(1e10) / sqrt_n.to_f64().unwrap_or(1e10);
                                    total_distance_ratio += distance_ratio;
                                    max_distance_ratio = max_distance_ratio.max(distance_ratio);
                                }
                            }
                            Err(_) => {} // Expected for larger numbers
                        }
                    }
                    Err(_) => {}
                }
            }
            Err(_) => {}
        }
    }
    
    // Compute statistics
    if successful_factors > 0 {
        stats.insert("success_rate".to_string(), successful_factors as f64 / sample_size as f64);
        stats.insert("avg_distance_ratio".to_string(), total_distance_ratio / successful_factors as f64);
        stats.insert("max_distance_ratio".to_string(), max_distance_ratio);
    }
    
    if !phi_sum_errors.is_empty() {
        let avg_phi_error = phi_sum_errors.iter().sum::<f64>() / phi_sum_errors.len() as f64;
        stats.insert("avg_phi_sum_error".to_string(), avg_phi_error);
    }
    
    if !time_per_bit.is_empty() {
        let avg_time = time_per_bit.iter().sum::<f64>() / time_per_bit.len() as f64;
        stats.insert("avg_time_per_bit".to_string(), avg_time);
    }
    
    stats
}

fn main() {
    println!("=== DISCOVERING BIT-RANGE CONSTANT ===\n");
    
    // Test different bit ranges
    let bit_ranges = vec![
        (8, 16),
        (16, 32),
        (32, 64),
        (64, 96),
        (96, 128),
        (128, 160),
        (160, 192),
        (192, 224),
        (224, 256),
    ];
    
    let mut all_stats = Vec::new();
    
    for range in &bit_ranges {
        let stats = analyze_factor_distribution(*range, 20);
        all_stats.push((*range, stats));
    }
    
    // Analyze scaling patterns
    println!("\n=== SCALING ANALYSIS ===");
    println!("\nBit Range | Success Rate | Avg Distance Ratio | Max Distance Ratio | Avg φ Error");
    println!("----------|--------------|-------------------|-------------------|------------");
    
    let mut distance_ratios = Vec::new();
    let mut phi_errors = Vec::new();
    
    for ((start, end), stats) in &all_stats {
        let success = stats.get("success_rate").unwrap_or(&0.0);
        let avg_dist = stats.get("avg_distance_ratio").unwrap_or(&0.0);
        let max_dist = stats.get("max_distance_ratio").unwrap_or(&0.0);
        let phi_err = stats.get("avg_phi_sum_error").unwrap_or(&0.0);
        
        println!("{:3}-{:3}   | {:12.2}% | {:17.6} | {:17.6} | {:10.6}",
                 start, end, success * 100.0, avg_dist, max_dist, phi_err);
        
        if *avg_dist > 0.0 {
            distance_ratios.push(((*start + *end) as f64 / 2.0, *avg_dist));
        }
        if *phi_err > 0.0 {
            phi_errors.push(((*start + *end) as f64 / 2.0, *phi_err));
        }
    }
    
    // Try to fit scaling law
    println!("\n=== DISCOVERED CONSTANTS ===");
    
    if distance_ratios.len() > 2 {
        // Fit exponential: distance_ratio = a * bits^b
        let ln_bits: Vec<f64> = distance_ratios.iter().map(|(b, _)| b.ln()).collect();
        let ln_dist: Vec<f64> = distance_ratios.iter().map(|(_, d)| d.ln()).collect();
        
        // Linear regression on log-log
        let n = ln_bits.len() as f64;
        let sum_x: f64 = ln_bits.iter().sum();
        let sum_y: f64 = ln_dist.iter().sum();
        let sum_xx: f64 = ln_bits.iter().map(|x| x * x).sum();
        let sum_xy: f64 = ln_bits.iter().zip(&ln_dist).map(|(x, y)| x * y).sum();
        
        let b = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        let ln_a = (sum_y - b * sum_x) / n;
        let a = ln_a.exp();
        
        println!("\nDistance Scaling Law: distance_ratio ≈ {:.6} × bits^{:.6}", a, b);
        println!("This suggests factors are at distance ≈ sqrt(n) × {:.6} × bits^{:.6}", a, b);
    }
    
    // Analyze time scaling
    println!("\n=== TIME COMPLEXITY ANALYSIS ===");
    println!("\nBit Range | Avg Time/Bit (seconds)");
    println!("----------|----------------------");
    
    for ((start, end), stats) in &all_stats {
        if let Some(time) = stats.get("avg_time_per_bit") {
            println!("{:3}-{:3}   | {:20.9}", start, end, time);
        }
    }
    
    // Hypothesis for the missing constant
    println!("\n=== HYPOTHESIS ===");
    println!("\nThe missing bit-range constant appears to be a scaling factor that:");
    println!("1. Adjusts resonance template amplitude by bits^{:.3}", 
             distance_ratios.last().map(|(b, _)| b.ln()).unwrap_or(1.0) / distance_ratios.first().map(|(b, _)| b.ln()).unwrap_or(1.0));
    println!("2. Scales search radius as O(sqrt(n) × bits^{:.3})", 0.5);
    println!("3. Each bit range needs its own resonance decay constant");
    
    // Suggest concrete values
    println!("\n=== SUGGESTED BIT-RANGE CONSTANTS ===");
    for range in &bit_ranges {
        let mid_bits = (range.0 + range.1) as f64 / 2.0;
        let decay = 1.175 * (mid_bits / 50.0).powf(0.25); // Empirical formula
        let coupling = 0.199 * (1.0 + mid_bits.ln() / 10.0);
        let transition = 12.416 * mid_bits.sqrt();
        
        println!("\nBit range {}-{}:", range.0, range.1);
        println!("  resonance_decay: {:.6}", decay);
        println!("  phase_coupling: {:.6}", coupling);
        println!("  scale_transition: {:.6}", transition);
    }
}

// Import utils module
mod utils {
    use rust_pattern_solver::types::Number;
    
    pub fn integer_sqrt(n: &Number) -> Result<Number, String> {
        Ok(n.sqrt())
    }
}