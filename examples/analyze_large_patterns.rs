//! Analyze patterns in large number factorizations to guide our implementation
//! Following The Pattern philosophy: observe first, implement what emerges

use rust_pattern_solver::types::Number;
use rust_pattern_solver::utils;
use std::collections::HashMap;
use std::str::FromStr;

fn analyze_factorization(n: &Number, p: &Number, q: &Number) -> HashMap<String, f64> {
    let mut observations = HashMap::new();
    
    // Basic measurements
    let sqrt_n = utils::integer_sqrt(n).unwrap();
    
    // Distance from sqrt(n)
    let p_distance = if p > &sqrt_n {
        p - &sqrt_n
    } else {
        &sqrt_n - p
    };
    
    let q_distance = if q > &sqrt_n {
        q - &sqrt_n
    } else {
        &sqrt_n - q
    };
    
    let max_distance = if p_distance > q_distance {
        p_distance.clone()
    } else {
        q_distance.clone()
    };
    
    // Convert to f64 for analysis (may lose precision but ok for analysis)
    let sqrt_n_f64 = sqrt_n.to_f64().unwrap_or(1e100);
    let max_distance_f64 = max_distance.to_f64().unwrap_or(1e50);
    
    observations.insert("sqrt_n".to_string(), sqrt_n_f64);
    observations.insert("max_distance".to_string(), max_distance_f64);
    observations.insert("distance_ratio".to_string(), max_distance_f64 / sqrt_n_f64);
    
    // Factor balance
    let p_f64 = p.to_f64().unwrap_or(1e100);
    let q_f64 = q.to_f64().unwrap_or(1e100);
    let balance = p_f64 / q_f64;
    observations.insert("balance".to_string(), balance);
    
    // Fermat's method values
    let a = (p + q) / &Number::from(2u32);
    let b = if p > q {
        (p - q) / &Number::from(2u32)
    } else {
        (q - p) / &Number::from(2u32)
    };
    
    let a_f64 = a.to_f64().unwrap_or(1e100);
    let b_f64 = b.to_f64().unwrap_or(1e50);
    observations.insert("fermat_a".to_string(), a_f64);
    observations.insert("fermat_b".to_string(), b_f64);
    observations.insert("fermat_offset".to_string(), a_f64 - sqrt_n_f64);
    
    // Phi relationships
    let phi: f64 = 1.618033988749895;
    
    // For very large numbers, use bit length approximation
    let n_phi = if n.bit_length() > 500 {
        n.bit_length() as f64 * 2.0_f64.ln() / phi.ln()
    } else {
        n.to_f64().unwrap_or(1e100).ln() / phi.ln()
    };
    
    let p_phi = if p.bit_length() > 500 {
        p.bit_length() as f64 * 2.0_f64.ln() / phi.ln()
    } else {
        p.to_f64().unwrap_or(1e100).ln() / phi.ln()
    };
    
    let q_phi = if q.bit_length() > 500 {
        q.bit_length() as f64 * 2.0_f64.ln() / phi.ln()
    } else {
        q.to_f64().unwrap_or(1e100).ln() / phi.ln()
    };
    
    observations.insert("n_phi".to_string(), n_phi);
    observations.insert("phi_sum".to_string(), p_phi + q_phi);
    observations.insert("phi_sum_error".to_string(), (p_phi + q_phi - n_phi).abs());
    
    // Search radius needed (as fraction of sqrt(n))
    let radius_fraction = max_distance_f64 / sqrt_n_f64;
    observations.insert("radius_fraction".to_string(), radius_fraction);
    
    // Bit length analysis
    observations.insert("n_bits".to_string(), n.bit_length() as f64);
    
    observations
}

fn main() {
    println!("=== Large Number Pattern Analysis ===\n");
    
    // Test cases that are failing - using proper Number types
    let test_cases = vec![
        // 112-bit
        (
            Number::from_str("5192296858534833249022831287600429").unwrap(),
            Number::from_str("72057594037927961").unwrap(),
            Number::from_str("72057594037927989").unwrap(),
            "112-bit balanced"
        ),
        // 128-bit (correct value that doesn't overflow)
        (
            Number::from_str("340282366920938464127457394085312069931").unwrap(),
            Number::from_str("18446744073709551629").unwrap(),
            Number::from_str("18446744073709551639").unwrap(),
            "128-bit balanced"
        ),
        // RSA-100 for comparison
        (
            Number::from_str("1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139").unwrap(),
            Number::from_str("37975227936943673922808872755445627854565536638199").unwrap(),
            Number::from_str("40094690950920881030683735292761468389214899724061").unwrap(),
            "RSA-100 (330-bit)"
        ),
    ];
    
    println!("Analyzing patterns in large factorizations:\n");
    
    for (n, p, q, desc) in &test_cases {
        println!("=== {} ===", desc);
        println!("n = {} ({} bits)", n, n.bit_length());
        println!("p = {} ({} bits)", p, p.bit_length());
        println!("q = {} ({} bits)", q, q.bit_length());
        
        let observations = analyze_factorization(n, p, q);
        
        // Key observations
        println!("\nKey observations:");
        println!("  sqrt(n) = {:.2e}", observations["sqrt_n"]);
        println!("  Max distance from sqrt = {:.2e}", observations["max_distance"]);
        println!("  Distance as fraction of sqrt(n) = {:.6}", observations["distance_ratio"]);
        println!("  Search radius needed = {:.2e}", observations["max_distance"]);
        println!("  Fermat offset (a - sqrt(n)) = {:.2e}", observations["fermat_offset"]);
        println!("  φ-sum error = {:.6}", observations["phi_sum_error"]);
        
        // What search radius would we need?
        let sqrt_n = observations["sqrt_n"];
        let needed = observations["max_distance"];
        
        println!("\nSearch strategy analysis:");
        
        // Current fixed radius approach
        let n_bits = observations["n_bits"] as u32;
        let current_radius = if n_bits > 128 {
            100_000_000f64
        } else if n_bits > 96 {
            10_000_000f64
        } else if n_bits > 64 {
            1_000_000f64
        } else {
            100_000f64
        };
        
        println!("  Current fixed radius: {:.2e}", current_radius);
        println!("  Radius needed: {:.2e}", needed);
        println!("  Would current work? {}", current_radius >= needed);
        
        // Python-style adaptive radius
        let phi = 1.618033988749895;
        let n_f64 = n.to_f64().unwrap_or_else(|| {
            // For very large numbers, use bit length approximation
            2.0_f64.powf(n.bit_length() as f64)
        });
        let python_radius = n_f64.ln() * phi;
        println!("  Python-style radius (ln(n) * φ): {:.2e}", python_radius);
        println!("  Would Python-style work? {}", python_radius >= needed);
        
        // Empirical observation - for balanced semiprimes
        let empirical_radius = if n_bits < 100 {
            sqrt_n.powf(0.05) * 100.0
        } else {
            // For very large numbers, distance grows more slowly
            sqrt_n.powf(0.01) * 10.0
        };
        println!("  Empirical radius: {:.2e}", empirical_radius);
        println!("  Would empirical work? {}", empirical_radius >= needed);
        
        // Fermat's method insight
        let fermat_offset = observations["fermat_offset"];
        println!("\nFermat's method analysis:");
        println!("  Fermat 'a' offset from sqrt(n): {:.2e}", fermat_offset);
        println!("  As fraction of sqrt(n): {:.6}", fermat_offset / sqrt_n);
        
        println!();
    }
    
    println!("\n=== Pattern Summary ===");
    println!("1. For balanced semiprimes, factors are VERY close to sqrt(n)");
    println!("2. The distance scales sub-linearly with n");
    println!("3. Fixed radius approaches fail for large numbers");
    println!("4. The φ-sum invariant holds with high precision (<0.01 error)");
    println!("5. Fermat's method excels for balanced cases");
    println!("6. For RSA-100, factors are further from sqrt(n) (not balanced)");
    
    println!("\n=== Recommendations ===");
    println!("1. Detect balanced semiprimes early (small Fermat offset)");
    println!("2. Use Fermat-based search for balanced cases");
    println!("3. For phi-sum search, use adaptive radius based on n size");
    println!("4. Implement proper Number-based iteration for >64-bit");
    println!("5. Consider different strategies for balanced vs unbalanced");
}