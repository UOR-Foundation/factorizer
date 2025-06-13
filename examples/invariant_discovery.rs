//! Discover the actual invariant relationships between factors
//!
//! This tool analyzes many factorizations to find the true invariants

use rust_pattern_solver::types::Number;
use std::collections::HashMap;

struct FactorProjection {
    n: u64,
    p: u64,
    q: u64,
    n_phi: f64,
    n_pi: f64,
    n_e: f64,
    n_unity: f64,
    p_phi: f64,
    p_pi: f64,
    p_e: f64,
    p_unity: f64,
    q_phi: f64,
    q_pi: f64,
    q_e: f64,
    q_unity: f64,
}

fn project_number(n: u64) -> (f64, f64, f64, f64) {
    let phi: f64 = 1.618033988749895;
    let pi = std::f64::consts::PI;
    let e = std::f64::consts::E;
    
    let n_phi = (n as f64).ln() / phi.ln();
    let n_pi = (n as f64 * phi) % pi;
    let n_e = ((n + 1) as f64).ln() / e;
    let n_unity = n as f64 / (n as f64 + phi + pi + e);
    
    (n_phi, n_pi, n_e, n_unity)
}

fn analyze_relationships(projections: &[FactorProjection]) {
    println!("=== Invariant Relationship Discovery ===\n");
    
    // Analyze various potential relationships
    println!("1. Golden Ratio Relationships:");
    for proj in projections {
        let ratio1 = proj.p_phi / proj.q_phi;
        let ratio2 = proj.p_phi + proj.q_phi - proj.n_phi;
        let ratio3 = (proj.p_phi * proj.q_phi) / proj.n_phi;
        println!("  n={}: p_φ/q_φ={:.4}, (p_φ+q_φ)-n_φ={:.4}, (p_φ*q_φ)/n_φ={:.4}", 
                proj.n, ratio1, ratio2, ratio3);
    }
    
    println!("\n2. Pi (Circular) Relationships:");
    for proj in projections {
        let sum_diff = (proj.p_pi + proj.q_pi) - proj.n_pi;
        let product = proj.p_pi * proj.q_pi;
        let modular = ((proj.p as f64 * proj.q as f64) * 1.618033988749895) % std::f64::consts::PI;
        println!("  n={}: (p_π+q_π)-n_π={:.4}, p_π*q_π={:.4}, (p*q*φ)%π={:.4}, n_π={:.4}", 
                proj.n, sum_diff, product, modular, proj.n_pi);
    }
    
    println!("\n3. Exponential Relationships:");
    for proj in projections {
        let product_ratio = (proj.p_e * proj.q_e) / proj.n_e;
        let sum_ratio = (proj.p_e + proj.q_e) / proj.n_e;
        let diff = proj.n_e - (proj.p_e + proj.q_e);
        println!("  n={}: (p_e*q_e)/n_e={:.4}, (p_e+q_e)/n_e={:.4}, n_e-(p_e+q_e)={:.4}", 
                proj.n, product_ratio, sum_ratio, diff);
    }
    
    println!("\n4. Combined Relationships:");
    for proj in projections {
        // Try combinations of coordinates
        let comb1 = (proj.p_phi * proj.q_e) / (proj.n_phi * proj.n_e);
        let comb2 = (proj.p_phi + proj.q_phi) / (proj.p_e + proj.q_e);
        let comb3 = (proj.p_pi * proj.q_pi) / (proj.p_phi * proj.q_phi);
        println!("  n={}: (p_φ*q_e)/(n_φ*n_e)={:.4}, (p_φ+q_φ)/(p_e+q_e)={:.4}, (p_π*q_π)/(p_φ*q_φ)={:.4}", 
                proj.n, comb1, comb2, comb3);
    }
    
    // Statistical analysis
    println!("\n5. Statistical Summary:");
    
    // For p_φ + q_φ - n_φ
    let phi_diffs: Vec<f64> = projections.iter()
        .map(|p| p.p_phi + p.q_phi - p.n_phi)
        .collect();
    let phi_mean = phi_diffs.iter().sum::<f64>() / phi_diffs.len() as f64;
    let phi_std = (phi_diffs.iter()
        .map(|x| (x - phi_mean).powi(2))
        .sum::<f64>() / phi_diffs.len() as f64)
        .sqrt();
    println!("  (p_φ + q_φ) - n_φ: mean={:.4}, std={:.4}", phi_mean, phi_std);
    
    // For ln(n) relationship
    for proj in projections {
        let expected_sum = 2.0 * (proj.n as f64).sqrt().ln() / 1.618033988749895_f64.ln();
        let actual_sum = proj.p_phi + proj.q_phi;
        let error = (actual_sum - expected_sum).abs();
        println!("  n={}: Expected p_φ+q_φ={:.4}, Actual={:.4}, Error={:.4}", 
                proj.n, expected_sum, actual_sum, error);
    }
}

fn main() {
    let test_cases = vec![
        (35, 5, 7),
        (77, 7, 11),
        (143, 11, 13),
        (221, 13, 17),
        (323, 17, 19),
        (437, 19, 23),
        (667, 23, 29),
        (899, 29, 31),
        (1147, 31, 37),
        (1517, 37, 41),
        (1763, 41, 43),
        (2021, 43, 47),
        (10403, 101, 103),
        (25117, 151, 167),
    ];
    
    let mut projections = Vec::new();
    
    for (n, p, q) in test_cases {
        let (n_phi, n_pi, n_e, n_unity) = project_number(n);
        let (p_phi, p_pi, p_e, p_unity) = project_number(p);
        let (q_phi, q_pi, q_e, q_unity) = project_number(q);
        
        projections.push(FactorProjection {
            n, p, q,
            n_phi, n_pi, n_e, n_unity,
            p_phi, p_pi, p_e, p_unity,
            q_phi, q_pi, q_e, q_unity,
        });
    }
    
    analyze_relationships(&projections);
    
    println!("\n=== Key Observations ===");
    println!("1. The π-coordinate is NOT conserved in the way we expected");
    println!("2. The relationship (p_φ + q_φ) - n_φ appears to be nearly constant!");
    println!("3. For balanced semiprimes, p_φ + q_φ ≈ 2*ln(sqrt(n))/ln(φ)");
    println!("4. The exponential relationship needs adjustment");
}