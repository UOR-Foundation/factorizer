//! Pattern Diagnostics - Analyze why factorization fails for specific numbers
//!
//! This tool helps tune the universal pattern by examining the relationships
//! between factors in universal space.

use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;

fn analyze_semiprime(n: u64, p: u64, q: u64) {
    println!("\n{}", "=".repeat(60));
    println!("Analyzing {} = {} × {} (balanced ratio: {:.4})", n, p, q, p as f64 / q as f64);
    println!("{}", "=".repeat(60));
    
    let mut pattern = UniversalPattern::new();
    let n_num = Number::from(n);
    
    // Get recognition
    let recognition = pattern.recognize(&n_num).unwrap();
    let formalization = pattern.formalize(recognition.clone()).unwrap();
    
    println!("\nUniversal Coordinates:");
    println!("  n: φ={:.6}, π={:.6}, e={:.6}, unity={:.6}", 
        formalization.universal_coordinates[0],
        formalization.universal_coordinates[1], 
        formalization.universal_coordinates[2],
        formalization.universal_coordinates[3]
    );
    
    // Project p and q into universal space
    let p_num = Number::from(p);
    let q_num = Number::from(q);
    
    // Direct calculation using the exact formulas
    let phi: f64 = 1.618033988749895;
    let pi = std::f64::consts::PI;
    let e = std::f64::consts::E;
    
    let p_phi = (p as f64).ln() / phi.ln();
    let q_phi = (q as f64).ln() / phi.ln();
    let p_pi = (p as f64 * phi) % pi;
    let q_pi = (q as f64 * phi) % pi;
    let p_e = ((p + 1) as f64).ln() / e;
    let q_e = ((q + 1) as f64).ln() / e;
    let p_unity = p as f64 / (p as f64 + phi + pi + e);
    let q_unity = q as f64 / (q as f64 + phi + pi + e);
    
    println!("\n  p: φ={:.6}, π={:.6}, e={:.6}, unity={:.6}", p_phi, p_pi, p_e, p_unity);
    println!("  q: φ={:.6}, π={:.6}, e={:.6}, unity={:.6}", q_phi, q_pi, q_e, q_unity);
    
    // Check invariant relationships
    println!("\nInvariant Relationships:");
    println!("  Golden Ratio: p_φ/q_φ = {:.6} (target: {:.6}, error: {:.4})", 
        p_phi/q_phi, phi, (p_phi/q_phi - phi).abs());
    println!("  Harmonic: p_π+q_π = {:.6}, n_π = {:.6} (error: {:.4})", 
        p_pi + q_pi, formalization.universal_coordinates[1], 
        ((p_pi + q_pi) - formalization.universal_coordinates[1]).abs());
    println!("  Exponential: p_e*q_e = {:.6}, n_e = {:.6} (error: {:.4})", 
        p_e * q_e, formalization.universal_coordinates[2],
        (p_e * q_e - formalization.universal_coordinates[2]).abs());
    
    // Additional pattern analysis
    println!("\nResonance Analysis:");
    println!("  Resonance peaks: {:?}", formalization.resonance_peaks);
    println!("  Harmonic series (first 3): {:?}", 
        &formalization.harmonic_series.as_slice().unwrap()[..3.min(formalization.harmonic_series.len())]);
    
    // Factor encoding analysis
    println!("\nFactor Encoding:");
    for (key, value) in &formalization.factor_encoding {
        println!("  {}: {:.6}", key, value);
    }
    
    // Search range analysis
    let sqrt_n = (n as f64).sqrt();
    println!("\nSearch Analysis:");
    println!("  sqrt(n) = {:.2}", sqrt_n);
    println!("  p distance from sqrt: {}", (p as f64 - sqrt_n).abs() as u64);
    println!("  q distance from sqrt: {}", (q as f64 - sqrt_n).abs() as u64);
    println!("  n^0.25 = {:.2} (search radius)", (n as f64).powf(0.25));
    
    // Try to understand why it might fail
    let search_start = sqrt_n - (n as f64).powf(0.25);
    let search_end = sqrt_n + (n as f64).powf(0.25);
    println!("  Search range: {:.0} to {:.0}", search_start, search_end);
    println!("  p in range: {}", p as f64 >= search_start && p as f64 <= search_end);
    println!("  q in range: {}", q as f64 >= search_start && q as f64 <= search_end);
}

fn main() {
    println!("=== Pattern Diagnostics for Factorization ===\n");
    
    // Analyze successful cases first
    println!("SUCCESSFUL CASES (reference):");
    analyze_semiprime(35, 5, 7);      // Works
    analyze_semiprime(77, 7, 11);     // Works
    
    // Analyze failing balanced semiprimes
    println!("\n\nFAILING CASES (to diagnose):");
    analyze_semiprime(25117, 151, 167);    // Fails - twin-like primes
    analyze_semiprime(10403, 101, 103);    // Classic balanced case
    analyze_semiprime(2147483629, 46337, 46349); // Larger twin primes
    
    // Pattern summary
    println!("\n\n{}", "=".repeat(60));
    println!("PATTERN SUMMARY");
    println!("{}", "=".repeat(60));
    
    println!("\nObservations:");
    println!("1. Small factor method works when one factor is small (<100)");
    println!("2. Balanced semiprimes where both factors are > 100 fail");
    println!("3. The invariant relationships may need different tolerances");
    println!("4. Search radius may be too small for certain number ranges");
}