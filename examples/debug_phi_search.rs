//! Debug the phi-sum guided search to understand why it's not finding factors

use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;

fn debug_search(n: u64, p: u64, q: u64) {
    println!("\n{}", "=".repeat(60));
    println!("Debugging search for {} = {} × {}", n, p, q);
    println!("{}", "=".repeat(60));
    
    let mut pattern = UniversalPattern::new();
    let n_num = Number::from(n);
    
    // Get coordinates
    let recognition = pattern.recognize(&n_num).unwrap();
    let n_phi = recognition.phi_component;
    
    println!("\nn_φ = {:.6}", n_phi);
    println!("Target φ (n_φ/2) = {:.6}", n_phi / 2.0);
    
    // Calculate what the search center should be
    let phi: f64 = 1.618033988749895;
    let center_estimate = phi.powf(n_phi / 2.0);
    println!("Center estimate: φ^(n_φ/2) = {:.2}", center_estimate);
    println!("Actual sqrt(n) = {:.2}", (n as f64).sqrt());
    println!("Actual p = {}, q = {}", p, q);
    
    // Check the actual phi values of p and q
    let p_phi = (p as f64).ln() / phi.ln();
    let q_phi = (q as f64).ln() / phi.ln();
    println!("\nActual p_φ = {:.6}, q_φ = {:.6}", p_phi, q_phi);
    println!("Sum: p_φ + q_φ = {:.6} (should equal n_φ = {:.6})", p_phi + q_phi, n_phi);
    println!("Difference from n_φ: {:.6}", (p_phi + q_phi - n_phi).abs());
    
    // Check search parameters
    let search_center = center_estimate as u64;
    let search_radius = (n as f64).powf(0.3).max(100.0) as u64;
    println!("\nSearch center: {}", search_center);
    println!("Search radius: {}", search_radius);
    println!("Search range: {} to {}", 
        search_center.saturating_sub(search_radius),
        search_center + search_radius
    );
    
    // Check if p and q are in range
    let p_in_range = p >= search_center.saturating_sub(search_radius) && p <= search_center + search_radius;
    let q_in_range = q >= search_center.saturating_sub(search_radius) && q <= search_center + search_radius;
    println!("p ({}) in search range: {}", p, p_in_range);
    println!("q ({}) in search range: {}", q, q_in_range);
    
    // If not in range, what radius would we need?
    if !p_in_range || !q_in_range {
        let needed_radius_p = (p as i64 - search_center as i64).abs();
        let needed_radius_q = (q as i64 - search_center as i64).abs();
        let needed_radius = needed_radius_p.max(needed_radius_q);
        println!("\nRadius needed to find factors: {}", needed_radius);
        println!("Current radius / needed radius = {:.2}", search_radius as f64 / needed_radius as f64);
    }
    
    // Test the actual search
    println!("\nTesting phi_sum_guided_search...");
    let formalization = pattern.formalize(recognition).unwrap();
    
    // Try to execute
    match pattern.execute(formalization) {
        Ok(factors) => {
            println!("✓ Found factors: {} × {}", factors.p, factors.q);
            println!("  Method: {}", factors.method);
        }
        Err(e) => {
            println!("✗ Search failed: {}", e);
        }
    }
}

fn main() {
    println!("=== Debug Phi-Sum Guided Search ===");
    
    // Test cases that are failing
    debug_search(10403, 101, 103);     // Classic balanced case
    debug_search(25117, 151, 167);     // Larger balanced case
    debug_search(2147483629, 46337, 46349); // Large twin primes
    
    // Also test a working case for comparison
    debug_search(143, 11, 13);         // This should work
}