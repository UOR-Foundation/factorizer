use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;

fn main() {
    let mut pattern = UniversalPattern::new();
    
    // Test 25117 = 151 × 167
    let n = Number::from(25117u32);
    println!("Testing phi-sum guided search directly for 25117 = 151 × 167");
    
    // Skip small factor search by setting bit length check
    println!("n bit length: {}", n.bit_length());
    
    // Get recognition and formalization
    let recognition = pattern.recognize(&n).unwrap();
    let formalization = pattern.formalize(recognition).unwrap();
    
    println!("n_φ = {:.6}", formalization.universal_coordinates[0]);
    
    // Calculate center more accurately
    let n_phi = formalization.universal_coordinates[0];
    let phi: f64 = 1.618033988749895;
    
    // For balanced semiprimes, both factors are close to sqrt(n)
    // So we need to search around sqrt(n), not φ^(n_φ/2)
    let sqrt_n = 25117f64.sqrt();
    println!("sqrt(n) = {:.2}", sqrt_n);
    println!("φ^(n_φ/2) = {:.2}", phi.powf(n_phi / 2.0));
    
    // The issue might be integer truncation
    let center_from_phi = phi.powf(n_phi / 2.0);
    println!("Center estimate as u64: {}", center_from_phi as u64);
    println!("Rounded center: {}", center_from_phi.round() as u64);
    
    // Check where 151 and 167 are relative to search
    let search_center = center_from_phi as u64;
    let search_radius = 25117f64.powf(0.3).max(100.0) as u64;
    println!("\nSearch center: {}", search_center);
    println!("Search radius: {}", search_radius);
    println!("151 distance from center: {}", (151i64 - search_center as i64).abs());
    println!("167 distance from center: {}", (167i64 - search_center as i64).abs());
    
    // Test divisibility directly
    println!("\nTesting divisibility:");
    println!("25117 % 151 = {}", n.clone() % &Number::from(151u32));
    println!("25117 % 167 = {}", n.clone() % &Number::from(167u32));
    
    // Calculate phi values
    let p_phi = 151f64.ln() / phi.ln();
    let q_phi = 167f64.ln() / phi.ln();
    println!("\np_φ = {:.6}, q_φ = {:.6}", p_phi, q_phi);
    println!("p_φ + q_φ = {:.6}", p_phi + q_phi);
    println!("Difference from n_φ: {:.6}", (p_phi + q_phi - n_phi).abs());
}