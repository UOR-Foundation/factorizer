fn main() {
    // Test 2147483629 = 46337 × 46349
    let n = 2147483629u64;
    let p = 46337u64;
    let q = 46349u64;
    
    println!("n = {} = {} × {}", n, p, q);
    println!("Verification: {} × {} = {}", p, q, p * q);
    
    // Phi calculations
    let phi: f64 = 1.618033988749895;
    let n_phi = (n as f64).ln() / phi.ln();
    let target_phi = n_phi / 2.0;
    let center_estimate = phi.powf(target_phi);
    
    println!("\nn_φ = {:.6}", n_phi);
    println!("Target φ (n_φ/2) = {:.6}", target_phi);
    println!("φ^(n_φ/2) = {:.2}", center_estimate);
    println!("sqrt(n) = {:.2}", (n as f64).sqrt());
    println!("Center as u64: {}", center_estimate as u64);
    
    // Check distance from factors
    let center = center_estimate as u64;
    println!("\nDistance from center:");
    println!("|{} - {}| = {}", p, center, (p as i64 - center as i64).abs());
    println!("|{} - {}| = {}", q, center, (q as i64 - center as i64).abs());
    
    // Check search radius
    let bit_length = 31; // From test output
    let search_radius = if bit_length > 30 { 10_000 } else { 100 };
    println!("\nSearch radius for bit length {}: {}", bit_length, search_radius);
    println!("Factors within radius: {}", 
        (p as i64 - center as i64).abs() <= search_radius as i64 &&
        (q as i64 - center as i64).abs() <= search_radius as i64
    );
}