fn main() {
    println!("=== Testing 25117 Factorization ===\n");
    
    // Basic math checks
    println!("25117 = 151 × 167? {}", 151 * 167 == 25117);
    println!("sqrt(25117) = {:.2}", (25117f64).sqrt());
    
    // Phi calculations
    let phi: f64 = 1.618033988749895;
    let n_phi = 25117f64.ln() / phi.ln();
    let p_phi = 151f64.ln() / phi.ln();
    let q_phi = 167f64.ln() / phi.ln();
    
    println!("\nPhi coordinates:");
    println!("n_φ = {:.10}", n_phi);
    println!("p_φ = {:.10}", p_phi);
    println!("q_φ = {:.10}", q_phi);
    println!("p_φ + q_φ = {:.10}", p_phi + q_phi);
    println!("Difference: {:.10}", (p_phi + q_phi - n_phi).abs());
    println!("Within 0.01 threshold? {}", (p_phi + q_phi - n_phi).abs() < 0.01);
    
    // Search center calculation
    let center_estimate = phi.powf(n_phi / 2.0);
    println!("\nSearch center:");
    println!("φ^(n_φ/2) = {:.6}", center_estimate);
    println!("As u64: {}", center_estimate as u64);
    println!("Rounded: {}", center_estimate.round() as u64);
    
    // Search logic
    let search_center = center_estimate as u64;
    let search_radius = 25117f64.powf(0.3) as u64;
    println!("\nSearch parameters:");
    println!("Center: {}", search_center);
    println!("Radius: {}", search_radius);
    println!("Range: {} to {}", search_center.saturating_sub(search_radius), search_center + search_radius);
    
    // Check if factors are in range
    println!("\nFactor positions:");
    println!("151 offset from center: {}", (151i64 - search_center as i64).abs());
    println!("167 offset from center: {}", (167i64 - search_center as i64).abs());
    
    // Simulate the search
    println!("\nSimulating search:");
    let mut found = false;
    for offset in 0..=search_radius {
        if offset == 0 {
            if search_center > 1 && 25117 % search_center == 0 {
                let other = 25117 / search_center;
                println!("  At offset 0: {} × {} = {}", search_center, other, search_center * other);
                if search_center * other == 25117 {
                    found = true;
                    break;
                }
            }
        } else {
            // Check positive direction
            let candidate = search_center + offset;
            if candidate > 1 && 25117 % candidate == 0 {
                let other = 25117 / candidate;
                println!("  At offset +{}: {} × {} = {}", offset, candidate, other, candidate * other);
                if candidate * other == 25117 {
                    // Check phi invariant
                    let p_phi_test = (candidate as f64).ln() / phi.ln();
                    let q_phi_test = (other as f64).ln() / phi.ln();
                    let diff = (p_phi_test + q_phi_test - n_phi).abs();
                    println!("    Phi sum difference: {:.10}", diff);
                    if diff < 0.01 {
                        found = true;
                        break;
                    }
                }
            }
            
            // Check negative direction
            if search_center >= offset {
                let candidate = search_center - offset;
                if candidate > 1 && 25117 % candidate == 0 {
                    let other = 25117 / candidate;
                    println!("  At offset -{}: {} × {} = {}", offset, candidate, other, candidate * other);
                    if candidate * other == 25117 {
                        // Check phi invariant
                        let p_phi_test = (candidate as f64).ln() / phi.ln();
                        let q_phi_test = (other as f64).ln() / phi.ln();
                        let diff = (p_phi_test + q_phi_test - n_phi).abs();
                        println!("    Phi sum difference: {:.10}", diff);
                        if diff < 0.01 {
                            found = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    
    println!("\nFactors found: {}", found);
}