use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;

fn main() {
    println!("=== Testing Phi Precision for 25117 ===\n");
    
    let mut pattern = UniversalPattern::new();
    let n = Number::from(25117u32);
    
    // Get the phi coordinate of n
    let recognition = pattern.recognize(&n).unwrap();
    let n_phi = recognition.phi_component;
    println!("n_φ = {:.10}", n_phi);
    
    // Test the actual factors
    let p = Number::from(151u32);
    let q = Number::from(167u32);
    
    // Calculate their phi coordinates with high precision
    let phi: f64 = 1.618033988749895;
    let p_phi = 151f64.ln() / phi.ln();
    let q_phi = 167f64.ln() / phi.ln();
    
    println!("p_φ = {:.10}", p_phi);
    println!("q_φ = {:.10}", q_phi);
    println!("p_φ + q_φ = {:.10}", p_phi + q_phi);
    println!("Difference: {:.10}", (p_phi + q_phi - n_phi).abs());
    
    // Test divisibility
    println!("\nDivisibility check:");
    println!("25117 % 151 = {}", n.clone() % &p);
    println!("25117 / 151 = {}", &n / &p);
    
    // Test the search logic directly
    println!("\nSearch logic test:");
    let center_estimate = phi.powf(n_phi / 2.0);
    let search_center = center_estimate as u64;
    println!("Center estimate: {:.2}", center_estimate);
    println!("Search center: {}", search_center);
    
    // Check if 151 is found
    let offset_for_151 = (151i64 - search_center as i64).abs() as u64;
    println!("Offset needed for 151: {}", offset_for_151);
    
    let search_radius = 25117f64.powf(0.3).max(100.0) as u64;
    println!("Search radius: {}", search_radius);
    println!("151 is within radius: {}", offset_for_151 <= search_radius);
    
    // Test the phi extraction method
    let formalization = pattern.formalize(recognition).unwrap();
    println!("\nTesting phi_sum_guided_search directly:");
    
    // Try to execute and see what happens
    match pattern.execute(formalization) {
        Ok(factors) => {
            println!("✓ Found factors: {} × {}", factors.p, factors.q);
            println!("  Method: {}", factors.method);
        }
        Err(e) => {
            println!("✗ Failed: {}", e);
        }
    }
}