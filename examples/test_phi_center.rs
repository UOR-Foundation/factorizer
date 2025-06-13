use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;

fn main() {
    println!("=== Testing Phi Center Calculation ===\n");
    
    let test_cases = vec![
        (25117u64, 151u64, 167u64),
        (10403, 101, 103),
        (143, 11, 13),
    ];
    
    for (n, p, q) in test_cases {
        println!("n = {} = {} × {}", n, p, q);
        
        // Calculate phi coordinates
        let phi: f64 = 1.618033988749895;
        let n_phi = (n as f64).ln() / phi.ln();
        let p_phi = (p as f64).ln() / phi.ln();
        let q_phi = (q as f64).ln() / phi.ln();
        
        println!("  n_φ = {:.6}", n_phi);
        println!("  p_φ = {:.6}, q_φ = {:.6}", p_phi, q_phi);
        println!("  p_φ + q_φ = {:.6} (diff = {:.6})", p_phi + q_phi, (p_phi + q_phi - n_phi).abs());
        
        // Calculate search center
        let center_from_phi = phi.powf(n_phi / 2.0);
        println!("  φ^(n_φ/2) = {:.2}", center_from_phi);
        println!("  sqrt(n) = {:.2}", (n as f64).sqrt());
        println!("  Center as u64: {}", center_from_phi as u64);
        
        // Check if factors are near center
        let center = center_from_phi as u64;
        println!("  |p - center| = {}", (p as i64 - center as i64).abs());
        println!("  |q - center| = {}", (q as i64 - center as i64).abs());
        
        println!();
    }
}