use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;

fn main() {
    let mut pattern = UniversalPattern::new();
    
    // Test case: 10403 = 101 × 103 (very balanced)
    let n = Number::from(10403u32);
    let p = 101u32;
    let q = 103u32;
    
    println!("Testing balanced semiprime: {} = {} × {}", n, p, q);
    println!("sqrt(n) = {}", 101.99);
    println!("p distance from sqrt: {}", (101.0_f64 - 101.99).abs());
    println!("q distance from sqrt: {}", (103.0_f64 - 101.99).abs());
    
    // Project factors into universal space
    let p_num = Number::from(p);
    let q_num = Number::from(q);
    
    // Compute projections manually
    println!("\nDirect projection of factors:");
    
    // φ-coordinates
    let p_phi = p_num.to_f64().unwrap().ln() / 1.618033988749895_f64.ln();
    let q_phi = q_num.to_f64().unwrap().ln() / 1.618033988749895_f64.ln();
    let n_phi = n.to_f64().unwrap().ln() / 1.618033988749895_f64.ln();
    println!("  p φ-coord: {:.6}, q φ-coord: {:.6}, n φ-coord: {:.6}", p_phi, q_phi, n_phi);
    println!("  p_phi / q_phi = {:.6} (should be ≈ 1.618 for golden ratio)", p_phi / q_phi);
    
    // π-coordinates
    let p_pi = (p as f64 * 1.618033988749895) % 3.141592653589793;
    let q_pi = (q as f64 * 1.618033988749895) % 3.141592653589793;
    let n_pi = (n.to_f64().unwrap() * 1.618033988749895) % 3.141592653589793;
    println!("  p π-coord: {:.6}, q π-coord: {:.6}, n π-coord: {:.6}", p_pi, q_pi, n_pi);
    println!("  p_pi + q_pi = {:.6}, n_pi = {:.6} (should be equal for harmonic)", p_pi + q_pi, n_pi);
    
    // e-coordinates
    let p_e = (p as f64 + 1.0).ln() / 2.718281828459045;
    let q_e = (q as f64 + 1.0).ln() / 2.718281828459045;
    let n_e = (n.to_f64().unwrap() + 1.0).ln() / 2.718281828459045;
    println!("  p e-coord: {:.6}, q e-coord: {:.6}, n e-coord: {:.6}", p_e, q_e, n_e);
    println!("  p_e * q_e = {:.6}, n_e = {:.6} (ratio: {:.6})", p_e * q_e, n_e, (p_e * q_e) / n_e);
    
    // Try recognition
    match pattern.recognize(&n) {
        Ok(recognition) => {
            println!("\nRecognition successful:");
            println!("  φ-component: {:.6}", recognition.phi_component);
            println!("  π-component: {:.6}", recognition.pi_component);
            println!("  e-component: {:.6}", recognition.e_component);
            
            // Try formalization
            match pattern.formalize(recognition) {
                Ok(formalization) => {
                    println!("\nFormalization successful:");
                    println!("  Resonance peaks: {:?}", formalization.resonance_peaks);
                    println!("  Universal coordinates: {:?}", formalization.universal_coordinates);
                    
                    // Try execution
                    match pattern.execute(formalization) {
                        Ok(factors) => {
                            println!("\n✓ Factorization successful!");
                            println!("  Factors: {} × {}", factors.p, factors.q);
                            println!("  Method: {}", factors.method);
                        }
                        Err(e) => {
                            println!("\n✗ Execution failed: {}", e);
                        }
                    }
                }
                Err(e) => println!("Formalization failed: {}", e),
            }
        }
        Err(e) => println!("Recognition failed: {}", e),
    }
}