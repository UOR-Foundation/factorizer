//! Tune Universal Pattern for balanced semiprimes (p ≈ q)
//! 
//! The goal is to identify patterns in how the Universal Pattern
//! performs on balanced semiprimes and tune parameters accordingly.

use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;
use rust_pattern_solver::utils;

fn analyze_balanced_semiprime(pattern: &mut UniversalPattern, p: u64, q: u64) {
    let n = Number::from(p) * Number::from(q);
    let ratio = p as f64 / q as f64;
    
    println!("\n=== Analyzing {} = {} × {} ===", n, p, q);
    println!("Bit length: {}", n.bit_length());
    println!("p/q ratio: {:.6}", ratio);
    
    // Analyze recognition
    if let Ok(recognition) = pattern.recognize(&n) {
        println!("\nRecognition:");
        println!("  φ-component: {:.6}", recognition.phi_component);
        println!("  π-component: {:.6}", recognition.pi_component);
        println!("  e-component: {:.6}", recognition.e_component);
        println!("  Unity phase: {:.6}", recognition.unity_phase);
        
        // Check relationships
        let phi_pi_ratio = recognition.phi_component / (recognition.pi_component + 1e-10);
        let pi_e_ratio = recognition.pi_component / (recognition.e_component + 1e-10);
        let phi_e_ratio = recognition.phi_component / recognition.e_component;
        
        println!("\nComponent ratios:");
        println!("  φ/π: {:.6}", phi_pi_ratio);
        println!("  π/e: {:.6}", pi_e_ratio);
        println!("  φ/e: {:.6}", phi_e_ratio);
        
        // Analyze formalization
        if let Ok(formalization) = pattern.formalize(recognition) {
            println!("\nFormalization:");
            println!("  Resonance peaks: {:?}", formalization.resonance_peaks);
            
            if let Some(product_phase) = formalization.factor_encoding.get("product_phase") {
                println!("  Product phase: {:.6}", product_phase);
            }
            if let Some(sum_resonance) = formalization.factor_encoding.get("sum_resonance") {
                println!("  Sum resonance: {:.6}", sum_resonance);
                
                // Check if sum resonance correctly estimates p + q
                let sqrt_n = utils::integer_sqrt(&n).unwrap();
                let phi = 1.618033988749895;
                let sum_estimate = sum_resonance * sqrt_n.to_f64().unwrap_or(1.0) / phi;
                let actual_sum = p + q;
                let error = ((sum_estimate - actual_sum as f64) / actual_sum as f64).abs() * 100.0;
                println!("  Sum estimate: {:.0} (actual: {}, error: {:.2}%)", sum_estimate, actual_sum, error);
            }
            
            // Try execution and see which method works
            match pattern.execute(formalization) {
                Ok(factors) => {
                    println!("\n✓ Factorization successful!");
                    println!("  Method: {}", factors.method);
                }
                Err(_) => {
                    println!("\n✗ Factorization failed");
                    
                    // Analyze why it might have failed
                    let sqrt_n = utils::integer_sqrt(&n).unwrap().to_f64().unwrap_or(1.0);
                    let p_distance = (p as f64 - sqrt_n).abs();
                    let q_distance = (q as f64 - sqrt_n).abs();
                    
                    println!("\nFailure analysis:");
                    println!("  sqrt(n): {:.0}", sqrt_n);
                    println!("  p distance from sqrt(n): {:.0}", p_distance);
                    println!("  q distance from sqrt(n): {:.0}", q_distance);
                    println!("  Search radius needed: {:.0}", p_distance.max(q_distance));
                }
            }
        }
    }
}

fn main() -> rust_pattern_solver::Result<()> {
    println!("=== Universal Pattern - Balanced Semiprime Analysis ===");
    
    let mut pattern = UniversalPattern::new();
    
    // Small balanced semiprimes (should work)
    analyze_balanced_semiprime(&mut pattern, 11, 13);
    analyze_balanced_semiprime(&mut pattern, 29, 31);
    analyze_balanced_semiprime(&mut pattern, 71, 73);
    
    // Medium balanced semiprimes  
    analyze_balanced_semiprime(&mut pattern, 101, 103);
    analyze_balanced_semiprime(&mut pattern, 151, 157);
    analyze_balanced_semiprime(&mut pattern, 311, 313);
    
    // Larger balanced semiprimes (likely to fail)
    analyze_balanced_semiprime(&mut pattern, 1009, 1013);
    analyze_balanced_semiprime(&mut pattern, 10007, 10009);
    
    // Very balanced (p and q differ by 2)
    println!("\n=== Twin Prime Products ===");
    analyze_balanced_semiprime(&mut pattern, 3, 5);
    analyze_balanced_semiprime(&mut pattern, 5, 7);
    analyze_balanced_semiprime(&mut pattern, 11, 13);
    analyze_balanced_semiprime(&mut pattern, 17, 19);
    analyze_balanced_semiprime(&mut pattern, 29, 31);
    analyze_balanced_semiprime(&mut pattern, 41, 43);
    
    Ok(())
}