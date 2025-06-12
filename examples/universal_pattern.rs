//! Example: Universal Pattern demonstration
//!
//! This example shows how The Pattern works through universal constants
//! and the three-stage process: Recognition, Formalization, Execution.

use rust_pattern_solver::pattern::universal_pattern::{UniversalPattern, UniversalConstants};
use rust_pattern_solver::types::Number;
use std::str::FromStr;

fn main() -> rust_pattern_solver::Result<()> {
    println!("=== The Universal Pattern ===\n");

    // Show universal constants
    let constants = UniversalConstants::default();
    println!("Universal Constants:");
    println!("  φ (Golden Ratio) = {:.15}", constants.phi);
    println!("  π (Pi)           = {:.15}", constants.pi);
    println!("  e (Euler's)      = {:.15}", constants.e);
    println!("  β (2 - φ)        = {:.15}", constants.beta);
    println!("  γ (Euler-M)      = {:.15}", constants.gamma);
    println!();

    // Create Universal Pattern instance
    let mut pattern = UniversalPattern::new();

    // Test cases
    let test_cases = vec![
        (Number::from(15u32), "3 × 5"),
        (Number::from(143u32), "11 × 13"),
        (Number::from(323u32), "17 × 19"),
        (Number::from(667u32), "23 × 29"),
        (Number::from(10403u32), "101 × 103"),
    ];

    for (n, description) in test_cases {
        println!("Testing {} = {}", n, description);
        
        // Stage 1: Recognition
        match pattern.recognize(&n) {
            Ok(recognition) => {
                println!("  Recognition:");
                println!("    φ-component: {:.6}", recognition.phi_component);
                println!("    π-component: {:.6}", recognition.pi_component);
                println!("    e-component: {:.6}", recognition.e_component);
                println!("    Unity phase: {:.6}", recognition.unity_phase);
                println!("    Resonance field size: {}", recognition.resonance_field.len());
                
                // Stage 2: Formalization
                match pattern.formalize(recognition) {
                    Ok(formalization) => {
                        println!("  Formalization:");
                        println!("    Universal coordinates: {:?}", 
                                 formalization.universal_coordinates.iter()
                                     .map(|x| format!("{:.3}", x))
                                     .collect::<Vec<_>>());
                        println!("    Resonance peaks: {:?}", formalization.resonance_peaks);
                        println!("    Factor encoding keys: {:?}", 
                                 formalization.factor_encoding.keys()
                                     .collect::<Vec<_>>());
                        
                        // Stage 3: Execution
                        match pattern.execute(formalization) {
                            Ok(factors) => {
                                println!("  ✓ Factors found: {} × {} (method: {})", 
                                         factors.p, factors.q, factors.method);
                            }
                            Err(e) => {
                                println!("  ✗ Execution failed: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        println!("  Formalization failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("  Recognition failed: {}", e);
            }
        }
        println!();
    }

    // Demonstrate universal relationships
    println!("=== Universal Relationships ===\n");
    
    // Golden ratio relationships
    println!("Golden Ratio Properties:");
    println!("  φ² = φ + 1: {:.15} = {:.15}", 
             constants.phi * constants.phi, 
             constants.phi + 1.0);
    println!("  1/φ = φ - 1: {:.15} = {:.15}", 
             1.0 / constants.phi, 
             constants.phi - 1.0);
    println!("  φ + β = 2: {:.15}", constants.phi + constants.beta);
    
    // Test with a larger semiprime
    println!("\n=== Larger Number Test ===\n");
    
    let large_n = Number::from(1073741827u64); // Large prime
    println!("Testing {}-bit number: {}", large_n.bit_length(), large_n);
    
    match pattern.recognize(&large_n) {
        Ok(recognition) => {
            println!("Universal signature extracted:");
            println!("  φ-component: {:.6}", recognition.phi_component);
            println!("  π-component: {:.6}", recognition.pi_component);
            println!("  e-component: {:.6}", recognition.e_component);
            
            // Show how components relate
            let phi_pi_ratio = recognition.phi_component / recognition.pi_component;
            let pi_e_ratio = recognition.pi_component / recognition.e_component;
            
            println!("\nComponent relationships:");
            println!("  φ/π ratio: {:.6}", phi_pi_ratio);
            println!("  π/e ratio: {:.6}", pi_e_ratio);
            
            // Check if ratios are near universal constants
            if (phi_pi_ratio - constants.phi).abs() < 0.1 {
                println!("  ✓ φ/π ratio is near φ!");
            }
            if (pi_e_ratio - constants.beta).abs() < 0.1 {
                println!("  ✓ π/e ratio is near β!");
            }
        }
        Err(e) => {
            println!("Recognition failed: {}", e);
        }
    }

    Ok(())
}