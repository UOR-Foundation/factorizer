//! Analyze the universal constants and their relationships to understand scaling

use std::f64::consts::{E, PI};

fn main() {
    println!("=== Universal Constants Analysis ===\n");
    
    // Universal constants
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let beta = 2.0 - phi;
    let gamma = 0.5772156649015329; // Euler-Mascheroni
    
    // Pattern constants from Python
    let resonance_decay_alpha = 1.1750566516490533;
    let phase_coupling_beta = 0.19968406830149554;
    let scale_transition_gamma = 12.41605776553433;
    
    println!("Universal Constants:");
    println!("  φ (golden ratio) = {:.15}", phi);
    println!("  π = {:.15}", PI);
    println!("  e = {:.15}", E);
    println!("  β (2-φ) = {:.15}", beta);
    println!("  γ (Euler-Mascheroni) = {:.15}", gamma);
    
    println!("\nPattern Constants (from empirical observation):");
    println!("  resonance_decay_alpha = {:.15}", resonance_decay_alpha);
    println!("  phase_coupling_beta = {:.15}", phase_coupling_beta);
    println!("  scale_transition_gamma = {:.15}", scale_transition_gamma);
    
    println!("\nDerived Relationships:");
    println!("  φ/π = {:.15}", phi / PI);
    println!("  e/φ = {:.15}", E / phi);
    println!("  2π = {:.15}", 2.0 * PI);
    println!("  scale_transition_gamma * ln(φ) = {:.15}", scale_transition_gamma * phi.ln());
    
    // Analyze scaling behavior
    println!("\n=== Scaling Analysis ===");
    println!("\nFor balanced semiprimes, empirical distance from sqrt(n):");
    
    let test_bits = vec![32, 64, 96, 113, 129, 161, 201, 330];
    let observed_distances = vec![
        0.0,    // 32-bit (effectively 0)
        3.0,    // 64-bit (observed ~3)
        7.0,    // 96-bit (observed ~7)
        14.0,   // 113-bit (observed ~14)
        5.0,    // 129-bit (observed ~5)
        -1.0,   // 161-bit (failed)
        -1.0,   // 201-bit (failed)
        -1.0,   // 330-bit (not balanced)
    ];
    
    println!("\nBits | Distance | Predicted (various models)");
    println!("-----|----------|---------------------------");
    
    for i in 0..test_bits.len() {
        let bits = test_bits[i] as f64;
        let dist = observed_distances[i];
        
        // Try different scaling models
        let model1 = scale_transition_gamma * bits.ln() / bits.ln().ln(); // sub-logarithmic
        let model2 = scale_transition_gamma * (bits / 50.0).powf(2.0);    // quadratic in bits/50
        let model3 = scale_transition_gamma * bits.ln() * phase_coupling_beta; // logarithmic with decay
        let model4 = bits.ln() * resonance_decay_alpha;                    // simple log scaling
        
        if dist >= 0.0 {
            println!("{:4} | {:8.1} | M1: {:6.1}, M2: {:6.1}, M3: {:6.1}, M4: {:6.1}", 
                    test_bits[i], dist, model1, model2, model3, model4);
        } else {
            println!("{:4} | FAILED   | M1: {:6.1}, M2: {:6.1}, M3: {:6.1}, M4: {:6.1}", 
                    test_bits[i], model1, model2, model3, model4);
        }
    }
    
    println!("\n=== Key Insights ===");
    println!("1. Distance doesn't grow linearly with bit size");
    println!("2. There's a 'jump' between 129-bit and 161-bit where methods fail");
    println!("3. The scale_transition_gamma constant (12.416) seems significant");
    println!("4. phase_coupling_beta (0.199) might represent a precision limit");
    
    // Analyze the threshold where methods fail
    println!("\n=== Failure Threshold Analysis ===");
    let failure_bits = 161.0;
    let success_bits = 129.0;
    
    println!("Last success: {} bits", success_bits);
    println!("First failure: {} bits", failure_bits);
    println!("Ratio: {:.3}", failure_bits / success_bits);
    println!("Difference: {} bits", failure_bits - success_bits);
    
    // Check if this relates to our constants
    println!("\nPossible relationships to constants:");
    println!("  failure_bits / scale_transition_gamma = {:.3}", failure_bits / scale_transition_gamma);
    println!("  (failure - success) / scale_transition_gamma = {:.3}", 
            (failure_bits - success_bits) / scale_transition_gamma);
    println!("  resonance_decay_alpha ^ (bits/100) at 161 bits = {:.6}", 
            resonance_decay_alpha.powf(failure_bits / 100.0));
    
    // u64 limit analysis
    println!("\n=== Numeric Limits ===");
    println!("u64::MAX = {}", u64::MAX);
    println!("sqrt(u64::MAX) ≈ 2^32 = {}", (u64::MAX as f64).sqrt());
    println!("Bits in sqrt(2^129) = 64.5");
    println!("Bits in sqrt(2^161) = 80.5 (exceeds u64!)");
    println!("\nThis suggests the failure at 161-bit is due to sqrt(n) exceeding u64::MAX");
}