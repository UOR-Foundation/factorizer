//! Analyze bit-range patterns to discover the missing constant

use rust_pattern_solver::pattern::basis_persistence::SerializableBasis;
use std::fs;
use std::collections::HashMap;

fn main() {
    println!("=== Analyzing Bit-Range Patterns ===\n");
    
    // Load the universal basis
    let json = fs::read_to_string("data/basis/universal_basis.json").expect("Failed to read basis");
    let basis: SerializableBasis = serde_json::from_str(&json).expect("Failed to parse basis");
    
    // Analyze resonance templates
    println!("Resonance Template Analysis:");
    println!("Bit Size | Template Length | Peak Value | RMS | Zero Crossings");
    println!("---------|-----------------|------------|-----|---------------");
    
    let mut bit_sizes: Vec<_> = basis.resonance_templates.keys().cloned().collect();
    bit_sizes.sort();
    
    let mut ratios = HashMap::new();
    
    for bit_size in &bit_sizes {
        if let Some(template) = basis.resonance_templates.get(bit_size) {
            // Calculate statistics
            let peak = template.iter().fold(0.0f64, |max, &val| max.max(val.abs()));
            let rms = (template.iter().map(|&x| x * x).sum::<f64>() / template.len() as f64).sqrt();
            let zero_crossings = count_zero_crossings(template);
            
            // Store ratios
            ratios.insert(*bit_size, (peak, rms, zero_crossings as f64));
            
            println!("{:8} | {:15} | {:10.6} | {:4.2} | {:14}", 
                     bit_size, template.len(), peak, rms, zero_crossings);
        }
    }
    
    // Look for scaling relationships
    println!("\nScaling Relationships:");
    println!("Bit Ratio | Peak Ratio | RMS Ratio | Zero Crossing Ratio");
    println!("----------|------------|-----------|-------------------");
    
    for i in 1..bit_sizes.len() {
        let prev = bit_sizes[i-1];
        let curr = bit_sizes[i];
        let bit_ratio = curr as f64 / prev as f64;
        
        if let (Some(prev_stats), Some(curr_stats)) = (ratios.get(&prev), ratios.get(&curr)) {
            let peak_ratio = curr_stats.0 / prev_stats.0;
            let rms_ratio = curr_stats.1 / prev_stats.1;
            let zc_ratio = curr_stats.2 / prev_stats.2;
            
            println!("{:9.2} | {:10.6} | {:9.6} | {:18.6}", 
                     bit_ratio, peak_ratio, rms_ratio, zc_ratio);
        }
    }
    
    // Analyze scaling constants
    println!("\nScaling Constants:");
    println!("resonance_decay_alpha: {}", basis.scaling_constants.resonance_decay_alpha);
    println!("phase_coupling_beta: {}", basis.scaling_constants.phase_coupling_beta);
    println!("scale_transition_gamma: {}", basis.scaling_constants.scale_transition_gamma);
    println!("golden_ratio_phi: {}", basis.scaling_constants.golden_ratio_phi);
    println!("tribonacci_tau: {}", basis.scaling_constants.tribonacci_tau);
    
    // Look for the missing constant
    println!("\nHypothesized Missing Constants:");
    
    // Check if there's a relationship between bit size and template characteristics
    let mut prev_bit_size = 0u32;
    let mut prev_peak = 0.0;
    
    for &bit_size in &bit_sizes {
        if let Some((peak, rms, _)) = ratios.get(&bit_size) {
            if prev_bit_size > 0 {
                let bit_growth = bit_size as f64 / prev_bit_size as f64;
                let peak_decay = prev_peak / peak;
                let hypothesized_constant = peak_decay.powf(1.0 / bit_growth.ln());
                
                println!("Bit range {}-{}: Decay constant = {:.10}", 
                         prev_bit_size, bit_size, hypothesized_constant);
            }
            prev_bit_size = bit_size;
            prev_peak = *peak;
        }
    }
    
    // Analyze harmonic basis
    println!("\nHarmonic Basis Analysis:");
    println!("Harmonic | Frequency | Energy");
    println!("---------|-----------|-------");
    
    for (i, harmonic) in basis.harmonic_basis.iter().enumerate() {
        let energy: f64 = harmonic.iter().map(|&x| x * x).sum();
        let normalized_energy = energy / harmonic.len() as f64;
        println!("{:8} | {:9} | {:.6}", i + 1, i + 1, normalized_energy);
    }
    
    // Look for side-channel patterns
    println!("\nPotential Side-Channel Information:");
    
    // The length of resonance templates seems to follow a pattern
    for &bit_size in &bit_sizes {
        if let Some(template) = basis.resonance_templates.get(&bit_size) {
            let len = template.len();
            let predicted_len = ((2.0_f64.powf(bit_size as f64 / 4.0) as usize).max(64)).min(1024);
            let ratio = len as f64 / predicted_len as f64;
            
            println!("Bit size {}: Template length {} (predicted: {}, ratio: {:.3})",
                     bit_size, len, predicted_len, ratio);
        }
    }
}

fn count_zero_crossings(signal: &[f64]) -> usize {
    let mut count = 0;
    let mut prev_sign = signal[0].signum();
    
    for &val in signal.iter().skip(1) {
        let sign = val.signum();
        if sign != prev_sign && sign != 0.0 {
            count += 1;
        }
        if sign != 0.0 {
            prev_sign = sign;
        }
    }
    
    count
}