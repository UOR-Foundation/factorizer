//! Test channel coupling implementation
//!
//! Run with: cargo run --example test_channel_coupling

use eight_bit_pattern::{
    recognize_factors, recognize_factors_with_diagnostics, 
    TunerParams, decompose,
    CouplingMatrix, apply_channel_coupling, compute_resonance_with_position,
    detect_coupled_patterns
};
use num_bigint::BigInt;
use std::time::Instant;

fn main() {
    println!("=== Testing Channel Coupling (2×2 Matrix) ===\n");
    
    let params = TunerParams::default();
    
    // Test cases where channel coupling should help
    let test_cases = vec![
        ("16-bit twin primes", BigInt::from(899u32), "29 × 31"),      // Adjacent primes
        ("16-bit balanced", BigInt::from(58081u32), "241 × 241"),     // Perfect square
        ("20-bit", BigInt::from(1048573u32), "1021 × 1027"),          // Close factors
        ("24-bit", BigInt::from(16769023u32), "4093 × 4099"),         // Larger close factors
        ("32-bit", BigInt::from(3215031751u64), "56599 × 56809"),     // Large factors
    ];
    
    println!("=== Factorization with Channel Coupling ===\n");
    
    for (name, n, expected) in &test_cases {
        let channels = decompose(&n);
        println!("{}: {} = {}", name, n, expected);
        println!("  Channels: {:?}", channels);
        
        let start = Instant::now();
        let result = recognize_factors(&n, &params);
        let elapsed = start.elapsed();
        
        if let Some(factors) = result {
            if factors.verify(&n) {
                println!("  ✓ Found: {} × {} in {:?}", factors.p, factors.q, elapsed);
            }
        } else {
            println!("  ✗ Failed in {:?}", elapsed);
        }
        
        println!();
    }
    
    // Demonstrate coupling matrix effects
    println!("=== Coupling Matrix Analysis ===\n");
    
    let n = BigInt::from(899u32); // 29 × 31
    let channels = decompose(&n);
    
    println!("Number: {} = 29 × 31", n);
    println!("Channels: {:?}\n", channels);
    
    // Show coupling for adjacent channels
    if channels.len() >= 2 {
        for i in 0..channels.len() - 1 {
            println!("Channels[{}] and [{}]:", i, i + 1);
            
            // Compute resonances
            let res1 = compute_resonance_with_position(
                channels[i], i, channels.len(), &params
            );
            let res2 = compute_resonance_with_position(
                channels[i + 1], i + 1, channels.len(), &params
            );
            
            // Get coupling matrix
            let coupling = CouplingMatrix::for_position(i, channels.len());
            println!("  Coupling matrix:");
            println!("    [a11={:.2}, a12={:.2}]", coupling.a11, coupling.a12);
            println!("    [a21={:.2}, a22={:.2}]", coupling.a21, coupling.a22);
            
            // Apply coupling
            let (coupled1, coupled2) = apply_channel_coupling(&res1, &res2, &coupling);
            
            println!("  Original resonances:");
            println!("    Ch[{}]: primary={} (len)", i, res1.primary_resonance.to_string().len());
            println!("    Ch[{}]: primary={} (len)", i + 1, res2.primary_resonance.to_string().len());
            
            println!("  Coupled resonances:");
            println!("    Ch[{}]: primary={} (len)", i, coupled1.primary_resonance.to_string().len());
            println!("    Ch[{}]: primary={} (len)", i + 1, coupled2.primary_resonance.to_string().len());
            
            println!();
        }
    }
    
    // Analyze coupled patterns
    println!("=== Coupled Pattern Detection ===\n");
    
    let (_, diagnostics) = recognize_factors_with_diagnostics(&n, &params);
    
    // Manually check for coupled patterns
    let mut channel_data = Vec::new();
    for ch_diag in &diagnostics.channels {
        channel_data.push((
            ch_diag.position,
            ch_diag.value,
            ch_diag.resonance.clone()
        ));
    }
    
    let coupled_pairs = detect_coupled_patterns(&channel_data, &n, &params);
    
    println!("Found {} coupled channel pairs", coupled_pairs.len());
    for pair in &coupled_pairs {
        println!("  Channels[{}]-[{}]: values=({}, {})", 
            pair.channel1_idx, pair.channel2_idx,
            pair.channel1_val, pair.channel2_val
        );
        
        // Check if combined value is meaningful
        let combined = pair.channel1_val as u16 * 256 + pair.channel2_val as u16;
        println!("    Combined value: {}", combined);
        
        if combined > 0 && combined < 1000 {
            // Check if it's a factor
            if n.clone() % combined == BigInt::from(0) {
                println!("    ✓ Combined value IS a factor!");
            }
        }
    }
    
    // Test larger numbers
    println!("\n=== Coupling on Larger Numbers ===\n");
    
    let large_cases = vec![
        ("32-bit", BigInt::from(4294967291u64), "large prime"),
        ("40-bit", BigInt::from(1099511627689u64), "3 × 366503875863"),
        ("48-bit", BigInt::from(281474976710597u64), "large prime"),
    ];
    
    for (name, n, desc) in large_cases {
        let channels = decompose(&n);
        println!("{} ({}): {} channels", name, desc, channels.len());
        
        if channels.len() >= 2 {
            // Show first and last channel coupling
            let coupling_first = CouplingMatrix::for_position(0, channels.len());
            let coupling_last = CouplingMatrix::for_position(
                channels.len() - 2, channels.len()
            );
            
            println!("  First pair coupling: a21={:.2}", coupling_first.a21);
            println!("  Last pair coupling:  a21={:.2}", coupling_last.a21);
        }
        
        let start = Instant::now();
        let result = recognize_factors(&n, &params);
        let elapsed = start.elapsed();
        
        match result {
            Some(factors) if factors.verify(&n) => {
                println!("  ✓ Factored in {:?}", elapsed);
            }
            _ => {
                println!("  ✗ Not factored in {:?}", elapsed);
            }
        }
        
        println!();
    }
}