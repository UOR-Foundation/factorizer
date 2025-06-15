//! Test position-aware channel resonance
//!
//! Run with: cargo run --example test_position_aware

use eight_bit_pattern::{
    recognize_factors, recognize_factors_with_diagnostics,
    TunerParams, decompose, bit_size, compute_resonance_with_position
};
use num_bigint::BigInt;
use std::time::Instant;

fn main() {
    println!("=== Testing Position-Aware Channel Resonance ===\n");
    
    let params = TunerParams::default();
    
    // Test different sized numbers to see position effects
    let test_cases = vec![
        ("8-bit", BigInt::from(143u32), "11 × 13"),
        ("16-bit", BigInt::from(58081u32), "241 × 241"),
        ("20-bit", BigInt::from(1048573u32), "1021 × 1027"),
        ("24-bit", BigInt::from(16769023u32), "4093 × 4099"),
        ("32-bit", BigInt::from(3215031751u64), "56599 × 56809"),
    ];
    
    println!("=== Factorization Performance ===\n");
    
    for (name, n, factors_str) in &test_cases {
        let channels = decompose(&n);
        let bits = bit_size(&n);
        
        println!("{} number: {} = {}", name, n, factors_str);
        println!("  Channels: {} (bits: {})", channels.len(), bits);
        println!("  Channel values: {:?}", channels);
        
        let start = Instant::now();
        let result = recognize_factors(&n, &params);
        let elapsed = start.elapsed();
        
        if let Some(factors) = result {
            if factors.verify(&n) {
                println!("  ✓ Found: {} × {} in {:?}", factors.p, factors.q, elapsed);
            } else {
                println!("  ✗ Invalid factors found");
            }
        } else {
            println!("  ✗ No factors found in {:?}", elapsed);
        }
        
        println!();
    }
    
    // Analyze channel-specific patterns
    println!("=== Channel Pattern Analysis ===\n");
    
    let n = BigInt::from(143u32); // 11 × 13
    let (result, diagnostics) = recognize_factors_with_diagnostics(&n, &params);
    
    println!("Number: {} = 11 × 13", n);
    println!("Channels with resonances:");
    
    for channel in &diagnostics.channels {
        println!("  Channel[{}] = {}", channel.position, channel.value);
        println!("    Primary resonance: {}", channel.resonance.primary_resonance);
        println!("    Harmonic signature: 0x{:016x}", channel.resonance.harmonic_signature);
        println!("    Phase offset: {}", channel.resonance.phase_offset);
        
        if !channel.aligned_with.is_empty() {
            println!("    Aligned with channels: {:?}", channel.aligned_with);
        }
    }
    
    if result.is_some() {
        println!("\n✓ Factorization successful!");
    }
    
    // Test position effects on larger numbers
    println!("\n=== Position Effects on Large Numbers ===\n");
    
    let large_n = BigInt::from(9999991u64) * BigInt::from(9999973u64); // Two large primes
    let channels = decompose(&large_n);
    
    println!("Large semiprime: {} (64-bit)", large_n);
    println!("Channel count: {}", channels.len());
    
    // Show first few and last few channels
    println!("\nFirst 4 channels:");
    for (i, &ch) in channels.iter().take(4).enumerate() {
        println!("  Channel[{}] = {} (0x{:02x})", i, ch, ch);
    }
    
    println!("\nLast 4 channels:");
    let start_idx = channels.len().saturating_sub(4);
    for (i, &ch) in channels[start_idx..].iter().enumerate() {
        println!("  Channel[{}] = {} (0x{:02x})", start_idx + i, ch, ch);
    }
    
    let start = Instant::now();
    let result = recognize_factors(&large_n, &params);
    let elapsed = start.elapsed();
    
    if let Some(factors) = result {
        if factors.verify(&large_n) {
            println!("\n✓ Large number factored in {:?}", elapsed);
            println!("  {} = {} × {}", large_n, factors.p, factors.q);
        }
    } else {
        println!("\n✗ Large number factorization failed in {:?}", elapsed);
    }
    
    // Compare resonance at different positions
    println!("\n=== Resonance Comparison by Position ===\n");
    
    let test_pattern = 143u8;
    let total_channels = 8;
    
    println!("Pattern: {} (0b{:08b})", test_pattern, test_pattern);
    println!("\nResonance characteristics by channel position:");
    
    for pos in 0..4 {
        let res = compute_resonance_with_position(test_pattern, pos, total_channels, &params);
        println!("\nChannel[{}]:", pos);
        println!("  Primary resonance scale: {}", 
            res.primary_resonance.to_string().len());
        println!("  Phase offset: {}", res.phase_offset);
        
        // Show relative magnitude
        if pos == 0 {
            println!("  Relative magnitude: 1.0 (baseline)");
        } else {
            let res0 = compute_resonance_with_position(test_pattern, 0, total_channels, &params);
            if res0.primary_resonance > BigInt::from(0) {
                // Just compare order of magnitude
                let ratio = res.primary_resonance.to_string().len() as f64 / 
                           res0.primary_resonance.to_string().len() as f64;
                println!("  Relative magnitude: ~{:.2}", ratio);
            }
        }
    }
}