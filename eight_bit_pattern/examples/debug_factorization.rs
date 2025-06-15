//! Debug a single factorization to understand what's happening
//!
//! Run with: cargo run --example debug_factorization

use eight_bit_pattern::{
    TunerParams, compute_basis, recognize_factors_with_diagnostics,
    decompose, Constants
};
use num_bigint::BigInt;

fn main() {
    // Test with a larger semiprime that will have multiple channels
    let n = BigInt::from(9409u32); // 97 * 97 (14-bit number)
    println!("Debugging factorization of {} = 97 × 97", n);
    println!("Bit length: {} bits\n", n.bits());
    
    // Decompose into channels
    let channels = decompose(&n);
    println!("Channel decomposition:");
    for (i, &byte) in channels.iter().enumerate() {
        println!("  Channel {}: {:3} = {:08b}", i, byte, byte);
    }
    
    // Show constant values
    println!("\nConstant values (Q32.224 encoding):");
    let constants = Constants::all();
    for (i, c) in constants.iter().enumerate() {
        println!("  Bit {} ({}): numerator = {}", i, c.symbol, &c.numerator.to_string()[..20]);
    }
    
    // Create basis with modified parameters for debugging
    let mut params = TunerParams::default();
    params.alignment_threshold = 1; // Lower threshold for debugging
    params.resonance_scaling_shift = 8; // Less aggressive scaling
    
    println!("\nTuner parameters:");
    println!("  alignment_threshold: {}", params.alignment_threshold);
    println!("  resonance_scaling_shift: {}", params.resonance_scaling_shift);
    println!("  harmonic_progression_step: {}", params.harmonic_progression_step);
    println!("  phase_coupling_strength: {}", params.phase_coupling_strength);
    
    let basis = compute_basis(16, &params); // Small basis for debugging
    
    // Run factorization with diagnostics
    println!("\nRunning factorization...");
    let (result, diagnostics) = recognize_factors_with_diagnostics(&n, &basis, &params);
    
    // Show diagnostic summary
    let summary = diagnostics.summary();
    println!("\nDiagnostic summary:");
    println!("  Channels analyzed: {}", summary.channels_analyzed);
    println!("  Peaks detected: {}", summary.peaks_detected);
    println!("  Unique patterns: {}", summary.unique_patterns);
    println!("  Candidates tested: {}", summary.candidates_tested);
    println!("  Success: {}", summary.success);
    
    if let Some((pattern, count)) = summary.most_common_pattern {
        println!("  Most common pattern: {:08b} ({} times)", pattern, count);
    }
    
    // Show channel diagnostics
    println!("\nChannel resonances:");
    for ch in &diagnostics.channels {
        println!("  Channel {}: value={:3} ({:08b})",
            ch.position, ch.value, ch.value
        );
        println!("    Primary resonance: {}...", &ch.resonance.primary_resonance.to_string()[..20]);
        println!("    Harmonic signature: {:#018x}", ch.resonance.harmonic_signature);
        println!("    Phase offset: {}", ch.resonance.phase_offset);
    }
    
    // Show peaks
    if diagnostics.peaks.is_empty() {
        println!("\nNo peaks detected!");
        println!("This suggests:");
        println!("- Alignment detection is too strict");
        println!("- Resonance values are not producing meaningful patterns");
        println!("- Constants need different values");
    } else {
        println!("\nDetected peaks:");
        for peak in &diagnostics.peaks {
            println!("  Channels {}-{}: pattern {:08b}, strength {}",
                peak.start_channel, peak.end_channel,
                peak.aligned_pattern, peak.alignment_strength
            );
        }
    }
    
    // Show result
    match result {
        Some(factors) => {
            println!("\nFactorization successful!");
            println!("  {} = {} × {}", n, factors.p, factors.q);
        }
        None => {
            println!("\nFactorization failed.");
            
            // Additional debugging for small numbers
            if n.bits() <= 16 {
                println!("\nTrying brute force to verify correctness...");
                for i in 2..n.sqrt().to_u32_digits().1[0] {
                    let candidate = BigInt::from(i);
                    if &n % &candidate == BigInt::from(0) {
                        let other = &n / &candidate;
                        println!("  Actual factors: {} × {}", candidate, other);
                        break;
                    }
                }
            }
        }
    }
}