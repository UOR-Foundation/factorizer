//! Resonance visualization tool for The 8-Bit Pattern
//!
//! Run with: cargo run --example visualize_resonance

use eight_bit_pattern::{
    TunerParams, compute_basis, decompose, Constants,
    recognize_factors_with_diagnostics
};
use num_bigint::BigInt;
use std::collections::HashMap;

fn main() {
    println!("=== Resonance Visualization for The 8-Bit Pattern ===\n");
    
    // Test numbers with known factors
    let test_numbers = vec![
        (BigInt::from(15u32), "3 × 5"),
        (BigInt::from(35u32), "5 × 7"),
        (BigInt::from(143u32), "11 × 13"),
        (BigInt::from(323u32), "17 × 19"),
        (BigInt::from(9409u32), "97 × 97"),
        (BigInt::from(10403u32), "101 × 103"),
    ];
    
    let params = TunerParams::default();
    let basis = compute_basis(32, &params);
    
    for (n, factors) in &test_numbers {
        println!("\n{}", "=".repeat(60));
        println!("N = {} = {}", n, factors);
        println!("Bit length: {} bits", n.bits());
        println!("{}\n", "=".repeat(60));
        
        // Decompose into channels
        let channels = decompose(n);
        
        // Show channel decomposition
        println!("Channel Decomposition:");
        for (i, &ch) in channels.iter().enumerate() {
            println!("  Channel {}: {:3} = {:08b}", i, ch, ch);
            
            // Show which constants are active
            let active = Constants::active_constants(ch);
            if !active.is_empty() {
                let names: Vec<_> = active.iter().map(|c| c.symbol).collect();
                println!("    Active constants: {}", 
                    names.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(", "));
            }
        }
        
        // Run factorization with diagnostics
        let (result, diagnostics) = recognize_factors_with_diagnostics(n, &basis, &params);
        
        // Visualize resonance patterns
        println!("\nResonance Patterns:");
        
        // Create ASCII visualization of resonance magnitudes
        if !diagnostics.channels.is_empty() {
            let max_resonance = diagnostics.channels.iter()
                .map(|ch| ch.resonance.primary_resonance.bits())
                .max()
                .unwrap_or(1);
            
            for ch in &diagnostics.channels {
                let magnitude = ch.resonance.primary_resonance.bits();
                let bar_length = (magnitude * 50 / max_resonance).min(50);
                let bar = "█".repeat(bar_length as usize);
                
                println!("  Ch{}: {} ({} bits)", 
                    ch.position, bar, magnitude);
            }
        }
        
        // Show detected peaks
        if !diagnostics.peaks.is_empty() {
            println!("\nDetected Peaks:");
            for peak in &diagnostics.peaks {
                println!("  Channels {}-{}: Pattern {:08b} (strength {})",
                    peak.start_channel, peak.end_channel,
                    peak.aligned_pattern, peak.alignment_strength);
                
                // Decode pattern
                let active = Constants::active_constants(peak.aligned_pattern);
                if !active.is_empty() {
                    let names: Vec<_> = active.iter().map(|c| c.name).collect();
                    println!("    Constants: {}", names.join(", "));
                }
            }
        } else {
            println!("\nNo peaks detected!");
        }
        
        // Show phase relationships
        if diagnostics.channels.len() >= 2 {
            println!("\nPhase Relationships:");
            for i in 0..diagnostics.channels.len()-1 {
                let ch1 = &diagnostics.channels[i];
                let ch2 = &diagnostics.channels[i+1];
                
                let phase_diff = (&ch2.resonance.phase_offset - &ch1.resonance.phase_offset)
                    .to_string();
                
                // Show first 20 digits of phase difference
                let phase_str = if phase_diff.len() > 20 {
                    format!("{}...", &phase_diff[..20])
                } else {
                    phase_diff
                };
                
                println!("  Ch{} → Ch{}: Δφ = {}", i, i+1, phase_str);
            }
        }
        
        // Show result
        match result {
            Some(factors) => {
                println!("\n✓ Factorization successful: {} × {}", 
                    factors.p, factors.q);
            }
            None => {
                println!("\n✗ Factorization failed");
            }
        }
    }
    
    // Pattern frequency analysis
    println!("\n\n{}", "=".repeat(60));
    println!("Pattern Frequency Analysis");
    println!("{}\n", "=".repeat(60));
    
    let mut pattern_freq: HashMap<u8, usize> = HashMap::new();
    let mut success_patterns: HashMap<u8, usize> = HashMap::new();
    
    for (n, _) in &test_numbers {
        let (result, diagnostics) = recognize_factors_with_diagnostics(n, &basis, &params);
        
        for peak in &diagnostics.peaks {
            *pattern_freq.entry(peak.aligned_pattern).or_default() += 1;
            if result.is_some() {
                *success_patterns.entry(peak.aligned_pattern).or_default() += 1;
            }
        }
    }
    
    // Sort by frequency
    let mut patterns: Vec<_> = pattern_freq.iter().collect();
    patterns.sort_by(|a, b| b.1.cmp(a.1));
    
    println!("Most common patterns:");
    for (pattern, freq) in patterns.iter().take(10) {
        let success = success_patterns.get(pattern).unwrap_or(&0);
        let rate = if **freq > 0 {
            *success as f64 / **freq as f64 * 100.0
        } else {
            0.0
        };
        
        println!("  Pattern {:08b}: {} occurrences, {:.0}% success",
            pattern, freq, rate);
        
        // Show active constants
        let active = Constants::active_constants(**pattern);
        if !active.is_empty() {
            let symbols: Vec<_> = active.iter().map(|c| c.symbol).collect();
            println!("    Active: {}", 
                symbols.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(" "));
        }
    }
    
    // Constant correlation matrix
    println!("\n\nConstant Activation Correlation:");
    println!("(Shows which constants appear together in successful patterns)\n");
    
    let mut cooccurrence = [[0usize; 8]; 8];
    
    for (pattern, _) in success_patterns.iter() {
        for i in 0..8 {
            if (pattern >> i) & 1 == 1 {
                for j in 0..8 {
                    if (pattern >> j) & 1 == 1 {
                        cooccurrence[i][j] += 1;
                    }
                }
            }
        }
    }
    
    let names = ["1", "τ", "φ", "ε", "δ", "γ", "β", "α"];
    print!("    ");
    for name in &names {
        print!("{:>4}", name);
    }
    println!();
    
    for (i, row_name) in names.iter().enumerate() {
        print!("{:>3} ", row_name);
        for j in 0..8 {
            if cooccurrence[i][j] > 0 {
                print!("{:4}", cooccurrence[i][j]);
            } else {
                print!("   .");
            }
        }
        println!();
    }
    
    // Recommendations
    println!("\n\nVisualization Insights:");
    
    if pattern_freq.is_empty() {
        println!("- No patterns detected - constants may need adjustment");
    } else {
        let total_patterns: usize = pattern_freq.values().sum();
        let successful_patterns: usize = success_patterns.values().sum();
        let success_rate = successful_patterns as f64 / total_patterns as f64;
        
        println!("- Overall pattern success rate: {:.1}%", success_rate * 100.0);
        
        if success_rate > 0.5 {
            println!("- Pattern recognition is working well");
        } else {
            println!("- Pattern recognition needs improvement");
        }
        
        // Find most correlated constant pairs
        let mut max_cooc = 0;
        let mut best_pair = (0, 0);
        for i in 0..8 {
            for j in i+1..8 {
                if cooccurrence[i][j] > max_cooc {
                    max_cooc = cooccurrence[i][j];
                    best_pair = (i, j);
                }
            }
        }
        
        if max_cooc > 0 {
            println!("- Strongest correlation: {} and {} (co-occur {} times)",
                names[best_pair.0], names[best_pair.1], max_cooc);
        }
    }
}