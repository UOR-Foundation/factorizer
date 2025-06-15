//! Test phase propagation model
//!
//! Run with: cargo run --example test_phase_propagation

use eight_bit_pattern::{
    recognize_factors,
    TunerParams, decompose, compute_resonance_with_position,
    propagate_phase_sequence, detect_phase_relations,
    detect_phase_alignments, extract_factors_from_phase,
    phase_states_to_wave
};
use num_bigint::BigInt;
use std::time::Instant;

fn main() {
    println!("=== Testing Phase Propagation Model ===\n");
    
    let params = TunerParams::default();
    
    // Test cases where phase propagation should reveal patterns
    let test_cases = vec![
        ("16-bit", BigInt::from(58081u32), "241 × 241"),           // Perfect square
        ("20-bit", BigInt::from(1048573u32), "1021 × 1027"),       // Close factors
        ("24-bit", BigInt::from(16769023u32), "4093 × 4099"),      // Larger close factors
        ("28-bit", BigInt::from(268386301u64), "16381 × 16391"),   // Twin primes
        ("32-bit", BigInt::from(3215031751u64), "56599 × 56809"),  // Large factors
    ];
    
    println!("=== Factorization with Phase Analysis ===\n");
    
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
    
    // Detailed phase analysis
    println!("=== Detailed Phase Analysis ===\n");
    
    let n = BigInt::from(1048573u32); // 1021 × 1027
    let channels = decompose(&n);
    
    println!("Number: {} = 1021 × 1027", n);
    println!("Channels: {:?}\n", channels);
    
    // Build resonance sequence
    let mut channel_resonances = Vec::new();
    for (pos, &ch_val) in channels.iter().enumerate() {
        let res = compute_resonance_with_position(ch_val, pos, channels.len(), &params);
        channel_resonances.push((pos, ch_val, res));
    }
    
    // Propagate phases
    let phase_states = propagate_phase_sequence(&channel_resonances, &n, &params);
    
    println!("Phase propagation through channels:");
    for (i, state) in phase_states.iter().enumerate() {
        println!("  Channel[{}]:", i);
        println!("    Accumulated phase: {} (mod n)", state.accumulated_phase);
        println!("    Phase velocity: {}", state.phase_velocity);
        println!("    Phase acceleration: {}", state.phase_acceleration);
    }
    
    // Detect phase relationships
    let phase_relations = detect_phase_relations(&phase_states, &n, &params);
    
    println!("\nPhase relationships between adjacent channels:");
    for (i, relation) in phase_relations.iter().enumerate() {
        println!("  Channels[{}]-[{}]:", i, i + 1);
        println!("    Phase difference: {}", relation.phase_diff);
        println!("    Coherence: {:.3}", relation.coherence);
        println!("    Phase locked: {}", if relation.is_locked { "YES" } else { "NO" });
    }
    
    // Detect phase alignments
    let phase_alignments = detect_phase_alignments(&phase_states, &phase_relations, &n, &params);
    
    println!("\nPhase alignment patterns:");
    if phase_alignments.is_empty() {
        println!("  No significant phase alignments detected");
    } else {
        for alignment in &phase_alignments {
            println!("  Channels[{}..={}]:", alignment.start_channel, alignment.end_channel);
            println!("    Phase period: {}", alignment.phase_period);
            println!("    Alignment strength: {:.3}", alignment.alignment_strength);
            
            // Check if phase period relates to factors
            if n.clone() % &alignment.phase_period == BigInt::from(0) {
                println!("    ✓ Phase period divides n!");
            }
        }
    }
    
    // Extract factors from phase
    let phase_factors = extract_factors_from_phase(&phase_alignments, &n);
    
    if !phase_factors.is_empty() {
        println!("\nPotential factors from phase analysis:");
        for factor in &phase_factors {
            println!("  Factor candidate: {}", factor);
            if n.clone() % factor == BigInt::from(0) {
                let other = n.clone() / factor;
                println!("    ✓ Valid: {} = {} × {}", n, factor, other);
            }
        }
    }
    
    // Visualize phase wave
    println!("\n=== Phase Wave Visualization ===\n");
    
    let phase_wave = phase_states_to_wave(&phase_states, &n);
    
    println!("Channel phase wave characteristics:");
    for (i, ((amp, phase), freq)) in phase_wave.amplitudes.iter()
        .zip(phase_wave.phases.iter())
        .zip(phase_wave.frequencies.iter())
        .enumerate()
    {
        println!("  Channel[{}]: amplitude={:.3}, phase={:.3}π, frequency={:.3}",
            i, amp, phase / std::f64::consts::PI, freq);
    }
    
    // Test larger numbers
    println!("\n=== Phase Analysis on Larger Numbers ===\n");
    
    let large_cases = vec![
        ("40-bit", BigInt::from(1099511627689u64), "3 × 366503875863"),
        ("48-bit", BigInt::from(281474976710597u64), "large prime"),
        ("56-bit", BigInt::from(72057594037927935u64), "5 × 14411518807585587"),
    ];
    
    for (name, n, desc) in large_cases {
        let channels = decompose(&n);
        println!("{} ({}): {} channels", name, desc, channels.len());
        
        // Quick phase analysis
        let mut channel_res = Vec::new();
        for (pos, &ch) in channels.iter().enumerate() {
            let res = compute_resonance_with_position(ch, pos, channels.len(), &params);
            channel_res.push((pos, ch, res));
        }
        
        let states = propagate_phase_sequence(&channel_res, &n, &params);
        let relations = detect_phase_relations(&states, &n, &params);
        
        // Count phase-locked channels
        let locked_count = relations.iter().filter(|r| r.is_locked).count();
        println!("  Phase-locked pairs: {}/{}", locked_count, relations.len());
        
        // Average coherence
        let avg_coherence: f64 = relations.iter().map(|r| r.coherence).sum::<f64>() 
            / relations.len() as f64;
        println!("  Average coherence: {:.3}", avg_coherence);
        
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