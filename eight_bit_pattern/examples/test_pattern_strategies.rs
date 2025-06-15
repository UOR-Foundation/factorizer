//! Test different pattern detection strategies to diagnose the regression
//!
//! Run with: cargo run --example test_pattern_strategies

use eight_bit_pattern::{
    recognize_factors, decompose, compute_basis, compute_resonance_with_position,
    detect_aligned_channels, extract_factors, TunerParams,
    detect_coupled_patterns, propagate_phase_sequence, detect_phase_relations,
    detect_phase_alignments, extract_factors_from_phase,
    analyze_channel_hierarchy, PeakLocation, ResonanceTuple
};
use num_bigint::BigInt;
use num_traits::{Zero, One};
use num_integer::Integer;
use std::time::Instant;

fn main() {
    println!("=== Testing Pattern Detection Strategies ===\n");
    
    let params = TunerParams::default();
    
    // Test cases that were working well before
    let test_cases = vec![
        (BigInt::from(143u32), "11 × 13"),
        (BigInt::from(221u32), "13 × 17"),
        (BigInt::from(323u32), "17 × 19"),
        (BigInt::from(437u32), "19 × 23"),
        (BigInt::from(667u32), "23 × 29"),
        (BigInt::from(899u32), "29 × 31"),
        (BigInt::from(259u32), "7 × 37"), // This was failing
    ];
    
    println!("=== Strategy 1: Original Simple Pattern Detection ===\n");
    test_simple_pattern_detection(&test_cases, &params);
    
    println!("\n=== Strategy 2: With Coupling Only ===\n");
    test_with_coupling_only(&test_cases, &params);
    
    println!("\n=== Strategy 3: With Phase Only ===\n");
    test_with_phase_only(&test_cases, &params);
    
    println!("\n=== Strategy 4: With Hierarchy Only ===\n");
    test_with_hierarchy_only(&test_cases, &params);
    
    println!("\n=== Strategy 5: Full Current Implementation ===\n");
    test_full_implementation(&test_cases, &params);
}

fn test_simple_pattern_detection(test_cases: &[(BigInt, &str)], params: &TunerParams) {
    let mut success_count = 0;
    
    for (n, expected) in test_cases {
        let channels = decompose(n);
        let basis = compute_basis(n, params);
        
        // Collect channel resonances
        let mut channel_resonances: Vec<(usize, u8, ResonanceTuple)> = Vec::new();
        for (pos, &channel_value) in channels.iter().enumerate() {
            if let Some(channel) = basis.get_channel(pos) {
                if let Some(pattern) = channel.get_pattern(channel_value) {
                    channel_resonances.push((pos, channel_value, pattern.resonance.clone()));
                }
            }
        }
        
        // Simple sliding window detection
        let mut peaks = Vec::new();
        for window_size in 1..=channels.len().min(4) {
            for start_pos in 0..=channels.len().saturating_sub(window_size) {
                let window: Vec<_> = channel_resonances[start_pos..start_pos + window_size]
                    .iter()
                    .cloned()
                    .collect();
                
                if let Some(pattern) = find_simple_alignment_pattern(&window, n) {
                    peaks.push(PeakLocation::new(
                        start_pos,
                        start_pos + window_size - 1,
                        pattern
                    ));
                }
            }
        }
        
        // Try to extract factors
        if let Some(factors) = extract_factors(n, &peaks, &channels, params) {
            if factors.verify(n) {
                println!("  ✓ {} = {} (found {} × {})", n, expected, factors.p, factors.q);
                success_count += 1;
            } else {
                println!("  ✗ {} = {} (invalid factors)", n, expected);
            }
        } else {
            println!("  ✗ {} = {} (no factors found)", n, expected);
        }
    }
    
    println!("\nSuccess rate: {}/{} ({:.1}%)", 
        success_count, test_cases.len(), 
        100.0 * success_count as f64 / test_cases.len() as f64);
}

fn find_simple_alignment_pattern(window: &[(usize, u8, ResonanceTuple)], n: &BigInt) -> Option<u8> {
    if window.is_empty() {
        return None;
    }
    
    // Direct factor check for small numbers
    if n.bits() <= 20 {
        if window.len() == 1 {
            let ch_val = BigInt::from(window[0].1);
            if ch_val > BigInt::one() && n % &ch_val == BigInt::zero() {
                return Some(window[0].1);
            }
        }
        
        if window.len() == 2 {
            let combined = BigInt::from(window[0].1) * 256 + BigInt::from(window[1].1);
            if combined > BigInt::one() && &combined <= &n.sqrt() && n % &combined == BigInt::zero() {
                return Some(window[0].1 ^ window[1].1);
            }
        }
    }
    
    // GCD check
    let mut gcd_accumulator = window[0].2.primary_resonance.clone();
    for i in 1..window.len() {
        gcd_accumulator = gcd_accumulator.gcd(&window[i].2.primary_resonance);
    }
    
    if gcd_accumulator > BigInt::one() && &gcd_accumulator < n && n % &gcd_accumulator == BigInt::zero() {
        let pattern = window.iter()
            .map(|(_, val, _)| val)
            .fold(0u8, |acc, &val| acc ^ val);
        return Some(pattern);
    }
    
    // Return pattern anyway
    let pattern = window.iter()
        .map(|(_, val, _)| val)
        .fold(0u8, |acc, &val| acc ^ val);
    Some(pattern)
}

fn test_with_coupling_only(test_cases: &[(BigInt, &str)], params: &TunerParams) {
    let mut success_count = 0;
    
    for (n, expected) in test_cases {
        let channels = decompose(n);
        let basis = compute_basis(n, params);
        
        // Collect channel resonances
        let mut channel_resonances: Vec<(usize, u8, ResonanceTuple)> = Vec::new();
        for (pos, &channel_value) in channels.iter().enumerate() {
            if let Some(channel) = basis.get_channel(pos) {
                if let Some(pattern) = channel.get_pattern(channel_value) {
                    channel_resonances.push((pos, channel_value, pattern.resonance.clone()));
                }
            }
        }
        
        let mut peaks = Vec::new();
        
        // Add simple patterns
        for window_size in 1..=channels.len().min(4) {
            for start_pos in 0..=channels.len().saturating_sub(window_size) {
                let window: Vec<_> = channel_resonances[start_pos..start_pos + window_size]
                    .iter()
                    .cloned()
                    .collect();
                
                if let Some(pattern) = find_simple_alignment_pattern(&window, n) {
                    peaks.push(PeakLocation::new(
                        start_pos,
                        start_pos + window_size - 1,
                        pattern
                    ));
                }
            }
        }
        
        // Add coupled patterns
        if channels.len() >= 2 {
            let coupled_pairs = detect_coupled_patterns(&channel_resonances, n, params);
            for pair in coupled_pairs {
                let pattern = pair.channel1_val ^ pair.channel2_val;
                peaks.push(PeakLocation::new(
                    pair.channel1_idx,
                    pair.channel2_idx,
                    pattern
                ));
            }
        }
        
        // Extract factors
        if let Some(factors) = extract_factors(n, &peaks, &channels, params) {
            if factors.verify(n) {
                println!("  ✓ {} = {} (found {} × {})", n, expected, factors.p, factors.q);
                success_count += 1;
            } else {
                println!("  ✗ {} = {} (invalid factors)", n, expected);
            }
        } else {
            println!("  ✗ {} = {} (no factors found)", n, expected);
        }
    }
    
    println!("\nSuccess rate: {}/{} ({:.1}%)", 
        success_count, test_cases.len(), 
        100.0 * success_count as f64 / test_cases.len() as f64);
}

fn test_with_phase_only(test_cases: &[(BigInt, &str)], params: &TunerParams) {
    let mut success_count = 0;
    
    for (n, expected) in test_cases {
        let channels = decompose(n);
        let basis = compute_basis(n, params);
        
        // Collect channel resonances
        let mut channel_resonances: Vec<(usize, u8, ResonanceTuple)> = Vec::new();
        for (pos, &channel_value) in channels.iter().enumerate() {
            if let Some(channel) = basis.get_channel(pos) {
                if let Some(pattern) = channel.get_pattern(channel_value) {
                    channel_resonances.push((pos, channel_value, pattern.resonance.clone()));
                }
            }
        }
        
        let mut peaks = Vec::new();
        
        // Add simple patterns
        for window_size in 1..=channels.len().min(4) {
            for start_pos in 0..=channels.len().saturating_sub(window_size) {
                let window: Vec<_> = channel_resonances[start_pos..start_pos + window_size]
                    .iter()
                    .cloned()
                    .collect();
                
                if let Some(pattern) = find_simple_alignment_pattern(&window, n) {
                    peaks.push(PeakLocation::new(
                        start_pos,
                        start_pos + window_size - 1,
                        pattern
                    ));
                }
            }
        }
        
        // Add phase patterns
        if channels.len() >= 2 {
            let phase_states = propagate_phase_sequence(&channel_resonances, n, params);
            let phase_relations = detect_phase_relations(&phase_states, n, params);
            let phase_alignments = detect_phase_alignments(&phase_states, &phase_relations, n, params);
            
            for alignment in phase_alignments {
                if alignment.alignment_strength > 0.5 {
                    let pattern = (alignment.phase_period.to_u32_digits().1.get(0).copied().unwrap_or(0) % 256) as u8;
                    peaks.push(PeakLocation::new(
                        alignment.start_channel,
                        alignment.end_channel,
                        pattern
                    ));
                }
            }
        }
        
        // Extract factors
        if let Some(factors) = extract_factors(n, &peaks, &channels, params) {
            if factors.verify(n) {
                println!("  ✓ {} = {} (found {} × {})", n, expected, factors.p, factors.q);
                success_count += 1;
            } else {
                println!("  ✗ {} = {} (invalid factors)", n, expected);
            }
        } else {
            println!("  ✗ {} = {} (no factors found)", n, expected);
        }
    }
    
    println!("\nSuccess rate: {}/{} ({:.1}%)", 
        success_count, test_cases.len(), 
        100.0 * success_count as f64 / test_cases.len() as f64);
}

fn test_with_hierarchy_only(test_cases: &[(BigInt, &str)], params: &TunerParams) {
    let mut success_count = 0;
    
    for (n, expected) in test_cases {
        let channels = decompose(n);
        let basis = compute_basis(n, params);
        
        // Collect channel resonances
        let mut channel_resonances: Vec<(usize, u8, ResonanceTuple)> = Vec::new();
        for (pos, &channel_value) in channels.iter().enumerate() {
            if let Some(channel) = basis.get_channel(pos) {
                if let Some(pattern) = channel.get_pattern(channel_value) {
                    channel_resonances.push((pos, channel_value, pattern.resonance.clone()));
                }
            }
        }
        
        let mut peaks = Vec::new();
        
        // Add simple patterns (baseline)
        for window_size in 1..=channels.len().min(4) {
            for start_pos in 0..=channels.len().saturating_sub(window_size) {
                let window: Vec<_> = channel_resonances[start_pos..start_pos + window_size]
                    .iter()
                    .cloned()
                    .collect();
                
                if let Some(pattern) = find_simple_alignment_pattern(&window, n) {
                    peaks.push(PeakLocation::new(
                        start_pos,
                        start_pos + window_size - 1,
                        pattern
                    ));
                }
            }
        }
        
        // Add hierarchical patterns
        if channels.len() >= 2 {
            let hierarchy_analysis = analyze_channel_hierarchy(&channel_resonances, n, params);
            
            for (_level, patterns) in &hierarchy_analysis.patterns_by_level {
                for pattern in patterns {
                    if pattern.strength > 0.6 {
                        let peak_pattern = if let Some(ref factor) = pattern.factor_candidate {
                            (factor.to_u32_digits().1.get(0).copied().unwrap_or(0) % 256) as u8
                        } else {
                            0u8
                        };
                        
                        // Add as peak if we can determine range
                        if !pattern.groups.is_empty() {
                            peaks.push(PeakLocation::new(0, channels.len() - 1, peak_pattern));
                        }
                    }
                }
            }
        }
        
        // Extract factors
        if let Some(factors) = extract_factors(n, &peaks, &channels, params) {
            if factors.verify(n) {
                println!("  ✓ {} = {} (found {} × {})", n, expected, factors.p, factors.q);
                success_count += 1;
            } else {
                println!("  ✗ {} = {} (invalid factors)", n, expected);
            }
        } else {
            println!("  ✗ {} = {} (no factors found)", n, expected);
        }
    }
    
    println!("\nSuccess rate: {}/{} ({:.1}%)", 
        success_count, test_cases.len(), 
        100.0 * success_count as f64 / test_cases.len() as f64);
}

fn test_full_implementation(test_cases: &[(BigInt, &str)], params: &TunerParams) {
    let mut success_count = 0;
    
    for (n, expected) in test_cases {
        let start = Instant::now();
        
        if let Some(factors) = recognize_factors(n, params) {
            if factors.verify(n) {
                println!("  ✓ {} = {} (found {} × {}) in {:?}", 
                    n, expected, factors.p, factors.q, start.elapsed());
                success_count += 1;
            } else {
                println!("  ✗ {} = {} (invalid factors) in {:?}", 
                    n, expected, start.elapsed());
            }
        } else {
            println!("  ✗ {} = {} (no factors found) in {:?}", 
                n, expected, start.elapsed());
        }
    }
    
    println!("\nSuccess rate: {}/{} ({:.1}%)", 
        success_count, test_cases.len(), 
        100.0 * success_count as f64 / test_cases.len() as f64);
}