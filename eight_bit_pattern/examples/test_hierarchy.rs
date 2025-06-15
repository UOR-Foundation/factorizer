//! Test hierarchical channel grouping
//!
//! Run with: cargo run --example test_hierarchy

use eight_bit_pattern::{
    recognize_factors, decompose, compute_resonance_with_position,
    analyze_channel_hierarchy, TunerParams,
    GroupingLevel
};
use num_bigint::BigInt;
use std::time::Instant;

fn main() {
    println!("=== Testing Hierarchical Channel Grouping ===\n");
    
    let params = TunerParams::default();
    
    // Test cases with different channel counts
    let test_cases = vec![
        // 2-channel numbers (16-bit range)
        (BigInt::from(58081u32), "241 × 241"),           // Perfect square
        (BigInt::from(63001u32), "241 × 261"),           // Close factors
        
        // 3-channel numbers (24-bit range)
        (BigInt::from(16769023u32), "4093 × 4099"),      // Twin primes
        (BigInt::from(16777619u32), "prime"),            // Large prime
        
        // 4-channel numbers (32-bit range)
        (BigInt::from(3215031751u64), "56599 × 56809"),  // Large factors
        (BigInt::from(4294967291u64), "prime"),          // Near 2^32
        
        // 5+ channel numbers
        (BigInt::from(1099511627689u64), "3 × 366503875863"), // 40-bit
        (BigInt::from(281474976710597u64), "large prime"),    // 48-bit
    ];
    
    println!("=== Channel Hierarchical Analysis ===\n");
    
    for (n, expected) in &test_cases {
        let channels = decompose(&n);
        println!("Number: {} = {}", n, expected);
        println!("Channels: {} ({})", channels.len(), 
            channels.iter().map(|c| format!("{:02x}", c)).collect::<Vec<_>>().join(" "));
        
        // Build channel resonances
        let mut channel_resonances = Vec::new();
        for (pos, &ch_val) in channels.iter().enumerate() {
            let res = compute_resonance_with_position(ch_val, pos, channels.len(), &params);
            channel_resonances.push((pos, ch_val, res));
        }
        
        // Perform hierarchical analysis
        let hierarchy = analyze_channel_hierarchy(&channel_resonances, &n, &params);
        
        // Show grouping levels
        println!("  Grouping levels available:");
        for level in GroupingLevel::levels_for_channels(channels.len()) {
            let group_count = hierarchy.groups_by_level.get(&level)
                .map(|g| g.len())
                .unwrap_or(0);
            if group_count > 0 {
                println!("    {:?}: {} groups", level, group_count);
            }
        }
        
        // Show detected patterns
        let mut pattern_count = 0;
        for (_level, patterns) in &hierarchy.patterns_by_level {
            pattern_count += patterns.len();
        }
        println!("  Total patterns detected: {}", pattern_count);
        
        // Show strongest patterns
        let mut all_patterns: Vec<_> = hierarchy.patterns_by_level
            .values()
            .flatten()
            .collect();
        all_patterns.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());
        
        if !all_patterns.is_empty() {
            println!("  Strongest patterns:");
            for (i, pattern) in all_patterns.iter().take(3).enumerate() {
                println!("    {}. Level {:?}, strength {:.3}", 
                    i + 1, pattern.level, pattern.strength);
                if let Some(ref factor) = pattern.factor_candidate {
                    println!("       Factor candidate: {}", factor);
                    if n % factor == BigInt::from(0) {
                        println!("       ✓ Valid factor!");
                    }
                }
            }
        }
        
        // Test factorization
        let start = Instant::now();
        let result = recognize_factors(&n, &params);
        let elapsed = start.elapsed();
        
        match result {
            Some(factors) if factors.verify(&n) => {
                println!("  ✓ Factored: {} × {} in {:?}", factors.p, factors.q, elapsed);
            }
            _ => {
                println!("  ✗ Not factored in {:?}", elapsed);
            }
        }
        
        println!();
    }
    
    // Detailed analysis of a specific case
    println!("=== Detailed Hierarchical Analysis ===\n");
    
    let n = BigInt::from(16769023u32); // 4093 × 4099
    let channels = decompose(&n);
    
    println!("Number: {} = 4093 × 4099", n);
    println!("Channels: {:?}\n", channels);
    
    // Build resonances
    let mut channel_resonances = Vec::new();
    for (pos, &ch_val) in channels.iter().enumerate() {
        let res = compute_resonance_with_position(ch_val, pos, channels.len(), &params);
        channel_resonances.push((pos, ch_val, res));
    }
    
    let hierarchy = analyze_channel_hierarchy(&channel_resonances, &n, &params);
    
    // Show channel groups at each level
    for level in GroupingLevel::levels_for_channels(channels.len()) {
        if let Some(groups) = hierarchy.groups_by_level.get(&level) {
            println!("Level {:?} ({} groups):", level, groups.len());
            
            for (i, group) in groups.iter().enumerate() {
                println!("  Group {}: channels[{}..={}] = {:?}",
                    i, group.start_idx, group.end_idx, group.channel_values);
                
                if group.has_factor_pattern(&n) {
                    println!("    ✓ Has factor pattern!");
                    
                    // Check if combined value is a factor
                    let combined = group.channel_values.iter()
                        .fold(BigInt::from(0), |acc, &val| acc * 256 + BigInt::from(val));
                    if combined > BigInt::from(1) && &combined <= &n.sqrt() {
                        if &n % &combined == BigInt::from(0) {
                            println!("    Combined value {} is a factor!", combined);
                        }
                    }
                }
            }
            println!();
        }
    }
    
    // Test large number with many channels
    println!("=== Large Number Hierarchical Test ===\n");
    
    let large_n = BigInt::from(72057594037927935u64); // 5 × 14411518807585587
    let large_channels = decompose(&large_n);
    
    println!("Number: {} (56-bit)", large_n);
    println!("Channels: {} total", large_channels.len());
    
    // Quick hierarchical analysis
    let mut large_resonances = Vec::new();
    for (pos, &ch_val) in large_channels.iter().enumerate() {
        let res = compute_resonance_with_position(ch_val, pos, large_channels.len(), &params);
        large_resonances.push((pos, ch_val, res));
    }
    
    let large_hierarchy = analyze_channel_hierarchy(&large_resonances, &large_n, &params);
    
    // Summary of hierarchy
    println!("\nHierarchical summary:");
    for level in GroupingLevel::levels_for_channels(large_channels.len()) {
        if let Some(groups) = large_hierarchy.groups_by_level.get(&level) {
            let patterns = large_hierarchy.patterns_by_level.get(&level)
                .map(|p| p.len())
                .unwrap_or(0);
            println!("  {:?}: {} groups, {} patterns", level, groups.len(), patterns);
        }
    }
    
    let start = Instant::now();
    let result = recognize_factors(&large_n, &params);
    let elapsed = start.elapsed();
    
    match result {
        Some(factors) if factors.verify(&large_n) => {
            println!("\n✓ Large number factored in {:?}", elapsed);
            println!("  {} = {} × {}", large_n, factors.p, factors.q);
        }
        _ => {
            println!("\n✗ Large number not factored in {:?}", elapsed);
        }
    }
}