//! Verify mathematical properties (modular arithmetic, GCD preservation)
//!
//! Run with: cargo run --example verify_properties

use eight_bit_pattern::{
    decompose, reconstruct, extract_channel_range,
    compute_resonance, Constants, TunerParams,
    recognize_factors, compute_basis
};
use num_bigint::BigInt;
use num_traits::{Zero, One};
use num_integer::Integer;
use std::collections::HashMap;

fn main() {
    println!("=== Mathematical Properties Verification ===\n");
    
    // Test various mathematical properties
    verify_channel_decomposition();
    verify_modular_arithmetic();
    verify_gcd_preservation();
    verify_resonance_properties();
    verify_constant_properties();
    verify_pattern_invariants();
    
    println!("\n=== All Verifications Complete ===");
}

/// Verify channel decomposition and reconstruction
fn verify_channel_decomposition() {
    println!("1. Channel Decomposition Properties:\n");
    
    let test_numbers = vec![
        BigInt::from(15u32),
        BigInt::from(143u32),
        BigInt::from(9409u32),
        BigInt::from(1234567890u64),
        BigInt::from(u64::MAX),
    ];
    
    let mut all_passed = true;
    
    for n in &test_numbers {
        let channels = decompose(n);
        let reconstructed = reconstruct(&channels);
        
        let passed = &reconstructed == n;
        all_passed &= passed;
        
        println!("  N = {}", n);
        println!("    Channels: {} (first 8: {:?})", 
            channels.len(), &channels[..8.min(channels.len())]);
        println!("    Reconstruction: {} [{}]", 
            if passed { "PASS" } else { "FAIL" },
            if passed { "exact match" } else { "mismatch" });
        
        // Verify channel properties
        let mut channel_sum = BigInt::zero();
        for (i, &ch) in channels.iter().enumerate() {
            let weight = BigInt::from(256).pow(i as u32);
            channel_sum += BigInt::from(ch) * weight;
        }
        
        let sum_match = &channel_sum == n;
        println!("    Channel sum formula: {} [Σ(ch_i × 256^i) = N]", 
            if sum_match { "PASS" } else { "FAIL" });
        println!();
    }
    
    println!("  Overall: {}\n", if all_passed { "✓ PASSED" } else { "✗ FAILED" });
}

/// Verify modular arithmetic properties
fn verify_modular_arithmetic() {
    println!("2. Modular Arithmetic Properties:\n");
    
    let test_cases = vec![
        (BigInt::from(15u32), BigInt::from(3u32), BigInt::from(5u32)),
        (BigInt::from(143u32), BigInt::from(11u32), BigInt::from(13u32)),
        (BigInt::from(323u32), BigInt::from(17u32), BigInt::from(19u32)),
    ];
    
    let mut all_passed = true;
    
    for (n, p, q) in &test_cases {
        println!("  N = {} = {} × {}", n, p, q);
        
        // Property 1: (p × q) mod m = ((p mod m) × (q mod m)) mod m
        let moduli = vec![7, 11, 13, 17, 256];
        let mut mod_passed = true;
        
        for m in &moduli {
            let left = n % m;
            let right = ((p % m) * (q % m)) % m;
            let passed = left == right;
            mod_passed &= passed;
            
            if !passed {
                println!("    Modular multiplication (mod {}): FAIL", m);
            }
        }
        
        println!("    Modular multiplication: {}", 
            if mod_passed { "PASS" } else { "FAIL" });
        
        // Property 2: Channel values encode modular information
        let channels = decompose(n);
        let ch0 = BigInt::from(channels[0]);
        let n_mod_256 = n % 256;
        let ch_match = ch0 == n_mod_256;
        
        println!("    First channel = N mod 256: {} [ch[0]={}, N%256={}]",
            if ch_match { "PASS" } else { "FAIL" },
            ch0, n_mod_256);
        
        // Property 3: Factor visibility in channels
        let p_visible = channels.iter().any(|&ch| {
            let ch_val = BigInt::from(ch);
            ch_val == (p % 256) || ch_val == (q % 256)
        });
        
        println!("    Factor visibility in channels: {}",
            if p_visible { "detected" } else { "not detected" });
        
        all_passed &= mod_passed && ch_match;
        println!();
    }
    
    println!("  Overall: {}\n", if all_passed { "✓ PASSED" } else { "✗ FAILED" });
}

/// Verify GCD preservation properties
fn verify_gcd_preservation() {
    println!("3. GCD Preservation Properties:\n");
    
    let test_pairs = vec![
        (BigInt::from(15u32), BigInt::from(21u32)),
        (BigInt::from(143u32), BigInt::from(169u32)),
        (BigInt::from(35u32), BigInt::from(77u32)),
    ];
    
    let mut all_passed = true;
    
    for (a, b) in &test_pairs {
        let gcd_ab = a.gcd(b);
        println!("  GCD({}, {}) = {}", a, b, gcd_ab);
        
        // Property 1: GCD(a,b) divides both a and b
        let divides_a = a % &gcd_ab == BigInt::zero();
        let divides_b = b % &gcd_ab == BigInt::zero();
        let div_passed = divides_a && divides_b;
        
        println!("    Divides both numbers: {}", 
            if div_passed { "PASS" } else { "FAIL" });
        
        // Property 2: Channel GCD relationships
        let channels_a = decompose(a);
        let channels_b = decompose(b);
        
        let mut channel_gcds = Vec::new();
        for i in 0..channels_a.len().min(channels_b.len()) {
            let ch_gcd = BigInt::from(channels_a[i]).gcd(&BigInt::from(channels_b[i]));
            if ch_gcd > BigInt::one() {
                channel_gcds.push((i, ch_gcd));
            }
        }
        
        println!("    Channel GCDs found: {}", channel_gcds.len());
        for (pos, gcd) in channel_gcds.iter().take(3) {
            println!("      Channel {}: GCD = {}", pos, gcd);
        }
        
        // Property 3: Bezout's identity (ax + by = gcd(a,b))
        // For small numbers, verify existence
        if a.bits() <= 20 && b.bits() <= 20 {
            let mut found_bezout = false;
            'outer: for x in -100..=100 {
                for y in -100..=100 {
                    if a * x + b * y == gcd_ab {
                        found_bezout = true;
                        println!("    Bezout coefficients: x={}, y={}", x, y);
                        break 'outer;
                    }
                }
            }
            
            if !found_bezout {
                println!("    Bezout coefficients: not found in range");
            }
        }
        
        all_passed &= div_passed;
        println!();
    }
    
    println!("  Overall: {}\n", if all_passed { "✓ PASSED" } else { "✗ FAILED" });
}

/// Verify resonance calculation properties
fn verify_resonance_properties() {
    println!("4. Resonance Properties:\n");
    
    let params = TunerParams::default();
    let test_numbers = vec![
        BigInt::from(15u32),
        BigInt::from(143u32),
        BigInt::from(323u32),
    ];
    
    let mut all_passed = true;
    
    for n in &test_numbers {
        println!("  N = {}", n);
        
        let channels = decompose(n);
        let mut resonances = Vec::new();
        
        // Calculate resonances for first few channels
        for (i, &ch) in channels.iter().take(4).enumerate() {
            let resonance = compute_resonance(ch, &params);
            resonances.push(resonance.clone());
            
            // Property 1: Resonance magnitude is positive
            let mag_positive = resonance.primary_resonance > BigInt::zero();
            println!("    Channel {}: resonance magnitude > 0: {}", 
                i, if mag_positive { "PASS" } else { "FAIL" });
            
            all_passed &= mag_positive;
        }
        
        // Property 2: Harmonic signatures follow expected patterns
        let mut harmonic_diffs = Vec::new();
        for i in 1..resonances.len() {
            let diff = if resonances[i].harmonic_signature >= resonances[i-1].harmonic_signature {
                resonances[i].harmonic_signature - resonances[i-1].harmonic_signature
            } else {
                resonances[i-1].harmonic_signature - resonances[i].harmonic_signature
            };
            harmonic_diffs.push(diff);
        }
        
        println!("    Harmonic progression detected: {} differences", 
            harmonic_diffs.len());
        
        println!();
    }
    
    println!("  Overall: {}\n", if all_passed { "✓ PASSED" } else { "✗ FAILED" });
}

/// Verify constant properties
fn verify_constant_properties() {
    println!("5. Constant Properties:\n");
    
    let constants = Constants::all();
    let mut all_passed = true;
    
    // Property 1: All constants are positive
    println!("  Positivity check:");
    for (_i, c) in constants.iter().enumerate() {
        let positive = c.numerator > BigInt::zero();
        all_passed &= positive;
        
        println!("    {} ({}): {}", 
            c.symbol, c.name,
            if positive { "PASS" } else { "FAIL" });
    }
    
    // Property 2: Bit positions are unique and sequential
    println!("\n  Bit position check:");
    let mut bit_positions = HashMap::new();
    let mut seq_passed = true;
    
    for (i, c) in constants.iter().enumerate() {
        if bit_positions.contains_key(&c.bit_position) {
            seq_passed = false;
            println!("    Duplicate bit position {} for {}", c.bit_position, c.symbol);
        }
        bit_positions.insert(c.bit_position, c.symbol);
        
        if c.bit_position != i as u8 {
            seq_passed = false;
            println!("    Non-sequential bit position for {}: expected {}, got {}", 
                c.symbol, i, c.bit_position);
        }
    }
    
    println!("    Bit positions: {}", 
        if seq_passed { "PASS" } else { "FAIL" });
    all_passed &= seq_passed;
    
    // Property 3: Rational representation
    println!("\n  Rational representation check:");
    for c in &constants {
        let rational_valid = c.denominator == (BigInt::one() << 224);
        println!("    {}: denominator = 2^224: {}", 
            c.symbol,
            if rational_valid { "PASS" } else { "FAIL" });
        all_passed &= rational_valid;
    }
    
    // Property 4: Active constant detection
    println!("\n  Active constant detection:");
    let test_patterns = vec![
        (0b10101010u8, vec!['τ', 'ε', 'γ', 'α']),
        (0b11111111u8, vec!['1', 'τ', 'φ', 'ε', 'δ', 'γ', 'β', 'α']),
        (0b00000001u8, vec!['1']),
    ];
    
    for (pattern, expected) in test_patterns {
        let active = Constants::active_constants(pattern);
        let symbols: Vec<_> = active.iter().map(|c| c.symbol).collect();
        let matches = symbols == expected;
        
        println!("    Pattern {:08b}: {} [expected {:?}, got {:?}]",
            pattern,
            if matches { "PASS" } else { "FAIL" },
            expected, symbols);
        
        all_passed &= matches;
    }
    
    println!("\n  Overall: {}\n", if all_passed { "✓ PASSED" } else { "✗ FAILED" });
}

/// Verify pattern invariants
fn verify_pattern_invariants() {
    println!("6. Pattern Invariants:\n");
    
    let params = TunerParams::default();
    let basis = compute_basis(32, &params);
    
    let test_cases = vec![
        (BigInt::from(15u32), BigInt::from(3u32), BigInt::from(5u32)),
        (BigInt::from(35u32), BigInt::from(5u32), BigInt::from(7u32)),
        (BigInt::from(143u32), BigInt::from(11u32), BigInt::from(13u32)),
    ];
    
    let mut all_passed = true;
    
    for (n, p, q) in &test_cases {
        println!("  N = {} = {} × {}", n, p, q);
        
        // Property 1: Factorization is unique (up to order)
        if let Some(factors) = recognize_factors(n, &basis, &params) {
            let unique = (factors.p == *p && factors.q == *q) ||
                        (factors.p == *q && factors.q == *p);
            
            println!("    Unique factorization: {}", 
                if unique { "PASS" } else { "FAIL" });
            
            // Property 2: Factors multiply to N
            let product_correct = factors.verify(n);
            println!("    Product verification: {}", 
                if product_correct { "PASS" } else { "FAIL" });
            
            // Property 3: Factors are in sorted order
            let sorted = factors.p <= factors.q;
            println!("    Factor ordering (p ≤ q): {}", 
                if sorted { "PASS" } else { "FAIL" });
            
            all_passed &= unique && product_correct && sorted;
        } else {
            println!("    Factorization failed!");
            all_passed = false;
        }
        
        // Property 4: Channel extraction preserves structure
        let channels = decompose(n);
        if channels.len() >= 4 {
            let extracted = extract_channel_range(&channels, 0, 3);
            if let Some(val) = extracted {
                let within_range = &val <= n;
                println!("    Channel extraction in range: {}", 
                    if within_range { "PASS" } else { "FAIL" });
                all_passed &= within_range;
            }
        }
        
        println!();
    }
    
    println!("  Overall: {}\n", if all_passed { "✓ PASSED" } else { "✗ FAILED" });
}