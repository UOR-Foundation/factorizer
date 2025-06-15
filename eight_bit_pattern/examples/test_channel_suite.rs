//! Comprehensive channel test suite (1-128 channels)
//!
//! Run with: cargo run --example test_channel_suite

use eight_bit_pattern::{
    recognize_factors, decompose, TunerParams, channels_for_bits
};
use num_bigint::BigInt;
use std::time::Instant;

fn main() {
    println!("=== Comprehensive Channel Test Suite ===\n");
    
    let params = TunerParams::default();
    
    // Test different channel counts
    test_by_channel_count(&params);
    
    // Test edge cases
    test_edge_cases(&params);
    
    // Test scaling behavior
    test_scaling_behavior(&params);
    
    // Test specific patterns
    test_specific_patterns(&params);
}

fn test_by_channel_count(params: &TunerParams) {
    println!("=== Testing by Channel Count ===\n");
    
    // Test cases organized by number of channels
    let test_cases = vec![
        // 1 channel (8-bit numbers)
        (1, vec![
            (BigInt::from(15u32), "3 × 5"),
            (BigInt::from(21u32), "3 × 7"),
            (BigInt::from(35u32), "5 × 7"),
            (BigInt::from(77u32), "7 × 11"),
            (BigInt::from(143u32), "11 × 13"),
            (BigInt::from(221u32), "13 × 17"),
        ]),
        
        // 2 channels (16-bit numbers)
        (2, vec![
            (BigInt::from(323u32), "17 × 19"),
            (BigInt::from(899u32), "29 × 31"),
            (BigInt::from(58081u32), "241 × 241"),
            (BigInt::from(63001u32), "241 × 261"),
            (BigInt::from(65279u32), "prime - 1"),
        ]),
        
        // 3 channels (24-bit numbers)
        (3, vec![
            (BigInt::from(16769023u32), "4093 × 4099"),
            (BigInt::from(8388607u32), "2^23 - 1 = 47 × 178481"),
            (BigInt::from(1048573u32), "1021 × 1027"),
        ]),
        
        // 4 channels (32-bit numbers)
        (4, vec![
            (BigInt::from(3215031751u64), "56599 × 56809"),
            (BigInt::from(4294967291u64), "prime"),
            (BigInt::from(2147483647u64), "2^31 - 1 (prime)"),
        ]),
        
        // 5+ channels (larger numbers)
        (5, vec![
            (BigInt::from(1099511627689u64), "3 × 366503875863"),
        ]),
        (6, vec![
            (BigInt::from(281474976710597u64), "large prime"),
        ]),
        (7, vec![
            (BigInt::from(72057594037927935u64), "5 × 14411518807585587"),
        ]),
        (8, vec![
            (BigInt::from(18446744073709551557u64), "large prime"),
        ]),
    ];
    
    for (expected_channels, cases) in test_cases {
        println!("Testing {}-channel numbers:", expected_channels);
        
        let mut success_count = 0;
        let mut total_time = 0u128;
        
        for (n, expected) in &cases {
            let channels = decompose(n);
            assert_eq!(channels.len(), expected_channels, 
                "Unexpected channel count for {}", n);
            
            let start = Instant::now();
            let result = recognize_factors(n, params);
            let elapsed = start.elapsed();
            total_time += elapsed.as_micros();
            
            if result.is_some() && result.as_ref().unwrap().verify(n) {
                if expected.contains("prime") {
                    println!("  ✗ {} incorrectly factored (should be {})", n, expected);
                } else {
                    println!("  ✓ {} = {} in {:?}", n, expected, elapsed);
                    success_count += 1;
                }
            } else {
                if expected.contains("prime") {
                    println!("  ✓ {} correctly identified as {}", n, expected);
                    success_count += 1;
                } else {
                    println!("  ✗ {} = {} (failed to factor)", n, expected);
                }
            }
        }
        
        let factorizable = cases.iter().filter(|(_, e)| !e.contains("prime")).count();
        println!("  Success: {}/{} factorizable, avg time: {:.1} μs\n",
            success_count, factorizable, 
            total_time as f64 / cases.len() as f64);
    }
}

fn test_edge_cases(params: &TunerParams) {
    println!("\n=== Testing Edge Cases ===\n");
    
    let edge_cases = vec![
        // Powers of 2 minus 1 (Mersenne numbers)
        (BigInt::from(3u32), "3 (prime)"),
        (BigInt::from(7u32), "7 (prime)"),
        (BigInt::from(15u32), "3 × 5"),
        (BigInt::from(31u32), "31 (prime)"),
        (BigInt::from(63u32), "7 × 9"),
        (BigInt::from(127u32), "127 (prime)"),
        (BigInt::from(255u32), "3 × 5 × 17"),
        (BigInt::from(511u32), "7 × 73"),
        (BigInt::from(1023u32), "3 × 11 × 31"),
        (BigInt::from(2047u32), "23 × 89"),
        
        // Powers of 2 plus 1 (Fermat numbers)
        (BigInt::from(5u32), "5 (prime)"),
        (BigInt::from(17u32), "17 (prime)"),
        (BigInt::from(257u32), "257 (prime)"),
        (BigInt::from(65537u32), "65537 (prime)"),
        
        // Channel boundary cases
        (BigInt::from(255u32), "3 × 5 × 17"),      // Max 1 channel
        (BigInt::from(256u32), "2^8"),             // Min 2 channels
        (BigInt::from(65535u32), "3 × 5 × 17 × 257"), // Max 2 channels
        (BigInt::from(65536u32), "2^16"),          // Min 3 channels
        
        // All 1s patterns
        (BigInt::from(0xFFu32), "3 × 5 × 17"),
        (BigInt::from(0xFFFFu32), "3 × 5 × 17 × 257"),
        (BigInt::from(0xFFFFFFu32), "3 × 5 × 17 × 257 × 65537"),
    ];
    
    for (n, expected) in edge_cases {
        let channels = decompose(&n);
        let channel_str = channels.iter()
            .map(|c| format!("{:02x}", c))
            .collect::<Vec<_>>()
            .join(" ");
        
        print!("{:10} ({} ch: {}) = {:25} ... ", n, channels.len(), channel_str, expected);
        
        let start = Instant::now();
        let result = recognize_factors(&n, params);
        let elapsed = start.elapsed();
        
        if result.is_some() && result.as_ref().unwrap().verify(&n) {
            if expected.contains("prime") || expected.contains("2^") {
                println!("✗ incorrectly factored");
            } else {
                println!("✓ factored in {:?}", elapsed);
            }
        } else {
            if expected.contains("prime") || expected.contains("2^") {
                println!("✓ correctly not factored");
            } else {
                println!("✗ failed to factor");
            }
        }
    }
}

fn test_scaling_behavior(params: &TunerParams) {
    println!("\n\n=== Testing Scaling Behavior ===\n");
    
    // Test how performance scales with bit size
    let bit_sizes = vec![8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64];
    
    println!("Bits | Channels | Example Number | Time");
    println!("-----|----------|----------------|------");
    
    for bits in bit_sizes {
        let channels = channels_for_bits(bits);
        
        // Generate a semiprime of approximately this bit size
        let p = BigInt::from(1u64) << (bits / 2 - 1) | BigInt::from(1u64);
        let q = BigInt::from(1u64) << (bits / 2) | BigInt::from(1u64);
        let n = &p * &q;
        
        let start = Instant::now();
        let _result = recognize_factors(&n, params);
        let elapsed = start.elapsed();
        
        println!("{:4} | {:8} | {:14} | {:?}", 
            bits, channels, 
            if n < BigInt::from(1_000_000u64) { n.to_string() } else { format!("~2^{}", n.bits()) },
            elapsed);
    }
}

fn test_specific_patterns(params: &TunerParams) {
    println!("\n\n=== Testing Specific Patterns ===\n");
    
    // Test numbers with specific channel patterns
    let pattern_tests = vec![
        // Repeating patterns
        ("Repeating AA", BigInt::from(0xAAAAu32), "2 × 21845"),
        ("Repeating 55", BigInt::from(0x5555u32), "5 × 17 × 257"),
        ("Repeating FF", BigInt::from(0xFFFFu32), "3 × 5 × 17 × 257"),
        
        // Sequential patterns
        ("Sequential 01 02", BigInt::from(0x0102u32), "2 × 129"),
        ("Sequential FE FD", BigInt::from(0xFEFDu32), "65277 (prime)"),
        
        // Palindromic patterns
        ("Palindrome ABBA", BigInt::from(0xABBAu32), "2 × 21949"),
        ("Palindrome 1221", BigInt::from(0x1221u32), "3 × 1547"),
    ];
    
    for (pattern_name, n, expected) in pattern_tests {
        let channels = decompose(&n);
        let channel_str = channels.iter()
            .map(|c| format!("{:02x}", c))
            .collect::<Vec<_>>()
            .join(" ");
        
        print!("{:20} ({}) = {:20} ... ", pattern_name, channel_str, expected);
        
        let start = Instant::now();
        let result = recognize_factors(&n, params);
        let elapsed = start.elapsed();
        
        if result.is_some() && result.as_ref().unwrap().verify(&n) {
            if expected.contains("prime") {
                println!("✗ incorrectly factored");
            } else {
                println!("✓ factored in {:?}", elapsed);
            }
        } else {
            if expected.contains("prime") {
                println!("✓ correctly identified");
            } else {
                println!("✗ failed to factor");
            }
        }
    }
    
    // Test very large numbers (many channels)
    println!("\n\nTesting very large numbers:");
    
    let large_tests = vec![
        (96, BigInt::from(79228162514264337593543950319u128), "3 × 59 × 72057594037927936676129"),
        (100, BigInt::parse_bytes(b"1267650600228229401496703205361", 10).unwrap(), "large prime"),
        (128, BigInt::parse_bytes(b"340282366920938463463374607431768211297", 10).unwrap(), "59 × 5765078824288161635987240802351393"),
    ];
    
    for (bits, n, expected) in large_tests {
        let channels = decompose(&n);
        println!("\n{}-bit number ({} channels): {}", bits, channels.len(), expected);
        println!("First 8 channels: {:?}", &channels[..8.min(channels.len())]);
        
        let start = Instant::now();
        let result = recognize_factors(&n, params);
        let elapsed = start.elapsed();
        
        if result.is_some() && result.as_ref().unwrap().verify(&n) {
            println!("✓ Factored in {:?}", elapsed);
        } else {
            if expected.contains("prime") {
                println!("✓ Correctly identified as prime in {:?}", elapsed);
            } else {
                println!("✗ Failed to factor in {:?}", elapsed);
            }
        }
    }
}