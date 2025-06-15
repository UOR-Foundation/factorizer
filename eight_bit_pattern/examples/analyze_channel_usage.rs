//! Analyze channel usage for different bit sizes
//!
//! Run with: cargo run --example analyze_channel_usage

use eight_bit_pattern::{decompose, bit_size};
use num_bigint::BigInt;
use num_traits::One;

fn main() {
    println!("=== Channel Usage Analysis for Different Bit Sizes ===\n");
    
    // Test specific bit sizes
    let bit_sizes = vec![4, 8, 16, 32, 64, 128, 256, 512, 1024];
    
    println!("Theoretical channel requirements:");
    println!("Bit Size | Channels Needed | Bytes");
    println!("---------|-----------------|-------");
    
    for bits in &bit_sizes {
        let channels_needed = (*bits + 7) / 8;  // ceil(bits/8)
        println!("{:8} | {:15} | {:6}", bits, channels_needed, channels_needed);
    }
    
    println!("\n\nActual channel usage for specific numbers:");
    println!("Number | Bit Size | Channels | Channel Values");
    println!("-------|----------|----------|---------------");
    
    // Small numbers
    analyze_number(&BigInt::from(15u32), "15 (3×5)");
    analyze_number(&BigInt::from(143u32), "143 (11×13)");
    analyze_number(&BigInt::from(65535u32), "65535 (255×257)");
    
    // Powers of 2 minus 1 (all bits set)
    analyze_number(&((BigInt::one() << 16) - 1), "2^16 - 1");
    analyze_number(&((BigInt::one() << 32) - 1), "2^32 - 1");
    analyze_number(&((BigInt::one() << 64) - 1), "2^64 - 1");
    analyze_number(&((BigInt::one() << 128) - 1), "2^128 - 1");
    analyze_number(&((BigInt::one() << 256) - 1), "2^256 - 1");
    analyze_number(&((BigInt::one() << 512) - 1), "2^512 - 1");
    analyze_number(&((BigInt::one() << 1024) - 1), "2^1024 - 1");
    
    // Specific semiprimes at different scales
    println!("\n\nSemiprimes at different scales:");
    
    // 16-bit semiprime
    let p16 = BigInt::from(251u32);
    let q16 = BigInt::from(257u32);
    analyze_number(&(&p16 * &q16), "251 × 257");
    
    // 32-bit semiprime  
    let p32 = BigInt::from(65521u32);
    let q32 = BigInt::from(65537u32);
    analyze_number(&(&p32 * &q32), "65521 × 65537");
    
    // 64-bit semiprime
    let p64 = BigInt::from(4294967291u64);
    let q64 = BigInt::from(4294967311u64);
    analyze_number(&(&p64 * &q64), "4294967291 × 4294967311");
    
    // Channel distribution analysis
    println!("\n\n=== Channel Distribution Analysis ===\n");
    
    // Analyze how many channels are typically non-zero
    analyze_channel_distribution();
}

fn analyze_number(n: &BigInt, label: &str) {
    let channels = decompose(n);
    let bits = bit_size(n);
    
    print!("{:<15} | {:8} | {:8} | ", label, bits, channels.len());
    
    // Show first few and last few channels
    if channels.len() <= 8 {
        println!("{:?}", channels);
    } else {
        print!("[");
        for i in 0..4 {
            print!("{}", channels[i]);
            if i < 3 { print!(", "); }
        }
        print!(", ..., ");
        for i in (channels.len()-4)..channels.len() {
            print!("{}", channels[i]);
            if i < channels.len() - 1 { print!(", "); }
        }
        println!("]");
    }
}

fn analyze_channel_distribution() {
    println!("Channel activity patterns:");
    
    // Test numbers with known bit patterns
    let test_cases = vec![
        ("Single bit set (2^n)", vec![
            BigInt::one() << 8,
            BigInt::one() << 16,
            BigInt::one() << 32,
            BigInt::one() << 64,
            BigInt::one() << 128,
        ]),
        ("All bits set (2^n - 1)", vec![
            (BigInt::one() << 8) - 1,
            (BigInt::one() << 16) - 1,
            (BigInt::one() << 32) - 1,
            (BigInt::one() << 64) - 1,
            (BigInt::one() << 128) - 1,
        ]),
        ("Random semiprimes", vec![
            BigInt::from(143u32),  // 11 × 13
            BigInt::from(9409u32), // 97 × 97
            BigInt::from(1299709u32), // 1009 × 1289
            BigInt::from(4294967291u64) * BigInt::from(4294967311u64),
        ]),
    ];
    
    for (category, numbers) in test_cases {
        println!("\n{}:", category);
        println!("Bits | Total Ch | Non-zero Ch | Zero Ch | Fill Rate");
        println!("-----|----------|-------------|---------|----------");
        
        for n in numbers {
            let channels = decompose(&n);
            let bits = bit_size(&n);
            let non_zero = channels.iter().filter(|&&ch| ch != 0).count();
            let zero = channels.len() - non_zero;
            let fill_rate = non_zero as f64 / channels.len() as f64 * 100.0;
            
            println!("{:4} | {:8} | {:11} | {:7} | {:6.1}%",
                bits, channels.len(), non_zero, zero, fill_rate);
        }
    }
    
    // Key observations
    println!("\n\nKey Observations:");
    println!("1. Little-endian ordering: least significant byte is channel[0]");
    println!("2. Channel count = number of bytes needed to represent the number");
    println!("3. Leading zeros are NOT included (unlike theoretical calculation)");
    println!("4. For n-bit number: channels ≈ ceil(n/8), but may be less due to no padding");
}