//! Generate authoritative test matrix of hard semiprimes
//! 
//! This creates an immutable test matrix with verified hard semiprimes
//! at each bit length for benchmarking The Pattern implementation.

use rust_pattern_solver::types::Number;
use rand::{thread_rng, Rng};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestCase {
    bit_length: usize,
    n: String,
    p: String,
    q: String,
    balanced: bool,
    p_bits: usize,
    q_bits: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct TestMatrix {
    version: String,
    generated: String,
    description: String,
    test_cases: BTreeMap<usize, Vec<TestCase>>,
}

/// Generate a random prime of approximately n bits
fn generate_random_prime(bits: usize) -> Number {
    let mut rng = thread_rng();
    loop {
        // Generate a random odd number with the desired bit length
        let mut candidate = Number::from(1u32) << (bits as u32 - 1); // Set high bit
        
        // Fill remaining bits randomly
        for bit_pos in 0..(bits as u32 - 1) {
            if rng.gen::<bool>() {
                candidate.set_bit(bit_pos, true);
            }
        }
        
        // Ensure odd for primality
        candidate.set_bit(0, true);
        
        // Check the bit length is correct
        if candidate.bit_length() != bits {
            continue;
        }
        
        // Miller-Rabin primality test
        if rust_pattern_solver::utils::is_probable_prime(&candidate, 40) {
            return candidate;
        }
    }
}

/// Generate hard semiprimes for a given bit length
fn generate_hard_semiprimes(target_bits: usize, count: usize) -> Vec<TestCase> {
    let mut cases = Vec::new();
    let mut rng = thread_rng();
    
    for i in 0..count {
        // Vary the balance of factors
        let (p_bits, q_bits) = if i < count / 3 {
            // Balanced case: both factors approximately equal size
            let p_bits = target_bits / 2;
            let q_bits = target_bits - p_bits;
            (p_bits, q_bits)
        } else if i < 2 * count / 3 {
            // Slightly unbalanced
            let delta = rng.gen_range(1..=(target_bits / 8).max(1));
            let p_bits = target_bits / 2 - delta;
            let q_bits = target_bits - p_bits;
            (p_bits, q_bits)
        } else {
            // More unbalanced (but not extreme)
            let delta = rng.gen_range(1..=(target_bits / 6).max(1));
            let p_bits = target_bits / 2 - delta;
            let q_bits = target_bits - p_bits;
            (p_bits, q_bits)
        };
        
        // Generate primes
        let p = generate_random_prime(p_bits);
        let q = generate_random_prime(q_bits);
        
        // Ensure p <= q for consistency
        let (p, q) = if p <= q { (p, q) } else { (q, p) };
        
        let n = &p * &q;
        let n_bits = n.bit_length();
        
        // Only keep if within tolerance of target
        if n_bits >= target_bits - 1 && n_bits <= target_bits + 1 {
            let balanced = (p.bit_length() as i32 - q.bit_length() as i32).abs() <= 2;
            
            cases.push(TestCase {
                bit_length: n_bits,
                n: n.to_string(),
                p: p.to_string(),
                q: q.to_string(),
                balanced,
                p_bits: p.bit_length(),
                q_bits: q.bit_length(),
            });
        }
    }
    
    cases
}

fn main() {
    println!("Generating authoritative test matrix for The Pattern...");
    
    let mut test_matrix = TestMatrix {
        version: "1.0.0".to_string(),
        generated: chrono::Utc::now().to_rfc3339(),
        description: "Authoritative test matrix of hard semiprimes for The Pattern implementation. \
                     This matrix contains verified random hard semiprimes at each bit length. \
                     The factors are provided for verification only and must not be used by \
                     the implementation during factorization.".to_string(),
        test_cases: BTreeMap::new(),
    };
    
    // Generate test cases for each bit length
    // Start small and go up to large sizes
    let bit_lengths = vec![
        8, 16, 24, 32, 40, 48, 56, 64,           // Small
        72, 80, 88, 96, 104, 112, 120, 128,      // Medium
        136, 144, 152, 160, 168, 176, 184, 192,  // Large
        200, 208, 216, 224, 232, 240, 248, 256,  // Very large
        264, 272, 280, 288, 296, 304, 312, 320,  // Huge
        384, 448, 512, 768, 1024,                // Extreme
    ];
    
    for &bits in &bit_lengths {
        println!("Generating {}-bit test cases...", bits);
        
        // More test cases for smaller sizes, fewer for larger
        let count = match bits {
            8..=64 => 20,
            65..=128 => 15,
            129..=256 => 10,
            257..=512 => 5,
            _ => 3,
        };
        
        let cases = generate_hard_semiprimes(bits, count);
        println!("  Generated {} test cases", cases.len());
        
        test_matrix.test_cases.insert(bits, cases);
    }
    
    // Save to file
    let json = serde_json::to_string_pretty(&test_matrix).unwrap();
    let mut file = File::create("data/test_matrix.json").unwrap();
    file.write_all(json.as_bytes()).unwrap();
    
    println!("\nTest matrix saved to data/test_matrix.json");
    
    // Print summary
    let total_cases: usize = test_matrix.test_cases.values().map(|v| v.len()).sum();
    println!("\nSummary:");
    println!("  Total bit lengths: {}", test_matrix.test_cases.len());
    println!("  Total test cases: {}", total_cases);
    
    // Print sample distribution
    println!("\nTest case distribution:");
    for (bits, cases) in &test_matrix.test_cases {
        let balanced_count = cases.iter().filter(|c| c.balanced).count();
        println!("  {}-bit: {} cases ({} balanced)", bits, cases.len(), balanced_count);
    }
}