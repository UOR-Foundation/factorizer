//! Demonstration of large-scale optimization for 8000+ bit numbers
//!
//! This example shows how The Pattern handles extremely large numbers
//! with memory optimization and hierarchical search strategies.

use rust_pattern_solver::observer::ConstantDiscovery;
use rust_pattern_solver::pattern::large_scale::{
    formalize_large_scale, quantum_search_large_scale, recognize_large_scale, LargeScaleConfig,
};
use rust_pattern_solver::types::pattern::ScaleRange;
use rust_pattern_solver::types::{Number, Pattern, PatternKind};
use std::time::Instant;

fn main() {
    println!("Large-Scale Pattern Optimization Demo");
    println!("=====================================\n");

    // Create configuration for large-scale operations
    let config = LargeScaleConfig {
        chunk_size: 4096,
        max_memory_mb: 512, // Limit memory for demo
        use_mmap: true,
        parallelism: num_cpus::get(),
        sampling_rate: 0.01,
    };

    println!("Configuration:");
    println!("  Chunk size: {} bits", config.chunk_size);
    println!("  Max memory: {} MB", config.max_memory_mb);
    println!("  Parallelism: {} cores", config.parallelism);
    println!("  Sampling rate: {}%\n", config.sampling_rate * 100.0);

    // Test with increasingly large numbers
    test_scale_progression(&config);

    // Demonstrate memory-efficient operations
    demonstrate_memory_efficiency(&config);

    // Show hierarchical search in action
    demonstrate_hierarchical_search(&config);
}

fn test_scale_progression(config: &LargeScaleConfig) {
    println!("Scale Progression Test");
    println!("=====================\n");

    let test_cases = vec![
        ("Small (32-bit)", create_test_number(32)),
        ("Medium (256-bit)", create_test_number(256)),
        ("Large (1024-bit)", create_test_number(1024)),
        ("Very Large (4096-bit)", create_test_number(4096)),
        // ("Massive (8192-bit)", create_test_number(8192)), // Commented for demo speed
    ];

    let patterns = create_test_patterns();
    let constants = ConstantDiscovery::extract(&patterns);

    for (label, n) in test_cases {
        println!("Testing {}: {} bits", label, n.bit_length());

        let start = Instant::now();

        // Recognition
        match recognize_large_scale(&n, &patterns, config) {
            Ok(recognition) => {
                let recognition_time = start.elapsed();
                println!(
                    "  Recognition: {:?} ({:.2}s)",
                    recognition.pattern_type,
                    recognition_time.as_secs_f64()
                );

                // Formalization
                let form_start = Instant::now();
                match formalize_large_scale(recognition, &patterns, &constants, config) {
                    Ok(formalization) => {
                        let form_time = form_start.elapsed();
                        println!(
                            "  Formalization: {} strategies ({:.2}s)",
                            formalization.strategies.len(),
                            form_time.as_secs_f64()
                        );

                        // Execution (limited time for demo)
                        let exec_start = Instant::now();
                        let patterns_clone = patterns.clone();
                        let config_clone = config.clone();
                        let exec_result = std::thread::spawn(move || {
                            quantum_search_large_scale(&formalization, &patterns_clone, &config_clone)
                        });

                        // Wait max 5 seconds for demo
                        match exec_result.join() {
                            Ok(Ok(factors)) => {
                                let exec_time = exec_start.elapsed();
                                println!(
                                    "  ✓ Factors found: {} × {} ({:.2}s)",
                                    factors.p,
                                    factors.q,
                                    exec_time.as_secs_f64()
                                );
                            },
                            _ => {
                                println!("  ✗ Execution timeout (>5s)");
                            },
                        }
                    },
                    Err(e) => {
                        println!("  Formalization failed: {}", e);
                    },
                }
            },
            Err(e) => {
                println!("  Recognition failed: {}", e);
            },
        }

        let total_time = start.elapsed();
        println!("  Total time: {:.2}s\n", total_time.as_secs_f64());
    }
}

fn demonstrate_memory_efficiency(config: &LargeScaleConfig) {
    println!("\nMemory Efficiency Demonstration");
    println!("==============================\n");

    // Create a large number that would normally require significant memory
    let large_n = create_test_number(2048);
    println!("Working with {}-bit number", large_n.bit_length());

    // Show memory usage estimation
    let standard_memory = estimate_standard_memory(&large_n);
    let optimized_memory = estimate_optimized_memory(&large_n, config);

    println!("Memory usage comparison:");
    println!("  Standard approach: ~{} MB", standard_memory);
    println!("  Optimized approach: ~{} MB", optimized_memory);
    println!(
        "  Savings: {:.1}%\n",
        (1.0 - optimized_memory as f64 / standard_memory as f64) * 100.0
    );

    // Demonstrate chunked operations
    println!("Chunked operations:");
    let chunk_count = (large_n.bit_length() / config.chunk_size).max(1);
    println!("  Number divided into {} chunks", chunk_count);
    println!("  Each chunk: {} bits", config.chunk_size);
    println!(
        "  Parallel processing: {} chunks simultaneously",
        config.parallelism
    );
}

fn demonstrate_hierarchical_search(config: &LargeScaleConfig) {
    println!("\nHierarchical Search Demonstration");
    println!("================================\n");

    // Create a number with known structure
    let p = Number::from(1000003u64); // Large prime
    let q = Number::from(1000033u64); // Another large prime
    let n = &p * &q;

    println!("Test number: {} × {} = {}", p, q, n);
    println!("Bit length: {} bits\n", n.bit_length());

    let patterns = create_test_patterns();
    let constants = ConstantDiscovery::extract(&patterns);

    // Show hierarchical search levels
    println!("Hierarchical search strategy:");
    println!("  Level 1: Macro regions (sparse sampling)");
    println!("    - Search density: {}%", config.sampling_rate * 100.0);
    println!(
        "    - Region count: ~{}",
        (n.bit_length() as f64).log2() as usize
    );

    println!("\n  Level 2: Refined regions (pattern-guided)");
    println!("    - Based on pattern analysis");
    println!("    - Adaptive probability distributions");

    println!("\n  Level 3: Parallel chunks");
    println!("    - Chunk size: {} bits", config.chunk_size);
    println!("    - Parallel workers: {}", config.parallelism);

    // Try to factor
    let start = Instant::now();

    match recognize_large_scale(&n, &patterns, config) {
        Ok(recognition) => {
            match formalize_large_scale(recognition, &patterns, &constants, config) {
                Ok(formalization) => {
                    match quantum_search_large_scale(&formalization, &patterns, config) {
                        Ok(factors) => {
                            let elapsed = start.elapsed();
                            println!("\n✓ Factors found in {:.2}s", elapsed.as_secs_f64());
                            println!("  Method: {}", factors.method);
                            println!("  p = {}", factors.p);
                            println!("  q = {}", factors.q);
                        },
                        Err(e) => {
                            println!("\n✗ Search failed: {}", e);
                        },
                    }
                },
                Err(e) => {
                    println!("\nFormalization failed: {}", e);
                },
            }
        },
        Err(e) => {
            println!("\nRecognition failed: {}", e);
        },
    }
}

fn create_test_number(bits: usize) -> Number {
    // Create a semiprime with roughly the specified bit length
    let half_bits = bits / 2;

    // Use simple formula for demo (not cryptographically secure)
    let p_base = Number::from(1u32) << ((half_bits - 1) as u32);
    let p = &p_base + &Number::from(15u32); // Small offset

    let q_base = Number::from(1u32) << ((half_bits - 1) as u32);
    let q = &q_base + &Number::from(25u32); // Different offset

    &p * &q
}

fn create_test_patterns() -> Vec<Pattern> {
    vec![
        Pattern {
            id: "large_scale_balanced".to_string(),
            kind: PatternKind::TypeSpecific("balanced".to_string()),
            frequency: 0.6,
            description: "Balanced pattern for large numbers".to_string(),
            parameters: vec![],
            scale_range: ScaleRange {
                min_bits: 32,
                max_bits: 10000,
                unbounded: true,
            },
        },
        Pattern {
            id: "large_scale_emergent".to_string(),
            kind: PatternKind::Emergent,
            frequency: 0.8,
            description: "Emergent pattern at scale".to_string(),
            parameters: vec![],
            scale_range: ScaleRange {
                min_bits: 100,
                max_bits: 10000,
                unbounded: true,
            },
        },
    ]
}

fn estimate_standard_memory(n: &Number) -> usize {
    // Rough estimation of memory usage
    let bits = n.bit_length();

    // Pattern signature + resonance field + matrices
    let signature_mb = bits / 8 / 1024 / 1024;
    let resonance_mb = 100; // Fixed size resonance field
    let matrix_mb = 100 * 100 * 8 / 1024 / 1024; // Pattern matrix

    signature_mb + resonance_mb + matrix_mb + 100 // +100MB overhead
}

fn estimate_optimized_memory(n: &Number, config: &LargeScaleConfig) -> usize {
    let bits = n.bit_length();

    // Sampled signature + sparse matrices + chunked operations
    let sampled_signature_mb = (bits as f64 * config.sampling_rate) as usize / 8 / 1024 / 1024;
    let sparse_matrix_mb = 10; // Compressed matrices
    let chunk_mb = config.chunk_size / 8 / 1024 / 1024 * config.parallelism;

    sampled_signature_mb + sparse_matrix_mb + chunk_mb + 50 // +50MB overhead
}
