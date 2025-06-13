//! Test with expanded pre-computed basis for better RSA performance

use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;
use std::str::FromStr;
use std::time::Instant;

fn main() {
    println!("=== Expanded Pre-computed Basis Analysis ===\n");
    
    // The current basis limitations
    println!("Current basis structure:");
    println!("- 5x5 factor relationship matrix (25 values)");
    println!("- 8 resonance templates (8, 16, 32... 1024 bits)");
    println!("- 7 harmonic basis functions");
    println!("- Total pre-computed values: ~1000 floats\n");
    
    println!("For RSA-scale numbers, we need:");
    println!("1. Larger factor relationship matrix (100x100 or more)");
    println!("2. More resonance templates with finer granularity");
    println!("3. Higher-order harmonic functions");
    println!("4. Pre-computed patterns for balanced semiprime signatures");
    println!("5. Lookup tables for common factor distances from sqrt(n)\n");
    
    // Test current performance
    let mut pattern = UniversalPattern::with_precomputed_basis();
    
    // Small test case
    let n = Number::from_str("9223372012704246007").unwrap(); // 64-bit
    println!("Testing 64-bit number: {}", n);
    
    let start = Instant::now();
    match pattern.recognize(&n) {
        Ok(recognition) => {
            match pattern.formalize(recognition) {
                Ok(formalization) => {
                    match pattern.execute(formalization) {
                        Ok(factors) => {
                            println!("✓ Factored in {:?} via {}", start.elapsed(), factors.method);
                            println!("  {} × {} = {}", factors.p, factors.q, &factors.p * &factors.q);
                        }
                        Err(e) => println!("✗ Execution failed: {}", e),
                    }
                }
                Err(e) => println!("✗ Formalization failed: {}", e),
            }
        }
        Err(e) => println!("✗ Recognition failed: {}", e),
    }
    
    println!("\n{}", "=".repeat(70));
    println!("RECOMMENDATIONS FOR EXPANDED BASIS");
    println!("{}", "=".repeat(70));
    
    println!("\n1. Pre-compute more pattern data:");
    println!("   - Factor distances for all bit sizes (store as percentages of sqrt(n))");
    println!("   - Common resonance patterns for balanced semiprimes");
    println!("   - Phi-sum relationships at different scales");
    
    println!("\n2. Use hierarchical basis:");
    println!("   - Coarse basis for initial recognition");
    println!("   - Fine basis for precise factor location");
    println!("   - Specialized basis for RSA-like numbers");
    
    println!("\n3. Memory vs Speed tradeoff:");
    println!("   - Current: ~10KB of pre-computed data");
    println!("   - Recommended: 1-10MB for general purpose");
    println!("   - RSA-optimized: 100MB-1GB for instant recognition");
    
    println!("\n4. Pattern-specific optimizations:");
    println!("   - RSA numbers have factors at ~0.5 * sqrt(n) ± small_offset");
    println!("   - Pre-compute offset distributions for each bit size");
    println!("   - Use statistical models of factor locations");
}