//! Basis tuner for optimizing pre-computed patterns
//! Analyzes failures and suggests basis improvements

use rust_pattern_solver::pattern::precomputed_basis::UniversalBasis;
use rust_pattern_solver::pattern::enhanced_basis::EnhancedUniversalBasis;
use rust_pattern_solver::types::Number;
use std::str::FromStr;
use std::time::Instant;
use std::collections::HashMap;

struct BasisAnalyzer {
    standard_basis: UniversalBasis,
    enhanced_basis: EnhancedUniversalBasis,
}

impl BasisAnalyzer {
    fn new() -> Self {
        println!("Loading basis configurations...");
        let standard = UniversalBasis::load_or_create(
            std::path::Path::new("data/basis/universal_basis.json")
        );
        let enhanced = EnhancedUniversalBasis::load_or_create(
            std::path::Path::new("data/basis/enhanced_basis.json")
        );
        
        BasisAnalyzer {
            standard_basis: standard,
            enhanced_basis: enhanced,
        }
    }
    
    fn analyze_number(&self, n: &Number) -> BasisAnalysis {
        let n_bits = n.bit_length();
        let sqrt_n = rust_pattern_solver::utils::integer_sqrt(n).unwrap();
        
        // Analyze with standard basis
        let std_start = Instant::now();
        let std_scaled = self.standard_basis.scale_to_number(n);
        let std_scale_time = std_start.elapsed();
        
        // Check resonance template coverage
        let template_size = std_scaled.scaled_resonance.len();
        let has_appropriate_template = self.standard_basis.resonance_templates
            .keys()
            .any(|&bits| bits as usize >= n_bits && bits as usize <= n_bits + 64);
        
        // Analyze universal coordinates
        let coords = std_scaled.universal_coords;
        let phi_coord = coords[0];
        let pi_coord = coords[1];
        let e_coord = coords[2];
        
        // Check if this is likely a balanced semiprime
        // For balanced semiprimes, both factors have approximately n_bits/2 bits
        let expected_factor_bits = n_bits / 2;
        let expected_factor_phi = (expected_factor_bits as f64) * 0.5; // Rough estimate
        let balance_indicator = (phi_coord / 2.0 - expected_factor_phi).abs();
        
        // Analyze with enhanced basis
        let enh_start = Instant::now();
        let enh_result = self.enhanced_basis.find_factors_enhanced(n);
        let enh_time = enh_start.elapsed();
        
        // Check offset distribution coverage
        let has_offset_dist = self.enhanced_basis.offset_distributions
            .contains_key(&((n_bits as u32 / 10) * 10));
        
        BasisAnalysis {
            bit_size: n_bits,
            sqrt_n_bits: sqrt_n.bit_length(),
            phi_coordinate: phi_coord,
            pi_coordinate: pi_coord,
            e_coordinate: e_coord,
            balance_indicator,
            has_appropriate_template,
            template_size,
            has_offset_distribution: has_offset_dist,
            standard_scale_time: std_scale_time,
            enhanced_search_time: enh_time,
            enhanced_found_factors: enh_result.is_ok(),
        }
    }
    
    fn suggest_improvements(&self, analyses: &[BasisAnalysis]) {
        println!("\n{}", "=".repeat(70));
        println!("BASIS IMPROVEMENT SUGGESTIONS");
        println!("{}", "=".repeat(70));
        
        // Group by bit size ranges
        let mut by_range: HashMap<usize, Vec<&BasisAnalysis>> = HashMap::new();
        for analysis in analyses {
            let range = (analysis.bit_size / 32) * 32; // 32-bit ranges
            by_range.entry(range).or_insert(Vec::new()).push(analysis);
        }
        
        println!("\n1. Template Coverage Gaps:");
        for (range, analyses) in &by_range {
            let missing_templates = analyses.iter()
                .filter(|a| !a.has_appropriate_template)
                .count();
            if missing_templates > 0 {
                println!("   - Add templates for {}-{} bit range ({} missing)",
                         range, range + 31, missing_templates);
            }
        }
        
        println!("\n2. Offset Distribution Gaps:");
        for (range, analyses) in &by_range {
            let missing_dists = analyses.iter()
                .filter(|a| !a.has_offset_distribution)
                .count();
            if missing_dists > 0 {
                println!("   - Add offset distributions for {}-{} bit range ({} missing)",
                         range, range + 31, missing_dists);
            }
        }
        
        println!("\n3. Balance Indicator Patterns:");
        let balanced: Vec<_> = analyses.iter()
            .filter(|a| a.balance_indicator < 1.0)
            .collect();
        if !balanced.is_empty() {
            let avg_phi = balanced.iter()
                .map(|a| a.phi_coordinate)
                .sum::<f64>() / balanced.len() as f64;
            println!("   - Balanced semiprimes show average φ-coordinate: {:.4}", avg_phi);
            println!("   - Consider pre-computing patterns around φ/2 = {:.4}", avg_phi / 2.0);
        }
        
        println!("\n4. Performance Optimization:");
        let slow_scaling = analyses.iter()
            .filter(|a| a.standard_scale_time.as_millis() > 10)
            .count();
        if slow_scaling > 0 {
            println!("   - {} numbers took >10ms to scale", slow_scaling);
            println!("   - Consider caching scaled bases for common bit sizes");
        }
        
        println!("\n5. Enhanced Basis Effectiveness:");
        let enhanced_success = analyses.iter()
            .filter(|a| a.enhanced_found_factors)
            .count();
        let total = analyses.len();
        println!("   - Enhanced basis success rate: {}/{} ({:.1}%)",
                 enhanced_success, total,
                 100.0 * enhanced_success as f64 / total as f64);
        
        if enhanced_success < total / 2 {
            println!("   - Enhanced basis needs more pre-computed patterns");
            println!("   - Focus on bit ranges where standard basis fails");
        }
        
        println!("\n6. Recommended Basis Expansions:");
        
        // Find bit ranges with poor performance
        let problem_ranges: Vec<_> = by_range.iter()
            .filter(|(_, analyses)| {
                let success = analyses.iter().filter(|a| a.enhanced_found_factors).count();
                success as f64 / analyses.len() as f64 < 0.5
            })
            .map(|(range, _)| *range)
            .collect();
        
        if !problem_ranges.is_empty() {
            println!("   Priority bit ranges for expansion: {:?}", problem_ranges);
            
            for range in problem_ranges {
                println!("\n   For {}-{} bit numbers:", range, range + 31);
                println!("   - Generate resonance templates every 2 bits");
                println!("   - Pre-compute factor offsets for top 1000 patterns");
                println!("   - Add harmonic functions up to 100th order");
                println!("   - Cache matrix decompositions for this range");
            }
        }
    }
}

#[derive(Debug)]
struct BasisAnalysis {
    bit_size: usize,
    sqrt_n_bits: usize,
    phi_coordinate: f64,
    pi_coordinate: f64,
    e_coordinate: f64,
    balance_indicator: f64,
    has_appropriate_template: bool,
    template_size: usize,
    has_offset_distribution: bool,
    standard_scale_time: std::time::Duration,
    enhanced_search_time: std::time::Duration,
    enhanced_found_factors: bool,
}

fn main() {
    println!("=== BASIS TUNER FOR HARD SEMIPRIMES ===\n");
    
    let analyzer = BasisAnalyzer::new();
    let mut analyses = Vec::new();
    
    // Test numbers of increasing size
    let test_numbers = vec![
        // Small (should work well)
        "143",              // 8-bit
        "10403",            // 14-bit
        "25217",            // 15-bit
        
        // Medium (may have issues)
        "852391",           // 20-bit
        "1000000007",       // 30-bit (prime)
        "9223372012704246007", // 64-bit
        
        // Large (likely to fail with current basis)
        "170141183460469231731687303715884105727", // 127-bit
        
        // Generate some balanced semiprimes
        "961",              // 31 × 31
        "169",              // 13 × 13
        "60073",            // 241 × 249 (close factors)
    ];
    
    println!("Analyzing basis performance on test numbers:\n");
    
    for n_str in test_numbers {
        print!("Analyzing {}: ", n_str);
        match Number::from_str(n_str) {
            Ok(n) => {
                let analysis = analyzer.analyze_number(&n);
                println!("{}-bit, φ={:.4}, balanced={:.4}, enhanced={}",
                         analysis.bit_size,
                         analysis.phi_coordinate,
                         analysis.balance_indicator,
                         if analysis.enhanced_found_factors { "✓" } else { "✗" });
                analyses.push(analysis);
            }
            Err(e) => println!("Error: {}", e),
        }
    }
    
    // Generate improvement suggestions
    analyzer.suggest_improvements(&analyses);
    
    println!("\n7. Specific Tuning Parameters:");
    println!("   Current basis sizes:");
    println!("   - Standard: 5×5 matrix, 8 templates, 7 harmonics");
    println!("   - Enhanced: 100×100 matrix, 511 templates, 50 harmonics");
    
    println!("\n   Recommended for 300-bit capability:");
    println!("   - Matrix: 500×500 or larger");
    println!("   - Templates: Every bit from 8 to 512");
    println!("   - Harmonics: Up to 200th order");
    println!("   - Offset distributions: Empirically derived from RSA numbers");
    println!("   - Factor lookup: Pre-compute for all balanced patterns");
    
    println!("\n8. Implementation Strategy:");
    println!("   1. Start with empirical data collection on known factors");
    println!("   2. Identify patterns in phi-coordinate relationships");
    println!("   3. Build lookup tables for common factor distances");
    println!("   4. Implement lazy loading for large basis components");
    println!("   5. Use memory-mapped files for huge pre-computed data");
}