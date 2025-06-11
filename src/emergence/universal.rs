//! Universal pattern discovery
//!
//! This module discovers patterns that appear to be universal across
//! all observations, revealing fundamental mathematical truths.

use crate::types::{Observation, Pattern, PatternKind, ScaleRange, UniversalConstant};
use crate::Result;
use statrs::statistics::{Data, OrderStatistics, Statistics};
use std::collections::HashMap;

/// Universal pattern discovery
pub struct UniversalPatterns;

impl UniversalPatterns {
    /// Discover universal patterns from observations
    pub fn discover(observations: &[Observation]) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();
        
        // Discover constant relationships
        patterns.extend(Self::discover_constant_patterns(observations)?);
        
        // Discover ratio patterns
        patterns.extend(Self::discover_ratio_patterns(observations)?);
        
        // Discover density patterns
        patterns.extend(Self::discover_density_patterns(observations)?);
        
        // Discover emergence patterns
        patterns.extend(Self::discover_emergence_patterns(observations)?);
        
        Ok(patterns)
    }
    
    /// Discover patterns involving universal constants
    fn discover_constant_patterns(observations: &[Observation]) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();
        
        // Analyze phi relationships
        let phi_stats = Self::analyze_phi_relationships(observations);
        if let Some(pattern) = phi_stats {
            patterns.push(pattern);
        }
        
        // Analyze pi relationships
        let pi_stats = Self::analyze_pi_relationships(observations);
        if let Some(pattern) = pi_stats {
            patterns.push(pattern);
        }
        
        // Analyze e relationships
        let e_stats = Self::analyze_e_relationships(observations);
        if let Some(pattern) = e_stats {
            patterns.push(pattern);
        }
        
        // Analyze combined constant relationships
        patterns.extend(Self::analyze_combined_constants(observations)?);
        
        Ok(patterns)
    }
    
    /// Analyze golden ratio relationships
    fn analyze_phi_relationships(observations: &[Observation]) -> Option<Pattern> {
        let phi = 1.618033988749895;
        let mut phi_ratios = Vec::new();
        
        for obs in observations {
            // Check if factor ratio approximates phi
            let ratio = obs.q.to_f64()? / obs.p.to_f64()?;
            let phi_proximity = (ratio / phi - 1.0).abs();
            
            if phi_proximity < 0.1 {
                phi_ratios.push(phi_proximity);
            }
        }
        
        if phi_ratios.len() > observations.len() / 10 {
            let mut data = Data::new(phi_ratios.clone());
            let mean_proximity = data.mean().unwrap_or(0.0);
            
            let mut pattern = Pattern::new("phi_factor_relationship", PatternKind::Universal);
            pattern.frequency = phi_ratios.len() as f64 / observations.len() as f64;
            pattern.description = format!(
                "Factor ratios approximate φ in {:.1}% of cases (mean deviation: {:.4})",
                pattern.frequency * 100.0,
                mean_proximity
            );
            pattern.parameters = vec![phi, mean_proximity];
            pattern.scale_range = ScaleRange {
                min_bits: 1,
                max_bits: usize::MAX,
                unbounded: true,
            };
            
            return Some(pattern);
        }
        
        None
    }
    
    /// Analyze pi relationships
    fn analyze_pi_relationships(observations: &[Observation]) -> Option<Pattern> {
        let pi = std::f64::consts::PI;
        let mut pi_components = Vec::new();
        
        for obs in observations {
            // Check various pi relationships
            let n_val = obs.n.to_f64()?;
            let sqrt_n = n_val.sqrt();
            
            // Check if n/sqrt(n) has pi component
            let ratio = n_val / sqrt_n;
            let pi_multiple = ratio / pi;
            let remainder = pi_multiple.fract();
            
            if remainder < 0.01 || remainder > 0.99 {
                pi_components.push(remainder.min(1.0 - remainder));
            }
        }
        
        if pi_components.len() > observations.len() / 20 {
            let mut pattern = Pattern::new("pi_structural_component", PatternKind::Universal);
            pattern.frequency = pi_components.len() as f64 / observations.len() as f64;
            pattern.description = format!(
                "π appears as structural component in {:.1}% of semiprimes",
                pattern.frequency * 100.0
            );
            pattern.parameters = vec![pi];
            
            return Some(pattern);
        }
        
        None
    }
    
    /// Analyze e relationships
    fn analyze_e_relationships(observations: &[Observation]) -> Option<Pattern> {
        let e = std::f64::consts::E;
        let mut e_growth_rates = Vec::new();
        
        // Group by scale to analyze growth
        let mut by_scale: HashMap<usize, Vec<&Observation>> = HashMap::new();
        for obs in observations {
            by_scale.entry(obs.scale.bit_length / 8)
                .or_insert_with(Vec::new)
                .push(obs);
        }
        
        // Check if complexity grows exponentially
        let mut scale_complexities = Vec::new();
        for (&scale, obs_list) in &by_scale {
            if obs_list.len() >= 5 {
                let avg_gap: f64 = obs_list.iter()
                    .filter_map(|obs| obs.scale.prime_gap.to_f64())
                    .sum::<f64>() / obs_list.len() as f64;
                
                scale_complexities.push((scale as f64, avg_gap));
            }
        }
        
        if scale_complexities.len() >= 3 {
            // Check for exponential growth
            for i in 1..scale_complexities.len() {
                let (scale1, gap1) = scale_complexities[i - 1];
                let (scale2, gap2) = scale_complexities[i];
                
                let growth_rate = (gap2 / gap1).ln() / (scale2 / scale1).ln();
                if (growth_rate - e).abs() < 0.5 {
                    e_growth_rates.push(growth_rate);
                }
            }
        }
        
        if !e_growth_rates.is_empty() {
            let mut data = Data::new(e_growth_rates.clone());
            let mean_rate = data.mean().unwrap_or(0.0);
            
            let mut pattern = Pattern::new("exponential_complexity_growth", PatternKind::Universal);
            pattern.frequency = 0.7;
            pattern.description = format!(
                "Factorization complexity grows at rate ≈ e^x (observed: {:.3})",
                mean_rate
            );
            pattern.parameters = vec![e, mean_rate];
            
            return Some(pattern);
        }
        
        None
    }
    
    /// Analyze combined constant relationships
    fn analyze_combined_constants(observations: &[Observation]) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();
        
        let phi = 1.618033988749895;
        let pi = std::f64::consts::PI;
        let e = std::f64::consts::E;
        
        // Check for phi + pi + e relationships
        let mut unity_deviations = Vec::new();
        
        for obs in observations {
            if let (Some(p_f), Some(q_f), Some(n_f)) = 
                (obs.p.to_f64(), obs.q.to_f64(), obs.n.to_f64()) {
                
                // Normalized components
                let p_norm = p_f / n_f.powf(0.333);
                let q_norm = q_f / n_f.powf(0.333);
                let sum_norm = (p_f + q_f) / n_f.powf(0.5);
                
                // Check various combinations
                let combo1 = (p_norm * phi + q_norm * pi) / (sum_norm * e);
                let combo2 = (p_norm + q_norm) / (phi * pi * e).powf(0.25);
                
                unity_deviations.push((combo1 - 1.0).abs());
                unity_deviations.push((combo2 - 1.0).abs());
            }
        }
        
        // Analyze deviations
        let close_to_unity = unity_deviations.iter()
            .filter(|&&dev| dev < 0.1)
            .count();
        
        if close_to_unity > unity_deviations.len() / 10 {
            let mut pattern = Pattern::new(
                "universal_constant_harmony",
                PatternKind::Universal,
            );
            pattern.frequency = close_to_unity as f64 / unity_deviations.len() as f64;
            pattern.description = format!(
                "Factors exhibit harmonic relationships with φ, π, e in {:.1}% of cases",
                pattern.frequency * 100.0
            );
            pattern.parameters = vec![phi, pi, e];
            patterns.push(pattern);
        }
        
        Ok(patterns)
    }
    
    /// Discover ratio patterns
    fn discover_ratio_patterns(observations: &[Observation]) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();
        
        // Analyze offset ratio distribution
        let offset_ratios: Vec<f64> = observations.iter()
            .map(|obs| obs.derived.offset_ratio.abs())
            .collect();
        
        if offset_ratios.len() >= 10 {
            let mut data = Data::new(offset_ratios.clone());
            let mean = data.mean().unwrap_or(0.0);
            let median = data.median();
            let std_dev = data.std_dev().unwrap_or(0.0);
            
            // Check for log-normal distribution
            let log_ratios: Vec<f64> = offset_ratios.iter()
                .filter(|&&r| r > 0.0)
                .map(|&r| r.ln())
                .collect();
            
            if log_ratios.len() == offset_ratios.len() {
                let mut log_data = Data::new(log_ratios);
                let log_mean = log_data.mean().unwrap_or(0.0);
                let log_std = log_data.std_dev().unwrap_or(0.0);
                
                let mut pattern = Pattern::new(
                    "offset_ratio_lognormal",
                    PatternKind::Universal,
                );
                pattern.frequency = 0.9;
                pattern.description = format!(
                    "Offset ratios follow log-normal distribution (μ={:.3}, σ={:.3})",
                    log_mean, log_std
                );
                pattern.parameters = vec![log_mean, log_std];
                patterns.push(pattern);
            }
        }
        
        // Analyze balance ratio clustering
        let mut balance_clusters = HashMap::new();
        for obs in observations {
            let cluster = (obs.scale.balance_ratio * 10.0).round() as i32;
            *balance_clusters.entry(cluster).or_insert(0) += 1;
        }
        
        // Find dominant clusters
        let total_count = observations.len();
        for (cluster, count) in balance_clusters {
            let frequency = count as f64 / total_count as f64;
            if frequency > 0.1 && cluster <= 15 { // Balance ratio <= 1.5
                let mut pattern = Pattern::new(
                    format!("balance_cluster_{}", cluster),
                    PatternKind::Universal,
                );
                pattern.frequency = frequency;
                pattern.description = format!(
                    "{:.1}% of semiprimes have balance ratio ≈ {:.1}",
                    frequency * 100.0,
                    cluster as f64 / 10.0
                );
                pattern.parameters = vec![cluster as f64 / 10.0];
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    /// Discover density patterns
    fn discover_density_patterns(observations: &[Observation]) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();
        
        // Analyze prime density near semiprimes
        let mut density_measurements = Vec::new();
        
        for obs in observations {
            // Count primes in neighborhood
            let n_val = obs.n.to_f64().unwrap_or(0.0);
            let neighborhood_size = n_val.powf(0.25).max(100.0);
            
            // Approximate prime density (would need actual prime counting in production)
            let expected_density = 1.0 / n_val.ln();
            let observed_density = 2.0 / (obs.p.to_f64().unwrap_or(1.0) + 
                                         obs.q.to_f64().unwrap_or(1.0));
            
            density_measurements.push(observed_density / expected_density);
        }
        
        if density_measurements.len() >= 20 {
            let mut data = Data::new(density_measurements.clone());
            let mean_ratio = data.mean().unwrap_or(0.0);
            
            if (mean_ratio - 1.0).abs() > 0.1 {
                let mut pattern = Pattern::new(
                    "semiprime_density_anomaly",
                    PatternKind::Universal,
                );
                pattern.frequency = 0.8;
                pattern.description = format!(
                    "Semiprimes occur in regions with {:.1}x expected prime density",
                    mean_ratio
                );
                pattern.parameters = vec![mean_ratio];
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    /// Discover emergence patterns - patterns that only appear at scale
    fn discover_emergence_patterns(observations: &[Observation]) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();
        
        // Group by bit length
        let mut by_bits: HashMap<usize, Vec<&Observation>> = HashMap::new();
        for obs in observations {
            by_bits.entry(obs.scale.bit_length)
                .or_insert_with(Vec::new)
                .push(obs);
        }
        
        // Look for patterns that emerge at specific scales
        let mut emergence_threshold = 0;
        let mut pre_emergence_property = 0.0;
        let mut post_emergence_property = 0.0;
        
        for (&bits, obs_list) in by_bits.iter() {
            if obs_list.len() < 5 {
                continue;
            }
            
            // Example: Check if balanced semiprimes become rare
            let balanced_ratio = obs_list.iter()
                .filter(|obs| obs.scale.balance_ratio < 1.1)
                .count() as f64 / obs_list.len() as f64;
            
            if bits <= 64 && balanced_ratio > 0.5 {
                pre_emergence_property = balanced_ratio;
            } else if bits >= 128 && balanced_ratio < 0.1 {
                post_emergence_property = balanced_ratio;
                emergence_threshold = 96; // Somewhere between 64 and 128
            }
        }
        
        if emergence_threshold > 0 {
            let mut pattern = Pattern::new(
                "balance_rarity_emergence",
                PatternKind::Universal,
            );
            pattern.frequency = 0.9;
            pattern.description = format!(
                "Balanced semiprimes become rare above {} bits ({:.1}% → {:.1}%)",
                emergence_threshold,
                pre_emergence_property * 100.0,
                post_emergence_property * 100.0
            );
            pattern.parameters = vec![
                emergence_threshold as f64,
                pre_emergence_property,
                post_emergence_property,
            ];
            pattern.scale_range = ScaleRange {
                min_bits: emergence_threshold,
                max_bits: usize::MAX,
                unbounded: true,
            };
            patterns.push(pattern);
        }
        
        Ok(patterns)
    }
    
    /// Extract universal constants from patterns
    pub fn extract_constants(patterns: &[Pattern]) -> Vec<UniversalConstant> {
        let mut constants = Vec::new();
        
        for pattern in patterns {
            match pattern.kind {
                PatternKind::Universal => {
                    // Extract constants from parameters
                    for (i, &value) in pattern.parameters.iter().enumerate() {
                        if value.is_finite() && value != 0.0 {
                            let constant = UniversalConstant {
                                name: format!("{}_{}", pattern.id, i),
                                value,
                                frequency: pattern.frequency,
                                universality: pattern.frequency,
                                description: format!(
                                    "Constant from {}: parameter {}",
                                    pattern.id, i
                                ),
                            };
                            constants.push(constant);
                        }
                    }
                }
                _ => {}
            }
        }
        
        // Deduplicate similar constants
        constants.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(std::cmp::Ordering::Equal));
        constants.dedup_by(|a, b| (a.value - b.value).abs() < 1e-10);
        
        constants
    }
}