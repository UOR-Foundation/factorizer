//! Enhanced execution with adaptive quantum search
//!
//! This module implements advanced factor decoding using enhanced quantum neighborhoods
//! with adaptive probability distributions.

use crate::error::PatternError;
use crate::pattern::verification::verify_factors;
use crate::types::quantum_enhanced::{DistributionType, EnhancedQuantumRegion};
use crate::types::recognition::{Factors, Formalization};
use crate::types::{Number, Pattern, PatternKind};
use crate::utils;
use crate::Result;

/// Enhanced decoding using adaptive quantum search
pub fn enhanced_quantum_search(
    formalization: &Formalization,
    patterns: &[Pattern],
) -> Result<Factors> {
    let n = &formalization.n;

    // Create enhanced quantum region
    let mut quantum_region = create_quantum_region_from_formalization(formalization, patterns)?;

    // Get initial search candidates
    let candidates = quantum_region.get_search_candidates(20);

    for candidate in candidates {
        // Check if it's a factor
        if candidate > Number::from(1u32) && &candidate < n && n % &candidate == Number::from(0u32)
        {
            let other = n / &candidate;

            // Found a factor, update quantum region
            quantum_region.update(&candidate, true, None);

            return Ok(Factors::new(candidate, other, "enhanced_quantum_search"));
        } else {
            // Not a factor, update region
            quantum_region.update(&candidate, false, None);
        }
    }

    // If initial candidates failed, use adaptive search
    adaptive_quantum_search(formalization, &mut quantum_region, patterns)
}

/// Create quantum region from formalization
fn create_quantum_region_from_formalization(
    formalization: &Formalization,
    patterns: &[Pattern],
) -> Result<EnhancedQuantumRegion> {
    let n = &formalization.n;
    let sqrt_n = utils::integer_sqrt(n)?;

    // Determine center from resonance peaks
    let center = if !formalization.resonance_peaks.is_empty() {
        let peak_idx = formalization.resonance_peaks[0];
        let field_size = formalization.harmonic_series.len().max(100) as f64;
        let ratio = peak_idx as f64 / field_size;
        &sqrt_n * Number::from((1.0 + ratio * 0.1) as u64)
    } else {
        sqrt_n
    };

    // Create region with pattern analysis
    let mut region = EnhancedQuantumRegion::from_pattern_analysis(center, patterns, n);

    // Set distribution type based on harmonic analysis
    if has_multiple_harmonics(&formalization.harmonic_series) {
        region.distribution_type = DistributionType::MultiModal;
    }

    Ok(region)
}

/// Check if harmonic series indicates multiple modes
fn has_multiple_harmonics(harmonics: &[f64]) -> bool {
    if harmonics.len() < 3 {
        return false;
    }

    // Count significant harmonics
    let threshold = 0.3;
    let significant = harmonics.iter().filter(|&&h| h.abs() > threshold).count();

    significant > 2
}

/// Adaptive quantum search with learning
fn adaptive_quantum_search(
    formalization: &Formalization,
    quantum_region: &mut EnhancedQuantumRegion,
    patterns: &[Pattern],
) -> Result<Factors> {
    let n = &formalization.n;
    let max_iterations = 100;

    for iteration in 0..max_iterations {
        // Adjust search strategy based on confidence
        let num_candidates = if quantum_region.confidence_metrics.overall > 0.7 {
            5 // High confidence, focus search
        } else if quantum_region.confidence_metrics.overall > 0.4 {
            10 // Medium confidence
        } else {
            20 // Low confidence, explore more
        };

        // Get next batch of candidates
        let candidates = quantum_region.get_search_candidates(num_candidates);

        for candidate in candidates {
            if candidate > Number::from(1u32) && &candidate < n {
                if n % &candidate == Number::from(0u32) {
                    let other = n / &candidate;

                    // Found factor! Update region with success
                    let matching_pattern = patterns.iter().find(|p| p.applies_to(n));
                    quantum_region.update(&candidate, true, matching_pattern);

                    // Verify and return
                    let factors =
                        Factors::new(candidate.clone(), other.clone(), "adaptive_quantum_search");
                    if let Ok(verification) =
                        verify_factors(&factors, n, patterns, "adaptive_quantum_search")
                    {
                        if verification.is_valid {
                            return Ok(factors);
                        }
                    }
                } else {
                    // Update region with failure
                    quantum_region.update(&candidate, false, None);
                }
            }
        }

        // Check if we should change strategy
        if iteration % 10 == 9 {
            adjust_search_strategy(quantum_region, iteration);
        }
    }

    Err(PatternError::ExecutionError(
        "Adaptive quantum search exhausted".to_string(),
    ))
}

/// Adjust search strategy based on performance
fn adjust_search_strategy(quantum_region: &mut EnhancedQuantumRegion, iteration: usize) {
    let confidence = quantum_region.confidence_metrics.overall;

    if confidence < 0.3 && iteration > 20 {
        // Low confidence after many iterations - try different distribution
        match quantum_region.distribution_type {
            DistributionType::Gaussian => {
                quantum_region.distribution_type = DistributionType::MultiModal;
            },
            DistributionType::MultiModal => {
                quantum_region.distribution_type = DistributionType::Empirical;
            },
            _ => {},
        }
    }

    // Adjust radius based on miss rate
    if quantum_region.observations.total_observations > 0 {
        let success_rate = quantum_region.observations.successes.len() as f64
            / quantum_region.observations.total_observations as f64;

        if success_rate < 0.01 {
            // Very low success rate - expand search
            quantum_region.radius.growth_rate = 2.0;
        }
    }
}

/// Multi-level quantum search
pub fn multi_level_quantum_search(
    formalization: &Formalization,
    patterns: &[Pattern],
) -> Result<Factors> {
    let n = &formalization.n;

    // Level 1: Coarse search with large regions
    let coarse_region = create_coarse_quantum_region(n, patterns)?;
    if let Ok(factors) = search_quantum_level(formalization, coarse_region, patterns, "coarse") {
        return Ok(factors);
    }

    // Level 2: Medium resolution
    let medium_regions = create_medium_quantum_regions(formalization, patterns)?;
    for region in medium_regions {
        if let Ok(factors) = search_quantum_level(formalization, region, patterns, "medium") {
            return Ok(factors);
        }
    }

    // Level 3: Fine-grained search
    let fine_regions = create_fine_quantum_regions(formalization, patterns)?;
    for region in fine_regions {
        if let Ok(factors) = search_quantum_level(formalization, region, patterns, "fine") {
            return Ok(factors);
        }
    }

    Err(PatternError::ExecutionError(
        "Multi-level quantum search failed".to_string(),
    ))
}

/// Create coarse quantum region for initial search
fn create_coarse_quantum_region(
    n: &Number,
    _patterns: &[Pattern],
) -> Result<EnhancedQuantumRegion> {
    let sqrt_n = utils::integer_sqrt(n)?;
    let radius = &sqrt_n / &Number::from(10u32);

    Ok(EnhancedQuantumRegion::new(sqrt_n, radius, n))
}

/// Create medium resolution quantum regions
fn create_medium_quantum_regions(
    formalization: &Formalization,
    _patterns: &[Pattern],
) -> Result<Vec<EnhancedQuantumRegion>> {
    let n = &formalization.n;
    let sqrt_n = utils::integer_sqrt(n)?;
    let mut regions = Vec::new();

    // Create regions around resonance peaks
    for &peak_idx in formalization.resonance_peaks.iter().take(3) {
        let field_size = formalization.harmonic_series.len().max(100) as f64;
        let ratio = peak_idx as f64 / field_size;
        let center = &sqrt_n * Number::from((1.0 + ratio * 0.2) as u64);
        let radius = &sqrt_n / &Number::from(50u32);

        let region = EnhancedQuantumRegion::new(center, radius, n);
        regions.push(region);
    }

    Ok(regions)
}

/// Create fine-grained quantum regions
fn create_fine_quantum_regions(
    formalization: &Formalization,
    patterns: &[Pattern],
) -> Result<Vec<EnhancedQuantumRegion>> {
    let n = &formalization.n;
    let mut regions = Vec::new();

    // Create regions based on pattern predictions
    for pattern in patterns.iter().filter(|p| p.applies_to(n)) {
        match &pattern.kind {
            PatternKind::Emergent => {
                let sqrt_n = utils::integer_sqrt(n)?;
                let center = sqrt_n.clone();
                let radius = Number::from(1000u32);
                regions.push(EnhancedQuantumRegion::new(center, radius, n));
            },
            PatternKind::Harmonic { base_frequency, .. } => {
                let sqrt_n = utils::integer_sqrt(n)?;
                let offset = (base_frequency * 1000.0) as u64;
                let center = &sqrt_n + &Number::from(offset);
                let radius = Number::from(offset / 10);

                let mut region = EnhancedQuantumRegion::new(center, radius, n);
                region.distribution_type = DistributionType::MultiModal;
                regions.push(region);
            },
            _ => {},
        }
    }

    Ok(regions)
}

/// Search within a quantum level
fn search_quantum_level(
    formalization: &Formalization,
    mut quantum_region: EnhancedQuantumRegion,
    _patterns: &[Pattern],
    level: &str,
) -> Result<Factors> {
    let n = &formalization.n;
    let max_attempts = match level {
        "coarse" => 50,
        "medium" => 100,
        "fine" => 200,
        _ => 100,
    };

    for _ in 0..max_attempts {
        let candidates = quantum_region.get_search_candidates(10);

        for candidate in &candidates {
            if candidate > &Number::from(1u32)
                && candidate < n
                && n % candidate == Number::from(0u32)
            {
                let other = n / candidate;
                let method = format!("quantum_search_{}", level);
                return Ok(Factors::new(candidate.clone(), other, method));
            }
        }

        // Update quantum region based on failures
        for candidate in &candidates {
            quantum_region.update(candidate, false, None);
        }
    }

    Err(PatternError::ExecutionError(format!(
        "Quantum search at {} level failed",
        level
    )))
}

/// Parallel quantum search across multiple regions
pub fn parallel_quantum_search(
    formalization: &Formalization,
    patterns: &[Pattern],
) -> Result<Factors> {
    use rayon::prelude::*;

    let n = &formalization.n;

    // Create multiple quantum regions
    let regions = create_parallel_quantum_regions(formalization, patterns)?;

    // Search in parallel
    let result = regions.into_par_iter().find_map_any(|region| {
        // Each thread searches its region
        for _ in 0..50 {
            let candidates = region.get_search_candidates(5);

            for candidate in candidates {
                if candidate > Number::from(1u32)
                    && &candidate < n
                    && n % &candidate == Number::from(0u32)
                {
                    let other = n / &candidate;
                    return Some(Factors::new(candidate, other, "parallel_quantum_search"));
                }
            }
        }
        None
    });

    result.ok_or_else(|| PatternError::ExecutionError("Parallel quantum search failed".to_string()))
}

/// Create regions for parallel search
fn create_parallel_quantum_regions(
    formalization: &Formalization,
    _patterns: &[Pattern],
) -> Result<Vec<EnhancedQuantumRegion>> {
    let n = &formalization.n;
    let sqrt_n = utils::integer_sqrt(n)?;
    let mut regions = Vec::new();

    // Create regions at different scales
    for i in 0..8 {
        let offset_factor = 1.0 + (i as f64 * 0.1);
        let center = &sqrt_n * Number::from(offset_factor as u64);
        let radius = &sqrt_n / &Number::from(20u32);

        let mut region = EnhancedQuantumRegion::new(center, radius, n);

        // Vary distribution types
        region.distribution_type = match i % 3 {
            0 => DistributionType::Gaussian,
            1 => DistributionType::MultiModal,
            _ => DistributionType::Skewed,
        };

        regions.push(region);
    }

    Ok(regions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiple_harmonics_detection() {
        let harmonics = vec![0.8, 0.4, 0.6, 0.2, 0.1];
        assert!(has_multiple_harmonics(&harmonics));

        let harmonics2 = vec![0.8, 0.1, 0.1];
        assert!(!has_multiple_harmonics(&harmonics2));
    }
}
