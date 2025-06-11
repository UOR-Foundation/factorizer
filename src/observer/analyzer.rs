//! Pattern analysis from observations
//!
//! This module discovers patterns in empirical data without presumption.

use crate::error::PatternError;
use crate::types::{Observation, Pattern, PatternKind};
use crate::types::pattern::ScaleRange;
use crate::Result;
use statrs::statistics::{Data, OrderStatistics, Distribution};
use std::collections::HashMap;

/// Analyzer for discovering patterns in observations
pub struct Analyzer;

impl Analyzer {
    /// Find patterns in observations
    pub fn find_patterns(observations: &[Observation]) -> Result<Vec<Pattern>> {
        if observations.is_empty() {
            return Err(PatternError::InsufficientData(
                "No observations to analyze".to_string(),
            ));
        }

        let mut patterns = Vec::new();

        // Analyze offset ratios
        if let Some(pattern) = Self::analyze_offset_ratios(observations)? {
            patterns.push(pattern);
        }

        // Analyze modular patterns
        patterns.extend(Self::analyze_modular_patterns(observations)?);

        // Analyze scale-dependent patterns
        patterns.extend(Self::analyze_scale_patterns(observations)?);

        // Analyze balance ratios
        if let Some(pattern) = Self::analyze_balance_patterns(observations)? {
            patterns.push(pattern);
        }

        Ok(patterns)
    }

    /// Analyze offset ratio patterns
    fn analyze_offset_ratios(observations: &[Observation]) -> Result<Option<Pattern>> {
        let offset_ratios: Vec<f64> =
            observations.iter().map(|obs| obs.derived.offset_ratio.abs()).collect();

        if offset_ratios.is_empty() {
            return Ok(None);
        }

        let mut data = Data::new(offset_ratios.clone());
        let mean = data.mean().unwrap_or(0.0);
        let std_dev = data.std_dev().unwrap_or(0.0);
        let median = data.median();

        // Check if offset ratios follow a pattern
        let mut pattern = Pattern::new("offset_ratio_distribution", PatternKind::ScaleDependent);

        pattern.frequency = 1.0; // Always present
        pattern.parameters = vec![mean, std_dev, median];
        pattern.description = format!(
            "Offset ratios have mean {:.6}, std dev {:.6}, median {:.6}",
            mean, std_dev, median
        );

        // Determine scale range
        let min_bits = observations.iter().map(|o| o.scale.bit_length).min().unwrap_or(0);
        let max_bits = observations.iter().map(|o| o.scale.bit_length).max().unwrap_or(0);

        pattern.scale_range = ScaleRange {
            min_bits,
            max_bits,
            unbounded: true, // Assume pattern continues
        };

        Ok(Some(pattern))
    }

    /// Analyze modular patterns
    fn analyze_modular_patterns(observations: &[Observation]) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();

        // Analyze each prime modulus
        let num_primes = observations[0].modular.modular_signature.len();

        for prime_idx in 0..num_primes.min(10) {
            let mut residue_counts: HashMap<u64, usize> = HashMap::new();

            for obs in observations {
                if prime_idx < obs.modular.modular_signature.len() {
                    let residue = obs.modular.modular_signature[prime_idx];
                    *residue_counts.entry(residue).or_insert(0) += 1;
                }
            }

            // Check if certain residues are more common
            let total = observations.len() as f64;
            for (residue, count) in residue_counts {
                let frequency = count as f64 / total;

                if frequency > 0.15 {
                    // Significant frequency
                    let mut pattern = Pattern::new(
                        format!("modular_preference_p{}_{}", prime_idx + 1, residue),
                        PatternKind::Probabilistic,
                    );

                    pattern.frequency = frequency;
                    pattern.parameters = vec![prime_idx as f64, residue as f64];
                    pattern.description = format!(
                        "Numbers prefer residue {} mod prime #{} ({:.1}% frequency)",
                        residue,
                        prime_idx + 1,
                        frequency * 100.0
                    );

                    patterns.push(pattern);
                }
            }
        }

        Ok(patterns)
    }

    /// Analyze patterns at different scales
    fn analyze_scale_patterns(observations: &[Observation]) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();

        // Group by bit length
        let mut by_scale: HashMap<usize, Vec<&Observation>> = HashMap::new();
        for obs in observations {
            by_scale.entry(obs.scale.bit_length).or_insert_with(Vec::new).push(obs);
        }

        // Analyze each scale
        for (bit_length, obs_at_scale) in by_scale {
            if obs_at_scale.len() < 10 {
                continue; // Need enough data
            }

            // Check offset ratio behavior at this scale
            let offset_ratios: Vec<f64> =
                obs_at_scale.iter().map(|o| o.derived.offset_ratio.abs()).collect();

            let mut data = Data::new(offset_ratios);
            let mean = data.mean().unwrap_or(0.0);

            let mut pattern = Pattern::new(
                format!("scale_{}_bit_offset", bit_length),
                PatternKind::ScaleDependent,
            );

            pattern.frequency = obs_at_scale.len() as f64 / observations.len() as f64;
            pattern.parameters = vec![bit_length as f64, mean];
            pattern.scale_range = ScaleRange {
                min_bits: bit_length,
                max_bits: bit_length,
                unbounded: false,
            };
            pattern.description = format!(
                "At {} bits, average offset ratio is {:.6}",
                bit_length, mean
            );

            patterns.push(pattern);
        }

        Ok(patterns)
    }

    /// Analyze balance ratio patterns
    fn analyze_balance_patterns(observations: &[Observation]) -> Result<Option<Pattern>> {
        let balance_ratios: Vec<f64> =
            observations.iter().map(|obs| obs.scale.balance_ratio).collect();

        // Group into categories
        let balanced = balance_ratios.iter().filter(|&&r| r < 1.1).count();
        let moderate = balance_ratios.iter().filter(|&&r| r >= 1.1 && r < 10.0).count();
        let harmonic = balance_ratios.iter().filter(|&&r| r >= 10.0).count();

        let total = observations.len() as f64;

        let mut pattern = Pattern::new("balance_distribution", PatternKind::Probabilistic);

        pattern.frequency = 1.0;
        pattern.parameters = vec![
            balanced as f64 / total,
            moderate as f64 / total,
            harmonic as f64 / total,
        ];
        pattern.description = format!(
            "Balance distribution: {:.1}% balanced, {:.1}% moderate, {:.1}% harmonic",
            pattern.parameters[0] * 100.0,
            pattern.parameters[1] * 100.0,
            pattern.parameters[2] * 100.0,
        );

        Ok(Some(pattern))
    }

    /// Find invariant relationships
    pub fn find_invariants(observations: &[Observation]) -> Result<Vec<Pattern>> {
        let mut invariants = Vec::new();

        // Check Fermat relationship: n = a² - b²
        let mut fermat_valid = true;
        for obs in observations {
            let a_squared = &obs.derived.fermat_a * &obs.derived.fermat_a;
            let b_squared = &obs.derived.fermat_b * &obs.derived.fermat_b;
            let computed_n = &a_squared - &b_squared;

            if computed_n != obs.n {
                fermat_valid = false;
                break;
            }
        }

        if fermat_valid {
            let mut pattern = Pattern::new("fermat_identity", PatternKind::Invariant);
            pattern.frequency = 1.0;
            pattern.description = "n = a² - b² where a = (p+q)/2, b = |p-q|/2".to_string();
            invariants.push(pattern);
        }

        // Check p*q = n
        let mut product_valid = true;
        for obs in observations {
            if &obs.p * &obs.q != obs.n {
                product_valid = false;
                break;
            }
        }

        if product_valid {
            let mut pattern = Pattern::new("fundamental_factorization", PatternKind::Invariant);
            pattern.frequency = 1.0;
            pattern.description = "n = p × q always holds".to_string();
            invariants.push(pattern);
        }

        Ok(invariants)
    }

    /// Validate that a pattern is invariant
    pub fn validate_invariant(pattern: &Pattern, observations: &[Observation]) -> bool {
        // For now, check frequency
        pattern.frequency >= 0.9999
    }

    /// Analyze how patterns scale
    pub fn analyze_scaling(pattern: &Pattern, observations: &[Observation]) -> ScalingBehavior {
        // Group observations by scale
        let mut by_scale: HashMap<usize, Vec<&Observation>> = HashMap::new();
        for obs in observations {
            by_scale
                .entry(obs.scale.bit_length / 8) // Group by bytes
                .or_insert_with(Vec::new)
                .push(obs);
        }

        // Simple scaling analysis
        let scales: Vec<usize> = by_scale.keys().copied().collect();
        let min_scale = scales.iter().min().copied().unwrap_or(0);
        let max_scale = scales.iter().max().copied().unwrap_or(0);

        ScalingBehavior {
            pattern_id: pattern.id.clone(),
            behavior_type: if pattern.kind == PatternKind::Invariant {
                "constant".to_string()
            } else {
                "variable".to_string()
            },
            scale_range: (min_scale * 8, max_scale * 8),
        }
    }
}

/// Describes how a pattern behaves across scales
#[derive(Debug)]
pub struct ScalingBehavior {
    /// Pattern identifier
    pub pattern_id: String,

    /// Type of scaling behavior
    pub behavior_type: String,

    /// Scale range observed
    pub scale_range: (usize, usize),
}

impl ScalingBehavior {
    /// Describe the scaling behavior
    pub fn describe(&self) -> String {
        format!(
            "{} scaling from {}-{} bits",
            self.behavior_type, self.scale_range.0, self.scale_range.1
        )
    }
}
