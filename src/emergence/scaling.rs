//! Scaling analysis - How patterns transform with scale
//!
//! This module discovers how patterns behave at different scales.

use crate::types::{Observation, Pattern, PatternKind};
use crate::types::pattern::ScaleRange;
use crate::Result;
use statrs::statistics::{Data, OrderStatistics, Distribution, Max};
use std::collections::HashMap;

/// Analysis of pattern scaling behavior
pub struct ScalingAnalysis;

impl ScalingAnalysis {
    /// Analyze patterns at all scales
    pub fn analyze_all(observations: &[Observation]) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();

        // Group observations by scale
        let by_scale = Self::group_by_scale(observations);

        // Analyze offset scaling
        patterns.extend(Self::analyze_offset_scaling(&by_scale)?);

        // Analyze balance ratio scaling
        patterns.extend(Self::analyze_balance_scaling(&by_scale)?);

        // Analyze modular pattern scaling
        patterns.extend(Self::analyze_modular_scaling(&by_scale)?);

        // Analyze computational complexity scaling
        patterns.extend(Self::analyze_complexity_scaling(&by_scale)?);

        Ok(patterns)
    }

    /// Group observations by bit length scale
    fn group_by_scale(observations: &[Observation]) -> HashMap<usize, Vec<&Observation>> {
        let mut by_scale = HashMap::new();

        for obs in observations {
            let scale_group = obs.scale.bit_length / 8; // Group by bytes
            by_scale.entry(scale_group).or_insert_with(Vec::new).push(obs);
        }

        by_scale
    }

    /// Analyze how offset ratios scale
    fn analyze_offset_scaling(
        by_scale: &HashMap<usize, Vec<&Observation>>,
    ) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();

        // Collect offset statistics at each scale
        let mut scale_stats = Vec::new();

        for (&scale, observations) in by_scale {
            if observations.len() < 5 {
                continue; // Need enough data
            }

            let offset_ratios: Vec<f64> =
                observations.iter().map(|obs| obs.derived.offset_ratio.abs()).collect();

            let mut data = Data::new(offset_ratios);
            let mean = data.mean().unwrap_or(0.0);
            let std_dev = data.std_dev().unwrap_or(0.0);
            let median = data.median();

            scale_stats.push((scale, mean, std_dev, median));
        }

        // Analyze scaling trend
        if scale_stats.len() >= 3 {
            // Check if offset ratio decreases with scale
            let decreasing = scale_stats.windows(2).all(|w| w[1].1 <= w[0].1);

            if decreasing {
                let mut pattern =
                    Pattern::new("offset_ratio_decreasing", PatternKind::ScaleDependent);
                pattern.frequency = 1.0;
                pattern.description = "Offset ratio decreases with number size".to_string();
                pattern.parameters = scale_stats.iter().map(|(_, mean, _, _)| *mean).collect();
                patterns.push(pattern);
            }

            // Check for logarithmic scaling
            let log_scales: Vec<f64> =
                scale_stats.iter().map(|(scale, _, _, _)| (*scale as f64 * 8.0).ln()).collect();
            let means: Vec<f64> = scale_stats.iter().map(|(_, mean, _, _)| *mean).collect();

            if let Some(slope) = Self::linear_regression(&log_scales, &means) {
                if slope.abs() > 0.01 {
                    let mut pattern =
                        Pattern::new("offset_logarithmic_scaling", PatternKind::ScaleDependent);
                    pattern.frequency = 0.9;
                    pattern.description =
                        format!("Offset ratio scales as O(log n) with slope {:.4}", slope);
                    pattern.parameters = vec![slope];
                    patterns.push(pattern);
                }
            }
        }

        Ok(patterns)
    }

    /// Analyze how balance ratios scale
    fn analyze_balance_scaling(
        by_scale: &HashMap<usize, Vec<&Observation>>,
    ) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();

        for (&scale, observations) in by_scale {
            let balanced_count =
                observations.iter().filter(|obs| obs.scale.balance_ratio < 1.1).count();

            let balanced_ratio = balanced_count as f64 / observations.len() as f64;

            if balanced_ratio > 0.8 {
                let mut pattern = Pattern::new(
                    format!("high_balance_at_{}_bytes", scale),
                    PatternKind::TypeSpecific("balanced".to_string()),
                );
                pattern.frequency = balanced_ratio;
                pattern.description = format!(
                    "{:.1}% of {}-byte numbers are balanced semiprimes",
                    balanced_ratio * 100.0,
                    scale
                );
                pattern.scale_range = ScaleRange {
                    min_bits: scale * 8 - 4,
                    max_bits: scale * 8 + 4,
                    unbounded: false,
                };
                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    /// Analyze how modular patterns scale
    fn analyze_modular_scaling(
        by_scale: &HashMap<usize, Vec<&Observation>>,
    ) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();

        // Check if certain residues become more common at larger scales
        let prime_count = 10; // Check first 10 primes

        for prime_idx in 0..prime_count {
            let mut scale_residue_freq = Vec::new();

            for (&scale, observations) in by_scale {
                let mut residue_counts = HashMap::new();

                for obs in observations {
                    if prime_idx < obs.modular.modular_signature.len() {
                        let residue = obs.modular.modular_signature[prime_idx];
                        *residue_counts.entry(residue).or_insert(0) += 1;
                    }
                }

                // Find most common residue
                if let Some((&residue, &count)) = residue_counts.iter().max_by_key(|(_, &c)| c) {
                    let frequency = count as f64 / observations.len() as f64;
                    scale_residue_freq.push((scale, residue, frequency));
                }
            }

            // Check if frequency increases with scale
            if scale_residue_freq.len() >= 3 {
                let increasing = scale_residue_freq.windows(2).all(|w| w[1].2 >= w[0].2);

                if increasing && scale_residue_freq.last().unwrap().2 > 0.3 {
                    let mut pattern = Pattern::new(
                        format!("modular_preference_scaling_p{}", prime_idx + 1),
                        PatternKind::ScaleDependent,
                    );
                    pattern.frequency = scale_residue_freq.last().unwrap().2;
                    pattern.description = format!(
                        "Preference for certain residues mod prime {} increases with scale",
                        prime_idx + 1
                    );
                    patterns.push(pattern);
                }
            }
        }

        Ok(patterns)
    }

    /// Analyze computational complexity scaling
    fn analyze_complexity_scaling(
        by_scale: &HashMap<usize, Vec<&Observation>>,
    ) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();

        // Analyze how the gap between factors scales
        let mut gap_scaling = Vec::new();

        for (&scale, observations) in by_scale {
            let gaps: Vec<f64> = observations
                .iter()
                .map(|obs| obs.scale.prime_gap.to_f64().unwrap_or(0.0))
                .collect();

            if !gaps.is_empty() {
                let mut data = Data::new(gaps);
                let mean_gap = data.mean().unwrap_or(0.0);
                let max_gap = data.max();

                gap_scaling.push((scale, mean_gap, max_gap));
            }
        }

        // Check gap growth pattern
        if gap_scaling.len() >= 3 {
            let scales: Vec<f64> = gap_scaling.iter().map(|(s, _, _)| *s as f64).collect();
            let mean_gaps: Vec<f64> = gap_scaling.iter().map(|(_, m, _)| *m).collect();

            if let Some(slope) = Self::linear_regression(&scales, &mean_gaps) {
                let mut pattern = Pattern::new("factor_gap_scaling", PatternKind::ScaleDependent);
                pattern.frequency = 0.8;
                pattern.description = format!("Factor gap grows with scale, slope = {:.2e}", slope);
                pattern.parameters = vec![slope];
                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    /// Simple linear regression
    fn linear_regression(x: &[f64], y: &[f64]) -> Option<f64> {
        if x.len() != y.len() || x.len() < 2 {
            return None;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y).map(|(xi, yi)| xi * yi).sum();
        let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return None;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        Some(slope)
    }

    /// Get scaling behavior description
    pub fn describe_scaling(pattern: &Pattern) -> String {
        match pattern.kind {
            PatternKind::ScaleDependent => {
                format!(
                    "{}: {} (parameters: {:?})",
                    pattern.id, pattern.description, pattern.parameters
                )
            },
            _ => pattern.description.clone(),
        }
    }
}
