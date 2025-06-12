//! Correlation analysis between patterns

use crate::relationship_discovery::PatternCorrelation;
use crate::types::{Observation, Pattern};
use crate::Result;
use statrs::statistics::{Data, Distribution};
use std::collections::HashMap;

/// Analyze correlations between patterns
#[derive(Debug)]
pub struct CorrelationAnalysis;

impl CorrelationAnalysis {
    /// Analyze all pattern correlations
    pub fn analyze(
        observations: &[Observation],
        patterns: &[Pattern],
    ) -> Result<Vec<PatternCorrelation>> {
        let mut correlations = Vec::new();

        // Extract pattern occurrences for each observation
        let pattern_matrix = Self::build_pattern_matrix(observations, patterns)?;

        // Compute pairwise correlations
        for i in 0..patterns.len() {
            for j in i + 1..patterns.len() {
                if let Some(correlation) = Self::compute_correlation(
                    &pattern_matrix,
                    i,
                    j,
                    &patterns[i].id,
                    &patterns[j].id,
                ) {
                    correlations.push(correlation);
                }
            }
        }

        // Sort by absolute correlation strength
        correlations.sort_by(|a, b| {
            b.correlation
                .abs()
                .partial_cmp(&a.correlation.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(correlations)
    }

    /// Build pattern occurrence matrix
    fn build_pattern_matrix(
        observations: &[Observation],
        patterns: &[Pattern],
    ) -> Result<Vec<Vec<f64>>> {
        let mut matrix = vec![vec![0.0; patterns.len()]; observations.len()];

        for (obs_idx, obs) in observations.iter().enumerate() {
            for (pat_idx, pattern) in patterns.iter().enumerate() {
                // Check if pattern applies to this observation
                if pattern.applies_to(&obs.n) {
                    // Use pattern frequency as strength
                    matrix[obs_idx][pat_idx] = pattern.frequency;

                    // Additional checks based on pattern type
                    match pattern.id.as_str() {
                        id if id.contains("balanced") => {
                            if obs.scale.balance_ratio < 1.1 {
                                matrix[obs_idx][pat_idx] *= 2.0;
                            }
                        },
                        id if id.contains("harmonic") => {
                            // Check for harmonic relationships
                            let ratio =
                                obs.q.to_f64().unwrap_or(1.0) / obs.p.to_f64().unwrap_or(1.0);
                            if ratio % 1.0 < 0.1 || ratio % 1.0 > 0.9 {
                                matrix[obs_idx][pat_idx] *= 1.5;
                            }
                        },
                        _ => {},
                    }
                }
            }
        }

        Ok(matrix)
    }

    /// Compute correlation between two patterns
    fn compute_correlation(
        matrix: &[Vec<f64>],
        idx_a: usize,
        idx_b: usize,
        name_a: &str,
        name_b: &str,
    ) -> Option<PatternCorrelation> {
        let col_a: Vec<f64> = matrix.iter().map(|row| row[idx_a]).collect();
        let col_b: Vec<f64> = matrix.iter().map(|row| row[idx_b]).collect();

        // Need variation in both columns
        let var_a = Data::new(col_a.clone()).variance().unwrap_or(0.0);
        let var_b = Data::new(col_b.clone()).variance().unwrap_or(0.0);

        if var_a < 1e-10 || var_b < 1e-10 {
            return None;
        }

        // Compute Pearson correlation
        let n = col_a.len() as f64;
        let sum_a: f64 = col_a.iter().sum();
        let sum_b: f64 = col_b.iter().sum();
        let sum_ab: f64 = col_a.iter().zip(&col_b).map(|(a, b)| a * b).sum();
        let sum_a2: f64 = col_a.iter().map(|a| a * a).sum();
        let sum_b2: f64 = col_b.iter().map(|b| b * b).sum();

        let numerator = n * sum_ab - sum_a * sum_b;
        let denominator = ((n * sum_a2 - sum_a * sum_a) * (n * sum_b2 - sum_b * sum_b)).sqrt();

        if denominator < 1e-10 {
            return None;
        }

        let correlation = numerator / denominator;

        // Compute p-value (simplified t-test)
        let t_stat = correlation * ((n - 2.0) / (1.0 - correlation * correlation)).sqrt();
        let p_value = Self::approximate_p_value(t_stat.abs(), n as usize - 2);

        Some(PatternCorrelation {
            pattern_a: name_a.to_string(),
            pattern_b: name_b.to_string(),
            correlation,
            p_value,
            sample_size: col_a.len(),
        })
    }

    /// Approximate p-value for correlation test
    fn approximate_p_value(t_stat: f64, df: usize) -> f64 {
        // Simplified p-value approximation
        // In production, use proper t-distribution
        let z = t_stat / (1.0 + t_stat * t_stat / df as f64).sqrt();

        // Approximate normal CDF
        let p = 0.5 * (1.0 + Self::erf(z / std::f64::consts::SQRT_2));
        2.0 * (1.0 - p) // Two-tailed test
    }

    /// Error function approximation
    fn erf(x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Find transitive relationships
    pub fn find_transitive_relationships(
        correlations: &[PatternCorrelation],
        threshold: f64,
    ) -> Vec<TransitiveRelationship> {
        let mut transitive = Vec::new();

        // Build correlation map
        let mut corr_map: HashMap<(&str, &str), f64> = HashMap::new();
        for corr in correlations {
            corr_map.insert((&corr.pattern_a, &corr.pattern_b), corr.correlation);
            corr_map.insert((&corr.pattern_b, &corr.pattern_a), corr.correlation);
        }

        // Find A→B→C relationships
        for corr_ab in correlations {
            if corr_ab.correlation.abs() < threshold {
                continue;
            }

            for corr_bc in correlations {
                if corr_bc.pattern_a != corr_ab.pattern_b {
                    continue;
                }

                if corr_bc.correlation.abs() < threshold {
                    continue;
                }

                // Check if A→C exists
                let a = &corr_ab.pattern_a;
                let c = &corr_bc.pattern_b;

                if let Some(&corr_ac) = corr_map.get(&(a.as_str(), c.as_str())) {
                    let predicted = corr_ab.correlation * corr_bc.correlation;
                    let error = (corr_ac - predicted).abs();

                    if error < 0.2 {
                        transitive.push(TransitiveRelationship {
                            pattern_a: a.clone(),
                            pattern_b: corr_ab.pattern_b.clone(),
                            pattern_c: c.clone(),
                            correlation_ab: corr_ab.correlation,
                            correlation_bc: corr_bc.correlation,
                            correlation_ac: corr_ac,
                            predicted_ac: predicted,
                            error,
                        });
                    }
                }
            }
        }

        transitive
    }
}

/// Transitive relationship between three patterns
#[derive(Debug, Clone)]
pub struct TransitiveRelationship {
    /// First pattern identifier
    pub pattern_a: String,
    /// Second pattern identifier
    pub pattern_b: String,
    /// Third pattern identifier
    pub pattern_c: String,
    /// Correlation between pattern A and B
    pub correlation_ab: f64,
    /// Correlation between pattern B and C
    pub correlation_bc: f64,
    /// Actual correlation between pattern A and C
    pub correlation_ac: f64,
    /// Predicted correlation between pattern A and C
    pub predicted_ac: f64,
    /// Error between predicted and actual correlation
    pub error: f64,
}
