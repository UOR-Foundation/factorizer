//! Enhanced quantum neighborhood with advanced probability distributions
//!
//! This module implements sophisticated probability models for quantum regions
//! based on empirical observations of factor distributions.

use crate::types::{Number, Pattern, PatternKind};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Distribution type for quantum regions
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum DistributionType {
    /// Single-peaked Gaussian
    Gaussian,
    /// Multi-modal distribution
    MultiModal,
    /// Skewed distribution (for harmonic patterns)
    Skewed,
    /// Uniform within bounds
    Uniform,
    /// Custom empirical distribution
    Empirical,
}

/// Enhanced quantum region with adaptive distributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedQuantumRegion {
    /// Center of the region
    pub center: Number,

    /// Adaptive radius
    pub radius: AdaptiveRadius,

    /// Distribution type
    pub distribution_type: DistributionType,

    /// Multi-modal peaks if applicable
    pub modes: Vec<QuantumMode>,

    /// Statistical model parameters
    pub model_params: StatisticalModel,

    /// Observation history for learning
    pub observations: ObservationHistory,

    /// Confidence metrics
    pub confidence_metrics: ConfidenceMetrics,
}

/// Adaptive radius that adjusts based on observations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveRadius {
    /// Current radius
    pub current: Number,

    /// Minimum radius
    pub min: Number,

    /// Maximum radius
    pub max: Number,

    /// Growth rate
    pub growth_rate: f64,

    /// Shrink rate
    pub shrink_rate: f64,

    /// Number of consecutive misses before expansion
    pub miss_threshold: u32,

    /// Current miss count
    pub miss_count: u32,
}

/// A mode in a multi-modal distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMode {
    /// Mode location (offset from center)
    pub offset: i64,

    /// Mode weight
    pub weight: f64,

    /// Mode width (standard deviation)
    pub width: f64,

    /// Associated pattern if any
    pub pattern_id: Option<String>,
}

/// Statistical model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalModel {
    /// Mean of the distribution
    pub mean: f64,

    /// Variance
    pub variance: f64,

    /// Skewness
    pub skewness: f64,

    /// Kurtosis
    pub kurtosis: f64,

    /// Mixture model parameters
    pub mixture_params: Option<MixtureParams>,

    /// Empirical distribution if learned
    pub empirical_dist: Option<Vec<f64>>,
}

/// Parameters for Gaussian mixture models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixtureParams {
    /// Component means
    pub means: Vec<f64>,

    /// Component variances
    pub variances: Vec<f64>,

    /// Component weights
    pub weights: Vec<f64>,
}

/// Observation history for adaptive learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationHistory {
    /// Successful factor locations
    pub successes: Vec<i64>,

    /// Failed search locations
    pub failures: Vec<i64>,

    /// Pattern associations
    pub pattern_hits: HashMap<String, u32>,

    /// Total observations
    pub total_observations: u32,
}

/// Confidence metrics for the quantum region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceMetrics {
    /// Overall confidence (0.0 - 1.0)
    pub overall: f64,

    /// Distribution fit quality
    pub distribution_fit: f64,

    /// Prediction accuracy
    pub prediction_accuracy: f64,

    /// Coverage (fraction of factors found in region)
    pub coverage: f64,
}

impl EnhancedQuantumRegion {
    /// Create a new enhanced quantum region
    pub fn new(center: Number, initial_radius: Number, n: &Number) -> Self {
        let radius = AdaptiveRadius {
            current: initial_radius.clone(),
            min: Number::from(1u32),
            max: utils::integer_sqrt(n).unwrap_or_else(|_| n.clone()),
            growth_rate: 1.5,
            shrink_rate: 0.8,
            miss_threshold: 3,
            miss_count: 0,
        };

        EnhancedQuantumRegion {
            center,
            radius,
            distribution_type: DistributionType::Gaussian,
            modes: vec![QuantumMode {
                offset: 0,
                weight: 1.0,
                width: initial_radius.to_f64().unwrap_or(100.0) / 3.0,
                pattern_id: None,
            }],
            model_params: StatisticalModel {
                mean: 0.0,
                variance: 1.0,
                skewness: 0.0,
                kurtosis: 3.0, // Normal distribution
                mixture_params: None,
                empirical_dist: None,
            },
            observations: ObservationHistory {
                successes: Vec::new(),
                failures: Vec::new(),
                pattern_hits: HashMap::new(),
                total_observations: 0,
            },
            confidence_metrics: ConfidenceMetrics {
                overall: 0.5,
                distribution_fit: 0.5,
                prediction_accuracy: 0.5,
                coverage: 0.5,
            },
        }
    }

    /// Initialize from pattern analysis
    pub fn from_pattern_analysis(center: Number, patterns: &[Pattern], n: &Number) -> Self {
        let mut region = Self::new(center.clone(), Number::from(1000u32), n);

        // Analyze patterns to determine distribution type
        let pattern_types: Vec<_> = patterns.iter().filter(|p| p.applies_to(n)).collect();

        if pattern_types.is_empty() {
            return region;
        }

        // Check for multi-modal patterns
        let harmonic_patterns = pattern_types
            .iter()
            .filter(|p| matches!(p.kind, PatternKind::Harmonic { .. }))
            .count();

        if harmonic_patterns > 0 {
            region.distribution_type = DistributionType::MultiModal;
            region.initialize_multimodal_distribution(&pattern_types);
        } else if pattern_types.len() > 2 {
            region.distribution_type = DistributionType::Empirical;
        }

        region
    }

    /// Initialize multi-modal distribution from patterns
    fn initialize_multimodal_distribution(&mut self, patterns: &[&Pattern]) {
        self.modes.clear();

        for (i, pattern) in patterns.iter().enumerate() {
            let offset = match &pattern.kind {
                PatternKind::Harmonic { base_frequency, .. } => {
                    (base_frequency * 1000.0 * (i as f64 + 1.0)) as i64
                },
                _ => (i as i64 * 100) - 200, // Default spacing
            };

            self.modes.push(QuantumMode {
                offset,
                weight: pattern.frequency,
                width: self.radius.current.to_f64().unwrap_or(100.0) / (patterns.len() as f64),
                pattern_id: Some(pattern.id.clone()),
            });
        }

        // Normalize weights
        let total_weight: f64 = self.modes.iter().map(|m| m.weight).sum();
        if total_weight > 0.0 {
            for mode in &mut self.modes {
                mode.weight /= total_weight;
            }
        }
    }

    /// Get probability at a specific location
    pub fn probability_at(&self, location: &Number) -> f64 {
        if !self.contains(location) {
            return 0.0;
        }

        let offset = if location >= &self.center {
            (location - &self.center).as_integer().to_i64().unwrap_or(0)
        } else {
            -(&self.center - location).as_integer().to_i64().unwrap_or(0)
        };

        match self.distribution_type {
            DistributionType::Gaussian => self.gaussian_probability(offset),
            DistributionType::MultiModal => self.multimodal_probability(offset),
            DistributionType::Skewed => self.skewed_probability(offset),
            DistributionType::Uniform => self.uniform_probability(offset),
            DistributionType::Empirical => self.empirical_probability(offset),
        }
    }

    /// Gaussian probability
    fn gaussian_probability(&self, offset: i64) -> f64 {
        let mode = &self.modes[0];
        let diff = (offset - mode.offset) as f64;
        let exponent = -diff * diff / (2.0 * mode.width * mode.width);
        mode.weight * exponent.exp()
    }

    /// Multi-modal probability
    fn multimodal_probability(&self, offset: i64) -> f64 {
        self.modes
            .iter()
            .map(|mode| {
                let diff = (offset - mode.offset) as f64;
                let exponent = -diff * diff / (2.0 * mode.width * mode.width);
                mode.weight * exponent.exp()
            })
            .sum()
    }

    /// Skewed probability (for harmonic patterns)
    fn skewed_probability(&self, offset: i64) -> f64 {
        let x = offset as f64 / self.radius.current.to_f64().unwrap_or(100.0);
        let alpha = self.model_params.skewness;

        // Skew-normal distribution
        let phi = |t: f64| (1.0 / (2.0 * std::f64::consts::PI).sqrt()) * (-t * t / 2.0).exp();
        let capital_phi = |t: f64| 0.5 * (1.0 + libm::erf(t / std::f64::consts::SQRT_2));

        2.0 * phi(x) * capital_phi(alpha * x)
    }

    /// Uniform probability
    fn uniform_probability(&self, offset: i64) -> f64 {
        let radius = self.radius.current.as_integer().to_i64().unwrap_or(100);
        if offset.abs() <= radius {
            1.0 / (2.0 * radius as f64 + 1.0)
        } else {
            0.0
        }
    }

    /// Empirical probability from observations
    fn empirical_probability(&self, offset: i64) -> f64 {
        if let Some(dist) = &self.model_params.empirical_dist {
            let radius = self.radius.current.as_integer().to_i64().unwrap_or(100);
            let idx = (offset + radius) as usize;
            if idx < dist.len() {
                dist[idx]
            } else {
                0.0
            }
        } else {
            self.gaussian_probability(offset) // Fallback
        }
    }

    /// Check if location is within region
    pub fn contains(&self, location: &Number) -> bool {
        let distance = if location >= &self.center {
            location - &self.center
        } else {
            &self.center - location
        };

        distance <= self.radius.current
    }

    /// Update distribution based on observation
    pub fn update(&mut self, location: &Number, found_factor: bool, pattern: Option<&Pattern>) {
        if !self.contains(location) {
            self.radius.miss_count += 1;
            if self.radius.miss_count >= self.radius.miss_threshold {
                self.expand_radius();
            }
            return;
        }

        let offset = if location >= &self.center {
            (location - &self.center).as_integer().to_i64().unwrap_or(0)
        } else {
            -(&self.center - location).as_integer().to_i64().unwrap_or(0)
        };

        // Update observation history
        self.observations.total_observations += 1;
        if found_factor {
            self.observations.successes.push(offset);
            self.radius.miss_count = 0; // Reset miss count

            if let Some(p) = pattern {
                *self.observations.pattern_hits.entry(p.id.clone()).or_insert(0) += 1;
            }
        } else {
            self.observations.failures.push(offset);
        }

        // Update distribution parameters
        self.update_model_params();

        // Update confidence metrics
        self.update_confidence_metrics();

        // Consider switching distribution type if needed
        self.adapt_distribution_type();
    }

    /// Expand the search radius
    fn expand_radius(&mut self) {
        let current_val = self.radius.current.to_f64().unwrap_or(100.0);
        let new_val = current_val * self.radius.growth_rate;
        let new_radius = Number::from(new_val as u64);
        if new_radius <= self.radius.max {
            self.radius.current = new_radius;
            self.radius.miss_count = 0;
        }
    }

    /// Update statistical model parameters
    fn update_model_params(&mut self) {
        if self.observations.successes.is_empty() {
            return;
        }

        // Calculate mean
        let mean = self.observations.successes.iter().map(|&x| x as f64).sum::<f64>()
            / self.observations.successes.len() as f64;

        // Calculate variance
        let variance = self
            .observations
            .successes
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / self.observations.successes.len() as f64;

        self.model_params.mean = mean;
        self.model_params.variance = variance;

        // Update modes if multi-modal
        if self.distribution_type == DistributionType::MultiModal {
            self.update_modes_from_observations();
        }
    }

    /// Update modes based on observations
    fn update_modes_from_observations(&mut self) {
        // Use clustering to identify modes
        let clusters = self.cluster_observations(&self.observations.successes, 3);

        if clusters.len() > 1 {
            self.modes.clear();
            for (center, points) in clusters {
                let weight = points.len() as f64 / self.observations.successes.len() as f64;
                let width = if points.len() > 1 {
                    let cluster_var = points
                        .iter()
                        .map(|&x| {
                            let diff = x as f64 - center;
                            diff * diff
                        })
                        .sum::<f64>()
                        / points.len() as f64;
                    cluster_var.sqrt()
                } else {
                    10.0
                };

                self.modes.push(QuantumMode {
                    offset: center as i64,
                    weight,
                    width,
                    pattern_id: None,
                });
            }
        }
    }

    /// Simple k-means clustering
    fn cluster_observations(&self, points: &[i64], k: usize) -> Vec<(f64, Vec<i64>)> {
        if points.len() <= k {
            return points.iter().map(|&p| (p as f64, vec![p])).collect();
        }

        // Initialize centers
        let mut centers: Vec<f64> =
            points.iter().step_by(points.len() / k).take(k).map(|&p| p as f64).collect();

        let mut clusters: Vec<Vec<i64>> = vec![Vec::new(); k];

        // Simple k-means iterations
        for _ in 0..10 {
            // Clear clusters
            for cluster in &mut clusters {
                cluster.clear();
            }

            // Assign points to nearest center
            for &point in points {
                let mut min_dist = f64::INFINITY;
                let mut best_cluster = 0;

                for (i, &center) in centers.iter().enumerate() {
                    let dist = (point as f64 - center).abs();
                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = i;
                    }
                }

                clusters[best_cluster].push(point);
            }

            // Update centers
            for (i, cluster) in clusters.iter().enumerate() {
                if !cluster.is_empty() {
                    centers[i] =
                        cluster.iter().map(|&p| p as f64).sum::<f64>() / cluster.len() as f64;
                }
            }
        }

        centers
            .into_iter()
            .zip(clusters.into_iter())
            .filter(|(_, cluster)| !cluster.is_empty())
            .collect()
    }

    /// Update confidence metrics
    fn update_confidence_metrics(&mut self) {
        let total = self.observations.total_observations as f64;
        if total == 0.0 {
            return;
        }

        // Prediction accuracy
        let successes = self.observations.successes.len() as f64;
        self.confidence_metrics.prediction_accuracy = successes / total;

        // Coverage (how well the region covers successful locations)
        let within_one_sigma = self
            .observations
            .successes
            .iter()
            .filter(|&&offset| {
                let z_score =
                    (offset as f64 - self.model_params.mean) / self.model_params.variance.sqrt();
                z_score.abs() <= 1.0
            })
            .count() as f64;

        self.confidence_metrics.coverage =
            if successes > 0.0 { within_one_sigma / successes } else { 0.0 };

        // Distribution fit (simplified)
        self.confidence_metrics.distribution_fit = if self.model_params.variance > 0.0 {
            1.0 / (1.0
                + self.model_params.variance.sqrt() / self.radius.current.to_f64().unwrap_or(100.0))
        } else {
            0.5
        };

        // Overall confidence
        self.confidence_metrics.overall = (self.confidence_metrics.prediction_accuracy * 0.4
            + self.confidence_metrics.coverage * 0.3
            + self.confidence_metrics.distribution_fit * 0.3)
            .min(1.0);
    }

    /// Adapt distribution type based on observations
    fn adapt_distribution_type(&mut self) {
        if self.observations.successes.len() < 10 {
            return; // Not enough data
        }

        // Check for multi-modality
        let clusters = self.cluster_observations(&self.observations.successes, 3);
        if clusters.len() > 1 && self.distribution_type != DistributionType::MultiModal {
            self.distribution_type = DistributionType::MultiModal;
            self.update_modes_from_observations();
        }

        // Check for skewness
        if self.model_params.skewness.abs() > 1.0
            && self.distribution_type == DistributionType::Gaussian
        {
            self.distribution_type = DistributionType::Skewed;
        }
    }

    /// Get the most promising search locations
    pub fn get_search_candidates(&self, num_candidates: usize) -> Vec<Number> {
        let mut candidates = Vec::new();

        match self.distribution_type {
            DistributionType::MultiModal => {
                // Sample from each mode
                for mode in &self.modes {
                    let center = &self.center + &Number::from(mode.offset.abs() as u64);
                    candidates.push(center);

                    // Add points around mode
                    for i in 1..=2 {
                        let offset = (mode.width * i as f64) as u64;
                        candidates
                            .push(&self.center + &Number::from(mode.offset.abs() as u64 + offset));
                        if mode.offset.abs() as u64 > offset {
                            candidates.push(
                                &self.center + &Number::from((mode.offset.abs() as u64) - offset),
                            );
                        }
                    }
                }
            },
            _ => {
                // Sample from highest probability regions
                let peak = &self.center + &Number::from(self.model_params.mean.abs() as u64);
                candidates.push(peak);

                // Add points at standard deviations
                let sigma = self.model_params.variance.sqrt();
                for i in 1..=3 {
                    let offset = (sigma * i as f64) as u64;
                    candidates.push(&self.center + &Number::from(offset));
                    if self.model_params.mean < 0.0 {
                        candidates.push(&self.center - &Number::from(offset));
                    }
                }
            },
        }

        // Limit to requested number
        candidates.truncate(num_candidates);
        candidates
    }
}

// Import utils at the module level
use crate::utils;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_radius() {
        let n = Number::from(10000u32);
        let region = EnhancedQuantumRegion::new(Number::from(100u32), Number::from(10u32), &n);

        assert_eq!(region.radius.current, Number::from(10u32));
        assert_eq!(region.distribution_type, DistributionType::Gaussian);
    }

    #[test]
    fn test_probability_calculation() {
        let n = Number::from(10000u32);
        let region = EnhancedQuantumRegion::new(Number::from(100u32), Number::from(50u32), &n);

        // Probability should be highest at center
        let p_center = region.probability_at(&Number::from(100u32));
        let p_offset = region.probability_at(&Number::from(110u32));

        assert!(p_center > p_offset);
    }
}
