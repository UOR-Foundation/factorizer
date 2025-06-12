//! Quantum neighborhood type
//!
//! This type emerges from the observation that factors exist in
//! specific regions identified by The Pattern.

use crate::types::Number;
use serde::{Deserialize, Serialize};

/// The quantum neighborhood where factors exist
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumRegion {
    /// Center of the quantum neighborhood
    pub center: Number,

    /// Radius of the neighborhood
    pub radius: Number,

    /// Probability distribution within the region
    pub probability_distribution: Vec<f64>,

    /// Peak probability location (offset from center)
    pub peak_offset: i64,

    /// Confidence in this region
    pub confidence: f64,
}

impl QuantumRegion {
    /// Create a new quantum region
    pub fn new(center: Number, radius: Number) -> Self {
        let mut region = QuantumRegion {
            center: center.clone(),
            radius: radius.clone(),
            probability_distribution: Vec::new(),
            peak_offset: 0,
            confidence: 0.0,
        };

        // Initialize probability distribution
        region.initialize_probability_distribution();
        region
    }

    /// Initialize probability distribution based on quantum principles
    pub fn initialize_probability_distribution(&mut self) {
        let radius_int = self.radius.as_integer().to_u64().unwrap_or(100) as usize;
        let size = 2 * radius_int + 1;
        self.probability_distribution = vec![0.0; size];

        // Create Gaussian-like distribution with quantum fluctuations
        let sigma = radius_int as f64 / 3.0;
        let center_idx = radius_int;

        for i in 0..size {
            let offset = i as f64 - center_idx as f64;

            // Base Gaussian
            let gaussian = (-offset * offset / (2.0 * sigma * sigma)).exp();

            // Quantum fluctuations
            let fluctuation = (offset.abs() / 10.0).sin() * 0.1;

            // Interference pattern
            let interference = (offset * std::f64::consts::PI / 20.0).cos() * 0.05;

            self.probability_distribution[i] = (gaussian + fluctuation + interference).max(0.0);
        }

        // Normalize
        let sum: f64 = self.probability_distribution.iter().sum();
        if sum > 0.0 {
            for p in &mut self.probability_distribution {
                *p /= sum;
            }
        }

        // Find peak
        self.find_peak();
    }

    /// Check if a value is within the region
    pub fn contains(&self, value: &Number) -> bool {
        let distance =
            if value >= &self.center { value - &self.center } else { &self.center - value };

        distance <= self.radius
    }

    /// Get the probability at a specific offset
    pub fn probability_at(&self, offset: i64) -> f64 {
        let idx = (offset + self.radius.as_integer().to_i64().unwrap_or(0)) as usize;
        if idx < self.probability_distribution.len() {
            self.probability_distribution[idx]
        } else {
            0.0
        }
    }

    /// Find the peak probability location
    pub fn find_peak(&mut self) {
        if let Some((idx, _)) = self
            .probability_distribution
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        {
            self.peak_offset = idx as i64 - self.radius.as_integer().to_i64().unwrap_or(0);
        }
    }

    /// Get search bounds
    pub fn bounds(&self) -> (Number, Number) {
        let lower = &self.center - &self.radius;
        let upper = &self.center + &self.radius;
        (lower, upper)
    }

    /// Get the most likely position
    pub fn most_likely(&self) -> Number {
        &self.center + &Number::from(self.peak_offset as u64)
    }

    /// Update probability distribution based on observations
    pub fn update_distribution(&mut self, observation: &Number, success: bool) {
        if !self.contains(observation) {
            return;
        }

        let offset = if observation >= &self.center {
            observation - &self.center
        } else {
            &self.center - observation
        };

        let idx = (offset.as_integer().to_u64().unwrap_or(0)
            + self.radius.as_integer().to_u64().unwrap_or(0)) as usize;

        if idx < self.probability_distribution.len() {
            // Bayesian update
            let prior = self.probability_distribution[idx];
            let likelihood = if success { 0.9 } else { 0.1 };
            let evidence = 0.5; // Simplified

            self.probability_distribution[idx] = (prior * likelihood) / evidence;
        }

        // Re-normalize
        let sum: f64 = self.probability_distribution.iter().sum();
        if sum > 0.0 {
            for p in &mut self.probability_distribution {
                *p /= sum;
            }
        }

        // Update peak
        self.find_peak();
    }

    /// Calculate confidence interval
    pub fn confidence_interval(&self, confidence_level: f64) -> (Number, Number) {
        let cumsum = self.cumulative_distribution();
        let lower_percentile = (1.0 - confidence_level) / 2.0;
        let upper_percentile = 1.0 - lower_percentile;

        let lower_idx = cumsum.iter().position(|&p| p >= lower_percentile).unwrap_or(0);
        let upper_idx =
            cumsum.iter().position(|&p| p >= upper_percentile).unwrap_or(cumsum.len() - 1);

        let lower_offset = lower_idx as i64 - self.radius.as_integer().to_i64().unwrap_or(0);
        let upper_offset = upper_idx as i64 - self.radius.as_integer().to_i64().unwrap_or(0);

        let lower = &self.center + &Number::from(lower_offset.abs() as u64);
        let upper = &self.center + &Number::from(upper_offset.abs() as u64);

        (lower, upper)
    }

    /// Get cumulative distribution
    fn cumulative_distribution(&self) -> Vec<f64> {
        let mut cumsum = vec![0.0; self.probability_distribution.len()];
        let mut sum = 0.0;

        for (i, &p) in self.probability_distribution.iter().enumerate() {
            sum += p;
            cumsum[i] = sum;
        }

        cumsum
    }

    /// Apply quantum tunneling effect
    pub fn apply_quantum_tunneling(&mut self, barrier_height: f64) {
        // Simulate quantum tunneling by modifying probability distribution
        let radius_int = self.radius.as_integer().to_u64().unwrap_or(100) as f64;

        for (i, p) in self.probability_distribution.iter_mut().enumerate() {
            let offset = i as f64 - radius_int;
            let tunneling_prob = (-barrier_height * offset.abs() / radius_int).exp();
            *p *= tunneling_prob;
        }

        // Re-normalize
        let sum: f64 = self.probability_distribution.iter().sum();
        if sum > 0.0 {
            for p in &mut self.probability_distribution {
                *p /= sum;
            }
        }
    }
}

/// Distribution analysis for quantum regions
#[derive(Debug, Clone)]
pub struct DistributionAnalysis {
    /// The quantum region being analyzed
    pub region: QuantumRegion,

    /// Statistical moments
    pub mean: f64,
    pub variance: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

impl DistributionAnalysis {
    /// Analyze a quantum region's distribution
    pub fn analyze(region: &QuantumRegion) -> Self {
        let dist = &region.probability_distribution;

        // Calculate mean
        let sum_p: f64 = dist.iter().sum::<f64>().max(1.0);
        let mean: f64 = dist.iter().enumerate().map(|(i, &p)| i as f64 * p).sum::<f64>() / sum_p;

        // Calculate variance
        let variance: f64 = dist
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let diff = i as f64 - mean;
                diff * diff * p
            })
            .sum::<f64>()
            / dist.iter().sum::<f64>().max(1.0);

        // Higher moments (simplified)
        let skewness = 0.0;
        let kurtosis = 0.0;

        DistributionAnalysis {
            region: region.clone(),
            mean,
            variance,
            skewness,
            kurtosis,
        }
    }

    /// Get the peak location in the distribution
    pub fn peak_location(&self) -> Number {
        self.region.most_likely()
    }

    /// Get a density map of the distribution
    pub fn density_map(&self) -> Vec<(i64, f64)> {
        self.region.probability_distribution
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let offset = i as i64 - self.region.radius.as_integer().to_i64().unwrap_or(0);
                (offset, p)
            })
            .filter(|(_, p)| *p > 0.001) // Only significant probabilities
            .collect()
    }
}
