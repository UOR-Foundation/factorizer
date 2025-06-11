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
        QuantumRegion {
            center,
            radius,
            probability_distribution: Vec::new(),
            peak_offset: 0,
            confidence: 0.0,
        }
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
        let n = dist.len() as f64;

        // Calculate mean
        let mean: f64 = dist.iter().enumerate().map(|(i, &p)| i as f64 * p).sum::<f64>()
            / dist.iter().sum::<f64>().max(1.0);

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
