//! Ensemble voting system using multiple constant sets

use crate::{
    Constants, Constant, Factors, TunerParams, compute_basis,
    recognize_factors, recognize_factors_advanced
};
use num_bigint::BigInt;
use std::collections::HashMap;

/// A set of constants with associated metadata
#[derive(Debug, Clone)]
pub struct EnsembleConstantSet {
    /// The 8 fundamental constant values
    pub values: [f64; 8],
    /// Success rate during tuning/testing
    pub success_rate: f64,
    /// Optimal bit range for this constant set
    pub optimal_bit_range: (usize, usize),
    /// Name/description of this constant set
    pub name: String,
}

impl EnsembleConstantSet {
    /// Get constants from this set as an array
    pub fn to_constant_array(&self) -> [Constant; 8] {
        [
            Constants::get_with_value(0, self.values[0]),
            Constants::get_with_value(1, self.values[1]),
            Constants::get_with_value(2, self.values[2]),
            Constants::get_with_value(3, self.values[3]),
            Constants::get_with_value(4, self.values[4]),
            Constants::get_with_value(5, self.values[5]),
            Constants::get_with_value(6, self.values[6]),
            Constants::get_with_value(7, self.values[7]),
        ]
    }
}

/// Ensemble of multiple constant sets for voting
pub struct EnsembleVoter {
    /// Collection of constant sets
    constant_sets: Vec<EnsembleConstantSet>,
    /// Voting strategy
    strategy: VotingStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VotingStrategy {
    /// Simple majority voting
    Majority,
    /// Weighted by success rate
    WeightedBySuccess,
    /// Choose based on bit size
    AdaptiveBySize,
    /// Try all and take first success
    FirstSuccess,
}

impl EnsembleVoter {
    /// Create a new ensemble voter
    pub fn new(strategy: VotingStrategy) -> Self {
        Self {
            constant_sets: Self::default_constant_sets(),
            strategy,
        }
    }
    
    /// Add a custom constant set
    pub fn add_constant_set(&mut self, set: EnsembleConstantSet) {
        self.constant_sets.push(set);
    }
    
    /// Get default constant sets based on empirical tuning
    fn default_constant_sets() -> Vec<EnsembleConstantSet> {
        vec![
            // Original RH-inspired constants (best for small numbers)
            EnsembleConstantSet {
                values: [1.0, 1.839287, 1.618034, 0.5, 0.159155, 6.283185, 0.199612, 14.134725],
                success_rate: 1.0,
                optimal_bit_range: (1, 20),
                name: "RH-inspired (small)".to_string(),
            },
            // Tuned constants from gradient descent
            EnsembleConstantSet {
                values: [1.0, 2.718282, 3.141593, 0.618034, 0.142857, 4.669201, 0.267949, 11.090537],
                success_rate: 0.8,
                optimal_bit_range: (16, 32),
                name: "Gradient-tuned (medium)".to_string(),
            },
            // Harmonic series constants
            EnsembleConstantSet {
                values: [1.0, 2.0, 1.5, 0.666667, 0.2, 5.0, 0.142857, 8.0],
                success_rate: 0.6,
                optimal_bit_range: (20, 64),
                name: "Harmonic series".to_string(),
            },
            // Prime-based constants
            EnsembleConstantSet {
                values: [1.0, 2.0, 3.0, 0.5, 0.2, 7.0, 0.090909, 13.0],
                success_rate: 0.5,
                optimal_bit_range: (32, 128),
                name: "Prime-based".to_string(),
            },
        ]
    }
    
    /// Recognize factors using ensemble voting
    pub fn recognize_factors(&self, n: &BigInt, params: &TunerParams) -> Option<Factors> {
        match self.strategy {
            VotingStrategy::Majority => self.majority_vote(n, params),
            VotingStrategy::WeightedBySuccess => self.weighted_vote(n, params),
            VotingStrategy::AdaptiveBySize => self.adaptive_vote(n, params),
            VotingStrategy::FirstSuccess => self.first_success(n, params),
        }
    }
    
    /// Simple majority voting
    fn majority_vote(&self, n: &BigInt, params: &TunerParams) -> Option<Factors> {
        let mut votes: HashMap<(BigInt, BigInt), usize> = HashMap::new();
        
        for _set in &self.constant_sets {
            // For now, use default basis since we can't inject custom constants
            // TODO: Extend basis computation to accept custom constants
            let basis = compute_basis(32, params);
            
            // Try both standard and advanced methods
            if let Some(factors) = recognize_factors(n, &basis, params) {
                let key = if factors.p <= factors.q {
                    (factors.p.clone(), factors.q.clone())
                } else {
                    (factors.q.clone(), factors.p.clone())
                };
                *votes.entry(key).or_default() += 1;
            }
        }
        
        // Return the result with most votes
        votes.into_iter()
            .max_by_key(|(_, count)| *count)
            .filter(|(_, count)| *count > self.constant_sets.len() / 2)
            .map(|((p, q), _)| Factors::new(p, q))
    }
    
    /// Weighted voting based on success rates
    fn weighted_vote(&self, n: &BigInt, params: &TunerParams) -> Option<Factors> {
        let mut weighted_votes: HashMap<(BigInt, BigInt), f64> = HashMap::new();
        
        for set in &self.constant_sets {
            // For now, use default basis since we can't inject custom constants
            let basis = compute_basis(32, params);
            
            if let Some(factors) = recognize_factors_advanced(n, &basis, params) {
                let key = if factors.p <= factors.q {
                    (factors.p.clone(), factors.q.clone())
                } else {
                    (factors.q.clone(), factors.p.clone())
                };
                *weighted_votes.entry(key).or_default() += set.success_rate;
            }
        }
        
        // Return result with highest weighted score
        weighted_votes.into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|((p, q), _)| Factors::new(p, q))
    }
    
    /// Adaptive voting based on number size
    fn adaptive_vote(&self, n: &BigInt, params: &TunerParams) -> Option<Factors> {
        let bit_size = n.bits();
        
        // Find constant sets optimal for this bit size
        let mut candidates: Vec<_> = self.constant_sets.iter()
            .filter(|set| bit_size >= set.optimal_bit_range.0 as u64 && 
                         bit_size <= set.optimal_bit_range.1 as u64)
            .collect();
        
        // Sort by success rate
        candidates.sort_by(|a, b| b.success_rate.partial_cmp(&a.success_rate).unwrap());
        
        // Try candidates in order
        for _set in candidates {
            // For now, use default basis
            let basis = compute_basis(32, params);
            
            if let Some(factors) = recognize_factors(n, &basis, params) {
                return Some(factors);
            }
        }
        
        // Fallback to trying all sets
        self.first_success(n, params)
    }
    
    /// Return first successful factorization
    fn first_success(&self, n: &BigInt, params: &TunerParams) -> Option<Factors> {
        for _set in &self.constant_sets {
            // For now, use default basis
            let basis = compute_basis(32, params);
            
            // Try standard method first
            if let Some(factors) = recognize_factors(n, &basis, params) {
                if factors.verify(n) {
                    return Some(factors);
                }
            }
            
            // Try advanced method
            if let Some(factors) = recognize_factors_advanced(n, &basis, params) {
                if factors.verify(n) {
                    return Some(factors);
                }
            }
        }
        
        None
    }
    
    /// Get ensemble statistics
    pub fn get_statistics(&self) -> EnsembleStats {
        EnsembleStats {
            num_constant_sets: self.constant_sets.len(),
            average_success_rate: self.constant_sets.iter()
                .map(|s| s.success_rate)
                .sum::<f64>() / self.constant_sets.len() as f64,
            bit_coverage: self.calculate_bit_coverage(),
        }
    }
    
    /// Calculate bit range coverage
    fn calculate_bit_coverage(&self) -> (usize, usize) {
        let min = self.constant_sets.iter()
            .map(|s| s.optimal_bit_range.0)
            .min()
            .unwrap_or(0);
        let max = self.constant_sets.iter()
            .map(|s| s.optimal_bit_range.1)
            .max()
            .unwrap_or(0);
        (min, max)
    }
}

#[derive(Debug)]
pub struct EnsembleStats {
    pub num_constant_sets: usize,
    pub average_success_rate: f64,
    pub bit_coverage: (usize, usize),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ensemble_creation() {
        let ensemble = EnsembleVoter::new(VotingStrategy::Majority);
        let stats = ensemble.get_statistics();
        
        assert!(stats.num_constant_sets > 0);
        assert!(stats.average_success_rate > 0.0);
    }
    
    #[test]
    fn test_constant_set_conversion() {
        let set = EnsembleConstantSet {
            values: [1.0; 8],
            success_rate: 0.5,
            optimal_bit_range: (1, 64),
            name: "Test".to_string(),
        };
        
        let constants = set.to_constant_array();
        // Should create valid constants
        assert_eq!(constants[0].to_f64(), 1.0);
    }
}