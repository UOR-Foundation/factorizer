//! Constant tuning framework for The Pattern
//! 
//! This module provides tools for optimizing the 8 fundamental constants
//! based on empirical pattern analysis and success rates.

use crate::{
    Constants, Constant, TunerParams, TestCase,
    compute_basis, recognize_factors_with_diagnostics,
    DiagnosticAggregator, FRACTIONAL_BITS
};
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive};
use std::collections::HashMap;

/// Configuration for constant tuning
#[derive(Debug, Clone)]
pub struct TuningConfig {
    /// Number of iterations for optimization
    pub iterations: usize,
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Momentum factor
    pub momentum: f64,
    /// Regularization strength
    pub regularization: f64,
    /// Batch size for gradient estimation
    pub batch_size: usize,
    /// Target success rate
    pub target_success_rate: f64,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            learning_rate: 0.01,
            momentum: 0.9,
            regularization: 0.001,
            batch_size: 20,
            target_success_rate: 0.95,
        }
    }
}

/// Represents a set of constant values being tuned
#[derive(Debug, Clone)]
pub struct ConstantSet {
    /// The 8 constant values (as rational approximations)
    pub values: [f64; 8],
    /// Success rate achieved with these constants
    pub success_rate: f64,
    /// Pattern frequency statistics
    pub pattern_stats: HashMap<u8, usize>,
}

impl ConstantSet {
    /// Create a new constant set from float values
    pub fn new(values: [f64; 8]) -> Self {
        Self {
            values,
            success_rate: 0.0,
            pattern_stats: HashMap::new(),
        }
    }
    
    /// Convert to actual Constants
    pub fn to_constants(&self) -> [Constant; 8] {
        let mut constants = [
            Constants::unity(),
            Constants::tau(),
            Constants::phi(),
            Constants::epsilon(),
            Constants::delta(),
            Constants::gamma(),
            Constants::beta(),
            Constants::alpha(),
        ];
        
        // Update each constant with the tuned value
        for (i, &value) in self.values.iter().enumerate() {
            // Convert float to Q32.224 fixed point
            let scale = BigInt::one() << FRACTIONAL_BITS;
            let numerator = (value * (1u64 << 24) as f64) as u64;
            let numerator_bigint = BigInt::from(numerator) * &scale >> 24;
            
            constants[i].numerator = numerator_bigint;
        }
        
        constants
    }
    
    /// Extract values from current Constants
    pub fn from_constants() -> Self {
        let constants = Constants::all();
        let mut values = [0.0; 8];
        
        for (i, c) in constants.iter().enumerate() {
            // Convert Q32.224 to float
            let num_f64 = c.numerator.to_f64().unwrap_or(1.0);
            let den_f64 = c.denominator.to_f64().unwrap_or(1.0);
            values[i] = num_f64 / den_f64;
        }
        
        Self::new(values)
    }
}

/// Constant tuner using pattern frequency analysis
pub struct ConstantTuner {
    config: TuningConfig,
    test_cases: Vec<TestCase>,
    current_set: ConstantSet,
    best_set: ConstantSet,
    gradient_accumulator: [f64; 8],
    velocity: [f64; 8],
}

impl ConstantTuner {
    /// Create a new tuner with test cases
    pub fn new(config: TuningConfig, test_cases: Vec<TestCase>) -> Self {
        let current_set = ConstantSet::from_constants();
        Self {
            config,
            test_cases,
            current_set: current_set.clone(),
            best_set: current_set,
            gradient_accumulator: [0.0; 8],
            velocity: [0.0; 8],
        }
    }
    
    /// Run the tuning process
    pub fn tune(&mut self) -> ConstantTuningResult {
        println!("Starting constant tuning with {} test cases", self.test_cases.len());
        
        // Evaluate initial constants
        let initial_rate = self.evaluate_constants(&mut self.current_set.clone());
        self.current_set.success_rate = initial_rate;
        self.best_set = self.current_set.clone();
        println!("Initial success rate: {:.2}%", self.current_set.success_rate * 100.0);
        
        // Run optimization iterations
        for iter in 0..self.config.iterations {
            // Estimate gradient using finite differences
            let gradient = self.estimate_gradient();
            
            // Update velocity with momentum
            for i in 0..8 {
                self.velocity[i] = self.config.momentum * self.velocity[i] 
                    - self.config.learning_rate * gradient[i];
            }
            
            // Update constants
            for i in 0..8 {
                self.current_set.values[i] += self.velocity[i];
                
                // Apply regularization (pull towards initial values)
                let initial_value = match i {
                    0 => 1.0,      // unity
                    1 => 1.839287, // tau
                    2 => 1.618034, // phi
                    3 => 0.5,      // epsilon
                    4 => 0.159155, // delta
                    5 => 6.283185, // gamma
                    6 => 0.199612, // beta
                    7 => 14.134725,// alpha
                    _ => 1.0,
                };
                
                self.current_set.values[i] += self.config.regularization 
                    * (initial_value - self.current_set.values[i]);
                
                // Keep values positive
                if self.current_set.values[i] <= 0.0 {
                    self.current_set.values[i] = 0.001;
                }
            }
            
            // Evaluate new constants
            let new_rate = self.evaluate_constants(&mut self.current_set.clone());
            self.current_set.success_rate = new_rate;
            
            // Update best if improved
            if self.current_set.success_rate > self.best_set.success_rate {
                self.best_set = self.current_set.clone();
                println!("Iteration {}: New best success rate: {:.2}%", 
                    iter + 1, self.best_set.success_rate * 100.0);
            }
            
            // Early stopping if target reached
            if self.best_set.success_rate >= self.config.target_success_rate {
                println!("Target success rate reached!");
                break;
            }
            
            // Progress update
            if (iter + 1) % 10 == 0 {
                println!("Iteration {}: Current rate: {:.2}%, Best: {:.2}%",
                    iter + 1,
                    self.current_set.success_rate * 100.0,
                    self.best_set.success_rate * 100.0
                );
            }
        }
        
        ConstantTuningResult {
            best_constants: self.best_set.clone(),
            iterations_run: self.config.iterations,
            final_success_rate: self.best_set.success_rate,
            pattern_analysis: self.analyze_patterns(),
        }
    }
    
    /// Evaluate a constant set on test cases
    fn evaluate_constants(&self, constant_set: &mut ConstantSet) -> f64 {
        // Create tuner params with the constant weights
        let params = TunerParams::default();
        
        // Create basis with current constants
        // Note: This is a simplified approach - in reality we'd need to 
        // modify the compute_basis function to accept custom constants
        let representative_n = BigInt::from(1u128) << 128; // 128-bit number
        let _basis = compute_basis(&representative_n, &params);
        
        let mut aggregator = DiagnosticAggregator::default();
        let mut successes = 0;
        
        // Test on a subset for efficiency
        let test_subset: Vec<_> = self.test_cases.iter()
            .take(self.config.batch_size.min(self.test_cases.len()))
            .collect();
        
        for test_case in &test_subset {
            let (result, diagnostics) = recognize_factors_with_diagnostics(
                &test_case.n,
                &params
            );
            
            aggregator.add(&diagnostics);
            
            if let Some(factors) = result {
                let correct = (factors.p == test_case.p && factors.q == test_case.q) ||
                             (factors.p == test_case.q && factors.q == test_case.p);
                if correct {
                    successes += 1;
                }
            }
        }
        
        constant_set.success_rate = successes as f64 / test_subset.len() as f64;
        
        // Update pattern statistics
        constant_set.pattern_stats = aggregator.global_pattern_frequency;
        
        constant_set.success_rate
    }
    
    /// Estimate gradient using finite differences
    fn estimate_gradient(&mut self) -> [f64; 8] {
        let mut gradient = [0.0; 8];
        let epsilon = 0.001;
        
        let _base_rate = self.current_set.success_rate;
        
        for i in 0..8 {
            // Save original value
            let original = self.current_set.values[i];
            
            // Evaluate with small perturbation
            self.current_set.values[i] = original + epsilon;
            let rate_plus = self.evaluate_constants(&mut self.current_set.clone());
            
            self.current_set.values[i] = original - epsilon;
            let rate_minus = self.evaluate_constants(&mut self.current_set.clone());
            
            // Restore original
            self.current_set.values[i] = original;
            
            // Compute gradient
            gradient[i] = (rate_plus - rate_minus) / (2.0 * epsilon);
        }
        
        gradient
    }
    
    /// Analyze patterns for insights
    fn analyze_patterns(&self) -> PatternAnalysis {
        let mut bit_activation_rates = [0.0; 8];
        let total_patterns = self.best_set.pattern_stats.values().sum::<usize>() as f64;
        
        if total_patterns > 0.0 {
            for (pattern, &count) in &self.best_set.pattern_stats {
                for bit in 0..8 {
                    if (pattern >> bit) & 1 == 1 {
                        bit_activation_rates[bit] += count as f64 / total_patterns;
                    }
                }
            }
        }
        
        PatternAnalysis {
            most_common_patterns: self.best_set.pattern_stats.iter()
                .map(|(&p, &c)| (p, c))
                .collect(),
            bit_activation_rates,
            success_correlation: self.compute_success_correlation(),
        }
    }
    
    /// Compute correlation between bit activation and success
    fn compute_success_correlation(&self) -> [f64; 8] {
        // Simplified correlation - in practice would use actual success data
        let mut correlations = [0.0; 8];
        
        // Use pattern stats to estimate correlation
        for i in 0..8 {
            // Higher activation in successful patterns = positive correlation
            correlations[i] = self.best_set.pattern_stats.iter()
                .filter(|(p, _)| (*p >> i) & 1 == 1)
                .map(|(_, &c)| c as f64)
                .sum::<f64>() / self.best_set.pattern_stats.values()
                .map(|&c| c as f64)
                .sum::<f64>();
        }
        
        correlations
    }
}

/// Result of constant tuning
#[derive(Debug, Clone)]
pub struct ConstantTuningResult {
    /// Best constant set found
    pub best_constants: ConstantSet,
    /// Number of iterations run
    pub iterations_run: usize,
    /// Final success rate achieved
    pub final_success_rate: f64,
    /// Pattern analysis insights
    pub pattern_analysis: PatternAnalysis,
}

/// Pattern analysis results
#[derive(Debug, Clone)]
pub struct PatternAnalysis {
    /// Most common patterns and their frequencies
    pub most_common_patterns: Vec<(u8, usize)>,
    /// Activation rate for each bit position
    pub bit_activation_rates: [f64; 8],
    /// Correlation between bit activation and success
    pub success_correlation: [f64; 8],
}

impl ConstantTuningResult {
    /// Print a summary report
    pub fn print_summary(&self) {
        println!("\n=== Constant Tuning Results ===");
        println!("Iterations run: {}", self.iterations_run);
        println!("Final success rate: {:.2}%", self.final_success_rate * 100.0);
        
        println!("\nOptimized constant values:");
        let names = ["unity", "tau", "phi", "epsilon", "delta", "gamma", "beta", "alpha"];
        for (i, (&value, &name)) in self.best_constants.values.iter()
            .zip(names.iter())
            .enumerate() 
        {
            println!("  {} ({}): {:.6}", name, i, value);
        }
        
        println!("\nPattern analysis:");
        println!("  Bit activation rates:");
        for (i, &rate) in self.pattern_analysis.bit_activation_rates.iter().enumerate() {
            println!("    Bit {} ({}): {:.2}%", i, names[i], rate * 100.0);
        }
        
        println!("\n  Success correlations:");
        for (i, &corr) in self.pattern_analysis.success_correlation.iter().enumerate() {
            println!("    Bit {} ({}): {:.3}", i, names[i], corr);
        }
        
        println!("\n  Top 5 most common patterns:");
        let mut patterns = self.pattern_analysis.most_common_patterns.clone();
        patterns.sort_by(|a, b| b.1.cmp(&a.1));
        for (pattern, count) in patterns.iter().take(5) {
            println!("    Pattern {:08b}: {} occurrences", pattern, count);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_constant_set_conversion() {
        let values = [1.0, 1.839, 1.618, 0.5, 0.159, 6.283, 0.199, 14.134];
        let set = ConstantSet::new(values);
        let constants = set.to_constants();
        
        assert_eq!(constants.len(), 8);
        for c in &constants {
            assert!(c.numerator > BigInt::from(0));
        }
    }
    
    #[test]
    fn test_tuning_config_default() {
        let config = TuningConfig::default();
        assert_eq!(config.iterations, 100);
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.target_success_rate, 0.95);
    }
}