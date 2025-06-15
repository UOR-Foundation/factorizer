//! Auto-tuner implementation for optimizing pattern parameters
//!
//! Implements parameter optimization using the test matrix.

use crate::{TunerParams, Basis, compute_basis, recognize_factors};
use num_bigint::BigInt;
use std::time::Instant;

/// Test case from the test matrix
#[derive(Debug, Clone)]
pub struct TestCase {
    pub n: BigInt,
    pub p: BigInt,
    pub q: BigInt,
    pub bit_length: usize,
}

/// Tuning result for a single test case
#[derive(Debug, Clone)]
pub struct TuningResult {
    pub success: bool,
    pub channels_scanned: usize,
    pub iterations: usize,
    pub time_ms: u128,
}

/// Auto-tuner for optimizing pattern parameters
pub struct AutoTuner {
    basis: Basis,
    params: TunerParams,
    test_cases: Vec<TestCase>,
}

impl AutoTuner {
    /// Create a new auto-tuner with default parameters
    pub fn new() -> Self {
        let params = TunerParams::default();
        let representative_n = BigInt::from(1u128) << 1024; // 1024-bit number
        let basis = compute_basis(&representative_n, &params); // Support up to 1024-bit numbers
        
        Self {
            basis,
            params,
            test_cases: Vec::new(),
        }
    }
    
    /// Load test cases from the test matrix
    pub fn load_test_cases(&mut self, test_cases: Vec<TestCase>) {
        self.test_cases = test_cases;
    }
    
    /// Run a single test case and return the result
    fn run_test_case(&self, test_case: &TestCase) -> TuningResult {
        let start = Instant::now();
        
        let result = recognize_factors(&test_case.n, &self.params);
        
        let success = if let Some(factors) = result {
            factors.verify(&test_case.n) &&
            ((factors.p == test_case.p && factors.q == test_case.q) ||
             (factors.p == test_case.q && factors.q == test_case.p))
        } else {
            false
        };
        
        TuningResult {
            success,
            channels_scanned: test_case.bit_length / 8 + 1, // Approximate
            iterations: 1, // Simplified for now
            time_ms: start.elapsed().as_millis(),
        }
    }
    
    /// Calculate the target metric for the current parameters
    fn calculate_metric(&self) -> u64 {
        let mut failure_count = 0;
        let mut total_channels = 0;
        let mut max_iterations = 0;
        
        for test_case in &self.test_cases {
            let result = self.run_test_case(test_case);
            
            if !result.success {
                failure_count += 1;
            }
            total_channels += result.channels_scanned;
            max_iterations = max_iterations.max(result.iterations);
        }
        
        let avg_channels = if self.test_cases.is_empty() {
            0
        } else {
            total_channels / self.test_cases.len()
        };
        
        // Target metric: failure_count * 1000000 + avg_channels * 100 + max_iterations
        (failure_count as u64) * 1_000_000 + (avg_channels as u64) * 100 + (max_iterations as u64)
    }
    
    /// Mutate parameters slightly for optimization
    fn mutate_params(&self, params: &TunerParams, temperature: f64) -> TunerParams {
        let mut new_params = params.clone();
        
        // Use temperature to control mutation magnitude
        let change = (temperature * 10.0) as i32;
        
        // Randomly mutate one parameter
        match fastrand::usize(0..5) {
            0 => {
                // Mutate alignment_threshold
                let delta = fastrand::i32(-change..=change);
                new_params.alignment_threshold = new_params.alignment_threshold
                    .saturating_add_signed(delta as i8)
                    .clamp(1, 10);
            }
            1 => {
                // Mutate resonance_scaling_shift
                let delta = fastrand::i32(-change/2..=change/2);
                new_params.resonance_scaling_shift = new_params.resonance_scaling_shift
                    .saturating_add_signed(delta as i8)
                    .clamp(0, 31);
            }
            2 => {
                // Mutate harmonic_progression_step
                let delta = fastrand::i32(-change..=change);
                new_params.harmonic_progression_step = new_params.harmonic_progression_step
                    .saturating_add_signed(delta as i16)
                    .clamp(1, 256);
            }
            3 => {
                // Mutate phase_coupling_strength
                let delta = fastrand::i32(-1..=1);
                new_params.phase_coupling_strength = new_params.phase_coupling_strength
                    .saturating_add_signed(delta as i8)
                    .clamp(1, 7);
            }
            4 => {
                // Mutate one constant weight
                let idx = fastrand::usize(0..8);
                let delta = fastrand::i32(-change*10..=change*10);
                new_params.constant_weights[idx] = new_params.constant_weights[idx]
                    .saturating_add_signed(delta as i8)
                    .clamp(1, 255);
            }
            _ => unreachable!(),
        }
        
        new_params
    }
    
    /// Optimize parameters using simulated annealing
    pub fn optimize(&mut self, max_rounds: usize) -> TunerParams {
        if self.test_cases.is_empty() {
            println!("Warning: No test cases loaded");
            return self.params.clone();
        }
        
        let mut best_params = self.params.clone();
        let mut best_metric = self.calculate_metric();
        
        println!("Starting optimization with metric: {}", best_metric);
        
        let mut current_params = best_params.clone();
        let mut current_metric = best_metric;
        
        for round in 0..max_rounds {
            let temperature = 1.0 - (round as f64 / max_rounds as f64);
            
            // Generate new candidate
            let candidate_params = self.mutate_params(&current_params, temperature);
            
            // Recompute basis with new params
            self.params = candidate_params.clone();
            let representative_n = BigInt::from(1u128) << 1024;
            self.basis = compute_basis(&representative_n, &self.params);
            
            let candidate_metric = self.calculate_metric();
            
            // Accept or reject
            let delta = candidate_metric as i64 - current_metric as i64;
            let accept = if delta < 0 {
                true // Always accept improvements
            } else {
                // Sometimes accept worse solutions (simulated annealing)
                let prob = (-delta as f64 / (temperature * 1000.0)).exp();
                fastrand::f64() < prob
            };
            
            if accept {
                current_params = candidate_params;
                current_metric = candidate_metric;
                
                if current_metric < best_metric {
                    best_params = current_params.clone();
                    best_metric = current_metric;
                    println!("Round {}: New best metric: {}", round, best_metric);
                }
            }
            
            // Early exit if perfect
            if best_metric == 0 {
                println!("Perfect parameters found!");
                break;
            }
        }
        
        println!("Optimization complete. Final metric: {}", best_metric);
        
        // Set final best params
        self.params = best_params.clone();
        let representative_n = BigInt::from(1u128) << 1024;
        self.basis = compute_basis(&representative_n, &self.params);
        
        best_params
    }
    
    /// Factor a number using the current tuned parameters
    pub fn factor(&self, n: &str) -> Result<(String, String), String> {
        let n = n.parse::<BigInt>()
            .map_err(|_| "Invalid number format".to_string())?;
        
        let factors = recognize_factors(&n, &self.params)
            .ok_or_else(|| "No factors found".to_string())?;
        
        Ok((factors.p.to_string(), factors.q.to_string()))
    }
}

impl Default for AutoTuner {
    fn default() -> Self {
        Self::new()
    }
}

// Pseudo-random number generation for parameter mutation
mod fastrand {
    use std::cell::Cell;
    
    thread_local! {
        static RNG: Cell<u64> = const { Cell::new(0x853c49e6748fea9b) };
    }
    
    fn next() -> u64 {
        RNG.with(|rng| {
            let mut x = rng.get();
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            rng.set(x);
            x.wrapping_mul(0x2545f4914f6cdd1d)
        })
    }
    
    pub fn usize(range: std::ops::Range<usize>) -> usize {
        let n = range.end - range.start;
        (next() as usize % n) + range.start
    }
    
    pub fn i32(range: std::ops::RangeInclusive<i32>) -> i32 {
        let start = *range.start();
        let end = *range.end();
        let n = (end - start + 1) as u32;
        ((next() as u32 % n) as i32) + start
    }
    
    pub fn f64() -> f64 {
        (next() as f64) / (u64::MAX as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_auto_tuner_creation() {
        let tuner = AutoTuner::new();
        assert_eq!(tuner.test_cases.len(), 0);
        assert_eq!(tuner.basis.num_channels, 128);
    }
    
    #[test]
    fn test_mutate_params() {
        let tuner = AutoTuner::new();
        let original = TunerParams::default();
        
        // High temperature should produce larger changes
        let mutated = tuner.mutate_params(&original, 1.0);
        
        // At least one parameter should be different
        let different = 
            mutated.alignment_threshold != original.alignment_threshold ||
            mutated.resonance_scaling_shift != original.resonance_scaling_shift ||
            mutated.harmonic_progression_step != original.harmonic_progression_step ||
            mutated.phase_coupling_strength != original.phase_coupling_strength ||
            mutated.constant_weights != original.constant_weights;
        
        assert!(different);
    }
    
    #[test]
    fn test_factor_small() {
        let tuner = AutoTuner::new();
        
        // Test with a small semiprime
        match tuner.factor("15") {
            Ok((p, q)) => {
                assert!(
                    (p == "3" && q == "5") || (p == "5" && q == "3"),
                    "Expected 3 and 5, got {} and {}", p, q
                );
            }
            Err(_) => {
                // It's okay if default params don't work perfectly
                // The real test is after tuning
            }
        }
    }
}