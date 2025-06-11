//! Data collection for empirical observation
//!
//! This module collects factorization data without presuming what matters.

use crate::types::{Number, Observation};
use crate::utils;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::sync::Mutex;

/// Collector for empirical observations
#[derive(Debug)]
pub struct Collector {
    /// Whether to show progress
    show_progress: bool,
    
    /// Detail level for observations
    detail_level: DetailLevel,
}

/// Level of detail for observations
#[derive(Debug, Clone, Copy)]
pub enum DetailLevel {
    /// Basic factorization only
    Basic,
    
    /// Include derived observations
    Standard,
    
    /// Full observation including harmonics
    Full,
}

impl Collector {
    /// Create a new collector
    pub fn new() -> Self {
        Collector {
            show_progress: true,
            detail_level: DetailLevel::Standard,
        }
    }
    
    /// Set progress display
    pub fn with_progress(mut self, show: bool) -> Self {
        self.show_progress = show;
        self
    }
    
    /// Set detail level
    pub fn with_detail_level(mut self, level: DetailLevel) -> Self {
        self.detail_level = level;
        self
    }
    
    /// Collect observations for a range of numbers
    pub fn collect_range(&self, range: std::ops::Range<u64>) -> Vec<Observation> {
        let total = range.end - range.start;
        let observations = Mutex::new(Vec::with_capacity(total as usize / 2));
        
        let progress = if self.show_progress {
            let pb = ProgressBar::new(total);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            Some(pb)
        } else {
            None
        };
        
        // Parallel collection
        (range.start..range.end).into_par_iter().for_each(|n| {
            if let Some(pb) = &progress {
                pb.inc(1);
            }
            
            // Factor the number
            if let Some((p, q)) = self.factor_number(n) {
                let obs = Observation::new(
                    Number::from(n),
                    Number::from(p),
                    Number::from(q),
                );
                observations.lock().unwrap().push(obs);
            }
        });
        
        if let Some(pb) = progress {
            pb.finish_with_message("Collection complete");
        }
        
        let mut result = observations.into_inner().unwrap();
        result.sort_by_key(|obs| obs.n.clone());
        result
    }
    
    /// Collect observations for specific numbers
    pub fn collect_numbers(&self, numbers: &[Number]) -> Vec<Observation> {
        let observations = Mutex::new(Vec::with_capacity(numbers.len()));
        
        numbers.par_iter().for_each(|n| {
            if let Some((p, q)) = self.factor_large(n) {
                let obs = Observation::new(n.clone(), p, q);
                observations.lock().unwrap().push(obs);
            }
        });
        
        observations.into_inner().unwrap()
    }
    
    /// Factor a small number (trial division)
    fn factor_number(&self, n: u64) -> Option<(u64, u64)> {
        if n < 2 {
            return None;
        }
        
        // Check small primes
        for p in 2..=(n as f64).sqrt() as u64 + 1 {
            if n % p == 0 {
                let q = n / p;
                if self.is_prime(p) && self.is_prime(q) {
                    return Some((p, q));
                }
            }
        }
        
        None
    }
    
    /// Factor a large number (using various methods)
    fn factor_large(&self, n: &Number) -> Option<(Number, Number)> {
        // For now, use trial division for numbers we can handle
        if n.bit_length() <= 64 {
            if let Some(n_u64) = n.as_integer().to_u64() {
                if let Some((p, q)) = self.factor_number(n_u64) {
                    return Some((Number::from(p), Number::from(q)));
                }
            }
        }
        
        // For larger numbers, we would implement more sophisticated methods
        // This is where The Pattern would emerge from observation
        None
    }
    
    /// Simple primality test
    fn is_prime(&self, n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }
        
        for i in (3..=(n as f64).sqrt() as u64 + 1).step_by(2) {
            if n % i == 0 {
                return false;
            }
        }
        
        true
    }
}

impl Default for Collector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_collector() {
        let collector = Collector::new().with_progress(false);
        let observations = collector.collect_range(10..20);
        
        // Should find composite numbers in range
        assert!(!observations.is_empty());
        
        // Check 15 = 3 × 5
        let obs_15 = observations.iter().find(|o| o.n == Number::from(15u32));
        assert!(obs_15.is_some());
        let obs_15 = obs_15.unwrap();
        assert_eq!(obs_15.p, Number::from(3u32));
        assert_eq!(obs_15.q, Number::from(5u32));
    }
}