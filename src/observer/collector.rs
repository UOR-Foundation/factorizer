//! Data collection for empirical observation
//!
//! This module collects factorization data without presuming what matters.

use crate::types::{Number, Observation};
use crate::utils;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde_json;
use std::sync::Mutex;

/// Collector for empirical observations
#[derive(Debug)]
pub struct Collector {
    /// Whether to show progress
    show_progress: bool,

    /// Detail level for observations
    detail_level: DetailLevel,

    /// Collected observations
    observations: Vec<Observation>,
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
            observations: Vec::new(),
        }
    }

    /// Observe a single number
    pub fn observe_single(&mut self, n: Number) -> crate::Result<Observation> {
        if let Some((p, q)) = self.factor_semiprime(&n) {
            let observation = Observation::new(n, p, q);
            self.observations.push(observation.clone());
            Ok(observation)
        } else {
            Err(crate::error::PatternError::FactorizationFailed(format!(
                "Could not factor {}",
                n
            )))
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
                let obs = Observation::new(Number::from(n), Number::from(p), Number::from(q));
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

        // For larger numbers, use our factor_semiprime method
        self.factor_semiprime(n)
    }

    /// Factor a semiprime (product of exactly two primes)
    /// This method emerges from observing patterns in semiprimes
    pub fn factor_semiprime(&self, n: &Number) -> Option<(Number, Number)> {
        // First check if n is even
        if n.is_even() {
            let half = n / &Number::from(2u32);
            if utils::is_probable_prime(&half, 10) {
                return Some((Number::from(2u32), half));
            }
        }

        // Use trial division first for small factors
        if let Ok(factors) = utils::trial_division(n, Some(&Number::from(10000u32))) {
            if factors.len() == 2 {
                return Some((factors[0].clone(), factors[1].clone()));
            }
            if factors.len() == 1 && &factors[0] == n {
                // n is prime, not a semiprime
                return None;
            }
        }

        // For larger semiprimes, use Fermat's method
        // This emerges from observing that semiprimes have factors near sqrt(n)
        let sqrt_n = utils::integer_sqrt(n).ok()?;
        let mut a = sqrt_n + 1u32;
        let max_iterations = 1000000; // Prevent infinite loops

        for _ in 0..max_iterations {
            let a_squared = &a * &a;
            let b_squared = &a_squared - n;

            // Check if b_squared is a perfect square
            if let Ok(b) = utils::integer_sqrt(&b_squared) {
                if &b * &b == b_squared {
                    // Found factors: n = (a+b)(a-b)
                    let p = &a + &b;
                    let q = &a - &b;

                    // Verify they are prime
                    if utils::is_probable_prime(&p, 10) && utils::is_probable_prime(&q, 10) {
                        return Some((q, p)); // Return smaller factor first
                    }
                }
            }

            a += 1u32;

            // If a gets too large, the factors are very unbalanced
            if &a > &(n / &Number::from(2u32)) {
                break;
            }
        }

        // If Fermat's method fails, try Pollard's rho
        self.pollard_rho(n)
    }

    /// Pollard's rho algorithm for finding factors
    /// Emerges from observing cyclic patterns in modular arithmetic
    fn pollard_rho(&self, n: &Number) -> Option<(Number, Number)> {
        if n.is_one() {
            return None;
        }

        if utils::is_probable_prime(n, 10) {
            return None;
        }

        let mut x = Number::from(2u32);
        let mut y = Number::from(2u32);
        let mut d = Number::from(1u32);

        let max_iterations = 100000;
        let mut iterations = 0;

        while d.is_one() && iterations < max_iterations {
            // x = (x^2 + 1) mod n
            x = (&x * &x + 1u32) % n;

            // y = (y^2 + 1) mod n, twice
            y = (&y * &y + 1u32) % n;
            y = (&y * &y + 1u32) % n;

            // d = gcd(|x - y|, n)
            let diff = if x > y { &x - &y } else { &y - &x };
            d = utils::gcd(&diff, n);

            iterations += 1;
        }

        if d == *n || d.is_one() {
            // Failed to find a factor
            None
        } else {
            // Found a factor
            let other_factor = n / &d;

            // Ensure both are prime
            if utils::is_probable_prime(&d, 10) && utils::is_probable_prime(&other_factor, 10) {
                if d < other_factor {
                    Some((d, other_factor))
                } else {
                    Some((other_factor, d))
                }
            } else {
                // One of the factors is composite, recurse
                // Note: This is for semiprimes only, so if one factor is composite,
                // the number is not a semiprime
                None
            }
        }
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

    /// Observe multiple numbers in parallel
    pub fn observe_parallel(&mut self, numbers: &[Number]) -> crate::Result<Vec<Observation>> {
        // Use collect_numbers which already handles parallel processing
        let results = self.collect_numbers(numbers);

        // Add to our stored observations
        self.observations.extend(results.clone());

        if results.is_empty() {
            Err(crate::error::PatternError::FactorizationFailed(
                "No numbers could be factored".to_string(),
            ))
        } else {
            Ok(results)
        }
    }

    /// Get all collected observations
    pub fn observations(&self) -> &[Observation] {
        &self.observations
    }

    /// Save observations to a file
    pub fn save_to_file(&self, path: &str) -> crate::Result<()> {
        use std::fs;
        use std::path::Path;

        // Create directory if it doesn't exist
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent)?;
        }

        // Serialize observations
        let json = serde_json::to_string_pretty(&self.observations)?;

        // Write to file
        fs::write(path, json)?;

        Ok(())
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
