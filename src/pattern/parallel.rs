//! Parallel pattern execution with caching
//!
//! This module provides parallel execution strategies that leverage
//! multiple CPU cores and caching for optimal performance.

use crate::error::PatternError;
use crate::pattern::cache::{CacheConfig, PatternCache};
use crate::types::{Factors, Formalization, Number, Pattern};
use crate::Result;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Parallel execution configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of worker threads
    pub num_threads: usize,

    /// Batch size for parallel processing
    pub batch_size: usize,

    /// Enable progress reporting
    pub show_progress: bool,

    /// Cache configuration
    pub cache_config: CacheConfig,

    /// Strategy for work distribution
    pub distribution_strategy: DistributionStrategy,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        ParallelConfig {
            num_threads: num_cpus::get(),
            batch_size: 100,
            show_progress: true,
            cache_config: CacheConfig::default(),
            distribution_strategy: DistributionStrategy::Dynamic,
        }
    }
}

/// Work distribution strategy
#[derive(Debug, Clone, Copy)]
pub enum DistributionStrategy {
    /// Static distribution - equal chunks
    Static,

    /// Dynamic distribution - work stealing
    Dynamic,

    /// Adaptive - based on number complexity
    Adaptive,
}

/// Parallel execution context
#[derive(Debug)]
pub struct ParallelExecutor {
    config: ParallelConfig,
    cache: Arc<PatternCache>,
    stats: Arc<Mutex<ExecutionStats>>,
}

/// Execution statistics
#[derive(Debug, Default, Clone)]
pub struct ExecutionStats {
    pub total_numbers: usize,
    pub successful: usize,
    pub failed: usize,
    pub cache_hits: usize,
    pub total_time: std::time::Duration,
    pub average_time_per_number: std::time::Duration,
}

impl ParallelExecutor {
    /// Create new parallel executor
    pub fn new(config: ParallelConfig) -> Self {
        let cache = Arc::new(PatternCache::new(config.cache_config.clone()));

        ParallelExecutor {
            config,
            cache,
            stats: Arc::new(Mutex::new(ExecutionStats::default())),
        }
    }

    /// Execute pattern recognition and factorization in parallel
    pub fn execute_batch(&self, numbers: &[Number], patterns: &[Pattern]) -> Vec<Result<Factors>> {
        let start = Instant::now();
        let total = numbers.len();

        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_numbers = total;
        }

        // Configure thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.config.num_threads)
            .build()
            .unwrap();

        let results = pool.install(|| match self.config.distribution_strategy {
            DistributionStrategy::Static => self.execute_static(numbers, patterns),
            DistributionStrategy::Dynamic => self.execute_dynamic(numbers, patterns),
            DistributionStrategy::Adaptive => self.execute_adaptive(numbers, patterns),
        });

        // Update final stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_time = start.elapsed();
            if total > 0 {
                stats.average_time_per_number = stats.total_time / total as u32;
            }
        }

        results
    }

    /// Static distribution execution
    fn execute_static(&self, numbers: &[Number], patterns: &[Pattern]) -> Vec<Result<Factors>> {
        let chunk_size = (numbers.len() / self.config.num_threads).max(1);

        numbers
            .par_chunks(chunk_size)
            .flat_map(|chunk| {
                chunk
                    .iter()
                    .map(|n| self.execute_single_cached(n, patterns))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Dynamic distribution execution
    fn execute_dynamic(&self, numbers: &[Number], patterns: &[Pattern]) -> Vec<Result<Factors>> {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let processed = AtomicUsize::new(0);
        let total = numbers.len();

        numbers
            .par_iter()
            .map(|n| {
                let result = self.execute_single_cached(n, patterns);

                // Update progress
                if self.config.show_progress {
                    let count = processed.fetch_add(1, Ordering::Relaxed) + 1;
                    if count % self.config.batch_size == 0 || count == total {
                        println!(
                            "Progress: {}/{} ({:.1}%)",
                            count,
                            total,
                            count as f64 / total as f64 * 100.0
                        );
                    }
                }

                result
            })
            .collect()
    }

    /// Adaptive distribution based on number complexity
    fn execute_adaptive(&self, numbers: &[Number], patterns: &[Pattern]) -> Vec<Result<Factors>> {
        // Sort numbers by estimated complexity
        let mut indexed_numbers: Vec<(usize, &Number)> = numbers.iter().enumerate().collect();

        indexed_numbers.par_sort_unstable_by_key(|(_, n)| n.bit_length());

        // Process in complexity order with work stealing
        let mut results_with_indices: Vec<(usize, Result<Factors>)> = indexed_numbers
            .into_par_iter()
            .map(|(idx, n)| {
                let result = self.execute_single_cached(n, patterns);
                (idx, result)
            })
            .collect();

        // Sort back to original order
        results_with_indices.sort_by_key(|(idx, _)| *idx);

        // Extract results
        results_with_indices.into_iter().map(|(_, result)| result).collect()
    }

    /// Execute single number with caching
    fn execute_single_cached(&self, n: &Number, patterns: &[Pattern]) -> Result<Factors> {
        // Try cache first
        self.cache.get_or_compute_factors(n, || self.execute_single(n, patterns))
    }

    /// Execute single number without caching
    fn execute_single(&self, n: &Number, patterns: &[Pattern]) -> Result<Factors> {
        // Recognition with caching
        let recognition = self.cache.get_or_compute_recognition(n, || {
            crate::pattern::recognition::recognize(n.clone(), patterns)
        })?;

        // Formalization with caching
        let constants = crate::observer::ConstantDiscovery::extract(patterns);
        let formalization = self.cache.get_or_compute_formalization(&recognition, || {
            crate::pattern::formalization::formalize(recognition.clone(), patterns, &constants)
        })?;

        // Execution (not cached at this level, delegated to factors cache)
        let result = crate::pattern::execution::execute(formalization, patterns);

        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            match &result {
                Ok(_) => stats.successful += 1,
                Err(_) => stats.failed += 1,
            }
        }

        result
    }

    /// Get execution statistics
    pub fn stats(&self) -> ExecutionStats {
        let stats = self.stats.lock().unwrap();
        ExecutionStats {
            total_numbers: stats.total_numbers,
            successful: stats.successful,
            failed: stats.failed,
            cache_hits: stats.cache_hits,
            total_time: stats.total_time,
            average_time_per_number: stats.average_time_per_number,
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> crate::pattern::cache::CacheStats {
        self.cache.stats()
    }
}

/// Parallel pattern matching with early termination
pub fn parallel_pattern_match(
    n: &Number,
    patterns: &[Pattern],
    max_patterns: usize,
) -> Vec<Pattern> {
    use std::sync::atomic::{AtomicBool, Ordering};

    let found_enough = AtomicBool::new(false);
    let matches = Arc::new(Mutex::new(Vec::new()));

    patterns
        .par_iter()
        .filter(|pattern| {
            // Early termination check
            if found_enough.load(Ordering::Relaxed) {
                return false;
            }

            // Check if pattern applies
            if pattern.applies_to(n) {
                let mut m = matches.lock().unwrap();
                m.push((*pattern).clone());

                if m.len() >= max_patterns {
                    found_enough.store(true, Ordering::Relaxed);
                }
                true
            } else {
                false
            }
        })
        .collect::<Vec<_>>();

    let matches_vec = Arc::try_unwrap(matches).unwrap().into_inner().unwrap();
    matches_vec
}

/// Parallel search across quantum regions
pub fn parallel_quantum_search(
    formalization: &Formalization,
    _patterns: &[Pattern],
    num_regions: usize,
) -> Result<Factors> {
    use crate::types::quantum_enhanced::EnhancedQuantumRegion;
    use crate::utils;

    let n = &formalization.n;
    let sqrt_n = utils::integer_sqrt(n)?;

    // Create multiple quantum regions
    let regions: Vec<EnhancedQuantumRegion> = (0..num_regions)
        .map(|i| {
            let offset = Number::from((i * 1000) as u64);
            let center = &sqrt_n + &offset;
            EnhancedQuantumRegion::new(center, &sqrt_n / &Number::from(50u32), n)
        })
        .collect();

    // Search regions in parallel
    let result = regions.into_par_iter().find_map_any(|mut region| {
        for _ in 0..100 {
            let candidates = region.get_search_candidates(5);

            for candidate in candidates {
                if candidate > Number::from(1u32) && &candidate < n {
                    if n % &candidate == Number::from(0u32) {
                        let other = n / &candidate;
                        return Some(Factors::new(candidate, other, "parallel_quantum_search"));
                    }

                    // Update region
                    region.update(&candidate, false, None);
                }
            }
        }
        None
    });

    result.ok_or_else(|| {
        PatternError::ExecutionError("Parallel quantum search exhausted".to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_executor() {
        let config = ParallelConfig {
            num_threads: 2,
            batch_size: 10,
            show_progress: false,
            ..Default::default()
        };

        let executor = ParallelExecutor::new(config);

        let numbers = vec![
            Number::from(143u32), // 11 × 13
            Number::from(323u32), // 17 × 19
        ];

        let patterns = vec![]; // Empty patterns for test

        let results = executor.execute_batch(&numbers, &patterns);
        assert_eq!(results.len(), 2);

        let stats = executor.stats();
        assert_eq!(stats.total_numbers, 2);
    }
}
