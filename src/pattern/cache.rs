//! Pattern caching for improved performance
//!
//! This module implements caching strategies for discovered patterns,
//! recognition results, and factorization outcomes to avoid redundant computations.

use crate::types::{Factors, Formalization, Number, Pattern, Recognition};
use crate::Result;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in memory
    pub max_entries: usize,

    /// Time-to-live for cache entries
    pub ttl: Duration,

    /// Enable persistent cache to disk
    pub persistent: bool,

    /// Path for persistent cache
    pub cache_path: Option<String>,

    /// Cache hit/miss statistics
    pub enable_stats: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        CacheConfig {
            max_entries: 10000,
            ttl: Duration::from_secs(3600), // 1 hour
            persistent: false,
            cache_path: None,
            enable_stats: true,
        }
    }
}

/// Cached entry with metadata
#[derive(Debug, Clone)]
struct CachedEntry<T> {
    value: T,
    created_at: Instant,
    access_count: usize,
    last_accessed: Instant,
}

/// Pattern cache statistics
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub total_time_saved: Duration,
}

impl CacheStats {
    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Main pattern cache structure
#[derive(Debug)]
pub struct PatternCache {
    /// Recognition cache
    recognition_cache: Arc<RwLock<HashMap<String, CachedEntry<Recognition>>>>,

    /// Formalization cache
    formalization_cache: Arc<RwLock<HashMap<String, CachedEntry<Formalization>>>>,

    /// Factors cache
    factors_cache: Arc<RwLock<HashMap<String, CachedEntry<Factors>>>>,

    /// Pattern similarity cache
    similarity_cache: Arc<RwLock<HashMap<(String, String), f64>>>,

    /// Configuration
    config: CacheConfig,

    /// Statistics
    stats: Arc<RwLock<CacheStats>>,
}

impl PatternCache {
    /// Create new cache with configuration
    pub fn new(config: CacheConfig) -> Self {
        PatternCache {
            recognition_cache: Arc::new(RwLock::new(HashMap::new())),
            formalization_cache: Arc::new(RwLock::new(HashMap::new())),
            factors_cache: Arc::new(RwLock::new(HashMap::new())),
            similarity_cache: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// Generate cache key for a number
    fn cache_key(n: &Number) -> String {
        // Use a hash of the number for efficient lookup
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        n.to_string().hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Get or compute recognition
    pub fn get_or_compute_recognition<F>(&self, n: &Number, compute: F) -> Result<Recognition>
    where
        F: FnOnce() -> Result<Recognition>,
    {
        let key = Self::cache_key(n);

        // Try to get from cache
        {
            let mut cache = self.recognition_cache.write().unwrap();
            if let Some(entry) = cache.get_mut(&key) {
                if entry.created_at.elapsed() < self.config.ttl {
                    entry.access_count += 1;
                    entry.last_accessed = Instant::now();

                    if self.config.enable_stats {
                        let mut stats = self.stats.write().unwrap();
                        stats.hits += 1;
                    }

                    return Ok(entry.value.clone());
                }
            }
        }

        // Cache miss - compute value
        if self.config.enable_stats {
            let mut stats = self.stats.write().unwrap();
            stats.misses += 1;
        }

        let start = Instant::now();
        let result = compute()?;
        let computation_time = start.elapsed();

        // Store in cache
        {
            let mut cache = self.recognition_cache.write().unwrap();

            // Evict old entries if needed
            if cache.len() >= self.config.max_entries {
                self.evict_lru(&mut cache);
            }

            cache.insert(
                key,
                CachedEntry {
                    value: result.clone(),
                    created_at: Instant::now(),
                    access_count: 1,
                    last_accessed: Instant::now(),
                },
            );
        }

        if self.config.enable_stats {
            let mut stats = self.stats.write().unwrap();
            stats.total_time_saved = stats.total_time_saved.saturating_add(computation_time);
        }

        Ok(result)
    }

    /// Get or compute formalization
    pub fn get_or_compute_formalization<F>(
        &self,
        recognition: &Recognition,
        compute: F,
    ) -> Result<Formalization>
    where
        F: FnOnce() -> Result<Formalization>,
    {
        let key = format!("{:?}", recognition.pattern_type);

        // Similar caching logic for formalization
        {
            let mut cache = self.formalization_cache.write().unwrap();
            if let Some(entry) = cache.get_mut(&key) {
                if entry.created_at.elapsed() < self.config.ttl {
                    entry.access_count += 1;
                    entry.last_accessed = Instant::now();

                    if self.config.enable_stats {
                        let mut stats = self.stats.write().unwrap();
                        stats.hits += 1;
                    }

                    return Ok(entry.value.clone());
                }
            }
        }

        let result = compute()?;

        {
            let mut cache = self.formalization_cache.write().unwrap();
            if cache.len() >= self.config.max_entries {
                self.evict_lru(&mut cache);
            }

            cache.insert(
                key,
                CachedEntry {
                    value: result.clone(),
                    created_at: Instant::now(),
                    access_count: 1,
                    last_accessed: Instant::now(),
                },
            );
        }

        Ok(result)
    }

    /// Get or compute factors
    pub fn get_or_compute_factors<F>(&self, n: &Number, compute: F) -> Result<Factors>
    where
        F: FnOnce() -> Result<Factors>,
    {
        let key = Self::cache_key(n);

        // Check cache first
        {
            let mut cache = self.factors_cache.write().unwrap();
            if let Some(entry) = cache.get_mut(&key) {
                if entry.created_at.elapsed() < self.config.ttl {
                    entry.access_count += 1;
                    entry.last_accessed = Instant::now();

                    if self.config.enable_stats {
                        let mut stats = self.stats.write().unwrap();
                        stats.hits += 1;
                    }

                    return Ok(entry.value.clone());
                }
            }
        }

        // Compute factors
        let result = compute()?;

        // Cache the result
        {
            let mut cache = self.factors_cache.write().unwrap();
            if cache.len() >= self.config.max_entries {
                self.evict_lru(&mut cache);
            }

            cache.insert(
                key,
                CachedEntry {
                    value: result.clone(),
                    created_at: Instant::now(),
                    access_count: 1,
                    last_accessed: Instant::now(),
                },
            );
        }

        Ok(result)
    }

    /// Cache pattern similarity
    pub fn cache_similarity(&self, pattern1_id: &str, pattern2_id: &str, similarity: f64) {
        let mut cache = self.similarity_cache.write().unwrap();
        let key = if pattern1_id < pattern2_id {
            (pattern1_id.to_string(), pattern2_id.to_string())
        } else {
            (pattern2_id.to_string(), pattern1_id.to_string())
        };
        cache.insert(key, similarity);
    }

    /// Get cached similarity
    pub fn get_similarity(&self, pattern1_id: &str, pattern2_id: &str) -> Option<f64> {
        let cache = self.similarity_cache.read().unwrap();
        let key = if pattern1_id < pattern2_id {
            (pattern1_id.to_string(), pattern2_id.to_string())
        } else {
            (pattern2_id.to_string(), pattern1_id.to_string())
        };
        cache.get(&key).copied()
    }

    /// Evict least recently used entry
    fn evict_lru<T>(&self, cache: &mut HashMap<String, CachedEntry<T>>) {
        if let Some((key, _)) = cache.iter().min_by_key(|(_, entry)| entry.last_accessed) {
            let key = key.clone();
            cache.remove(&key);

            if self.config.enable_stats {
                let mut stats = self.stats.write().unwrap();
                stats.evictions += 1;
            }
        }
    }

    /// Clear all caches
    pub fn clear(&self) {
        self.recognition_cache.write().unwrap().clear();
        self.formalization_cache.write().unwrap().clear();
        self.factors_cache.write().unwrap().clear();
        self.similarity_cache.write().unwrap().clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let stats = self.stats.read().unwrap();
        CacheStats {
            hits: stats.hits,
            misses: stats.misses,
            evictions: stats.evictions,
            total_time_saved: stats.total_time_saved,
        }
    }

    /// Save cache to disk (if persistent)
    pub fn save_to_disk(&self) -> Result<()> {
        if !self.config.persistent || self.config.cache_path.is_none() {
            return Ok(());
        }

        // TODO: Implement disk persistence
        Ok(())
    }

    /// Load cache from disk (if persistent)
    pub fn load_from_disk(&mut self) -> Result<()> {
        if !self.config.persistent || self.config.cache_path.is_none() {
            return Ok(());
        }

        // TODO: Implement disk loading
        Ok(())
    }
}

lazy_static::lazy_static! {
    /// Global cache instance
    static ref GLOBAL_CACHE: PatternCache = PatternCache::new(CacheConfig::default());
}

/// Get global cache instance
pub fn global_cache() -> &'static PatternCache {
    &GLOBAL_CACHE
}

/// Parallel pattern search with caching
pub fn parallel_pattern_search(
    numbers: &[Number],
    patterns: &[Pattern],
    max_workers: usize,
) -> Vec<Result<Factors>> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let processed = AtomicUsize::new(0);
    let total = numbers.len();

    // Configure thread pool
    let pool = rayon::ThreadPoolBuilder::new().num_threads(max_workers).build().unwrap();

    pool.install(|| {
        numbers
            .par_iter()
            .map(|n| {
                // Use cache for each number
                let result = global_cache().get_or_compute_factors(n, || {
                    // Recognition
                    let recognition = crate::pattern::recognition::recognize(n.clone(), patterns)?;

                    // Formalization
                    let constants = crate::observer::ConstantDiscovery::extract(patterns);
                    let formalization = crate::pattern::formalization::formalize(
                        recognition,
                        patterns,
                        &constants,
                    )?;

                    // Execution
                    crate::pattern::execution::execute(formalization, patterns)
                });

                // Update progress
                let count = processed.fetch_add(1, Ordering::Relaxed) + 1;
                if count % 100 == 0 {
                    println!(
                        "Progress: {}/{} ({:.1}%)",
                        count,
                        total,
                        count as f64 / total as f64 * 100.0
                    );
                }

                result
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic() {
        let cache = PatternCache::new(CacheConfig::default());
        let n = Number::from(143u32);

        // First call should miss
        let result1 = cache
            .get_or_compute_recognition(&n, || {
                Ok(Recognition::new(
                    crate::types::PatternSignature::new(n.clone()),
                    crate::types::PatternType::SmallFactor,
                ))
            })
            .unwrap();

        // Second call should hit
        let result2 = cache
            .get_or_compute_recognition(&n, || {
                panic!("Should not compute again");
            })
            .unwrap();

        assert_eq!(result1.pattern_type, result2.pattern_type);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }
}
