//! Performance optimization module
//!
//! This module contains optimized implementations of critical algorithms
//! and memory-efficient data structures for The Pattern.

pub mod arithmetic;
pub mod memory;
pub mod simd;

use crate::types::Number;
use crate::Result;

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Enable SIMD operations where available
    pub enable_simd: bool,

    /// Use memory pooling for temporary allocations
    pub use_memory_pool: bool,

    /// Cache size for arithmetic operations
    pub arithmetic_cache_size: usize,

    /// Enable profile-guided optimization hints
    pub enable_pgo: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        OptimizationConfig {
            enable_simd: cfg!(target_feature = "avx2") || cfg!(target_feature = "neon"),
            use_memory_pool: true,
            arithmetic_cache_size: 1024,
            enable_pgo: false,
        }
    }
}

/// Global optimization configuration
static mut OPTIMIZATION_CONFIG: Option<OptimizationConfig> = None;
static INIT: std::sync::Once = std::sync::Once::new();

/// Initialize optimization subsystem
pub fn initialize(config: OptimizationConfig) {
    unsafe {
        INIT.call_once(|| {
            OPTIMIZATION_CONFIG = Some(config);
        });
    }
}

/// Get current optimization configuration
pub fn config() -> &'static OptimizationConfig {
    unsafe { OPTIMIZATION_CONFIG.as_ref().unwrap_or(&DEFAULT_CONFIG) }
}

const DEFAULT_CONFIG: OptimizationConfig = OptimizationConfig {
    enable_simd: false,
    use_memory_pool: true,
    arithmetic_cache_size: 1024,
    enable_pgo: false,
};

/// Optimized modular arithmetic
#[inline(always)]
pub fn fast_mod(n: &Number, m: u64) -> u64 {
    if config().enable_simd {
        arithmetic::simd_mod(n, m)
    } else {
        arithmetic::standard_mod(n, m)
    }
}

/// Optimized GCD computation
#[inline(always)]
pub fn fast_gcd(a: &Number, b: &Number) -> Number {
    arithmetic::binary_gcd(a, b)
}

/// Optimized integer square root
#[inline(always)]
pub fn fast_isqrt(n: &Number) -> Result<Number> {
    arithmetic::newton_isqrt(n)
}
