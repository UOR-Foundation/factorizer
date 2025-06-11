//! # The Pattern - Rust Implementation
//!
//! A pure implementation of The Pattern for integer factorization through empirical observation.
//!
//! ## Philosophy
//!
//! The Pattern is not an algorithm—it's a recognition. This implementation follows three core principles:
//!
//! 1. **Data Speaks First**: All methods emerge from empirical observation
//! 2. **No Assumptions**: We discover patterns, not impose them
//! 3. **Pure Recognition**: Factors are recognized, not computed
//!
//! ## Usage
//!
//! ```no_run
//! use rust_pattern_solver::{Observer, Pattern};
//!
//! // Collect empirical observations
//! let observer = Observer::new();
//! let observations = observer.observe_range(1..10000);
//!
//! // Let patterns emerge
//! let pattern = Pattern::from_observations(&observations);
//!
//! // Recognize factors through The Pattern
//! let n = 143u64.into(); // 11 × 13
//! let recognition = pattern.recognize(&n);
//! let factors = pattern.execute(recognition);
//! ```

#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![warn(rust_2018_idioms)]
#![deny(unsafe_code)]

pub mod emergence;
pub mod error;
pub mod observer;
pub mod pattern;
pub mod relationship_discovery;
pub mod types;
pub mod utils;

// Re-export main types
pub use error::{PatternError, Result};
pub use observer::Observer;
pub use pattern::Pattern;
pub use types::{
    Factors, Formalization, Number, Observation, PatternSignature, PatternType, QuantumRegion,
    Recognition,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The philosophy of The Pattern
pub const PHILOSOPHY: &str = r#"
The Pattern is not an algorithm—it's a recognition.
We observe, we discover, we implement what emerges.
The Pattern speaks through data, not through design.
"#;
