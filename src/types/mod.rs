//! Core type definitions for The Pattern
//!
//! These types emerge from empirical observation of factorization patterns.

pub mod number;
pub mod observation;
pub mod pattern;
pub mod quantum;
pub mod recognition;
pub mod signature;

// Re-export main types
pub use number::Number;
pub use observation::Observation;
pub use pattern::{Pattern, PatternKind, UniversalConstant};
pub use quantum::QuantumRegion;
pub use recognition::{Factors, Formalization, PatternType, Recognition};
pub use signature::PatternSignature;
