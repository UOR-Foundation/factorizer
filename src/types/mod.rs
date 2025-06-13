//! Core type definitions for The Pattern
//!
//! These types emerge from empirical observation of factorization patterns.

pub mod number;
pub mod observation;
pub mod pattern;
pub mod quantum;
pub mod quantum_enhanced;
pub mod recognition;
pub mod signature;
pub mod rational;
pub mod constants;

// Re-export main types
pub use number::Number;
pub use observation::Observation;
pub use pattern::{Pattern, PatternKind, UniversalConstant};
pub use quantum::QuantumRegion;
pub use recognition::{Factors, Formalization, PatternType, Recognition};
pub use signature::PatternSignature;
pub use rational::Rational;
pub use constants::{get_constant, ConstantType, FundamentalConstantsRational, integer_sqrt};
