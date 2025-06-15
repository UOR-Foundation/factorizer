//! The 8-Bit Pattern Implementation
//! 
//! A standalone implementation of The Pattern focusing on the 8 fundamental
//! constants and their channel-based architecture for constant-time integer
//! factorization.

pub mod constants;
pub mod types;
pub mod channel;
pub mod basis;
pub mod pattern;
pub mod tuner;
pub mod diagnostics;
pub mod tuning;
pub mod resonance_extraction;
pub mod ensemble;
pub mod special_cases;
pub mod parallel;
pub mod coupling;

// Public API exports
pub use constants::{Constants, Constant, FRACTIONAL_BITS};
pub use types::{
    ResonanceTuple, Pattern, Channel, Basis, PeakLocation, 
    Factors, TunerParams
};
pub use channel::{
    decompose, reconstruct, extract_channel_range, 
    channels_for_bits, bit_size, coupling_strength, 
    is_harmonic_progression, ChannelWindow
};
pub use basis::{
    compute_resonance, compute_resonance_with_position, 
    compute_channel_patterns, compute_channel_patterns_with_context,
    compute_basis, verify_basis, serialize_basis
};
pub use pattern::{
    detect_aligned_channels, extract_factors, recognize_factors,
    recognize_factors_with_diagnostics
};
pub use tuner::{AutoTuner, TestCase, TuningResult};
pub use diagnostics::{
    ChannelDiagnostic, FactorizationDiagnostics, DiagnosticSummary, DiagnosticAggregator
};
pub use tuning::{
    TuningConfig, ConstantSet, ConstantTuner, ConstantTuningResult, PatternAnalysis
};
pub use resonance_extraction::{
    extract_factors_resonance, recognize_factors_advanced
};
pub use ensemble::{
    EnsembleVoter, VotingStrategy, EnsembleConstantSet,
    EnsembleStats
};
pub use special_cases::{
    detect_special_cases, try_special_cases, SpecialCase, SpecialCaseResult
};
pub use parallel::{
    decompose_parallel, compute_resonances_parallel, detect_peaks_parallel,
    extract_factors_parallel, recognize_factors_parallel
};
pub use coupling::{
    CouplingMatrix, CoupledChannelPair, apply_channel_coupling,
    detect_coupled_patterns, extract_factor_from_coupled_pair
};