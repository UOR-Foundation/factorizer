//! Error types for The Pattern implementation

use thiserror::Error;

/// Main error type for pattern operations
#[derive(Error, Debug)]
pub enum PatternError {
    /// Error in pattern recognition
    #[error("Recognition failed: {0}")]
    RecognitionError(String),

    /// Error in pattern formalization
    #[error("Formalization failed: {0}")]
    FormalizationError(String),

    /// Error in pattern execution
    #[error("Execution failed: {0}")]
    ExecutionError(String),

    /// Error in data collection
    #[error("Collection failed: {0}")]
    CollectionError(String),

    /// Error in pattern analysis
    #[error("Analysis failed: {0}")]
    AnalysisError(String),

    /// Error in constant discovery
    #[error("Constant discovery failed: {0}")]
    ConstantError(String),

    /// Arithmetic error
    #[error("Arithmetic error: {0}")]
    ArithmeticError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Pattern not found
    #[error("Pattern not found for: {0}")]
    PatternNotFound(String),

    /// Insufficient data
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
}

/// Result type alias for pattern operations
pub type Result<T> = std::result::Result<T, PatternError>;