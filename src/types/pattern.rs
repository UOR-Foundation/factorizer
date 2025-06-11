//! Pattern types that emerge from observation
//!
//! These types represent the patterns discovered in factorization data.

use crate::types::Number;
use serde::{Deserialize, Serialize};

/// A discovered pattern in factorization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// Pattern identifier
    pub id: String,
    
    /// Pattern kind
    pub kind: PatternKind,
    
    /// How often this pattern appears
    pub frequency: f64,
    
    /// At which scales this pattern appears
    pub scale_range: ScaleRange,
    
    /// Pattern parameters
    pub parameters: Vec<f64>,
    
    /// Description of what was observed
    pub description: String,
}

/// Different kinds of patterns that emerge
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PatternKind {
    /// Invariant relationship (always true)
    Invariant,
    
    /// Scale-dependent pattern
    ScaleDependent,
    
    /// Type-specific pattern (balanced, harmonic, etc)
    TypeSpecific(String),
    
    /// Probabilistic pattern
    Probabilistic,
    
    /// Emergent pattern (unexpected discovery)
    Emergent,
}

/// Scale range where pattern appears
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleRange {
    /// Minimum bit length
    pub min_bits: usize,
    
    /// Maximum bit length
    pub max_bits: usize,
    
    /// Whether pattern continues beyond observed range
    pub unbounded: bool,
}

impl Pattern {
    /// Create a new pattern
    pub fn new(id: impl Into<String>, kind: PatternKind) -> Self {
        Pattern {
            id: id.into(),
            kind,
            frequency: 0.0,
            scale_range: ScaleRange {
                min_bits: 0,
                max_bits: 0,
                unbounded: false,
            },
            parameters: Vec::new(),
            description: String::new(),
        }
    }
    
    /// Check if pattern is universal (appears at all scales)
    pub fn is_universal(&self) -> bool {
        matches!(self.kind, PatternKind::Invariant) || 
        (self.scale_range.unbounded && self.frequency > 0.99)
    }
    
    /// Check if pattern applies to a given number
    pub fn applies_to(&self, n: &Number) -> bool {
        let bits = n.bit_length();
        bits >= self.scale_range.min_bits && 
        (self.scale_range.unbounded || bits <= self.scale_range.max_bits)
    }
    
    /// Describe the pattern
    pub fn describe(&self) -> String {
        format!(
            "{} ({:?}): {} (frequency: {:.2}%, {}-{} bits)",
            self.id,
            self.kind,
            self.description,
            self.frequency * 100.0,
            self.scale_range.min_bits,
            if self.scale_range.unbounded { 
                "∞".to_string() 
            } else { 
                self.scale_range.max_bits.to_string() 
            }
        )
    }
}

/// A mathematical relationship discovered in patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    /// Name of the relationship
    pub name: String,
    
    /// Mathematical expression
    pub expression: String,
    
    /// Variables involved
    pub variables: Vec<String>,
    
    /// Accuracy of the relationship
    pub accuracy: f64,
    
    /// Number of observations supporting this
    pub support_count: usize,
}

/// A discovered constant in patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalConstant {
    /// Constant name
    pub name: String,
    
    /// Constant value
    pub value: f64,
    
    /// Where this constant appears
    pub appearances: Vec<String>,
    
    /// How universal is this constant (0-1)
    pub universality: f64,
    
    /// Mathematical meaning if known
    pub meaning: Option<String>,
}

impl UniversalConstant {
    /// Check if constant appears in observations with given threshold
    pub fn appears_in_ratio(&self, threshold: f64) -> bool {
        self.universality >= threshold
    }
}