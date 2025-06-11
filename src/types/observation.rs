//! Observation type for empirical data collection
//!
//! This structure emerges from the need to capture all potentially
//! relevant information about factorizations without presuming importance.

use crate::types::Number;
use serde::{Deserialize, Serialize};

/// Complete observation of a factorization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// The composite number
    pub n: Number,
    
    /// First prime factor (p ≤ q)
    pub p: Number,
    
    /// Second prime factor
    pub q: Number,
    
    /// Derived observations
    pub derived: DerivedObservations,
    
    /// Modular observations
    pub modular: ModularObservations,
    
    /// Harmonic observations
    pub harmonic: HarmonicObservations,
    
    /// Scale observations
    pub scale: ScaleObservations,
}

/// Observations derived from basic factorization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedObservations {
    /// Square root of n
    pub sqrt_n: Number,
    
    /// Fermat's a = (p + q) / 2
    pub fermat_a: Number,
    
    /// Fermat's b = |p - q| / 2
    pub fermat_b: Number,
    
    /// Offset from sqrt(n) to a
    pub offset: Number,
    
    /// Ratio of offset to sqrt(n)
    pub offset_ratio: f64,
}

/// Modular arithmetic observations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModularObservations {
    /// n mod first 30 primes
    pub modular_signature: Vec<u64>,
    
    /// p mod first 30 primes
    pub p_mod_signature: Vec<u64>,
    
    /// q mod first 30 primes
    pub q_mod_signature: Vec<u64>,
    
    /// Quadratic residues
    pub quadratic_residues: Vec<i8>,
}

/// Harmonic and phase observations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicObservations {
    /// Harmonic residues at different frequencies
    pub harmonic_residues: Vec<f64>,
    
    /// Phase relationships between p and q
    pub phase_relationships: Vec<f64>,
    
    /// Resonance strength at different scales
    pub resonance_strength: Vec<f64>,
}

/// Scale-related observations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleObservations {
    /// Bit length of n
    pub bit_length: usize,
    
    /// Decimal digit length
    pub digit_length: usize,
    
    /// Gap between p and q
    pub prime_gap: Number,
    
    /// Balance ratio max(p,q) / min(p,q)
    pub balance_ratio: f64,
    
    /// Pattern classification
    pub pattern_type: PatternClass,
}

/// Classification of factorization pattern
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum PatternClass {
    /// Balanced semiprime (p ≈ q)
    Balanced,
    
    /// Harmonic pattern (large ratio)
    Harmonic,
    
    /// Perfect square (p = q)
    Square,
    
    /// Small factor pattern
    SmallFactor,
    
    /// Other pattern
    Other,
}

impl Observation {
    /// Create a new observation from factorization
    pub fn new(n: Number, p: Number, q: Number) -> Self {
        // Ensure p ≤ q
        let (p, q) = if p <= q { (p, q) } else { (q, p) };
        
        // Compute derived observations
        let sqrt_n = n.sqrt();
        let fermat_a = (&p + &q) / &Number::from(2u32);
        let fermat_b = (&q - &p) / &Number::from(2u32);
        let offset = &fermat_a - &sqrt_n;
        let offset_ratio = offset.to_f64().unwrap_or(0.0) / sqrt_n.to_f64().unwrap_or(1.0);
        
        let derived = DerivedObservations {
            sqrt_n,
            fermat_a,
            fermat_b,
            offset,
            offset_ratio,
        };
        
        // Compute modular observations
        let primes = crate::utils::generate_primes(30);
        let modular_signature: Vec<u64> = primes.iter()
            .map(|prime| (&n % prime).as_integer().to_u64().unwrap_or(0))
            .collect();
        let p_mod_signature: Vec<u64> = primes.iter()
            .map(|prime| (&p % prime).as_integer().to_u64().unwrap_or(0))
            .collect();
        let q_mod_signature: Vec<u64> = primes.iter()
            .map(|prime| (&q % prime).as_integer().to_u64().unwrap_or(0))
            .collect();
        
        // Quadratic residues
        let quadratic_residues: Vec<i8> = primes.iter().take(20)
            .map(|prime| {
                let r = &n % prime;
                if r.is_zero() {
                    0
                } else {
                    // Simple quadratic residue check
                    let exp = (prime - &Number::from(1u32)) / &Number::from(2u32);
                    let result = r.pow(exp.as_integer().to_u32().unwrap_or(0));
                    if result.is_one() {
                        1
                    } else {
                        -1
                    }
                }
            })
            .collect();
        
        let modular = ModularObservations {
            modular_signature,
            p_mod_signature,
            q_mod_signature,
            quadratic_residues,
        };
        
        // Compute harmonic observations (simplified for now)
        let harmonic = HarmonicObservations {
            harmonic_residues: vec![0.0; 10],
            phase_relationships: vec![0.0; 10],
            resonance_strength: vec![0.0; 10],
        };
        
        // Compute scale observations
        let bit_length = n.bit_length();
        let digit_length = n.digit_length();
        let prime_gap = &q - &p;
        let balance_ratio = q.to_f64().unwrap_or(1.0) / p.to_f64().unwrap_or(1.0);
        
        let pattern_type = if p == q {
            PatternClass::Square
        } else if balance_ratio < 1.1 {
            PatternClass::Balanced
        } else if balance_ratio > 10.0 {
            PatternClass::Harmonic
        } else if p < Number::from(1000u32) {
            PatternClass::SmallFactor
        } else {
            PatternClass::Other
        };
        
        let scale = ScaleObservations {
            bit_length,
            digit_length,
            prime_gap,
            balance_ratio,
            pattern_type,
        };
        
        Observation {
            n,
            p,
            q,
            derived,
            modular,
            harmonic,
            scale,
        }
    }
    
    /// Check if this observation matches a pattern
    pub fn matches_pattern(&self, other: &Observation) -> f64 {
        // Simple pattern matching based on modular signatures
        let mut similarity = 0.0;
        let mut count = 0.0;
        
        for (a, b) in self.modular.modular_signature.iter()
            .zip(&other.modular.modular_signature) {
            if a == b {
                similarity += 1.0;
            }
            count += 1.0;
        }
        
        similarity / count
    }
}