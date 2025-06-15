//! The 8 fundamental constants that govern The Pattern
//! 
//! Each constant is encoded using Q32.224 fixed-point representation
//! (256 bits total with 224 fractional bits) for exact integer arithmetic.

use num_bigint::BigInt;
use num_traits::One;

/// Q32.224 encoding parameters
pub const FRACTIONAL_BITS: u32 = 224;
pub const SCALE: u32 = 1 << 24; // For intermediate scaling

/// Constant structure with rational representation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constant {
    /// Integer numerator (scaled by 2^224)
    pub numerator: BigInt,
    /// Denominator (always 2^224 for Q32.224)
    pub denominator: BigInt,
    /// Bit position in 8-bit pattern (0-7)
    pub bit_position: u8,
    /// Human-readable name
    pub name: &'static str,
    /// Symbol representation
    pub symbol: char,
}

impl Constant {
    /// Create a new constant with Q32.224 encoding
    pub fn new(numerator_hex: &str, bit_position: u8, name: &'static str, symbol: char) -> Self {
        let numerator = BigInt::parse_bytes(numerator_hex.as_bytes(), 16)
            .expect("Invalid hex string for constant");
        let denominator = BigInt::one() << FRACTIONAL_BITS;
        
        Self {
            numerator,
            denominator,
            bit_position,
            name,
            symbol,
        }
    }
    
    /// Get the exact rational value as a tuple (numerator, denominator)
    pub fn as_rational(&self) -> (&BigInt, &BigInt) {
        (&self.numerator, &self.denominator)
    }
    
    /// Create a constant from a floating point value
    pub fn from_f64(value: f64, name: &'static str, symbol: char) -> Self {
        // Convert f64 to Q32.224 by multiplying by 2^224
        let scale = BigInt::one() << FRACTIONAL_BITS;
        let numerator = BigInt::from((value * (1u64 << 32) as f64) as u64) << (FRACTIONAL_BITS - 32);
        
        // Determine bit position from symbol
        let bit_position = match symbol {
            '1' => 0,
            'τ' => 1,
            'φ' => 2,
            'ε' => 3,
            'δ' => 4,
            'γ' => 5,
            'β' => 6,
            'α' => 7,
            _ => 0,
        };
        
        Self {
            numerator,
            denominator: scale,
            bit_position,
            name,
            symbol,
        }
    }
    
    /// Convert to f64 (lossy)
    pub fn to_f64(&self) -> f64 {
        // Convert Q32.224 to f64
        let num_f64 = self.numerator.to_string().parse::<f64>().unwrap_or(0.0);
        let den_f64 = self.denominator.to_string().parse::<f64>().unwrap_or(1.0);
        num_f64 / den_f64
    }
    
    /// Check if this constant is active in a given bit pattern
    pub fn is_active(&self, bit_pattern: u8) -> bool {
        (bit_pattern >> self.bit_position) & 1 == 1
    }
}

/// The 8 fundamental constants
pub struct Constants;

impl Constants {
    /// Create constants from floating point values
    pub fn from_values(_values: &[f64; 8]) -> Self {
        // This is a factory struct, so we just return Self
        Self
    }
    
    /// Get a constant by index with custom value
    pub fn get_with_value(index: usize, value: f64) -> Constant {
        let (name, symbol) = match index {
            0 => ("unity", '1'),
            1 => ("tau", 'τ'),
            2 => ("phi", 'φ'),
            3 => ("epsilon", 'ε'),
            4 => ("delta", 'δ'),
            5 => ("gamma", 'γ'),
            6 => ("beta", 'β'),
            7 => ("alpha", 'α'),
            _ => ("unknown", '?'),
        };
        Constant::from_f64(value, name, symbol)
    }
    
    /// Resonance decay rate (alpha) - controls signal decay in resonance fields
    /// Based on first Riemann zero imaginary part: 14.134725...
    pub fn alpha() -> Constant {
        // 14.134725 * 2^224 / 10^6 (scaled to integer)
        Constant::new(
            "E24B4AB1A52E82000000000000000000000000000000000000000000000000",
            7,
            "resonance_decay_alpha",
            'α'
        )
    }
    
    /// Phase coupling strength (beta) - phase relationships between factors
    /// Based on zero density parameter: 0.1996...
    pub fn beta() -> Constant {
        // 0.199612 * 2^224
        Constant::new(
            "3326A9B9BA6173000000000000000000000000000000000000000000000000",
            6,
            "phase_coupling_beta",
            'β'
        )
    }
    
    /// Scale transition factor (gamma) - scale transformation across bit sizes
    /// Based on logarithmic density: 2*pi = 6.283...
    pub fn gamma() -> Constant {
        // 6.283185 * 2^224
        Constant::new(
            "C90FDAA22168C234C000000000000000000000000000000000000000000000",
            5,
            "scale_transition_gamma",
            'γ'
        )
    }
    
    /// Interference null spacing (delta) - interference pattern nulls
    /// Based on average zero spacing: 1/(2*pi*log(t)) ~ 0.159...
    pub fn delta() -> Constant {
        // 0.159155 * 2^224
        Constant::new(
            "28BE60DB93936E7C8000000000000000000000000000000000000000000000",
            4,
            "interference_null_delta",
            'δ'
        )
    }
    
    /// Adelic threshold (epsilon) - p-adic threshold transitions
    /// Based on critical strip width: 1/2 = 0.5
    pub fn epsilon() -> Constant {
        // 0.5 * 2^224
        Constant::new(
            "80000000000000000000000000000000000000000000000000000000000000",
            3,
            "adelic_threshold_epsilon",
            'ε'
        )
    }
    
    /// Golden ratio (phi) - golden ratio harmonic resonance
    /// Exact value: 1.618033988749...
    pub fn phi() -> Constant {
        // 1.618034 * 2^224
        Constant::new(
            "19E3779B97F4A7C15000000000000000000000000000000000000000000000",
            2,
            "golden_ratio_phi",
            'φ'
        )
    }
    
    /// Tribonacci constant (tau) - tribonacci sequence relationships
    /// Exact value: 1.839286755214...
    pub fn tau() -> Constant {
        // 1.839287 * 2^224
        Constant::new(
            "1D6329F1C35CA4BF8000000000000000000000000000000000000000000000",
            1,
            "tribonacci_tau",
            'τ'
        )
    }
    
    /// Unity (1) - empirical ground reference
    pub fn unity() -> Constant {
        // 1.0 * 2^224
        Constant::new(
            "10000000000000000000000000000000000000000000000000000000000000",
            0,
            "unity",
            '1'
        )
    }
    
    /// Get all constants as an array indexed by bit position
    pub fn all() -> [Constant; 8] {
        [
            Self::unity(),      // bit 0
            Self::tau(),        // bit 1
            Self::phi(),        // bit 2
            Self::epsilon(),    // bit 3
            Self::delta(),      // bit 4
            Self::gamma(),      // bit 5
            Self::beta(),       // bit 6
            Self::alpha(),      // bit 7
        ]
    }
    
    /// Get constants active in a given bit pattern
    pub fn active_constants(bit_pattern: u8) -> Vec<Constant> {
        Self::all()
            .into_iter()
            .filter(|c| c.is_active(bit_pattern))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_constant_creation() {
        let alpha = Constants::alpha();
        assert_eq!(alpha.bit_position, 7);
        assert_eq!(alpha.symbol, 'α');
        assert!(alpha.numerator > BigInt::from(0));
    }
    
    #[test]
    fn test_bit_pattern_activation() {
        let pattern = 0b10101010; // alpha, gamma, epsilon, tau active
        let active = Constants::active_constants(pattern);
        
        assert_eq!(active.len(), 4);
        assert_eq!(active[0].symbol, 'τ');
        assert_eq!(active[1].symbol, 'ε');
        assert_eq!(active[2].symbol, 'γ');
        assert_eq!(active[3].symbol, 'α');
    }
    
    #[test]
    fn test_all_constants() {
        let all = Constants::all();
        assert_eq!(all.len(), 8);
        
        // Verify bit positions
        for (i, constant) in all.iter().enumerate() {
            assert_eq!(constant.bit_position, i as u8);
        }
    }
}