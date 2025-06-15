//! Core data structures for The Pattern implementation
//! 
//! All types use integer arithmetic only - no floating point operations.

use num_bigint::BigInt;
use num_traits::Zero;
use std::fmt;

/// Resonance tuple returned by the resonance function R(b)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResonanceTuple {
    /// Primary resonance: product of active constant numerators scaled by 2^256
    pub primary_resonance: BigInt,
    /// Harmonic signature: XOR accumulation of active constant patterns
    pub harmonic_signature: u64,
    /// Phase offset: sum of bit positions times constant denominators
    pub phase_offset: BigInt,
}

impl ResonanceTuple {
    /// Create a new resonance tuple
    pub fn new(primary_resonance: BigInt, harmonic_signature: u64, phase_offset: BigInt) -> Self {
        Self {
            primary_resonance,
            harmonic_signature,
            phase_offset,
        }
    }
    
    /// Check if this tuple aligns with another according to modular congruence rules
    pub fn aligns_with(&self, other: &Self, n: &BigInt) -> bool {
        // Primary resonances must be congruent mod N
        let congruent = (&self.primary_resonance - &other.primary_resonance) % n == BigInt::zero();
        
        // Phase offsets must differ by exactly 8 (channel width)
        let phase_diff = &other.phase_offset - &self.phase_offset;
        let phase_aligned = phase_diff == BigInt::from(8);
        
        congruent && phase_aligned
    }
}

/// Pattern stored for each bit combination in a channel
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pattern {
    /// The 8-bit mask indicating which constants are active
    pub bit_mask: u8,
    /// Pre-computed resonance tuple for this bit pattern
    pub resonance: ResonanceTuple,
    /// Indices where factors are likely to appear
    pub peak_indices: Vec<usize>,
}

impl Pattern {
    /// Create a new pattern
    pub fn new(bit_mask: u8, resonance: ResonanceTuple, peak_indices: Vec<usize>) -> Self {
        Self {
            bit_mask,
            resonance,
            peak_indices,
        }
    }
    
    /// Check if pattern has any peak indices
    pub fn has_peaks(&self) -> bool {
        !self.peak_indices.is_empty()
    }
}

/// Channel containing patterns for all 256 bit combinations
#[derive(Debug, Clone)]
pub struct Channel {
    /// Position of this channel (0-127 for 1024-bit numbers)
    pub position: usize,
    /// Pre-computed patterns for each possible 8-bit value
    pub patterns: Vec<Pattern>,
}

impl Channel {
    /// Create a new channel with 256 empty patterns
    pub fn new(position: usize) -> Self {
        let patterns = Vec::with_capacity(256);
        Self { position, patterns }
    }
    
    /// Get pattern for a specific 8-bit value
    pub fn get_pattern(&self, value: u8) -> Option<&Pattern> {
        self.patterns.get(value as usize)
    }
    
    /// Set pattern for a specific 8-bit value
    pub fn set_pattern(&mut self, value: u8, pattern: Pattern) {
        if self.patterns.len() <= value as usize {
            self.patterns.resize(value as usize + 1, Pattern::new(0, ResonanceTuple::new(BigInt::zero(), 0, BigInt::zero()), vec![]));
        }
        self.patterns[value as usize] = pattern;
    }
}

/// Complete basis containing all channels
#[derive(Debug, Clone)]
pub struct Basis {
    /// Number of channels (e.g., 128 for 1024-bit support)
    pub num_channels: usize,
    /// All channels with their pre-computed patterns
    pub channels: Vec<Channel>,
}

impl Basis {
    /// Create a new empty basis
    pub fn new(num_channels: usize) -> Self {
        let channels = (0..num_channels)
            .map(Channel::new)
            .collect();
        
        Self {
            num_channels,
            channels,
        }
    }
    
    /// Get a specific channel
    pub fn get_channel(&self, position: usize) -> Option<&Channel> {
        self.channels.get(position)
    }
    
    /// Get a mutable reference to a specific channel
    pub fn get_channel_mut(&mut self, position: usize) -> Option<&mut Channel> {
        self.channels.get_mut(position)
    }
}

/// Location where a peak was detected
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeakLocation {
    /// Starting channel index
    pub start_channel: usize,
    /// Ending channel index
    pub end_channel: usize,
    /// The bit pattern that aligned
    pub aligned_pattern: u8,
    /// Strength of the alignment (number of aligned channels)
    pub alignment_strength: usize,
}

impl PeakLocation {
    /// Create a new peak location
    pub fn new(start: usize, end: usize, pattern: u8) -> Self {
        Self {
            start_channel: start,
            end_channel: end,
            aligned_pattern: pattern,
            alignment_strength: end - start + 1,
        }
    }
    
    /// Get the channel span
    pub fn span(&self) -> usize {
        self.end_channel - self.start_channel + 1
    }
}

/// Result of factorization
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Factors {
    /// First factor (p)
    pub p: BigInt,
    /// Second factor (q)
    pub q: BigInt,
}

impl Factors {
    /// Create new factors ensuring p <= q
    pub fn new(p: BigInt, q: BigInt) -> Self {
        if p <= q {
            Self { p, q }
        } else {
            Self { p: q, q: p }
        }
    }
    
    /// Verify that p * q equals n
    pub fn verify(&self, n: &BigInt) -> bool {
        &self.p * &self.q == *n
    }
}

impl fmt::Display for Factors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} times {}", self.p, self.q)
    }
}

/// Tunable parameters for the auto-tuner
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TunerParams {
    /// Minimum channels needed for alignment (0-255)
    pub alignment_threshold: u8,
    /// Bit shift for resonance calculations (0-31)
    pub resonance_scaling_shift: u8,
    /// Expected step in harmonic signatures (1-256)
    pub harmonic_progression_step: u16,
    /// How many adjacent channels to check (0-7)
    pub phase_coupling_strength: u8,
    /// Relative contribution of each constant (0-255 each)
    pub constant_weights: [u8; 8],
}

impl Default for TunerParams {
    fn default() -> Self {
        Self {
            alignment_threshold: 3,
            resonance_scaling_shift: 16,
            harmonic_progression_step: 1,
            phase_coupling_strength: 3,
            constant_weights: [255; 8], // All constants equally weighted initially
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_resonance_tuple_alignment() {
        let n = BigInt::from(143);
        
        let tuple1 = ResonanceTuple::new(
            BigInt::from(100),
            0xABCD,
            BigInt::from(24),
        );
        
        let tuple2 = ResonanceTuple::new(
            BigInt::from(100) + &n, // Congruent mod n
            0xABCE,
            BigInt::from(32), // Phase offset differs by 8
        );
        
        assert!(tuple1.aligns_with(&tuple2, &n));
    }
    
    #[test]
    fn test_factors_ordering() {
        let f1 = Factors::new(BigInt::from(13), BigInt::from(11));
        assert_eq!(f1.p, BigInt::from(11));
        assert_eq!(f1.q, BigInt::from(13));
        
        let f2 = Factors::new(BigInt::from(11), BigInt::from(13));
        assert_eq!(f2.p, BigInt::from(11));
        assert_eq!(f2.q, BigInt::from(13));
    }
    
    #[test]
    fn test_peak_location() {
        let peak = PeakLocation::new(5, 8, 0b11010110);
        assert_eq!(peak.span(), 4);
        assert_eq!(peak.alignment_strength, 4);
    }
}