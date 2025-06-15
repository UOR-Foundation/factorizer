//! Basis computation and management for The Pattern
//!
//! Implements the resonance function R(b) and pre-computation of basis patterns.

use crate::{Constants, ResonanceTuple, Pattern, Channel, Basis, TunerParams};
use num_bigint::BigInt;
use num_traits::{One, Zero};

/// The precision for resonance calculations (256 bits)
const RESONANCE_PRECISION: u32 = 256;

/// Compute the resonance function R(b) for a given 8-bit value
/// 
/// Returns a tuple of (primary_resonance, harmonic_signature, phase_offset) where:
/// - primary_resonance: product of active constant numerators scaled by 2^256
/// - harmonic_signature: XOR accumulation of active constant patterns  
/// - phase_offset: sum of bit positions times constant denominators
pub fn compute_resonance(bit_pattern: u8, params: &TunerParams) -> ResonanceTuple {
    let constants = Constants::all();
    let scale = BigInt::one() << RESONANCE_PRECISION;
    
    // Initialize accumulation variables
    let mut primary_resonance = scale.clone();
    let mut harmonic_signature: u64 = 0;
    let mut phase_offset = BigInt::zero();
    
    // Process each bit position
    for (bit_pos, constant) in constants.iter().enumerate() {
        if constant.is_active(bit_pattern) {
            // Apply weight scaling
            let weight = params.constant_weights[bit_pos] as u32;
            
            // Primary resonance: multiply by weighted numerator
            let weighted_num = &constant.numerator * weight;
            primary_resonance = primary_resonance * weighted_num / &constant.denominator;
            
            // Harmonic signature: XOR with hash of numerator
            // Use a simple hash combining bit position and part of numerator
            let num_bytes = constant.numerator.to_bytes_be().1;
            let hash_input = if num_bytes.len() >= 8 {
                u64::from_be_bytes([
                    num_bytes[0], num_bytes[1], num_bytes[2], num_bytes[3],
                    num_bytes[4], num_bytes[5], num_bytes[6], num_bytes[7]
                ])
            } else {
                // For smaller numbers, pad with bit position
                (bit_pos as u64) << 56 | num_bytes.iter().fold(0u64, |acc, &b| (acc << 8) | b as u64)
            };
            harmonic_signature ^= hash_input.rotate_left(bit_pos as u32 * 8);
            
            // Phase offset: based on bit position
            phase_offset += BigInt::from(bit_pos + 1) * &constant.denominator / BigInt::from(256);
        }
    }
    
    // Apply resonance scaling shift
    primary_resonance >>= params.resonance_scaling_shift;
    
    ResonanceTuple::new(primary_resonance, harmonic_signature, phase_offset)
}

/// Compute peak indices for a given resonance pattern
/// 
/// Peak indices indicate where factors are likely to appear based on
/// the resonance characteristics of the bit pattern.
fn compute_peak_indices(resonance: &ResonanceTuple, channel_pos: usize) -> Vec<usize> {
    let mut peaks = Vec::new();
    
    // Extract pattern from harmonic signature
    let pattern = (resonance.harmonic_signature >> (channel_pos % 8)) & 0xFF;
    
    // Peaks occur at positions related to the pattern
    // This is empirically derived from observations
    if pattern != 0 {
        // Primary peak at pattern value
        peaks.push(pattern as usize);
        
        // Secondary peak at complement
        peaks.push((!pattern) as usize);
        
        // Tertiary peaks at bit positions
        for bit in 0..8 {
            if (pattern >> bit) & 1 == 1 {
                peaks.push(1 << bit);
            }
        }
    }
    
    // Sort and deduplicate
    peaks.sort_unstable();
    peaks.dedup();
    
    peaks
}

/// Pre-compute all patterns for a single channel
pub fn compute_channel_patterns(channel_pos: usize, params: &TunerParams) -> Channel {
    let mut channel = Channel::new(channel_pos);
    
    // Compute pattern for each possible 8-bit value
    for bit_value in 0..=255u8 {
        let mut resonance = compute_resonance(bit_value, params);
        
        // Adjust phase offset to incorporate channel position for proper alignment
        // This ensures consecutive channels have phase offsets that differ by 8
        resonance.phase_offset = resonance.phase_offset + BigInt::from(channel_pos * 8);
        
        let peak_indices = compute_peak_indices(&resonance, channel_pos);
        
        let pattern = Pattern::new(bit_value, resonance, peak_indices);
        channel.set_pattern(bit_value, pattern);
    }
    
    channel
}

/// Pre-compute the complete basis for all channels needed by a number
/// 
/// The number of channels is determined by the bit size of the input:
/// channels = ceil(bits/8)
pub fn compute_basis(n: &BigInt, params: &TunerParams) -> Basis {
    use crate::bit_size;
    
    // Calculate required channels based on input size
    let bits = bit_size(n);
    let num_channels = ((bits + 7) / 8) as usize;
    
    // Ensure at least 32 channels for small numbers (backward compatibility)
    // This provides headroom for resonance calculations
    let num_channels = num_channels.max(32);
    
    let mut basis = Basis::new(num_channels);
    
    // Compute patterns for each channel
    for pos in 0..num_channels {
        let channel = compute_channel_patterns(pos, params);
        basis.channels[pos] = channel;
    }
    
    basis
}

/// Verify that a basis is valid and complete
pub fn verify_basis(basis: &Basis) -> Result<(), String> {
    // Check number of channels
    if basis.channels.is_empty() {
        return Err("Basis has no channels".to_string());
    }
    
    // Verify each channel
    for (pos, channel) in basis.channels.iter().enumerate() {
        // Check channel position
        if channel.position != pos {
            return Err(format!("Channel {} has incorrect position {}", pos, channel.position));
        }
        
        // Check all 256 patterns exist
        if channel.patterns.len() != 256 {
            return Err(format!("Channel {} has {} patterns, expected 256", pos, channel.patterns.len()));
        }
        
        // Verify each pattern
        for (value, pattern) in channel.patterns.iter().enumerate() {
            if pattern.bit_mask != value as u8 {
                return Err(format!("Pattern at {} has incorrect bit mask {}", value, pattern.bit_mask));
            }
            
            // Check resonance is non-zero for non-zero patterns
            if value != 0 && pattern.resonance.primary_resonance == BigInt::zero() {
                return Err(format!("Non-zero pattern {} has zero resonance", value));
            }
        }
    }
    
    Ok(())
}

/// Serialize basis to binary format
pub fn serialize_basis(basis: &Basis) -> Vec<u8> {
    let mut data = Vec::new();
    
    // Header: "8BITPATT"
    data.extend_from_slice(b"8BITPATT");
    
    // Version: 1
    data.extend_from_slice(&1u32.to_be_bytes());
    
    // Number of channels
    data.extend_from_slice(&(basis.num_channels as u32).to_be_bytes());
    
    // Each channel
    for channel in &basis.channels {
        // Channel ID
        data.extend_from_slice(&(channel.position as u32).to_be_bytes());
        
        // Pattern count (always 256)
        data.extend_from_slice(&256u32.to_be_bytes());
        
        // Each pattern
        for pattern in &channel.patterns {
            // Bit mask
            data.push(pattern.bit_mask);
            
            // Resonance data (simplified for now - just store harmonic signature)
            data.extend_from_slice(&pattern.resonance.harmonic_signature.to_be_bytes());
            
            // Peak count
            data.extend_from_slice(&(pattern.peak_indices.len() as u32).to_be_bytes());
            
            // Peak indices
            for &peak in &pattern.peak_indices {
                data.extend_from_slice(&(peak as u32).to_be_bytes());
            }
        }
    }
    
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_compute_resonance_zero() {
        let params = TunerParams::default();
        let resonance = compute_resonance(0, &params);
        
        // With no bits set, primary resonance should be just the scale
        assert_eq!(resonance.harmonic_signature, 0);
        assert_eq!(resonance.phase_offset, BigInt::zero());
    }
    
    #[test]
    fn test_compute_resonance_unity_only() {
        let params = TunerParams::default();
        let resonance = compute_resonance(0b00000001, &params); // Only unity bit set
        
        // Should have non-zero values
        assert!(resonance.primary_resonance > BigInt::zero());
        assert!(resonance.harmonic_signature != 0);
        assert!(resonance.phase_offset > BigInt::zero());
    }
    
    #[test]
    fn test_compute_resonance_all_bits() {
        let params = TunerParams::default();
        let resonance = compute_resonance(0b11111111, &params); // All bits set
        
        // Should have large values
        assert!(resonance.primary_resonance > BigInt::zero());
        assert!(resonance.harmonic_signature != 0);
        assert!(resonance.phase_offset > BigInt::zero());
    }
    
    #[test]
    fn test_compute_channel_patterns() {
        let params = TunerParams::default();
        let channel = compute_channel_patterns(0, &params);
        
        assert_eq!(channel.position, 0);
        assert_eq!(channel.patterns.len(), 256);
        
        // Verify first and last patterns
        assert_eq!(channel.patterns[0].bit_mask, 0);
        assert_eq!(channel.patterns[255].bit_mask, 255);
    }
    
    #[test]
    fn test_compute_and_verify_basis() {
        let params = TunerParams::default();
        let n = BigInt::from(u32::MAX); // 32-bit number
        let basis = compute_basis(&n, &params);
        
        // Should have at least 32 channels (our minimum)
        assert!(basis.num_channels >= 32);
        assert_eq!(basis.channels.len(), basis.num_channels);
        
        // Verify the basis
        verify_basis(&basis).expect("Basis should be valid");
    }
    
    #[test]
    fn test_dynamic_basis_sizing() {
        let params = TunerParams::default();
        
        // Test various bit sizes
        let test_cases = vec![
            (BigInt::from(255u32), 32),        // 8-bit -> 32 channels (minimum)
            (BigInt::from(u16::MAX), 32),      // 16-bit -> 32 channels (minimum)
            (BigInt::from(u32::MAX), 32),      // 32-bit -> 32 channels
            (BigInt::from(u64::MAX), 32),      // 64-bit -> 32 channels (still within minimum)
            (BigInt::from(1u128) << 256, 33),  // 257-bit -> 33 channels
            (BigInt::from(1u128) << 512, 65),  // 513-bit -> 65 channels
        ];
        
        for (n, expected_min) in test_cases {
            let basis = compute_basis(&n, &params);
            assert!(basis.num_channels >= expected_min, 
                "For {:?}-bit number, expected at least {} channels, got {}", 
                crate::bit_size(&n), expected_min, basis.num_channels);
        }
    }
    
    #[test]
    fn test_peak_indices() {
        let params = TunerParams::default();
        let resonance = compute_resonance(0b10101010, &params);
        let peaks = compute_peak_indices(&resonance, 0);
        
        // Should have some peaks
        assert!(!peaks.is_empty());
        
        // Peaks should be sorted and unique
        for i in 1..peaks.len() {
            assert!(peaks[i] > peaks[i-1]);
        }
    }
}