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
    compute_resonance_with_position(bit_pattern, 0, 1, params)
}

/// Compute position-aware resonance for a channel
/// 
/// Takes into account:
/// - channel_pos: absolute position of this channel (0 = LSB)
/// - total_channels: total number of channels in the number
/// - Different resonance patterns based on position
pub fn compute_resonance_with_position(
    bit_pattern: u8, 
    channel_pos: usize,
    total_channels: usize,
    params: &TunerParams
) -> ResonanceTuple {
    let constants = Constants::all();
    let scale = BigInt::one() << RESONANCE_PRECISION;
    
    // Initialize accumulation variables
    let mut primary_resonance = scale.clone();
    let mut harmonic_signature: u64 = 0;
    let mut phase_offset = BigInt::zero();
    
    // Calculate position-based modifiers
    let position_factor = calculate_position_factor(channel_pos, total_channels);
    let _distance_from_lsb = channel_pos;
    let distance_from_msb = total_channels.saturating_sub(channel_pos + 1);
    
    // Process each bit position
    for (bit_pos, constant) in constants.iter().enumerate() {
        if constant.is_active(bit_pattern) {
            // Apply weight scaling with position awareness
            let base_weight = params.constant_weights[bit_pos] as u32;
            
            // Adjust weight based on channel position
            let position_adjusted_weight = adjust_weight_for_position(
                base_weight,
                bit_pos,
                channel_pos,
                total_channels
            );
            
            // Primary resonance: multiply by position-adjusted weighted numerator
            let weighted_num = &constant.numerator * position_adjusted_weight;
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
            
            // Phase offset: based on bit position and channel position
            // Channels further from LSB have larger phase offsets
            let channel_phase_contrib = BigInt::from(channel_pos * 8);
            let bit_phase_contrib = BigInt::from(bit_pos + 1) * &constant.denominator / BigInt::from(256);
            phase_offset += bit_phase_contrib + channel_phase_contrib;
        }
    }
    
    // Apply resonance scaling shift with position adjustment
    let position_scaling = if channel_pos == 0 {
        // LSB channel: less scaling for direct factor detection
        params.resonance_scaling_shift.saturating_sub(4)
    } else if distance_from_msb == 0 {
        // MSB channel: more scaling to prevent overflow
        params.resonance_scaling_shift + 4
    } else {
        params.resonance_scaling_shift
    };
    primary_resonance >>= position_scaling;
    
    // Apply position factor to primary resonance
    primary_resonance = primary_resonance * BigInt::from(position_factor.0) / BigInt::from(position_factor.1);
    
    ResonanceTuple::new(primary_resonance, harmonic_signature, phase_offset)
}

/// Calculate position-based scaling factor
fn calculate_position_factor(channel_pos: usize, total_channels: usize) -> (u32, u32) {
    if total_channels <= 1 {
        return (1, 1);
    }
    
    // Channels closer to LSB get higher weight (factors often appear there)
    let distance_from_lsb = channel_pos;
    let max_distance = total_channels - 1;
    
    if distance_from_lsb == 0 {
        (3, 2)  // 1.5x for LSB
    } else if distance_from_lsb == 1 {
        (5, 4)  // 1.25x for second channel
    } else if distance_from_lsb >= max_distance {
        (2, 3)  // 0.67x for MSB
    } else {
        (1, 1)  // 1.0x for middle channels
    }
}

/// Adjust constant weight based on channel position
fn adjust_weight_for_position(
    base_weight: u32,
    bit_pos: usize,
    channel_pos: usize,
    total_channels: usize,
) -> u32 {
    // Constants have different importance at different positions
    match bit_pos {
        0 => {
            // Unity: more important in LSB channels
            if channel_pos == 0 {
                base_weight * 2
            } else {
                base_weight
            }
        }
        1 | 2 => {
            // Tau & Phi: important for middle channels
            if channel_pos > 0 && channel_pos < total_channels - 1 {
                (base_weight * 3) / 2
            } else {
                base_weight
            }
        }
        7 => {
            // Alpha: more important for higher channels
            if channel_pos > total_channels / 2 {
                (base_weight * 3) / 2
            } else {
                base_weight
            }
        }
        _ => base_weight,
    }
}

/// Compute peak indices for a given resonance pattern
/// 
/// Peak indices indicate where factors are likely to appear based on
/// the resonance characteristics of the bit pattern.
fn compute_peak_indices(resonance: &ResonanceTuple, channel_pos: usize) -> Vec<usize> {
    let mut peaks = Vec::new();
    
    // Extract pattern from harmonic signature
    let pattern = (resonance.harmonic_signature >> (channel_pos % 8)) & 0xFF;
    
    // Channel position affects peak detection strategy
    if channel_pos == 0 {
        // LSB channel: Direct factor encoding is common
        if pattern != 0 {
            // Primary peak at pattern value (often factor % 256)
            peaks.push(pattern as usize);
            
            // For small factors, check multiples of pattern
            if pattern < 128 {
                peaks.push((pattern * 2) as usize);
            }
            
            // Check GCD-related values
            for divisor in [2, 3, 5, 7, 11, 13].iter() {
                if pattern % divisor == 0 {
                    peaks.push((pattern / divisor) as usize);
                }
            }
        }
    } else {
        // Higher channels: Use standard peak detection
        if pattern != 0 {
            // Primary peak at pattern value
            peaks.push(pattern as usize);
            
            // Secondary peak at complement
            peaks.push((!pattern) as usize);
            
            // Position-adjusted peaks
            let position_shift = (channel_pos % 4) * 2;
            peaks.push(pattern.rotate_left(position_shift as u32) as usize);
            peaks.push(pattern.rotate_right(position_shift as u32) as usize);
            
            // Tertiary peaks at bit positions
            for bit in 0..8 {
                if (pattern >> bit) & 1 == 1 {
                    peaks.push(1 << bit);
                }
            }
        }
    }
    
    // Sort and deduplicate
    peaks.sort_unstable();
    peaks.dedup();
    
    // Limit peaks to prevent excessive computation
    peaks.truncate(16);
    
    peaks
}

/// Pre-compute all patterns for a single channel with position awareness
pub fn compute_channel_patterns(channel_pos: usize, params: &TunerParams) -> Channel {
    compute_channel_patterns_with_context(channel_pos, 1, params)
}

/// Pre-compute patterns with full position context
pub fn compute_channel_patterns_with_context(
    channel_pos: usize, 
    total_channels: usize,
    params: &TunerParams
) -> Channel {
    let mut channel = Channel::new(channel_pos);
    
    // Compute pattern for each possible 8-bit value
    for bit_value in 0..=255u8 {
        // Use position-aware resonance computation
        let resonance = compute_resonance_with_position(
            bit_value, 
            channel_pos,
            total_channels,
            params
        );
        
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
    
    // Compute patterns for each channel with position awareness
    for pos in 0..num_channels {
        let channel = compute_channel_patterns_with_context(pos, num_channels, params);
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
    
    #[test]
    fn test_position_aware_resonance() {
        let params = TunerParams::default();
        let pattern = 0b11001100;
        
        // Test different channel positions
        let res_lsb = compute_resonance_with_position(pattern, 0, 8, &params);
        let res_mid = compute_resonance_with_position(pattern, 4, 8, &params);
        let res_msb = compute_resonance_with_position(pattern, 7, 8, &params);
        
        // LSB should have different characteristics than MSB
        assert_ne!(res_lsb.primary_resonance, res_msb.primary_resonance);
        assert_ne!(res_lsb.phase_offset, res_msb.phase_offset);
        
        // Phase offsets should increase with position
        assert!(res_mid.phase_offset > res_lsb.phase_offset);
        assert!(res_msb.phase_offset > res_mid.phase_offset);
    }
    
    #[test]
    fn test_position_factor_calculation() {
        // LSB gets boost
        assert_eq!(calculate_position_factor(0, 8), (3, 2));
        
        // Second channel gets smaller boost
        assert_eq!(calculate_position_factor(1, 8), (5, 4));
        
        // Middle channels get no boost
        assert_eq!(calculate_position_factor(4, 8), (1, 1));
        
        // MSB gets reduction
        assert_eq!(calculate_position_factor(7, 8), (2, 3));
    }
    
    #[test]
    fn test_channel_specific_peaks() {
        let params = TunerParams::default();
        
        // Test LSB channel peak detection
        let res_lsb = compute_resonance_with_position(143, 0, 4, &params);
        let peaks_lsb = compute_peak_indices(&res_lsb, 0);
        
        // LSB should check for direct factor encoding
        assert!(peaks_lsb.contains(&143));
        
        // Test higher channel peak detection
        let res_high = compute_resonance_with_position(143, 2, 4, &params);
        let peaks_high = compute_peak_indices(&res_high, 2);
        
        // Should have different peak patterns
        assert_ne!(peaks_lsb, peaks_high);
    }
}