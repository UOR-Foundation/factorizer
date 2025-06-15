//! Channel coupling implementation for multi-channel coordination
//!
//! Implements 2×2 coupling matrices to model interactions between adjacent channels.

use crate::{ResonanceTuple, TunerParams};
use num_bigint::BigInt;
use num_traits::{One, Zero, ToPrimitive};
use num_integer::Integer;

/// 2×2 coupling matrix for adjacent channel interactions
#[derive(Debug, Clone)]
pub struct CouplingMatrix {
    /// Coupling strength from channel i to channel i
    pub a11: f64,
    /// Coupling strength from channel i+1 to channel i
    pub a12: f64,
    /// Coupling strength from channel i to channel i+1
    pub a21: f64,
    /// Coupling strength from channel i+1 to channel i+1
    pub a22: f64,
}

impl CouplingMatrix {
    /// Create identity coupling (no interaction)
    pub fn identity() -> Self {
        Self {
            a11: 1.0,
            a12: 0.0,
            a21: 0.0,
            a22: 1.0,
        }
    }
    
    /// Create standard coupling matrix for adjacent channels
    pub fn standard() -> Self {
        Self {
            a11: 0.9,   // Self-coupling (conservative)
            a12: 0.1,   // Influence from next channel
            a21: 0.1,   // Influence to next channel
            a22: 0.9,   // Self-coupling (conservative)
        }
    }
    
    /// Create strong coupling for highly correlated channels
    pub fn strong() -> Self {
        Self {
            a11: 0.7,
            a12: 0.3,
            a21: 0.3,
            a22: 0.7,
        }
    }
    
    /// Create position-dependent coupling
    pub fn for_position(channel_pos: usize, total_channels: usize) -> Self {
        if channel_pos == 0 {
            // LSB has moderate forward coupling (carries)
            Self {
                a11: 0.92,
                a12: 0.08,
                a21: 0.15,
                a22: 0.85,
            }
        } else if channel_pos >= total_channels - 2 {
            // MSB has weaker coupling
            Self {
                a11: 0.95,
                a12: 0.05,
                a21: 0.05,
                a22: 0.95,
            }
        } else {
            // Middle channels use standard coupling
            Self::standard()
        }
    }
}

/// Channel pair with coupling information
#[derive(Debug, Clone)]
pub struct CoupledChannelPair {
    /// First channel index
    pub channel1_idx: usize,
    /// Second channel index
    pub channel2_idx: usize,
    /// Channel 1 value
    pub channel1_val: u8,
    /// Channel 2 value
    pub channel2_val: u8,
    /// Channel 1 resonance
    pub resonance1: ResonanceTuple,
    /// Channel 2 resonance
    pub resonance2: ResonanceTuple,
    /// Coupling matrix
    pub coupling: CouplingMatrix,
    /// Coupled resonance (after applying coupling)
    pub coupled_resonance: ResonanceTuple,
}

/// Apply coupling matrix to adjacent channel resonances
pub fn apply_channel_coupling(
    res1: &ResonanceTuple,
    res2: &ResonanceTuple,
    coupling: &CouplingMatrix,
) -> (ResonanceTuple, ResonanceTuple) {
    // Apply coupling to primary resonances
    let coupled_primary1 = apply_coupling_to_bigint(
        &res1.primary_resonance,
        &res2.primary_resonance,
        coupling.a11,
        coupling.a12,
    );
    
    let coupled_primary2 = apply_coupling_to_bigint(
        &res1.primary_resonance,
        &res2.primary_resonance,
        coupling.a21,
        coupling.a22,
    );
    
    // Apply coupling to harmonic signatures (bitwise operations)
    let coupled_harmonic1 = 
        ((res1.harmonic_signature as f64 * coupling.a11) as u64) ^
        ((res2.harmonic_signature as f64 * coupling.a12) as u64);
    
    let coupled_harmonic2 = 
        ((res1.harmonic_signature as f64 * coupling.a21) as u64) ^
        ((res2.harmonic_signature as f64 * coupling.a22) as u64);
    
    // Apply coupling to phase offsets
    let coupled_phase1 = apply_coupling_to_bigint(
        &res1.phase_offset,
        &res2.phase_offset,
        coupling.a11,
        coupling.a12,
    );
    
    let coupled_phase2 = apply_coupling_to_bigint(
        &res1.phase_offset,
        &res2.phase_offset,
        coupling.a21,
        coupling.a22,
    );
    
    (
        ResonanceTuple::new(coupled_primary1, coupled_harmonic1, coupled_phase1),
        ResonanceTuple::new(coupled_primary2, coupled_harmonic2, coupled_phase2),
    )
}

/// Apply coupling coefficients to BigInt values
fn apply_coupling_to_bigint(
    val1: &BigInt,
    val2: &BigInt,
    coeff1: f64,
    coeff2: f64,
) -> BigInt {
    // Convert to f64 for coupling calculation (with precision loss for very large numbers)
    let v1 = val1.to_f64().unwrap_or(1.0);
    let v2 = val2.to_f64().unwrap_or(1.0);
    
    let coupled = v1 * coeff1 + v2 * coeff2;
    
    // Convert back to BigInt
    if coupled >= 0.0 {
        BigInt::from(coupled as u64)
    } else {
        BigInt::zero()
    }
}

/// Detect coupled channel patterns in a window
pub fn detect_coupled_patterns(
    channels: &[(usize, u8, ResonanceTuple)],
    n: &BigInt,
    params: &TunerParams,
) -> Vec<CoupledChannelPair> {
    let mut coupled_pairs = Vec::new();
    
    // Process adjacent channel pairs
    for i in 0..channels.len().saturating_sub(1) {
        let (pos1, val1, res1) = &channels[i];
        let (pos2, val2, res2) = &channels[i + 1];
        
        // Check if channels are actually adjacent
        if pos2 - pos1 != 1 {
            continue;
        }
        
        // Create coupling matrix based on position
        let coupling = CouplingMatrix::for_position(*pos1, channels.len());
        
        // Apply coupling
        let (coupled1, coupled2) = apply_channel_coupling(res1, res2, &coupling);
        
        // Create combined resonance for the pair
        let combined_resonance = combine_coupled_resonances(&coupled1, &coupled2);
        
        // Check if the coupled resonance indicates a factor
        if is_coupled_factor_indicator(&combined_resonance, n, params) {
            coupled_pairs.push(CoupledChannelPair {
                channel1_idx: *pos1,
                channel2_idx: *pos2,
                channel1_val: *val1,
                channel2_val: *val2,
                resonance1: res1.clone(),
                resonance2: res2.clone(),
                coupling: coupling.clone(),
                coupled_resonance: combined_resonance,
            });
        }
    }
    
    coupled_pairs
}

/// Combine two coupled resonances into a single resonance
fn combine_coupled_resonances(res1: &ResonanceTuple, res2: &ResonanceTuple) -> ResonanceTuple {
    // Multiply primary resonances (they're already in product form)
    let combined_primary = &res1.primary_resonance * &res2.primary_resonance;
    
    // XOR harmonic signatures
    let combined_harmonic = res1.harmonic_signature ^ res2.harmonic_signature;
    
    // Add phase offsets
    let combined_phase = &res1.phase_offset + &res2.phase_offset;
    
    ResonanceTuple::new(combined_primary, combined_harmonic, combined_phase)
}

/// Check if a coupled resonance indicates a potential factor
fn is_coupled_factor_indicator(
    resonance: &ResonanceTuple,
    n: &BigInt,
    params: &TunerParams,
) -> bool {
    // Check if the resonance has factor-like properties
    // Be more conservative to avoid false positives
    
    // 1. Primary resonance should have GCD relationship with n
    let gcd = resonance.primary_resonance.gcd(n);
    if gcd > BigInt::one() && &gcd < n && n % &gcd == BigInt::zero() {
        // Extra verification: GCD must actually divide n
        return true;
    }
    
    // 2. Harmonic signature should match expected patterns
    let pattern_byte = (resonance.harmonic_signature & 0xFF) as u8;
    if pattern_byte > 1 {
        let pattern_bigint = BigInt::from(pattern_byte);
        let gcd_pattern = pattern_bigint.gcd(n);
        if gcd_pattern > BigInt::one() && &gcd_pattern < n && n % &gcd_pattern == BigInt::zero() {
            return true;
        }
    }
    
    // 3. Phase offset alignment (stricter threshold)
    let phase_mod = &resonance.phase_offset % n;
    let threshold = BigInt::from((params.alignment_threshold / 2).max(1) as u64);
    if phase_mod <= threshold {
        // Additional check: phase should relate to factor structure
        let phase_gcd = phase_mod.gcd(n);
        if phase_gcd > BigInt::one() && &phase_gcd < n {
            return true;
        }
    }
    
    false
}

/// Extract factor from coupled channel pair
pub fn extract_factor_from_coupled_pair(
    pair: &CoupledChannelPair,
    n: &BigInt,
    _params: &TunerParams,
) -> Option<BigInt> {
    // Try combined channel value interpretation
    let combined_value = BigInt::from(pair.channel1_val) * 256 + BigInt::from(pair.channel2_val);
    
    // Check if combined value is a factor
    if combined_value > BigInt::one() && &combined_value <= &n.sqrt() {
        if n % &combined_value == BigInt::zero() {
            return Some(combined_value);
        }
    }
    
    // Try reverse combination
    let reverse_combined = BigInt::from(pair.channel2_val) * 256 + BigInt::from(pair.channel1_val);
    if reverse_combined > BigInt::one() && &reverse_combined <= &n.sqrt() {
        if n % &reverse_combined == BigInt::zero() {
            return Some(reverse_combined);
        }
    }
    
    // Try GCD of coupled resonance with n
    let gcd = pair.coupled_resonance.primary_resonance.gcd(n);
    if gcd > BigInt::one() && &gcd < n {
        return Some(gcd);
    }
    
    // Try harmonic pattern extraction
    let pattern = pair.coupled_resonance.harmonic_signature & 0xFFFF;
    if pattern > 1 {
        let pattern_bigint = BigInt::from(pattern);
        if n % &pattern_bigint == BigInt::zero() && &pattern_bigint < n {
            return Some(pattern_bigint);
        }
    }
    
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_coupling_matrix_creation() {
        let identity = CouplingMatrix::identity();
        assert_eq!(identity.a11, 1.0);
        assert_eq!(identity.a12, 0.0);
        
        let standard = CouplingMatrix::standard();
        assert!(standard.a11 < 1.0);
        assert!(standard.a12 > 0.0);
    }
    
    #[test]
    fn test_channel_coupling() {
        let res1 = ResonanceTuple::new(
            BigInt::from(100u32),
            0x1234,
            BigInt::from(10u32),
        );
        
        let res2 = ResonanceTuple::new(
            BigInt::from(200u32),
            0x5678,
            BigInt::from(20u32),
        );
        
        let coupling = CouplingMatrix::standard();
        let (coupled1, coupled2) = apply_channel_coupling(&res1, &res2, &coupling);
        
        // Verify coupling was applied
        assert_ne!(coupled1.primary_resonance, res1.primary_resonance);
        assert_ne!(coupled2.primary_resonance, res2.primary_resonance);
    }
    
    #[test]
    fn test_position_dependent_coupling() {
        let lsb_coupling = CouplingMatrix::for_position(0, 8);
        let mid_coupling = CouplingMatrix::for_position(4, 8);
        let msb_coupling = CouplingMatrix::for_position(7, 8);
        
        // LSB should have stronger forward coupling
        assert!(lsb_coupling.a21 > msb_coupling.a21);
        
        // MSB should have weaker coupling overall
        assert!(msb_coupling.a12 < mid_coupling.a12);
    }
}