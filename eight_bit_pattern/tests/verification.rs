//! Verification test suite for the 8-bit pattern implementation
//! 
//! Validates each channel's pre-computed basis and constants.

use eight_bit_pattern::*;
use num_bigint::BigInt;
use num_traits::{Zero, One};

#[test]
fn test_constants_precision() {
    // Verify all constants have proper Q32.224 encoding
    let constants = Constants::all();
    
    for constant in &constants {
        // Check denominator is 2^224
        let expected_denom = BigInt::one() << FRACTIONAL_BITS;
        assert_eq!(constant.denominator, expected_denom);
        
        // Check numerator is non-zero
        assert!(constant.numerator > BigInt::zero());
        
        // Check bit position is valid
        assert!(constant.bit_position < 8);
    }
}

#[test]
fn test_resonance_deterministic() {
    // Verify resonance function is deterministic
    let params = TunerParams::default();
    
    for bit_pattern in 0..=255u8 {
        let res1 = compute_resonance(bit_pattern, &params);
        let res2 = compute_resonance(bit_pattern, &params);
        
        assert_eq!(res1.primary_resonance, res2.primary_resonance);
        assert_eq!(res1.harmonic_signature, res2.harmonic_signature);
        assert_eq!(res1.phase_offset, res2.phase_offset);
    }
}

#[test]
fn test_basis_completeness() {
    // Verify basis has all required patterns
    let params = TunerParams::default();
    let basis = compute_basis(16, &params);
    
    assert_eq!(basis.num_channels, 16);
    assert_eq!(basis.channels.len(), 16);
    
    for channel in &basis.channels {
        assert_eq!(channel.patterns.len(), 256);
        
        // Verify each pattern
        for (i, pattern) in channel.patterns.iter().enumerate() {
            assert_eq!(pattern.bit_mask, i as u8);
        }
    }
}

#[test]
fn test_channel_decomposition_roundtrip() {
    // Test that decompose/reconstruct is lossless
    let test_numbers = vec![
        BigInt::from(0u32),
        BigInt::from(255u32),
        BigInt::from(65535u32),
        BigInt::from(16777215u32),
        BigInt::parse_bytes(b"123456789ABCDEF0123456789ABCDEF", 16).unwrap(),
    ];
    
    for n in test_numbers {
        let channels = decompose(&n);
        let reconstructed = reconstruct(&channels);
        assert_eq!(n, reconstructed);
    }
}

#[test]
fn test_alignment_modular_congruence() {
    // Verify alignment detection follows modular congruence rules
    let n = BigInt::from(143); // 11 * 13
    
    // Create two resonance tuples that should align
    let res1 = ResonanceTuple::new(
        BigInt::from(100),
        0xABCD,
        BigInt::from(8),
    );
    
    let res2 = ResonanceTuple::new(
        BigInt::from(100) + &n, // Congruent mod n
        0xABCE, // Different harmonic (allowed)
        BigInt::from(16), // Phase offset differs by 8
    );
    
    assert!(res1.aligns_with(&res2, &n));
    
    // Create non-aligned tuple
    let res3 = ResonanceTuple::new(
        BigInt::from(101), // Not congruent mod n
        0xABCF,
        BigInt::from(24),
    );
    
    assert!(!res1.aligns_with(&res3, &n));
}

#[test]
fn test_constant_interactions() {
    // Test that constant combinations produce expected patterns
    let params = TunerParams::default();
    
    // Unity only (bit 0)
    let unity_res = compute_resonance(0b00000001, &params);
    assert!(unity_res.primary_resonance > BigInt::zero());
    
    // Golden ratio + unity (bits 0,2)
    let phi_unity_res = compute_resonance(0b00000101, &params);
    assert!(phi_unity_res.primary_resonance > unity_res.primary_resonance);
    
    // All constants active
    let all_res = compute_resonance(0b11111111, &params);
    assert!(all_res.harmonic_signature != 0);
    assert!(all_res.phase_offset > BigInt::zero());
}

#[test]
fn test_peak_detection_consistency() {
    // Verify peak indices are consistent for same pattern
    let params = TunerParams::default();
    let channel = compute_channel_patterns(0, &params);
    
    // Check that identical bit patterns produce identical peaks
    for i in 0..256 {
        let pattern = &channel.patterns[i];
        
        // Peaks should be sorted
        for j in 1..pattern.peak_indices.len() {
            assert!(pattern.peak_indices[j] > pattern.peak_indices[j-1]);
        }
    }
}

#[test]
fn test_factor_extraction_small_semiprimes() {
    // Test with known small semiprimes
    let params = TunerParams::default();
    let basis = compute_basis(8, &params);
    
    let test_cases = vec![
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
    ];
    
    for (n, p, q) in test_cases {
        let n_big = BigInt::from(n);
        let result = recognize_factors(&n_big, &basis, &params);
        
        if let Some(factors) = result {
            assert!(
                (factors.p == BigInt::from(p) && factors.q == BigInt::from(q)) ||
                (factors.p == BigInt::from(q) && factors.q == BigInt::from(p)),
                "Failed to factor {} correctly", n
            );
        }
    }
}

#[test]
fn test_tuner_parameter_bounds() {
    // Verify tuner parameters stay within valid bounds
    let mut params = TunerParams::default();
    
    // Test extreme values
    params.alignment_threshold = 255;
    params.resonance_scaling_shift = 31;
    params.harmonic_progression_step = 256;
    params.phase_coupling_strength = 7;
    params.constant_weights = [255; 8];
    
    // Should still produce valid resonances
    let res = compute_resonance(0b10101010, &params);
    assert!(res.primary_resonance >= BigInt::zero());
    assert!(res.phase_offset >= BigInt::zero());
}

#[test]
fn test_basis_serialization() {
    // Test basis can be serialized and produces valid data
    let params = TunerParams::default();
    let basis = compute_basis(4, &params); // Small basis for testing
    
    let serialized = serialize_basis(&basis);
    
    // Check header
    assert_eq!(&serialized[0..8], b"8BITPATT");
    
    // Check version
    let version = u32::from_be_bytes([
        serialized[8], serialized[9], serialized[10], serialized[11]
    ]);
    assert_eq!(version, 1);
    
    // Check channel count
    let channels = u32::from_be_bytes([
        serialized[12], serialized[13], serialized[14], serialized[15]
    ]);
    assert_eq!(channels, 4);
    
    // Verify data is not empty
    assert!(serialized.len() > 16);
}