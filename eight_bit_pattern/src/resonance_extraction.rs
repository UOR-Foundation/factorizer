//! Advanced resonance-based factor extraction
//! 
//! Implements the 8-dimensional factorization space theory where factors
//! emerge from phase intersections and interference patterns.

use crate::{
    PeakLocation, Factors, TunerParams, Basis,
    Constants, decompose
};
use num_bigint::BigInt;
use num_traits::{Zero, One, Signed};
use num_integer::Integer;

/// Advanced factor extraction using resonance interference
pub fn extract_factors_resonance(
    n: &BigInt,
    peaks: &[PeakLocation],
    channels: &[u8],
    basis: &Basis,
    params: &TunerParams,
) -> Option<Factors> {
    // Try multiple extraction strategies
    
    // Strategy 1: Phase intersection method
    if let Some(factors) = extract_via_phase_intersection(n, peaks, channels, basis, params) {
        return Some(factors);
    }
    
    // Strategy 2: Resonance modulation method
    if let Some(factors) = extract_via_resonance_modulation(n, peaks, channels, basis, params) {
        return Some(factors);
    }
    
    // Strategy 3: Harmonic synthesis method
    if let Some(factors) = extract_via_harmonic_synthesis(n, peaks, channels, basis, params) {
        return Some(factors);
    }
    
    None
}

/// Extract factors via phase intersection in 8D space
fn extract_via_phase_intersection(
    n: &BigInt,
    peaks: &[PeakLocation],
    channels: &[u8],
    basis: &Basis,
    _params: &TunerParams,
) -> Option<Factors> {
    for peak in peaks {
        // Get resonances for channels in the peak
        let mut resonances = Vec::new();
        for ch_idx in peak.start_channel..=peak.end_channel {
            if ch_idx < channels.len() {
                if let Some(channel) = basis.get_channel(ch_idx) {
                    if let Some(pattern) = channel.get_pattern(channels[ch_idx]) {
                        resonances.push((ch_idx, &pattern.resonance));
                    }
                }
            }
        }
        
        if resonances.len() < 2 {
            continue;
        }
        
        // Compute phase differences between consecutive channels
        for i in 0..resonances.len() - 1 {
            let (_idx1, res1) = resonances[i];
            let (_idx2, res2) = resonances[i + 1];
            
            // Phase difference should encode factor information
            let phase_diff = (&res2.phase_offset - &res1.phase_offset).abs();
            
            // Try phase difference modulo n as potential factor encoding
            let phase_mod = &phase_diff % n;
            
            // The phase modulo might directly encode a factor
            if phase_mod > BigInt::one() && &phase_mod < n {
                let gcd = phase_mod.gcd(n);
                if gcd > BigInt::one() && &gcd < n {
                    if n % &gcd == BigInt::zero() {
                        return Some(Factors::new(gcd.clone(), n / &gcd));
                    }
                }
            }
            
            // Try harmonic relationship
            let harmonic_diff = (res2.harmonic_signature ^ res1.harmonic_signature) as u32;
            if harmonic_diff > 1 {
                let h_factor = BigInt::from(harmonic_diff);
                if n % &h_factor == BigInt::zero() && &h_factor < n {
                    return Some(Factors::new(h_factor.clone(), n / &h_factor));
                }
            }
        }
    }
    
    None
}

/// Extract factors via resonance modulation patterns
fn extract_via_resonance_modulation(
    n: &BigInt,
    peaks: &[PeakLocation],
    channels: &[u8],
    basis: &Basis,
    _params: &TunerParams,
) -> Option<Factors> {
    for peak in peaks {
        // Collect resonances for the peak
        let mut resonances = Vec::new();
        for ch_idx in peak.start_channel..=peak.end_channel {
            if ch_idx < channels.len() {
                if let Some(channel) = basis.get_channel(ch_idx) {
                    if let Some(pattern) = channel.get_pattern(channels[ch_idx]) {
                        resonances.push(&pattern.resonance);
                    }
                }
            }
        }
        
        if resonances.is_empty() {
            continue;
        }
        
        // Compute modulation pattern
        let mut modulation = BigInt::one();
        for res in &resonances {
            // Modulate by primary resonance
            modulation = (modulation * &res.primary_resonance) % n;
        }
        
        // The modulation pattern might reveal factors
        if modulation > BigInt::one() && modulation < *n {
            let gcd = modulation.gcd(n);
            if gcd > BigInt::one() && &gcd < n {
                if n % &gcd == BigInt::zero() {
                    return Some(Factors::new(gcd.clone(), n / &gcd));
                }
            }
        }
        
        // Try inverse modulation
        if let Some(inv_mod) = mod_inverse(&modulation, n) {
            let gcd = inv_mod.gcd(n);
            if gcd > BigInt::one() && &gcd < n {
                if n % &gcd == BigInt::zero() {
                    return Some(Factors::new(gcd.clone(), n / &gcd));
                }
            }
        }
    }
    
    None
}

/// Extract factors via harmonic synthesis
fn extract_via_harmonic_synthesis(
    n: &BigInt,
    peaks: &[PeakLocation],
    channels: &[u8],
    _basis: &Basis,
    _params: &TunerParams,
) -> Option<Factors> {
    // Get active constants from peak patterns
    for peak in peaks {
        let active_constants = Constants::active_constants(peak.aligned_pattern);
        if active_constants.len() < 2 {
            continue;
        }
        
        // Synthesize factor from constant relationships
        let mut synthesized = BigInt::zero();
        
        for (i, c1) in active_constants.iter().enumerate() {
            for c2 in active_constants.iter().skip(i + 1) {
                // Compute relationship between constants
                let num_prod = &c1.numerator * &c2.numerator;
                let den_prod = &c1.denominator * &c2.denominator;
                
                // Scale to integer domain
                let scaled = &num_prod / &den_prod;
                
                // Add to synthesis with channel weighting
                if peak.start_channel < channels.len() {
                    let weight = BigInt::from(channels[peak.start_channel]);
                    synthesized = (synthesized + scaled * weight) % n;
                }
            }
        }
        
        // Check if synthesized value relates to factors
        if synthesized > BigInt::one() && synthesized < *n {
            let gcd = synthesized.gcd(n);
            if gcd > BigInt::one() && &gcd < n {
                if n % &gcd == BigInt::zero() {
                    return Some(Factors::new(gcd.clone(), n / &gcd));
                }
            }
        }
        
        // Try Pollard-rho style iteration
        let mut x = synthesized.clone();
        let mut y = synthesized.clone();
        let c = BigInt::from(peak.aligned_pattern);
        
        for _ in 0..100 {
            x = (x.pow(2) + &c) % n;
            y = (y.pow(2) + &c) % n;
            y = (y.pow(2) + &c) % n;
            
            let d = (x.clone() - y.clone()).abs().gcd(n);
            if d > BigInt::one() && &d < n {
                return Some(Factors::new(d.clone(), n / &d));
            }
        }
    }
    
    None
}

/// Compute modular inverse
fn mod_inverse(a: &BigInt, m: &BigInt) -> Option<BigInt> {
    let (gcd, x, _) = extended_gcd(a, m);
    if gcd == BigInt::one() {
        Some((x % m + m) % m)
    } else {
        None
    }
}

/// Extended Euclidean algorithm
fn extended_gcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    if b.is_zero() {
        (a.clone(), BigInt::one(), BigInt::zero())
    } else {
        let (gcd, x1, y1) = extended_gcd(b, &(a % b));
        let x = y1.clone();
        let y = x1 - (a / b) * y1;
        (gcd, x, y)
    }
}

/// Advanced pattern recognition using 8D space theory
pub fn recognize_factors_advanced(
    n: &BigInt,
    basis: &Basis,
    params: &TunerParams,
) -> Option<Factors> {
    let channels = decompose(n);
    
    // First try standard pattern detection
    let peaks = crate::pattern::detect_aligned_channels(n, basis, params);
    
    if !peaks.is_empty() {
        // Try advanced resonance extraction
        if let Some(factors) = extract_factors_resonance(n, &peaks, &channels, basis, params) {
            return Some(factors);
        }
    }
    
    // If no peaks found, try full-space search
    full_space_factor_search(n, &channels, basis, params)
}

/// Search the full 8D space for factor patterns
fn full_space_factor_search(
    n: &BigInt,
    channels: &[u8],
    _basis: &Basis,
    _params: &TunerParams,
) -> Option<Factors> {
    // For each possible bit pattern, check if it reveals factors
    for pattern in 1u8..=255u8 {
        let active_constants = Constants::active_constants(pattern);
        if active_constants.is_empty() {
            continue;
        }
        
        // Compute pattern-specific resonance
        let mut pattern_resonance = BigInt::one();
        for c in &active_constants {
            pattern_resonance = (pattern_resonance * &c.numerator) / &c.denominator;
        }
        
        // Apply channel modulation
        for (i, &ch_val) in channels.iter().enumerate() {
            if ch_val > 0 {
                let ch_weight = BigInt::from(ch_val) << (i * 8);
                pattern_resonance = (pattern_resonance + ch_weight) % n;
            }
        }
        
        // Check for factors
        if pattern_resonance > BigInt::one() && pattern_resonance < *n {
            let gcd = pattern_resonance.gcd(n);
            if gcd > BigInt::one() && &gcd < n && n % &gcd == BigInt::zero() {
                return Some(Factors::new(gcd.clone(), n / &gcd));
            }
        }
    }
    
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mod_inverse() {
        let a = BigInt::from(3);
        let m = BigInt::from(11);
        let inv = mod_inverse(&a, &m).unwrap();
        assert_eq!((a * inv) % m, BigInt::one());
    }
    
    #[test]
    fn test_extended_gcd() {
        let a = BigInt::from(35);
        let b = BigInt::from(15);
        let (gcd, x, y) = extended_gcd(&a, &b);
        assert_eq!(gcd, BigInt::from(5));
        assert_eq!(a * x + b * y, gcd);
    }
}