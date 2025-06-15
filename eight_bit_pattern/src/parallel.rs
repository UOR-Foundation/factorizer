//! Parallel channel processing for performance optimization

use crate::{
    decompose, compute_resonance, TunerParams, ResonanceTuple,
    Basis, PeakLocation, Factors, bit_size
};
use num_bigint::BigInt;
use num_integer::Integer;
use rayon::prelude::*;
use std::sync::Arc;

/// Parallel channel decomposition for large numbers
pub fn decompose_parallel(n: &BigInt) -> Vec<u8> {
    let bit_count = bit_size(n);
    let channel_count = (bit_count + 7) / 8;
    
    if channel_count <= 8 {
        // For small numbers, serial is faster
        return decompose(n);
    }
    
    // For now, just use serial decomposition
    // Parallel decomposition of BigInt requires more careful handling
    decompose(n)
}

/// Parallel resonance computation across channels
pub fn compute_resonances_parallel(
    channels: &[u8],
    _basis: &Basis,
    params: &TunerParams,
) -> Vec<(usize, u8, ResonanceTuple)> {
    if channels.len() <= 16 {
        // For small channel counts, serial is faster
        return channels.iter()
            .enumerate()
            .filter_map(|(i, &ch)| {
                if ch == 0 { return None; }
                let resonance = compute_resonance(ch, params);
                Some((i, ch, resonance))
            })
            .collect();
    }
    
    // Parallel resonance computation
    channels.par_iter()
        .enumerate()
        .filter_map(|(i, &ch)| {
            if ch == 0 { return None; }
            let resonance = compute_resonance(ch, params);
            Some((i, ch, resonance))
        })
        .collect()
}

/// Parallel peak detection in channel alignments
pub fn detect_peaks_parallel(
    n: &BigInt,
    resonances: &[(usize, u8, ResonanceTuple)],
    params: &TunerParams,
) -> Vec<PeakLocation> {
    let n_arc = Arc::new(n.clone());
    let window_size = params.phase_coupling_strength as usize;
    
    if resonances.len() <= 32 {
        // Serial processing for small inputs
        return detect_peaks_serial(n, resonances, params);
    }
    
    // Parallel sliding window analysis
    (0..resonances.len())
        .into_par_iter()
        .filter_map(|start| {
            let n_ref = n_arc.clone();
            let end = (start + window_size).min(resonances.len() - 1);
            let window = &resonances[start..=end];
            
            // Check for alignment in this window
            if check_window_alignment(window, n_ref.as_ref(), params) {
                let pattern = calculate_window_pattern(window);
                Some(PeakLocation::new(start, end, pattern))
            } else {
                None
            }
        })
        .collect()
}

/// Serial peak detection (fallback for small inputs)
fn detect_peaks_serial(
    n: &BigInt,
    resonances: &[(usize, u8, ResonanceTuple)],
    params: &TunerParams,
) -> Vec<PeakLocation> {
    let mut peaks = Vec::new();
    let window_size = params.phase_coupling_strength as usize;
    
    for start in 0..resonances.len() {
        let end = (start + window_size).min(resonances.len() - 1);
        let window = &resonances[start..=end];
        
        if check_window_alignment(window, n, params) {
            let pattern = calculate_window_pattern(window);
            peaks.push(PeakLocation::new(start, end, pattern));
        }
    }
    
    peaks
}

/// Check if a window of resonances shows alignment
fn check_window_alignment(
    window: &[(usize, u8, ResonanceTuple)],
    n: &BigInt,
    _params: &TunerParams,
) -> bool {
    if window.len() < 2 {
        return window.len() == 1 && window[0].1 > 0;
    }
    
    // Check for congruence relationships
    for i in 0..window.len() - 1 {
        if window[i].2.aligns_with(&window[i + 1].2, n) {
            return true;
        }
    }
    
    false
}

/// Calculate alignment pattern for a window
fn calculate_window_pattern(window: &[(usize, u8, ResonanceTuple)]) -> u8 {
    window.iter()
        .map(|(_, ch, _)| ch)
        .fold(0u8, |acc, &ch| acc ^ ch)
}

/// Parallel factor extraction from multiple peaks
pub fn extract_factors_parallel(
    n: &BigInt,
    peaks: &[PeakLocation],
    channels: &[u8],
    params: &TunerParams,
) -> Option<Factors> {
    if peaks.is_empty() {
        return None;
    }
    
    let n_arc = Arc::new(n.clone());
    
    // Try to extract factors from peaks in parallel
    peaks.par_iter()
        .find_map_any(|peak| {
            let n_ref = n_arc.clone();
            extract_factor_from_peak_parallel(n_ref.as_ref(), peak, channels, params)
        })
}

/// Extract factor from a single peak (parallel-safe)
fn extract_factor_from_peak_parallel(
    n: &BigInt,
    peak: &PeakLocation,
    channels: &[u8],
    _params: &TunerParams,
) -> Option<Factors> {
    // Direct pattern check
    if peak.aligned_pattern > 1 {
        let pattern_as_factor = BigInt::from(peak.aligned_pattern);
        if n % &pattern_as_factor == BigInt::from(0) && &pattern_as_factor < n {
            let other_factor = n / &pattern_as_factor;
            return Some(Factors::new(pattern_as_factor, other_factor));
        }
    }
    
    // Channel value extraction
    if peak.start_channel < channels.len() {
        let ch_val = BigInt::from(channels[peak.start_channel]);
        if ch_val > BigInt::from(1) {
            let gcd = ch_val.gcd(n);
            if gcd > BigInt::from(1) && &gcd < n {
                let other = n / &gcd;
                return Some(Factors::new(gcd, other));
            }
        }
    }
    
    None
}

/// Parallel pattern recognition pipeline
pub fn recognize_factors_parallel(
    n: &BigInt,
    basis: &Basis,
    params: &TunerParams,
) -> Option<Factors> {
    // Special case detection first
    if let Some(factors) = crate::special_cases::try_special_cases(n) {
        return Some(factors);
    }
    
    // Parallel channel decomposition
    let channels = decompose_parallel(n);
    
    // Parallel resonance computation
    let resonances = compute_resonances_parallel(&channels, basis, params);
    
    // Parallel peak detection
    let peaks = detect_peaks_parallel(n, &resonances, params);
    
    // Parallel factor extraction
    extract_factors_parallel(n, &peaks, &channels, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_basis;
    
    #[test]
    fn test_parallel_decompose() {
        let n = BigInt::from(u64::MAX);
        let serial = decompose(&n);
        let parallel = decompose_parallel(&n);
        assert_eq!(serial, parallel);
    }
    
    #[test]
    fn test_parallel_recognition() {
        let n = BigInt::from(143u32); // 11 × 13
        let params = TunerParams::default();
        let basis = compute_basis(&n, &params);
        
        let result = recognize_factors_parallel(&n, &basis, &params);
        assert!(result.is_some());
        
        let factors = result.unwrap();
        assert!(factors.verify(&n));
    }
}