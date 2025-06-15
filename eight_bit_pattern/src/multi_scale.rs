//! Multi-scale alignment detection for channel patterns
//!
//! Detects patterns at different scales (local window vs global) without
//! disrupting the core pattern detection that's already working well.

use crate::{ResonanceTuple, TunerParams, PeakLocation};
use num_bigint::BigInt;
use num_traits::{Zero, One};
use num_integer::Integer;

/// Multi-scale alignment result
#[derive(Debug, Clone)]
pub struct MultiScaleAlignment {
    /// Scale level (1 = local, higher = more global)
    pub scale: usize,
    /// Start position in channel array
    pub start_pos: usize,
    /// End position in channel array
    pub end_pos: usize,
    /// Alignment quality (0.0 to 1.0)
    pub quality: f64,
    /// Detected pattern signature
    pub pattern_signature: u64,
    /// Potential factor if detected
    pub factor_hint: Option<BigInt>,
}

/// Detect alignments at multiple scales
pub fn detect_multi_scale_alignments(
    channel_resonances: &[(usize, u8, ResonanceTuple)],
    n: &BigInt,
    params: &TunerParams,
) -> Vec<MultiScaleAlignment> {
    let mut alignments = Vec::new();
    
    if channel_resonances.is_empty() {
        return alignments;
    }
    
    // Define scales based on channel count
    let max_scale = (channel_resonances.len() as f64).log2().ceil() as usize;
    
    for scale in 1..=max_scale {
        let window_size = 2_usize.pow(scale as u32).min(channel_resonances.len());
        let stride = (window_size / 2).max(1);
        
        // Scan at this scale
        for start in (0..channel_resonances.len().saturating_sub(window_size)).step_by(stride) {
            let window = &channel_resonances[start..start + window_size];
            
            if let Some(alignment) = analyze_window_alignment(window, n, scale, start, params) {
                alignments.push(alignment);
            }
        }
    }
    
    // Sort by quality and remove overlaps
    alignments.sort_by(|a, b| b.quality.partial_cmp(&a.quality).unwrap());
    filter_overlapping_alignments(alignments)
}

/// Analyze a window for alignment patterns
fn analyze_window_alignment(
    window: &[(usize, u8, ResonanceTuple)],
    n: &BigInt,
    scale: usize,
    start_pos: usize,
    params: &TunerParams,
) -> Option<MultiScaleAlignment> {
    if window.is_empty() {
        return None;
    }
    
    // Calculate alignment metrics
    let (pattern_signature, coherence) = calculate_pattern_signature(window);
    let factor_hint = detect_factor_hint(window, n);
    let quality = calculate_alignment_quality(window, n, coherence, &factor_hint);
    
    // Only return significant alignments
    if quality > params.alignment_threshold as f64 / 100.0 {
        Some(MultiScaleAlignment {
            scale,
            start_pos,
            end_pos: start_pos + window.len() - 1,
            quality,
            pattern_signature,
            factor_hint,
        })
    } else {
        None
    }
}

/// Calculate pattern signature and coherence for a window
fn calculate_pattern_signature(window: &[(usize, u8, ResonanceTuple)]) -> (u64, f64) {
    if window.is_empty() {
        return (0, 0.0);
    }
    
    // Combine resonance information into a signature
    let mut signature = 0u64;
    let mut coherence_sum = 0.0;
    
    for (i, (_, ch_val, res)) in window.iter().enumerate() {
        // Build signature from channel values and harmonic signatures
        signature = signature.rotate_left(8) ^ (*ch_val as u64);
        signature ^= (res.harmonic_signature as u64) << (i % 8);
        
        // Check coherence with neighbors
        if i > 0 {
            let prev_res = &window[i - 1].2;
            let phase_diff = &res.phase_offset - &prev_res.phase_offset;
            
            // Coherence based on phase relationship
            if phase_diff < BigInt::from(256) {
                coherence_sum += 1.0;
            } else if phase_diff < BigInt::from(65536) {
                coherence_sum += 0.5;
            }
        }
    }
    
    let coherence = if window.len() > 1 {
        coherence_sum / (window.len() - 1) as f64
    } else {
        1.0
    };
    
    (signature, coherence)
}

/// Detect potential factor hints from window
fn detect_factor_hint(window: &[(usize, u8, ResonanceTuple)], n: &BigInt) -> Option<BigInt> {
    // Try GCD of all primary resonances
    let mut gcd_accumulator = window[0].2.primary_resonance.clone();
    
    for (_, _, res) in window.iter().skip(1) {
        gcd_accumulator = gcd_accumulator.gcd(&res.primary_resonance);
        
        // Early exit if GCD becomes 1
        if gcd_accumulator == BigInt::one() {
            break;
        }
    }
    
    // Check if GCD is a meaningful factor
    if gcd_accumulator > BigInt::one() && &gcd_accumulator < n && n % &gcd_accumulator == BigInt::zero() {
        return Some(gcd_accumulator);
    }
    
    // Try combined channel values at different scales
    for group_size in [1, 2, 4].iter() {
        if *group_size <= window.len() {
            for start in (0..window.len()).step_by(*group_size) {
                let end = (start + group_size).min(window.len());
                let combined = window[start..end].iter()
                    .fold(BigInt::zero(), |acc, (_, ch_val, _)| {
                        acc * 256 + BigInt::from(*ch_val)
                    });
                
                if combined > BigInt::one() && &combined <= &n.sqrt() && n % &combined == BigInt::zero() {
                    return Some(combined);
                }
            }
        }
    }
    
    None
}

/// Calculate overall alignment quality
fn calculate_alignment_quality(
    window: &[(usize, u8, ResonanceTuple)],
    _n: &BigInt,
    coherence: f64,
    factor_hint: &Option<BigInt>,
) -> f64 {
    let mut quality = coherence * 0.5; // Base quality from coherence
    
    // Boost quality if we have a factor hint
    if factor_hint.is_some() {
        quality += 0.3;
    }
    
    // Check for special patterns in channel values
    let channel_values: Vec<u8> = window.iter().map(|(_, v, _)| *v).collect();
    
    // Repeating patterns boost quality
    if has_repeating_pattern(&channel_values) {
        quality += 0.1;
    }
    
    // Arithmetic progressions boost quality
    if has_arithmetic_progression(&channel_values) {
        quality += 0.1;
    }
    
    quality.min(1.0)
}

/// Check for repeating patterns in channel values
fn has_repeating_pattern(values: &[u8]) -> bool {
    if values.len() < 2 {
        return false;
    }
    
    // Check for period-2 patterns
    if values.len() >= 4 {
        let period2 = values.chunks(2)
            .map(|chunk| (chunk[0], chunk.get(1).copied().unwrap_or(0)))
            .collect::<Vec<_>>();
        
        if period2.windows(2).all(|w| w[0] == w[1]) {
            return true;
        }
    }
    
    // Check for constant values
    if values.windows(2).all(|w| w[0] == w[1]) {
        return true;
    }
    
    false
}

/// Check for arithmetic progressions
fn has_arithmetic_progression(values: &[u8]) -> bool {
    if values.len() < 3 {
        return false;
    }
    
    // Check if differences are constant
    let diffs: Vec<i16> = values.windows(2)
        .map(|w| w[1] as i16 - w[0] as i16)
        .collect();
    
    diffs.windows(2).all(|w| w[0] == w[1])
}

/// Filter out overlapping alignments, keeping highest quality
fn filter_overlapping_alignments(mut alignments: Vec<MultiScaleAlignment>) -> Vec<MultiScaleAlignment> {
    if alignments.is_empty() {
        return alignments;
    }
    
    let mut filtered = vec![alignments.remove(0)];
    
    for alignment in alignments {
        let overlaps = filtered.iter().any(|existing| {
            alignment.start_pos <= existing.end_pos && 
            alignment.end_pos >= existing.start_pos
        });
        
        if !overlaps {
            filtered.push(alignment);
        }
    }
    
    filtered
}

/// Convert multi-scale alignments to peak locations
pub fn alignments_to_peaks(alignments: &[MultiScaleAlignment]) -> Vec<PeakLocation> {
    alignments.iter().map(|alignment| {
        // Use lower 8 bits of pattern signature as aligned pattern
        let pattern = (alignment.pattern_signature & 0xFF) as u8;
        
        PeakLocation::new(
            alignment.start_pos,
            alignment.end_pos,
            pattern,
        )
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ResonanceTuple;
    
    #[test]
    fn test_repeating_pattern_detection() {
        assert!(has_repeating_pattern(&[1, 2, 1, 2, 1, 2]));
        assert!(has_repeating_pattern(&[5, 5, 5, 5]));
        assert!(!has_repeating_pattern(&[1, 2, 3, 4]));
    }
    
    #[test]
    fn test_arithmetic_progression() {
        assert!(has_arithmetic_progression(&[1, 3, 5, 7]));
        assert!(has_arithmetic_progression(&[10, 20, 30, 40]));
        assert!(!has_arithmetic_progression(&[1, 2, 4, 8]));
    }
    
    #[test]
    fn test_pattern_signature() {
        let window = vec![
            (0, 10u8, ResonanceTuple::new(BigInt::from(100), 0x1234, BigInt::from(0))),
            (1, 20u8, ResonanceTuple::new(BigInt::from(200), 0x5678, BigInt::from(10))),
        ];
        
        let (sig, coherence) = calculate_pattern_signature(&window);
        assert!(sig != 0);
        assert!(coherence >= 0.0 && coherence <= 1.0);
    }
}