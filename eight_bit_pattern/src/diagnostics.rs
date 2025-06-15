//! Diagnostics module for pattern analysis and tuning
//!
//! Provides instrumentation to understand how The Pattern recognizes factors.

use crate::{ResonanceTuple, PeakLocation};
use num_bigint::BigInt;
use std::collections::HashMap;

/// Diagnostic data for a single channel
#[derive(Debug, Clone)]
pub struct ChannelDiagnostic {
    pub position: usize,
    pub value: u8,
    pub resonance: ResonanceTuple,
    pub aligned_with: Vec<usize>, // Other channel positions this aligns with
}

/// Complete diagnostic trace of a factorization attempt
#[derive(Debug, Clone)]
pub struct FactorizationDiagnostics {
    /// The number being factored
    pub n: BigInt,
    
    /// Diagnostics for each channel
    pub channels: Vec<ChannelDiagnostic>,
    
    /// Which 8-bit patterns appear at detected peaks
    pub peak_patterns: HashMap<u8, usize>,
    
    /// Resonance values at peak locations
    pub peak_resonances: Vec<BigInt>,
    
    /// All detected peak locations
    pub peaks: Vec<PeakLocation>,
    
    /// Which patterns led to successful factor extraction
    pub successful_patterns: Vec<u8>,
    
    /// Factor candidates that were tested
    pub factor_candidates: Vec<BigInt>,
    
    /// Whether factorization succeeded
    pub success: bool,
}

impl FactorizationDiagnostics {
    /// Create new diagnostics for a number
    pub fn new(n: BigInt) -> Self {
        Self {
            n,
            channels: Vec::new(),
            peak_patterns: HashMap::new(),
            peak_resonances: Vec::new(),
            peaks: Vec::new(),
            successful_patterns: Vec::new(),
            factor_candidates: Vec::new(),
            success: false,
        }
    }
    
    /// Add channel diagnostic
    pub fn add_channel(&mut self, diagnostic: ChannelDiagnostic) {
        self.channels.push(diagnostic);
    }
    
    /// Record a peak pattern
    pub fn record_peak_pattern(&mut self, pattern: u8) {
        *self.peak_patterns.entry(pattern).or_insert(0) += 1;
    }
    
    /// Record a peak resonance value
    pub fn record_peak_resonance(&mut self, resonance: BigInt) {
        self.peak_resonances.push(resonance);
    }
    
    /// Record a detected peak
    pub fn record_peak(&mut self, peak: PeakLocation) {
        self.record_peak_pattern(peak.aligned_pattern);
        self.peaks.push(peak);
    }
    
    /// Record a successful pattern
    pub fn record_success(&mut self, pattern: u8) {
        self.successful_patterns.push(pattern);
        self.success = true;
    }
    
    /// Record a factor candidate that was tested
    pub fn record_candidate(&mut self, candidate: BigInt) {
        self.factor_candidates.push(candidate);
    }
    
    /// Get summary statistics
    pub fn summary(&self) -> DiagnosticSummary {
        DiagnosticSummary {
            channels_analyzed: self.channels.len(),
            peaks_detected: self.peaks.len(),
            unique_patterns: self.peak_patterns.len(),
            candidates_tested: self.factor_candidates.len(),
            success: self.success,
            most_common_pattern: self.most_common_pattern(),
        }
    }
    
    /// Find the most common pattern at peaks
    fn most_common_pattern(&self) -> Option<(u8, usize)> {
        self.peak_patterns
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(pattern, count)| (*pattern, *count))
    }
}

/// Summary statistics from diagnostics
#[derive(Debug, Clone)]
pub struct DiagnosticSummary {
    pub channels_analyzed: usize,
    pub peaks_detected: usize,
    pub unique_patterns: usize,
    pub candidates_tested: usize,
    pub success: bool,
    pub most_common_pattern: Option<(u8, usize)>,
}

/// Aggregate diagnostics across multiple factorizations
#[derive(Debug, Default)]
pub struct DiagnosticAggregator {
    /// Pattern frequency across all factorizations
    pub global_pattern_frequency: HashMap<u8, usize>,
    
    /// Pattern frequency only in successful factorizations
    pub success_pattern_frequency: HashMap<u8, usize>,
    
    /// Average resonance values by pattern
    pub pattern_resonances: HashMap<u8, Vec<BigInt>>,
    
    /// Success rate by bit size
    pub success_by_size: HashMap<usize, (usize, usize)>, // (successes, total)
    
    /// Total factorizations analyzed
    pub total_analyzed: usize,
    
    /// Total successful factorizations
    pub total_success: usize,
}

impl DiagnosticAggregator {
    /// Add diagnostics from a single factorization
    pub fn add(&mut self, diagnostics: &FactorizationDiagnostics) {
        self.total_analyzed += 1;
        
        if diagnostics.success {
            self.total_success += 1;
            
            // Record successful patterns
            for pattern in &diagnostics.successful_patterns {
                *self.success_pattern_frequency.entry(*pattern).or_insert(0) += 1;
            }
        }
        
        // Record all peak patterns
        for (pattern, count) in &diagnostics.peak_patterns {
            *self.global_pattern_frequency.entry(*pattern).or_insert(0) += count;
        }
        
        // Record resonances by pattern
        for (i, peak) in diagnostics.peaks.iter().enumerate() {
            if i < diagnostics.peak_resonances.len() {
                self.pattern_resonances
                    .entry(peak.aligned_pattern)
                    .or_default()
                    .push(diagnostics.peak_resonances[i].clone());
            }
        }
        
        // Track success by bit size
        let bit_size = diagnostics.n.bits() as usize;
        let entry = self.success_by_size.entry(bit_size).or_insert((0, 0));
        entry.1 += 1; // Total count
        if diagnostics.success {
            entry.0 += 1; // Success count
        }
    }
    
    /// Get patterns most correlated with success
    pub fn success_patterns(&self) -> Vec<(u8, f64)> {
        let mut patterns: Vec<(u8, f64)> = self.success_pattern_frequency
            .iter()
            .map(|(pattern, success_count)| {
                let total_count = self.global_pattern_frequency.get(pattern).unwrap_or(&0);
                let success_rate = if *total_count > 0 {
                    *success_count as f64 / *total_count as f64
                } else {
                    0.0
                };
                (*pattern, success_rate)
            })
            .collect();
        
        patterns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        patterns
    }
    
    /// Get overall statistics
    pub fn overall_stats(&self) -> String {
        let success_rate = if self.total_analyzed > 0 {
            self.total_success as f64 / self.total_analyzed as f64
        } else {
            0.0
        };
        
        format!(
            "Total: {}, Success: {}, Rate: {:.2}%",
            self.total_analyzed,
            self.total_success,
            success_rate * 100.0
        )
    }
}